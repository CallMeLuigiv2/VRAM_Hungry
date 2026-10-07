# FlashMLA — tutor's reference notes

- **Repo:** `~/refs/inference/deepseek/FlashMLA` (github.com/deepseek-ai/FlashMLA)
- **Commit studied:** `2e5429fc5653bab6e081f09477126f731882a6a9` — "Open-source release for Ascend 950 (2026.09.30)", committed 2026-09-30. Shallow clone (single commit), so older revisions are **not** checked out locally.
- **Read on:** 2026-09-30. All file:line refs below were verified against this commit with `grep -n` / `sed -n`.
- **Audience:** the tutor, not the learner. Private reference; contains more than Luigi needs at any single milestone.

## Marking convention used in these notes

- `[code]` — read directly out of the repo, with file:line.
- `[docs]` — stated in `README.md` or `docs/*.md`, with line number.
- `[inference]` — my reasoning/derivation, not stated by DeepSeek. Never present these as DeepSeek's numbers.
- **Applies to:** each deep dive ends with `CPU engine` / `sm_89 CUDA` / `neither` so you know whether it can be taught as something Luigi will actually build.

## Executive summary (the 10 things that matter for this project)

1. **This release is Blackwell-only.** `TORCH_CHECK(arch.is_sm100f(), ...)` gates both entry points (`csrc/api/sparse_decode.cpp:190`, `csrc/api/sparse_prefill.cpp:104`); `setup.py:27` builds `sm_100a` + `sm_103a` only. The 2026-09-30 release **removed Hopper** and the older models [docs: README.md:3]. So the sm90 kernels the briefing asks about are *not in the tree* — only the two deep-dive blogs that describe them remain (`docs/20250422-*.md`, `docs/20250929-*.md`). I flag this everywhere it matters.
2. **MLA decode is compute-bound, and that is the whole point.** The arithmetic intensity of MLA decode is ≈ `2·h_q·s_q` FLOP/byte because K and V are *the same latent tensor* and there is one KV head for 128 query heads [docs: 20250422:11]. A plain MHA llama model (stories15M: 6 heads, head_dim 48) gets **0.5 FLOP/byte** with an f32 KV cache — three orders of magnitude lower [inference, derivation below]. This single contrast is the best roofline lesson in the repo and maps straight onto M0/role-project 4.
3. **Split-KV (flash-decoding) + a log-sum-exp combine kernel.** A one-warp scheduler kernel (`get_decoding_sched_meta.cu`) greedily cuts all requests' KV blocks into `num_sm_parts` equal-payload partitions; the attention kernel writes partial `(o_accum, lse_accum)`; a tiny combine kernel merges them with `exp2`/`log2` LSE math (`combine.cu:110-174`). The math is exactly what Luigi will need to split one long sequence across 24 CPU threads in M6.
4. **The LSE merge is worth a whole M6 lesson**, and vLLM has a CPU version of it to compare against: `~/refs/inference/vllm/csrc/cpu/mla_decode.cpp:128-148` (per-block merge) and `:322-350` (cross-thread merge). llama2.c parallelizes over *heads* (`run.c:283`), which leaves 18 of Luigi's 24 threads idle on a 6-head model — split-KV is the fix.
5. **"skip-scale"** — the running max used for softmax is allowed to lag the true max by up to 6 (in log2 units), which skips the expensive rescale of the output accumulator: `__any_sync(0xffffffff, cur_pi_max - mi > 6.0f)` (`kernel.cuh:224`), documented as a named technique in the Ascend blog [docs: 20260930:56-64, 240-248]. Cheap and portable to the CPU.
6. **Paged KV cache, but the kernel never reads a block table.** The caller pre-flattens `block_idx * page_block_size + offset` into the `indices` tensor [docs: README.md:150]; the kernel's TMA view is a flat 2-D `[rows, bytes/4]` tensor (`kv_cache_utils.cuh:53-61`). `page_block_size` is a *runtime* value (tests use 1, 2, 5, 61, 64, 69, 123, 256, 576). The `64` in this release is `B_TOPK`, the topk tile, not the page size — a correction to the briefing's premise.
7. **FP8 / FP4 KV cache with interleaved per-token scales.** 528 B/token (512× e4m3 + 16× ue8m0 scales, one per 32 values) or 288 B/token (512× e2m1 + 32× e4m3 scales, one per 16) [docs: README.md:140-141; code: `kv_cache_format.h:7-20`]. Dequantization happens **in the kernel, into shared memory**, by a dedicated warpgroup; the MMAs are bf16×bf16→f32.
8. **Warp specialization is the structural idea**: 3 warpgroups = softmax/epilogue (128 thr) | MMA-issue + TMA-gather + index-transform (4 warps) | dequant (128 thr), with 2 KV buffers and 4 index buffers (`kernel.cuh:165-655`, `config.h:63`). On sm_89 the roles survive; the mechanisms (TMA, TMEM, `setmaxnreg`, clusters) do not.
9. **Their correctness method is a 3-part check**: per-element (abs OR rel) tolerance, a *cosine-difference* gate on the whole tensor, and structural equality of inf/NaN positions (`tests/kernelkit/compare.py:44-105`), plus a bitwise determinism check across 5 runs (`tests/test_fmha_sm100.py:141-149`). Compare with D1's single measured `atol=5e-4, rtol=0`: their `rel_tol` values are literally `2.01/128`, `8.01/128`, `2.01/65536` — i.e. derived from bf16/fp32 mantissa ULPs, not measured. Good material for a D-decision discussion.
10. **The API is a two-call (metadata, then kernel) API that is *not* a deep module** in Ousterhout's sense: `block_table` and `cache_seqlens` are still positional parameters that the kernel ignores [code: `flash_mla_interface.py:87-88`], and reusing metadata with changed `topk_length` "silently reuses stale split-KV scheduling metadata" (`:90-95`). Great worked example of interface cruft caused by backwards compatibility.

---

# 1. Repo map

```
FlashMLA/
  README.md                          373 lines: claims, KV formats, usage
  docs/
    20250422-new-kernel-deep-dive.md            77  Hopper dense MLA decode: roofline + "seesaw"
    20250929-hopper-fp8-sparse-deep-dive.md     52  Hopper FP8 sparse decode: cycle model + "crossover"
    20260930-ascend-prefill-deep-dive.md       252  Ascend 950 sparse prefill/decode (EN)
    20260930-ascend-prefill-deep-dive-zh.md    255  same (ZH)
    assets/MLA Kernel Sched.drawio.svg              the seesaw schedule diagram
  flash_mla/
    __init__.py                       22  public names
    flash_mla_interface.py           552  get_mla_metadata, flash_mla_with_kvcache,
                                          flash_mla_sparse_fwd, flash_attn_varlen_* (autograd)
    fused_norm_rope_attn_rope_cast.py 211  prefill/decode + weight-permutation helpers
  csrc/
    params.h                         174  ALL kernel parameter structs + DecodingSchedMeta
    api/                                  torch/pybind layer, one .cpp per op
      api.cpp 16, common.h 277 (Arch, ImplBase, feature dispatch),
      sparse_decode.cpp 465, sparse_prefill.cpp 232,
      dense_fwd.cpp 15, dense_bwd.cpp 15, fused_norm_rope_attn_rope_cast_fwd.cpp 564
    cuda_kernels/
      defines.h 12, utils.h 47, kv_cache_format.h 59   (KV layout as a compile-time type)
      smxx/decode/                        architecture-independent CUDA
        get_decoding_sched_meta/ 148       the tile scheduler (1 block, 32 threads)
        combine/combine.cu       240       split-KV LSE merge, PDL-launched
      sm100/
        helpers.h 232, common_subroutine.h 225 (softmax helpers), dequant_utils.cuh 116,
        kv_cache_utils.cuh 64 (TMA tensormap for the paged cache)
        decode/sparse/head64/    kernel.cuh 741 + config.h 191 + 4 instantiations
        decode/sparse/head128/README.md    (a pointer: the h_q=128 decode kernel lives
                                            under prefill/sparse/fwd_for_small_topk/head128)
        prefill/sparse/fwd/head64|head128/         phase1.cuh 681 / 711
        prefill/sparse/fwd_for_small_topk/head128/ phase1.cuh 1075  (prefill AND decode)
        prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.cuh 1685
        prefill/dense/                     CUTLASS FMHA fwd+bwd (NVIDIA PR #76)
      3rdparty/kerutils/         PTX intrinsic wrappers, split sm80/sm90/sm100 (+ascend)
      3rdparty/cutlass/          submodule (not checked out here)
    ascend_kernels/prefill/sparse/         Ascend C (.asc) sparse prefill/decode
  tests/
    lib.py 504 (testcase gen + FLOP/byte accounting), ref.py 112 (PyTorch reference),
    quant.py 229 (quantize/dequantize + block-table flattening),
    test-sparse-decode.py 358, test-sparse-prefill.py 188,
    test-fused-norm-rope-attn-rope-cast.py 537, test_fmha_sm100.py 184,
    kernelkit/ (bench.py 303 kineto timing, compare.py 108 tolerances, build.py, platform.py)
  scripts/generate_instantiations.py 64
  setup.py 318   (arch flags, register-spill gate)
```

## 1.1 Kernels and the arch each supports

All from `setup.py:33-65` (source list), `csrc/api/*.cpp` (dispatch) and the file headers.

| Kernel | Entry point | Arch | Notes |
|---|---|---|---|
| Sparse MLA **decode**, h_q=64 | `flash_mla_with_kvcache` | sm100a/103a | `csrc/cuda_kernels/sm100/decode/sparse/head64/kernel.cuh`; 4 instantiations: {V41, V41_FP4 extra} × {split-KV, no-split} |
| Sparse MLA **decode**, h_q=128 | same | sm100a/103a | reuses the *prefill* `fwd_for_small_topk/head128` kernel in `Decode`/`DecodeWithSplitKV` mode (`sparse_decode.cpp:117`); 2-CTA cluster |
| Sparse MLA **prefill**, h_q=64 / 128 | `flash_mla_sparse_fwd` | sm100a/103a | `prefill/sparse/fwd/head{64,128}/phase1.cuh` |
| Sparse MLA prefill, small topk (≤1280) | same, auto-selected | sm100a/103a | `sparse_prefill.cpp:207`; same file as the h_q=128 decoder |
| **Fused** norm+RoPE+attn+RoPE+cast (prefill and decode) | `fused_norm_rope_attn_rope_cast.{prefill,decode}` | sm100a/103a | `prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.cuh` + two weight-permute kernels |
| **Dense MHA** fwd / bwd | `flash_attn_varlen_func` etc. | sm100 (CUTLASS FMHA) | contributed by NVIDIA [docs: README.md:28]; bwd rejects GQA (`flash_mla_interface.py:310`) |
| Tile-scheduler metadata | internal, first call of the decode op | any CUDA (`smxx`) | `get_decoding_sched_meta.cu`, grid 1×1×1, block 32 |
| Split-KV **combine** | internal | any CUDA (`smxx`) | `combine.cu`, launched with PDL |
| Ascend sparse prefill/decode | same Python API | Ascend 950 | `.asc` kernels; **no split-KV** (`sparse_decode.cpp:221`) |

Note the naming trap worth telling Luigi about: the h_q=128 *decode* kernel is a `prefill/` file, because decode with `s_q>1` (MTP/spec-decode) and prefill-with-topk are the same computation with a different scheduler. `csrc/cuda_kernels/sm100/decode/sparse/head128/README.md` is a one-line signpost to that effect. The repo treats prefill and decode as *modes of one kernel* (`SparseAttnFwdMode::{Prefill, Decode, DecodeWithSplitKV}`, `params.h:44-51`) — which is exactly the distinction M4 makes explicit in the engine.

---

# 2. The README's claims, and how the numbers are produced

## 2.1 The claims, verbatim conditions

| Claim | Hardware / conditions | Source |
|---|---|---|
| "up to 1460 TFlops during prefill and 950 TFlops during decoding" (fused kernel) | B200, CUDA 13.3 | README.md:41 |
| "up to 1350 TFlops" (sparse prefill) | B200, CUDA 13.3 | README.md:51 |
| "up to 1024 TFlops" (sparse decode) | B200, CUDA 13.3 | README.md:61 |
| "1460 TFlops in forward and 1000 TFlops in backward" (dense MHA) | B200, "as reported by NVIDIA" | README.md:71 |
| "410 TFlops (95% hardware peak)" prefill, "360 TFlops (83% of hardware peak)" decode | Huawei Ascend 950 NPU | README.md:13, 53, 63 |
| "5% ~ 15% performance improvement for compute-bound workloads, achieving up to 660 TFlops" | H800 SXM5 | README.md:29 |
| previous version: "3000 GB/s in memory-intensive settings and 580 TFlops in compute-bound settings" | H800 SXM5 | 20250422:3 |
| "80% Tensor Core utilization (of the throttled theoretical peak) and 3 TB/s memory bandwidth" | H800 SXM5 | 20250422:57 |
| "410 TFLOPS … (batch_size=128, num_heads=128, s_q=2, topk=2048)", vs "250 TFLOPS" without crossover; "up to 460 TFLOPS" at topk=32768; prior bf16 dense decode peak "640 TFLOPS" | H800 SXM5, FP8 sparse decode | 20250929:48-50 |
| 128K-token request needs `576·2·62·128·1024 = 8.72 GiB` of KV cache | DeepSeek-V3.2 | 20250929:3 |

Two honesty notes for the tutor: (a) *every* headline number is a peak ("up to") from a benchmark sweep, not a single fixed configuration; (b) the memory-bound number is reported in GB/s and the compute-bound number in TFLOP/s — the same kernel is measured against *whichever* roof it is under. That is the professional habit to transplant into the engine's README: report the metric that corresponds to the binding constraint, and say which one binds.

## 2.2 How the numbers are computed (benchmark code)

**FLOPs and bytes — the model.** `tests/lib.py:490-495`:

```python
compute_flop = 2 * p.h_q * num_attended_tokens * (p.d_qk + p.d_v)
mem_vol = sum([2*b*s_q*h_q*d_qk,            # Q, bf16
               kv_bytes(kv_scope) + ...,     # KV: unique tokens × bytes_per_token
               2*b*s_q*h_q*d_v])             # O, bf16
```

- `num_attended_tokens` counts *attended* (topk, or `topk_length.sum()`) tokens (`lib.py:467-472`) — so FLOPs are counted as work the kernel really does.
- `kv_bytes` uses **unique** indices (`indices.unique().numel()`, `lib.py:474-483`) times the format's bytes/token (`quant.py:19-24`). That is the honest count for a sparse gather: a token selected by two query positions is fetched once (from L2, in the best case).
- Note what is *excluded*: the `indices` tensor traffic, the scale bytes are included (they are inside `bytes_per_token`), and the `lse` output. [inference] These are small relative to the KV, but a careful benchmark table for the engine should say what is counted, as they do here in code.

**Achieved rates** — `tests/test-sparse-decode.py:234-236`:

```python
theoritical_compute_memory_ratio = flops_and_mem_vol.flop / flops_and_mem_vol.mem_vol
achieved_tflops = flops_and_mem_vol.flop / e2e_time_usage_s / 1e12
achieved_gBps  = flops_and_mem_vol.mem_vol / e2e_time_usage_s / 1e9
```

The results table prints `C/M` (that intensity), `TFlops`, `GBps`, `us` side by side (`:312-335`), and a **geometric mean** of TFLOPS across the sweep (`:338-354`). Printing the arithmetic intensity next to the achieved rates *is* a roofline table in text form; steal this layout for M5/M6/M7.

**Timing** — `tests/kernelkit/bench.py`:
- kineto/CUPTI device-side kernel time ranges, not wall clock (`_bench_kineto`, `:112-150`).
- **L2 is flushed before every iteration** with an 8 GB `zero_()` (`:119-136`): "an excessive 8GB memset to give the GPU some (literal) chill time without full idle" — thermal *and* cache control.
- A marker kernel delimits the profiling range; one warmup pass then one active pass (`:121-131`).
- `get_kernel_time` = Σ(kernel durations)/num_tests (`:51-81`); `get_e2e_time` = per-iteration `max(end) − min(start)` over the named kernels (`:83-110`), i.e. it *includes* the gap between the split-KV kernel and the combine kernel — which is why PDL (overlapping them) shows up as a real win.
- The decode test reports splitkv time, combine time, and e2e separately (`test-sparse-decode.py:213-247`), and sleeps 0.3 s between perf cases for cooldown (`:292-293`).
- The dense test instead uses `triton.testing.do_bench(warmup=2, rep=3)` and computes `FLOPS = total_attn_compute * h * 2 * (d + dv)` where `total_attn_compute` is the number of *unmasked* (i, j) pairs (`test_fmha_sm100.py:69-70, 151-155`). Counting only unmasked pairs is the honest way to report causal-attention FLOPS.

**Teaching hook (M0, role project 4).** Have Luigi write the same two functions for the CPU engine *before* the engine exists: `flops(cfg, pos)` and `bytes(cfg, pos)` for one decode step, then `predicted_tok_per_s = 1 / max(flops/F_peak, bytes/BW)`. Then the M4 measurement fills in the "measured" column. FlashMLA's `C/M` column is the proof that professionals keep the predicted intensity in the results table forever, not just in the design doc.

**Applies to:** CPU engine (accounting + table layout), sm_89 CUDA (same, plus L2 flush and event/CUPTI timing).

---

# 3. Deep dive: the decode attention problem and why intensity decides everything

## 3.1 What / where

**What.** In decode you have 1 query token per request (or a few, `s_q>1`, with MTP / speculative decoding — `README.md:161` calls `s_q` "the number of q tokens per q sequence. If MTP (speculative decoding) is disabled, it should be 1") against a KV cache of thousands to 128k tokens. Per request the kernel does two skinny GEMMs: `P = Q·Kᵀ` (`[h_q, d_k] × [d_k, s_k]`) and `O = softmax(P)·V` (`[h_q, s_k] × [s_k, d_v]`).

**Where.** The canonical derivation is `docs/20250422-new-kernel-deep-dive.md:11-15`:

> The number of FLOPs is roughly `2 (h_q s_q · d_k · s_k + h_q s_q · s_k · d_v) = 2 h_q s_q s_k (d_k+d_v)`, and the memory access volume (in bytes) is `sizeof(bfloat16) × (h_q s_q d_k + s_k d_k + h_q s_q d_v) ≈ 2 s_k d_k`. Thus, the compute-memory ratio is `h_q s_q · (d_k+d_v)/d_k ≈ 2 h_q s_q`. [docs: 20250422:11]

and the crossover [docs: 20250422:13]:

> An NVIDIA H800 SXM5 GPU has a peak memory bandwidth of 3.35 TB/s and peak FLOPs of 990 TFlops. However, due to throttling (reducing to ~1600 MHz in our case), the practical peak FLOPs drops to ~865 TFlops. Therefore, when `h_q s_q ≥ ½ · 865/3.35 = 128`, the kernel is compute-bound.

Then the punchline [docs: 20250422:15]: DeepSeek runs decode instances **without tensor parallelism**, so `h_q = 128` on one GPU and the kernel *is* compute-bound. The hardware constant (`865/3.35 ≈ 258` FLOP/byte machine balance) and the algorithm constant (`2·h_q·s_q`) are compared directly. That is role project 4 in three lines of arithmetic.

## 3.2 Why MLA's absorbed form changes the intensity

Write `g = h_q / h_kv` (query heads per KV head) and ignore Q/O traffic for a long context [inference]:

| Attention shape | bytes per KV token | FLOPs per KV token | intensity |
|---|---|---|---|
| MHA/GQA, bf16 K and V separate | `2·(d_k + d_v)` | `2·g·s_q·(d_k+d_v)` | `g·s_q` |
| MLA (K and V are one latent), bf16 | `2·d_k` (d_v ⊂ d_k) | `2·g·s_q·(d_k+d_v)` | `g·s_q·(d_k+d_v)/d_k` |

For V3-era MLA (`d_k=576`, `d_v=512`, `h_kv=1`, so `g = h_q = 128`): `(576+512)/576 = 1.889 ≈ 2`, hence the doc's `2·h_q·s_q` = **256 FLOP/byte** at `s_q=1` [docs: 20250422:11 + inference for the substitution]. Two independent factors of ~100× each:
1. **MQA-like sharing.** One KV head serves 128 query heads, so one byte of KV feeds 128 dot products. This is the "absorbed" form: the up-projections `W_UK`/`W_UV` are folded into `W_Q`/`W_O`, so decode attends in the 512-dim latent space and the cache stores the latent, not per-head K/V.
2. **K and V are the same tensor.** `docs/20250422:19` says it outright: "remember that in MLA, `K` and `V` are the same with different names". So the PV GEMM re-reads the *same* shared-memory tile as the QK GEMM — see `kernel.cuh:488` and `:496`, where `sK` and `sV` are both `plan.u.kv.dequant[rs.buf_idx].quant_part`, differing only in the CuTe layout (`SmemLayoutKTiles_DualGemm_SW128` vs `SmemLayoutKTilesTransposed_SW128`).

For the V4.1 format in *this* commit (`d_qk = d_v = 512`, fp8 cache at 528 B/token) [inference, using the repo's own formulas at `lib.py:490-495`]:

```
per request, s_q = 1, topk T = 2048, h_q = 128:
  flops = 2 · 128 · 2048 · (512+512)      = 536,870,912
  bytes = 2048·528  +  2·128·512·2 (Q)  +  2·128·512·2 (O)
        = 1,081,344 + 131,072 + 131,072   = 1,343,488
  intensity ≈ 400 FLOP/byte     (≈ 497 if you count KV traffic only)
```

FP8 roughly doubles the intensity relative to bf16 — quantizing the KV cache does not merely save memory, it **moves the kernel to the right on the roofline**, which is why the FP8 kernel then becomes *dequantization*-bound (§7) rather than bandwidth-bound.

## 3.3 The same arithmetic for the engine's own model (stories15M)

Verified header of `~/refs/inference/models/stories15M.bin`: `(dim=288, hidden=768, n_layers=6, n_heads=6, n_kv_heads=6, vocab=32000, seq_len=256)`. `head_size = 288/6 = 48`. llama2.c keeps the KV cache in **f32** (`run.c:63, 86`). All numbers below are [inference], derived the same way as the doc:

**Attention only, one decode step at context length L, all 6 layers:**
```
FLOPs = 6 layers · 6 heads · (2·48·L + 2·48·L)      = 6912·L
bytes = 6 layers · (K: L·288·4 + V: L·288·4)        = 13824·L
intensity = 0.5 FLOP/byte        (f32 KV)
          = 1.0 FLOP/byte        (bf16/f16 KV)
          = 2.0 FLOP/byte        (int8 KV)
```
Per the table in §3.2: `g = h_q/h_kv = 1`, `s_q = 1`, so intensity = `g·s_q` = 1 for bf16 — the formula and the concrete count agree, which is a nice check to make Luigi do himself.

**Whole model, one decode step:** ~15.2 M params, f32 → the 58 MB file is streamed per token for `2·15.2M ≈ 30.4 MFLOP` → **0.5 FLOP/byte**; int8 weights → ~1 FLOP/byte. Attention's 13.8 kB/token·L only rivals the 58 MB of weights when `L > 4400` — impossible here (`seq_len = 256`, and `256·13824 = 3.5 MB`, 6 % of the weight traffic).

**Consequences to teach:**
- For stories15M, **decode is weight-streaming-bound**; the KV cache is a rounding error. The M0 performance model should predict decode tok/s from `weight_bytes / achievable_BW`, and TTFT from the prefill FLOPs, and say so.
- Intensity 0.5–1 FLOP/byte versus a CPU machine balance in the tens (e.g. [inference] ~1 TFLOP/s achievable f32 with AVX2+FMA on 24 threads over ~40 GB/s of DDR ⇒ ~25 FLOP/byte) means the engine is **~25–50× to the left of the ridge point**. Every M6 optimisation should therefore be judged by bytes moved, not FLOPs saved. Quantization (M5) is not a memory-capacity trick here — it is *the* throughput lever.
- On the GPU (M7) the same model is even more lopsided: 12 GB of HBM at (spec) 504 GB/s, versus tens of TFLOP/s of bf16 tensor-core throughput ⇒ machine balance in the hundreds of FLOP/byte [inference]. **Have Luigi measure both roofs** (a big `cudaMemcpy`/STREAM-style kernel for BW, a large cuBLAS GEMM for FLOPS) instead of trusting the 504 GB/s spec figure — that is exactly what DeepSeek did when they used the *throttled* 865 TFLOPS rather than the 990 nameplate [docs: 20250422:13].

## 3.4 The tensor-core trap for decode on sm_89 [inference, important for M7]

With `s_q = 1` and a GQA/MHA model, the QK product is a **matrix-vector** product per head: the smallest bf16 `mma.sync` shape on sm_89 is `m16n8k16`, so an M=1 problem wastes 15/16 of the tensor core. FlashMLA never has this problem because MQA + no tensor parallelism gives it `M = h_q = 64 or 128` rows of *real* work per KV tile (`MMA_M = 64` at `config.h:33`, "Head block size of the MMA; equals h_q").

So the honest plan for Luigi's M7 decode kernel: **CUDA cores, vectorized loads, warp-level reductions** — a bandwidth kernel. Tensor cores earn their keep in (a) batched prefill (matrix × matrix), and (b) models with a large `h_q/h_kv` group. Say this early; it prevents a month of chasing `mma.sync` for a kernel that is bandwidth-bound anyway.

**Applies to:** CPU engine (§3.3 numbers, M0), sm_89 CUDA (§3.4), and the roofline method everywhere.

---

# 4. Deep dive: paged KV cache and the block table

## 4.1 What the layout actually is

`k_cache: [num_blocks, page_block_size, num_heads_k, bytes_per_token]` [code: `flash_mla_interface.py:82-86`; shape-checked at `sparse_decode.cpp:289`]. `h_kv` must be 1 ("only MQA is supported", `:230`). The last dim is **bytes**, not elements, because a token's quantized payload and its scales share the row (§7).

Requirements the API states [code: `flash_mla_interface.py:86`]: the cache must be "contiguously valid" — one address range, possibly a slice of a larger array, never a list of disjoint blocks; and `kv.stride(1) == bytes_per_token` (`sparse_decode.cpp:291`), i.e. rows within a block are dense, though `kv.stride(0)` (the block stride) may be padded, as long as it is a multiple of the token stride (`kv_cache_utils.cuh:51`).

## 4.2 How the kernel indexes it — two levels

**Level 1: host/indexer side.** The block table is applied *before* the kernel: `tests/quant.py:197-229`

```python
indices_in_kvcache = block_table[i][abs_idx // block_size] * block_size + abs_idx % block_size
```

which is **exactly** vLLM PagedAttention's logical→physical mapping, just hoisted out of the kernel. README.md:150 states the contract and adds: "Since the index of the page block has already been encoded into `indices_in_kvcache`, the kernel does not use the `block_table` parameter". Invalid slots are `-1` (`README.md:151`), and the decode kernel "treats an index as invalid only when it is exactly -1; it performs no upper-bound check" (`flash_mla_interface.py:104-106`) — any other out-of-range value produces out-of-bounds TMA addresses. A sharp, documented footgun.

**Level 2: kernel side.** One warp (warp 7) turns the flat indices into TMA coordinates and a validity bitmask (`kernel.cuh:534-614`):

```c
int kv_block_idx  = (unsigned)cur_idx / cur_block_size;   // page_block_size, runtime
int idx_in_block  = (unsigned)cur_idx % cur_block_size;
...
tma_coords[i] = is_token_valid ? block_idx_arr[i]*cur_tma_coords_step_per_block
                               + idx_in_block_arr[i]*tma_coords_step_per_token : -1;
```
with `tma_coords_step_per_block = params.stride_kv_block / TMA_K_STRIDE` (`:539`) — i.e. it divides out the *logical* page size and multiplies back in the *physical* row stride, so a padded block stride works. Validity is packed 2 bits per lane into a `char` mask via `__shfl_xor_sync` (`:585-590`) and stored in `plan.is_token_valid[buf][64/8]`.

The TMA view of the whole cache is created once per launch (`kv_cache_utils.cuh:53-61`): a 2-D `uint32` tensor of shape `{RAW_TOKEN_DATA_BYTES/4, num_blocks·(block_stride/token_stride)}` — *paging has been flattened away into a row index*. Loads use `ku::tma_gather4`, four rows per instruction, 16 instructions per 64-token tile (`kernel.cuh:515-529`), with `EVICT_FIRST` as the L2 hint.

Invalid tokens get coordinate `-1`: TMA zero-fills out-of-range coordinates, and the comment at `:583` explains why that matters — "we must manually fill tma_coords with -1 to avoid copying-in NaN". The score for such a token is then forced to `-inf` *before* the cross-half reduction, because "(-inf) + anything (except nan and +inf) is (-inf)" (`common_subroutine.h:103-119`).

## 4.3 Why: relation to vLLM PagedAttention and to the post-v1 server

Same idea as vLLM — fixed-size physical blocks, a per-request table of physical block ids, no contiguous-per-request allocation, so no fragmentation and easy prefix sharing. Two differences worth a decision entry:

1. **Who dereferences the table.** vLLM's GPU kernels take `block_tables` and dereference per block. FlashMLA's sparse decode takes pre-flattened row ids. Cost: the caller (the DSA indexer) must do the translation and the kernel loses the ability to bounds-check. Benefit: the kernel's addressing becomes a single flat gather, so the *sparse* and *paged* cases collapse into one code path — the kernel never knows it is paged.
2. **Page size.** `page_block_size` is a runtime value here; the tests deliberately use non-power-of-two and tiny page sizes (1, 2, 5, 61, 69, 123, 576 — `test-sparse-decode.py:56-72`). What *is* fixed at 64 is `B_TOPK` (`config.h:48`) — the kernel's KV tile — and `block_size_topk = 64` in the scheduler (`sparse_decode.cpp:68`), plus `KU_ASSERT(params.topk % B_TOPK == 0)` (`kernel.cuh:671`). **Correction to the briefing:** block size 64 in this release is the topk tile, not the page size. (Older Hopper FlashMLA did require page size 64; not verifiable in this shallow clone.)

**Teaching hook (post-v1, and M3 as a seed).** In M3 Luigi's KV cache will be `[layer][pos][kv_dim]`, contiguous per sequence — the llama2.c layout (`run.c:63`: `float* key_cache; // (layer, seq_len, dim)`). The paged version changes exactly one thing: `k_ptr(layer, t)` goes from `base + layer*stride_l + t*stride_t` to `base + block_table[t/BS]*stride_block + (t%BS)*stride_t`. Have him write the accessor as a *function* in M3 (not inline pointer arithmetic), so paging in the server is a one-function change. That is the whole architectural payoff, and it costs nothing now.

**Pitfalls:** (a) `%` and `/` by a runtime page size are expensive in a hot loop — FlashMLA hoists them to a dedicated warp and prefetches next-block indices (`kernel.cuh:597-604`); the CPU analog is to compute the block base pointer once per block, never per element. (b) Invalid/padding slots must be masked to `-inf` *before* any summation, or NaNs propagate. (c) A power-of-two page size lets `/` and `%` become shifts — a real argument for 64 or 128 even when the kernel allows anything.

**Applies to:** CPU engine (accessor design now, paging post-v1), sm_89 CUDA (same indexing; `tma_gather4` → a per-thread gather with `cp.async` or plain `ld.global.nc` of 16 B chunks).

---

# 5. Deep dive: split-KV (flash-decoding), the tile scheduler, and the combine kernel

This is the single most transferable section: it carries M6 (24 CPU threads) and M7 (many SMs) at once.

## 5.1 The problem

One decode step with batch `b` produces only `b · s_q` (or `b · s_q · h_kv`) independent tiles of work. On a 132-SM GPU with `b=4`, 128 SMs idle. Worse, sequence lengths differ wildly, so even a per-request assignment leaves most SMs waiting on the longest request. Flash-decoding's answer: split the *KV axis*, let several SMs each attend to a slice of the same request, then merge the partial softmaxes.

FlashMLA turns it on only when there is enough work per request (`sparse_decode.cpp:223`):

```c
bool enable_split_kv = !enable_batch_invariant && !(topk + extra_topk <= 640);
```

So: ≤ 640 attended tokens ⇒ no split (the fixed overhead would dominate), and `enable_batch_invariant=True` ⇒ no split ever, so results do not depend on how the batch was partitioned (`flash_mla_interface.py:110-112`). That flag is a *determinism* switch, and it is the same tension Luigi's "fixed seed, reproducible from one command" rule creates.

## 5.2 `get_mla_metadata` → the tile scheduler

The Python `get_mla_metadata()` is now a placeholder returning an empty `FlashMLASchedMeta` dataclass (`flash_mla_interface.py:43-56`); the real metadata is generated inside the first `flash_mla_with_kvcache` call (`:141-183`) and cached in that object. The metadata is produced by a GPU kernel with **grid 1×1×1, block 32** — one warp (`get_decoding_sched_meta.cu:15-16`).

The descriptor, one per SM partition (`params.h:84-95`, 32 B, `static_assert(sizeof(DecodingSchedMeta) == 32)`):

```c
int begin_req_idx, end_req_idx;      // inclusive
int begin_block_idx, end_block_idx;  // [begin, end)
int begin_split_idx;                 // where this partition's partial output goes
int is_first_req_splitted, is_last_req_splitted;
```

Algorithm (`get_decoding_sched_meta.cu:42-122`):

1. Each lane handles a strided subset of requests, computing that request's block count `ceil(topk_length / 64)` and accumulating `num_blocks + fixed_overhead_num_blocks` into a warp-reduced `total_num_blocks` (`:43-61`). `cur_s_k == 0` is forced to 1 so "the main loop will never be empty" (`:45`).
2. One elected lane computes the per-partition **payload**:
   ```c
   int payload = ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks;  // :65
   ```
3. It then walks requests in order, filling partitions greedily: consume a whole request if `remain_payload >= now_remain_blocks + fixed_overhead`, else consume `remain_payload - fixed_overhead` blocks of it and record a split (`:79-102`).
4. `num_splits_ptr` is a **cumulative** array of length `b+1`: request *i* owns split slots `[num_splits[i], num_splits[i+1])` (`:85-86`, `:119-121`).
5. A device-side assertion checks the walk consumed everything: `KU_TRAP_ONLY_DEVICE_ASSERT(now_req_idx == batch_size && now_block == 0 && ...)` (`:115`).

The constants come from the kernel implementation itself (`sparse_decode.cpp:63-69` for h_q=64, `:99-106` for h_q=128):

| | `num_sm_parts` | `fixed_overhead_num_blocks` | `block_size_topk` |
|---|---|---|---|
| h_q = 64 | `max(num_sms / s_q, 1)` | 5 | 64 |
| h_q = 128 | `max(num_sms / s_q / 2, 1)` | 3 (`// TODO Tune`) | 64 |

Two things to point out to Luigi:
- **`fixed_overhead_num_blocks` is a cost model inside the scheduler.** Every request charges ~3–5 extra "virtual blocks" for its prologue/epilogue (load Q, TMA descriptors, write the partial output). Without it, the scheduler would happily give a partition 30 one-block requests and call it balanced. This is role project 4 living inside the load balancer — the same trick the post-v1 request router will need (a request costs `a + b·tokens`, not `b·tokens`).
- **`num_sm_parts` divides by `s_q`** because the grid's x dimension is `s_q` (`kernel.cuh:732`: `dim3(params.s_q, ENABLE_SPLITKV ? params.num_sm_parts : params.b, 1)`) and by an extra 2 for h_q=128 because that kernel runs 2-CTA clusters [inference: its grid is `[2*b]` with cluster `[2,1,1]`, `phase1.cuh:14, 51`]. The metadata is a function of the *machine* (SM count) and the *batch shape* — so it is computed once per decoding step shape and reused across the 60+ layers (`README.md:111-124`).

Buffers for the partial results (`sparse_decode.cpp:420-431`):
```c
const int total_num_splits = b + params.num_sm_parts;
lse_accum = empty({total_num_splits, s_q, h_q});          // float32
o_accum   = empty({total_num_splits, s_q, h_q, d_v});     // float32
```
`b + num_sm_parts` is the tight bound [inference]: each request needs ≥1 slot, and each partition boundary can add at most one extra split. For `h_q=128, d_v=512, s_q=1, b=128, num_sm_parts=64`: `192·128·512·4 B = 50 MB` of f32 scratch — split-KV is not free in memory, which is part of why it is gated on `topk > 640`.

## 5.3 What the attention kernel writes

Non-split path (`kernel.cuh:294-299`), natural log, straight to the user's tensors:
```c
float cur_lse = fma(mi, CUDART_LN2_F, logf(li));      // m·ln2 + ln(l)
cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
```
and the output is scaled by `1/(li + exp2f(attn_sink - mi))` (`:327`) — the sink folded in.

Split path (`:300-304`), **log2** domain, into the scratch buffer:
```c
float cur_lse = log2f(li) + mi;
*gSoftmaxLseAccum = cur_lse;
```
and the partial output is normalized by its own `li` only (`:362`): "Here we leave attn_sink to the combine kernel, otherwise attn_sink will take effect for multiple times". The partial `o_accum` rows are written with `SM90_BULK_COPY_S2G` per head row, skipping padding rows beyond `h_q` (`:384-397`).

## 5.4 The combine kernel — exact math

`csrc/cuda_kernels/smxx/decode/combine/combine.cu`. Grid `[b, s_q, ceil(h_q/8)]`, block 256 = 8 warps, **one warp per query head** (`:20-22`, `:48`). Key steps:

- Early exit when the request was not split: `if (my_num_splits == 1) return;` (`:67-69`) — the unsplit path already wrote the final answer.
- Split range from the cumulative array (`:64-66`).
- `cudaGridDependencySynchronize()` (`:88`) = PDL: this kernel starts before the attention kernel has fully retired and waits only for the data dependency. The attention kernel calls `cudaTriggerProgrammaticLaunchCompletion()` when it finishes its last request (`kernel.cuh:311-313`), and the combine kernel is launched with `cudaLaunchAttributeProgrammaticStreamSerialization` via `cudaLaunchKernelEx` (`combine.cu:218-230`). Note `__ldg` is avoided for the prefetch because it "is incompatible with PDL" (`:97`).
- The reduction (`:100-147`), all in base 2:

```c
local_lse[i] = split_idx < my_num_splits ? gLseAccum(split_idx, warp_idx) : -INFINITY;
max_lse = warp_max(local_lse);  max_lse = max_lse == -INFINITY ? 0.0f : max_lse;   // :110-117
sum_lse = Σ exp2f(local_lse[i] - max_lse);  (warp-reduced)                          // :119-125
global_lse = (sum_lse == 0 || sum_lse == -INF) ? INFINITY : log2f(sum_lse) + max_lse;  // :127
gLse(warp_idx) = global_lse / M_LOG2E;                        // → natural log      // :129
smem_buf[warp_idx][split_idx] = exp2f(local_lse[i] - global_lse);   // per-split weight :146
out = Σ_j smem_buf[·][j] · o_accum[j]                                               // :159-174
```

The math, written out [inference, derived from the code above]. Let split *j* cover token set `S_j`, with per-split max `m_j` and sum `l_j = Σ_{t∈S_j} 2^{p_t − m_j}`, storing `lse_j = log₂ l_j + m_j` and `o_j = (1/l_j)·Σ_{t∈S_j} 2^{p_t−m_j} v_t`. Then `2^{lse_j} = Σ_{t∈S_j} 2^{p_t}`, so

```
L      = log2( Σ_j 2^{lse_j} )                   (computed max-shifted, :119-127)
w_j    = 2^{lse_j − L}   with   Σ_j w_j = 1       (:146)
Σ_j w_j · o_j = Σ_t 2^{p_t} v_t / Σ_t 2^{p_t} = softmax(p)·v      ✓ exact
```

Numerical guards worth naming: the `max_lse == -inf → 0` guard "In case all local LSEs are -inf" (`:117`), and `sum_lse == 0 → global_lse = +inf` so all weights become `exp2(finite − inf) = 0` (`:127`, `:135`). `attn_sink` is applied *only* to the output scaling, via `global_lse += log2f(1 + exp2f(attn_sink·log2e − global_lse))` (`:137`), leaving the returned LSE untouched (`:131-142`).

`MAX_SPLITS` is a compile-time template arg, dispatched in steps of 32 up to 160 (`:189-209`), so `local_lse` stays in registers (`NUM_LSE_PER_THREAD = ceil(MAX_SPLITS/32)`, `:102`). Each lane holds `512/(32·4) = 4` `float4`s of the output (`:91-98`) and software-pipelines the next split's load inside the accumulation loop (`:169-171`).

## 5.5 Teaching hook: the tiny worked example (use this verbatim at M3 and M7)

Four keys, scores already scaled, in log2 units: `p = [1, 3, 2, 5]`, values `v1..v4`.

**One pass, two tiles of 2 (online softmax, §6):** final `l = 5.75`, `m = 3`, `o/l = [0.04348, 0.17391, 0.08696, 0.69565]·v`. Check: `2^p = [2,8,4,32]`, sum 46, and `5.75·2^3 = 46` ✓.

**Two splits, merged with LSE (this section):**

| | tokens | m_j | l_j | `lse_j = log2 l_j + m_j` | `o_j` (normalized) |
|---|---|---|---|---|---|
| split A | p=1,3 | 3 | 1.25 | 3.3219 | 0.2·v1 + 0.8·v2 |
| split B | p=2,5 | 5 | 1.125 | 5.1699 | 0.1111·v3 + 0.8889·v4 |

Merge: `max = 5.1699`; `sum = 2^{3.3219−5.1699} + 2^0 = 0.2778 + 1 = 1.2778`; `L = log2(1.2778) + 5.1699 = 5.5235`. Weights `w_A = 2^{3.3219−5.5235} = 0.2174`, `w_B = 0.7826` (sum = 1). Output:

```
0.2174·(0.2 v1 + 0.8 v2) + 0.7826·(0.1111 v3 + 0.8889 v4)
 = 0.04348 v1 + 0.17391 v2 + 0.08696 v3 + 0.69565 v4     ← identical to the one-pass result
```

and `L·ln2 = 3.829 = ln 46` ✓. Two splits, one pass, same answer, to 5 digits. That is the whole lesson; the numbers are small enough to do on paper.

## 5.6 The CPU analog — exactly what M6 needs

**The problem on Luigi's machine.** llama2.c parallelizes attention over heads (`run.c:283`: `#pragma omp parallel for private(h)` over `p->n_heads`). stories15M has **6 heads** → 6 of 24 threads work, 18 idle, for the whole attention phase. Splitting the *time axis* fixes it.

**The reference implementation to show him** — vLLM's CPU MLA decode, `~/refs/inference/vllm/csrc/cpu/mla_decode.cpp`:
- per-thread accumulators sized `max_threads · num_heads · V_HEAD_DIM` plus `acc_lse` (`:262-267`), initialized to `0` and `-FLT_MAX` (`:289-290`);
- `#pragma omp for` over the request's **blocks**, each thread merging its blocks into its own `(acc_out, acc_lse)` via the block-level LSE merge at `:128-148`;
- then `#pragma omp for` over **heads**, each thread merging `num_threads` partial states for one head (`:322-350`): max over partial LSEs, `exp`, sum, `1/sum`, scaled accumulate.
- Their merge is in **natural log** (`std::log`, `std::exp`), and they cite "section 2.2 in https://arxiv.org/pdf/2501.01005" at `:129` and `:323`.

The natural-log form of the same math, for the CPU engine:
```
merge((o_a, lse_a), (o_b, lse_b)):
    m   = max(lse_a, lse_b)
    ea  = exp(lse_a - m);  eb = exp(lse_b - m)
    s   = ea + eb
    lse = log(s) + m
    o   = (ea/s)·o_a + (eb/s)·o_b        // both o already normalized
```
Associative and commutative up to floating point, so a tree reduction over 24 threads works. **Pitfall to plant:** the order of the merge changes the last bits, so a 24-thread split is not bitwise equal to a 1-thread run — which is exactly what `enable_batch_invariant` exists for on the GPU side. Decide *in DECISIONS.md* whether M6's benchmark mode is deterministic (fixed split count) or throughput-optimal (dynamic), and how the M3 tolerance check is run.

**Design sketch for M6 (mirroring FlashMLA's structure):**
1. Choose `n_splits` per (layer, head) as `clamp(L / MIN_TOKENS_PER_SPLIT, 1, n_threads/…)` with a `fixed_overhead` term, as in `payload` above. With one request and `L=256` on 24 threads, `MIN_TOKENS_PER_SPLIT` will often say "don't split" — measure it.
2. Partial state per (thread, head): `o[head_dim]` f32 + `lse` f32. Tiny: `6 heads · 48 · 4 B = 1.2 kB` per thread.
3. Merge tree, then the FFN continues single-threaded per token.
4. Report before/after with the same table as §2.2, including the intensity column.

**Applies to:** CPU engine (M6, directly), sm_89 CUDA (M7, directly — split-KV is the reason a decode kernel fills a GPU at all), post-v1 server (the scheduler's cost model).

---

# 6. Deep dive: online softmax as the kernel actually does it

## 6.1 The algorithm, from DeepSeek's own pseudocode

`docs/20260930-ascend-prefill-deep-dive.md:39-75` writes it out in 8 steps. State per row: `running_max`, `running_max_for_softmax`, `running_sumexp`, `out_accum`, all f32, initialized `-inf, -inf, 0, 0` (`:45`). Per KV block: gather KV, `P = Q·gathered_kvᵀ`, row max `cur_max`, update `running_max`, and then **only if** `running_max − running_max_for_softmax > RESCALE_THRES` (6) rescale `out_accum` and `running_sumexp` and adopt the new max (`:55-69`); otherwise "skip scale". Then `S = exp(P − running_max_for_softmax)`, cast to bf16, `out_accum += S·gathered_kv`, `running_sumexp += S.sum()`.

## 6.2 How the CUDA kernel implements it

`csrc/cuda_kernels/sm100/decode/sparse/head64/kernel.cuh:190-279`, warpgroup 0:

```c
float mi = MAX_INIT_VAL;      // -1e30, NOT -inf: see below     (:190, config.h:64)
float li = 0.0f;              // running sum of exp2
float real_mi = -CUDART_INF_F;// the true max, tracked separately (:192)
...
float cur_pi_max = get_max<N>(p) * params.sm_scale_div_log2;     (:216-217)
// cross-half row reduction through smem, since the 64 rows are split over 2 half-warps
plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;             (:219)
cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup^64]);  (:222)
real_mi = max(real_mi, cur_pi_max);
bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);     (:224)
...
if (!should_scale_o) { scale_for_old = 1.0f; new_max = mi; }
else { new_max = max(cur_pi_max, mi); scale_for_old = exp2f(mi - new_max); }   (:231-238)
// S = exp2(p·sm_scale·log2e − new_max), accumulate its sum, write bf16 S to smem
d = fma(p, scale, -new_max); d = exp2f(d); cur_sum += d; s = bf16(d);          (:245-252)
li = fma(li, scale_for_old, cur_sum.x + cur_sum.y);                            (:253)
if (block_idx != args.start_block_idx && should_scale_o) rescale_O<...>(scale_for_old);  (:262-271)
```

Details that are pure gold for teaching:

- **Base 2 everywhere.** `sm_scale_div_log2 = sm_scale · log2(e)` is precomputed on the host (`params.h:57`, `sparse_decode.cpp:349`, `LOG_2_E = 1.44269504f` at `common.h:17`), so the kernel uses `exp2f`/`log2f`, which map to single hardware instructions (`MUFU.EX2`, `MUFU.LG2`). Conversion back to natural log happens once, at the end (`kernel.cuh:296`, `combine.cu:129`).
- **`MAX_INIT_VAL = -1e30f` "To avoid (-inf) - (-inf) = NaN"** (`config.h:64`). A finite sentinel instead of `-inf`, because the first block computes `exp2f(mi - new_max)` before anyone knows whether any token was valid.
- **Two maxima.** `mi` is the max *used for scaling*; `real_mi` is the true max, used only to detect "no valid token at all" (`:281-286`): then `li = 0, mi = -inf`, which produces `lse = +inf` and an all-zero output — matching the documented edge case (`README.md:206`: "A query token that has no valid index … returns `max_logits = -inf`, `lse = +inf` and an all-zero output … whereas the pseudo-code would produce a NaN").
- **skip-scale is warp-uniform.** `__any_sync` makes the decision identical across the warp so the branch never diverges, and the comment at `:226-227` documents the invariant ("`should_scale_o` is identical among every warp, and is identical among threads that controls the same row").
- **Rescaling the output is the expensive half.** `rescale_O` (`common_subroutine.h:141-173`) loads O out of TMEM in chunks, multiplies, stores back. For `h_q=64, d_v=512` that is 32 k f32 values touched per rescale. Skipping it when the max grew by < 6 (in log2 ⇒ ≤ 2^6 = 64×) is worth 10-15 % on Ascend by their account [docs: 20260930:244-248 lists the benefits: fewer FixPipe copies, less VF compute, lower power ⇒ higher clocks].
- **Unit subtlety to flag:** the CUDA kernel compares in log2 units (`cur_pi_max - mi > 6.0f` with `exp2`), the Ascend doc phrases the same threshold as `exp(6)` in natural units (`:63`). Same constant 6, different base ⇒ a 64× vs 403× slack [inference]. Do not let Luigi "derive" one from the other.

## 6.3 Worked example for M3 (tiny, on paper)

Same numbers as §5.5, one pass, tiles of 2, `RESCALE_THRES = 6`:

| step | p tile | `cur_max` | `> mi+6`? | `mi` | `scale_for_old` | `s = 2^{p−mi}` | `li` |
|---|---|---|---|---|---|---|---|
| init | | | | −1e30 | | | 0 |
| tile 0 | [1, 3] | 3 | yes (first) | 3 | (skipped: first block) | [0.25, 1] | 1.25 |
| tile 1 | [2, 5] | 5 | **no** (5−3=2 ≤ 6) | 3 | 1.0 | [0.5, 4] | 5.75 |

Final `o = 0.25v1 + 1v2 + 0.5v3 + 4v4`, `out = o/li`, weights `[0.0435, 0.1739, 0.0870, 0.6957]` ✓ (checked against `2^p/46`). Note tile 1 produced `s = 4 > 1` — allowed, because the guarantee is only `s ≤ 2^6 = 64`, comfortably inside bf16's range. Show him that the *naive* version (rescale every tile) gives the same answer with one extra pass over `o`; the skip made no difference to the result, only to the work.

For M3's CPU decode step, note what llama2.c does instead (`run.c:290-318`): fill `att[0..pos]`, call `softmax(att, pos+1)` (two passes: max then exp/sum, `run.c:197-215`), then the weighted sum. **That is fine and simpler**, because the whole score row is in memory already. Online softmax earns its place when (a) you tile the KV axis so you never materialize the full score row (M6 batched prefill), or (b) you split the KV axis across threads (§5.6). Teach two-pass first, then show online softmax as the *enabling* trick for tiling, not as an optimization of the scalar case.

**Pitfalls to plant deliberately:** `exp(x - max)` with `max` from the *wrong* row (a classic when the score matrix is tiled 2-D); forgetting to rescale the accumulator when the max changes; using `-inf` as the initial max and hitting `inf - inf`; summing `exp` in bf16 (`li` must be f32 — note `li` here is f32 while `s` is cast to bf16 only for the MMA input, `:242-252`).

**Applies to:** CPU engine (M1 softmax, M3 attention, M6 tiling), sm_89 CUDA (identical code shape; `exp2f` is `MUFU.EX2` on sm_89 too).

---

# 7. Deep dive: kernel-level techniques, and the sm_89 translation

## 7.1 Warp specialization (the structure of the h_q=64 decode kernel)

`NUM_THREADS = 128*3` with the roles spelled out in a comment (`config.h:63`): "128 exp (wg0) + 1/32 utcmma + 1/32 raw KV producer + 32 index+valid_mask producer (wg1, its warp 6 is idle) + 128 dequant (wg2)".

| warpgroup | warps | job | registers |
|---|---|---|---|
| 0 | 0–3 | softmax (scale/exp/row-max/row-sum), rescale O, epilogue | `warpgroup_reg_alloc<224>()` (`kernel.cuh:168`) |
| 1 | 4 | issue the two UMMAs (QK, then SV) | `warpgroup_reg_dealloc<72>()` (`:405`) |
| 1 | 5 | issue `tma_gather4` for the raw KV tile | ” |
| 1 | 7 | index → TMA coordinate + validity mask | ” |
| 1 | 6 | idle | ” |
| 2 | 8–11 | dequantize fp8/fp4 → bf16 into the shared K tile | `warpgroup_reg_alloc<208>()` (`:621`) |

Pipelining: `NUM_BUFS = 2` KV buffers, `NUM_INDEX_BUFS = 4` index buffers (`config.h:49-50`), with a `RingState` that flips buffer index and mbarrier phase together (`kernel.cuh:151-162`). The barrier set (`config.h:166-171`) is a textbook producer/consumer chain: `bar_raw_ready/free`, `bar_quant_part_dequant_ready`, `bar_qk_done`, `bar_so_ready`, `bar_sv_done`, `bar_valid_coord_ready/free`. Five named barriers for intra-warpgroup syncs (`config.h:22-28`).

**sm_89 translation.** The *structure* ports; the mechanisms do not:
- Warp specialization itself: yes. Use `__syncthreads()` plus PTX `barrier.sync <id>` / cutlass-style named barriers (sm_80 supports 16 per CTA) so producer and consumer warps sync without a full block barrier.
- `setmaxnreg` (`warpgroup_reg_alloc/dealloc`): **sm_90+ only.** On sm_89 all warps share one budget set by `__launch_bounds__`; the dequant/softmax split must be register-frugal or spill. (FlashMLA's build *fails* on any spill — `setup.py:113-141`. Adopt that gate in M7: `--ptxas-options=-v,--warn-on-spills` is in their nvcc flags at `setup.py:173`.)
- mbarrier with transaction counts (`arrive_and_expect_tx`): sm_90+ semantics. On sm_89 use `cp.async.commit_group` / `cp.async.wait_group N` for a multi-stage pipeline, or `cp.async.mbarrier.arrive` with sm_80 mbarriers.

## 7.2 "Seesaw" scheduling (Hopper-only, but the reasoning is the lesson)

`docs/20250422-new-kernel-deep-dive.md:19-37`. Why FlashAttention-3's ping-pong does not apply: the WGMMA accumulator must live in registers, and "Each `64 × 512` output matrix occupies 32,768 32-bit registers. With only 65,536 32-bit registers per SM, we can store only one output matrix per SM" (`:19`). So they cannot hold two output tiles and alternate. Their answer: split `O` vertically into `O_L`/`O_R` (64×256 each, one per warpgroup), take **two** KV blocks per step, and interleave 11 micro-steps so that while warpgroup 0 is doing CUDA-core softmax on `p_0`, warpgroup 1 is doing tensor-core work on `p_1`, sharing one running max `m` (`:21-33`). "This schedule can be viewed as a 'ping-pong' variant using one output matrix—we call it 'seesaw' scheduling" (`:35`). Diagram: `docs/assets/MLA Kernel Sched.drawio.svg`.

On sm_100 this problem *disappears*: the accumulator lives in **TMEM**, not registers (`config.h:78-85`: O at tmem col 0, Q at 256, P at 400; 512 columns allocated at `kernel.cuh:81`), which is why the current kernel uses plain warpgroup specialization instead of seesaw.

**sm_89 relevance [inference]:** on sm_89 the `mma.sync` accumulator lives in registers, so the Hopper constraint is *back*, and worse (64 KB of registers per SM, 255 per thread). For MLA-like `d_v = 512` you must tile the output columns. For Luigi's models (`head_dim` 48–128, and `h_q` rows only when several heads share a K tile) the accumulator is tiny, so this is a non-issue — but it is the right lens: **always ask where the accumulator lives and how big it is.** That question also decides his CPU register blocking in M6 (how many accumulator `__m256`s fit in 16 YMM registers).

## 7.3 Overlapping CUDA-core softmax with tensor-core matmul

Two mechanisms in the current kernel:
1. **Different warpgroups**: the MMA warp (4) issues `utcmma_ts`/`utcmma_ss` and immediately waits on the *next* buffer's barrier, while warpgroup 0 does `exp2f` on the previous tile (`kernel.cuh:481-500` vs `:194-279`).
2. **Dual-rail QK GEMM.** `TiledMMA_P` uses `B_TOPK*2` as N "for dual gemm" (`config.h:174-176`), splitting the 512 QK dims into halves; the two partial score matrices land in TMEM rows 0–63 and 64–127 and are **summed in shared memory** by `retrieve_mask_and_reduce_p` (`common_subroutine.h:67-139`), with the invalid-token mask applied *before* the sum (`:103-119`). The comment at `:472-479` draws the interleaving of the RoPE half into the dual-GEMM view.

**sm_89 translation:** split-K within a warp's `mma.sync` chain is normal; accumulate in registers and reduce across warps via shared memory — same pattern, smaller scale. Overlap of CUDA-core and tensor-core work on sm_89 comes from **instruction-level** mixing inside a warp (the SM co-issues MMA and ALU from different warps), not from named warpgroup roles. Concretely: give each warp its own KV tile stage, so while warp A's `mma.sync` results are being `exp2f`'d, warp B's `mma.sync` is in flight.

## 7.4 Loading the KV cache

**Hopper/Blackwell (in-tree).**
- `tma_gather4`: one instruction gathers 4 non-contiguous rows into shared memory with mbarrier completion (`kernel.cuh:519-529`, wrapper at `kerutils/device/cuda/sm100/intrinsics.cuh:12`). 16 of them per 64-token tile, then one `arrive_and_expect_tx(B_TOPK * RAW_TOKEN_SMEM_STRIDE)` (`:529`).
- **Fine-grained TMA↔GEMM pipelining** [docs: 20250422:51]: "For a `64 × 576` K block, we launch 9 TMA copies (each moving a `64 × 64` block). GEMM operations begin as soon as each TMA copy completes".
- **L2 cache hints:** `TMA::CacheHintSm90::EVICT_FIRST` for both Q and KV (`kernel.cuh:423`, `:525`) — streaming data should not evict reused data. [docs: 20250422:53] "improves L2 cache hit rates, as shown by experiments".
- **L2 promotion** `CU_TENSOR_MAP_L2_PROMOTION_L2_128B` on the KV tensormap (`kv_cache_utils.cuh:44`).
- **Prefetch of TMA descriptors** at kernel start (`kernel.cuh:55-59`).
- **Index prefetch**: warp 7 loads block *i+1*'s indices while processing block *i* (`kernel.cuh:597-613`), and the KV producer warp prefetches the next `int4` of coordinates inside its gather loop (`:513-528`).

**sm_89 equivalents [inference, with repo evidence for the primitives]:**

| FlashMLA mechanism | arch | sm_89 replacement |
|---|---|---|
| `cp.async.bulk.tensor` / TMA + tensormap | sm_90+ | `cp.async.cg.shared.global.L2::128B [dst],[src],16` per thread — literally `kerutils/.../sm80/intrinsics.cuh:10-31`, which supports 64B/128B/256B L2 prefetch and a predicate |
| `tma_gather4` (4 gathered rows, 1 instr) | sm_90+ | one thread (or 4 lanes) per token issuing `cp.async` of 16 B chunks; 528 B/token = 33 chunks ⇒ assign 33 lanes or loop 16B×2 per lane |
| mbarrier transaction accounting | sm_90+ | `cp.async.commit_group` + `cp.async.wait_group N` for an N-stage pipeline |
| TMA zero-fill of out-of-range coords | sm_90+ | `cp.async` has a predicate (`pred=false` writes nothing) — use `SM80_CP_ASYNC_CACHEALWAYS_ZFILL`-style zero-fill, which the repo itself uses for scales at `kv_cache_utils.cuh:36` |
| `EVICT_FIRST` / L2 policy | sm_80+ | available: `createpolicy` + `ld.global.L2::cache_hint`, wrapped at `sm80/intrinsics.cuh:35-95` |
| UTCMMA / TMEM (`tcgen05`) | sm_100 only | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, operands staged via `ldmatrix.sync.aligned.m8n8.x4` |
| WGMMA (`wgmma.mma_async`) | sm_90 only | same as above |
| 2-CTA clusters, DSM, `st.async`, cluster barriers | sm_90+ | none. No cluster on sm_89 ⇒ no cross-CTA shared memory (see §7.5) |
| PDL (`cudaTriggerProgrammaticLaunchCompletion`) | sm_90+ | back-to-back kernels on one stream (eat the ~2–5 µs), or fuse the combine into the attention kernel with an atomic "last CTA does the merge" pattern |
| `setmaxnreg` | sm_90+ | none; single register budget per CTA |
| FP8 e4m3 → bf16 with `.scaled::n2::ue8m0` | sm_100+ (`setup.py:24-27`) | `cvt.rn.f16x2.e4m3x2` (native on sm_89) then scale, or the 4-step Hopper path [docs: 20250929:21-27] |
| FP4 e2m1 → bf16 | sm_100 (`cvt` for e2m1) | 16-entry LUT via `__byte_perm`/shifts (`tests/quant.py:29` has the magnitude table) |

## 7.5 The Hopper "crossover" trick (why you should read it even though it doesn't port)

`docs/20250929-hopper-fp8-sparse-deep-dive.md` is the best short case study of a *quantitative* kernel diagnosis in the repo, and it maps 1:1 onto the method for role project 4:

1. **Budget per unit of work in cycles.** "Each SM can process 4096 MMA Flops per clock cycle (calculated as `989 TFlops / 1830 MHz / 132 SMs` on H800)" ⇒ 64 heads × (576+512) × 2 / 4096 ≈ **34 cycles of MMA per KV token** (`:17`).
2. **Budget the competing work.** H800 cannot cast e4m3→bf16 directly, so dequantizing one token costs `(1/64 + 1/64 + 1/16 + 1/256) × 512 ≈ 50 cycles` from NVIDIA's published instruction throughputs (`:21-27`). 50 > 34 ⇒ "the kernel is **dequantization-bound**".
3. **Find the algebraic escape.** Every query head in a token attends the *same* KV (MQA), so two CTAs handling different head halves can share dequantized KV: each dequantizes half and ships it to its partner. They named it "crossover" after chromosomal crossover (`:31-33`).
4. **Use the hardware feature that makes it possible.** Hopper's Distributed Shared Memory: clusters of 2, `__ldg` 128-bit loads, dequantize, store to own smem *and* `st.async` into the partner's smem, sync with a cluster transaction barrier (`:37-45`).
5. **Report the delta with the configuration:** 410 TFLOPS vs 250 TFLOPS without crossover, at `batch=128, heads=128, s_q=2, topk=2048` on H800 SXM5 (`:48`).

In the current sm100 tree the same "two CTAs, one token" idea survives as a **2-CTA cluster with `2x1SM` UMMA**: `SM100_MMA_F16BF16_2x1SM_TS_NOELECT<..., H_Q=128, B_TOPK*2, ...>` (`fwd_for_small_topk/head128/config.h:130-138`), with per-CTA halves of each token (`KVCachePart<F, 2, CTA>`, `kv_cache_format.h:24-35`: CTA0 takes the first half of the bytes and stages scales separately, CTA1's TMA view includes the trailing scales) and a permutation layout so "CTA0 takes V[:, 0:256] and CTA1 takes V[:, 256:512]" (`config.h:137`).

**sm_89:** no clusters, no DSM ⇒ the crossover trick has **no equivalent**. The sm_89 answer to a dequant-bound decode kernel is different [inference]: (a) sm_89 *has* native fp8→fp16 conversion (`cvt.rn.f16x2.e4m3x2`), so the 4-step Hopper penalty largely disappears; (b) give one CTA all the heads that share a K tile so the dequantized tile is amortized over more MMA work; (c) or keep the cache in bf16 and quantize only weights. Mark the whole section as "read for the method, not the mechanism".

## 7.6 Shared-memory layout and bank conflicts

Three distinct tricks, all worth teaching because two of them have CPU analogs:

1. **Swizzled layouts for MMA operands.** `UMMA::Layout_K_SW128_Atom<bf16>` tiled to shape (`config.h:88-139`) — the 128-B XOR swizzle that lets `ldmatrix`/UMMA read a K-major tile conflict-free. The dequantizer computes the swizzled destination address by hand: `dst_offset = group_idx/8*(8*128) + row_in_atom*128 + (idx_in_group ^ row_in_atom)*16` (`dequant_utils.cuh:64-67`) — "the 16 B lane index is XORed with the row's position in the atom".
2. **Padding to break a bad stride.** `SmemLayoutOAccumBuf` uses `Stride<Int<520>, _1>` for a 512-wide f32 tile — "We use stride = 520 here to avoid bank conflict" (`config.h:107-110`). Likewise the raw KV row stride is `QUANT_BYTES + 64` = 576 B for the fp8 format, chosen so "adjacent rows start 64 B apart modulo 128 B, avoiding bank conflicts for the dequantizer's LDS.64" (`config.h:54-57`), and the fp4 case asserts `RAW_TOKEN_SMEM_STRIDE/4 % 16 == 8` (`dequant_utils.cuh:44`).
3. **Reordering thread→row mapping** instead of padding: `group_idx = (group_idx & ~3) | ((group_idx & 1) << 1) | ((group_idx & 2) >> 1)` gives "row order 0,2,1,3 in each warp" for one stride case (`dequant_utils.cuh:29-30, 58-60`).

**CPU analog (M6):** a 512-float row stride (2048 B) is a power of two, so rows alias in the same cache sets — pad the stride (the identical trick to `520`). Also the same lesson for the KV cache layout: `[layer][pos][dim]` with a power-of-two `dim` can alias across positions; measure with `perf stat -e cache-misses` before and after padding. **sm_89:** identical bank-conflict arithmetic (32 banks × 4 B); `ldmatrix` exists on sm_89 (since sm_75), so the swizzle pattern is directly reusable.

## 7.7 Small but professional details worth copying

- **Register spilling is a build error.** `setup.py:113-145` disassembles the built `.so` and fails on any spill from this repo's own kernels (CUTLASS exempt); `FLASH_MLA_SKIP_REG_SPILL_CHECK=1` to bypass [docs: README.md:105]. nvcc flags include `--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage` (`setup.py:173`).
- **Comments that record *negative* results.** `kernel.cuh:96` "NOTE Putting the following code outside the warpgroup specialization switch results in register spilling"; `:731` "NOTE Don't use PDL because of potential compiler bugs!"; `phase1.cuh:41` "if we use kerutils::st_shared, warpgroup 0 suffers from register spilling. However, if we change to this dump implementation with `__cvta_generic_to_shared`, there is no register spilling"; `phase1.cuh:406` "Missing this leads to reg spilling". This is the single most imitable habit in the repo: **when a rewrite was forced by the compiler, the comment says so.** Tell Luigi to do this in his `unsafe`/SIMD blocks alongside the `// SAFETY:` note.
- **Every kernel file has a header block** stating purpose, template parameters, grid/block shape, and "I/O: See <struct> in csrc/params.h" (e.g. `kernel.cuh:1-18`, `combine.cu:1-24`, `get_decoding_sched_meta.cu:1-19`). A one-paragraph contract per file. Cheap, and exactly what ARCHITECTURE.md should contain per module.
- **All parameters in one struct per op** (`csrc/params.h`), passed by `__grid_constant__ const` value (`combine.cu:45`, `kernel.cuh:665`). Strides are `int` with an explicit `int64_stride_to_int` conversion at the boundary (`sparse_decode.cpp:366-375`) — a deliberate 32-bit-indexing decision, guarded by `KU_ASSERT(... <= INT32_MAX ...)` (`kv_cache_utils.cuh:52`).

**Applies to:** sm_89 CUDA (most of §7), CPU engine (§7.6 padding, §7.7 practices).

---

# 8. Deep dive: the FP8 / FP4 KV cache

## 8.1 Format (current release)

`csrc/cuda_kernels/kv_cache_format.h:5-20` encodes the layout as a compile-time type:

```c
// Each V4.1 token stores [raw data, 1-byte scales], including the RoPE dims:
// FP8: 512 B e4m3 + 16 B ue8m0; FP4: 256 B e2m1 + 32 B e4m3. See tests/quant.py.
D_QK = 512, D_ROPE = 64, QUANT_TILE_SIZE = IS_FP4 ? 16 : 32
NUM_SCALES_EACH_TOKEN = D_QK / QUANT_TILE_SIZE     // 16 (fp8) / 32 (fp4)
BYTES_PER_TOKEN = QUANT_BYTES + NUM_SCALES_EACH_TOKEN  // 528 / 288
```

- **V4.1 fp8, 528 B/token:** 512 `e4m3` covering *all* 512 dims **including RoPE**, then 16 `e8m0` (power-of-two-only) scales, one per 32 consecutive values [docs: README.md:140].
- **V4.1 fp4, 288 B/token:** 256 B holding 512 `e2m1` (2 per byte, even index in the low nibble), then 32 `e4m3` scales, one per 16 values. Only legal as the `extra_k_cache` beside an fp8 main cache [docs: README.md:141; enforced at `sparse_decode.cpp:287-288`, `kv_cache_format.h:47-49`].
- Format is **detected from the tensor's last dimension** (`detect_kv_cache_format_for_headdim_512(kv.size(3))`, `sparse_decode.cpp:285`). Cute, and a self-describing-data lesson.
- The *interleaving* of data and scales per token is a deliberate change from the previous release, and the Ascend doc says why [docs: 20260930:218-220]: the old layout put all of a block's data first, then all of its scales, which "is not friendly to the Huawei Ascend: each token needs two MTE2 copies". Interleaving lets one gather fetch both. Note the cost: `RAW_TOKEN_SMEM_STRIDE` must then be padded to 576 B in shared memory (`config.h:57`).

**Contrast with the Hopper-era format** [docs: 20250929:9]: tile size 1×128 over the first 512 dims → 512 `e4m3` + 4 `f32` scales, and the **64 RoPE dims kept in bf16** "as they are sensitive to precision loss" ⇒ 656 B/token. The new format quantizes RoPE too, with a finer tile (32) and a cheaper scale type (`e8m0`, 1 B). Excellent material for an M5 discussion: *group size vs scale-type vs what you refuse to quantize*, and how DeepSeek's own answer changed between releases.

## 8.2 Scale granularity, quantization math

`tests/quant.py:58-136`:
- fp8: per 32-element tile, `scale_inv = amax/448`, then **rounded up to a power of two**: `torch.pow(2, clamp_min(s,1e-4).log2().ceil())` (`:26-27`) so it fits `e8m0`; the comment ties the `1e-4` to "Tile Kernel's FP8_AMAX_MARGIN". 448 is e4m3's max finite magnitude.
- fp4: per 16-element tile, `scale = clamp(amax/6, 2^-9, 448)` cast to `e4m3` (6 = max e2m1 magnitude), with NaN tiles poisoned deliberately (`:113-119`).
- Dequantization (`:139-194`) is a plain `value * scale` in bf16, with a precision note at `:187`: "e2m1 x e4m3 has at most 2 + 4 significant bits, so the product is exact in bf16, as in the kernel" — the same claim appears in the kernel (`dequant_utils.cuh:96`). That is a *proof* that dequant introduces no extra rounding, which is why they can compare against a reference that dequantizes in PyTorch and still use a tight tolerance.

## 8.3 Where dequantization happens

In-kernel, in a dedicated warpgroup, into shared memory, before any MMA:
- `KVBlockDequantizer` (`dequant_utils.cuh:32-114`): 8 lanes per token, each converting 64 elements per step (`ELEMS_PER_STEP = 64`), reading the raw row with `LDS.64` (fp8) or `LDS.32` (fp4), and writing bf16 with `st.weak.shared::cta.b128` into the swizzled tile (`:107-110`).
- The scale for a step is picked with a compile-time base plus a per-thread part; for fp4 the scale byte is extracted with **one `PRMT`** and replicated into both halves of a `bf16x2` (`:45-47`, `:93`).
- Warpgroup 2 drives it (`kernel.cuh:619-654`): wait `bar_raw_ready`, wait `bar_sv_done` (the tile's previous consumer is done), dequantize, `fence_view_async_shared()`, signal `bar_quant_part_dequant_ready` and `bar_raw_free`.
- The MMAs then consume **bf16** (`TiledMMA_P/O` are `F16BF16` with f32 accumulate, `config.h:174-181`), exactly as the Hopper blog describes [docs: 20250929:11].

So the KV cache is quantized **in memory only**; the math is bf16×bf16→f32. That is the key design statement for M5.

## 8.4 Accuracy impact — what is and isn't documented

- No end-to-end quality number (no perplexity, no logit drift) is given for the KV quantization in this repo. The motivation is stated as memory pressure: 8.72 GiB for one 128 k request ⇒ "OOM errors or under-utilized GPUs due to small batch sizes" [docs: 20250929:3].
- The tests **hide** quantization error on purpose: `KVScope.quant_and_dequant_` quantizes the cache and then feeds the *dequantized* values to the PyTorch reference (`tests/lib.py:256-266`): "the quantization error may be too large to be distinguished from wrong kernels, so we de-quantize kvcache here to mitigate quantization error". The kernel test therefore measures *kernel* error against the same quantized inputs, not quantization error.
- The only accuracy-driven design statements are qualitative: RoPE dims were left unquantized in the old format because they are "sensitive to precision loss" [docs: 20250929:9]; `e2m1 × e4m3` is exact in bf16 [code: `quant.py:187`].

**Teaching hook (M5).** This is a two-decision milestone, and FlashMLA separates them cleanly:
1. **Quantize weights** (llama2.c `runq.c` Q8_0-style, group 64) — pure throughput win on a weight-bound model (§3.3).
2. **Quantize the KV cache** — a *capacity* win, and on a small model a negligible throughput win (§3.3: KV is 6 % of the traffic at `L=256`). So M5's KV quantization should be justified as preparation for the server, and its benefit measured at long context, not at 256 tokens.
And copy their measurement protocol exactly: **test the kernel against a reference fed the dequantized values** (isolates kernel bugs), then *separately* measure the quantization's quality drift against the f32 engine (isolates format error). Two experiments, two numbers. D1's structure already anticipates this ("quantization (M5) and the GPU (M7) will get their own tolerances, measured the same way").

**Pitfalls:** NaN handling in a tile (fp4 poisons the whole tile's scale, `quant.py:113`); scales rounded *up* to powers of two lose up to 1 bit of range but make dequant a multiply-free exponent add; `e8m0` has no sign/mantissa so `clamp_min` is needed before `log2().ceil()`; an out-of-range value maps to NaN in torch's fp8 cast, hence the clamp at `quant.py:114` with the comment "torch maps overflow to NaN".

**Applies to:** CPU engine (M5 formats, group size, the two-experiment protocol), sm_89 CUDA (fp8 has native conversion; fp4 needs a LUT).

---

# 9. Deep dive: the sparse (DSA) kernels

## 9.1 Inputs and semantics

`indices`: `[batch, s_q, topk]` int32 for decode (`README.md:148-151`), `[s_q, h_kv, topk]` for prefill (`:169`). Values are pre-flattened cache row ids (§4.2); invalid = `-1` (decode) or `-1`/`≥ s_kv` (prefill, `:181`). Optional `topk_length: [b]` lets each request use only the leftmost *k* indices "so that we can save some computation, compared to masking" (`flash_mla_interface.py:109`). Optional `attn_sink: [h_q]` scales the output by `exp(lse)/(exp(lse)+exp(sink))` (`:107`). Optional `extra_k_cache` + `extra_indices_in_kvcache` add a second cache (possibly fp4) attended in the same pass (`:108`).

The exact semantics are given as runnable PyTorch in the README (`:185-205`) — gather, scale, mask invalid to `-inf`, `max_logits`, `logsumexp`, softmax, `S @ focused_kv` — plus the documented divergence for a query with no valid index (`:206`). **Publishing the reference implementation in the README** is a practice worth stealing for the engine's kernels.

Where the top-k comes from (the "indexer") is *not* in this repo; the sparse kernels take the indices as data. Clean separation: the selection policy is a model concern, the gather is a kernel concern.

## 9.2 How sparsity changes memory access

- Access granularity becomes **one token (528 B) at a random row**, not a contiguous block. Hence `tma_gather4` (4 rows per instruction), and on Ascend the `gather2` trick (`burst_count=2` with a computed `src_stride` so one MTE2 request fetches two tokens, because the issue queue holds only 16 outstanding requests) [docs: 20260930:214-216].
- **L2 becomes the real bandwidth source.** The benchmark counts *unique* tokens (`lib.py:474-483`), acknowledging that duplicate selections hit cache. The Ascend analysis budgets against **L2** bandwidth (5 TB/s total, divided by 32 cores) rather than HBM [docs: 20260930:85-90].
- Prologue/epilogue overhead stops being negligible: "With a smaller topk, the relative overhead of the kernel's prologue and epilogue becomes larger compared with dense decoding with long context length" — 410 TFLOPS at topk=2048 vs 460 at topk=32768 [docs: 20250929:50].
- A separate code path exists for **small topk** (`fwd_for_small_topk`, chosen when `topk <= 1280`, `sparse_prefill.cpp:207`) — evidence that one kernel cannot span the whole topk range efficiently.
- The economic argument [docs: 20250929:52]: the FP8 sparse decode kernel's runtime at topk=2048 "is comparable to that of the dense decoding kernel when the sequence length is around 3000" — so DSA pays off beyond ~3 k context.

**Applies to:** neither, directly — Luigi's v1 has no sparse attention. But two transferable ideas: (a) *the gather is the kernel's whole memory behaviour*, so batch it (4 rows/instr, 2 tokens/request) rather than issuing one load per element; (b) a **separate kernel for the small-N regime** is a legitimate answer, not a failure — the same reason M6 will want a different code path for prefill (matrix×matrix) and decode (matrix×vector).

---

# 10. Deep dive: how they test

## 10.1 The reference

`tests/ref.py` — pure PyTorch, f32 accumulation, gather-then-dense (`ref_sparse_attn_decode`, `:60-112`): clamp indices, `index_select`, `q.float() @ gathered.transpose`, scale, mask invalid to `-inf`, `logsumexp`, `exp(P - lse)`, `@ V`, apply sink, then fix the lonely-query case (`output=0`, `lse=+inf`, `:107-110`). The prefill reference additionally returns `max_logits` and merges the sink into the LSE with a `logsumexp` of two terms (`:7-17`, `:46-57`).

Note `gathered_kv.masked_fill_(gathered_kv != gathered_kv, 0.0)` (`:89`) — NaN-scrubbing the gathered values, because invalid rows may contain garbage.

## 10.2 The comparison function

`tests/kernelkit/compare.py:44-105`, `check_is_allclose(name, ans, ref, abs_tol, rel_tol, cos_diff_tol)`:

1. **Anomaly structure first** (`:60-73`): for each of `+inf, -inf, NaN`, check the *positions* match exactly, then zero them out. A NaN in the wrong place fails immediately and prints counts.
2. **Per-element pass if EITHER tolerance passes** (`:76-80`):
   ```python
   rel_err = raw_rel_err.masked_fill(raw_abs_err < abs_tol, 0)
   abs_err = raw_abs_err.masked_fill(raw_rel_err < rel_tol, 0)
   pass_mask = (abs_err < abs_tol) | (rel_err < rel_tol)
   ```
   (`raw_rel_err = |a-r| / (|r| + 1e-6)`.)
3. **A whole-tensor cosine-difference gate** (`:27-42`, `:102-104`): `1 - 2·⟨a,r⟩/(‖a‖²+‖r‖²)`, computed in float64, compared against `cos_diff_tol`. This catches a *systematic* small bias that per-element tolerances would wave through.
4. On failure it prints max abs err with its index, max rel err with its index, the pass percentage, and the cosine diff (`:86-99`).

Tolerances actually used:

| test | tensor | abs_tol | rel_tol | cos_diff_tol | ref |
|---|---|---|---|---|---|
| sparse decode | `out` (bf16) | 1e-3 | 2.01/128 ≈ 0.0157 | 5e-6 | `test-sparse-decode.py:254` |
| sparse decode | `lse` (f32) | 1e-6 | 8.01/65536 ≈ 1.2e-4 | — | `:255` |
| sparse prefill | `out` | 8e-4 | 3.01/128 | 1e-5 | `test-sparse-prefill.py:38` |
| sparse prefill | `max_logits`, `lse` | 1e-6 | 2.01/65536 | — | `:39-40` |
| dense MHA | `out`, `dq/dk/dv` | 1e-3 | 8.01/128 | 7e-6 | `test_fmha_sm100.py:125, 130-132` |

`x.01/2^n` is the tell: the relative tolerances are **derived from ULPs** (bf16 has 8 mantissa bits ⇒ `2^-7 = 1/128`; f32-ish `1/65536`), with `.01` added so the comparison is strict-less-than at exactly `x` ULPs. When the test injects deliberately huge K values (`k_amplifier_portion`), tolerances are relaxed to `1.0/1.0` — i.e. that case checks only the anomaly structure and cosine diff (`test-sparse-prefill.py:38-40`).

## 10.3 Test shapes and edge cases

`tests/test-sparse-decode.py:27-150` builds a Cartesian product, ~thousands of cases:
- `h_q ∈ {64, 128}`; `s_q ∈ {1, 3}` (decode *and* MTP); `b ∈ {1, 4, 74, 321}`.
- `(s_k, topk, block_size)` combos including `(512, 64, 5)`, `(512, 64, 69)`, `(1024, 576, 2)`, `(2046, 2048, 1)`, `(2046, 2048, 576)` — i.e. **page sizes of 1, 2, 5, 61, 69, 123, 576**, and `topk > s_k` (2048 > 2046, so most indices are invalid).
- `is_varlen` toggles; a comment records why one combination matters: "With b=74 and no varlen/topk_len/extra_topk_len, no request should be split-kv" (`:75`) — i.e. a case chosen to exercise the *non*-split path.
- Corner-case block: `is_all_indices_invalid`, `have_zero_seqlen_k`, `enable_attn_sink` (including `±inf` sinks, `lib.py:294-298`), `have_topk_length`, `have_extra_topk_length`, and `extra_topk=100` (not a multiple of `B_TOPK=64`) with a comment explaining which kernel handles it (`:114-127`).
- Q values are clamped to `[-1, 1]` (`lib.py:290`) and tensors deliberately made non-contiguous (`kk.non_contiguousify`, `:291`) to test stride handling.
- **Ordering discipline:** run the kernel *first* and keep its output, before allocating reference tensors, "otherwise when allocating tensors for storing answers, it may re-use memory that contains the correct answer, leading to false positives" (`test-sparse-decode.py:183-188`). And the perf run happens before the reference is generated "to avoid interference" (`:191`).
- **Determinism:** the dense test runs forward 5× and asserts `torch.equal` on `out` and `lse` (`test_fmha_sm100.py:141-149`).
- Failure handling: exit on first failure unless `--run-to-finish`, then print every failing `TestParam` so it can be pasted back as a repro (`:299-301`, `:346-349`).

## 10.4 Compare with D1 (the learner's tolerance decision)

D1: `|ours − ref| ≤ atol + rtol·|ref|` with **atol = 5e-4, rtol = 0**, measured — 9× above the observed worst noise of run.c-vs-PyTorch (5.5e-5) and 560× below the smallest planted bug; plus three buckets (PASS / PASS near-tie / FAIL) and a near-tie rule based on the reference's top-2 gap.

| | D1 | FlashMLA |
|---|---|---|
| tolerance chosen by | **measurement** (20 prompts × 256 positions × 32000 logits, worst case) + planted bugs | ULP arithmetic (`2.01/128` etc.), not measured |
| combination | `atol + rtol·|ref|`, rtol = 0 | pass if **either** abs or rel passes (more permissive per element) |
| global check | none yet | **cosine diff** over the whole tensor |
| special values | n/a (logits are finite) | inf/NaN position equality, checked first |
| what is compared | logits at every position | `out`, `lse`, `max_logits`, gradients |
| determinism | fixed seed, one command | explicit `torch.equal` across runs; `enable_batch_invariant` flag in the kernel |

**Two things to pull into the engine.** (1) The **cosine-diff gate**: it catches a uniform small bias (e.g. a slightly wrong scale factor) that a per-element atol of 5e-4 on logits would pass. Cheap to add to the M3 checker, and it is a different *kind* of evidence. (2) **inf/NaN structural equality** — as soon as M4's sampling or M5's quantization can produce `-inf` masks, "did the infinities land in the same places" becomes a real check. (3) Their `rel_tol` is worth a decision entry as an *alternative rejected with a reason*: it is principled for bf16 kernels (error scales with magnitude) but D1 already measured that this engine's error does *not* scale with logit magnitude, so rtol = 0 is the better-evidenced choice. That contrast is exactly what a DECISIONS.md entry should look like.

**Applies to:** CPU engine (M3 checker, M5/M7 tolerances), sm_89 CUDA (M7: reference = the CPU engine, same three-part check).

---

# 11. API design through the Ousterhout lens

## 11.1 The shape of the interface

```python
tile_scheduler_metadata, num_splits = get_mla_metadata()          # once per shape
for layer in range(num_layers):
    o, lse = flash_mla_with_kvcache(q, kvcache, block_table, cache_seqlens, dv,
                                    tile_scheduler_metadata, num_splits, indices=indices)
```
[docs: README.md:111-124]

**Why two calls at all:** the scheduling plan depends on the batch shape and the per-request lengths, not on the layer, so it is computed once and reused across ~60 layers. That is real: it removes a 1-block/32-thread kernel launch plus two allocations from every layer. The cost is a **temporal coupling** the user must respect.

## 11.2 Is it a deep module?

**Deep, in the good sense:** one function hides a scheduler kernel, a split/no-split decision, FP8/FP4 format detection, TMA descriptor construction, a 3-warpgroup pipelined kernel, and a combine kernel. The caller supplies tensors and gets `(out, lse)`.

**But the interface leaks, and the repo says so in its own docstrings:**

1. **Vestigial parameters.** `block_table` and `cache_seqlens` are positional and "currently ignored. We leave it here to be compatible with the old interface" (`flash_mla_interface.py:87-88`); `num_splits` "must be None" (`:96-97`); `causal` "Must be False" and `is_fp8_kvcache` "Must be True" (`:99-100`). Five parameters that exist only for history — classic interface cruft, and exactly the thing Ousterhout says accumulates when you preserve compatibility instead of versioning.
2. **The metadata object is stateful and silently stale-able.** "You may reuse the same `tile_scheduler_metadata` across different invocations, but only when the tensor shapes and the values of `topk_length` and `extra_topk_length` remain the same. Note that the values are NOT checked at runtime: reusing it with different `topk_length` values silently reuses stale split-KV scheduling metadata" (`:90-95`). The shapes *are* checked (`:160-174` asserts b, s_q, h_q, page_block_size, h_k, causal, topk, …, with a helpful message), but the values cannot be without a device sync — so correctness rests on a documented convention. An honest, well-documented hole; a *deeper* module would hash the lengths on device, or make the metadata immutable and keyed.
3. **Information leakage into the caller.** The caller must know the FP8/FP4 byte layout to build the cache, must pre-apply the block table, must set invalid indices to exactly `-1`, and must know that `lse`'s shape is transposed relative to `out` (`README.md:157`). Some of this is inherent (the cache is written by another kernel), some is not.
4. **Errors as `TORCH_CHECK`/asserts with good messages** (`sparse_decode.cpp:227-299`) — dozens of shape/dtype/contiguity checks with named tensors. Defensive, cheap, and the right default for a kernel library.
5. **A nice piece of depth:** `detect_kv_cache_format_for_headdim_512(kv.size(3))` (`:285`) — the tensor's own shape identifies the format, so there is no redundant `format=` argument to get wrong. Self-describing data instead of a parameter.
6. **Internal layering is genuinely deep.** `ImplBase<Params, Features>` + `DECLARE_SUPPORTED_FEATURES` + `dispatch_kv_formats` (`csrc/api/common.h:183-277`) turns "which kernel supports attn_sink + topk_length + fp4-extra-cache + batch-invariance?" into a declarative list per implementation, with the dispatcher verifying that every requested feature is supported before running (`:203`, `:273`). The feature enum is `SparseDecodeFeatures` (`sparse_decode.cpp:15-25`). This is a clean answer to combinatorial kernel selection and would suit Luigi's future `Device::{Cpu, Cuda} × {f32, int8, int4}` matrix.

## 11.3 What to take into the engine's own attention API

- **Separate "plan" from "execute"** if and only if the plan is reused. Luigi's engine will have the same choice in M6/post-v1: compute a thread work-plan once per decode step (all layers share it) versus every layer.
- **Do not keep parameters you no longer use.** Delete them and bump a version; his project has no external users, so there is no excuse.
- **Make the plan immutable and validated.** If the plan object records `(b, s_q, lengths_hash)`, a stale reuse is a panic, not silent corruption. Rust makes this easy (own the plan, borrow it in `forward`), and it is a good place to show why the borrow checker is an asset here.
- **Return the LSE.** FlashMLA always returns `(out, lse)` even for the non-split path. That is what makes split/merge composable — and, later in the server, what makes chunked prefill and prefix-cache reuse composable. Cheap now, enabling later.

**Applies to:** CPU engine (API design, M3/M6), post-v1 server.

---

# 12. Engineering practices inventory (things to copy verbatim)

1. **One header comment per kernel file**: purpose, template params, grid/block, "I/O: see struct" (`kernel.cuh:1-18`, `combine.cu:1-24`).
2. **All params in one POD struct per op**, passed `__grid_constant__` (`csrc/params.h`).
3. **Compile-time format types** instead of runtime branching (`kv_cache_format.h`; `KVCacheFormat<MT>` with `static_assert`s for every layout invariant, e.g. `:26`, `config.h:58-60`).
4. **`static_assert` the layout math**, not just the types: `static_assert(sizeof(DecodingSchedMeta) == 32)` (`params.h:94`), `4 * RAW_TOKEN_SMEM_STRIDE % 128 == 0` (`config.h:60`), `smem_size <= 227*1024` (`kernel.cuh:728`).
5. **Device-side assertions** in debug-capable form: `KU_TRAP_ONLY_DEVICE_ASSERT` (`get_decoding_sched_meta.cu:115`, `combine.cu:71`).
6. **Build fails on register spills** (`setup.py:113-145`) with an env-var escape hatch, documented in the README (`:105`).
7. **Comments record rejected alternatives and compiler-forced rewrites** (§7.7). Also `// TODO Tune` on a magic constant (`sparse_decode.cpp:103`) — honest about what is unmeasured.
8. **Benchmarks flush L2 and cool down** (`bench.py:119-136`, `test-sparse-decode.py:292-293`); timings come from CUPTI per-kernel ranges, with e2e defined as `max(end) − min(start)` so inter-kernel gaps are visible (`bench.py:83-110`).
9. **The results table shows the predicted intensity next to the achieved rates** (`test-sparse-decode.py:312-335`), plus a geomean.
10. **Reference implementation published in the README as runnable PyTorch** (`README.md:185-205`), including the documented divergence for the degenerate case (`:206`).
11. **Tests generate adversarial shapes programmatically**, with comments naming the invariant each family probes (`test-sparse-decode.py:75`, `:114-115`).
12. **Failures print a paste-able repro** (`:346-349`).
13. **A determinism switch in the kernel itself** (`enable_batch_invariant`) so bitwise reproducibility is a supported mode, not an accident.
14. **Platform detection at build time** with an override env var (`README.md:99-101`), and a clear error when an op is missing on a backend (`flash_mla_interface.py:225-234`).

---

# 13. Milestone map

| Milestone | FlashMLA idea to use | Where it lives in FlashMLA | Applies to |
|---|---|---|---|
| **M0** perf model | Arithmetic intensity vs machine balance; measure the *throttled* peak, not the nameplate; keep the `C/M` column in the results table forever | `docs/20250422:11-15`; `tests/lib.py:490-495`; `test-sparse-decode.py:234-245` | CPU + GPU |
| **M0** | Cycle-level budgeting of a competing unit (MMA vs dequant) | `docs/20250929:17-27`; Ascend's Python cycle model `docs/20260930:85-116` | both |
| **M1** softmax | `exp2` + precomputed `scale·log2(e)`; `-1e30` instead of `-inf`; row max/sum in f32 | `kernel.cuh:190-253`, `config.h:64`, `common.h:17` | CPU (then GPU) |
| **M1** kernels | Padding a power-of-two stride to kill conflicts (520 vs 512) | `config.h:107-110` | CPU + GPU |
| **M2** loading | Self-describing layout: detect the format from the tensor's shape; layout invariants as `static_assert` | `sparse_decode.cpp:285`, `kv_cache_format.h` | CPU |
| **M3** forward/KV cache | KV accessor as a *function* so paging is a one-line change later; return `(out, lse)` from attention | `kv_cache_utils.cuh:53-61`; `README.md:157` | CPU |
| **M3** attention | Online softmax with running max/sum + rescale; mask invalid to `-inf` **before** any reduction | `docs/20260930:39-75`; `common_subroutine.h:103-119` | CPU |
| **M3** correctness | Reference in PyTorch, f32, gather-then-dense; publish it; add the cosine-diff gate and inf/NaN structural check to D1's checker | `tests/ref.py`, `compare.py:44-105` | CPU |
| **M4** prefill/decode split | Prefill and decode as *modes of one kernel* + separate schedulers; `s_q > 1` is the spec-decode/MTP case | `params.h:44-51`; `sparse_decode.cpp:117` | CPU |
| **M5** quantization | Group size vs scale dtype (32/ue8m0 vs 16/e4m3); interleave scales with data per token; dequant in-kernel, math in bf16/f32; **test the kernel against a dequantized reference, measure drift separately** | `kv_cache_format.h:5-20`, `quant.py:58-194`, `lib.py:256-266` | CPU + GPU |
| **M6** threading | Split-KV across threads + LSE merge (and the reason: 6 heads ≠ 24 threads); scheduler with a `fixed_overhead` cost term; skip-scale | `get_decoding_sched_meta.cu:42-122`; `combine.cu:110-174`; vLLM `csrc/cpu/mla_decode.cpp:279-355`; `kernel.cuh:224` | CPU |
| **M6** batched prefill | Why matrix×matrix changes everything (tensor cores / AVX2 blocking); a separate code path for small-N is legitimate | `fwd_for_small_topk` (`sparse_prefill.cpp:207`) | CPU |
| **M6** measurement | L2/cache flush before timing; cooldown; per-kernel timing; geomean over a sweep | `bench.py:112-150` | CPU + GPU |
| **M7** CUDA decode | Warp specialization (producer/consumer roles), multi-stage smem pipeline, `cp.async` + L2 hints, swizzled/padded smem, spill-free build gate | `kernel.cuh:165-655`; `kerutils/.../sm80/intrinsics.cuh:10-95`; `setup.py:113-173` | sm_89 |
| **M7** CUDA decode | Split-KV + combine as **two kernels**, partials in f32; PDL is Hopper+, so eat the launch gap or fuse | `combine.cu`; `kernel.cuh:731` | sm_89 |
| **M7** validation | CPU engine as oracle, three-part tolerance, determinism check | `compare.py`, `test_fmha_sm100.py:141-149` | sm_89 |
| **M7** don't | TMA, TMEM/`tcgen05`, WGMMA, clusters/DSM/`st.async`, `setmaxnreg`, PDL, `cvt…scaled::ue8m0` | §7.4 table | not on sm_89 |
| **M8** real model | GQA means `g = h_q/h_kv` heads per KV byte — recompute the intensity; MLA/MQA is why decode can be compute-bound at all | §3.2 | both |
| **post-v1** paged KV | Block table applied by the caller vs in the kernel; page size as a runtime value; "contiguously valid" cache | `README.md:150`; `quant.py:197-229`; `flash_mla_interface.py:86` | server |
| **post-v1** load balancing | Greedy equal-payload partitioning with a per-request fixed overhead; metadata computed once per step | `get_decoding_sched_meta.cu:65-102` | server (role project 3) |
| **post-v1** determinism/faults | `enable_batch_invariant` as an explicit mode; split plans are recomputable from lengths alone (so a lost worker's plan is cheap to rebuild) | `sparse_decode.cpp:223` | server (roles 5, 6) |

---

# 14. Concrete learning path

**Step 1 — M3, naive CPU decode attention.** Per head: dot products over `0..pos`, two-pass softmax, weighted sum of V. Mirror llama2.c (`run.c:290-318`) but with Luigi's tensor type and a **KV accessor function**. Return `(out, lse)` even though nothing uses `lse` yet — 10 minutes now, and it is the hook for everything below. *FlashMLA ideas used:* the `(out, lse)` contract; mask-before-reduce discipline; the PyTorch reference published alongside.

**Step 2 — M3/M6 boundary, tiled + online softmax on CPU.** Rewrite the same attention as: for each 64-token tile, compute scores, update `(m, l, o)` online. Verify bitwise-ish agreement with step 1 on the §6.3 example, then on real prompts against D1's tolerance. *Ideas:* online softmax; `exp2` + precomputed `scale·log2e`; `-1e30` sentinel; skip-scale with threshold 6 (measure whether it helps at `head_dim=48`; predict "barely", then confirm — a good calibration exercise).

**Step 3 — M6, threaded split-KV + LSE merge on CPU.** Split the `0..pos` range across 24 threads (`rayon` or `std::thread` scope), per-thread `(o, lse)`, tree-merge with the natural-log formula in §5.6. Compare against vLLM's `csrc/cpu/mla_decode.cpp:279-355`. Measure: 6-head model on 24 threads, before/after, with the `C/M` column. Then add the scheduler idea: a work plan with a `fixed_overhead` term, and show a batch of unequal-length requests being balanced. *Ideas:* split-KV, the combine math, the metadata/plan separation, the determinism tension.

**Step 4 — M7a, naive sm_89 decode kernel.** One block per (request, head); `q` in registers, loop over KV in global memory, CUDA cores, warp-shuffle reductions, two-pass softmax over a `__shared__` score buffer (or online if the context exceeds the buffer). Validate against the CPU engine with the three-part check. Expect it to be bandwidth-bound; measure achieved GB/s versus a measured `memcpy` roof. *Ideas:* CPU-as-oracle; the intensity table; spill-free build gate from day one.

**Step 5 — M7b, shared-memory tiles + a cp.async pipeline.** Stage KV tiles into shared memory with `cp.async` (16 B/thread, `L2::128B` prefetch, predicated) in a 2–3 stage circular buffer with `commit_group`/`wait_group`; keep the score/accumulator math in registers; pad or swizzle the tile stride. *Ideas:* the multi-stage pipeline and the `NUM_BUFS`/barrier-pair structure (`config.h:166-171`), L2 cache hints, smem stride padding (`config.h:109`).

**Step 6 — M7c, online softmax + warp specialization.** Move to online softmax so the tile size is independent of context length; optionally split warps into loader/compute roles with named barriers. *Ideas:* the warpgroup table in §7.1 (roles, not mechanisms); skip-scale; `__any_sync` to keep the branch warp-uniform.

**Step 7 — M7d, split-KV + a combine kernel on sm_89.** Port step 3's plan to the GPU: a plan buffer (compute it on the host — his batch sizes are tiny, no need for a scheduler kernel), `o_accum [splits, heads, head_dim] f32` + `lse_accum [splits, heads] f32`, then a combine kernel with one warp per head, `exp2`/`log2`, early-exit when `num_splits == 1`. Mirror `combine.cu` closely — it is 240 lines and almost all of it is transferable. No PDL on sm_89: measure the launch-gap cost and report it. *Ideas:* everything in §5.

**Step 8 — M7e, quantized KV in the GPU kernel.** int8 or fp8 KV with per-32 group scales interleaved per token; dequantize into shared memory (a dedicated warp if warps are specialized), MMA/FMA in f32; verify against the dequantized reference, then measure quality drift separately. On sm_89 use native `cvt.rn.f16x2.e4m3x2` for fp8 (no ue8m0 scaled-cvt). *Ideas:* §8 in full.

**Step 9 — post-v1, paged KV.** Swap the KV accessor for a block-table lookup; decide (and write down) whether the kernel dereferences the table or the caller pre-flattens indices. Reuse the M6 scheduler for prefill/decode mixing. *Ideas:* §4, plus the `fixed_overhead` cost model for the request router (role project 3).

---

# 15. Quiz questions (with answers, for the tutor)

**Roofline / M0**
1. MLA decode with `h_q=128, s_q=1` has intensity ≈ 256 FLOP/byte; stories15M's attention has ≈ 0.5. Name the two independent factors and give each one's size. *(MQA sharing: 128 query heads per KV head; K and V being one tensor: ≈ 1.9×. And f32-vs-bf16 accounts for the last 2×.)*
2. On H800 DeepSeek used 865 TFLOPS, not 990. Why, and what would using 990 have done to the compute-bound threshold? *(Clock throttling to ~1600 MHz; the threshold `h_q·s_q ≥ ½·peak/BW` would have moved from 128 to ~148, i.e. it would have wrongly predicted memory-bound at h_q=128.)*
3. For stories15M at `L=256`, what fraction of the per-token bytes is KV cache traffic? At what `L` would KV equal weight traffic? *(≈6 %; L ≈ 4400 — unreachable since seq_len=256.)*
4. Why does the benchmark count *unique* indices for KV bytes but *all* attended tokens for FLOPs? *(Duplicate selections are served from cache, so they cost FLOPs but not DRAM traffic. See `lib.py:474-490`.)*

**Online softmax / M3**
5. Why `MAX_INIT_VAL = -1e30` instead of `-inf`? *(`exp2f(mi - new_max)` with both `-inf` gives NaN; `config.h:64`.)*
6. The kernel keeps both `mi` and `real_mi`. What is each for? *(`mi` scales the softmax and may lag; `real_mi` detects "no valid token at all" so the row can be forced to `lse=+inf`, `out=0`; `kernel.cuh:192, 281-286`.)*
7. With `RESCALE_THRES = 6` in log2 units, what is the largest value `exp2(p − mi)` can take, and why is that safe? *(2^6 = 64; far inside bf16/f32 range, and the running sum is f32.)*
8. Why must the invalid-token mask be applied before the two score halves are summed? *(`-inf + finite = -inf`, but if you mask after summing you may already have added garbage — and `-inf + inf = NaN`. `common_subroutine.h:103-105`.)*

**Split-KV / M6-M7**
9. Write the merge of two partial states `(o_a, lse_a)`, `(o_b, lse_b)` and prove it equals the single-pass softmax. *(§5.4 derivation; `2^{lse_j} = Σ_{t∈S_j} 2^{p_t}` is the key identity.)*
10. Why does the combine kernel return immediately when `my_num_splits == 1`? *(That request was never split; the attention kernel wrote the final out/lse directly, including the sink. `combine.cu:67`, `kernel.cuh:294-299`.)*
11. `payload = ceil(total_blocks / num_sm_parts) + fixed_overhead`, and each request also charges `fixed_overhead`. What failure mode does the overhead term prevent? *(A partition stuffed with many tiny requests: block count alone under-counts per-request prologue/epilogue cost.)*
12. `o_accum` is sized `b + num_sm_parts`. Why is that enough? *(One slot per request minimum, plus at most one extra split per partition boundary.)*
13. Split-KV is disabled when `topk + extra_topk <= 640`. What is the trade-off in one sentence? *(Below that, the fixed per-split cost plus the combine pass costs more than the extra parallelism buys.)*
14. Luigi's model has 6 heads and he has 24 threads. Why does llama2.c's `#pragma omp parallel for` over heads cap his speedup at 4×, and what does split-KV change? *(Only 6 tasks exist; splitting the time axis creates 24+ tasks. `run.c:283`.)*
15. Why is a 24-thread split-KV result not bitwise identical to a single-thread run, and what does FlashMLA offer for that? *(Different summation order; `enable_batch_invariant` disables splitting to make results independent of partitioning, `sparse_decode.cpp:223`.)*

**Kernels / sm_89**
16. Which of these exist on sm_89: `cp.async`, `ldmatrix`, `mma.sync`, TMA, WGMMA, TMEM, thread-block clusters, `setmaxnreg`, PDL, native fp8→fp16 `cvt`? *(Exist: cp.async, ldmatrix, mma.sync, native fp8 cvt. Absent: TMA, WGMMA, TMEM/tcgen05, clusters/DSM, setmaxnreg, PDL.)*
17. Why did the Hopper kernel need "seesaw" and the sm100 kernel does not? *(Hopper's WGMMA accumulator lives in registers — one 64×512 f32 tile is 32 k registers, half the SM's file; sm100 keeps O in TMEM. `docs/20250422:19`, `config.h:78-85`.)*
18. What made the Hopper FP8 decode kernel dequantization-bound, and what was the fix? *(34 cycles of MMA vs ~50 cycles of dequant per token, because H800 lacks a direct e4m3→bf16 cast; fix = "crossover", two CTAs each dequantizing half a token and exchanging via distributed shared memory. `docs/20250929:17-45`.)*
19. Why is `stride = 520` used for a 512-wide f32 shared-memory tile, and what is the CPU analog? *(Break the power-of-two stride so rows land in different banks; on CPU, avoid cache-set aliasing with a padded row stride.)*
20. In the sparse decode kernel, `sK` and `sV` point at the same shared-memory array. Why is that legal? *(In MLA, K and V are the same latent tensor; only the CuTe layout differs. `kernel.cuh:488, 496`.)*

**API / practices**
21. Name three vestigial parameters of `flash_mla_with_kvcache` and say what the docstring says about each. *(`block_table`, `cache_seqlens` ignored; `num_splits` must be None; also `causal` must be False, `is_fp8_kvcache` must be True. `flash_mla_interface.py:87-100`.)*
22. What silently breaks if you reuse the scheduler metadata after changing `topk_length`, and why can't the library check it? *(Stale split plan ⇒ wrong partitioning of work, and the values live on device, so checking would force a sync. `:90-95`.)*
23. Their `rel_tol` values look like `2.01/128` and `8.01/65536`. Where do those come from, and why did D1 choose `rtol = 0` instead? *(bf16/f32 mantissa ULPs times a small integer; D1 measured that the engine's error does not grow with logit magnitude, and rtol would loosen exactly the largest logits, which decide the token.)*
24. What does the cosine-diff gate catch that a per-element tolerance does not? *(A small systematic bias spread over every element — e.g. a slightly wrong `sm_scale`.)*
25. Why does the decode test run the kernel before allocating the reference tensors? *(Otherwise the reference allocation can reuse the buffer holding the answer, producing a false pass. `test-sparse-decode.py:183-188`.)*

---

# 16. Open questions / things I could not verify

1. **The sm90 kernels are not in this tree.** The 2026-09-30 release removed Hopper support [docs: README.md:3] and the clone is shallow (1 commit), so `ba89a3466e9470ad08ab39738d4e7bb66989e1e7` (the last Hopper-capable commit) cannot be inspected locally. Everything I say about warp specialization on Hopper, the seesaw schedule, DSM crossover, the 656 B KV format, and the dense bf16 MLA decode kernel comes from the two blog posts, not from code. If the tutor wants the Hopper code, someone must `git fetch --unshallow` or clone that commit separately.
2. **No dense (non-sparse) MLA decode kernel exists in this release** — decode requires `indices` and an FP8/FP4 cache (`flash_mla_interface.py:131-133`: "Sparse attention is required"; "only sparse attention with a quantized KV cache is supported"). So the "dense decode path" in the briefing no longer exists here; the closest thing in-tree is the CUTLASS dense **prefill** fwd/bwd.
3. **Page block size 64** is not a requirement of this release (tests use 1–576); the `64` is `B_TOPK`/`block_size_topk`. I could not verify the historical Hopper requirement from this clone.
4. **No accuracy numbers for FP8/FP4 KV** anywhere in the repo (no perplexity, no logit-drift table). The justification given is memory capacity. If Luigi wants a published KV-quantization quality number, it will have to come from the DeepSeek-V3.2 paper or elsewhere.
5. **Hardware peaks for B200 / Ascend 950 are not stated** in the repo, only achieved TFLOPS and, for Ascend, a percentage-of-peak. So the "95 % / 83 % of hardware peak" claims cannot be decomposed without an external spec. Do not let a derived peak be presented as DeepSeek's.
6. **`fixed_overhead_num_blocks = 3` for h_q=128 is marked `// TODO Tune`** (`sparse_decode.cpp:103`) — the constants are empirical, and the repo does not say how the h_q=64 value (5) was chosen.
7. **The DSA indexer (how top-k is selected) is out of scope** for this repo; only the consumer of `indices` is here.
8. **`enable_batch_invariant`'s exact guarantee** (bitwise across batch compositions? across `num_sm_parts`?) is only described as "results do not depend on how the batch is partitioned" (`flash_mla_interface.py:110-112`). I did not find a test asserting it.
9. **Ascend numbers use an L2 bandwidth of 5 TB/s and 4096 FMA/cycle/core** in a Python snippet (`docs/20260930:85-101`) without citing the source; treat as DeepSeek's own assumptions.
10. **CUTLASS is not checked out** (`csrc/3rdparty/cutlass` is an uninitialized submodule here), so the CUTLASS FMHA internals (tile schedulers `fmha_tile_scheduler.hpp` / `fmha_causal_tile_scheduler.hpp` are present, but their CUTLASS dependencies are not) could only be read at the FlashMLA-local level.
11. **I did not verify the kernels run** — this machine has an RTX 4070 (sm_89) and the build requires sm_100a/103a plus CUDA 13.1+. Nothing in this repo can be compiled or benchmarked locally. All performance statements are DeepSeek's, all structural statements are from reading the source.
