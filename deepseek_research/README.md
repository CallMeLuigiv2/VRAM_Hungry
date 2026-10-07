# DeepSeek research: tutor's index

Research into DeepSeek's GitHub (github.com/deepseek-ai), done 2026-09-29/30 at Luigi's request: "take notes on repos we could use for examples, structure and high-performance techniques for our engine and kernels ... notes for yourself so you have enough info to guide me."

**Audience: Claude, as tutor.** These are private reference notes, denser than anything Luigi needs at one milestone. Luigi can read them, but they aren't his docs (his are README.md, DECISIONS.md and ARCHITECTURE.md).

**How to use them:**
- Follow just-in-time learning. At a milestone, open the map below, read the linked sections *before* explaining, and give Luigi one idea at a time.
- Nothing here is a decision. Every "option" in these notes is for Luigi to choose (5-step loop, step 3).
- File:line references are pinned to the commits below. If a repo is re-cloned at a newer commit, re-verify before quoting.

## The six notes

| File | Repos (commit) | Lines | Core value |
|---|---|---|---|
| [01_reference_inference.md](01_reference_inference.md) | DeepSeek-V3 `9b4e978`, DeepSeek-V3.2-Exp `87e509a` | 813 | A llama2.c-sized reference with FP8: one `forward(tokens, start_pos)` for prefill and decode, MLA naive vs absorb (prefill vs decode as two algorithms), block-scaled FP8 GEMM = runq.c group matmul, the sort-free exponential-race sampler, tensor-parallel → threads |
| [02_deepgemm_deepjit.md](02_deepgemm_deepjit.md) | DeepGEMM `057ca59`, DeepJIT `3732a3b` | 1554 | Two-level accumulation (quant group = accumulation chunk), a 40-line config heuristic as a performance model, wave quantization = load imbalance, the `calc_diff` metric, benchmark method, JIT/caching options for the M7 host, and the **M7 sm_89 GEMM ladder** |
| [03_flashmla.md](03_flashmla.md) | FlashMLA `2e5429f` | 897 | Decode arithmetic intensity (MLA ≈256 FLOP/B vs stories15M 0.5), split-KV + LSE combine (M6 CPU threads, M7 GPU), online softmax, skip-scale, paged KV, FP8/FP4 KV cache, a 3-part correctness check, and the **M3→M7 attention learning path** |
| [04_deepselect_tilekernels.md](04_deepselect_tilekernels.md) | DeepSelect `bfa4507`, TileKernels `66258df` | 517 | **Sampling (role project 1)**: threshold-filter top-k with radix select, the exact llama2.c top-p algorithm and its traps, CPU AVX2 and sm_89 sampling kernels K1–K7; quantization formats (FP8 E4M3, FP4 E2M1, per-token groups = Q8_0), rounding and zero-amax traps, property/statistical tests |
| [05_serving_system.md](05_serving_system.md) | open-infra-index `56d8685`, profile-data `4496024`, EPLB `d52c72d`, LPLB `0490f79`, DeepEP `93eb6eb`, DeepSpec `005e03b`, DualPipe `030ce43`, 3FS `22fca04` | 1138+ | DeepSeek's production serving: prefill/decode disaggregation (EP32 vs EP144), published numbers that reproduce (role 4), trace analysis, EPLB line by line (role 3), fault tolerance derived from 3FS (role 5), speculative decoding, interview mapping, and a one-box post-v1 server roadmap |
| [06_rust_codebases.md](06_rust_codebases.md) | deepseek-recipe `8cadfed`, 3FS Rust `22fca04` | 725 | Real DeepSeek Rust: workspace layout facts for the **M0 crate-layout decision**, error handling, API style, streaming state machine and streaming detokenization (M4), FFI (cxx, bindgen, PyO3), 3FS `AlignedBuffer` as an unsafe counter-example, and 24 idioms |

Clones: `~/refs/inference/deepseek/<repo>` (shallow, `--depth 1`, no submodules).

## Hardware reality check (read before promising Luigi anything)

- **None of DeepSeek's fast kernels run on the RTX 4070 (sm_89).**
  - DeepGEMM needs SM90/SM100 + CUDA ≥ 12.9 (DeepGEMM `README.md:31-36`).
  - FlashMLA's current release is **Blackwell-only**; Hopper was removed on 2026-09-30 (`FlashMLA/README.md:3`, `sparse_decode.cpp:190`).
  - DeepSelect is built for sm_100a/103a and needs about 203 KiB of shared memory (the 4070 allows 99 KiB per block).
  - TileKernels needs SM90/SM100.

  They are idea sources, never dependencies or baselines.
- **Techniques that don't exist on sm_89:** TMA, WGMMA, clusters/DSM, `setmaxnreg`, PDL (sm_90+); tcgen05/TMEM/UTCCP (sm_100+). **The sm_89 path:** `mma.sync` + `ldmatrix` + `cp.async` multi-stage pipelines + L2 hints. The equivalence tables are in 02 §10 and 03 §7.
- **Toolchain:** nvcc **12.0** rejects FP8 `mma.sync` for sm_89 at the assembler stage, but **INT8 `mma.sync` compiles** (01 §2.5, tested by the agent). So the M7 low-precision kernel can be int8 today; FP8 needs a newer CUDA toolkit (version unverified, probably 12.4+). This goes into the M7 decision (see memory `m7-kernel-language`).
- **CPU:** AMD Ryzen 9 5900X (Zen 3), 12 cores / 24 threads, AVX2 + FMA, **no AVX-512, no VNNI, no FP8**. Q8_0 int8 with AVX2 is the natural M5 path. On synthetic data, int8 in groups of 32 had lower error than FP8 (01 §2.5).
- **llama2.c weights start at byte 28** (`run.c:160`, a 7×int32 header). An mmapped checkpoint is therefore never 32-byte aligned, so aligned AVX2 loads would fault. Use unaligned loads or copy (candle checks and copies: `safetensors.rs:115-135`) (06 §8).

## Cross-cutting lessons (the ideas that recur across repos)

1. **Prefill and decode are different regimes, so they get different algorithms and parallelism.** MLA naive (prefill) vs absorb (decode) on the same weights (01 §2.3.2; V3.2 branches at `model.py:574,590`). EP32 prefill vs EP144 decode units (05; D6 doc `:22-23`). Decode is weight-streaming-bound at about 0.5 FLOP/B for stories15M (03 §3.3; 05 quiz 1). This is the backbone of Luigi's end goal and the M0 performance model.
2. **Quantization group = accumulation chunk; apply each scale once per group.** `runq.c:336` ≡ `kernel.py:162` ≡ DeepGEMM's `final_accum += (sa*sb)*accum` (`sm90_fp8_gemm_1d2d.cuh:251-345`). The M5 core lesson (01, 02 §3, 04 §6).
3. **A performance model only needs the terms that vary.** DeepGEMM's heuristic comment: "HBM bandwidth and total compute ... are constant across configs" (`heuristics/sm90.hpp:227`). A template for role project 4 and for M0/M6 predicted-vs-measured tables (02 §5).
4. **Load imbalance is its own term.** Wave quantization (02 §4.7, with a 24-thread CPU worked example). FlashMLA's split-KV scheduler has a per-request fixed-overhead cost (03 §5.2). EPLB/LPLB greedy and LP balancing (05 §5). llama2.c's head-parallel attention leaves 18 of 24 threads idle on 6 heads (`run.c:283`); split-KV fixes it (03 §5.6).
5. **Online softmax + log-sum-exp merge** enables tiling and splitting, from CPU threads to GPU blocks. Use 03 §5.5 and §6.3 verbatim as worked examples. Use `-1e30` rather than `-inf`; skip-scale lets the running max lag by up to 6 before rescaling (`kernel.cuh:224`).
6. **Sampling without a full sort:**
   - The exponential-race (Gumbel-max) sampler (`generate.py:27`).
   - Threshold-filter + radix select top-k (DeepSelect, 04 §1).
   - llama2.c's cutoff-then-qsort top-p (`run.c:624-665`).
   - Measure sampling's share of per-token time before optimizing (04 §4, open questions 1–2).
7. **Testing methodology, richer than one tolerance.** None of them replace D1; they add to it:
   - D1's measured max-abs check (keep it).
   - DeepGEMM `calc_diff = ‖x−y‖²/(‖x‖²+‖y‖²)` (`testing/numeric.py:5-11`).
   - FlashMLA's 3-part check: inf/NaN positions, per-element abs-or-rel, and a cosine gate (`tests/kernelkit/compare.py:44-105`).
   - Double references: vs the exact f32 result *and* vs dequantized inputs, which separates kernel bugs from quantization error.
   - Bitwise determinism reruns.
   - Two paths must agree: batched prefill == token-by-token decode (DeepGEMM `tests/utils.py:20-38`).
   - Property tests for top-k (ties make index equality wrong).
   - A statistical rounding-bias test (TileKernels `check_bias`).
8. **Benchmark methodology:** flush L2/cache before each timed run, warm up, time per kernel, use effective GB/s for memory-bound kernels (no FLOPs to count), and a geometric mean across sweeps (02 §8; 03 §2.2; 04 §3).
9. **The reference model repos have no tests,** and an indexer RoPE bug shipped and stayed for about 7 weeks (`DeepSeek-V3.2-Exp/README.md:77`). Use this as the argument for D1 and as the M8 RoPE-layout warning.
10. **Tricks expire.** The FFMA SASS interleaving hack was retired when NVCC 12.9 started doing it automatically (DeepGEMM `README.md:22`). Measure rather than cargo-cult.

## Milestone map (merged; see each file's own table for detail)

| Milestone | Read | What to bring (one idea at a time) |
|---|---|---|
| **M0 · D2 perf model** | 01 §2.3.2, 03 §3, 05 §3 and §6.2, 02 §5 | Roofline: decode ≈ weights ÷ bandwidth, about 0.5 FLOP/B. DeepSeek redesigned attention because of this arithmetic. "Model only what varies." Their published node counts reproduce from throughput (05 quiz 10) |
| **M0 · crate layout** | 06 §3 (pros/cons), 01 §2.1 | Facts: a bin-only package can't have `tests/` importing its code (the golden-logits test needs a lib); profiles are root-only; candle keeps nvcc out with `exclude` + a `cuda` feature + a dummy backend. Present, don't decide |
| **M1** | 03 §6, 04 §9, 02 §3.2, 06 §4 and §8 | Softmax via `exp2` with f32 max/sum; wider accumulators; an error-handling choice for shape mismatches (assert vs Result); aligned-buffer design (3FS `AlignedBuffer` as the counter-example) |
| **M2** | 01 §2.8, 06 §8, 02 §7.2 | Byte-28 misalignment; named vs positional weights (the swapped w1/w3 quiz); mmap context |
| **M3** | 01 §2.1–2.3, 03 §5.5, §6.3, §11.3 | Where the KV cache lives (DeepSeek in-model / llama2.c RunState / candle `&mut Cache`): Luigi decides. **The API must return logits at every position** (D1 needs them). Mask `(s, start+s)`. A KV accessor function. Return `(out, lse)`. RoPE position-offset bugs. Speculative decoding will later need a KV "crop" |
| **M4** | 01 §2.6, 04 §2–5, 06 §6–7, 05 §3.5 | Prefill → decode loop, TTFT (includes tokenization) and decode tok/s. The exact llama2.c top-p and its traps (unstable qsort, `/=` temperature). The exponential-race sampler. Streaming detokenization (a byte buffer that emits complete UTF-8; `safe_printf` drops split characters at `run.c:436-440`). A tokenizer behind a trait? Luigi's call |
| **M5** | 02 §3, 04 §6, 01 §2.5, 03 §8 | Group = accumulation chunk; scale dtype; group size; **round-half: C `round()` −24.5 → −25 vs torch → −24** (`runq.c:167` vs `export.py:62`); zero-amax guard (`runq.c` divides by zero); a dequantized reference to separate kernel error from quantization drift |
| **M6** | 02 §4.7, 03 §5.6, 01 §2.7, 04 §6 | Wave quantization over 24 threads; split-KV over threads + LSE merge (compare vLLM `csrc/cpu/mla_decode.cpp:279-355`); row vs column split (the vocab head is about 61% of multiply-adds); AVX2 hit-mask top-k; batched prefill == decode test |
| **M7** | 02 §10, 03 §14, 04 §5, 02 §7.5 | GEMM ladder, attention ladder and sampling kernels for sm_89; int8 `mma.sync` today; host compile options (AOT build.rs / NVRTC / cache); CPU-as-oracle checks; an RNG scheme so CPU and GPU sample the same tokens (a D-decision) |
| **M8** | 01 §2.3.3, 03 §3.2, 04 §4 | GQA intensity; RoPE interleaved vs half-split (DeepSeek's shipped bug); vocab 32K vs about 150K makes sampling about 5× more expensive; one layer owns BOS |
| **post-v1** | 05 §5, §8, §9, §11; 03 §4; 06 §5.3 | Paged KV; EPLB-style router (role 3); requeue = re-prefill prompt + streamed tokens (role 5); prefix cache; spec decoding; `InferenceChunk`-style engine↔server contract |
| **Interviews** | 05 §10 | Batching API, 500 GB broadcast (**use the 10 Gbps table**), distributed mode = hierarchical histogram reduce |

## Review log (what the lead Claude verified or changed)

Agents verified their own file:line refs. On top of that, I re-checked these against the sources:
- 01: `V3/model.py:772-790` (the mask only when seqlen>1), `kernel.py:162` ≡ `runq.c:336`, `generate.py:14-27`, `V3.2-Exp/README.md:77`, CPU = Ryzen 9 5900X.
- 02: DeepGEMM `README.md:22,25,31-36`, `testing/numeric.py:5-11`, `heuristics/sm90.hpp:227`.
- 03: FlashMLA `README.md:3` (Hopper removed), `sparse_decode.cpp:190`, `kernel.cuh:224`, llama2.c `run.c:283`.
- 04: `runq.c:167` (C `round`) vs `export.py:62` (`torch.round`), `run.c:642` (qsort).
- 05: Day 6 doc `:22-23,67,74-76` (EP32/EP144, 226.75 nodes, 608B/168B, 73.7k/14.8k). The node count reproduces (≈226.9).
- 06: `run.c:160` (byte-28 weights), candle `Cargo.toml:13-17` (exclude kernels), `run.c:436-440` (`safe_printf`).

**Corrections made:**
- 05 §10.2's broadcast table assumed 50 GB/s links. I added a table with the interview's actual **10 Gbps** parameters: chain ≈ 450 s against a 400 s lower bound, star ≈ 4.6 days.

**Premises the agents corrected** (things my briefing got wrong):
- FlashMLA's "block size 64" is the top-k tile (`B_TOPK`); the page size is a runtime value.
- FlashMLA has no dense decode path any more.
- DeepSelect contains no sampler code; the sampler flow in 04 was reconstructed from vLLM + DeepSeek-V3.
- DeepGEMM's README has no performance table at this commit; the only number is "up to 1550 TFLOPS on H800".
- The DeepEP checkout is V2.5, which removed V1's zero-SM low-latency mode.

## Open questions worth measuring (collected from all six files)

- **n0 (M4):** how many tokens survive llama2.c's cutoff per step on stories15M at T=1, top-p=0.9, and what share of per-token time sampling takes, before and after M6 (04 open questions 1–2).
- **M5 format:** which group size and scale dtype minimize drift per byte on stories15M, measured D1-style (04 open question 5).
- **M7 toolchain:** which CUDA version enables FP8 `mma.sync` on sm_89 (01 §7, 02 open questions); check the PTX ISA before choosing FP8 over int8.
- **M7 sampler RNG:** can the CPU oracle and the GPU sampler produce identical tokens (a counter-based RNG per (seed, step, row)), or is the check statistical only? (04 open question 6)
- **05's MTP hypothesis:** the decode trace looks like MTP speculative decoding with about 85% acceptance. The acceptance rate must come from the V3 paper; it isn't in these repos.
- **Q8_0/Q4_0 layouts:** exact layouts and the AVX2 int8 dot trick in ggml. Verify when llama.cpp is cloned at M5 (not yet in `~/refs/inference`).

## Repos considered and skipped

- DeepSeek-R1, Coder/Coder-V2, LLM, MoE, VL/VL2, OCR/OCR-2, Janus, Math/Math-V2, Prover and Engram: model or paper releases with no engine or kernel code relevant here.
- awesome-* lists: links only.
- deepseek-harness: a TypeScript agent harness.
- smallpond: DuckDB data processing.
- ESFT: fine-tuning.
- DeepGEMM-Ascend, DeepEP-Ascend, clangd-ascend: Huawei NPU ports.

Revisit DeepSpec in more depth if speculative decoding becomes a post-v1 project, and llama.cpp (not DeepSeek) at M5.
