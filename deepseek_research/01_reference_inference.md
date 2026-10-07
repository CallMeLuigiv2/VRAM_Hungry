# 01: DeepSeek reference inference code (V3 and V3.2-Exp), tutor notes

Private reference for the tutor. Do not paste it to Luigi wholesale. Teach one idea at a time, at the milestone given in §5.

## Header

| Repo | Local path | Commit (date) |
|---|---|---|
| DeepSeek-V3 | `~/refs/inference/deepseek/DeepSeek-V3` | `9b4e978` (2025-08-27), shallow clone |
| DeepSeek-V3.2-Exp | `~/refs/inference/deepseek/DeepSeek-V3.2-Exp` | `87e509a` (2025-11-18), shallow clone |
| llama2.c (comparison) | `~/refs/inference/llama2.c` | `350e04f` |
| candle (comparison) | `~/refs/inference/candle` | `66a8cf1` |
| FlashMLA (pointer only) | `~/refs/inference/deepseek/FlashMLA` | `2e5429f` (2026-09-30) |
| DeepGEMM (pointer only) | `~/refs/inference/deepseek/DeepGEMM` | `057ca59` (2026-09-30) |

Studied 2026-09-30. Path shorthands used below: `V3/` = `DeepSeek-V3/inference/`, `V32/` = `DeepSeek-V3.2-Exp/inference/`, `l2c/` = `llama2.c/`. Every file:line was checked with `sed -n`/`grep -n` at the commits above.

**Evidence tags.** Each claim carries one:
- **[code]**: read in the source at the cited line.
- **[doc]**: stated in a README or the V3.2 PDF (quoted).
- **[measured]**: I ran it on this machine (commands in §8).
- **[inference]**: my reasoning, not stated by the sources.
- **[paper, not local]**: from DeepSeek papers that are not in `~/refs`. Verify before presenting it as fact.

**Summary.** DeepSeek's "inference demo" is about 1,100 lines of PyTorch (V3: `model.py` 808 lines, `kernel.py` 196, `generate.py` 185, `convert.py` 96, `fp8_cast_bf16.py` 112). It is the reference *semantics* for a 671B-parameter MoE model, not a fast server. The README itself calls it "example only" (`DeepSeek-V3/README.md:252`) and points to SGLang, vLLM and others for performance.

For this project it is valuable because it is a second complete, readable "llama2.c-sized" engine that is closer to modern production:
- **One forward, both phases.** A single `Transformer.forward(tokens, start_pos)` serves prefill (many tokens) and decode (one token), with the KV cache hidden in module buffers.
- **MLA has two equivalent algorithms.** "Naive" is MHA mode and "absorb" is MQA mode. The official V3.2 paper says the MHA mode is used for prefill and the MQA mode for decode. That is exactly Luigi's end goal (prefill and decode as separate phases) shown in a real model.
- **FP8 block quantization.** 128×128 weight blocks and 1×128 per-token activation groups, with an fp32 "promotion" accumulate. This is the same pattern as llama2.c `runq.c`'s int8 group matmul, so it maps directly onto the M5 Q8_0 plan.
- **One-line sampler.** An exponential-race (Gumbel-max-equivalent) sampler with no sort (role project 1).
- **Tensor parallelism in about 60 lines.** Column/row-parallel linears plus `all_reduce`, which map directly onto M6 thread-splitting.
- **V3.2-Exp adds DeepSeek Sparse Attention.** A lightning indexer (FP8, ReLU, 64 heads) picks the top 2048 past tokens per query. The reference code shows the semantics (a dense compute plus a −inf mask), not the speedup. The speed lives in FlashMLA and DeepGEMM kernels, which need SM90/SM100 and do not run on the RTX 4070.

**Two findings that matter for this machine:**
1. **[measured]** The installed `nvcc 12.0` (PTX ISA 8.0) *rejects* `mma.sync ... .e4m3.e4m3` for `sm_89` (ptxas: "Unexpected instruction types specified for 'mma'"). The INT8 `mma.sync.m16n8k32 .s8.s8.s32` compiles fine. So FP8 tensor-core kernels in M7 need a newer CUDA toolkit, while an INT8 (Q8_0-style) W8A8 tensor-core kernel works today.
2. **[measured]** Casting an out-of-range value to `float8_e4m3fn` in PyTorch gives **NaN**, not saturation (500.0 → nan). This is why the scale is `amax/448` and why V3.2's TileLang kernel clamps.

---

## 1. Repo map

```
DeepSeek-V3/
├── README.md                  model card; §6 "How to Run Locally" (:227–342): demo = "example only" (:252), convert + torchrun commands (:288, :296)
├── README_WEIGHTS.md          HF checkpoint layout: 61 layers + 1 MTP layer (id 61), FP8 e4m3 with 128x128 block scale "weight_scale_inv" (:60–92)
└── inference/
    ├── model.py        (808)  whole model: ModelArgs, TP linears, RMSNorm, YaRN RoPE, MLA, MLP, Gate, Expert, MoE, Block, Transformer
    ├── kernel.py       (196)  Triton: act_quant (1x128 dynamic FP8), weight_dequant (128x128), fp8_gemm (block-scaled GEMM, autotuned)
    ├── generate.py     (185)  sample() (exponential race), generate() (batched prefill→decode loop), main() (torchrun, chat loop)
    ├── convert.py       (96)  HF safetensors → renamed + TP-sharded "model{rank}-mp{N}.safetensors"
    ├── fp8_cast_bf16.py(112)  offline FP8→BF16 dequantization of a whole HF checkpoint
    ├── requirements.txt       torch==2.4.1, triton==3.0.0, transformers==4.46.3, safetensors==0.4.5
    └── configs/               config_16B.json (V2-Lite shape), config_236B.json (V2), config_671B.json (V3), config_v3.1.json (+scale_fmt ue8m0)

DeepSeek-V3.2-Exp/
├── README.md                  DSA intro; 2025.11.17 RoPE-layout bug notice (:77); kernel pointers (TileLang, DeepGEMM #200, FlashMLA #98) (:79–83)
├── DeepSeek_V3_2.pdf          6-page tech report: indexer formula (p1), MQA-mode MLA (p2), training (p3), costs (p4–5), MHA vs MQA mode appendix (p6)
└── inference/
    ├── model.py        (923)  V3 model + Indexer, LayerNorm, Hadamard rotate, per-phase MLA (MHA prefill / MQA decode), fused residual RMSNorm
    ├── kernel.py       (274)  TileLang: act_quant, fp8_gemm, fp8_index (indexer scores)
    ├── generate.py     (186)  ≈ V3 (softmax in fp32, different seed/defaults)
    ├── convert.py      (100)  V3 + 4 indexer tensor names
    ├── config_671B_v3.2.json  V3.1 config + index_n_heads 64, index_head_dim 128, index_topk 2048
    └── requirements.txt       torch, transformers, safetensors, fast_hadamard_transform, tilelang==0.1.6
```

There are no test files in either repo. The only self-check is a `__main__` smoke test (`V3/model.py:801–808`, `V32/model.py:916–923`) that builds a random-weight model and prints the logits shape [code].

---

## 2. Deep dives

### 2.1 Code structure, module map, and comparison with llama2.c

**What.** One file holds the whole model as `nn.Module`s. The math primitives (`linear`, RMSNorm, RoPE) are at the top, then the layers, then `Transformer`. Kernels live in a separate file. Generation, loading and conversion are separate scripts.

**Module map: `V3/model.py`** [code]

| Symbol | Lines | Job |
|---|---|---|
| `world_size, rank, block_size, gemm_impl, attn_impl` | 13–17 | Module globals: TP size and rank, FP8 block = 128, GEMM strategy (default `"bf16"`), attention strategy (default `"absorb"`) |
| `ModelArgs` | 19–86 | Hyperparameter dataclass. Defaults ≈ `config_16B.json` shape. JSON overrides via `ModelArgs(**json.load(f))` (`generate.py:112–113`) |
| `ParallelEmbedding` | 89–128 | Vocab-sharded embedding: shift ids into the local range, zero the out-of-shard rows, `all_reduce` |
| `linear()` | 131–163 | The single GEMM entry point with 3 strategies: bf16 `F.linear`; FP8 weight → `weight_dequant` → bf16 `F.linear`; or `act_quant` + `fp8_gemm` |
| `Linear` | 166–205 | `weight (out,in)` in `Linear.dtype`. If 1-byte, it adds an fp32 `scale (ceil(out/128), ceil(in/128))` and **attaches it as `weight.scale`** (:187) |
| `ColumnParallelLinear` | 208–234 | Shards `out_features` by `world_size`. No communication |
| `RowParallelLinear` | 237–267 | Shards `in_features`. `all_reduce` of partial outputs, then bias |
| `RMSNorm` | 270–294 | `F.rms_norm`, eps 1e-6 |
| `precompute_freqs_cis` | 297–375 | RoPE table `(max_seq_len, rope_dim/2)` complex64, with YaRN frequency correction |
| `apply_rotary_emb` | 378–393 | Rotates adjacent pairs (interleaved) via `view_as_complex` |
| `MLA` | 396–497 | Multi-head Latent Attention. **Owns the KV cache** as non-persistent buffers (:439–444) |
| `MLP` | 500–532 | Dense SwiGLU `w2(silu(w1 x) * w3 x)`. w1, w3 column-parallel; w2 row-parallel |
| `Gate` | 535–598 | MoE router: scores, bias, group-limited top-k |
| `Expert` | 601–633 | SwiGLU expert with plain `Linear` (whole experts live on one rank) |
| `MoE` | 636–693 | Dispatches tokens to local experts, adds the shared expert, `all_reduce` |
| `Block` | 696–735 | Pre-norm residual: `x += attn(norm(x))`, `x += ffn(norm(x))`. Dense MLP for `layer_id < n_dense_layers`, else MoE (:716) |
| `Transformer` | 738–798 | Sets globals (:757–761), embed → blocks → norm → **last position only** → head → `all_gather` |

**`V3/kernel.py`** [code]: `act_quant_kernel` 9–35 / `act_quant` 38–57; `weight_dequant_kernel` 60–86 / `weight_dequant` 89–110; `fp8_gemm_configs` 113–116; `fp8_gemm_kernel` 118–172 / `fp8_gemm` 175–196.

**`V3/generate.py`** [code]: `sample` 14–27; `generate` 30–78; `main` 81–158 (process group 100–104, silence non-rank-0 prints 105–107, `set_default_dtype(bf16)` 109, seed 111, build model on CUDA 115–116, warm-up generate **before** weights load 118, `load_model` 119, chat loop 121–144, batch file mode 145–155); CLI 161–185.

**What V3.2 changes structurally** (`V32/model.py`) [code]:
- `gemm_impl` and `attn_impl` globals are gone. `linear()` always uses `act_quant` + `fp8_gemm` for FP8 weights (:159–163) and asserts `bias is None` (:157).
- The attention algorithm is picked **per phase**: `if mask is not None: # MHA prefill` / `else: # MQA decode` (:574, :590).
- `RMSNorm.forward(x, residual)` fuses the residual add, computes in fp32 (:286–306). `Block` returns `(x, residual)` (:831–851).
- `RowParallelLinear(reduce_output=...)` (:247–269). The shared expert skips its own all-reduce, so each MoE layer does 1 all-reduce instead of 2 (:778, :801–803).
- More fp32 islands: SwiGLU (:643, :744), gate (:687), MoE accumulator (:793), logits head in fp32 (:886, :908).
- New: `LayerNorm` (:309–321), `rotate_activation` (Hadamard, :428–432), `Indexer` (:435–487), a torch-only `weight_dequant` (:490–495).
- `wkv_b` is dequantized once and cached (`self.dequant_wkv_b`, :543, :591–593). V3 re-dequantizes it on **every** forward call (`V3/model.py:481`).

**Comparison with llama2.c and candle**

| Concern | llama2.c `run.c` | llama2.c `model.py` | DeepSeek V3 | candle `llama2_c.rs` |
|---|---|---|---|---|
| Forward signature | `float* forward(Transformer*, int token, int pos)` :231 | `forward(tokens, targets=None)` :249 | `forward(tokens[b,s], start_pos=0)` :773 | `forward(&self, x, index_pos, cache: &mut Cache)` :347 |
| Tokens per call | exactly 1 | whole sequence | 1..s (per call) | 1..s |
| KV cache lives in | `RunState.key_cache/value_cache` :63–64 (calloc :86–87), separate from weights | nowhere: `generate` re-runs the whole sequence each step (:323–325, docstring says "no key/value cache") | module buffers inside each `MLA` :439–444 | `Cache.kvs: Vec<Option<(Tensor,Tensor)>>` :84, passed in `&mut`, grown with `Tensor::cat` :188–192 |
| Cache sizing | `n_layers*seq_len*kv_dim` f32 | none | preallocated `[max_batch_size, max_seq_len, …]` at construction | grows per step |
| Causal mask | implicit: loop `t <= pos` :290 | `triu` buffer :116–117 or `is_causal=True` :148 | `(s,s)` triu only if `s>1` :787–789 | `(s, index_pos+s)`, memoized by `(s, kv_len)` :123–130; builder `utils.rs:17–23` |
| Returned logits | this position | last position only (inference path) :266 | last position only `h[:, -1]` :792 | **all positions** :347–355 |
| Prefill | one token at a time via `forward` in `generate` :747–759 | n/a | one call with all prompt tokens | one call with all prompt tokens |

**Why DeepSeek chose this** [inference]. `nn.Module` buffers keep the demo tiny: the cache moves with `.to(device)`, the TP sharding of heads is automatic (cache shape uses `n_local_heads`), and the caller only passes `start_pos`. The cost is that the model is **stateful**. Only one "conversation batch" can exist per model instance. There is no API to reset, evict, fork or page a sequence. `max_batch_size × max_seq_len` memory is committed up front.

**What it suggests for the Rust engine** [inference]. These are options to put to Luigi as questions (step 2 of the loop is his), not prescriptions:
- **(a) Where the cache lives.**
  - DeepSeek/llama2.c style is `fn forward(&mut self, tokens: &[u32], start_pos: usize) -> …`, with the cache inside the model.
  - candle style is `fn forward(&self, tokens, start_pos, cache: &mut KvCache)`.
  - The candle shape gives immutable weights (shareable across threads with `&`, no `RefCell`), one cache per request (post-v1 server), and fresh caches per test (D1). The borrow checker also likes it: `&self` for weights and `&mut cache` are disjoint borrows. llama2.c already separates `TransformerWeights` (:29–48) from `RunState` (:50–65), which is the C version of this split.
- **(b) What `forward` returns.** DeepSeek returns only last-position logits (saves one vocab-sized matmul per prompt token). But Luigi's D1 correctness check compares logits **at every position** (`DECISIONS.md` D1), and M5 perplexity needs every position too. So the M3/M4 API must be able to produce all-position logits, at least in test mode. That is a real design decision.
- **(c) The `start_pos` contract.** It needs to be written down (DeepSeek never writes it down): "the cache holds positions `[0, start_pos)` of this sequence; this call writes `[start_pos, start_pos+n)` and attends to `[0, start_pos+n)`; requires `start_pos + n ≤ max_seq_len`". This is Ousterhout's "interface comment" and a natural place for a `debug_assert!`.

**Pitfalls.**
- Stateful forward plus a warm-up call: `generate.py:118` runs a 2-token generate on **uninitialized** weights before `load_model` (:119). It leaves garbage in cache positions 0–1. That is harmless only because every later `generate` restarts at `prev_pos=0` and overwrites [inference].
- Both V3 and V3.2 construct `nn.Parameter(torch.empty(...))` under `with torch.device("cuda")` (`generate.py:115–116`). Uninitialized memory until `load_model`, so a missing tensor would be garbage, not zeros [inference: `load_model` defaults to strict, which should catch missing keys; unverified].

---

### 2.2 `Transformer.forward(tokens, start_pos)`: one function, prefill and decode

**Where.** `V3/model.py:772–798`, `V32/model.py:889–913`; attention use of `start_pos` at `V3/model.py:459–495`.

**How, step by step (V3)** [code]:
1. `seqlen = tokens.size(1)` (:784). Prefill: `seqlen = prompt length`. Decode: `seqlen = 1`.
2. `h = self.embed(tokens)` (:785): `(b, s, dim)`.
3. `freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]` (:786). RoPE angles for the **absolute** positions of these tokens.
4. Causal mask (:787–789):
   ```python
   mask = None
   if seqlen > 1:
       mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
   ```
   It is **skipped in decode**. One new query at position `start_pos` may attend to every cached key `0..start_pos` (all are in its past), so nothing needs masking.
5. Every layer gets `(h, start_pos, freqs_cis, mask)` (:790–791).
6. Inside `MLA.forward`: `end_pos = start_pos + seqlen` (:460). The new keys/values are **written** at `cache[:bsz, start_pos:end_pos]` (:477–478 naive; :484–485 absorb). Scores are computed against `cache[:bsz, :end_pos]` (:479, :486–487), i.e. all past tokens plus this chunk. The mask is added as `scores += mask.unsqueeze(1)` (:489). Scores are `(b, s, h, t)`, the mask becomes `(s, 1, s)`.
7. `h = self.norm(h)[:, -1]` (:792). Only the last position goes through the vocab head. Returns `(b, vocab)` logits (after `all_gather` over vocab shards, :794–797).

**The hidden constraint** [measured]. The mask is `(seqlen, seqlen)` but scores have last dim `end_pos = start_pos + seqlen`. So a multi-token call only works when `start_pos == 0`. With `start_pos=4, seqlen=3` the broadcast fails: *"The size of tensor a (7) must match the size of tensor b (3) at non-singleton dimension 3"*. In other words:
- no chunked prefill,
- no prefix reuse,
- no multi-token speculative verification.

`generate.py` never violates this: its first call is `tokens[:, 0:min_prompt_len]` at `start_pos=0`, and all later calls are 1 token. V3.2's MHA-prefill branch has the same constraint for a second reason: it attends only to the **current chunk's** `k, v` (`V32/model.py:576–580`), not the cache. The general mask is candle's `(seq_len, index_pos+seq_len)` with `mask[i][j] = (j > index_pos + i)` (`candle-transformers/src/utils.rs:17–23`).

**Why.** [inference] One code path means one set of weights and kernels, and the demo stays small. The price is that prefill and decode get the *same algorithm* in V3 (`attn_impl` is global). V3.2 fixes that by branching on `mask is not None` (see 2.3).

**Applicability.**
- **CPU (M3/M4).** This shape is exactly what M4 needs for "explicit prefill phase": call forward once with all prompt tokens (`start_pos=0`), then once per generated token. llama2.c instead feeds prompt tokens one at a time (`run.c:747–759`), which is why its "prefill" is just slow decode. The first M4 version can use the matrix-vector kernels in a loop over prompt tokens inside the prefill call. M6 "batched prefill" then replaces that loop with matrix × matrix.
- **GPU (M7).** Same API. Prefill is compute-bound GEMMs, decode is GEMV-like and memory-bound. That is why real engines pick different kernels per phase.
- **Server (post-v1).** Continuous batching needs *per-sequence* `start_pos` (a vector, not one int) and mixed prefill/decode batches. DeepSeek's single `start_pos: int` forces all rows of a batch to the same position (see 2.6).

**Teaching hook (M3).** "Draw the score matrix for `start_pos=0, seqlen=4` (4×4, upper triangle −inf). Now draw it for `start_pos=4, seqlen=1` (1×5, nothing masked). Now for `start_pos=4, seqlen=3` (3×7): which cells are −inf?" Answer: row i masks columns `j > 4+i`, so the top-right 2+1 cells. Then ask why DeepSeek's `(s,s)` mask can't express that.

**Pitfalls.**
- Off-by-one on `end_pos` versus the `t <= pos` loop in llama2.c (inclusive) (`run.c:290`).
- Forgetting the RoPE slice offset (`freqs_cis[start_pos:…]`) gives correct prefill and wrong decode: a classic M3 bug that D1's every-position check catches.

---

### 2.3 MLA attention: naive vs absorb, KV cache sizes, RoPE and YaRN

#### 2.3.1 The shapes (V3 671B config)

From `configs/config_671B.json` [code]: `dim 7168, n_heads 128, q_lora_rank 1536, kv_lora_rank 512, qk_nope_head_dim 128, qk_rope_head_dim 64, v_head_dim 128, n_layers 61`.

Projections (`V3/model.py:424–433`) [code]:
- **Query path.** `wq_a: 7168→1536` (compress), `q_norm`, then `wq_b: 1536→128·(128+64)` column-parallel. Split per head into `q_nope (128)` and `q_pe (64)` (:461–466).
- **Key/value path.** `wkv_a: 7168→512+64` (not parallel, replicated). Split into latent `kv (512)` and a **single shared** `k_pe (64)` (:468–469).
- **Up-projection.** `wkv_b: 512→128·(128+128)` column-parallel. Per head it is `W_UK (128×512)` stacked on `W_UV (128×512)`: `wkv_b.view(n_local_heads, -1, kv_lora_rank)` (:482), first 128 rows = K part, last 128 = V part (:483, :495).
- **Output.** `wo: 128·128→7168` row-parallel.
- **Decoupled RoPE** [inference: the standard MLA reasoning]. RoPE is applied only to the 64-dim `q_pe` / `k_pe` parts (:467, :470). `k_pe` is computed straight from `x` and **shared by all heads** (`k_pe.expand(-1,-1,n_local_heads,-1)`, :476). Rotation is position-dependent, so it can't be folded into `W_UK` (see absorb below). That's why the position signal gets its own small MQA-style channel.

#### 2.3.2 Naive (MHA mode) vs absorb (MQA mode)

**Naive** (`attn_impl == "naive"`, :471–479, :491–492) [code]:
```python
kv = self.wkv_b(self.kv_norm(kv))                     # up-project latent → per-head k_nope, v
k = torch.cat([k_nope, k_pe.expand(...)], dim=-1)     # (b, s, H, 192)
self.k_cache[:bsz, start_pos:end_pos] = k             # cache FULL per-head K (192) ...
self.v_cache[:bsz, start_pos:end_pos] = v             # ... and V (128)
scores = einsum("bshd,bthd->bsht", q, k_cache[:bsz,:end_pos]) * softmax_scale
out    = einsum("bsht,bthd->bshd", probs, v_cache[:bsz,:end_pos])
```

**Absorb** (default, :480–487, :493–495) [code]:
```python
wkv_b = wkv_b.view(n_local_heads, -1, kv_lora_rank)                   # (H, 256, 512)
q_nope = einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope_head_dim]) # q̃ = W_UKᵀ q  → (b,s,H,512)
self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)              # cache the 512-d LATENT
self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)               # and the 64-d shared rope key
scores = (einsum("bshc,btc->bsht", q_nope, kv_cache[:bsz,:end_pos]) +
          einsum("bshr,btr->bsht", q_pe,   pe_cache[:bsz,:end_pos])) * softmax_scale
x = einsum("bsht,btc->bshc", probs, kv_cache[:bsz,:end_pos])           # attend in latent space
x = einsum("bshc,hdc->bshd", x, wkv_b[:, -v_head_dim:])                # then up-project with W_UV
```

**Why they're equal (associativity):**
- Keys: `q_hᵀ (W_UK,h c_t) = (W_UK,hᵀ q_h)ᵀ c_t`.
- Values: `Σ_t p_t (W_UV,h c_t) = W_UV,h (Σ_t p_t c_t)`.

[measured] With random fp64 tensors (H=4, latent 16, T=7), the two paths agree to 3.6e-15 (scores) and 1.8e-15 (outputs).

**What each caches, per token per layer** [code shapes; arithmetic mine]:

| | naive | absorb |
|---|---|---|
| Elements | `H·(192+128)` = 128·320 = **40,960** | `512 + 64` = **576** |
| Bytes (bf16 buffers, default dtype) | 81,920 | 1,152 |
| Per token, all 61 layers | 4,997,120 B ≈ **4.77 MiB** | 70,272 B ≈ **68.6 KiB** |
| Ratio | 71.1× larger | 1 |
| Preallocated at `max_batch_size=8, max_seq_len=16384` | ≈ **610 GiB** (÷ world_size, heads are sharded) | ≈ **8.58 GiB**, **replicated** on every rank (no head dim) |

[measured with the script in §8]. Note the TP subtlety [inference]: the naive cache is sharded by heads, but the latent cache is identical on every rank. At `world_size=16` the per-rank ratio shrinks to 2,560 vs 576 = 4.4×. That is why production serving uses **DP-attention** (attention data-parallel, MoE expert-parallel). The V3.2 SGLang launch uses `--tp 8 --dp 8 --enable-dp-attention` (`V3.2-Exp/README.md:123`), and V3's README mentions SGLang "DP Attention" (`README.md:307`).

In deployment the latent cache is FP8. FlashMLA's doc says each token's MLA cache is **656 bytes** = 512 fp8 + 4 fp32 scales (1×128 tiles) + 64 bf16 RoPE dims kept unquantized "as they are sensitive to precision loss" [doc: `FlashMLA/docs/20250929-hopper-fp8-sparse-deep-dive.md:9`]. V3.2's reference *simulates* this: quantize to FP8 and back before writing the bf16 cache ("we use fp8 kv cache in actual deployment, so here we simulate the precision…", `V32/model.py:569–571`) [code].

**Why absorb exists: decode memory traffic** [inference, arithmetic from config].

Per cached token, per query token, per layer (bf16, batch 1):

| | naive | absorb |
|---|---|---|
| MACs per head per (query, key) pair | 192 (QK) + 128 (PV) = 320 | 576 (QK) + 512 (PV) = 1,088, i.e. **3.4× more math** |
| Bytes read per cached token | 81,920 | 1,152, i.e. **71× fewer** |
| Arithmetic intensity | 2·128·320 FLOP / 81,920 B = **1.0 FLOP/B** | 2·128·1088 / 1,152 = **≈242 FLOP/B** |

Decode is memory-bound, so trading 3.4× more FLOPs for 71× fewer bytes is a big win. Absorb turns attention into **MQA with one 576-dim KV head shared by 128 query heads**. FlashMLA's doc describes decode exactly that way: "128 query heads and 1 key head, where head_dim_k = 576 and head_dim_v = 512" [doc: same file :9].

In prefill (compute-bound, many queries per key) the 3.4× extra math is a loss, so the MHA (naive) mode is better there. The V3.2 paper states the split outright: *"For DeepSeek-V3.1-Terminus, the MHA mode is used for training and prefilling, while the MQA mode is used for decoding."* [doc: `DeepSeek_V3_2.pdf` p6, Fig. 4 caption]. V3.2's code implements it: `if mask is not None: # MHA prefill` materializes per-head K/V via `wkv_b` (:574–589); `else: # MQA decode` uses the absorbed path (:590–606) [code]. **This is the single best real-world example of "prefill and decode are different workloads, so use different algorithms on the same weights"**, which is Luigi's stated end goal.

V3 (not V3.2) uses one global `attn_impl` for both phases, and in absorb mode it re-dequantizes `wkv_b` on every call (`V3/model.py:481`). [inference] With FP8 weights that is a 128·256×512 = 16.8M-element bf16 materialization per layer per step, pure wasted bandwidth in decode. V3.2 caches it (:591–593). That is a textbook "hoist loop-invariant work out of the decode loop" lesson for M6.

The DeepSeek-V2 paper also describes absorbing `W_UV` into `W_O` [paper, not local]. This code does **not** do that: it applies `W_UV` explicitly (:495).

**Applicability.**
- **stories15M (M3):** not needed. Plain MHA with 6 heads, `n_kv_heads = 6`. KV cache per token = 2·6·288·4 B = **13,824 B**; full 256-token context = 3.54 MB [measured arithmetic from the checkpoint header `(288, 768, 6, 6, 6, 32000, 256)`]. At the last position, attention reads ≈3.5 MB versus ≈60.75 MB of f32 weights, so the KV cache is <6% of decode traffic [inference]. Compressing it would be premature for M3–M6.
- **M8 (GQA models):** GQA is the same idea in a gentler form (share K/V across groups of query heads). llama2.c already supports it (`kv_mul`, `run.c:240`, `:292`), so M3 code should index `h / kv_mul` from day one. [inference, configs from memory, verify at M8]: TinyLlama-1.1B has 22 layers, 4 KV heads × 64 dims, giving 2·22·4·64·2 B = 22,528 B/token in fp16.
- **CPU:** the absorb trick is pure linear algebra and would work on CPU. It only pays off when KV traffic rivals weight traffic: long contexts, big batches.
- **GPU sm_89:** an MQA-mode decode kernel (one KV head, many query heads) is a great M7 stretch exercise: load each cached row once into shared memory, use it for all heads. FlashMLA itself is SM90/SM100 only (`FlashMLA/README.md:79` lists SM100/SM103 at HEAD; `:3` says Hopper support moved to commit `ba89a34`) [doc]. So it can't be run here, only read.
- **Server:** KV bytes per token set the max batch size, i.e. throughput. Worth quoting in the post-v1 capacity model (role project 4).

**Teaching hook (M3, optional; really M8/post-v1).** One head, latent `c = [1, 2]`, `W_UK = [3, 1]` (1×2), query `q = 2`.
- Naive: `k = 3·1 + 1·2 = 5`, score `2·5 = 10`.
- Absorb: `q̃ = W_UKᵀ q = [6, 2]`, score `q̃·c = 6 + 4 = 10`.
- Ask: "Which one needs you to store `c` and which one needs `k`? Which is bigger when there are 128 heads?"

**Pitfalls.**
- `softmax_scale` uses the *full* `qk_head_dim = 192` (:434), not 128 or 64.
- `kv_norm` is applied before caching in absorb mode (:484) and before `wkv_b` in naive mode (:473). Mixing them up breaks equivalence.

#### 2.3.3 RoPE with YaRN (`precompute_freqs_cis`, `V3/model.py:297–375`)

**Formulas** (d = `qk_rope_head_dim` = 64, base = 10,000, original context L = 4096, factor s = 40, β_fast = 32, β_slow = 1) [code]:
1. Base frequencies: `θ_i = base^(−2i/d)`, i = 0..d/2−1 (:366). Wavelength `λ_i = 2π/θ_i`. Rotations over the original context: `r_i = L/λ_i`.
2. `find_correction_dim(r) = d·ln(L/(2πr)) / (2·ln base)` (:314–327) solves `r_i = r` for i.
3. `low = floor(find_correction_dim(β_fast))`, `high = ceil(find_correction_dim(β_slow))`, clamped (:329–345). [measured] For V3: raw (10.47, 22.51) → **low = 10, high = 23** (pair indices out of 32).
4. Ramp `γ_i = clamp((i − low)/(high − low), 0, 1)`; `smooth = 1 − γ` (:347–364, :369).
5. `θ'_i = θ_i/s · (1 − smooth) + θ_i · smooth` (:370):
   - i ≤ 10 (fast dims, >32 rotations within 4K): unchanged ("extrapolate").
   - i ≥ 23 (slow dims, <1 rotation): divided by 40 ("interpolate").
   - In between: blended.
   - [measured] ratios θ'/θ at i = 0, 10, 11, 16, 22, 23, 31: 1, 1, 0.925, 0.55, 0.1, 0.025, 0.025.
6. `freqs_cis = polar(1, outer(t, θ'))` for t = 0..max_seq_len−1 (:372–374).
7. Attention temperature ("mscale"): `softmax_scale *= (0.1·mscale·ln s + 1)²` (`MLA.__init__`, :434–437). [measured] With `mscale = 1.0` (671B config sets none, so the default applies) that is 1.36889² = **1.87385**, giving `softmax_scale = 192^−0.5 · 1.874 = 0.13523` (vs 0.07217 plain). The 16B config sets `mscale 0.707`, giving 1.5896.
8. YaRN (steps 3–5 and 7) is applied **only if `max_seq_len > original_seq_len`** (:367, :435). `max_seq_len` is a *runtime buffer-size* setting (default 16,384). [inference] Running the 671B checkpoint with `max_seq_len ≤ 4096` silently changes the model's math (no YaRN, different softmax scale), which is a coupling bug waiting to happen.

**Interleaved vs half-split pairs** [code]. `apply_rotary_emb` pairs **adjacent** elements `(x[2i], x[2i+1])` via `view_as_complex(x.view(..., -1, 2))` (:390). llama2.c does the same (`run.c:265–278`, pairs `i, i+1`; `model.py:64–65` reshapes `(-1, 2)`). V3.2's indexer uses the **non-interleaved** layout, pairs `(x[i], x[i+d/2])`, via the `interleaved=False` path (`V32/model.py:418–419`, :423–424, :463–470). DeepSeek shipped that wrong for ~7 weeks: *"previous versions of the inference demo code contained an implementation discrepancy in Rotary Position Embedding (RoPE) within the indexer module, potentially leading to degraded model performance… the input tensor to RoPE in the indexer module requires a non-interleaved layout, whereas RoPE in the MLA module expects an interleaved layout."* [doc: `V3.2-Exp/README.md:77`, dated 2025.11.17; kernels released 2025.09.29 per `FlashMLA/README.md:26`].

**What a llama2-style model needs** [inference]:
- Only step 1 plus the rotation. No YaRN, no mscale.
- stories15M: head_size 48, base 10000, rotate the whole head.
- llama2.c recomputes `powf/cosf/sinf` for every pair of every token of every layer (`run.c:265–270`). DeepSeek precomputes a `(max_seq_len, d/2)` table once. That's a small M6 item (for stories15M: 144 pairs × 6 layers × 3 transcendental calls per token; measure before claiming it matters).
- **M8 pitfall:** HF Llama/Qwen checkpoints use the half-split ("rotate_half") layout. A llama2.c-style interleaved rotation on HF weights is wrong unless q/k rows are permuted at load. DeepSeek's own bug is the perfect war story.
- Qwen2.5 uses base 1e6 [from memory, verify at M8].

**Teaching hook (M3).** "Your RoPE matches the reference at position 0 but not at position 5. Why does position 0 always pass?" (Because the rotation by angle 0 is the identity.) Then: "Your RoPE matches llama2.c but not HF TinyLlama at M8. What's the one-line hypothesis?" (Pair layout.)

---

### 2.4 MoE Gate: only what M8 and post-v1 need

**Where.** `Gate` (`V3/model.py:535–598`, `V32/model.py:646–709`), `MoE` (:636–693 / :747–804).

**How (671B config: 256 routed experts, 8 active, 8 groups of 32, top-4 groups, sigmoid, route_scale 2.5)** [code]:
1. `scores = linear(x, W_gate)`: `(T, 256)` (:576). V3.2 does it in fp32 (:687).
2. `softmax` (V2 and 16B configs) or `sigmoid` (V3; `"score_func": "sigmoid"` in config_671B.json) (:577–580).
3. `original_scores = scores` (:581). **The bias is added only for selection** (:582–583). The bias parameter exists only `if self.dim == 7168` (:564), a magic number standing in for "is V3". In the checkpoint it's `e_score_correction_bias`, renamed to `bias` (`convert.py:61`).
4. Group-limited routing (`n_groups > 1`, :584–592): view as `(T, 8, 32)`.
   - Group score is `amax` without bias, or the **sum of the top-2** with bias (:586–589).
   - Keep the top `topk_groups = 4` groups and set the other groups' scores to −inf (:590–592).
5. `indices = topk(scores, 8)` (:593). `weights = original_scores.gather(indices)` (:594): **weights come from the un-biased scores**.
6. Sigmoid only: renormalize weights to sum to 1 (:595–596). Then `× route_scale` (:597).
7. `MoE.forward`:
   - `bincount(...).tolist()` (:683) gives per-expert counts (a **GPU→host sync every MoE layer**, i.e. 58 per forward for 61 − 3 dense layers [inference]).
   - Loop over *local* experts, `torch.where(indices == i)` gathers their tokens, `y[idx] += expert(x[idx]) * weight` (:684–689).
   - Shared expert on all tokens (:690), then `all_reduce` (:691–692).

**Why** [paper, not local; README confirms the name]:
- **"Auxiliary-loss-free load balancing"** (`README.md:49`, :65). During training a per-expert bias is nudged down when an expert is overloaded and up when it is underloaded. It steers *which* experts get picked without distorting *how much* each one contributes (hence weights from `original_scores`). The update rule is in the V3 paper, not this code.
- **Group limiting** caps how many device/node groups a token's experts span (the paper says at most 4 nodes), which bounds all-to-all traffic. Here it's just masking.

**Teaching hook (post-v1, role project 3).** 8 experts in 2 groups of 4, top-1 group, top-2 experts, sigmoid scores `[0.9, 0.5, 0.1, 0.1 | 0.6, 0.7, 0.2, 0.1]`.
- **No bias:** group sums of top-2 are 1.4 and 1.3, so group 0 wins and the picks are experts {0, 1}.
- **Bias −0.5 on the overloaded expert 0:** group 0 becomes 0.4 + 0.5 = 0.9 < 1.3, so group 1 wins and the picks are {5, 4}. Weights come from the *original* 0.7 and 0.6, normalized to 0.538 and 0.462, then × 2.5.
- Ask: "What did the bias change, and what didn't it change? Why is that a load *balancer* rather than a *model change*?" This is a feedback controller, the same shape of problem as routing requests across CPU/GPU workers. `~/refs/inference/deepseek/EPLB` (expert-parallel load balancer) and `LPLB` are the local follow-ups (not covered here).

**Applicability.**
- **M8:** the planned models (TinyLlama, Qwen2.5-0.5B class) are dense. MoE is **not needed for v1**.
- **CPU:** `config_16B.json` (64 experts, 6 active, dim 2048, 27 layers) is the smallest MoE here, and at ~16B parameters it does not fit 15 GB RAM in bf16 [inference].
- **Server:** expert-level load imbalance (hot experts) and the per-layer host sync are exactly the kind of latency sources role projects 3 and 6 care about.
- **Not applicable:** the all-reduce-based expert parallelism is only simple because activations are replicated under TP. Real EP uses all-to-all dispatch (DeepEP, local), which doesn't matter on one GPU.

---

### 2.5 FP8 block quantization and `kernel.py`, compared with Q8_0

**Formats** [code/doc]:
- Weights are `torch.float8_e4m3fn`: 1 sign, 4 exponent, 3 mantissa bits, **max 448**, no infinities, the "fn" NaN encoding (`README_WEIGHTS.md:77–78`).
- [measured] `torch.finfo`: max 448, min normal 2⁻⁶ = 0.015625, eps 0.125. e5m2 (not used here) has max 57,344, eps 0.25.
- [measured] Casts: 448 → 448, **464 → 448**, **500 → NaN** (no saturation), 1/1024 → 0 (underflow).
- Scales are fp32. `scale_fmt="ue8m0"` (V3.1/V3.2 configs) rounds each activation scale **up to a power of two**, an exponent-only "UE8M0" scale (`V3/kernel.py:29–31`; TileLang bit tricks `V32/kernel.py:20–33`). DeepGEMM notes that SM100 *requires* packed UE8M0 scales while SM90 takes fp32 (`DeepGEMM/README.md:65–68`) [doc].

**Granularity** [doc/code]:
- **Weights:** one fp32 `weight_scale_inv` per **128×128 block**. Dequantize = `(128x128 block) * weight_scale_inv`. Non-multiples of 128 are zero-padded for the scale computation (`README_WEIGHTS.md:83–90`).
- The name "scale_inv": the stored number *multiplies* to dequantize, so the quantizer divided by it.
- **Activations:** dynamic, "per-token-per-128-channel" (`README_WEIGHTS.md:81`, :92). One scale per token per 128 contiguous channels, computed at runtime.

#### `act_quant` (`V3/kernel.py:9–57`), line by line [code]
- :51–52: input must be contiguous with the last dim divisible by 128, so a flat run of 128 elements is always inside one row: one group = one program.
- :53–54: output `y` is fp8, same shape. `s` has shape `(*x.shape[:-1], K/128)` in fp32.
- :55: 1-D grid, `numel/128` programs.
- :23–25: program `pid` loads its 128 elements and upcasts to fp32.
- :26–27: `amax = max|x|`, clamped to ≥ 1e-4 (avoids a zero or denormal scale for an all-zero group).
- :28: `s = amax / 448`, so the largest element maps exactly to the fp8 max.
- :29–31: `ue8m0` gives `s = 2^ceil(log2 s)` (rounding up keeps `|x/s| ≤ 448`).
- :32–35: `y = x / s`, cast to fp8, store `y` and one scale.
- V3.2's TileLang version (`V32/kernel.py:36–111`) tiles 32 rows × 128 channels per block (128 threads), does `reduce_absmax`, and **clamps** to ±448 before the cast (:75–78). That is safer given the NaN-on-overflow cast above.

#### `weight_dequant` (`V3/kernel.py:60–110`) [code]
- :76–80: 2-D grid over 128×128 tiles, row and column offsets.
- :81–83: load the tile with edge masks, upcast.
- :78, :84: exactly **one scalar scale per tile**: `s_ptr[pid_m * cdiv(N,128) + pid_n]`.
- :85–86: `y = x * s`, stored in the default dtype (bf16).

It is used by `linear()` when `gemm_impl == "bf16"`, **the V3 default** (`model.py:16`, :155–157), and by `fp8_cast_bf16.py`. So by default V3 stores weights in FP8 but computes in bf16, materializing a bf16 copy of each weight on every call [inference: an extra ~2+2 bytes of traffic per weight per call on top of the 1-byte read].

#### `fp8_gemm` (`V3/kernel.py:113–196`) [code]
Computes `C[M,N] = A[M,K] · B[N,K]ᵀ`. A is the activations (fp8, scales `a_s[M, K/128]`); B is the weight (fp8 `(out,in)` row-major, scales `b_s[N/128, K/128]`).
- :113–118: 36 autotune configs (`BLOCK_M ∈ {16,32,64}`, `BLOCK_N ∈ {32,64,128}`, stages 3–6), with **`BLOCK_K` fixed at 128** so every K-step lines up with exactly one scale group. Tuned per `(N, K)` (weight shape), with M (tokens) left dynamic.
- :148–149: `offs_m % M`, `offs_n % N` wrap out-of-range rows so loads stay in bounds. The store is masked later (:171).
- :151–152: A tile pointers are row-major `(BM, BK)`. B pointers `offs_n[None,:]*K + offs_k[:,None]` read a `(BK, BN)` tile of Bᵀ from row-major B.
- :153–154: activation scale row = `offs_m * k`. Weight scale row = `(offs_n // 128) * k`. This reuses `BLOCK_SIZE_K` as the weight block size along N, a silent coupling that only works because both are 128.
- :156–166, the K loop, and the key line (:162):
  ```python
  accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
  ```
  Each 128-deep partial product comes out of the tensor core in fp32, is rescaled by (per-token activation scale × per-block weight scale), and is added to an fp32 accumulator. The scales change with k, so they can't be factored out of the sum: this is the **per-K-block "promotion"**. V3.2's TileLang GEMM does the same explicitly, commented `# Promote to enable 2xAcc` (`V32/kernel.py:159–163`). Because `block_N = 128 = group_size` there, the weight scale is a *scalar* per tile per k (:155–157). The V3 paper motivates promotion by limited accumulation precision inside Hopper FP8 tensor cores [paper, not local].
- :167–172: cast to the default dtype and do a masked store.

**Comparison with the planned M5 Q8_0 (int8, block 32, one scale)** and llama2.c `runq.c`:

| | DeepSeek FP8 | Q8_0 (ggml) / llama2.c `runq.c` |
|---|---|---|
| Element type | e4m3 float (3-bit mantissa, relative precision ~6%) | int8 uniform grid, 255 levels |
| Weight scale granularity | 128×128 tile (2-D) | 1×32 (Q8_0) or 1×GS, GS = 64 by default in `export.py:182` |
| Activation granularity | 1×128 per token, dynamic | `runq.c` quantizes activations 1×GS dynamically (`quantize` :145–171) |
| Scale value | `amax/448` (optionally power of 2) | `amax/127` (`runq.c:161`) |
| Storage overhead | 4 B per 16,384 weights ≈ 8.002 bits/weight; activations 8.25 bits | Q8_0: 34 B per 32 = 8.5 bits/weight (fp16 scale) [from ggml, not local yet] |
| Inner loop | fp8 MMA over 128, fp32 promote × a_s × b_s (`kernel.py:162`) | int32 dot over GS, then `val += ival * w_s * x_s` (`runq.c:332–337`) |

The last row is the teaching gold: **`runq.c:336` and `kernel.py:162` are the same algorithm.** Integer or low-precision dot inside a group, float rescale per group, float accumulation across groups.

**Worked example** [measured], block `x = [0.9, −0.05, 0.3, 4.48]`:
- **FP8:** `s = 4.48/448 = 0.01`, so `x/s = [90, −5, 30, 448]`. In e4m3 that becomes `[88, −5, 30, 448]` (90 is not representable: 3 mantissa bits give 88 or 96). Dequantized: `[0.88, −0.05, 0.30, 4.48]`.
- **Q8_0-style int8:** `d = 4.48/127 = 0.035276`, so `q = [26, −1, 9, 127]`. Dequantized: `[0.917, −0.035, 0.317, 4.48]`.
- **The lesson:** FP8 keeps *relative* precision, so the small −0.05 is exact while the mid-size 0.9 gets a 2% error. Int8 keeps *absolute* precision (step 0.035): every value is off by up to half a step, and the small value gets a 30% error.
- **With UE8M0:** `s` rounds up to 2⁻⁶ = 0.015625. The values now span only up to 286.7 instead of 448, which barely matters for a float grid [inference]. For an int grid, a power-of-two scale would waste up to 1 bit for *every* value.

**Synthetic error comparison** [measured; 4096×4096 Gaussian, 8 outlier columns ×20, relative RMSE; teaching illustration only, not a claim about real weights]:
- int8 1×32: **0.0068**
- int8 1×128: 0.0122
- fp8 1×32: 0.0223
- fp8 1×128: 0.0233
- fp8 128×128: 0.0263

So on bell-shaped data, int8 with small blocks is *more* accurate than FP8. FP8's advantages are hardware (native tensor-core FP8 on sm_89+), wide dynamic range inside a block, and training (DeepSeek trained in FP8, so the FP8 weights are "native" with no post-hoc quantization loss: "Since FP8 training is natively adopted in our framework, we only provide FP8 weights", `README.md:240`) [doc].

**Why 128×128 for weights and 1×128 for activations** [inference + paper, not local]:
- Activations have per-token outlier channels, so they need fine per-token groups.
- Weights are smoother, and 2-D tiles mean a GEMM tile needs one scale per K-step (cheap epilogue math).
- A square block is transpose-symmetric, so the same quantized weight serves `W` and `Wᵀ` in training (paper).

**Applicability.**
- **CPU (M5).** This machine is a Ryzen 9 5900X (Zen 3): AVX2 + FMA + F16C, **no AVX-512, no AVX-VNNI, no BF16/FP8 instructions** [measured: `/proc/cpuinfo`].
  - FP8 on CPU is pure software: a 256-entry `u8 → f32` lookup table per element. So **Q8_0 int8 remains the right M5 format**. It gets the AVX2 `maddubs`/`madd` integer dot path (ggml's approach) and matches `runq.c`.
  - Keep the DeepSeek lessons: dynamic per-token activation groups, fp32 promotion per group, clamp before cast, scale from amax.
- **GPU (M7).**
  - The block-scaled GEMM structure (`kernel.py:162`) is the M7 quantized-matmul design.
  - **[measured] with the installed nvcc 12.0, FP8 `mma.sync` for sm_89 fails in ptxas**, while INT8 `mma.sync.m16n8k32.s8.s8.s32` compiles. So the "low-precision GPU kernel" (role project 2) is buildable today as **W8A8 int8 with per-group scales**, the direct GPU twin of the CPU Q8_0 path. That is also what the D1-style CPU-oracle tolerance method wants.
  - FP8 needs a toolkit upgrade (see §7).
  - TMA/WGMMA-based FP8 GEMMs (DeepGEMM) are Hopper/Blackwell-only (`DeepGEMM/README.md:31`) [doc].
- **Server:** FP8 KV cache (656 B/token/layer) is a capacity lever for post-v1.

**Pitfalls.**
- Overflow gives NaN with a plain cast [measured]. Always derive the scale from amax (or clamp).
- The amax floor (1e-4) matters for all-zero groups (padding).
- Scale tensors must be sharded consistently with weights (see 2.8).
- In V3, `ColumnParallelLinear.forward` / `RowParallelLinear.forward` call `linear(x, self.weight, self.bias)` **without `scale_fmt`** (`V3/model.py:233`, :262), unlike `Linear.forward` (:205). With `gemm_impl="fp8"` and a ue8m0 config, the parallel layers would silently use fp32 scales [inference; V3.2 passes it, `V32/model.py:233`, :263].

---

### 2.6 `generate.py`: the loop and `sample()`

#### The loop (`V3/generate.py:30–78`) [code]
- :51–53: `prompt_lens`. `total_len = min(max_seq_len, max_new_tokens + max(prompt_lens))`.
- :54–56: a `(B, total_len)` token matrix filled with −1, prompts **left-aligned** from position 0.
- :57–59: `prev_pos = 0`, `finished`, `prompt_mask = tokens != −1`.
- :60–71, the loop, which runs `for cur_pos in range(min(prompt_lens), total_len)`:
  - :61 `logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)`. The **first iteration is the prefill** of the first `min_prompt_len` tokens of every row. Every later iteration feeds exactly **one** token (the one written at `cur_pos−1`) at `start_pos = prev_pos`.
  - :62–65: sample (temperature > 0) or argmax.
  - :66: `torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)`. Rows whose prompt is longer than `cur_pos` are **teacher-forced**: the model's prediction is discarded and the prompt token kept.
  - :67–70: write the token, update `finished` (EOS, only for generated positions), `prev_pos = cur_pos`, and stop when all rows are finished.
- :72–78: slice each row's `[prompt_len : prompt_len + max_new_tokens]` and cut at EOS.

**Worked example (teaching hook, M4).** Prompts of lengths 3 and 5, `max_new_tokens = 4`, so `total_len = 9`.

| cur_pos | call | row 0 (len 3) | row 1 (len 5) |
|---|---|---|---|
| 3 | `forward(tokens[:,0:3], 0)`: **prefill ×3** | samples pos 3 | forced prompt token at pos 3 |
| 4 | `forward(tokens[:,3:4], 3)` | samples pos 4 | forced pos 4 (its prompt "prefill" continues **one token per step**) |
| 5–8 | 1 token each | samples | samples 5..8 |

Row 0 generated positions 3–8 (6 tokens) but keeps only 4. Row 1's last 2 prompt tokens were processed at decode speed. This static batching wastes work in both directions. It is the motivation for per-sequence positions, continuous batching and chunked prefill in the post-v1 server [inference].

**Other loop facts** [code]:
- Chat mode rebuilds the **whole conversation** and regenerates from position 0 every turn (`messages.append` → `apply_chat_template(messages)` → `generate`, :139–141). No prefix/KV reuse across turns (a prefix-caching motivation).
- `finished.all()` (:70) and MoE `.tolist()` are host syncs every step.
- `torch.set_num_threads(8)` (:110) is for host-side ops only.
- Both files use `torch.inference_mode()` (`generate.py:30`, `model.py:772`).

**Mapping to M4** [inference]: TTFT = time of the first `forward` (prefill) + first sample. Decode tok/s = (generated − 1) / time of the later steps. For batch 1 (Luigi's CLI) this loop reduces to: one prefill call, then decode calls. That's the explicit two-phase structure M4 requires, and unlike llama2.c (`run.c:747–759`) it measures the phases separately.

#### `sample()` (`V3/generate.py:14–27`) [code]
```python
logits = logits / max(temperature, 1e-5)
probs = torch.softmax(logits, dim=-1)                 # V3.2: dtype=torch.float32 (V32/generate.py:26)
return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
```

**Math: why `argmax_i p_i / E_i` with `E_i ~ Exp(1)` samples from `p`** [standard result; measured]:
1. `p_i/E_i` is largest ⇔ `E_i/p_i` is smallest.
2. If `E ~ Exp(1)` then `E/p ~ Exp(rate = p)`.
3. For independent exponentials with rates λ_i, `P(argmin = j) = λ_j / Σλ_i`. With λ = p and Σp = 1 that equals `p_j`. ∎
4. Equivalently, `log(p_i/E_i) = log p_i + G_i` where `G_i = −log E_i ~ Gumbel(0,1)`: this is the **Gumbel-max trick**.
5. **Corollary:** softmax normalization is unnecessary. `argmax_i (logit_i / T − log E_i)` gives the same distribution, because the log-partition constant is shared by all i.
6. [measured] Target `p = [0.5, 0.3, 0.15, 0.05]`; empirical frequencies over 400k draws are `[0.5003, 0.2995, 0.1501, 0.0501]`.

**Tiny hook.** `p = [0.7, 0.3]`, draws `E = [1.2, 0.2]` give `p/E = [0.58, 1.5]`, so token 1 wins even though it's less likely. "Over many draws, how often does token 1 win?" (30%.)

**What this means for low-latency sampling (role project 1)** [inference]:
- **No sort, no prefix sum.** llama2.c's top-p does a `qsort` of the candidates (`sample_topp`, `run.c:624`), which is O(n log n). Inverse-CDF sampling (`sample_mult`, `run.c:603`) needs a cumulative sum. The exponential race is one elementwise pass plus one argmax reduction, both trivially parallel and SIMD/GPU-friendly, with no data-dependent control flow.
- **It distributes across vocab shards.** Each thread, or each GPU with a vocab-sharded head (`ColumnParallelLinear` head, `model.py:769`), can compute its local `max(logit/T − log E)` and index, then reduce `(value, index)` pairs. That is 2 numbers per shard instead of `all_gather`-ing the full vocab like `model.py:794–797`. DeepSeek's demo doesn't do this, but it's a clean optimization story for the server.
- **It can fuse into the LM-head epilogue** on GPU: add noise to each logits tile and keep a running argmax, never writing all logits to memory.
- **Cost:** V random numbers plus V logs per step, vs 1 random number for inverse CDF. On CPU, 32,000 `ln` calls per token must be measured against llama2.c's sort before claiming a win (M6).
- **Top-p/top-k** still need global information (a threshold). They combine by masking first. Filtering then sampling is where sort-free threshold search (e.g. by bisection on the probability mass) becomes a nice project.
- **Determinism:** reproducibility needs a fixed seed *and* the same noise per vocab entry. Under TP, every rank must draw identical noise (the same `torch.manual_seed`, `generate.py:111`). Otherwise ranks would sample different tokens.

**Applicability.** CPU M4 (correctness: compare distributions, not exact tokens, when the RNG differs from llama2.c's xorshift), M6 (measure sampling's share of per-token time, as the plan requires), M7 (GPU sampling kernel), server (batched sampling).

**Pitfalls.**
- `probs.div_` is in-place on the softmax output.
- `temperature=0` goes through a separate argmax branch; `max(T, 1e-5)` only guards the division.
- V3 computes softmax in bf16 unless the logits are fp32 (V3.2 forces fp32). With a 129k vocab, bf16 probabilities are coarse [inference].

---

### 2.7 Tensor-parallel inference → M6 multithreading

**Where** [code]: `ParallelEmbedding` `V3/model.py:89–128`; `ColumnParallelLinear` :208–234; `RowParallelLinear` :237–267; usage in MLA (:425–433), MLP (:518–520), head (:769, :794–797), MoE (:658–666, :691–692); globals set from `torch.distributed` (:757–759); launch via `torchrun` (`README.md:296`), one process per GPU, NCCL (`generate.py:100–104`).

**Math.** For `y = W x` with `W ∈ R^{out×in}` (PyTorch layout `(out, in)`):
- **Column-parallel** (split output rows, `ColumnParallelLinear`): rank r holds `W_r = W[r·out/P : (r+1)·out/P, :]` and computes its **own slice of y**. No communication. Heads split naturally: `n_local_heads = n_heads / world_size` (:416).
- **Row-parallel** (split input columns, `RowParallelLinear`): rank r holds `W[:, r·in/P : …]` and the matching slice of x (which is exactly the output slice of the previous column-parallel layer), and computes a **partial sum** `y_r = W_{:,r} x_r`. Then `y = Σ_r y_r` via `all_reduce` (:263–264).
- **The Megatron pattern:** column → local elementwise or per-head work → row → **one all_reduce**.
  - MLP: w1, w3 column, SwiGLU local, w2 row.
  - MLA: wq_b and wkv_b column (per head), attention local per head, wo row.
  - The small `wq_a` and `wkv_a` are plain `Linear`, computed redundantly on every rank (:427, :430).
- **Embedding:** vocab-sharded. Each rank embeds ids in its range, zeroes the others, and `all_reduce` sums (:120–127).
- **Head:** column (vocab) parallel, then `all_gather` of logits (:794–797).
- **MoE:** whole experts per rank, zeros elsewhere, `all_reduce` sums (:665–666, :691–692).
- V3.2 does the all-reduce **in fp32** (`y = y.float(); dist.all_reduce(y)`, `V32/model.py:264–266`), a numerics fix for summing bf16 partials across 8–16 ranks.

**Mapping to CPU threads (M6)** [inference, to be measured]:

| TP concept | CPU-thread analogue | Reference |
|---|---|---|
| Column-parallel | Each thread computes a disjoint block of output rows. No reduction, one barrier at the end. The natural matvec split for decode. | llama2.c `matmul` `#pragma omp parallel for` over rows `i < d` (`run.c:221–222`); `runq.c:323–324` |
| Per-head parallel attention | Threads own heads | `run.c:283–284` (`omp parallel for` over heads) |
| Row-parallel | Threads split the K (input) dimension and produce partial output vectors, then a reduction. Extra buffer plus a sync; worse than column-split for a lone matvec. | none in llama2.c |
| Column→row pair (Megatron MLP) | Thread t computes `h_t = silu(W1_t x) ⊙ (W3_t x)` for **its** hidden slice, then `y_t = W2[:, slice_t] h_t`, then **one** reduction of 24 small `dim`-length partials | [inference] fewer barriers (1 vs 2) and `h_t` stays in the thread's L1/L2. Worth an M6 experiment |
| Vocab-sharded head | Split the 32,000 classifier rows across threads | For stories15M the head is 32000×288 = 9.2M of ~15.2M MACs per token (**≈61%**) [arithmetic from the header]: the single biggest matmul |
| `all_reduce` | Sum of per-thread partial buffers (or atomics; don't) | none |
| `all_gather` | Threads write disjoint slices of one shared output: free on shared memory | none |

Key contrast to teach: on GPUs across a network, communication is the bottleneck, so TP minimizes collectives. On one CPU socket, "communication" is cache-coherence traffic plus barrier latency. The column split wins by default because it needs no reduction. The 5900X is Zen 3 with 12 cores / 24 threads; hyperthreads share a core's FMA units, so a memory-bound decode may not scale to 24 [inference: measure 1, 6, 12, 24 threads in M6]. (`lscpu` reports L2 6 MiB = 12×512 KiB; L3 shows as 32 MiB in WSL.)

**Applicability.**
- **Server:** multi-GPU TP is out of scope (one GPU). The ideas return in the server as "one engine per worker" plus a router (role project 3).
- **Fault tolerance (role project 5):** a TP group is all-or-nothing. If one rank dies, NCCL collectives hang for the whole group and its KV caches (module buffers on every rank) are lost, so the request must be requeued and re-prefilled [inference].
- V3.2 adds a **cross-rank consistency check**: it broadcasts rank 0's indexer top-k and asserts equality (`V32/model.py:484–486`). Non-deterministic top-k across ranks would make the ranks attend to different tokens and silently diverge.

**Teaching hook (M6).** "`W` is 4×4, you have 2 threads. Split A: thread 0 takes rows 0–1. Split B: thread 0 takes columns 0–1. Which split needs an extra step after both threads finish, and what is it?"

**Pitfalls.**
- `world_size` must divide heads, vocab, expert count and 128-row scale blocks (asserts at :101, :219, :248, :658; `convert.py:73`).
- With threads, uneven splits and false sharing on output buffers (align per-thread slices to 64-byte cache lines).

---

### 2.8 `convert.py` / checkpoint layout → lessons for M2

**Where.** `V3/convert.py:11–85`; `README_WEIGHTS.md`; `fp8_cast_bf16.py:12–103`.

**How** [code]:
1. `mapping` (:11–30): the HF module name maps to `(engine name, shard dim)`. Shard dim is 0 = split output rows (column-parallel: `embed`, `wq`, `wq_b`, `wkv_b`, `w1`, `w3`, `head`), 1 = split input columns (row-parallel: `wo`, `w2`), `None` = replicate (norms, `wq_a`, `wkv_a`, `gate`).
2. Iterate every tensor of every `*.safetensors` shard (:50–52).
3. Skip `model.layers.61` = the MTP module (:53–54). The demo has no speculative decoding. The MTP layer is 11.5B unique parameters (`README_WEIGHTS.md:38`).
4. String renames (:56–61): strip `model.`, `self_attn→attn`, `mlp→ffn`, `weight_scale_inv→scale`, `e_score_correction_bias→bias`.
5. `key = name.split(".")[-2]` (:62), which must be in `mapping` (assert :63), then `name.replace(key, new_key)` (:64–65). The FP8 scale inherits its weight's shard dim: `...q_b_proj.scale` has key `q_b_proj`, so dim 0, so the `(out/128, in/128)` scale grid is split in the same proportion. [inference] The `"scale": ("scale", None)` entry (:29) appears unreachable.
6. Per rank (:66–76): routed experts are kept whole on the rank that owns index `idx` (from `name.split(".")[-3]`, :68–71). Other tensors are `narrow(dim, i·shard, shard).contiguous()` (:72–75), after asserting divisibility (:73).
7. Save `model{i}-mp{mp}.safetensors` (:80–81), copy the tokenizer files (:83–85). `generate.py:119` loads exactly the file for `rank`, `world_size`.
8. The V3.2 `convert.py` only adds 4 indexer names (`wq_b`, `wk`, `k_norm`, `weights_proj`, :30–33) [code: diff].

`fp8_cast_bf16.py` [code]:
- It uses the HF `model.safetensors.index.json` `weight_map` (tensor name → shard file, :34–37) because a weight's `_scale_inv` may be in **another** shard file (:44–61).
- It keeps at most 2 loaded shard files on the GPU (:90–94) and rewrites the index without the scale entries (:96–103).

**Lessons for M2 (llama2.c `.bin`) and M8 (safetensors/GGUF)** [inference]:
- **Order vs names.** llama2.c's format is a 7-int header plus a flat f32 blob whose *order* is the contract (`run.c` Config :19–27; `memory_map_weights` :111–140 walks pointers in a fixed sequence). A single order mistake gives plausible-looking garbage. DeepSeek/HF use **named** tensors, so a missing or renamed tensor is a hard error (the `assert` at `convert.py:63`, strict `load_model`). For M2: after mapping, print each tensor's name, shape and first values and compare with `model.py`'s `state_dict` (that *is* the M2 checkpoint evidence).
- **Keep a name-mapping layer** (checkpoint name → engine field) separate from the model code. DeepSeek puts it in an offline script, so the runtime model only knows its own names. For the Rust engine: one `load` module that owns all format knowledge (a deep module) and returns a `Weights` struct.
- **Validate shapes against the config at load**, not at first use. DeepSeek gets this implicitly from `load_model`. In Rust, make a shape mismatch a `Result::Err` at load.
- **Quantized formats come with their scales as sibling tensors** (`X.weight` + `X.weight_scale_inv`), and scales must travel with the weights through every transform (sharding, transposes). This matters for the M5 on-disk format decision.
- **Sharded checkpoints plus an index file** are the norm for real models (M8). A loader that handles "tensor name → file → offset" generalizes to TinyLlama/Qwen safetensors.
- **mmap vs read (the M2 decision):** DeepSeek copies (`get_tensor`, `narrow().contiguous()`), and its `convert.py` holds **all ranks' state dicts in RAM at once** (:48, :76). That is fine on a big server, not on 15 GB. safetensors and llama2.c's format are both designed to be mmap-able; llama2.c mmaps (`read_checkpoint` :142–162).

**Teaching hook (M2).** "llama2.c's loader has no tensor names. What goes wrong, and when do you find out, if `w1` and `w3` are swapped on disk?" (Nothing crashes. SwiGLU becomes `silu(W3 x) ⊙ W1 x`, so logits are wrong, and D1's comparison is the only thing that catches it.)

---

### 2.9 V3.2-Exp: DeepSeek Sparse Attention (DSA)

**What it adds (exactly)** [doc + code]. The paper says the *only* architectural change from V3.1-Terminus is DSA, which has two parts (`DeepSeek_V3_2.pdf` p1):
1. A **lightning indexer**. The score between query token t and past token s is
   `I_{t,s} = Σ_{j=1..H_I} w_{t,j} · ReLU(q_{t,j} · k_s)`
   with H_I = 64 indexer heads (`index_n_heads`), d_I = 128 (`index_head_dim`). ReLU was chosen "for throughput consideration", and the indexer "can be implemented in FP8" [doc p1].
2. **Fine-grained token selection.** Each query attends only to the KV entries with the top-k index scores, k = 2048 (`index_topk`) [doc p1, p3; config].

It is instantiated on MLA's **MQA mode**, because "each key-value entry must be shared across multiple queries for computational efficiency" [doc p2].

**Where** [code]:
- **`Indexer` (`V32/model.py:435–487`):**
  - `wq_b: q_lora_rank → 64·128`. It reuses MLA's compressed, normalized query `qr` (:560, :583).
  - `wk: dim → 128`, a **single** key head shared by all 64 index heads (MQA-style).
  - `k_norm` is a **LayerNorm** (:447).
  - `weights_proj: dim → 64` (the per-head weights `w_{t,j}`, fp32, :449).
  - Caches: `k_cache` **fp8** `[B, S, 128]` plus `k_scale_cache` fp32 `[B, S, 1]` (:453–454).
- **Forward (:457–487):**
  - Split q and k into rope and nope parts; RoPE **non-interleaved** (:462–471).
  - Hadamard-rotate q and k (:472–473).
  - FP8-quantize q and k with 1×128 groups (:474–475).
  - Write the k cache at `[start_pos:end_pos]` (:476–477).
  - Head weights: `w = weights_proj(x) · H_I^{-1/2} · q_scale · softmax_scale` (:478–479).
  - `fp8_index(...)` gives `(B, S, end_pos)` scores (:480). Add the causal mask (:481–482). `topk(min(2048, end_pos))` (:483). Cross-rank broadcast and assert (:484–486).
- **Use in `MLA.forward`:** prefill (:583–586) builds `index_mask (B, S, S)` = −inf everywhere except the selected indices, adds the causal mask, and adds it to the dense MHA scores. Decode (:600–602) does the same with a `(B, 1, end_pos)` mask on the MQA scores.
- **`fp8_index_kernel` (TileLang, `V32/kernel.py:199–274`):**
  - Grid `(b, m, ceil(n/512))`. Each block loads one query's `(H, d)` fp8 matrix into shared memory once (:217–218).
  - It streams 128 keys at a time, pipelined 2 stages (:223–228).
  - `logits = k_smem · q_smemᵀ` on tensor cores (:231–238).
  - `relu(logits) * w_h` (:240–241), sum over heads (:243–244), `× k_scale` (:246–247), store (:249).
  - The docstring spells out the math (:269–272). The `max(…,0)` happens on raw fp8 products *before* the positive scales. That is fine because the scales are positive, and w (which may be negative) is applied after the ReLU, matching the formula [inference].
- **Other kernels:**
  - V3.2's `kernel.py` is **TileLang** (Python DSL → CUDA; `requirements.txt`: `tilelang==0.1.6`).
  - `pass_configs` disable warp specialization and TMA lowering (`V32/kernel.py:9–13`), so the comment `# TMA store` at :164 is aspirational.
  - The Hadamard transform comes from the `fast_hadamard_transform` CUDA package (:430).
  - Production kernels: the indexer "weighted ReLU MQA logits" are in DeepGEMM (#200). There is a non-paged version for prefill and a paged one for decode (`DeepGEMM/README.md:19–20`, :88–110). Sparse attention is in FlashMLA (#98) [doc: `V3.2-Exp/README.md:79–83`]. FlashMLA claims "up to 640 TFlops during prefilling and 410 TFlops during decoding" for these kernels [doc: `FlashMLA/README.md:26`; Hopper numbers, not reproducible here].

**The reference does not realize the savings** [code + inference]. Both branches compute the **full dense** score matrix and then add −inf outside the top-k (:580–588, :596–604). So the demo's attention cost is dense plus the indexer. It is a semantics spec. The paper says the same for short prefill: *"for short-sequence prefilling, we specially implement a masked MHA mode to simulate DSA, which can achieve higher efficiency under short-context conditions"* [doc p5].

**How it changes decode cost** [doc for the O() claims; my arithmetic for the numbers]. The paper: DSA "reduces the core attention complexity of the main model from O(L²) to O(Lk)… Although the lightning indexer still has a complexity of O(L²), it requires much less computation" [doc p4]. Per decode token, *per layer*, using the deployed FP8 layouts (656 B/token MLA entry, 128 + 4 B/token indexer key):

| Context L | Dense MQA-mode MLA (bytes, MACs) | DSA: indexer + 2048 selected (bytes, MACs) | Bytes ratio | MAC ratio |
|---|---|---|---|---|
| 4,096 | 2.69 MB, 0.57 G | 0.54 + 1.34 = 1.88 MB, 0.32 G | 1.4× | 1.8× |
| 32,768 | 21.5 MB, 4.56 G | 4.33 + 1.34 = 5.67 MB, 0.55 G | 3.8× | 8.2× |
| 131,072 | 86.0 MB, 18.25 G | 17.3 + 1.34 = 18.6 MB, 1.36 G | 4.6× | 13.4× |

[measured arithmetic, §8] The indexer's MACs are FP8 (cheaper per MAC on tensor cores) and the MLA MACs are bf16, so the real speed ratio differs. The indexer becomes the dominant *bytes* term at long L: it's still linear in L per decode token. Also, the selected 2048 entries are a **gather** (scattered rows), so it needs a paged, gather-friendly cache layout.

The paper's Figure 3 shows cost per million tokens vs position for prefill and decode, "estimated from benchmarking the actual service deployed on H800 GPUs, at a rental price of 2 USD per GPU hour" [doc p4–5]. Exact values are only readable off the plot, so don't quote numbers.

**Training facts** [doc p2–3]:
- The indexer is warmed up with a KL loss against the head-summed main attention (1000 steps, 2.1B tokens, lr 1e-3).
- Then sparse training selects 2048 KV tokens per query (15,000 steps, 943.7B tokens).
- The indexer input is **detached**: the indexer learns only from its KL loss.

**Other V3.2 techniques worth stealing**:
- **Hadamard rotation before FP8 quantization** (`rotate_activation`, :428–432). `H/√d` is orthogonal, so `(Hq)·(Hk) = q·k` exactly, but the rotation spreads outlier energy across all 128 channels, so the per-group amax (and so the quantization error) shrinks [inference; the same idea as the QuaRot/SpinQuant line of 4-bit work]. For M5's 4-bit format this is a real option: a fast Walsh-Hadamard transform is O(d log d) butterflies, AVX2-friendly.
- **FP8 KV simulation** (:569–571): a cheap way to measure quality drift of a quantized cache *before* writing a quantized-cache kernel. Exactly the M5 "measure drift first" discipline.

**Applicability.**
- **CPU/M3–M8:** not needed. stories15M's context is 256 and M8 models are 2–32K. Top-k attention is still a nice post-v1 experiment on long contexts [inference].
- **GPU sm_89:** the TileLang kernels avoid TMA and warp specialization, so they *might* compile for Ada, but they use FP8 `T.gemm`, which runs into the FP8-toolchain issue found in §2.5 [unverified]. FlashMLA and DeepGEMM are SM90/SM100 only.
- **Server:** DSA changes the long-context cost curve, which a scheduler or cost model must know (per-token cost grows more slowly with position). That is role project 4's "quantitative model" territory.
- **Pitfall** [inference]: `Indexer.forward` calls `dist.broadcast` unconditionally (:485), but `generate.py` only initializes the process group when `WORLD_SIZE > 1` (`V32/generate.py:103–104`). A single-process run should therefore raise on the first indexer call. Unverified: not run, as it needs the weights.

---

## 3. Engineering practices

**Config handling** [code]:
- There are two sources of truth: a hand-written `configs/*.json` for the demo, and the HF `config.json` that ships with the weights (read only by the tokenizer and by `fp8_cast_bf16`).
- `ModelArgs(**json.load(f))` (`generate.py:113`) rejects unknown keys (good), but **missing keys silently take defaults**. The 671B config omits `mscale` (default 1.0) and `max_seq_len` (default 16384, which switches YaRN on).
- Runtime knobs (`max_batch_size`, `max_seq_len`) live in the same dataclass as architecture constants, and one of them changes the math (2.3.3).
- Magic-number dispatch: `Gate.bias` exists iff `dim == 7168` (`V3/model.py:564`, `V32/model.py:675`).
- **Lesson for M0/M2:** separate *architecture* (from the checkpoint) from *runtime limits* (from the CLI). Derive booleans such as "has router bias" from data, not from a dimension.

**Dtype handling** [code]:
- **Set globally:** `torch.set_default_dtype(bf16)` (`generate.py:109`) and `Linear.dtype = float8_e4m3fn` if `args.dtype == "fp8"` (`model.py:760`). Buffers such as the KV cache take the default (bf16).
- **Scales ride along as an attribute** on the weight Parameter (`self.weight.scale = self.scale`, :187), which is how a free function `linear()` finds them.
- **Precision islands grew from V3 to V3.2:** fp32 softmax in attention (`V3:490`), then in V3.2 also fp32 RMSNorm, SwiGLU, gate, MoE accumulation, the TP all-reduce, and logits (`V32:284–306`, :643, :687, :793, :264–266, :886/:908), plus fp32 sampling softmax (`V32/generate.py:26`).
- **Lesson for M5/M7:** quantize storage and GEMM inputs, but keep norms, softmax, reductions, residual sums and logits in f32. Luigi's f32 engine starts there, which is good. The discipline matters once quantized kernels arrive.

**What they test.**
- Nothing automated [code: no test files; §1].
- The only checks are the `__main__` shape smoke test, runtime `assert`s on shapes and divisibility, and V3.2's cross-rank top-k assert (`V32/model.py:486`).
- The indexer RoPE-layout bug shipped and was fixed ~7 weeks later (`V3.2-Exp/README.md:77`) [doc]. A reference-logit comparison at every position, i.e. Luigi's D1, would catch that class of bug immediately *if* a reference exists [inference]. This is the strongest argument for Luigi's M0 tolerance decision, and a good interview anecdote.

**Readability through Ousterhout's "deep module" lens** (see `design_considerations/ousterhout_research.md`) [inference]:
- **Deep:** `Transformer.forward(tokens, start_pos) -> logits` hides caches, masks, RoPE slicing, TP collectives and quantization behind two parameters. `linear()` hides three execution strategies behind `y = xWᵀ`. Both are exemplary interfaces.
- **Information leakage and hidden coupling:**
  - The `start_pos`/mask contract (multi-token calls only at 0) is undocumented.
  - Scales are smuggled on `weight.scale`.
  - Mutable module globals (`world_size`, `rank`, `gemm_impl`, `attn_impl`) and class attributes (`Linear.dtype`, `Linear.scale_fmt`) are set inside `Transformer.__init__` (:757–761). That is temporal coupling: modules read them at construction, and two models with different dtypes can't coexist in one process.
  - `BLOCK_SIZE_K` is reused as the weight block size along N (`kernel.py:154`).
  - The shard dims live in `convert.py`'s table while the matching split lives in the layer classes: two places must agree.
- **Shallow-ish:** `ColumnParallelLinear` in V3 only changes a shape at init. Its value is naming intent, which is fine.
- **Comments:** nearly every function has a docstring that restates the signature ("Forward pass for the MLP layer"), which Ousterhout calls repeating the code. The valuable comments are the rare *why* comments:
  - "we use fp8 kv cache in actual deployment, so here we simulate…" (`V32/model.py:569`)
  - "rope in indexer is not interleaved" (:463, :469)
  - "lm_head in the checkpoint is stored in bf16, while the parameter here is stored in fp32…" (:885)
  - "Promote to enable 2xAcc" (`V32/kernel.py:160`)

  Show Luigi these as examples of comments worth writing.
- **Tactical tells:** V3 re-dequantizing `wkv_b` every call (:481), the unreachable `"scale"` mapping entry, and a warm-up call on uninitialized weights (`generate.py:118`). V3.2 fixed the first (cached) and replaced the third with a print.

---

## 4. Applicability summary: CPU, GPU sm_89, server

| Technique | (a) Rust CPU (AVX2, 24 threads) | (b) CUDA C++ on sm_89 | (c) post-v1 server | Not applicable here |
|---|---|---|---|---|
| One forward for prefill + decode, `start_pos` | M3/M4 core API | same API | needs per-sequence positions | none |
| Causal mask only when `s > 1` | M3 (general `(s, start+s)` form) | M7 attention kernel | chunked prefill | DeepSeek's `(s,s)` form for `start_pos > 0` |
| MLA absorb / MQA-mode decode | not needed (stories15M, GQA models) | nice stretch kernel | KV-capacity model | FlashMLA (SM90/100) |
| YaRN | not needed (plain RoPE) | not needed | long-context models only | none |
| Precomputed RoPE table | M6 micro-opt | M7 (table in constant/global memory) | none | none |
| MoE Gate + aux-free bias | none | none | load-balancing analogy (role 3) | running any MoE in 15 GB RAM |
| FP8 block quant | software only (LUT); **use Q8_0** | FP8 MMA needs newer CUDA [measured]; **INT8 mma.sync works now** | FP8 KV cache | DeepGEMM (SM90/100), TMA, WGMMA |
| Per-group promotion `acc += dot·s_a·s_w` | M5 (= `runq.c:336`) | M7 quantized GEMM epilogue | none | none |
| Hadamard before quant | M5 4-bit option | M7 option | none | none |
| Exponential-race sampling | M4 correctness, M6 measure vs sort | M7 sampling kernel | batched and sharded sampling | none |
| Column/row TP | M6 thread splits | multi-GPU: n/a (1 GPU) | one engine per worker | NCCL, torchrun |
| Named, sharded checkpoints | M2 lessons, M8 loader | none | none | 671B-scale conversion |
| DSA (indexer + top-k) | post-v1 experiment only | post-v1 experiment | long-context cost model | FlashMLA/DeepGEMM kernels |

---

## 5. Milestone map: technique → milestone → how to use it when teaching

| Technique (section) | Milestone | How to use it |
|---|---|---|
| Decode AI: naive 1 FLOP/B vs absorb 242 FLOP/B (2.3.2) | **M0** perf model (D2) | After Luigi finishes the stories15M memory-bound decode estimate (60 MB ÷ bandwidth), show that DeepSeek redesigned attention *because* of that same arithmetic. Role project 4 |
| `forward(tokens, start_pos)` + where the cache lives (2.1, 2.2) | **M0** crate layout; **M3** | Present DeepSeek (cache in model), llama2.c (cache in RunState) and candle (`&mut Cache` param) as three data points. Luigi chooses. Log as a D-decision |
| Last-position vs all-position logits (2.1) | **M3/M4** API | Point out D1 needs every position, so his API must support it. Ask him to design how |
| Causal mask only when `s > 1`; `(s,s)` vs `(s,start+s)` (2.2) | **M3** | Draw-the-matrix exercise. Then the DeepSeek chunked-prefill failure [measured] |
| RoPE table + interleaved vs half-split (2.3.3) | **M3**, again at **M8** | The position-0-always-passes quiz; DeepSeek's 7-week indexer bug as the M8 warning |
| GQA ↔ MLA as KV-reduction family (2.3.2) | **M3** (`kv_mul`), **M8** | "Why does llama2.c index `h / kv_mul`?" Then at M8, bytes per token for TinyLlama vs stories15M |
| Explicit prefill then decode loop; static batching waste (2.6) | **M4** | Walk the `prev_pos/cur_pos` table. Define TTFT and decode tok/s on it |
| Exponential race sampler (2.6) | **M4** (correctness), **M6** (speed) | Proof in 3 lines. M6: measure sampling share vs `sample_topp`'s sort. Role project 1 |
| Precision islands (3) | **M1** (kernels in f32), **M5** | "Which ops stay f32 even when weights are int8?" |
| Named tensors, mapping table, scales travel with weights (2.8) | **M2**; **M8** | The swapped-w1/w3 quiz. mmap vs read decision context |
| Block-scaled GEMM = `runq.c` group matmul (2.5) | **M5** | Put `runq.c:332–337` next to `kernel.py:156–166`. Q8_0 vs FP8 worked example. Why Q8_0 on Zen 3 |
| Hadamard rotation before quant; FP8-KV simulation (2.9) | **M5** (4-bit) | Optional technique for 4-bit outliers; "simulate the quantized cache first" |
| Column/row parallel → threads (2.7) | **M6** | Row vs column split quiz. Megatron MLP-on-threads experiment. Vocab-split head (61% of MACs) |
| Hoist loop-invariant work (V3 `wkv_b` dequant per call) (2.3.2) | **M6** | "Find the work that doesn't change between decode steps" |
| INT8 `mma.sync` works, FP8 needs newer nvcc (2.5, §7) | **M7** | Decide the M7 low-precision kernel format with this constraint on the table |
| MQA-mode decode kernel (2.3.2) | **M7** stretch | One KV row loaded once, used by all heads |
| DP-attention, prefix reuse, continuous batching, KV bytes/token capacity (2.3, 2.6) | **post-v1** | The server's reasons to exist |
| Aux-loss-free bias as a load balancer (2.4) | **post-v1** (role 3) | Feedback-controller framing |
| TP group failure loses KV → requeue (2.7) | **post-v1** (role 5) | Fault model |
| DSA cost curve (2.9) | **post-v1** (role 4) | Cost vs position; an indexer that is still O(L) per token |

---

## 6. Quiz questions (with expected answers for the tutor)

**M0 (performance model)**
1. DeepSeek's naive MLA decode reads 81,920 bytes per cached token per layer and does 81,920 FLOPs. Is that compute-bound or memory-bound on any modern chip, and why? *(Memory-bound: 1 FLOP/B is far below any CPU or GPU ridge point.)*
2. Absorb does 3.4× more math per (query, key) pair. Why is it still faster for decode but slower for prefill? *(Decode: bytes dominate, 71× fewer. Prefill: many queries share each key, so it's compute-bound and the extra math costs.)*
3. For stories15M at position 255, how does KV-cache traffic compare with weight traffic per token? *(≈3.5 MB vs ≈61 MB, <6%, so a KV-compression trick would be premature.)*

**M1 (kernels)**
4. DeepSeek V3.2 computes RMSNorm and softmax in fp32 even with bf16/FP8 weights. Which of your M1 kernels need a wider accumulator if inputs ever become f16/int8, and why? *(RMSNorm's sum of squares, softmax's sum of exps, matmul's dot products: long sums lose low bits.)*

**M2 (loading)**
5. llama2.c's `.bin` has no tensor names. If `w1` and `w3` were swapped on disk, what would crash? *(Nothing. Logits are wrong. Only a reference comparison catches it.)*
6. In DeepSeek's `convert.py`, which dimension of `wo` is split across ranks, and why that one? *(dim 1 (input features): `wo` is row-parallel, consuming the per-head outputs that each rank already holds.)*

**M3 (forward + KV cache)**
7. Why does DeepSeek build a causal mask only when `seqlen > 1`? *(A single new query's keys are all in its past.)*
8. DeepSeek's mask is `(seqlen, seqlen)`. Give a call where that's wrong. *(Any multi-token call with `start_pos > 0`: chunked prefill, or feeding a second prompt chunk. The mask must be `(s, start_pos+s)`.)*
9. Your RoPE passes at position 0 and fails at position 5. What's the first thing you check? *(The `start_pos` offset into the angle table, or the pair layout. Position 0 is the identity rotation.)*
10. In llama2.c, `k` for head h is read at `(h / kv_mul) * head_size`. What does `kv_mul > 1` mean, and what's the extreme version of this idea in DeepSeek? *(GQA: query heads share KV heads. MLA decode is MQA with one 576-dim KV "head".)*

**M4 (tokenizer, sampling, generate)**
11. In DeepSeek's `generate`, prompts of lengths 3 and 5 are batched. How many tokens does the first `forward` call process per row, and how is row 1's 4th prompt token processed? *(3; then one token per step in "decode" with the prediction overwritten by the prompt token.)*
12. Prove that `argmax(p_i / E_i)` with `E_i ~ Exp(1)` samples token j with probability `p_j`. *(`E_i/p_i ~ Exp(p_i)`; the min of independent exponentials is index j with probability `λ_j/Σλ`.)*
13. Why can you skip the softmax entirely in that sampler? *(`argmax(logit/T − log E)`: the normalizer is a shared constant.)*
14. Where exactly do TTFT and decode tok/s start and stop in your loop?

**M5 (quantization)**
15. `runq.c:336` and DeepSeek `kernel.py:162` do the same thing. What is it, and why can't the scales be factored out of the whole dot product? *(Integer/FP8 dot per group, then rescale per group and accumulate in float. Scales differ per group along K.)*
16. Quantize `[0.9, −0.05, 0.3, 4.48]` with Q8_0-style int8 and with FP8 e4m3. Which values does each format hurt, and why? *(int8: small values, fixed absolute step. FP8: mid-size values like 0.9 → 0.88, 3-bit mantissa.)*
17. What happens if you cast 500.0 to `float8_e4m3fn` in PyTorch? What does that mean for your scale choice? *(NaN. Scale = amax/max and/or clamp before cast.)*
18. Why does Q8_0 make more sense than FP8 on a Zen 3 CPU? *(No FP8 hardware; AVX2 has an int8 multiply-add path; int8 is more accurate on bell-shaped blocks [measured on synthetic data].)*
19. Why might rotating vectors with a Hadamard matrix before quantizing reduce error without changing dot products? *(Orthogonal, so dots are preserved; it spreads outliers, so the per-group amax is lower.)*

**M6 (speed)**
20. Splitting a matvec across threads by output rows vs by input columns: which needs a reduction? Which does llama2.c use? *(Columns need one; llama2.c splits rows, `run.c:221`.)*
21. For stories15M, which single matmul should you parallelize first, and what fraction of MACs is it? *(The classifier: 32000×288 ≈ 61%.)*
22. DeepSeek V3 dequantized `wkv_b` on every forward call. What's the general optimization lesson? *(Hoist loop-invariant work out of the per-token loop.)*

**M7 (GPU)**
23. With nvcc 12.0 on sm_89, which low-precision tensor-core MMA can you use today? *(INT8 `mma.sync.m16n8k32 .s8.s8.s32`; FP8 fails in ptxas [measured].)*
24. How would you validate a GPU quantized GEMM against the CPU engine? *(The D1 method: per-position logits with a tolerance measured, not guessed.)*

**M8 (real model)**
25. Your M8 model's output is fluent for 3 tokens then degrades. RoPE matches llama2.c. What's the DeepSeek-inspired hypothesis? *(Interleaved vs half-split RoPE layout mismatch with HF weights.)*

**Post-v1 server**
26. DeepSeek's chat loop re-runs the whole conversation each turn. What server feature removes that cost, and what must the KV cache API support for it? *(Prefix caching: caches keyed by token prefix, shareable and forkable, which requires caches outside the model.)*
27. Under TP=16, why is the MLA latent cache no longer 71× smaller per GPU than the naive cache, and what deployment trick fixes it? *(The latent is replicated per rank, so 2,560 vs 576 = 4.4×; DP-attention.)*
28. A TP rank dies mid-request. What state is lost, and what must the router do? *(Every rank's KV slice for that request; requeue and redo prefill.)*

---

## 7. Open questions / unverified items

1. **FP8 `mma.sync` toolchain requirement.** [measured] nvcc 12.0 (PTX ISA 8.0) rejects `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` for sm_89. My recollection is that FP8 `mma.sync` needs PTX ISA 8.4 (CUDA 12.4+). **Unverified**: check the PTX ISA docs before M7. `cuda_fp8.h` (types and conversions) *is* present in `/usr/include`.
2. Whether Triton 3.0 (`V3/kernel.py`) or TileLang 0.1.6 (`V32/kernel.py`) FP8 kernels run on sm_89. Triton bundles its own ptxas, so it may accept FP8 MMA where system nvcc 12.0 does not. Not tested. The venv has Triton 3.7.0 (not the pinned 3.0.0) and no TileLang. A quick M7-prep experiment is to run a tiny Triton FP8 `tl.dot` on the 4070 to learn whether the *hardware* path works independently of the system toolkit.
3. V3.2 single-process run: `dist.broadcast` without an initialized process group (`V32/model.py:485` vs `V32/generate.py:103–104`). Expected to raise. Not run.
4. `safetensors.torch.load_model` strictness and dtype casting (e.g. bf16 `lm_head` into the fp32 `head` parameter in V3.2, per the comment at `V32/model.py:885`). Behavior inferred, not executed.
5. From DeepSeek papers not in `~/refs` [paper, not local]:
   - the aux-loss-free bias update rule;
   - node-limited routing ("at most 4 nodes");
   - FP8 accumulation-precision motivation for promotion;
   - 128×128 blocks being transpose-friendly for training;
   - V2's absorbing `W_UV` into `W_O`.

   Add the V2/V3 papers to `~/refs` if these are to be taught as facts.
6. `convert.py`'s `"scale": ("scale", None)` mapping entry looks unreachable (the key is always the module name). Inferred from string logic, not executed.
7. M8 model configs quoted from memory (TinyLlama: 22 layers, 4 KV heads × 64; Qwen2.5 rope base 1e6). Verify from their `config.json` at M8.
8. Figure 3 cost curves in the V3.2 PDF (p5) can only be read off the plot. No exact numbers quoted.
9. The FlashMLA HEAD (`2e5429f`, 2026-09-30) says it removed Hopper and V3/V3.2 support (`FlashMLA/README.md:3`). Kernels for V3.2 are at commit `ba89a34`. Irrelevant for sm_89, but matters if the tutor cites FlashMLA code lines.
10. The exponential-race vs top-p sort CPU cost comparison is a hypothesis to measure in M6, not a result.
11. The DSA decode-cost table uses the deployed FP8 layouts (FlashMLA 656 B; indexer fp8 + fp32 scale from `V32/model.py:453–454`). It ignores paging overhead, the top-k selection cost, and dtype throughput differences.

---

## 8. Verification log (what I ran)

All scratch work is in the session scratchpad (`…/scratchpad/ds01_verify.py`, `ds01_fp8mma.cu`, ephemeral). To reproduce, run with `~/refs/inference/venv/bin/python` (torch 2.12.0+cu130, CPU):
- **MLA equivalence:** random fp64 `q_nope, q_pe, c, k_pe, wkv_b` with H=4, C=16, DN=8, DR=4, DV=8, T=7. Naive (materialize K, V) vs absorb einsums as in `V3/model.py:471–495`. Max diff 3.6e-15 (scores) and 1.8e-15 (outputs).
- **Exponential race:** `p=[0.5,0.3,0.15,0.05]`, 400k draws of `(p / Exp(1)).argmax()` gives `[0.5003, 0.2995, 0.1501, 0.0501]`.
- **YaRN:** the same formulas as `V3/model.py:314–370` with d=64, base=1e4, L=4096, s=40, β=(32,1) give low/high = 10/23 and the ratios above. mscale² = 1.87385 (mscale 1.0) and 1.58963 (0.707).
- **FP8:** `torch.finfo(torch.float8_e4m3fn)` gives max 448, tiny 0.015625, eps 0.125. Casting `[448, 464, 500, 0.1, 2⁻⁹, 2⁻¹⁰]` gives `[448, 448, nan, 0.1015625, 0.001953125, 0]`.
- **Chunked-prefill mask:** `zeros(1,3,2,7) + full((3,3),-inf).triu(1).unsqueeze(1)` raises the size-mismatch error.
- **Synthetic quant error:** 4096² Gaussian with 8 columns ×20, relative RMSE for int8 1×32/1×128 and FP8 1×32/1×128/128×128 (values in 2.5).
- **Sizes:** KV-cache and DSA arithmetic from the configs; stories15M header read with `struct.unpack('7i')` = `(288, 768, 6, 6, 6, 32000, 256)`.
- **PTX:** `nvcc -arch=sm_89 -c` on a kernel with inline `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` fails (ptxas: "Unexpected instruction types specified for 'mma'"). The same file with only `...s32.s8.s8.s32` compiles. The PTX header says `.version 8.0`.
- **Hardware:** `lscpu` reports an AMD Ryzen 9 5900X, 12C/24T; `/proc/cpuinfo` has avx2, fma, f16c, no avx512 or avx_vnni. `nvidia-smi` reports an RTX 4070, compute 8.9, 12,282 MiB.
