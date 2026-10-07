# VRAM Hungry (`WeegeeInfer`)

From-scratch LLM inference in Rust: load a small transformer’s weights, run the forward pass ourselves (no ML framework), then make it fast. **CPU first** as the correctness oracle; **CUDA later** on an RTX 4070 (sm_89), with every GPU kernel checked against the CPU.

First model: **TinyStories 15M** in [llama2.c](https://github.com/karpathy/llama2.c) checkpoint format. Metrics we care about once generate works: **prefill vs decode**, TTFT, and decode tokens/sec — then int8/4-bit quant, AVX2/threads, and CUDA.

This repo is a **work in progress**. The engine is not built yet. What *is* here today is the plan, one locked correctness decision (**D1**), measurement scripts for that decision, and a Rust crate scaffold.

## Current status

| Piece | State |
| --- | --- |
| `WeegeeInfer` | Scaffold (`cargo run` → `Hello, world!`); empty deps |
| **D1** | Locked — f32 engine vs reference logit tolerance (see below) |
| Success criteria table | Planned in M0; measured columns empty until M4+ |
| Performance model (predicted TTFT / tok/s) | Not written yet (M0) |
| Crate layout (lib vs bin / workspace) | Still open (M0) |
| Research notes | `deepseek_research/`, `design_considerations/` — background for learning, **not** the product |
| Public docs | This README + `DECISIONS.md` |

**Not claimed:** working forward pass, CUDA kernels, quantization, or a serving stack.

## Architecture (planned)

```
checkpoint (llama2.c / TinyStories)
    → load weights
    → tensor ops (matmul, rmsnorm, softmax, silu, …)
    → forward (+ KV cache)
    → tokenize / sample
    → generate with explicit prefill vs decode
         │
         ├─ CPU (oracle, AVX2 target) ──► measured TTFT & tok/s
         └─ CUDA (cudarc + custom kernels on sm_89)
              every kernel checked vs CPU under a measured tolerance
```

v1.0 done-condition (from the plan): `generate "Once upon a time"` is coherent from a real checkpoint; logits match a reference within tolerance; a reproducible benchmark table (f32 vs int8 vs 4-bit, CPU vs GPU) next to llama2.c on the same machine.

## Decision D1 — correctness tolerance

Engines that are both correct still disagree in the last digits of f32. D1 draws a line between **rounding noise** and **a bug**, measured before the engine exists — we do not loosen it later to make a test pass.

For the **f32 CPU engine vs reference**, check **every position’s logits** (not only the last):

`|ours − ref| ≤ atol + rtol · |ref|` with **atol = 5e-4**, **rtol = 0**.

Buckets: **PASS** / **PASS near-tie** / **FAIL**. A near-tie is a flipped winner when the reference’s top-2 scores were &lt; `2 × atol` apart (rounding alone could explain the flip).

Measured on TinyStories stories15M (~164M logits: 20 prompts × 256 positions × 32k vocab):

| Signal | Worst \|diff\| |
| --- | --- |
| Naive f32 engines vs each other (`run.c -O3` vs PyTorch f32) | ~5.5e-5 |
| Smallest planted bug (wrong RMSNorm ε) | ~0.28 |

`atol = 5e-4` sits ~10× above that noise and far below planted bugs. Quant (M5) and GPU (M7) get their **own** tolerances, measured the same way.

Full write-up: `DECISIONS.md`, `test_ref/tolerance.md`. Scripts: `scripts/tolerance/` (`refs.py`, `noise_pytorch.py`, `noise_runc.py`, `planted_bugs.py`, `dump.c`). Those scripts expect local clones/checkpoints under `~/refs/inference/` (not shipped in this repo).

## Success criteria (M0 — fill as we measure)

| Metric | Target | Predicted | Measured |
| --- | --- | --- | --- |
| Logit match vs reference (f32) | D1 (`atol=5e-4`) | — | — |
| TTFT (prefill) | TBD in M0 | — | — |
| Decode tok/s (f32, 1 thread) | TBD in M0 | — | — |
| Peak memory | TBD in M0 | — | — |

## Roadmap

| # | Milestone | Checkpoint evidence |
| --- | --- | --- |
| **M0** | Plan + criteria + crate layout + this README | `cargo build` / `cargo test`; criteria table with predictions |
| **M1** | Tensors + math kernels | Unit tests vs hand-computed values |
| **M2** | Load checkpoint | Config + weight values match Python reference |
| **M3** | Forward + KV cache | Logits vs reference under D1 |
| **M4** | Tokenizer, sampling, generate | Real text; baseline TTFT + decode tok/s (f32, 1 thread) |

Later (only after M4 matches D1): **M5** quant, **M6** CPU speed (profile first), **M7** CUDA on sm_89 (int8 `mma` path is realistic today; many H100/Blackwell kernels are **idea sources only** on a 4070), **M8** larger open model, **M9** write-up. Post-v1 serving ideas stay out of scope until the engine is real.

## Hardware note

Dev machine in the plan: WSL2 Ubuntu, Ryzen-class CPU (AVX2 + FMA, no AVX-512), RTX **4070 12 GB (sm_89)**. DeepSeek’s fastest published kernels (DeepGEMM / FlashMLA / etc.) generally need newer GPUs — useful for ideas, not as drop-in deps here.

## Build / run (what exists today)

Needs a Rust toolchain that supports edition **2024** (see `WeegeeInfer/Cargo.toml`).

```bash
cd WeegeeInfer
cargo run
# → Hello, world!
```

Tolerance re-runs need llama2.c + TinyStories checkpoints under `~/refs/inference/` — see `test_ref/tolerance.md`.

## Layout

```
WeegeeInfer/              # Rust crate (scaffold)
README.md                 # This file
DECISIONS.md              # Locked decisions (D1, …)
CLAUDE.md                 # Full working plan / milestone notes
deepseek_research/        # Tutor research notes (not product docs)
design_considerations/    # Design-principle notes
scripts/tolerance/        # D1 measurement scripts
test_ref/tolerance.md     # D1 experiment write-up
```

## License

TBD.
