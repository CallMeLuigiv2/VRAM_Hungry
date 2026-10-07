# inference-engine: a CPU LLM inference engine in Rust

AI-infra project #5 on the list (quantized CPU runtime). Luigi chose on 2026-09-24 to do it **first**, and **in Rust** (the list said C++). The folder name is a placeholder; Luigi can rename it.

## What it is
A program that loads a small transformer language model's weights from disk, runs the forward pass on the CPU, and generates text one token at a time. Then it makes that fast: a KV cache, int8/4-bit quantized weights, AVX2 SIMD, multithreading. No ML framework: every piece of math is ours.

Machine: WSL2 Ubuntu, 24 threads, AVX2 + FMA (no AVX-512), 15 GB RAM, **NVIDIA RTX 4070 12 GB** (compute 8.9, visible in WSL, `nvcc` at `/usr/bin/nvcc`). Rust 1.96 in `~/.cargo/bin`.

**Plan (Luigi, 2026-09-24): CPU first, GPU as a later milestone.** The CPU engine is built first and becomes the test oracle: every GPU kernel is checked against its CPU result. Real-world precedent: llama.cpp (CPU first, CUDA backend added later), candle (`Device::Cpu | Cuda`, CPU is the reference). GPU path = Rust host (`cudarc`, as candle does) + CUDA C kernels.

**Model (Luigi, 2026-09-24): start small, scale later.** Build and debug on TinyStories 15M (llama2.c checkpoint format, Python reference in `model.py`), then scale to a real open model in M8.

**End goal (Luigi, 2026-09-24):** demonstrate that he understands inference serving: **prefill vs decode** as separate phases with their own metrics (TTFT = time to first token; decode tokens/sec), and the vLLM-style ideas that grow from them. Note llama2.c does NOT separate them (`generate` feeds the prompt through `forward` one token at a time), so this is a deliberate difference from the reference.

**Done-condition (v1.0):** `generate "Once upon a time"` produces coherent text from a real checkpoint; our logits match a reference implementation within a stated tolerance; a benchmark table (tokens/sec, memory, quality drift) for f32 vs int8 vs 4-bit, CPU vs GPU, next to llama2.c on the same model and machine, reproducible from one command.

**Target role (Luigi, 2026-09-25):** the project is aimed at Anthropic's **Performance Engineer** role. Its representative projects, and where each one lives in this plan:

| Role project | Where it lives |
|---|---|
| 1. Low-latency, high-throughput sampling | M4 makes it correct; M6/M7 make it fast (measure sampling's share of per-token time: llama2.c sorts all 32k probs for every top-p step; GPU sampling kernel); batched sampling in the server |
| 2. GPU kernels for low-precision inference | M5 + M7: quantized CUDA matmul, validated against the CPU with a measured tolerance (the D1 method) |
| 3. Custom load-balancing algorithm for serving efficiency | post-v1 server: route requests across CPU and GPU workers, aware of prefill vs decode |
| 4. Quantitative models of system performance | starts in M0: predict TTFT and decode tok/s from hardware numbers before building, then measure against the prediction (M4, M6, M7) |
| 5. Fault-tolerant distributed system | post-v1 server: several engine workers; a worker dies mid-request → its KV cache is lost → requeue and redo prefill |
| 6. Kernel-level latency spikes in containers ("kernel" = Linux kernel / network stack, not GPU kernels) | post-v1 server in containers: track p99 latency, hunt spikes with `perf` |

2 and 4 are the heart of v1; 1 needs speed work, not just correctness; 3, 5, 6 belong to the server this engine becomes. When a milestone touches one of these, point out the link and make its evidence role-relevant.

## How we work: pair programming, milestone → folder → file → function
Luigi's request (2026-09-24): pair program, **going function by function, file by file, folder by folder**, organised in **milestones like CodePath AI201**, so the project is well documented and he understands the architecture.

Every component goes through the 5-step loop (Luigi's contract, 2026-09-23):

1. **Explain.** Claude explains how it works using real projects: llama2.c, candle, llama.cpp/ggml (file:line), the papers, and why they chose what they chose. Lead with a small worked example, not his code.
2. **Luigi designs it in his own words:** the module, struct, function signature, data layout. This is what makes the engine his and not a llama2.c port.
3. **Pros/cons + fine-tune.** Claude says what works, what hurts later (perf, memory, borrow-checker friction, API), and what the real projects do differently. Luigi decides. Log it in `DECISIONS.md` (D1, D2, ...: choice, alternatives, why).
4. **Express it in code.** Claude shows the Rust idiom on a small parallel toy, a snippet from a real project, or pseudocode mapped to his words. **Luigi types the repo code.** Claude reviews it afterwards. If he asks Claude to write repo code outright, confirm first.
5. **Explain-back / quiz.** Claude checks understanding before moving to the next function. Luigi's Rust questions always get a question back before the next hint.

Pacing: **one decision at a time.** Present one, elaborate, wait until Luigi says it's clear, then the next. Never dump several options in one message. No deadline: the metric is programming and learning every day, not speed.

Order of work inside a milestone: folder (what lives here and why) → file (its one job) → function (signature first, then a test, then the body).

## Milestones (CodePath-style)
Each milestone has a goal, deliverables, and a **checkpoint**: evidence pasted into the README (real command output, not a description of it). A milestone is closed when the checkpoint holds and Luigi can explain the architecture it added.

| # | Milestone | Deliverables | Checkpoint (evidence) |
|---|---|---|---|
| M0 | **Plan + success criteria** | Luigi picks the model/checkpoint and writes the success criteria with targets (like AI201 M1: correctness tolerance, tok/s, memory). A paper performance model: predicted TTFT and decode tok/s from hardware numbers (role project 4). He designs the crate layout (single crate vs workspace, lib vs bin). `cargo` project created by him, README + DECISIONS.md started. | `cargo build` + `cargo test` pass; criteria table in README with predicted values and empty measured columns |
| M1 | **Tensors + math kernels** | The tensor/buffer type (his design; keep a later GPU device in mind), naive matmul, rmsnorm, softmax, silu. Unit tests against hand-computed values. | test output; one kernel explained in README |
| M2 | **Loading weights** | Read the checkpoint header + weights (mmap vs read = a decision). Config struct, weight layout. | prints config + a few weight values that match the Python reference |
| M3 | **The forward pass** | One transformer block → all layers: embedding, RoPE, attention, SwiGLU FFN, final logits. KV cache. | our logits vs reference logits for a fixed prompt, max abs diff under the M0 tolerance |
| M4 | **Tokenizer + sampling + generate loop** | Encode/decode, greedy + temperature + top-p sampling, CLI. Generate loop with an explicit **prefill phase** (prompt) and **decode phase** (one token per step). | real generated text; baseline TTFT + decode tokens/sec (f32, 1 thread) |
| M5 | **Quantization** | int8 per-group (Q8_0-style) first, then a 4-bit format. Quantize/dequantize + quantized matmul. | quality drift (logit diff or perplexity) + memory + tok/s vs f32 |
| M6 | **Make it fast** | Profile first (perf / flamegraph), then AVX2 via `std::arch`, multithreading, memory layout. **Batched prefill**: all prompt tokens through a layer at once (matrix × matrix instead of matrix × vector). Every change measured. | before/after benchmark table, the command that produced it, comparison with llama2.c `runq` |
| M7 | **GPU backend (CUDA)** | Rust host via `cudarc`, CUDA C kernels: matmul, rmsnorm, softmax, RoPE, attention, then quantized matmul. Device abstraction so the model code runs on either backend. | every GPU kernel matches its CPU result within tolerance; tok/s CPU vs GPU table; `nsys`/`ncu` profile of one token |
| M8 | **Real model** | A real open model (TinyLlama-1.1B or Qwen-0.5B class; GGUF vs safetensors = a decision then), its tokenizer, on CPU and GPU. | real answers to real prompts; TTFT + decode tok/s + memory, CPU vs GPU, next to llama.cpp |
| M9 | **Write-up (v1.0)** | Final README, ARCHITECTURE.md (data flow diagram in his words), benchmark table, AI-usage reflection (as in AI201 M5), git tag v1.0. | Luigi explains the whole pipeline end to end without notes |

After v1.0: this engine becomes the core of project #1, a vLLM-style server (continuous batching, paged KV cache, prefill/decode scheduling), plus the role projects that need a server: a custom load balancer across CPU/GPU workers (3), worker-failure handling with requeue (5), and containerized deployment with p99 latency tracking (6).

## Documentation (Luigi writes it, in his words)
- `README.md`: CodePath-style sections filled per milestone: what it is, success criteria table, architecture, evidence per checkpoint, benchmark table, AI-usage reflection.
- `DECISIONS.md`: every design decision (D1, D2, ...).
- `ARCHITECTURE.md`: grows each milestone; a module map and the path one token takes through the engine.
- Claude reviews these docs for accuracy and clarity; it does not write them for him.

## References (explain from these)
All in WSL `~/refs/inference/`:
- **llama2.c** (Karpathy): the smallest complete version. `run.c` is 973 lines: `rmsnorm` :182, `softmax` :197, `matmul` :217, `forward` :231, tokenizer `encode` :452, `sample_topp` :624, `generate` :729. `runq.c` = the int8 version (`quantize` :145, quantized `matmul` :317). `model.py` + `export.py` = the Python reference and checkpoint format. TinyStories checkpoints (15M/42M/110M) are small enough to iterate fast.
- **candle** (Hugging Face, Rust): how a real Rust ML library does it. `candle-transformers/src/models/llama2_c.rs` and `quantized_llama.rs`; `candle-core/src/quantized/` (`k_quants.rs`, `avx.rs`, `gguf_file.rs`).
- **llama.cpp / ggml**: the industry reference for quant formats (Q8_0, Q4_0, K-quants) and GGUF. Clone into `~/refs/inference` when M5 starts.
- Papers: "Attention Is All You Need" (2017); LLaMA (Touvron et al. 2023); RoFormer / RoPE (Su et al. 2021); RMSNorm (Zhang & Sennrich 2019); LLM.int8() (Dettmers et al. 2022).

## Rules
- Correctness before speed. The Python reference (llama2.c `model.py` or PyTorch) is the ground truth; every kernel gets a test.
- Measure, don't claim. Every perf claim comes with a number and the command that produced it (`cargo build --release`, fixed prompt, fixed seed).
- `cargo clippy` and `cargo fmt` clean before a commit. `unsafe` only where needed (mmap, SIMD), each block with a `// SAFETY:` comment saying why it is sound.
- Commit at the end of every function that passes its test. Small commits, his messages.
- Don't pre-scaffold. Claude creates nothing beyond what Luigi has designed.
