# 05: DeepSeek's inference serving system (tutor reference)

Private reference for the tutor. Not Luigi's documentation. It covers how DeepSeek serves V3/R1, and what that means for M4 (prefill/decode split), M6/M7 (overlap, profiling), and the post-v1 server (role projects 3, 4 and 5).

- **Written:** 2026-09-30.
- **Repo root:** `~/refs/inference/deepseek/`. All paths below are relative to it unless they start with `~`.
- **Source tags:**
  - `[doc]`: README or design doc.
  - `[code]`: source code.
  - `[fig]`: an image in the repo.
  - `[trace]`: a number I measured myself from `profile-data/*.json` with a Python script (method in §4.4).
  - `[inference]`: my own math or reasoning. It is not in any repo.
- **Line numbers** were checked against the files at the commits below.

| Repo | Commit | Commit date |
|---|---|---|
| open-infra-index | `56d86855fcf6e08fdfd45ce6280bd24322c93351` | 2025-05-15 |
| profile-data | `449602428a1b023acb8a505d4f34fef536535db6` | 2025-03-21 |
| EPLB | `d52c72d5b2f2fb4c41afbf8eb21366820239913d` | 2025-03-24 |
| LPLB | `0490f79452f7ef277e814449600b1b1dd4c663b3` | 2025-11-19 |
| DeepEP | `93eb6eb238127e96c6d7a4a625a6dad158348509` | 2026-09-30 (V2.5; shallow clone, 1 commit) |
| DeepSpec | `005e03b81cec38b7da6399833d609ee89a2587f2` | 2026-07-09 |
| DualPipe | `030ce4325f4ebeb437da4ebc6d00a70469dd58ae` | 2026-01-14 |
| 3FS | `22fca04564c7cc230fd8b9523b8b92864e1dad47` | 2026-05-07 |
| DeepSeek-V3 (config and MTP facts only) | `9b4e9788e4a3a731f7567338ed15d3ec549ce03b` | 2025-08-28 |
| vLLM (cross-refs only), `~/refs/inference/vllm` | `22bbe3f1023a68a7d1f2de566dd4242fdcfd36c3` | 2026-09-25 |

Abbreviations:
- **D6** = `open-infra-index/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md`
- **OII** = `open-infra-index/README.md`
- **PDR** = `profile-data/README.md`

## Summary (the 10 things worth knowing)

1. **Prefill and decode run on different machines, with different parallelism.**
   - Prefill: 4-node units, routed experts EP32, attention/shared expert DP32. Each GPU holds 9 routed + 1 shared expert.
   - Decode: 18-node units, EP144/DP144. Each GPU holds 2 routed + 1 shared expert.
   - Both phases add 32 redundant experts [doc D6:21-23].
   - For DeepSeek, the reason to split is mostly *different parallelism per phase*, not throughput. vLLM's docs say plainly that PD disaggregation "DOES NOT improve throughput" (it helps latency control) [doc `~/refs/inference/vllm/docs/features/disagg_prefill.md:12-16`].
2. **Expert parallelism exists to make the per-expert batch big.** Only 8 of 256 experts fire per token [doc D6:19], so each expert sees roughly batch × 8/256 tokens.
   - Decode at EP144: about 512 tokens per physical expert per micro-batch, so the GEMMs are compute-efficient.
   - The same batch inside one 8-GPU node: about 32 tokens per expert, so the GEMMs are memory-bound. [inference, §3.3]
3. **Overlap: split every batch into two micro-batches**, so one micro-batch's all-to-all runs under the other's compute.
   - Prefill gives 24 of 132 SMs to communication kernels [fig D6 prefill figure; trace name `...-sm24.json`].
   - Decode uses **0 SMs** for communication: RDMA is issued, then the SMs are freed [doc PDR:30]. It also splits attention into two parts to form a 5-stage pipeline [doc D6:32].
4. **The traces show this directly** [trace]:
   - Prefill: 91% of communication-kernel time runs at the same time as compute.
   - Decode: everything is on one stream, and communication appears only as short send/recv kernels.
   - The decode trace also appears to contain an **MTP speculative layer** after sampling (§7.5).
5. **Three load balancers, each with its own objective** [doc D6:41-54]:
   - Prefill LB: attention compute + input tokens per GPU.
   - Decode LB: KV cache usage + request count per GPU.
   - EP LB: the maximum expert load on any GPU.
   - With synchronous all-to-all, step time = the **max** over GPUs, so every objective is min-max.
6. **EPLB is about 160 lines of greedy code**, with two policies:
   - Hierarchical: pack expert groups onto nodes (LPT greedy), then replicate hot experts within each node (max-per-replica greedy), then pack replicas onto GPUs (LPT again).
   - Global: the same steps with a single group and a single node.
   - DeepSeek's configs fit the README's advice: 8 groups over 4 prefill nodes gives hierarchical, and 18 decode nodes gives global [inference].
7. **LPLB adds a per-batch LP**: minimize the max GPU load by moving token fractions along original→replica edges.
   - It is solved with 5 affine-scaling interior-point iterations on one SM per group.
   - It is "early research stage" [doc `LPLB/README.md:5`]. Its limits are listed in §5.5.
8. **The published numbers are internally consistent** [inference, §6]:
   - 608B input / 73.7k per node gives about 95.5 prefill nodes. 168B output / 14.8k per node gives about 131.4 decode nodes. The sum, about 226.9, matches the published 226.75 average node count.
   - Cost and revenue can be reproduced to the dollar.
9. **Prefix KV cache on disk (3FS)**: 56.3% of input tokens hit it [doc D6:74]. Clients peak at 40 GiB/s per node [doc OII:82, `3FS/README.md:48`].
   - Reading a cached token's KV (~69 KiB) costs about 1/17 of recomputing it [inference, §6.5].
   - This is the economics behind cheaper cached-input pricing.
10. **Fault tolerance is barely documented for inference.**
    - The docs give elastic day/night scaling [doc D6:65-67], an optional external KV store [fig D6 diagram], and 3FS's lease/heartbeat, chain-version and balanced-recovery designs [doc `3FS/docs/design_notes.md:128-147,167,177-179`].
    - Role project 5 has to be designed from these principles (§9).

---

## 1. Map of the repos

| Repo | What it is | Key files (this note) | Serving relevance |
|---|---|---|---|
| open-infra-index | Index of the Feb-2025 Open Source Week, plus the Day 6 inference system overview and the Apr-2025 "path to open-sourcing the inference engine" | `README.md` (days 1-6 summaries, lines 32-105); D6 (88 lines, 5 figures); `OpenSourcing_DeepSeek_Inference_Engine/README.md` | **Core.** Architecture, parallelism degrees, LB objectives, production statistics |
| profile-data | PyTorch-profiler Chrome traces: `train.json` (3.1 MB), `prefill.json` (17.5 MB), `decode.json` (4.7 MB), plus overlap diagrams | `README.md` (30 lines), `assets/*.jpg` | Evidence of overlap; the model for Luigi's M7 nsys work |
| EPLB | Expert-parallel load balancer: replication + placement | `eplb.py` (164 lines), `README.md`, `example.png` | Role project 3: bin packing, replication of hot items, hierarchy |
| LPLB | LP-based per-batch token redistribution over redundant experts (research) | `README.md`, `lplb/planner.py`, `lplb/resources/csrc-tmpl/minilp.cu` | Dynamic balancing, min-max LP, hierarchical histogram allreduce |
| DeepEP | EP all-to-all communication library. **The checkout is V2.5**: "Fully remove V1, including its APIs, NVSHMEM backend, and legacy documentation" [doc `DeepEP/README.md:18`] | `README.md` (533 lines), `figures/low-latency.png`, `figures/normal.png` | Normal vs low-latency kernels, hook-style overlap, NVLink/RDMA |
| DeepSpec | Training/eval of draft models for speculative decoding (DSpark, DFlash, Eagle3) on Qwen3/Gemma targets | `README.md`, `deepspec/eval/base_evaluator.py`, `deepspec/utils/sampling.py`, `config/*` | Speculative decode loop, acceptance metrics, KV rollback |
| DualPipe | Bidirectional pipeline parallelism for **training** | `README.md` | Pipeline bubbles and chunked pipelines (interview Q2) |
| 3FS | Distributed file system (SSD + RDMA), CRAQ replication | `README.md`, `docs/design_notes.md` | On-disk KV cache (prefix caching), failure detection, recovery balancing |

Not covered here, even though they sit in the same folder: DeepGEMM, FlashMLA, DeepSeek-V3.2-Exp, DeepJIT, DeepSelect, TileKernels, deepseek-recipe. The only exceptions are the day summaries in OII:32-63.

Two facts from the index that the tutor will reuse:
- **FlashMLA** (Day 1): "Paged KV cache (block size 64)"; "3000 GB/s memory-bound | BF16 580 TFLOPS compute-bound on H800" [doc OII:39-40]. These are the H800 "achieved roofline" numbers for §6.
- **DeepGEMM** (Day 3): "Up to 1350+ FP8 TFLOPS on Hopper GPUs" [doc OII:59].

DeepSeek's production engine "is based on an early fork of vLLM from over a year ago" [doc `open-infra-index/OpenSourcing_DeepSeek_Inference_Engine/README.md:12-13,20-21`]. So vLLM concepts (continuous batching, paged KV) sit under everything in D6, even where D6 doesn't name them.

---

## 2. The system end to end (Day 6)

### 2.1 Design principles, quoted

- Objectives: "**higher throughput and lower latency**" [doc D6:3].
- The tool is cross-node EP [doc D6:5]:
  - "EP significantly scales the batch size, enhancing GPU matrix computation efficiency and boosting throughput" [D6:6].
  - "each GPU processing only a small subset of experts (reducing memory access demands), thereby lowering latency" [D6:7].
- The costs of EP [D6:9-11]:
  1. Cross-node communication, which needs overlap.
  2. Multiple nodes imply DP, which needs load balancing between DP instances.
- Why EP has to be large: "only 8 out of 256 experts per layer are activated ... necessitates an extremely large overall batch size" [D6:19].

### 2.2 Architecture diagram (ASCII; faithful to the D6 figure, with my annotations)

The figure is `open-infra-index/202502OpenSourceWeek/figures/Diagram of DeepSeek's Online Inference System.jpg`. Its boxes and arrows are:
- API Server → Prefill LB → stacked Prefill Service boxes.
- Prefill Service → Decode LB → stacked Decode Service boxes.
- Decode Service → API Server.
- Each service box contains an "Expert-Parallel Load Balancer".
- Prefill ↔ External KVCache Storage (Optional): arrows go both ways.
- Decode → storage: a single arrow down.

Everything marked `[inf]` below is my annotation.

```
                         +------------------+
    web / app / API ---->|    API Server    |<------------------ generated tokens ---------------+
                         +--------+---------+                                                    |
                                  | request (prompt)                                             |
                       +----------v-----------+   objectives [D6:41-45]:                         |
                       | Prefill Load Balancer|   - core-attention compute per GPU               |
                       +----------+-----------+   - input tokens per GPU (dispatch send)         |
               +------------------+-------------------+                                          |
     +---------v----------+  +----v-------+     ~24 units on average [inf, §6.2]                 |
     | PREFILL unit       |  | prefill    | ...                                                  |
     | 4 nodes x 8 H800   |  | unit       |                                                      |
     | routed experts EP32|  +------------+                                                      |
     | MLA+shared DP32    |<====== read prefix KV (56.3% of input tokens hit) =====+             |
     | 9 routed+1 shared  |======= write KV ======================================+|             |
     |  expert / GPU      |                                                        ||            |
     | [EPLB: hierarchical|                                        +---------------v+-------+    |
     |  policy inf]       |                                        | External KVCache       |    |
     | dual micro-batch,  |                                        | Storage (optional)     |    |
     | 24 SMs for comm    |                                        | = 3FS on NVMe [inf]    |    |
     +---------+----------+                                        +-----------^------------+    |
               | first token + KV handoff (mechanism NOT documented)           |                 |
     +---------v-----------+   objectives [D6:46-50]:                          | write KV        |
     | Decode Load Balancer|   - KV cache usage per GPU                        | (arrow in fig)  |
     +---------+-----------+   - request count per GPU                         |                 |
     +---------v----------+                                                    |                 |
     | DECODE unit        |----------------------------------------------------+                 |
     | 18 nodes x 8 H800  |                                                                      |
     | EP144 / DP144      |   ~7.3 units on average [inf, §6.2]                                  |
     | 2 routed+1 shared  |----------------------------------------------------------------------+
     |  expert / GPU      |
     | [EPLB: global      |
     |  policy inf]       |
     | 5-stage overlap,   |
     | 0 SMs for comm     |
     +--------------------+
```

One MoE layer inside a unit, on GPU *i*, which owns a DP shard of requests:

```
 tokens of my requests --> [RMSNorm, MLA attention on MY KV cache (DP: no comm)]
                        --> [shared expert (every GPU has a copy)]
                        --> [gate: pick top-8 of 256 experts, group-limited]
                        --> DISPATCH all-to-all: send each token's hidden (FP8) to the GPUs owning its experts
                        --> [grouped GEMM over tokens received for MY routed experts]
                        --> COMBINE all-to-all: return expert outputs (BF16), weighted sum
                        --> residual add --> next layer
```

Precision is stated in the doc: "matrix multiplications and dispatch transmissions adopt the FP8 format aligned with training, while core MLA computations and combine transmissions use the BF16 format" [doc D6:62-63].

### 2.3 Parallelism table

| | Prefill | Decode | Source |
|---|---|---|---|
| Unit size | 4 nodes (32 GPUs) | 18 nodes (144 GPUs) | [doc D6:22-23] |
| Routed experts | EP32 | EP144 | [doc D6:22-23] |
| MLA + shared expert | DP32 | DP144 | [doc D6:22-23] |
| Redundant experts | 32 | 32 | [doc D6:22-23] |
| Routed experts per GPU | 9 = (256+32)/32 | 2 = (256+32)/144 | [doc D6:22-23]; arithmetic [inference] |
| Profile config | EP32, TP1, 4K prompt, **16K tokens/GPU**, 2 micro-batches | EP**128**, TP1, 4K prompt, **128 requests/GPU**, 2 micro-batches | [doc PDR:22,30] |
| Comm SMs | 24 of 132 | 0 | [fig D6 figures; doc PDR:30; trace name `dsv3-600B-tp1-ep32-input4096-output1-bs16384-split1-sm24.json`] |
| EPLB policy (inferred) | hierarchical (8 groups ÷ 4 nodes) | global (8 groups not ÷ 18 nodes) | [code `EPLB/eplb.py:150-156`] + `n_expert_groups: 8` [code `DeepSeek-V3/inference/configs/config_671B.json:12`]; match with the EPLB README advice [doc `EPLB/README.md:24-25,29-31`] [inference] |

Model facts used later, all from `DeepSeek-V3/inference/configs/config_671B.json`:
- `dim` 7168 (:3), `moe_inter_dim` 2048 (:5), `n_layers` 61 (:6), `n_dense_layers` 3 (:7), `n_heads` 128 (:8).
- 256 routed experts, 1 shared, 8 activated (:9-11); 8 expert groups with 4 limited groups (:12-13).
- `q_lora_rank` 1536 (:16), `kv_lora_rank` 512 (:17), `qk_rope_head_dim` 64 (:19).
- 671B total / 37B activated [doc `DeepSeek-V3/README.md:47`].
- One MTP module (layer 61) [doc `DeepSeek-V3/README_WEIGHTS.md:36,47`].

### 2.4 Production statistics (24 h, 2025-02-27 12:00 to 02-28 12:00, UTC+8)

| Quantity | Value | Source |
|---|---|---|
| Hardware | H800, 8 per node | [doc D6:61,67] |
| Peak nodes (V3 + R1) | 278 | [doc D6:67] |
| Average nodes | 226.75 | [doc D6:67] |
| Assumed lease price | $2 / GPU-hour | [doc D6:68] |
| Daily cost | $87,072 | [doc D6:68] |
| Input tokens | 608B, of which 342B (56.3%) hit the on-disk KV cache | [doc D6:74] |
| Output tokens | 168B | [doc D6:75] |
| Per-request output speed | 20–22 tok/s on average | [doc D6:75] |
| Average KV length per output token | 4,989 tokens | [doc D6:75] |
| Prefill throughput | ~73.7k tok/s per node, **including cache hits** | [doc D6:76], also OII:97 |
| Decode throughput | ~14.8k tok/s per node | [doc D6:76] |
| Theoretical revenue at R1 prices | $562,027/day, "cost profit margin of 545%" | [doc D6:78] |
| R1 prices | $0.14/M input (hit), $0.55/M input (miss), $2.19/M output | [doc D6:80] |
| Why real revenue is lower | V3 is cheaper; web/app are free; night discounts | [doc D6:82-85] |

**TTFT is not published anywhere in these repos. Neither is p99 latency.** Only the mean output speed is given. TPOT ≈ 1/21 s ≈ 45–50 ms [inference].

### 2.5 Elastic scaling (a planned form of "fault tolerance")

"deploy inference services across all nodes during peak daytime hours. During low-load nighttime periods, we reduce inference nodes and allocate resources to research and training" [doc D6:65-66].

The node-count chart [fig `.../H800 Node Count For Inference Service.jpg`] shows:
- 278 nodes from 12:00 to 00:00.
- Then stepping down to about 68 nodes, flat from roughly 03:30 to 06:30.
- Then stepping back up to 278 by about 08:30.
- The steps, read by eye, are about 30 nodes each, with a first drop of about 60.

**[inference, speculative: values read off a chart]** One 30-node step = 1 decode unit (18) + 3 prefill units (12).
- Then 68 = 2 decode units + 8 prefill units.
- And 278 = 9 decode + 29 prefill units.
- The peak prefill share would be 116/278 = 0.42. That matches the 0.42 prefill share derived independently from the throughput numbers in §6.2.

This is a nice role-project-4 exercise ("decompose the chart into deployment units"). Present it as a hypothesis, never as fact.

**Teaching hook:** "scale down at night" means draining whole deployment units. A unit can't be half-removed, because EP144 needs all 144 GPUs. This is the same constraint that makes a worker death in an EP unit so expensive: the whole unit stalls (§9).

---

## 3. Why disaggregate prefill and decode (first principles, with reusable numbers)

### 3.1 The one formula

For a weight matrix of size `d_in × d_out` stored at `b` bytes per weight, and `T` tokens multiplied through it in one pass:

- FLOPs = `2 · T · d_in · d_out`
- Weight bytes read = `b · d_in · d_out` (read once per pass, if the pass is batched)
- Arithmetic intensity (AI) = FLOPs / bytes = `2T / b` FLOP per byte

A machine with peak compute `P` and bandwidth `BW` has a ridge point `P / BW`:
- Below the ridge, the pass is memory-bound.
- Above it, compute-bound.

Where each phase lands:
- **Decode:** T = batch size (1 in M4).
- **Prefill:** T = prompt length × batch.

That single `T` is the whole story [inference; standard roofline reasoning].

### 3.2 At Luigi's scale (stories15M, his D2 numbers)

The header of `~/refs/inference/models/stories15M.bin` reads (288, 768, 6, 6, 6, 32000, 256). The file is 60,816,028 bytes, so about 60 MB of f32 weights.

- **Decode, batch 1:**
  - About 30 MFLOP per token (2 × 15M) against 60 MB read.
  - AI = 0.5 FLOP/byte.
  - Luigi's D2 memory ceiling is 60 MB ÷ 50 GB/s (assumed) = 1.2 ms/token ≈ 830 tok/s. The naive single-core compute estimate is about 10 ms/token ≈ 100 tok/s.
  - So M4 decode is compute-bound because the math is naive, and M6 turns it memory-bound. (This is from the memory file of the 2026-09-25 session.)
- **Prefill, P prompt tokens as one matrix × matrix pass:** AI ≈ 0.5 · P.
  - Assume a CPU peak around 1.5 TFLOP/s: 12 cores × AVX2 2×FMA × 8 lanes × 2 × ~4 GHz. **This must be measured, not assumed.**
  - The ridge is then about 30 FLOP/byte.
  - So prefill becomes compute-bound once P ≳ 60 tokens. [inference]
- **RTX 4070 (M7):** public spec is about 500 GB/s. [inference: public spec, not in these repos; measure it.]
  - Decode ceiling ≈ 60 MB / 500 GB/s ≈ 0.12 ms/token.
  - At that point kernel-launch overhead (a few µs × dozens of kernels per token) becomes a visible share of the time. §4.6 covers what to look for.
- **KV cache for stories15M:** 2 × 6 layers × 288 × 4 B = **13,824 B/token**.
  - The full 256-token context is 3.5 MB.
  - So moving KV between workers on one box costs next to nothing compared with the weights. This matters for the post-v1 PD experiment. [inference]

Teaching sequence that works: first ask Luigi for decode AI at batch 1 (0.5). Then ask for prefill of a 100-token prompt (50). Then ask which side of a ~30 ridge each falls on. The prefill/decode split then feels forced by physics rather than by an architecture fashion.

### 3.3 At DeepSeek's scale: why EP, and why a *different* EP per phase

Expert sizes:
- One routed expert = gate + up + down = 3 × 7168 × 2048 = **44.0M params ≈ 44 MB in FP8** [inference from config].
- All routed experts: 256 × 58 MoE layers × 44M ≈ **654B params**.
- So the routed experts are about 97% of the 671B total.
- Non-routed (attention, shared experts, 3 dense layers, embedding, head) ≈ 17B [inference].

**Tokens per expert decide GEMM efficiency.** With `N` GPUs each holding `t` tokens per micro-batch, top-8 routing and `E_phys` physical experts, each physical expert gets about `N·t·8 / E_phys` tokens.

- **Decode, EP144, t = 128 tokens per micro-batch.** In the trace, `_layer_norm_kernel` grid = 128 inside MoE micro-batches [trace].
  - 144·128·8/288 ≈ **512 tokens per expert**.
  - FP8 weights give AI ≈ 2·512/1 ≈ 1024 FLOP/byte.
  - The DeepGEMM 1350 TFLOPS / 3 TB/s ridge is about 450, so this is compute-leaning.
- **Same per-GPU batch in a single node (EP8, 256 experts):** 8·128·8/256 = **32 tokens per expert**. AI ≈ 64, so it is badly memory-bound.
- **Prefill, EP32, t = 8,192 tokens per micro-batch:** 32·8192·8/288 ≈ **7,282 tokens per expert**, far past the ridge.

So prefill already has plenty of tokens per expert from a small unit. Decode only gets enough by pooling 144 GPUs' requests. That is the quantitative reason for the two EP degrees [inference].

**The trace confirms decode expert GEMMs are efficient** [trace, decode.json]:
- The routed-expert gate+up GEMM `fp8_gemm_kernel<4096u, 7168u, ...GemmType 2>` takes about 75–80 µs per micro-batch.
- 1024 token-expert pairs × 2 × 4096 × 7168 ≈ 60 GFLOP, which is about **760 TFLOPS** FP8 [inference].

**Memory is the other reason for the EP split** [inference]:
- Per-GPU routed weights:
  - Prefill: 9 × 58 × 44 MB ≈ **23 GB**.
  - Decode: 2 × 58 × 44 MB ≈ **5.1 GB**.
- Plus about 17 GB of replicated non-routed weights on each GPU.
- A decode GPU therefore has at most about 58 GB left for KV and activations.
- MLA stores 576 values per token per layer (`kv_lora_rank` 512 + rope 64), BF16, over 61 layers = **70,272 B ≈ 68.6 KiB per token**. The decode trace's MLA kernel is templated on 576: `Flash_fwd_kernel_traits_mla<576, ...>` [trace].
- 58 GB / 68.6 KiB ≈ 824k tokens ≈ 165 requests at the published 4,989-token average.
- Throughput implies about 88 active requests per GPU (§6.4), so memory is not binding at the average. At peak it gets closer.
- Decode wants KV room. Prefill wants FLOPs. Separate pools let each be sized for its own bottleneck.

### 3.4 Two reasons to disaggregate; DeepSeek's is the second

1. **Latency isolation (vLLM's framing).**
   - Without disaggregation, a long prefill inserted into the running batch stalls every decoding request. That hurts tail ITL.
   - With it, TTFT and ITL can be tuned separately.
   - vLLM: "Tuning time-to-first-token (TTFT) and inter-token-latency (ITL) separately" and "Controlling tail ITL". Chunked prefill can do the same but "it's hard to figure out the correct chunk size". And: "Disaggregated prefill DOES NOT improve throughput." [doc `~/refs/inference/vllm/docs/features/disagg_prefill.md:12-16`]
2. **Different optimal parallelism and hardware per phase (DeepSeek's framing).**
   - "As we have adopted prefill-decode disaggregation architecture, we employ different degrees of parallelisms during the prefill and decode phases" [doc D6:21].
   - With MoE plus EP, this is a throughput argument: EP144 for everything would waste prefill capacity on communication, and EP32 for decode would starve the experts of tokens [inference].

A pitfall to warn about: Luigi's single-box server will have a dense 15M–110M model and no EP. The *latency-isolation* reason carries over directly; the *parallelism* reason mostly does not. Don't let "DeepSeek does PD disaggregation" become "PD disaggregation is always faster".

### 3.5 Connection to M4 (explicit prefill phase + decode phase, TTFT and decode tok/s)

- llama2.c's `generate` feeds the prompt one token at a time (CLAUDE.md). DeepSpec's loop is the pattern Luigi is about to build:
  - One prefill call over the whole prompt, `target_model(input_ids=input_ids, ...)`, then sample the first token [code `DeepSpec/deepspec/eval/base_evaluator.py:345-357`].
  - Then a decode loop [code `base_evaluator.py:385-430`].
- **Two kinds of metric.** DeepSeek reports:
  - per-request speed: 20–22 tok/s, i.e. latency;
  - per-node throughput: 14.8k tok/s, i.e. aggregate.
  - At batch 1 (M4) the two are the same number. In the server they diverge. It's worth making Luigi name both in M4 so the server metrics later are an extension rather than a redefinition. [inference]
- TTFT = queue wait + prefill compute + first sample. At M4 there is no queue, so TTFT ≈ prefill time. Record them as separate fields now so the server can fill in queue wait later. [inference]
- **A future requirement to keep in mind at M3/M4 design time (don't scaffold it):** speculative decoding needs a "truncate the KV cache back to length n" operation. DeepSpec calls `past_key_values_target.crop(start)` after each verify [code `base_evaluator.py:418,425`]. A KV cache API that only ever appends will have to change at that point.

---

## 4. Overlap: dual micro-batch, the 5-stage decode pipeline, and what the traces show

### 4.1 Prefill: dual micro-batch overlap

The quote: "splitting a batch of requests into two microbatches... these two microbatches executed alternately and the communication cost of one microbatch is hide behind the computation of the other" [doc D6:26-27].

The figure [fig `open-infra-index/202502OpenSourceWeek/figures/Communication-Computation Overlapping during Prefilling Phase.png`, same as `profile-data/assets/prefill.jpg`] has two lanes:
- Compute lane, 108 SMs: `ATTN(mb1) | SHARED(mb0) | ATTN(mb0) | MLP(mb1) | MLP(mb0) | SHARED(mb1)`
- Communication lane, 24 SMs: `COMBINE(mb0) | DISPATCH(mb1) | DISPATCH(mb0) | COMBINE(mb1)`

ATTN = MLA plus the MoE routing gate. SHARED = shared experts.

Attention load is balanced across the two micro-batches: "the same prompt may be split between them" [doc PDR:22]. Splitting by request count alone would leave one micro-batch with the long prompts [inference].

### 4.2 Decode: 5-stage pipeline, 0 communication SMs

"During the decoding phase, the execution durations of different stages are unbalanced. Hence, we subdivide the attention layer into two steps and use a 5-stage pipeline" [doc D6:32].

The figure [fig `.../Communication-Computation Overlapping during Decoding Phase.png`] has two lanes:
- Compute lane, all 132 SMs: `SHARED(mb0) ATTN-0(mb1) | MLP(mb0) | ATTN-1(mb1) | SHARED(mb1) ATTN-0(mb0) | MLP(mb1) | ATTN-1(mb0)`
- Communication lane, **0 SMs**: `DISPATCH(mb0)`, `COMBINE(mb0)`, `DISPATCH(mb1)`, `COMBINE(mb1)`, each running under the compute block that doesn't depend on it.

Legend: ATTN-0 = "MLA down/up projection and other ops after combine all-to-all and before core attention". ATTN-1 = "Core attention, attention output projection and MoE routing gate".

"after RDMA messages are issued, all GPU SMs are freed, and the system waits for the all-to-all communication to complete after the computation has finished" [doc PDR:30].

vLLM has since implemented the same idea as "Dual Batch Overlap" (DBO).
- Two CPU "UBatch" threads ping-pong at yield points [doc `~/refs/inference/vllm/docs/design/dbo.md:9`].
- The schedule is written out at `dbo.md:23-26`: `Comp: |-A0₀-A1₀-||-MLP₁-||-S₁-MLP₀-||-S₀-A0₁-A1₁-|`.
- vLLM uses DeepEP low-latency for decode-heavy workloads and high-throughput for prefill-heavy ones [doc `dbo.md:37`; `~/refs/inference/vllm/docs/serving/expert_parallel_deployment.md:27-28`].
- All DP ranks must agree to micro-batch or none do [doc `dbo.md:52`]. This is the same lock-step constraint again.

### 4.3 DeepEP: normal vs low-latency, hooks, NVLink/RDMA (conceptual)

**Version caveat, important.** The local checkout is **V2.5**:
- "Fully remove V1, including its APIs, NVSHMEM backend, and legacy documentation" [doc `DeepEP/README.md:18`].
- "EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported" [doc `README.md:39`].

The zero-SM, hook-based low-latency mode in D6 and PDR:30 was a **V1** feature, and V1 is what the March-2025 decode trace shows (`dispatch_ll`/`combine_ll` kernels). The V1 README's bandwidth/latency tables **are not in this checkout. Don't quote them from memory.** Use the trace-derived numbers in §4.4 instead.

**What survives in the repo as figures:**
- `DeepEP/figures/normal.png` (high-throughput kernels):
  - CPU: "Launch notify" → wait → "Tensor allocation" ("Notify tensor size ASAP") → "Launch dispatch" → "Launch computation" → "Launch combine" ("Reuse layout information").
  - GPU: Notify → Dispatch (interleaved "IB chunk" and "NVL chunk") → computation → Combine (NVL and IB chunks).
  - The note says "real cases may have hundreds of chunks".
  - So the normal kernels are **hierarchical**: RDMA (InfiniBand) between nodes, then NVLink forwarding within the node, pipelined in chunks.
  - They need a **CPU sync** to learn how many tokens will arrive before output buffers can be allocated.
- `DeepEP/figures/low-latency.png`:
  - Top half: "Traditional overlapping with communication SMs". Two streams: Attention/Dispatch/MoE/Combine for batch 0 on stream 0 and for batch 1 on stream 1.
  - Bottom half: "Overlapping without communication SMs → Faster computation with more SMs". One stream: `Attention 0 | Attention 1 with background RDMA | MoE 0 with background RDMA | MoE 1 ... `.
  - The labels between blocks read "Dispatch 0 issue", "Dispatch 0 receive / Dispatch 1 issue", "Dispatch 1 receive / Combine 0 issue", "Combine 0 receive / Combine 1 issue".
  - That is the hook model: a send kernel returns right after posting RDMA writes. A later **receive hook** kernel waits for arrival. The other micro-batch's compute kernels run in between.

**What a server designer needs from DeepEP V2.5's README:**
- High-throughput and low-latency now share one `EPBuffer` API [doc `README.md:29`]. SM/QP counts are computed analytically [doc :31].
- `do_cpu_sync=True` "obtains exact output sizes ... commonly used for training and prefill". Decode uses `False` with GPU-side counts, and "outputs are allocated to the configured capacity" [doc `README.md:162`].
  - This is the prefill/decode split again. Prefill can afford a CPU round trip per layer. Decode can't, because it wants a fixed-shape, CUDA-graph-able step [inference].
- The overlap pattern: launch communication, do independent work, then `event.current_stream_wait()` [doc `README.md:261-277`].
- Network operations:
  - Traffic isolation via InfiniBand virtual lanes: EP traffic vs everything else [doc `README.md:454-463`].
  - Adaptive routing always on [:465-467].
  - Congestion control off for max bandwidth [:469-471].
  - These matter for role project 6 (p99 spikes can come from the network fabric) [inference].
- Dynamic redundant experts (`lb_prefetch_weights`) push expert weights over NVLink into replica slots before compute [doc `README.md:15,332-337`]. EPLB's replication is static; this makes it per-batch.
- Tools for slow ranks: `antgroup/DeepXTrace`, "A diagnostic analyzer for efficient and precise localization of slow ranks" [doc `README.md:510`]. Stragglers are the enemy of synchronous EP.

**What "low-latency decode" means, in one paragraph for the tutor** [inference]:
- A decode step moves tiny messages: 128 tokens × 7 KB per micro-batch per GPU. There are 2 all-to-alls per MoE layer × 58 layers × 2 micro-batches, so about 232 all-to-alls per step.
- Latency, not bandwidth, dominates. The design goals are:
  1. no CPU involvement: GPU-initiated RDMA, fixed-capacity buffers, CUDA-graph friendly;
  2. no SMs held while bytes are in flight, so compute gets all 132 SMs;
  3. direct sends to each destination expert's GPU, without the two-hop NVLink forwarding that the high-throughput kernels use for bandwidth.
- The price is exposed receive-wait time when the overlap isn't perfect (§4.4).

### 4.4 profile-data: what exists, how to open it, what it shows

How to open (from the README):
- "captured using the PyTorch Profiler ... navigating to chrome://tracing in the Chrome browser (or edge://tracing in the Edge browser)" [doc PDR:3].
- The Perfetto UI (ui.perfetto.dev) also loads PyTorch Chrome-trace JSON [inference]. It works better for the 17 MB prefill file.
- **"we simulate an absolutely balanced MoE routing strategy for profiling"** [doc PDR:3]. So the traces show the *best case*. Real imbalance, which is EPLB's job, only makes things slower.

JSON structure [trace]:
- Top-level keys: `schemaVersion`, `deviceProperties` (H800: 132 SMs, 84.9 GB, sm_90), `distributedInfo` (`{"backend":"nccl","rank":0,"world_size":N}`), `traceEvents`, `traceName`.
- Events use Chrome trace format. `ph: "X"` is a complete event with `ts`/`dur` in µs. Categories are `kernel`, `cpu_op`, `cuda_runtime`, `ac2g` (flow arrows from CPU launch to GPU kernel), `user_annotation`, and a few more.
- `tid` is the CUDA stream. Metadata events name them, e.g. `stream 7`.
- Kernel `args` hold `grid`, `block`, registers, shared memory and estimated occupancy.

| File | Size / events | Config | GPU streams | What I measured [trace] |
|---|---|---|---|---|
| `train.json` | 3.1 MB / 14,240 | world_size 64 (EP64, TP1, 4K seq), 8 GPUs in deviceProperties; a forward+backward chunk pair with 4 MoE layers each, no PP comm [doc PDR:11-12] | 7 = compute (907 kernels); 27 = comm (40: dispatch/combine/notify); 23 = 16 grouped GEMMs | 129 ms kernel window; comm kernels busy 113 ms, 108.9 ms of it concurrent with compute. User annotations `attn(F)`, `mlp(B)`, `mlp(W)`, `dispatch(F)`, `combine(B)`, … ×4, plus `1F1B` |
| `prefill.json` | 17.5 MB / 85,422 | `traceName` = `dsv3-600B-tp1-ep32-input4096-output1-bs16384-split1-sm24.json`; world_size 32 | 7 = compute (3,525 kernels, busy 1,930 ms); **16 = comm** (580 = 116 × {dispatch, combine, notify_dispatch, cached_notify, get_dispatch_layout}, busy 1,799 ms); 13 = 6 NCCL allreduces | Kernel window **2,120 ms** for one full forward over 16,384 tokens/GPU. 122 attention kernels = 61 layers × 2 micro-batches; 116 dispatches = 58 MoE layers × 2. **Comm kernel time 1,783 ms, of which 1,625 ms (91%) is concurrent with compute kernels** |
| `decode.json` | 4.7 MB / 19,417 | `traceName` = `./decode-debug-rank0.json`; world_size 128 | **Everything on stream 7** (3,642 kernels); 2 stray LL kernels on 16; 3 allreduces on 13 | Kernel window 95.9 ms; stream-7 span 95.5 ms, idle gaps 6.8 ms (7%). Median kernel 9 µs; 33% of kernels < 5 µs. `dispatch_ll` ×235 (11.9 ms total), `combine_ll` ×235 (5.1 ms total). **Comm kernels ≈ 17 ms ≈ 18% of the step, with no concurrency** (a single stream, so overlap happens on the NIC, not between kernels) |

**Per-kernel prefill numbers** [trace], per micro-batch-layer (8,192 tokens):

| Kernel | Time |
|---|---|
| Core attention `flash::compute_attn_ws` | ~2.1–2.2 ms |
| Routed-expert gate+up `fp8_gemm<4096,7168,…GemmType 1>` | 3.6 ms average |
| Expert down | 2.4 ms |
| **Dispatch** (comm) | **4.6 ms** average |
| **Combine** (comm) | **8.7 ms** average |

- Compute and comm per layer are about equal: roughly 33 ms of compute vs 31 ms of comm-kernel time per layer.
- Run serially, they would add to about 3.7 s of kernel time. Overlapped, the pass takes 2.1 s. So overlap gives about a **1.75× speedup** [inference; this ignores the SMs the comm kernels take].

**Sanity check of the prefill comm times** [inference; the 400 Gb/s ≈ 50 GB/s NIC per GPU is an assumption, not stated in these repos]:
- Group-limited routing sends each token to at most 4 groups (`n_limited_groups: 4`). With hierarchical EPLB placing 2 groups per node, that is at most 3 remote nodes.
- DeepEP's normal kernels send once per node over IB and fan out over NVLink (normal.png).
- Dispatch: ≤ 3 × 7,168 B (FP8) per token ≈ 21.5 KB × 8,192 tokens ≈ 176 MB → about 3.5 ms at 50 GB/s. Measured: 4.6 ms.
- Combine in BF16: about 352 MB → about 7 ms. Measured: 8.7 ms.
- Same order of magnitude, so the comm is bandwidth-bound.

**Decode kernel sequence** [trace], one repeating unit from the middle of `decode.json`. The column "my label" is **[inference]**: it maps kernels to stages using the GEMM shapes (e.g. 2112 = q_lora 1536 + kv 576; 24576 = 128 heads × 192; 16384 = 128 × 128 → 7168 is o_proj; 4096 = 2 × 2048 is expert gate+up).

```
t(ms)   dur    kernel                                             my label [inference]
21.355    4us  _layer_norm_kernel                                 ATTN-0 (micro-batch A)
21.360   13us  fp8_gemm<2112,7168>   q_a + kv_a down-projection   ATTN-0
21.379   19us  fp8_gemm<24576,1536>  q up-projection              ATTN-0
21.398   16us  bf16 gemm (nn)        absorbed k up-projection     ATTN-0
21.415    7us  rotary_embedding_with_kv_cache                     ATTN-0 (+ KV write)
21.423   65us  dispatch_ll           ← receive hook: waits        DISPATCH recv (B)
21.489   75us  fp8_gemm<4096,7168,GemmType 2> routed experts      MLP (B)
21.565    7us  swiglu                                             MLP (B)
21.573   32us  fp8_gemm<7168,2048,GemmType 2> expert down         MLP (B)
21.606   19us  combine_ll            ← send                       COMBINE send (B)
21.626  241us  flash_fwd_splitkv_mla core attention               ATTN-1 (A)
21.868   11us  flash_fwd_splitkv_mla_combine (split-KV reduce)    ATTN-1 (A)
21.901   49us  fp8_gemm<7168,16384>  o_proj                       ATTN-1 (A)
21.968    6us  top2_sum_gate         group-limited top-k gate     ATTN-1 (A)
21.975   17us  combine_ll            ← receive                    COMBINE recv (B)
21.996   17us  dispatch_ll           ← send                       DISPATCH send (A)
22.018   18us  fp8_gemm<4096,7168,GemmType 0> shared expert       SHARED
... repeats with A and B swapped; ~0.72 ms per micro-batch-layer
```

What to read from it:
- **Sends are about 17 µs.** They only post RDMA.
- **Receives are 65–86 µs** where the data isn't there yet. That is the exposed part of the latency.
- Core MLA attention is the biggest block, 241 µs of every 720 µs.
  - It reads about 302 MB of latent KV (64 requests × 4K × 1,152 B), so about 1.25 TB/s.
  - At 2 query tokens per request (see below) it does about 146 GFLOP ≈ 600 TFLOPS.
  - MLA's 128 heads share one 576-dim latent KV, so decode attention has AI ≈ 480 FLOP/byte and is **compute-leaning**. This is unlike Luigi's MHA model, whose decode attention is memory-bound at about 1 FLOP/byte per head [inference].

**Evidence of MTP speculative decoding inside the decode trace** [trace + inference]:
1. The three dense layers run as one un-split batch. `_layer_norm_kernel` and `rotary` have grid = **256**, and the dense FFN GEMM `fp8_gemm<36864,7168>` appears 3 times, not 6. The MoE micro-batches have grid = 128 each. So each step processes **256 tokens for "128 requests per GPU"** [doc PDR:30], i.e. 2 tokens per request.
2. After the last MoE layer there is:
   - an LM-head GEMM (`sm90_xmma_gemm_bf16f32…`, 756 µs);
   - two sampling passes (`apply_penalty_kernel`, `topk_kernel`, `radixSort`, `mask_top_p_kernel`, softmax; 23 kernels, 0.92 ms total);
   - then embedding lookup, **two RMSNorms**, a concat, a 93 µs projection GEMM, and **one more full attention layer over the whole batch** (MLA 467 µs = 2 × 241). Its MoE follows, and a second head GEMM (669 µs at 94.5 ms).
   - That matches the MTP module exactly: "enorm & hnorm: RMSNorm parameters required for speculative decoding", "eh_proj", and "Additional Transformer Hidden Layer ... model.layers.61" [doc `DeepSeek-V3/README_WEIGHTS.md:44-47`]. MTP "can also be used for speculative decoding" [doc `DeepSeek-V3/README.md:66-67`].
3. Conclusion (hypothesis, strong): each decode step verifies 1 draft token per request, taking 2 tokens through the main model, and drafts the next with the MTP layer. §6.4 shows this explains the published 20–22 tok/s.

**Sampling's share (role project 1)** [trace]:
- LM head about 0.76 ms plus sampling kernels about 0.92 ms, out of about 95 ms. That is **about 1.8% of the step**, at batch 128 on a 671B model.
- For stories15M on CPU the share will be far larger, because the model is tiny and top-p sorts 32k probs every step. That is exactly the M6 measurement CLAUDE.md asks for [inference].

### 4.5 A trace-reading recipe (for the tutor; Python, analysis only)

This is what produced the tables above. Scripts are in the scratchpad, not the repo:

```python
d = json.load(open("prefill.json")); ev = d["traceEvents"]
k = [e for e in ev if e.get("cat") == "kernel"]          # GPU kernels
by_stream = group k by e["tid"]                           # tid == CUDA stream
comm = intervals of kernels whose name contains dispatch/combine/notify
comp = intervals of all other kernels
busy(x) = length of union(intervals); overlap = length of intersection(union(comm), union(comp))
```

Luigi can run the same analysis on his own nsys export (`nsys export --type sqlite` or JSON) in M7. The metric "fraction of comm time hidden under compute" transfers directly to "fraction of H2D copy time hidden under compute" [inference].

### 4.6 What Luigi's M7 nsys profile should look for (checklist) [inference]

1. **Gaps between kernels on the compute stream.**
   - The DeepSeek decode trace is 7% idle, even with CUDA-graph-style engineering.
   - For a 15M model at batch 1, kernels will be a few µs each, so launch overhead may dominate.
   - Measure idle % and count kernels per token.
2. **Memcpy on the critical path.** Are logits copied device→host every step, synchronously, for CPU sampling? Does the step wait on it?
3. **Stream concurrency.** If he later overlaps KV transfer or weight upload with compute, it should appear as a copy-engine lane running under compute kernels. That is the single-GPU analogue of the prefill figure.
4. **Achieved bandwidth per decode matmul** = bytes of weights / kernel time, against a measured ~500 GB/s. This is the roofline check for role project 2.
5. **Sampling share.** LM head + softmax + top-p as a % of the step (role project 1), compared with the 1.8% above.
6. **Prefill vs decode signature.** Prefill should show a few long GEMM kernels. Decode should show many short ones. Seeing both on one timeline is the M4/M7 "two phases" evidence for the README.

### 4.7 DualPipe (brief; training, but two ideas transfer)

- "bidirectional pipeline parallelism ... achieves full overlap of forward and backward computation-communication phases, also reducing pipeline bubbles" [doc `DualPipe/README.md:3`].
- The bubble table [doc `DualPipe/README.md:26-36`]:
  - 1F1B: `(PP-1)(F+B)`.
  - DualPipe: `(PP/2-1)(F&B+B-3W)`, at the cost of 2× parameters per device.
- The training trace (`profile-data/assets/train.jpg`, 112 compute SMs / 20 comm SMs) pairs a forward chunk with a backward chunk, so one chunk's communication hides under the other's compute. It's the same trick as the dual micro-batch overlap, applied to training.
- **Transfer to serving** [inference]:
  1. Any pipeline pays a fill/drain cost of about (stages − 1) × per-stage time. The same formula governs chunked chain broadcast in interview Q2 (§10.2).
  2. Overlap needs **two independent streams of work**. In training, DualPipe gets them from the two directions. In inference, you split the batch.

---

## 5. Load balancing: the three balancers, EPLB, LPLB, and role project 3

### 5.1 Why min-max, and why it's so important under EP

"if a single GPU is overloaded with computation or communication, it becomes a performance bottleneck, slowing the entire system while leaving other GPUs idle" [doc D6:39].

With a synchronous all-to-all every layer, **step time = max over the 144 GPUs**. A 20% overload on one GPU costs 20% for all 144 [inference]. Every objective below therefore balances the max, not the mean.

The metric the LPLB script prints is `balance = max / mean` [code `LPLB/scripts/run_ep16_cube8p2e.py:90`]. Use the same metric for the post-v1 router.

### 5.2 Prefill LB [doc D6:41-45]

- Problem: "Varying request counts and sequence lengths across DP instances lead to imbalanced core-attention computation and dispatch send load."
- Objectives:
  - (a) balance core-attention computation across GPUs;
  - (b) equalize input token counts per GPU ("dispatch send load").

Worked example [inference]. Two GPUs, each given 8,192 prompt tokens:
- GPU A: one 8,192-token prompt. Causal attention pairs ≈ L²/2 = 33.6M.
- GPU B: eight 1,024-token prompts. 8 × 0.52M = 4.2M pairs.
- Tokens (objective b) are perfectly balanced. Attention (objective a) is off by **8×**.
- So the LB needs two numbers per candidate placement: Σ L (dispatch) and Σ L² (attention).
- With a cached prefix `p` and `n` new tokens, attention cost ≈ n·(p + n/2). Cache hits shrink the token count but not the attention to the cached part.

### 5.3 Decode LB [doc D6:46-50]

- Problem: "Uneven request counts and sequence lengths across DP instances cause disparities in core-attention computation (linked to KVCache usage) and dispatch send load."
- Objectives:
  - (a) "Balance KVCache usage across GPUs";
  - (b) "Equalize request counts per GPU".

Worked example [inference]:
- GPU A has 100 requests × 1K context. GPU B has 20 × 5K.
- KV usage is equal (100K tokens each), so attention time is about equal.
- A sends 5× more tokens per all-to-all, so A is the dispatch straggler.
- The decode LB must weigh both. In role-project terms, placement is **two-dimensional bin packing** (KV bytes, request slots). A single "least loaded" scalar is not enough.

### 5.4 EPLB: the algorithm line by line

Interface [code `EPLB/eplb.py:131-162`]:

```
rebalance_experts(weight[layers, n_logical], num_replicas, num_groups, num_nodes, num_gpus)
  -> phy2log[layers, num_replicas]      # which logical expert each physical slot holds
     log2phy[layers, n_logical, maxcnt] # replica slots of each logical expert (-1 padded)
     logcnt[layers, n_logical]          # number of replicas per logical expert
```

- `weight` is the **estimated load**. "the exact method to predict the loads of experts is out of this repo's scope. A common method is to use moving average of historical statistics" [doc `EPLB/README.md:11-13`].
- **EPLB is a periodic, static planner. It is not per-batch.**
- Policy choice [code `eplb.py:150-156`]:
  - `num_groups % num_nodes == 0` → hierarchical.
  - Otherwise, call the same function with `num_groups=1, num_nodes=1`. That is the "global" policy.
  - `log2phy` is built by one scatter [code `eplb.py:157-161`].

**`balanced_packing(weight[X, n], num_packs)`** [code `eplb.py:5-41`]:
- Pack n items into m packs of **exactly n/m items each** [:19-20]. There's a fast path for one item per pack [:22-25].
- Otherwise it is **LPT greedy** (longest processing time first):
  - Sort items by weight, descending [:27].
  - For each item, choose the lightest pack that still has room [:34-35].
  - Record the pack and the rank inside it [:37-40].
- Complexity is O(n·m) per layer, in a Python loop on CPU.

**`replicate_experts(weight[X, num_log], num_phy)`** [code `eplb.py:44-71`]:
- Start with one replica each [:62-64].
- For each extra slot, give it to the expert with the **highest current per-replica load** `weight/logcnt` [:66-70].
- This greedy is optimal for minimizing the max per-replica load `max_i w_i/c_i` subject to Σc_i = num_phy: each increment removes the current bottleneck [inference].

**`rebalance_experts_hierarchical`** [code `eplb.py:74-129`]:
- **Step 1, groups → nodes** [:103-108]:
  - Sum the load per expert group [:104].
  - `balanced_packing` puts the groups onto nodes [:105].
  - Build a permutation `log2mlog` that renumbers experts so each node's experts are contiguous ("middle logical" ids) [:106-108].
- **Step 2, replicate within each node** [:110-113]:
  - Reshape to `[layers × nodes, experts_per_node]`.
  - Replicate to `num_physical/num_nodes` slots per node.
  - So **replicas never cross nodes**: node locality is kept.
- **Step 3, replicas → GPUs** [:115-120]:
  - Per-replica load = load / count [:117].
  - `balanced_packing` onto GPUs within the node, with exactly `phy_experts_per_gpu` replicas each [:118].
- **Steps 4–5, map back to original ids** [:122-128].

The motivation: "thanks to the group-limited expert routing used in DeepSeek-V3, we also attempt to place the experts of the same group to the same node to reduce inter-node data traffic" [doc `EPLB/README.md:6-8`].

**Worked example.** This is the README example, run as-is from the repo with the scratchpad script, plus the intermediate values I printed [code + inference]:
- Input, layer 0, 12 experts: `[90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86]`.
- Setup: 4 groups of 3, 2 nodes, 8 GPUs, 16 physical slots, i.e. 4 redundant [doc `EPLB/README.md:37-51`].

1. **Group loads:**
   - g0 = {0,1,2} = 262
   - g1 = {3,4,5} = 330
   - g2 = {6,7,8} = 116
   - g3 = {9,10,11} = 325
2. **LPT to 2 nodes, 2 groups each.** Sorted order: g1, g3, g0, g2.
   - g1 → node0 (330).
   - g3 → node1 (325).
   - g0 → node1, the lighter one: 587. Node1 is now full.
   - g2 → node0: 446.
   - Result: **node0 = {g1, g2} = 446**, **node1 = {g3, g0} = 587**.
   - This greedy happens to be optimal here. The alternatives give max 592 or 655.
   - Note the node imbalance already caps what the later steps can achieve: node1's 4 GPUs must carry 587, i.e. 146.8 each on average, while the ideal is 129.1.
3. **Replicate within each node** (8 slots per node, 6 experts → 2 replicas):
   - Node0 loads (experts 3,4,5,6,7,8) = [61, 104, 165, 39, 4, 73]. Replicate e5 (165 → 82.5 per replica), then e4 (104 → 52).
   - Node1 loads (experts 9,10,11,0,1,2) = [56, 183, 86, 90, 132, 40]. Replicate e10 (183 → 91.5), then e1 (132 → 66).
4. **Pack replicas to GPUs** (2 per GPU). Node0 replica loads sorted: 82.5, 82.5, 73, 61, 52, 52, 39, 4.
   - LPT gives GPU0 = {e5, e6} = 121.5, GPU1 = {e5, e7} = 86.5, GPU2 = {e8, e4} = 125, GPU3 = {e3, e4} = 113.
   - Node1 gives 147.5, 131.5, 156, 152.
5. **Result:** `phy2log[0] = [5,6, 5,7, 8,4, 3,4, 10,9, 10,2, 0,1, 11,1]`. This matches the README output [doc `EPLB/README.md:55`] and the figure `EPLB/example.png`.

Comparison (max GPU load / ideal) [code run + inference]:

| Layer | No replication, 12 experts on 4 GPUs, contiguous | EPLB hierarchical (8 GPUs) | EPLB global (same call with num_nodes=3, so the fallback) |
|---|---|---|---|
| 0 | 330 / 258.25 = **1.28** | 156 / 129.1 = **1.21** | 138.5 / 129.1 = **1.07** |
| 1 | 516 / 289 = **1.79** | 179.5 / 144.5 = **1.24** | 172 / 144.5 = **1.19** |

The lesson: **hierarchy trades balance for locality.** Global balances better, but replicas and groups scatter across nodes, so more RDMA traffic. DeepSeek picks hierarchical where it's possible (prefill, 4 nodes) and global where it isn't (decode, 18 nodes) [inference; README:21-31].

Pitfalls when teaching EPLB:
- The equal-count constraint in `balanced_packing` exists because each GPU has exactly k expert slots. A request router doesn't need it.
- Replication (step 2) and packing (step 3) are optimized separately, so the result is not jointly optimal.
- It uses predicted loads, so it is only as good as the forecast.
- It does nothing about per-batch noise. That gap is what LPLB addresses.

### 5.5 LPLB: an LP over redundant-expert edges (research)

- **What:** "solves optimal token assignments for each batch to achieve dynamic load balancing ... embedded LP solver implements single-SM Interior Point Method (IPM)" using cuSolverDx/cuBLASDx [doc `LPLB/README.md:3`]. "currently in the early research stage" [:5].
- **When it helps over EPLB:** EPLB "handles static imbalances (e.g., consistently overloaded experts due to data distribution), LPLB targets per-batch fluctuations caused by small-batch randomness during training" [doc `LPLB/README.md:59`].
  - Note the stated target is **training**.
  - EPLB is still used, but "reordering only, no replication". The heaviest experts are then replicated according to a fixed topology [:65].

**Formulation**, reconstructed from `minilp.cu`. The notation is mine; the structure is the code's [code `LPLB/lplb/resources/csrc-tmpl/minilp.cu:61-62,203-292`].

For one group of `G` ranks, each with `D` redundant slots (defaults G=8, D=2, "Cube8P2E" [:20-26]):
- `F_i` = fixed load on rank *i*: experts that aren't replicated [:223-233, as `b[i] = -Σ`].
- `w[r][d]` = load of rank *r*'s d-th hot expert [:203-220].
- Its replica lives on the rank `h` with `r2o[h][d] == r` [:248-256].

Variables:
- `x[r][d]` = fraction kept on the original.
- `y[r][d]` = fraction sent to the replica.
- `s_i ≥ 0` = slack.
- `t` = max load.
- `z` = big-M artificial.
- That is NV = 2GD + G + 2 variables and NC = G + GD constraints [:61-62].

```
minimize     t + 1000·z                                          c vector [:284-292]
subject to   F_i + Σ_d w[i][d]·x[i][d] + Σ_{(r,d): h(r,d)=i} w[r][d]·y[r][d] + s_i − t = 0   ∀ rank i   [:239-264]
             x[r][d] + y[r][d] = 1                                ∀ (r,d)    [:265-272]
             x, y, s, t, z ≥ 0 ;  z's column = b − A·1 so that x = 1 is feasible (Big M)      [:275-281]
```

**Solver** [code `minilp.cu:331-367`]:
- Start at x = **1** and run **5 iterations of primal affine scaling**:
  - `ax2 = A·X²` [:336-340], `ax2a = A X² Aᵀ` [:343], `ax2c = A X² c` [:346];
  - solve (A X² Aᵀ) y = A X² c with a cuSolverDx Cholesky `posv` [:29-39, 349];
  - reduced costs r = yᵀA [:352]; direction d = x ∘ (c − r) [:357];
  - step α = 0.999/max(d) [:361]; x ← x ∘ (1 − α d) [:365].
- One CUDA block per group [code `LPLB/csrc/plugin.cpp:561`].
- **Feasibility check:** d_max < 0.1, z < 1e-4, residual < 0.05 [:381-383].
- **If infeasible, fall back to a 50/50 split** [:395-397]. That is graceful degradation, worth pointing out.

**Getting the global load vector** [code `minilp.cu:110-176`]: a **hierarchical all-reduce of a tiny histogram** (one float per expert).
- Inter-node all-gather with NVSHMEM `putmem_signal` [:116-135].
- Sum across nodes [:139-144].
- Intra-node NVLink reduction through shared buffers and atomics [:146-162].
- Normalize by the max [:164-175].
- This matches the README: "optimized using NVLINK and NVSHMEM instead of torch.distributed.allreduce" [doc `LPLB/README.md:65`]. It's also the answer pattern for interview Q3 (§10.3).

**Mapping tokens to replicas** [code `minilp.cu:445-515`]:
- Each token routed to logical expert *e* takes an arrival counter.
- The counter is scrambled with `(count*499 + 41) % total` [:505], so the split doesn't depend on SM order.
- The token goes to the replica if the scrambled index ≥ the original's expected share `x·total` [:506-508].
- Since 499 is prime, the map is a permutation unless `total` is a multiple of 499 [inference].

**Topologies** [doc `LPLB/README.md:75-78`; code `LPLB/tests/utils.py:97-119`]:
- Cube: 8 GPUs, ≥2 experts per GPU.
- Hypercube: 16 GPUs.
- Torus: one intra-node neighbour and one inter-node neighbour.
- `r2o` constraints [code `LPLB/lplb/planner.py:48-59`]:
  - Every rank has the same number of redundant slots.
  - Slot d on a rank copies slot d of exactly one other rank.
  - So `max_logcnt == 2` is asserted [:173].

**Stated limitations** [doc `LPLB/README.md:69-71`]:
1. "balances only total token count, not accounting for non-linearity in grouped matrix multiplication time costs".
2. "The solver takes ~100 µs for intra-node optimization (longer for inter-node), which may be non-negligible for small batches".
3. "Under extreme global load imbalance, LPLB may perform worse than EPLB ... (LPLB avoids assigning multiple replicas to the same original expert)".

Limitation 1 is the same trap as "token count ≠ attention cost" in §5.2. **Cost models must be non-linear.**

### 5.6 Role project 3: from EPLB/LPLB to a CPU+GPU request router on one box

The concept mapping [inference throughout]:

| DeepSeek concept | Where | Post-v1 server analogue (one box, CPU and GPU workers) |
|---|---|---|
| Min-max objective (synchronous EP) | D6:39,54 | Workers are independent (no collective), so the objective becomes **SLO attainment**: p99 TTFT/TPOT under capacity limits. Min-max utilization is only a proxy |
| LPT `balanced_packing` | eplb.py:5-41 | Assign a batch of queued requests to workers: heaviest first, to the worker with the earliest *expected finish* = (queued work + w) / speed. Heterogeneous speeds replace "equal pack sizes" |
| Hot-item replication (`replicate_experts`) | eplb.py:44-71 | Replicate **hot prefixes**: the KV of a popular system prompt on both workers, allocated by "max per-replica load" to decide which prefixes get copies |
| Hierarchy: groups → nodes → GPUs | eplb.py:103-120 | Tier 1: choose a pool (prefill-capable GPU vs CPU; or prefill pool vs decode pool). Tier 2: choose the worker. Keep a conversation's turns together (prefix affinity), like keeping an expert group on one node |
| Static plan from moving averages | EPLB/README.md:11-13 | Periodic re-planning of pool sizes and prefix placement from windowed stats |
| Per-batch LP (LPLB) | minilp.cu | Per-request or per-tick routing: fractions of each traffic class sent to each worker, min max-utilization LP. With two workers this is solvable in closed form, which is a nice exercise |
| Two objectives per phase (tokens vs attention; KV vs request count) | D6:41-50 | Prefill routing uses a cost of about ΣL and ΣL². Decode admission uses KV bytes free *and* slots free. Two-dimensional packing |
| LPLB's fallback to 50/50 on infeasibility | minilp.cu:395-397 | Always have a trivial safe policy when the smart one fails or times out |
| Linear-cost limitation | LPLB/README.md:69 | Measure worker cost curves (latency vs batch size, vs context length) in M6/M7 before designing the router. Role project 4 feeds role project 3 |

A design point to give Luigi as a *question*, not a decision:
- On one box, the GPU beats the CPU at *both* phases. Decode is memory-bound: about 500 GB/s vs about 50 GB/s, roughly 10× [inference; both to be measured]. So what is the CPU worker for?
- Candidates:
  - (a) overflow capacity for low-priority or batch traffic when the GPU queue is full;
  - (b) the KV tier (host RAM as "external KV storage"), not a compute worker;
  - (c) a CPU-decode / GPU-prefill split, *if* measurements show it helps TTFT under load;
  - (d) the draft model for speculative decoding (§7).
- The router's value depends on this choice. Let Luigi reason it out from his own M6/M7 numbers.

---

## 6. Numbers → a quantitative model (role project 4). All math here is [inference]

### 6.1 Cost and revenue reproduce exactly

- Cost = 226.75 nodes × 8 GPUs × $2 × 24 h = **$87,072**. This equals D6:68.
- Revenue at R1 prices:
  - Hits: 342B × $0.14/M = $47,880.
  - Misses: (608 − 342) = 266B × $0.55/M = $146,300.
  - Output: 168B × $2.19/M = $367,920.
  - Total: **$562,100**. D6:78 says $562,027; the gap comes from rounding the token counts to billions.
- Margin: 562,027 / 87,072 − 1 = **5.455 → "545%"**. ✓

### 6.2 The per-node throughputs are consistent with the node count

- Input rate = 608e9 / 86,400 = **7.04M tok/s**. Output rate = 168e9 / 86,400 = **1.94M tok/s**. That is 3.62 input tokens per output token.
- Prefill nodes = 7.04M / 73.7k = **95.5**. Decode nodes = 1.94M / 14.8k = **131.4**. Sum = **226.9**, against 226.75 published.
- So "per-node throughput" = total tokens / node-seconds *of that pool*, averaged over the day, idle time included.
- The prefill share is 0.42, which agrees with the chart decomposition in §2.5.
- Units: 95.5 / 4 ≈ 24 prefill units; 131.4 / 18 ≈ 7.3 decode units on average.

### 6.3 Prefill: production vs the trace

- Production: 73.7k tok/s per node includes hits. Computed (miss) tokens = 73.7k × 266/608 = **32.2k per node = 4.0k per GPU**.
  - At about 2 × 37B = 74 GFLOP per token for the linear layers, that is about **300 TFLOPS per GPU** on average (attention not counted).
- Trace: 16,384 tokens / 2.12 s = **7.7k tok/s per GPU = 61.8k per node** ≈ **570 TFLOPS** linear. That is 42% of DeepGEMM's 1350 peak [OII:59], with perfectly balanced routing.
- Production computed-token rate / trace rate ≈ **0.52**. Plausible reasons, none of them stated in the docs:
  - day-average load below peak, including step-scaling slack;
  - real routing imbalance, since the trace simulates perfect balance [PDR:3];
  - prompt lengths different from 4K, with attention growing as L²;
  - scheduling gaps and KV handoff.
- An exercise for Luigi: which of these could you measure, and how?

### 6.4 Decode: concurrency, TPOT, KV, and why 20–22 tok/s

- Per GPU: 14.8k / 8 = **1,850 tok/s**.
- At 20–22 tok/s per request: **84–93 concurrent requests per GPU**, and about **88k–97k concurrent streams** fleet-wide. TPOT ≈ 45–50 ms.
- KV per GPU at 88 requests × 4,989 tokens × 70,272 B ≈ **31 GB**. That fits in the roughly 58 GB of free HBM (§3.3).
- **The trace gives a step of about 93 ms** for 128 requests per GPU (≈ 61 layers × 2 × 0.72–0.76 ms, plus head, sampling and MTP). With MTP, tokens per request per step = 1 + a, where a is the acceptance rate of the draft token:

  | a | per-request tok/s | per-GPU tok/s | per-node tok/s |
  |---|---|---|---|
  | 0 (no MTP) | 10.8 | 1,376 | 11.0k |
  | 0.80 | 19.4 | 2,477 | 19.8k |
  | 0.85 | 19.9 | 2,546 | 20.4k |
  | 0.90 | 20.4 | 2,615 | 20.9k |

  - Without MTP the trace predicts about 11 tok/s per request. The published speed is 20–22. MTP with a ≈ 0.8–0.9 closes the gap.
  - **The acceptance rate itself is not in these repos.** The V3 technical report (arXiv 2412.19437) discusses MTP acceptance; look it up before quoting a number.
  - Production per-node throughput (14.8k) is below the trace's MTP prediction (~20k). That is consistent with a day-average load below the profiled 128 requests per GPU.
- Cost per million tokens, GPU time only:
  - Decode: $2 / (1,850 × 3,600 / 1e6) = **$0.30/M output**, against a $2.19 price.
  - Prefill: $2 / (9,212 × 3,600 / 1e6) = **$0.06/M input** including hits, or **$0.14/M** per computed token, against $0.55 for a miss.

### 6.5 Why cache hits are cheap: recompute vs load

- Recompute cost per token ≈ 1 / (4.0k tok/s/GPU) ≈ **0.25 ms of H800 time**.
- Loading its KV: 70,272 B. Assume about 5 GB/s per GPU, i.e. 40 GiB/s per client node ÷ 8 [OII:82]. That is about **14 µs**.
- So a hit is about **17× cheaper** than a miss.
- The price ratio is $0.55 / $0.14 = 3.9×. The price discount is smaller than the cost saving.
- Disk read volume: 342B hit tokens × 70 KB ≈ **24 PB/day ≈ 278 GB/s average** fleet-wide, about 2.9 GB/s per prefill node on average.
  - That is well under the 40 GiB/s client peak and about 4% of 3FS's 6.6 TiB/s aggregate stress-test number [doc `3FS/README.md:30`].
  - This assumes the on-disk format is BF16 MLA latents. The actual format is not documented.

### 6.6 EP network load in decode (a sanity check of "why overlap")

- Per GPU per micro-batch-layer:
  - Dispatch: 128 tokens × 8 × 7,168 B (FP8) = 7.3 MB.
  - Combine: the same in BF16 = 14.7 MB.
- × 2 micro-batches × 58 layers = **2.55 GB per step**, i.e. **27 GB/s at a 93 ms step**.
- That is about 55% of a 50 GB/s NIC. The NIC figure is assumed, not stated in these repos.
- So the network is busy half the time. Without overlap it would add something like 50 ms to a 93 ms step.

### 6.7 A fill-in template for Luigi's own model (M0 → M4 → M6 → M7)

```
decode_time_per_token  ≈ max( weight_bytes / BW_measured ,  2·params / FLOPS_measured ) + overhead
prefill_time(P)        ≈ max( weight_bytes / BW ,  2·params·P / FLOPS ) + attention(P²) + overhead
TTFT(P)                ≈ queue_wait + prefill_time(P) + sample_time
server decode tok/s    ≈ B / step_time(B)   with step_time(B) ≈ max(weights/BW + B·KV_bytes·ctx/BW, 2·params·B/FLOPS)
spec-decode speedup    ≈ E[accepted+1] / (1 + k·c)   (see §7)
```

Every term is a measurement Luigi takes himself. DeepSeek's docs show how to check the published numbers against each other (§6.2). He should do the same for his own benchmark table.

---

## 7. Speculative decoding: DeepSpec, MTP, and where it fits after v1

### 7.1 First principles, with a worked example on stories15M [inference]

- In decode at batch 1, one forward pass reads all weights (60 MB) to produce 1 token. When memory-bound (after M6), the time is about 60 MB / BW regardless of how many token positions go through.
- **Verification is one forward over k+1 positions:** the current token plus k draft tokens. DeepSpec does exactly this: `verify_length = draft_token_count + 1` and a single `target_model(input_ids=proposal.verify_input_ids, ...)` [code `DeepSpec/deepspec/eval/base_evaluator.py:214-223`].
  - The weights are read once and the FLOPs go up k+1 times.
  - k = 4 costs 150 MFLOP. With about 125 GFLOP/s available (a multithreaded AVX2 rate, not the naive M4 rate) that's about 1.2 ms, the same as the memory time. **So verifying 5 positions costs about the same as decoding 1.**
- If each draft token is accepted with probability α (i.i.d. model), expected tokens committed per verify = `(1 − α^{k+1}) / (1 − α)`:
  - α = 0.8, k = 4 → **3.36**
  - α = 0.8, k = 7 → 4.16
  - α = 0.6, k = 4 → 2.31
  - α = 0.9, k = 4 → 4.10
- With a draft costing a fraction c of a target step (e.g. c = 0.05), the speedup ≈ 3.36 / (1 + 4 × 0.05) = **2.8×**.
- **Quiz hook:** in M4's naive, compute-bound regime, verifying 5 positions costs 5× as much, so speculation gains ~nothing. Speculation only pays once decode is memory-bound. This is the same `min(compute, memory)` reasoning as his D2 question ("20× faster math → ~830 tok/s").
- At large batch the server is closer to compute-bound, so the benefit shrinks. That's a real serving trade-off: spec decode helps latency at low load more than throughput at high load.

### 7.2 Correctness: the rejection sampling in the code

- Accept draft token i with probability `min(1, p_target(x_i) / p_draft(x_i))` [code `base_evaluator.py:252-256`].
- Take the accepted prefix via `cumprod` [:257-258].
- At the first rejection, sample from the **residual** `max(0, p_target − p_draft)`, normalized [code `DeepSpec/deepspec/utils/sampling.py:34-44`, called at `base_evaluator.py:278-283`].
- If everything is accepted, sample a bonus token from the last target position [:284-285].
- This keeps the output distribution identical to the target's. At temperature 0, `logits_to_probs` becomes one-hot argmax [code `sampling.py:6-11`], so acceptance = "draft equals target argmax".
- For Luigi's D1-style testing: spec decoding at temperature 0 must reproduce plain greedy output **token for token**. That's a free correctness oracle [inference].

### 7.3 Metrics (DeepSpec's definitions) [code `base_evaluator.py:407-424,482-503`]

- Per verify, the committed length = accepted draft tokens + 1 [:423]. At an early stop token, it is just the accepted count [:416].
- `acceptance_length` = Σ committed / number of proposals [:482-484]. This is "tokens per target forward", often called τ.
- `verify_rate` = Σ committed / (Σ proposal lengths + number of proposals) [:488-490]. It is the fraction of verified positions that got committed.
- `accept_rate@pos` = accepted at position i / proposals that reached position i [:495-503]. It shows how fast acceptance decays along the draft.
- The eval runs batch size 1 only: `assert input_ids.size(0) == 1` [:331]. The published numbers are therefore latency-oriented, not serving throughput.

### 7.4 The three draft models, from the configs [code `DeepSpec/config/*`]

- **Eagle3** (`eagle3_qwen3_4b.py:10-16`): a 1-layer autoregressive draft head (`draft_num_hidden_layers=1`) fed by target hidden states from layers `[1, 9, 17, 25, 33]`. It is trained with test-time-training rollout length 7 (`ttt_length=7`). It drafts one token at a time.
- **DFlash** (`dflash_qwen3_4b.py:10-28`): a 5-layer draft that predicts a whole **block of 7 tokens in parallel** from mask tokens (`block_size=7`, `mask_token_id`), conditioned on the same target layers.
  - In DeepSpec it is literally DSpark with the Markov head off (`markov_rank=0`), the confidence head off, and CE-only loss.
- **DSpark** (`dspark_qwen3_4b.py:10-30`; paper title "Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation", `DeepSpec/README.md:91`) = DFlash plus two heads:
  - A **low-rank Markov head** (`markov_rank=256`). It adds a bias from the previous token's embedding to each position's logits, `markov_w2(markov_w1(prev_token))` [code `deepspec/modeling/dspark/markov_head.py:17-31`]. This is "semi-autoregressive": cheap dependence between neighbouring draft tokens.
  - A **confidence head** that truncates the block to its confident prefix before verification (`_confident_prefix_length(..., threshold)`) [code `deepspec/eval/dspark/draft_ops.py:117-131`]. Shorter proposals waste less verify compute when the draft is unsure.
- Results (Table 1) are in the paper (arXiv 2607.05147), **not in the repo**. The README only lists checkpoints [doc `DeepSpec/README.md:55-62`].
- Training needs a "target cache" of about **38 TB** for Qwen3-4B [doc `DeepSpec/README.md:29`]. It's heavy, and not something to replicate.

### 7.5 MTP: DeepSeek's own speculative decoding in production

See §4.4. The decode trace contains an MTP-shaped layer after sampling, and 2 tokens per request go through the main model. MTP is a *built-in* draft: one extra transformer layer that shares the embedding and head [doc `DeepSeek-V3/README_WEIGHTS.md:43-48`]. That makes it the k = 1 case of §7.1. With k = 1 there is at most 1 bonus token, so the gain is ≤ 2×, and about 1.85× at a = 0.85 [inference].

### 7.6 Where it fits after v1 [inference]

- **Natural first project:**
  - Use stories15M as the draft for stories110M. llama2.c ships 15M/42M/110M (CLAUDE.md), and they share the 32k tokenizer.
  - The target is about 7× bigger, so c ≈ 0.14 per draft token. The speedup is bounded by acceptance.
  - It's measurable with the M4 metrics: TTFT is unchanged; decode tok/s should rise.
- **Engine requirements it creates:**
  1. A forward over k+1 positions with a KV cache. This is the M6 batched-prefill code path, reused.
  2. KV rollback, `crop` (§3.5).
  3. Two models resident at once (memory accounting).
  4. Sampling with explicit probabilities for the rejection rule. That touches role project 1.
- **In the server:** it interacts with batching. Per-request acceptance varies, so batch rows commit different numbers of tokens per step. That is a real scheduler problem, and it's worth a decision entry when it arrives.

---

## 8. KV cache on disk: 3FS and prefix caching

### 8.1 3FS in five lines

- Four components: cluster manager, metadata service, storage service, client. All are on RDMA [doc `3FS/docs/design_notes.md:5`].
- Heartbeats go to the cluster manager, which has a primary/standby [:7].
- Metadata services are stateless, over FoundationDB [:9, 63].
- Storage uses CRAQ, **write-all-read-any**: writes go head→tail along a chain, and reads go to any replica [:11, 105, 151].
- The native client is `io_uring`-style and zero-copy (`Iov` shared memory, `Ior` ring) and batches small reads [:43-49].
- From the README: "Disaggregated Architecture ... locality-oblivious"; "KVCache for Inference: Provides a cost-effective alternative to DRAM-based caching, offering high throughput and significantly larger capacity" [doc `3FS/README.md:9,17`].

### 8.2 Headline numbers

| Number | Source |
|---|---|
| 6.6 TiB/s aggregate read: 180 storage nodes (2×200 Gbps IB, 16 × 14 TiB NVMe each), 500+ clients, with background training traffic | [doc `3FS/README.md:30`; OII:80] |
| GraySort: 110.5 TiB in 30 min 14 s = 3.66 TiB/min, on 25 storage + 50 compute nodes | [doc `3FS/README.md:40`; OII:81] |
| KVCache read: peak up to 40 GiB/s per client node, 1×400 Gbps NIC | [doc `3FS/README.md:48`; OII:82 says "40+ GiB/s peak throughput per client node for KVCache lookup"] |
| Average KVCache read in the same chart ≈ 2–4 GiB/s per client (read by eye) | [fig `3FS/docs/images/kvcache_read_throughput.png`] |
| GC (removal) IOPS: periodic bursts up to ~1.0–1.4 MIOPS, about once a minute, near zero in between (by eye) | [fig `3FS/docs/images/kvcache_gc_iops.png`] |

Readings [inference]:
- 40 GiB/s is about 86% of a 400 Gb/s NIC (46.6 GiB/s).
- The **peak-to-average ratio is about 10×**. KV loads are bursty, because a long cached prompt arrives all at once. Provisioning for the average fails the TTFT SLO.
- Eviction runs in **periodic batches**, not continuously. That's a design choice worth copying: evict with hysteresis rather than per-insert.

### 8.3 Connection to prefix caching and API prompt caching [inference, except where cited]

- D6 counts **"on-disk KV cache"** hits: 56.3% of input tokens [doc D6:74]. The diagram labels the store "External KVCache Storage (Optional)" [fig D6 diagram].
- So there's a tier below GPU HBM: HBM (hot) → host DRAM (maybe) → shared NVMe via 3FS (warm, fleet-wide).
- Fleet-wide sharing is the point. Any prefill unit can hit a prefix computed by any other, so routing doesn't need prefix affinity to get hits. That is a different trade-off from single-node vLLM prefix caching, where affinity matters.
- vLLM's in-memory prefix caching and its paged KV design are covered in its docs: `~/refs/inference/vllm/docs/design/prefix_caching.md`, `paged_attention.md`, `hybrid_kv_cache_manager.md`. FlashMLA's paged KV uses block size 64 [doc OII:39].
- **API prompt caching** (DeepSeek, Anthropic, others) exposes this economics to users: cached input is billed cheaper. DeepSeek's ratio is 3.9× [doc D6:80], while the cost ratio is about 17× (§6.5).
- Don't quote other providers' prices from memory.

Scaled down for Luigi's box:
- The "external KV storage" can be host RAM, or an NVMe file of KV blocks keyed by a hash of the token prefix.
- For stories15M, one full 256-token context is 3.5 MB. The experiment is about **TTFT with and without a hit**, and about eviction policy, not bandwidth.

---

## 9. Fault tolerance (role project 5)

### 9.1 What the docs say, and don't

| Topic | What exists | Source |
|---|---|---|
| Inference worker failure, KV loss, rescheduling | **Not documented** anywhere in these repos | (absence) |
| Elastic capacity | Nodes are drained at night for research/training and added back by day | [doc D6:65-67; fig node count] |
| KV durability | External KV store: prefill reads and writes, decode writes (arrows in the figure). This implies a lost decode worker's context *could* be partly recovered from storage instead of fully re-prefilled [inference] | [fig D6 diagram] |
| Failure detection | Heartbeats as **leases**: declared failed after T s without a heartbeat; a service that can't reach the manager for T/2 s **stops serving and exits itself** | [doc `3FS/docs/design_notes.md:177`] |
| Stateless failover | Clients fail over to another metadata service on failure or timeout | [doc `design_notes.md:63,179`] |
| Membership / versioning | Chain version incremented on every change; a stale-version request is rejected; an offline target moves to the end of the chain | [doc `design_notes.md:122,155,205`] |
| In-flight request survival | A write forwarded to a successor that dies is re-forwarded to the new successor once the chain table updates; the receiver may reject until it catches up | [doc `design_notes.md:167`] |
| Self-fencing | "If a storage service finds public state of any local storage target is lastsrv or offline, it exits immediately" (network partition) | [doc `design_notes.md:207`] |
| Load balance **during** recovery | Spread a failed node's read traffic over *all* peers (chain tables built as a balanced incomplete block design, solved by an integer program) instead of 2 neighbours | [doc `design_notes.md:128-147`] |
| Recovery overlapping normal traffic | Recovery streams full-chunk writes while serving | [doc `design_notes.md:232-240`] |
| Degraded mode | LPLB falls back to a 50/50 split when the LP fails its feasibility check | [code `LPLB/.../minilp.cu:395-397`] |
| Stragglers | Slow-rank localization tool listed by DeepEP | [doc `DeepEP/README.md:510`] |

### 9.2 Role project 5 design notes (options for Luigi; don't decide for him) [inference]

Scenario: a worker dies mid-request, so its KV cache is lost. Requeue and redo prefill.

1. **Detection.**
   - A lease/heartbeat from each worker to the router (the 3FS pattern).
   - The worker **self-fences**: it stops emitting tokens if it can't renew, so no "zombie" keeps streaming after the router has reassigned its requests.
   - Trade-off: a short T means fast recovery but false positives under GC or CPU stalls. That links to role project 6 (latency spikes can look like deaths).
2. **What to replay.** If the client has already received m generated tokens, the replacement must prefill **prompt + those m tokens**, then continue from token m+1.
   - It must not regenerate them. Regenerating would change text the user already saw, unless sampling is seeded per (request, position).
   - So the router must keep the tokens it has already streamed per request. That makes the router stateful, which is a design decision.
3. **Cost of recovery** = TTFT(prompt + m). The M4 metric gives the recovery-latency budget directly.
   - With a KV tier (§8), recovery = load KV (cheap) instead of recompute.
   - That is the DeepSeek-style answer to "KV is lost": make it not the only copy.
4. **Blast radius.** In DeepSeek's EP144, one GPU death stalls the whole 144-GPU unit (synchronous all-to-all). On Luigi's box, workers are independent, so one death affects only its own requests.
   - That's a good interview talking point: parallelism choice determines failure domain.
5. **Rebalancing after failure.** The surviving worker(s) inherit the load. Do what 3FS does: spread it evenly and admit less (backpressure) rather than overload one survivor.
6. **Testing.** Kill a worker process during a streaming request. Assert:
   - the final text equals the no-failure run at temperature 0 (the D1-style oracle);
   - the extra latency ≈ the predicted TTFT(prompt + m).

---

## 10. Mapping to the interview system-design questions

### 10.1 "High-concurrency LLM inference API with request batching"

What DeepSeek adds beyond textbook continuous batching:
- **Separate pools** for prefill and decode, each sized from its own bottleneck: FLOPs vs KV capacity and bandwidth (§3.3).
- **Balancers with explicit two-part objectives** per phase (§5.2–5.3). Not round-robin.
- **Prefix cache as a fleet-wide tier** (56.3% hits), with pricing that passes the savings on (§6.5, §8).
- **Elastic capacity** following diurnal load, scaled in whole units (§2.5).
- **Speculative decoding (MTP)** to raise per-user speed (§6.4).
- **Metrics:** per-request speed (TPOT) vs per-node throughput, reported separately (§2.4).
- The back-of-envelope in §6.2 is a model for the interview: throughput per node × node count = total tokens. Check it both ways.

### 10.2 "Deploy a 500 GB model to 100–1000 GPU workers" [inference, assuming each worker needs a full copy and a 50 GB/s link per node]

| Scheme | Time for N = 1000 | Notes |
|---|---|---|
| Star from one source | N × S / B = 1000 × 500 GB / 50 GB/s = **10,000 s** | The source NIC is the bottleneck |
| Binary tree, whole file per hop | 2S/B per level × ~10 levels ≈ **200 s** | Each node sends twice |
| Binary tree, chunked + pipelined | ≈ 2S/B + depth × chunk time ≈ **20 s** | |
| **Chain (pipeline), chunked** (64 MB chunks) | S/B + (N−1) × chunk/B ≈ 10 + 1.3 ≈ **11.3 s** | Every node receives once and sends once. Near optimal. Fill cost = (N−1) × chunk time, the same shape as a pipeline bubble [DualPipe/README.md:26-36] |
| Everyone reads a parallel FS (3FS-like) | N × S / aggregate = 500 TB / 6.6 TiB/s ≈ **69 s** | Aggregate storage bandwidth is the limit [3FS/README.md:30] |

**Same table with the interview's actual parameters** (added by the lead Claude after review; the table above assumes 50 GB/s, but the reported question gives **10 Gbps external and 10 Gbps per worker**, i.e. B = 1.25 GB/s, so S/B = 500 GB / 1.25 GB/s = **400 s**). [inference]

| Scheme (N = 1000, 64 MB chunks, chunk time ≈ 0.05 s) | Time |
|---|---|
| Lower bound: every worker must receive S at B | **400 s** (the external download alone is also 400 s) |
| Star from one source | 1000 × 400 s = **400,000 s ≈ 4.6 days** |
| Binary tree, whole file per hop | 2 × 400 s × ~10 levels ≈ **8,000 s** |
| Binary tree, chunked + pipelined | ≈ 2 × 400 s + 10 × 0.05 s ≈ **800 s** (each node's uplink carries 2S) |
| **Chain, chunked** (starts forwarding as soon as chunks arrive from outside) | ≈ 400 s + 999 × 0.05 s ≈ **450 s**, about 1.1× the lower bound |

With these numbers, the chain's fill cost (about 50 s) is not negligible, which is why chunk size matters in the interview discussion. Several parallel chains from the source (source bandwidth permitting) or a k-ary tree with chunking trade fill cost against uplink load. At 10k workers the chain's fill cost becomes about 500 s, so switch to several chains or a pipelined tree.

Failures:
- A chain is fragile. One dead node stalls everyone downstream.
- The 3FS answer is exactly CRAQ's: a versioned chain membership, move the dead node to the end, re-forward the in-flight chunk to the new successor, and the receiver rejects stale versions [doc `3FS/docs/design_notes.md:122,167,205`].
- Resume by chunk index. Chunk hashes give integrity.
- Topology-aware: send one copy per node over the network, then fan out inside the node over NVLink/PCIe. This is EPLB's "node first, then GPU" hierarchy, and DeepEP normal's "IB then NVL" forwarding (`DeepEP/figures/normal.png`).

### 10.3 "Distributed mode lookup under a bandwidth limit"

LPLB's workload all-reduce is the pattern [code `minilp.cu:110-176`]:
- Never ship raw items. Ship a **histogram** (one counter per key: 256 floats ≈ 1 KB).
- Reduce **hierarchically**: across nodes over the slow network, within the node over the fast links.
- Then take the argmax.

When the key space is too big for a full histogram:
- Each node sends only its local top-K with counts, plus an upper bound on any unsent count (its K-th count).
- Or use heavy-hitter sketches: Misra-Gries, count-min.
- A second round verifies the candidates.

[inference: standard techniques, not in these repos.] EPLB's "moving average of historical statistics" [doc `EPLB/README.md:12-13`] is the streaming variant: maintain decayed counts rather than recomputing.

---

## 11. Post-v1 server roadmap: DeepSeek's ideas scaled to one box (options, not decisions) [inference]

Building blocks, roughly in dependency order. Each row says what it demonstrates, so Luigi can choose the order himself.

| # | Building block | DeepSeek / vLLM reference | Options to present | Evidence it produces | Role |
|---|---|---|---|---|---|
| S1 | Single-worker server: request queue, iteration-level (continuous) batching, streaming | vLLM base of DeepSeek's engine [OpenSourcing README:12-13] | Scheduler policy: FCFS vs shortest-prompt-first vs priority; max batch by tokens vs by requests | TTFT / TPOT / throughput vs offered load; p50/p99 | 4, 1 |
| S2 | Paged KV cache | FlashMLA block 64 [OII:39]; vLLM `paged_attention.md` | Block size (16/32/64); allocation on admission vs on demand; preemption by recompute vs swap | Max concurrent requests at fixed memory; fragmentation % | 4 |
| S3 | Prefix cache + a lower KV tier | D6:74; 3FS KVCache [3FS/README.md:45-51] | Tier = host RAM vs NVMe file; hash granularity = block; eviction LRU vs batched GC (§8.2) | Hit rate; TTFT hit vs miss; recompute-vs-load ratio (§6.5 at his scale) | 4 |
| S4 | Second worker (CPU and GPU) + router | D6 prefill/decode LBs; EPLB/LPLB (§5.6) | Round-robin → least-loaded → cost model (ΣL, ΣL², KV-free, slots-free) → prefix affinity; static vs per-request | balance = max/mean (LPLB script); p99 under skewed prompt-length mixes | **3** |
| S5 | PD disaggregation on one box | D6:21; vLLM `disagg_prefill.md` | GPU prefill → KV over PCIe → CPU or GPU decode; vs chunked prefill in one worker | Tail ITL with long prompts mixed in, with and without; KV handoff time (tiny for 15M: 13.8 KB/token) | 3, 4 |
| S6 | Overlap inside a worker | Dual micro-batch [D6:26-33]; vLLM DBO | Overlap CPU scheduling/sampling with GPU forward; overlap KV copies with compute on a second stream | nsys: hidden-copy fraction (§4.5); idle % (§4.6) | 2, 4 |
| S7 | Fault tolerance | 3FS leases/versions/balanced recovery (§9) | Router-held generated tokens vs client-held; recompute vs KV-tier restore; T (lease) size | Kill-test: correctness at T = 0 + recovery latency vs predicted TTFT(prompt + m) | **5** |
| S8 | Speculative decoding | DeepSpec; MTP (§7) | Draft = stories15M → target 110M; k fixed vs confidence-scheduled (DSpark idea); per-request vs batch-level | acceptance_length, verify_rate (DeepSpec definitions); speedup vs batch size | 1, 4 |
| S9 | Elasticity + containers | D6:65-67 day/night scaling | Worker autoscaling under a load generator; container CPU limits; `perf` on p99 spikes | p99 over time; spike root causes | **6** |

Dependencies worth pointing out:
- S2 before S3: prefix caching is naturally block-granular.
- S1 metrics before everything else.
- S4's cost model needs M6/M7 measurements of latency vs batch and vs context.
- S7 is much nicer with S3 (restore instead of recompute).

What **not** to copy at this scale:
- EP, DeepEP and all-to-all (dense model, one GPU).
- LP solvers for two workers.
- Disk-scale 3FS. A file or RAM tier teaches the same lesson.

---

## 12. Milestone and role-project map

| Milestone | Use from this note | Role project |
|---|---|---|
| M0 (perf model) | §3.1–3.2 roofline, AI = 2T/b; §6.2 "check the numbers both ways" | 4 |
| M3 (KV cache) | §3.3 KV bytes per token formula; §3.5 note that spec decode will need `crop` | (4) |
| M4 (prefill/decode, TTFT, tok/s) | §3.5 DeepSpec's prefill→decode loop; per-request vs aggregate metrics; TTFT = queue + prefill + sample | 4, 1 |
| M5 (quantization) | FP8 matmul/dispatch vs BF16 attention/combine split [D6:62-63]: precision chosen per op | 2 |
| M6 (make it fast, batched prefill) | §3.2 prefill becomes compute-bound at P ≳ 60 on CPU; sampling share (§4.4: 1.8% at 671B) | 1, 4 |
| M7 (GPU, nsys) | §4.4–4.6: trace structure, overlap measurement, checklist | 2, 4 |
| M8 (real model) | §3.3 MoE/MLA contrast if Luigi picks an MoE-class model; KV/token formula | 4 |
| Post-v1 server | §5 (LB), §8 (prefix cache), §9 (faults), §11 (roadmap) | 3, 5, 6 |
| Interviews | §10 | all |

---

## 13. Quiz questions (with answer keys for the tutor)

1. **Decode at batch 1 on stories15M reads 60 MB and does 30 MFLOP. What is its arithmetic intensity, and what does that tell you?**
   0.5 FLOP/B. Far below any CPU/GPU ridge, so once the math is fast it's memory-bound.
2. **Why does DeepSeek use EP144 for decode but only EP32 for prefill?**
   Decode needs many GPUs' requests pooled so each expert gets about 512 tokens per micro-batch (vs 32 in one node); it also frees HBM for KV. Prefill already has about 7k tokens per expert per micro-batch in a 4-node unit.
3. **vLLM says PD disaggregation "does not improve throughput". DeepSeek uses it anyway. Reconcile the two.**
   vLLM's benefit is latency isolation. DeepSeek's is choosing different parallelism per phase for an MoE model with EP, which is a throughput win in that setting.
4. **In the prefill figure, communication gets 24 SMs. In decode it gets 0. Why the difference?**
   Prefill messages are huge and bandwidth-bound, handled by chunked IB→NVLink forwarding kernels that need SMs. Decode messages are tiny and latency-bound, so the SM-free RDMA issue/receive-hook design lets compute keep all 132 SMs.
5. **The decode trace has everything on one stream. How can there be any overlap?**
   Overlap happens on the NIC. The send kernel posts RDMA and returns, the other micro-batch's compute runs, then the receive kernel waits. The 65–86 µs receive kernels are the part that isn't hidden.
6. **Prefill LB balances both "input tokens" and "core-attention compute". Give two GPUs with equal tokens but 8× different attention.**
   One 8K prompt vs eight 1K prompts: L²/2 = 33.6M vs 4.2M pairs.
7. **Walk EPLB's hierarchical policy on the README example. Why is node1 already over the ideal before any replication?**
   Groups sum to 446 vs 587. Node1's GPUs average 146.8 vs the ideal 129.1. Hierarchy caps the achievable balance.
8. **`replicate_experts` gives each new slot to argmax(w/count). Why does that minimize the max per-replica load?**
   Each step reduces the current bottleneck. A classic greedy for separable min-max.
9. **What does LPLB optimize, and name one stated limitation that also bites a request router.**
   Min max rank load via token fractions over replica edges. It balances token *counts*, but real cost is non-linear. The router equivalent: tokens ≠ attention cost; KV length matters.
10. **Check D6's numbers: from 608B input, 168B output, 73.7k and 14.8k tok/s per node, derive the average node count.**
    7.04M / 73.7k + 1.94M / 14.8k ≈ 95.5 + 131.4 ≈ 226.9 vs 226.75.
11. **Why is a cache hit about 17× cheaper than a miss for DeepSeek, yet priced only 3.9× cheaper?**
    Load about 14 µs vs recompute about 0.25 ms per token. Pricing is a business choice, and storage/bandwidth isn't free.
12. **Verification of k draft tokens costs "about one decode step". Under what condition is that false, and does it hold in M4?**
    It's false when compute-bound. In naive M4 (compute-bound), k+1 positions cost about (k+1)× as much, so spec decoding gains nothing until M6 makes decode memory-bound.
13. **α = 0.8, k = 4: expected tokens per verify?**
    (1 − 0.8⁵)/(1 − 0.8) = 3.36.
14. **A worker dies after streaming 50 tokens of a reply. What must the replacement prefill, and why not just resample?**
    Prompt + the 50 streamed tokens. Resampling could change text the user already saw, unless sampling is deterministic per position.
15. **3FS makes a service exit if it can't reach the manager for T/2. Why T/2 and not T?**
    So the node fences itself *before* the manager declares it dead at T and reassigns its work. No two owners at once.
16. **Chain vs tree to broadcast 500 GB to 1000 nodes: which is faster, and what is the chain's weakness?**
    Chunked chain ≈ S/B + (N−1)·chunk/B ≈ 11 s. It's fragile: any failure stalls downstream, so it needs versioned re-linking (CRAQ).

---

## 14. Open questions (not answerable from these repos)

1. **TTFT and p99 latency** of DeepSeek's service: not published. Only mean output speed (20–22 tok/s) is [D6:75].
2. **How KV moves from prefill to decode:** direct RDMA, or via the external store? The diagram shows only Prefill → Decode LB and the storage arrows.
3. **Does the 20–22 tok/s include MTP?** The trace strongly suggests MTP runs every decode step (§4.4). The acceptance rate must come from the V3 technical report, not from memory.
4. **How often EPLB re-plans, and how weights move when it does.** D6 and EPLB are silent. DeepEP V2.5's `lb_prefetch_weights` is a per-batch mechanism, not EPLB's.
5. **The decode LB algorithm itself.** Only the objectives are given [D6:46-50]. Same for the prefill LB.
6. **Failure handling in the inference engine:** undocumented. Everything in §9.2 is inference from 3FS's design.
7. **On-disk KV format and eviction policy.** Only the GC IOPS chart hints at periodic batch deletion. §6.5 assumes BF16 MLA latents.
8. **Why production computed-prefill throughput is about 52% of the profiled rate** (§6.3). This needs load-over-time data that isn't published.
9. **The decode profile uses EP128 while production uses EP144** [PDR:30 vs D6:23]. The per-step numbers in §6.4 are an approximation to production, not a measurement of it.
10. **DeepEP V1 numbers** (latency/bandwidth tables) are not in this V2.5 checkout. If the tutor wants them, clone an older tag rather than quoting from memory.
