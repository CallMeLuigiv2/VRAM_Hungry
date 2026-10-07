# 04: DeepSelect + TileKernels, top-k/sampling and quantization kernels (tutor notes)

These are private reference notes for the tutor. They are not learner-facing and not engine code.

| Item | Value |
|---|---|
| Repo 1 | `~/refs/inference/deepseek/DeepSelect/` @ `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18` ("Add Kernels for Huawei Ascend NPU (#23)", 2026-09-30) |
| Repo 2 | `~/refs/inference/deepseek/TileKernels/` @ `66258df6175d2f630ffecb04c5ab66bff8a2ae6a` ("Merge pull request #34 … v2.0.0", 2026-09-30) |
| Cross-refs | llama2.c @ `350e04fe35433e6d2941dce5a1f53308f87058eb`; vLLM @ `22bbe3f1023a68a7d1f2de566dd4242fdcfd36c3`; DeepSeek-V3 @ `9b4e9788e4a3a731f7567338ed15d3ec549ce03b` |
| Written | 2026-09-30 |
| Machine facts used | Measured here: RTX 4070 = 46 SMs, 100 KiB shared memory per SM, **99 KiB per block (opt-in)**, 36 MiB L2 (`torch.cuda.get_device_properties`). The CPU is a Ryzen 9 5900X (Zen 3): 12C/24T, 32 KiB L1d and 512 KiB L2 per core, 32 MiB L3, with AVX2, FMA, BMI2 and POPCNT (`lscpu`, `/proc/cpuinfo`). |

**Labels.** `[code]` / `[doc]` = verified in the file:line cited. `[inference]` = my reasoning or estimate; nothing measured. `[external]` = general knowledge that is not in these repos; verify before teaching it as fact. Applicability tags: **CPU** (the Rust engine, M4–M6), **sm_89** (M7 CUDA on the 4070), **neither** (Blackwell/Hopper-only or training-only).

## Summary (the 10 things that matter)

1. **DeepSelect's algorithm is not "radix select over the whole row".** It is a **threshold filter + small candidate buffer + periodic radix-select compaction**. It streams the row once in blocks visited in a pseudo-random order. Each block keeps only elements `> T`, where `T` is the current k-th largest. When the buffer grows past `k + B2`, it runs an 8-bit-digit radix select in shared memory, which cuts the buffer back to k and raises `T` [doc `docs/DeepSelect-deep-dive.md:9-39`]. The expected compaction work is `O((1+k/B2)(k+B+B2)·ln(N/B))`, far below N [doc :86-102].
2. **Ties are resolved by an "equal quota".** The kernel counts `> pivot` and `== pivot` per thread, then prefix-sums both in one packed u32. The first `k − #gt` equal elements *in buffer order* win [code `common_parts.cuh:477-504`]. The buffer order follows the permuted scan order, so **the lowest index does not necessarily win** (`v3_fp32/topk_select.cuh:6`). TileKernels' MoE top-k is the opposite: it breaks ties to the smallest index and is tested bitwise against a stable sort.
3. **Sampling scenario**: fp32, vocab ≈ 128K (DeepSeek's 129,280), k ≤ 4096, benchmarked with k=512, `sorted=True`, and values returned [code `tests/test.py:249-254`]. A 129,280-entry fp32 row is **505 KiB**. Top-k is memory-bound, so the metric is **effective bandwidth** [doc README:43-44].
4. **Headline numbers** [doc]: "2 ~ 20x speedup compared to vanilla `torch.topk`" (README:3), and `return_value=False` is "~10% faster" (README:91). Values read off the fp32 sampler plot (≈, GPU not named, the build targets sm_100a/sm_103a) are 2.3–2.8 TB/s vs 0.6–1.15 TB/s for torch at batch 256–4096, and only **~0.1 TB/s at batch 6**, which is latency-bound.
5. **llama2.c top-p, verified** (`run.c:624-665`): a cutoff filter `p ≥ (1−topp)/(n−1)`, then `qsort` of the survivors, then a cumulative sum up to `> topp`, then an inverse-CDF sample scaled by `coin*cumulative_prob`. The filter is safe (proof in §3.3). The cost is dominated by `n0·log n0`, where `n0` depends on how peaked the distribution is, **so measure `n0`**.
6. **CPU mapping (M6)**: DeepSelect's per-thread `hit_mask` + `__ffs` + `mask &= mask-1` translates almost literally to AVX2. The chain is `_mm256_cmp_ps` → `_mm256_movemask_ps` → `trailing_zeros` → `m & (m-1)`. The same threshold-filter idea turns top-p into "filter, then partial-select by cumulative mass" instead of a full sort. Sampling's share of per-token time is small now but grows after M6 speeds up the matmuls (Amdahl). Measure it before and after.
7. **sm_89 portability**: DeepSelect as shipped does not run on the 4070. It is built only for `sm_100a/sm_103a` (`setup.py:119-127`). It uses TMA, mbarrier transaction counts, `elect.sync`, and clusters (sm_90+), and its fp32 shared-memory plan is ~203 KiB against 99 KiB available. **The algorithm ports; the plumbing does not.** A reduced config (~91 KiB, see §2.6 and §6) plus `cp.async` or plain LDG.128 is the Ada path.
8. **TileKernels quantization** uses FP8 **E4M3** (max 448) and FP4 **E2M1** (max 6, two per byte). There is **no E5M2** anywhere. The scale formula is `sf = max(amax, clamp_min)/max_q` with `sf_inv = max_q/amax` computed separately (`quant/common.py:251-282`). Scales can be stored as fp32, UE8M0 (a power-of-two exponent byte, rounded **up** via a bit trick), or E4M3. Groupings are per-token `(1, 16|32|64|128)`, per-block `(32|128, 32|128)`, and per-channel `(32|128, 1)`. All of it is symmetric, with no zero points.
9. **The M5 bridge**: llama2.c's `runq.c` Q8_0 is exactly TileKernels' "per-token cast" with int8 instead of FP8. It has two traps. First, `runq.c:167` uses C `round()` (ties away from zero) while `export.py:62` uses `torch.round` (ties to even), so bitwise tests disagree on ties. Second, there is no zero-amax guard (`runq.c:161,166`), so an all-zero group gives `x/0`. TileKernels guards this with `clamp_min` (`config.py:10-15`).
10. **Testing lessons**: DeepSelect checks **properties** (range, uniqueness, value==input[idx], `min(selected) ≥ max(unselected)`, sortedness), not index equality with `torch.topk`, because ties make index equality wrong (`tests/test.py:80-147`). TileKernels checks **bitwise equality** against a PyTorch reference that mirrors the kernel's arithmetic order, plus a **statistical rounding-bias test** (`testing/numeric.py:41-64`). It has test levels (`TK_TEST_LEVEL` 0/1/2) and benchmarks gated behind `--run-benchmark`, with a JSONL baseline and a 5%/0.8 µs regression gate.

---

## 1. Repo maps

### 1.1 DeepSelect

```
README.md                         scenarios, recommendations, perf plots, usage
docs/DeepSelect-deep-dive.md      algorithm + expected-cost proof + implementation notes (114 lines; links point to older commit 8e70df71, line numbers differ from HEAD)
assets/perf_*.png                 bar charts (effective TB/s vs torch.topk)
deep_select/interface.py          Python API `topk(...)` (:34-109), output stride alignment (:78-86)
csrc/api.cpp                      torch binding + arg checks (:56-67) + config dispatch (:145-221)
csrc/structs.h                    TopkSelectArgs; INPUT stride align 1024 B, MAX_VOCAB 2^23 (:17-22)
csrc/cuda_kernels/config.h        TopkSelectConfig template (B = elements_per_round, B2 = reconstruct_threshold)
csrc/cuda_kernels/bit_utils.cuh   distort/un_distort: float total order -> uint order (:22-88)
csrc/cuda_kernels/utils.cuh       warp prefix/suffix sums via shfl.sync (:5-54)
csrc/cuda_kernels/common_parts.cuh  the real engine (1517 lines): epilogue, smem plan, perm, pivot search,
                                    census/eq-quota, TMA issue, bf16 radix passes, scan_segs main loop
csrc/cuda_kernels/v3/             bf16 "normal" kernel (one CTA per row)
csrc/cuda_kernels/v3_fp32/        fp32 kernel = the SAMPLER path (603 lines, self-contained main loop)
csrc/cuda_kernels/v3_cluster/     bf16 small-batch kernel: 16-CTA cluster per row
csrc/cuda_kernels/*/instantiations/  one .cu per template config (generated by scripts/generate_instantiations.py)
csrc/ascend_kernels/              same algorithm on Ascend SIMD vector units (kernel.asc, 946 lines)
tests/test.py, tests/lib.py       correctness sweep + perf cases; input distributions
tests/kernelkit/                  bench (CUPTI via torch.profiler), compare, build-time spill check
```

### 1.2 TileKernels

```
tile_kernels/<area>/<op>_kernel.py   Python API: arg checks, config, allocation, backend dispatch
tile_kernels/<area>/<op>_cuda.py     TileLang CUDA kernel (@tilelang.jit)
tile_kernels/<area>/<op>_asc.py      TileLang Ascend kernel
tile_kernels/torch/<op>.py           PyTorch REFERENCE implementations (the oracles)
tile_kernels/quant/                  per_token / per_block / per_channel cast, cast_back, swiglu(+quant), norm(+quant), common.py
tile_kernels/moe/                    topk_gate (plain top-k), moe_topk_gate_forward/backward (routing), normalize_weight
tile_kernels/transform/              RoPE
tile_kernels/rand/                   randn (TileLang kernel is Ascend-only; CUDA path = torch.randn, randn_kernel.py:31-32)
tile_kernels/config.py               runtime knobs incl. amax clamp minimums (:10-15)
tile_kernels/testing/                numeric checks, generators, test levels, pytest plugins (benchmark, seeds, gpu mem, precompile)
tests/<area>/test_<op>.py            correctness + @pytest.mark.benchmark tests
```

Requirements: SM90 or SM100, CUDA 13.1+, TileLang 0.1.15+ (README:22-28). TileLang is **not installed** on this machine (`ModuleNotFoundError`). Treat it as a reading reference: **neither** directly runnable on sm_89 as-is [inference: some memory-bound kernels might compile for sm_89, but it is unsupported].

---

## 2. DeepSelect deep dive

### 2.1 What and why
TopK is used by DeepSeek Sparse Attention's "lightning indexer" (bf16, k=512 in V4 and 2048 in V3.2-Exp; see `DeepSeek-V3.2-Exp/inference/model.py:90`) and by the **sampler** (fp32, V≈128K) [doc README:3, 17-37]. `torch.topk` is a general algorithm. DeepSelect specializes for the case **k ≪ N** with the input read exactly once.

### 2.2 The algorithm as documented [doc `DeepSelect-deep-dive.md`]
```
DeepSelectTopk(x[0:N), k, B, B2):                           # :17-39
    topk_candidate = []            # shared memory
    T = -inf                       # current k-th largest; never decreases
    p = random_permutation(ceil(N/B))
    for block in p:
        for j in block: if x[j] > T: topk_candidate.append((x[j], j))   # Filter
        if len(topk_candidate) >= k + B2:                                # Compact
            topk_candidate = RadixSelectTopK(topk_candidate, k)
            T = min(v for v,_ in topk_candidate)
    return RadixSelectTopK(topk_candidate, k)
```
- Invariants: `len(buffer) ≤ k + B2 + B`. Every element is read once, in contiguous blocks. Extra space is O(k+B+B2), so it fits in shared memory [doc :45-49].
- Why the random order: with a sorted-ascending input and a linear scan, every element beats the threshold and the filter never filters. A random block order makes the expected append count at step i at most `L/i` [doc :69-76]. That gives `E[W] ≤ (1 + k/B2)·(k+B+B2)·H_m`, with `m = ceil(N/B)` and `H_m` the harmonic number [doc :86-92]. With `B, B2 = Θ(k)`, the extra work is `O(k log(N/k))` [doc :102].
- Suggested config for k=512: `B = B2 = 1024` [doc :15]. The shipped fp32 config is `B = 8192, B2 = 4096` (`api.cpp:201-205`; `TopkSelectConfig<…, MAX_TOPK, NUM_THREADS, 1, B, 4096, 3>`, where the positional args are occupancy, elements_per_round=B, reconstruct_threshold=B2, tma_depth; see `config.h:5-27`).

Bound, evaluated [inference, arithmetic from the doc's formula]:

| N | k | B | B2 | m | H_m | E[W] ≤ | W/N |
|---|---|---|---|---|---|---|---|
| 129,280 | 512 | 8192 | 4096 | 16 | 3.38 | 48.7K | 0.38 |
| 129,280 | 512 | 1024 | 1024 | 127 | 5.43 | 20.8K | 0.16 |
| 32,000 | 40 | 256 | 256 | 125 | 5.41 | 3.5K | 0.11 |

W counts elements fed to the in-shared-memory radix select; the one global read of N is extra. It is an upper bound under a truly uniform permutation.

**Teaching hook A (the filter, tiny).** Use N=16 byte keys, k=3, B=4, and compaction when `len ≥ k+B2 = 6` (B2=3).
Keys by index: `0:3A 1:7F 2:12 3:7C | 4:55 5:7F 6:08 7:61 | 8:7A 9:2E 10:44 11:19 | 12:70 13:03 14:5B 15:66`. Visit order: blocks `[2, 0, 3, 1]`.
- Block 2: T=−∞, all 4 appended → buffer {7A,2E,44,19} (size 4).
- Block 0: all 4 appended → size 8 ≥ 6 → compact to top-3 {7F,7C,7A}, T=7A.
- Block 3: 70, 03, 5B, 66 are all ≤ 7A → **0 appended**.
- Block 1: only 7F > 7A → buffer {7F,7C,7A,7F}.
- End: select top-3 → {7F(1), 7F(5), 7C(3)}.
Ask the learner: which visit order makes this algorithm worst? (Ascending blocks: every block beats T.) Then ask what the random order buys. (The expected number of appends decays like 1/i.)

### 2.3 Radix select inside the buffer (the "RadixSelectTopK")

**Step 0: make floats sortable as unsigned integers ("distort")** [code `bit_utils.cuh:22-48`]. For positives, flip the sign bit. For negatives, flip all bits. Examples: `-inf 0xFF800000→0x007FFFFF < -1.0 0xBF800000→0x407FFFFF < -0.0 →0x7FFFFFFF < +0.0 →0x80000000 < +1.0 0x3F800000→0xBF800000 < +inf →0xFF800000`. `un_distort` is its inverse (:72-88).
*Rust hook:* `f32::total_cmp` implements IEEE totalOrder with the same bit trick. The learner can build radix keys from `to_bits()` with this map.
*Pitfall [inference]:* for fp32 the **top byte is sign + the 7 high exponent bits**, so one top-byte bucket spans a factor of 4 in magnitude: the buckets are exponent pairs, i.e. [0.5,2), [2,8), [8,32), and so on. The first 8-bit pass therefore barely separates the top logits; passes 2–3 (the remaining exponent bit plus the mantissa) do the work. DeepSelect handles fp32 with up to **4 byte passes plus early exit** (`v3_fp32/topk_select.cuh:164-196`), and bf16 with exactly 2 (`common_parts.cuh:1101-1129`).

**Step 1: histogram of the current digit.** There are 256 buckets plus a "sink" slot. Elements whose prefix does not match the pivot prefix are counted into the sink rather than skipped, so the compiler emits uniform `ATOMS`/`red.shared.add` [code `v3_fp32:170-180`; comment :171]. The layout has 2 rows × 260 slots: 256 buckets + sink + 3 pad to keep 16 B alignment (`common_parts.cuh:364-370`).

**Step 2: find the pivot bucket** [code `common_parts.cuh:525-582`]. One warp handles it: each lane loads 8 counters, and a warp exclusive **suffix** sum (`utils.cuh:30-54`) gives `S[j]` = the number of elements in buckets ≥ j. The owning lane satisfies `S[8·lane+8] < k ≤ S[8·lane]`. The pivot bucket j is the largest j with `S[j] ≥ k`. The count still needed from that bucket is `k − S[j+1]`. If `S[j] == k`, the whole bucket is selected, and that is the **early exit** (`:567-574`).

**Step 3: the census and the equal quota** [code `common_parts.cuh:477-504`; fp32 census `v3_fp32:206-233`]. Each thread counts `gt` (>pivot) and `eq` (==pivot) over its slice. It packs `(gt<<16)|eq` into **one u32**, so a single warp reduction or prefix computes both counts; this needs counts < 65536 (:481). The quota rule: `total_eq_quota = k − total_gt`. A thread's `eq_quota = max(0, total_eq_quota − eq_before_me)`. Its write offset is `gt_before + min(eq_before, total_eq_quota)`.

**Step 4: compaction.** Each thread writes its selected pairs to the *other* survivor buffer (A/B ping-pong, `common_parts.cuh:436-441`; `survivor_buf_idx ^= 1` at `v3_fp32:300`). The new threshold T is the pivot (`v3_fp32:543-545`).

**Teaching hook B: top-3 of 16 via a 4-bit radix (2 passes over 8-bit keys).** Same keys as hook A, k=3.
- Pass 1, high nibble histogram: `0:2 1:2 2:1 3:1 4:1 5:2 6:2 7:5 (8..F:0)`. Suffix sums: `S[8]=0, S[7]=5` → pivot bucket 7, need `3−S[8] = 3`. `S[7]=5 ≠ 3`, so this is not a whole bucket and pass 2 is needed.
- Pass 2 counts only keys with high nibble 7 (the others go to the sink). Low nibbles `F,C,F,A,0` give `0:1 A:1 C:1 F:2`. `S[F]=2, S[C]=3` → pivot digit C, need `3−S[D]=1`. `S[C]==3` means a whole bucket, so exit early; for fp32 this is where passes 3–4 get skipped.
- Pivot = 0x7C. Census: gt=2 (7F, 7F), eq=1. Quota = 3−2 = 1. Output = {idx1, idx5, idx3}.
- **Tie variant**: change idx 8 to 0x7C. Now eq=2 and quota=1. Whichever of idx3/idx8 comes first in the *buffer* wins. The buffer order is the permuted visit order, so the result is "not necessarily the lowest index".

### 2.4 Implementation map, fp32 sampler kernel (`csrc/cuda_kernels/v3_fp32/topk_select.cuh`) [code]

| Phase | Lines | What happens |
|---|---|---|
| Caveats | 1-7 | vocab < 2^23; do not assume prefix index on ties |
| Shortcut | 84-93 | if `end ≤ k`: emit indices 0..end, then pad (EpilogueRunner `IS_SHORTCUT`, `common_parts.cuh:129-185`) |
| Segmentation | 102-107 | 512-element segments. If the row has more segments than fit in the 32 KiB init window (16 for fp32, 32 for bf16), a prefix whose length is a multiple of 8 segments is permuted. The remaining 1–8 segments, i.e. the last ≤4096 elements (the "tail"), are visited in order inside the init window (`common_parts.cuh:311-323`) |
| Pseudo-random order | `common_parts.cuh:345-348, 452-462` | `seg' = (seg·(P mod n) + 0x22262226) mod n`, with P=0xB559EB75. **Verified prime here** (python), so for n < P `gcd(P mod n, n)=1` and the map is a bijection. It is fixed and input-independent, unlike the doc's "random" permutation [inference: an adversarial input could target it]. The Ascend version seeds from the row's first 8 bytes XOR its length (`kernel.asc:737-738`) |
| TMA issue | 312-329; `common_parts.cuh:628-710` | Up to 4 "issue warps" (`:332`) TMA-load whole segments into a 3-deep ring (`tma_load_buf`) with `EVICT_FIRST` (`:661-665`), 128B swizzle (`:389-399, 431`), and mbarrier `arrive_and_expect_tx` |
| Init window | 331-430 | A 32 KiB window (tail + first permuted segments, `common_parts.cuh:319-323`) is histogrammed (`:345-367`). **Exact** radix select sets the initial threshold (`:388`), survivors are appended (`:393-426`), and `threshold = pivot` (`:428`). This replaces "T = −∞ until the first compaction" |
| Main loop compare | 470-487 | Each thread loads **16 contiguous elements** per round (`common_parts.cuh:309`). The loop is `setp.gtu.f32` plus a predicated `or` into a 32-bit `hit_mask`. `.gtu` = unordered-true, **so NaN always counts as a hit** (`:481`) |
| Block prefix of hits | 488-515 | `__popc(mask)` → warp `__reduce_add_sync` → smem `warp_cnt[]` → `__syncthreads` → cross-warp and in-warp exclusive prefix (`utils.cuh:5-28`) |
| Append | 517-532 | `__ffs(mask)`, `mask &= mask-1` loop; store 64-bit `{value,index}` pairs into `incoming_topk_pairs`. The first hit's smem load is hoisted to overlap the barrier (`:500-506`) |
| Compaction trigger | 543-546 | `if num_incomers ≥ B2: T = reconstruct(...)` |
| Reconstruct | 236-302 | Histogram of survivors (k) + incomers → pivot, quota → write to the other buffer. Per-thread unit counts are forced odd for bank-conflict-free gathers (`:247`) |
| Final | 549-562 | Last reconstruct; block-wide `__syncthreads_or(have_nan)` → trap or `0x3F3F3F3F` (`common_parts.cuh:834-848`); epilogue |
| Launch | 569-601 | `grid = batch_size` (**one CTA per row**), `__launch_bounds__(threads, occupancy, 1)` |

The bf16 main loop (`common_parts.cuh:1395-1506`) is the same with 4 elements per step. Its hit mask is built by `set.gtu.s32.bf16x2` + `prmt` + `dp4a` with byte weights 1/2/4/8 (`:1413-1428`). It computes the lane prefix with **5 independent `__ballot_sync`s**, one per bit of the ≤16 hit count, instead of a 5-step shuffle scan (`:1455-1463`). Long rows reconstruct early at B2/4 (`:1385-1390`).

### 2.5 Output options [code]
- **`return_value`**: when false, the value loads and stores are compiled out (`common_parts.cuh:152-156, 172-184, 199-208`). "~10% faster" [doc README:91].
- **`sorted` (by value, fp32 only)**: `cub::BlockRadixSort<distorted uint, …, 4 radix bits>::SortDescending` over k (`common_parts.cuh:117-124, 230-246`). It requires `return_value` and excludes `sorted_index` (`api.cpp:60-61, 148`).
- **`sorted_index`**: segments are visited whole and compaction is stable, so survivors are already index-sorted *within* each 512-element window. The epilogue then does a **counting sort by window id**: count per window with `atomicAdd_block`, block prefix sum, then subtract the first-in-window position (`common_parts.cuh:722-799`). The README says it costs performance and should be disabled unless needed (README:26).
- **`end`** (per-row length) and **`output_idx_offset`** are supported. `begin` and `hint` are not (`interface.py:94-95`).
- **Stride contract**: input row stride must be a multiple of 1024 B (`structs.h:17`; TMA box and OOB safety, `common_parts.cuh:392-393`). Output strides are 32 B-aligned, so outputs may be non-contiguous (`interface.py:78-86`).

### 2.6 Parallelization, shared memory, registers; large vs small batch
- **Per-row**: one CTA per row, 512 threads for fp32. Each warp owns one 512-element segment per round (`static_assert(NUM_SEGS_PER_ROUND == NUM_WARPS)`, `common_parts.cuh:401`).
- **Registers**: the budget formula is `65536/(threads·occupancy)` → 128 regs/thread for 512×1 (`common_parts.cuh:116`). Builds use `--register-usage-level=10 --warn-on-spills`, and **fail the build on any spill** (`setup.py:149, 168-181`).
- **Shared-memory plan** (`common_parts.cuh:436-450`): the `surviving_topk_pairs[2][MAX_TOPK]` ping-pong, `incoming_topk_pairs[B2+B]` (aliased as the init-window buffer → candidate buffer → epilogue temp, `:383-386`), `tma_load_buf[depth][B]`, barriers, and 2×260 histogram slots. Sizes computed from the struct layout [inference, not `sizeof`-measured]:

| Config | Shared memory |
|---|---|
| fp32 k≤512 (512 threads, B=8192, B2=4096, TMA depth 3) | **203 KiB** |
| fp32 k≤1024 | 211 KiB |
| fp32 k≤4096 (256 threads, B=4096) | 179 KiB |
| bf16 occupancy 2 (256 threads, B=4096, depth 4) | 107 KiB (×2 CTAs/SM) |
| *Hypothetical sm_89 fp32 (256 threads, B=4096, B2=2048, depth 2)* | *~91 KiB, fits in 99 KiB* |

- **Dispatch** (`api.cpp:145-221`):
  - bf16 with `batch ≤ 6 && V ≥ 512K && k ≤ 1024` → **cluster kernel**: 16 CTAs per row, each computes a local top-k, then all ship to CTA0 via `st.async` into distributed shared memory, and CTA0 does a final radix select (`v3_cluster/topk_select.cuh:1-8, 156-255`). The purpose is to fill SMs when batch < #SMs ("wave quantization", doc :114).
  - bf16 with one wave → 512 threads, occupancy 1.
  - bf16 otherwise → 256 threads, occupancy 2.
  - **fp32 has no cluster variant**. The sampler at small batch is simply latency-bound (see the plot numbers in §4).
- *Batch-invariance note [inference]*: the fp32 config depends only on k, so tie-breaks should not change with batch size. The bf16 config depends on `num_waves`, which changes round grouping, which could in principle change which tied elements survive. For serving, "the same request gives the same tokens regardless of batch" is a real requirement; worth a quiz question.

### 2.7 Micro-optimizations: know them, don't copy them
- **Integer adds done as float adds**: for `0 ≤ x,y ≤ 2^22`, `x+y` equals `float_as_uint(uint_as_float(x)+uint_as_float(y))` (denormals). The kernel uses this for smem pointer bumps (`add.f32 %0, %0, 0f00000008`, `common_parts.cuh:1041, 1086`; `v3_fp32:71`) and index arithmetic (`v3_fp32:413-415`). The reason: compare and bitwise instructions contend with integer adds [doc :110]. This **requires `--ftz=false`** (`setup.py:148`) and limits vocab to < 2^23 (`structs.h:20-22`). Census counts accumulate on the FP pipe via `set.gt.f32` → 1.0/0.0 (`v3_fp32:206-223`). **neither**: the lesson is "find the saturated pipe/port with a profiler", not the trick itself.
- NaN detection by `min.NaN` accumulation (`v3_fp32:219, 227`); `bfe`/`prmt`/`lop3` bit tricks (`common_parts.cuh:911-924, 945-962`). **sm_89**: mostly available. The bf16x2 `set`/`setp` forms are newer [external: check the PTX ISA target table before relying on them].
- Note: the deep-dive links (e.g. `common_parts.cuh#L1466`) point to commit `8e70df71`. At HEAD, the `__ffs` loop is at `:1476-1484` and the mask build at `:1413-1428`.

### 2.8 Pitfalls to pre-empt
- Comparing a top-k implementation's **indices** to `torch.topk` or a sort will fail on ties even when both are correct. Use DeepSelect's properties instead (§10).
- `>` versus `≥` for the threshold. Appending only `> T` is safe because T is the current k-th value; equal elements are already represented, and the quota handles them. Using `≥` makes the buffer explode on many-ties inputs (e.g. `UniformUIntDistribution(0x0, 0x1)`, `tests/test.py:226`).
- NaN: `x > T` is false for NaN, so NaN silently disappears. DeepSelect deliberately uses unordered compares so NaN is detected. In Rust, `partial_cmp().unwrap()` panics on NaN and `total_cmp` puts NaN at the extremes. Make it a decision.

---

## 3. The sampling scenario

### 3.1 What the repo says and implies
- The README's sampling scenario is fp32, any batch, V ≈ 128K, k ≤ 4096; fp32 is CUDA-only (README:29-37). The benchmarked sampler cases are `TestParam(b, 129280, 512, sorted_value=True, sorted_index=False, return_value=True, fp32, int64)` for `b ∈ {6,256,512,768,4096}` (`tests/test.py:249-254`). 129,280 is DeepSeek-V3's vocab (`DeepSeek-V3/inference/configs/config_671B.json:2`).
- There is **no sampler code** in DeepSelect. The flow in §3.2 is [inference] plus cross-references to real sampler code.
- Cost implications [inference]: one row is 129,280 × 4 B = **505 KiB**. At batch 4096 that is ~2.1 GB per call, which is bandwidth-bound. At batch 1–6 it is 0.5–3 MB, which is **latency-bound**: one CTA per row leaves most SMs idle, and launch overhead is comparable to the transfer time. On the 4070 (36 MiB L2), a decode-time logits row would typically still be L2-resident right after the LM-head matmul, unless the benchmark flushes L2 as DeepSelect's does.

### 3.2 How a sampler uses top-k [inference, grounded in vLLM and DeepSeek code]
```
logits[V] --(select top-k on raw logits; T>0 scaling is monotone so selection is invariant)-->
cand[k] sorted desc by value
  -> apply temperature on k values only
  -> softmax over k (or over V using a full-row logsumexp Z, see below)
  -> cumsum; cut at top-p (keep at least one)
  -> draw u ~ U[0,1), inverse CDF over the survivors (or exponential race)
```
- vLLM's PyTorch path **sorts the whole vocab** (`logits.sort`). It masks below the k-th, softmaxes over the remainder, cumsums, masks `cumsum ≤ 1−p`, and forces "at least one" (`vllm/v1/sample/ops/topk_topp_sampler.py:483-503`, esp. :499). Its top-k-only path avoids a full sort by using `topk` (:506-525). The Triton path documents the order "Top-k is applied first (by logit value), then top-p is applied to the remaining k values" and cites the "Qrita" pivot-based top-k/top-p paper (`topk_topp_triton.py:1-9, 1621-1622`). FlashInfer's path uses **rejection sampling to avoid sorting** and only guarantees *statistical* equivalence (`topk_topp_sampler.py:569-606`, esp. :577-583).
- **Exactness trap [inference]**: using a top-k=512 prefilter for pure top-p (no top-k) is exact only if the nucleus fits inside the 512. Checking that requires the **full-vocab normalizer** `Z = Σ exp(l_i/T)` (a streaming logsumexp in the same pass). If the top-512 mass is below p, fall back to the full path.
- **The exponential-race (Gumbel-max) trick**: `argmax(p_i / E_i)` with `E_i ~ Exp(1)` samples from the categorical. It needs no sort and no cumsum, and it is a pure reduction (`DeepSeek-V3/inference/generate.py:25-27`; vLLM `random_sample`, `topk_topp_sampler.py:535-566`, used to avoid the CPU-GPU sync of `torch.multinomial`). The cost is V random numbers per step. It is exact in distribution, but it will **not** pick the same token as llama2.c for the same seed.

### 3.3 llama2.c's sampler, verified [code `run.c`]
- `sample()` (`:691-714`): at T=0 it is `sample_argmax` (`:590-601`, strict `>`, so first max index wins). Otherwise it divides the logits in place by T (`:699`, a **division**, not a multiply by 1/T), runs `softmax` (`:197-215`: max pass, `expf` + sequential sum, divide), draws `coin = random_f32` (xorshift, 24-bit, `:680-689`), then `sample_mult` (`:603-614`) or `sample_topp`.
- `sample_topp` (`:624-665`):
  1. **Pre-filter**: `cutoff = (1−topp)/(n−1)`; keep `p_i ≥ cutoff` into `probindex` (`:634-641`).
  2. `qsort` the n0 survivors descending (`:642`, comparator `:616-622`; ties return 0).
  3. Cumulative sum until `> topp`, giving `last_idx` (`:645-653`).
  4. `r = coin·cumulative_prob`, then inverse CDF over `[0..last_idx]` (`:656-664`).
- **Why the cutoff is safe** (for topp ≥ 1/n) [inference, proof]: if sorted element j (j ≥ 1) is in the nucleus, the mass before it is ≤ topp. So the tail starting at j has mass ≥ 1−topp spread over ≤ n−1 elements, and p_j, the tail's largest, is ≥ (1−topp)/(n−1). For j=0, p_0 ≥ 1/n ≥ (1−topp)/(n−1) iff topp ≥ 1/n. For V=32000 and topp=0.9 the cutoff is 3.1e-6; for V=151,936 it is 6.6e-7.
- Defaults: `temperature=1.0, topp=0.9, steps=256` (`:911-913`).
- **M4 reproducibility traps**: (a) `qsort` is not stable, so the order of equal probabilities is unspecified and can change which token a given coin lands on. Decide on "prob desc, index asc" and document the rare divergence. (b) Match `/= temperature` and the sequential sums exactly if the checkpoint compares text with llama2.c at the same seed. (c) Any faster top-p (M6) changes *which* token a coin maps to unless it preserves the sorted order. Switch that checkpoint to a statistical test (§10).

---

## 4. Performance methodology

- **Metric**: effective memory bandwidth. "TopK does no floating-point math, so a FLOP rate would not be meaningful" [doc README:43-44]. `total_size = (Σend or B·V)·elem + bytes(values_out) + bytes(indices_out)`, and `TB/s = total_size / kernel_time` (`tests/test.py:149-157`).
- **Timing** (`tests/kernelkit/bench.py:112-174`): the kernel is timed on the GPU via `torch.profiler` (CUPTI/kineto), not by host wall clock. There is a warmup cycle, and a Triton "marker kernel" delimits the measured range (:23-28, :130-131, :158-164). Before each run it **flushes L2 with an 8 GB memset** (:119-134). It filters kernel names containing "topk" (`test.py:152-156`), runs `num_runs=10` per perf case, and sleeps 0.2 s between perf cases (`test.py:278-279`).
- **Baseline**: `torch.topk(x, k, dim=1, sorted=p.sorted_value)` on the same tensor, timed the same way, and printed as a speedup (`test.py:159-170`). It is only compared when `end` and `offset` are absent.
- **Numbers** [doc]: README:3 "2 ~ 20x", README:91 "~10% faster" without values. From `assets/perf_fp32_cuda.png` (V=129,280, k=512; values read off bars, so ≈; the GPU is not named, but the build targets sm_100a/sm_103a):

| batch | DeepSelect TB/s | torch.topk TB/s | ≈ speedup |
|---|---|---|---|
| 6 | ~0.1 | ~0.03 | ~3× |
| 256 | ~2.3 | ~0.6 | ~3.8× |
| 512 | ~2.4 | ~0.9 | ~2.6× |
| 768 | ~2.4 | ~0.9 | ~2.7× |
| 4096 | ~2.8 | ~1.15 | ~2.4× |

  From `perf_bf16_cuda.png` (indexer, k=512): at batch 4096 and V=1M, ~5.3 vs ~0.27 TB/s (~20×, the top of the "2~20x" claim). At batch 6, every bar is < 0.6 TB/s.
  *Reading [inference]*: batch 6 × 505 KiB ≈ 3.1 MB at ~0.1 TB/s is ≈ 30 µs, so small-batch sampling is latency-bound. That is the regime of single-user decode on the learner's machine.
- **TileKernels uses the same idea**: `bandwidth_gbs = count_bytes(inputs, outputs)/t_us/1e3` (e.g. `tests/quant/test_per_token_cast.py:255-264`). Timing comes from `tilelang.profiler.bench.do_bench(backend='cupti', warmup=0, rep=30)` (`testing/pytest/benchmark.py:779-802`).

**Transfer to the learner** [inference]. For the 4070, first *measure* its achievable bandwidth: a read-only reduction kernel over ≥ 1 GB. The spec sheet says ~504 GB/s [external]. Then report each sampling kernel as µs and as % of measured peak. On CPU the analogous metric is bytes of logits touched per pass ÷ time, compared with a measured L2 and DRAM stream bandwidth. This is role project 4 in miniature.

---

## 5. Complexity comparison for role project 1 (CPU view)

### 5.1 Algorithms and costs [inference: textbook costs, arithmetic here]

| Method | Cost model | V=32,000 (stories15M) | V=129,280 | V=151,936 (Qwen2.5-class [external]) |
|---|---|---|---|---|
| Full sort | V·log2 V compares | ~479K | ~2.20M | ~2.62M |
| llama2.c top-p | V (filter) + n0·log2 n0 (qsort) + ≤n0 (cumsum) | depends on **n0**: ~50K if n0≈3000; ~479K if the distribution is flat | same | same |
| Quickselect / `select_nth_unstable` | ~2V + 2k·ln(V/k) compares, plus a V-sized mutable copy | ~65K (k=40) | ~260K | ~305K |
| Heap of size k (random order) | V root compares + k·ln(V/k)·log2 k sift ops; worst case V·log2 k | ~33K expected (k=40); 170K worst | ~131K / 688K | ~154K / 809K |
| Radix select (8-bit digits) | pass 1 over V + passes on the pivot bucket (≤4 for fp32; the first pass is weak on logits, §2.3) | ~V + a few ×(bucket size) | same | same |
| DeepSelect-style filter | V compares (SIMD) + E[W] buffer work | 32K + ≤3.5K (k=40, B=B2=256) | 129K + ≤21K (k=512, B=B2=1024) | similar |
| Exponential race (no top-p) | V RNG draws + V divides + argmax | 32K RNG | 129K RNG | 152K RNG |

Data volume: 32,000 fp32 is 125 KiB, which fits in a 5900X core's 512 KiB L2. At 129K–152K it is 505–594 KiB, just over the per-core L2, so the multi-pass softmax streams from L3. A `(prob, index)` pair array for a full sort is 2× larger again.

**Key insight to teach: top-p is selection by *cumulative mass*, not by count** [inference]. DeepSelect finds the pivot bucket by suffix *counts* (`find_pivot_in_histogram`). A top-p variant does the same with suffix *sums of probability mass* per bucket. Walk buckets from high to low until the running mass exceeds p, then only sort or scan the one crossing bucket. This is the "pivot-based truncation" idea the vLLM Triton kernel cites. Because monotone maps preserve order, bucketing on probabilities (exponents spread well in (0,1]) can separate better than bucketing raw logits.

### 5.2 Mapping to AVX2 + threads (M6) [inference; intrinsic names from `core::arch::x86_64`]
The filter loop is DeepSelect's hit mask with 8 lanes instead of 16 elements per thread:
```text
t = _mm256_set1_ps(T)
for i in (0..V).step_by(8):
    v = _mm256_loadu_ps(&x[i])
    m = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_NLE_UQ>(v, t))   // "not <= T", unordered→true: NaN counts as a hit,
                                                                  // exactly DeepSelect's .gtu / Ascend's !(v <= T) (kernel.asc:277-278)
    while m != 0 { j = m.trailing_zeros(); push((x[i+j], i+j)); m &= m - 1; }   // tzcnt + blsr (BMI1)
    if buf.len() >= k + B2 { compact(&mut buf, k); T = kth; t = set1(T) }
```
- AVX2 has no compress-store (that is AVX-512 `vpcompressd`). The bit loop above, or a 256-entry permutation LUT with `_mm256_permutevar8x32_ps`, is the standard substitute. Hits are rare after warm-up, so the bit loop is fine.
- Histograms on CPU are scalar: no scatter and no conflict detection in AVX2. Repeated bins stall on store→load forwarding. Several sub-histograms merged at the end is the usual cure [external].
- Visit order: DeepSelect randomizes block order for GPU reasons *and* for the adversarial-order bound. On CPU a simpler guard is to seed T from a small sample first, like DeepSelect's exact top-k of a 32 KiB init window (`v3_fp32:331-430`). Random block order hurts hardware prefetching less than one might fear at 125 KiB, but that needs measuring.
- Threads: for **one row** of 32K, splitting the sampler across threads likely costs more in wake-up and sync than it saves [inference; measure]. For the **batched server**, parallelize across rows (one row per worker), which mirrors DeepSelect's one CTA per row.
- `expf` is the real arithmetic cost of softmax (V calls per token). A vectorized polynomial `exp` (8 lanes) is the M6 win. Then keep bitwise or tolerance equivalence explicit: TileKernels' "fast certificate" (§7.4) is the model for fast math that stays provably exact.

### 5.3 Does sampling matter next to the matmuls? How to *measure* (not guess)
Rough scale [inference]: stories15M does ~15M MACs per token (≈6M in the layers + 9.2M in the 32000×288 classifier), while sampling is ~3–5 passes over 32K floats plus `n0·log n0`. Sampling is small against a naive f32 matmul. Once M6 makes the matmuls 10–50× faster, sampling's share grows (Amdahl), especially at high temperature where n0 → V. Protocol for the learner:
1. **Phase timers** in the generate loop: time `forward` and `sample` separately per token with `Instant`. Report mean/p50/p99 and the sampling share (%). Use a `--release` build, a fixed prompt and seed, and 1 thread pinned (`taskset -c 2`).
2. **Log n0 and nucleus size per step**, and plot the histogram. Run at T ∈ {0 (argmax), 0.8, 1.0, 1.5}, topp ∈ {0.9, off}.
3. **Replay benchmark**: dump real logits for ~256 decode steps once, then benchmark each sampler implementation on the *same* rows (criterion). Real peaky distributions, not `randn`, determine n0. DeepSelect's synthetic distributions are for correctness, not for this.
4. **Profile**: `perf record -g ./target/release/...` and a flamegraph. Also, with no patch needed, `perf record ./run stories15M.bin -t 1.0 -p 0.9 -s 42 -n 256 -i "Once upon a time"` on llama2.c gives the reference share of `qsort`/`compare`/`expf`.
5. **Predict first** (role project 4): from measured L2 bandwidth and measured `expf` throughput, predict the sampling µs per token, then compare. Repeat after M6.

### 5.4 Teaching pseudocode for M6 (not engine code; the learner designs the real one)

(a) **Top-p by mass-histogram pivot.** The same shape as DeepSelect's `find_pivot_in_histogram`, but with mass instead of counts [inference]:
```text
Z = Σ w_i  where w_i = exp((l_i - max)/T)          # unnormalized; target mass = p·Z
cand = { i : w_i ≥ cutoff·Z }                      # llama2.c cutoff (1-p)/(n-1), vectorized hit-mask filter
bucket(i) = high bits of w_i's float exponent/mantissa  # monotone in w_i (w_i > 0, so raw bits already order correctly)
mass[b] += w_i ; cnt[b] += 1        for i in cand
walk b from highest to lowest: acc += mass[b]; stop at first b* where acc > p·Z
nucleus = all i in buckets > b*  ∪  (sort only bucket b* desc, take until acc crosses p·Z)
sample: r = u · mass(nucleus); inverse CDF over nucleus IN THE SAME ORDER the reference uses
```
The last line is the reproducibility knob. Walking in "bucket desc, then within-bucket sorted" order equals llama2.c's sorted order only if the within-bucket order and the ties match.

(b) **Top-k by bounded min-heap** (the simplest CPU baseline to beat qsort):
```text
heap = first k (value, idx) as a min-heap          # root = current k-th largest = DeepSelect's T
for i in k..V: if x[i] > root: replace root, sift-down  # "x > T" filter, again
sort heap desc (k log k)
```
Question to ask the learner: what is the heap root, in DeepSelect's vocabulary? (It is the threshold T. The heap is a candidate buffer compacted after *every* hit, i.e. B2=1.)

---

## 6. GPU learning path for M7 on sm_89 (Ada)

### 6.1 What ports [external for the arch facts; verify each against the CUDA/PTX docs when M7 starts]
| DeepSelect ingredient | sm_89? | Ada substitute |
|---|---|---|
| Threshold filter + candidate buffer + radix compaction | ✅ algorithm | same |
| `__ballot_sync`, `__popc`, `__ffs`, `shfl.sync` scans, `__reduce_add_sync` (sm_80+) | ✅ | same |
| smem atomics / `red.shared.add` histograms | ✅ | same |
| `dp4a`, `prmt`, `lop3`, `bfe` | ✅ | same (inline PTX or intrinsics) |
| TMA (`SM90_TMA_LOAD_3D`), mbarrier `expect_tx`, `elect.sync` | ❌ sm_90+ | `cp.async` (sm_80+) double buffer, or plain vectorized `ld.global.v4` |
| Thread-block clusters + DSMEM `st.async` (small-batch variant) | ❌ sm_90+ | split the row over several CTAs, write per-CTA top-k to global, then a second tiny kernel (or last-block-done atomic counter) merges |
| 203 KiB smem plan | ❌ (99 KiB/block) | smaller B/B2/depth (≈91 KiB config in §2.6) or register-resident rows |
| LDG/STG 256-bit (`common_parts.cuh:18-37`) | ❌ sm_100 | 128-bit loads |
| float-add-as-int-add | ✅ but needs `--ftz=false`, which fights `--use_fast_math` | don't |

### 6.2 A simple-to-fast kernel sequence (each validated against the Rust CPU sampler = oracle)
- **K0 (CPU oracle)**: the M4 Rust sampler, plus a property checker and a statistical checker (§10).
- **K1 argmax**: one CTA per row, block reduction (warp shuffle, then smem). Define the tie rule (lowest index) and test bitwise vs CPU.
- **K2 softmax (+T)**: block max + sum-exp, or an online single-pass softmax. Tolerance from a measured D1-style study, not guessed.
- **K3 sample**: inverse CDF via block prefix scan and "first cdf > u", or the exponential race with a device RNG. Decide RNG reproducibility: seeded per (request, step) counter-based RNG so CPU and GPU can agree [inference].
- **K4 radix top-k in smem**: DeepSelect's reconstruct path alone (histogram → pivot via suffix scan → census → eq-quota compaction). V=32000 fp32 = 125 KiB > 99 KiB, so either re-read from global/L2 per pass (4 passes × 125 KiB, trivially L2-resident) or keep the row in registers (e.g. 1024 threads × 32 values). Watch spills with `-Xptxas -v`, as DeepSelect's build does.
- **K5 threshold filter**: blocks of B, hit masks via `__ballot_sync`/per-thread bits, warp-aggregated append, and compaction when the buffer ≥ k+B2. `cp.async` prefetch of the next block.
- **K6 top-p on k candidates**: sort k (`cub::BlockRadixSort` as DeepSelect's epilogue does, or bitonic for k ≤ 1024), block scan of probabilities, cut, sample. Also a "mass-histogram pivot" variant (§5.1, §5.4).
- **K7 batched**: grid = batch. Measure the wave quantization effect with 46 SMs (batch 46 vs 47). Then try split-row for batch < 46 (the Ada version of the cluster idea).
K4 skeleton, one CTA per row, radix select over distorted keys [inference; mirrors `v3_fp32:147-302`]:
```text
prefix = 0; need = k; shift = 24
repeat up to 4 times:
    clear hist[256+1]                                   # +1 = sink slot, as DeepSelect does
    for key in my slice: b = (key>>shift)&0xFF if (key>>(shift+8))==prefix else 256; atomicAdd_smem(hist[b])
    __syncthreads
    warp 0: suffix-scan hist -> pivot byte j, need = need - S[j+1], whole = (S[j]==need_before)
    prefix = (prefix<<8)|j; shift -= 8; if whole: break
pivot = prefix << (shift+8)   (fill low bits correctly; for negatives DeepSelect fills 1s in raw space, v3_fp32:197-204)
census gt/eq -> block prefix of packed (gt<<16|eq) -> eq_quota -> write exactly k pairs
```
Check it on the same 16-key example as §2.3 with 4-bit digits before running on V=32000.

- **Measure each step** DeepSelect-style: µs, effective GB/s vs measured peak, speedup vs `torch.topk`/`torch.softmax` on the same GPU (torch with CUDA exists in `~/refs/inference/venv`), with L2 flushed and not flushed, since decode logits are usually L2-hot [inference].

---

## 7. TileKernels quantization

### 7.1 Formats [code]
- **E4M3** (`torch.float8_e4m3fn`, finite-only, max **448**): `torch/cast.py:21-24`; the Ascend kernels hardcode 448 (`quant/per_token_cast_asc.py:28`).
- **E2M1 FP4** (max **6.0**; levels 0, 0.5, 1, 1.5, 2, 3, 4, 6), **packed two per byte in `torch.int8`**: the type mapping is `common.py:33`, packing `lo | hi<<4` is `torch/cast.py:278-281`, and the decoder with s|e|m layout and bias 1 is `common.py:319-367`.
- **No E5M2** anywhere (grep). All quantization is **symmetric**, with no zero points.
- Hardware relevance [external]: Ada (sm_89) has FP8 E4M3/E5M2 conversion and FP8 tensor-core MMA. FP4 conversion and MMA are Blackwell (sm_100+). The CPU has neither, so FP8 on CPU means a 256-entry dequant LUT.

### 7.2 Scale computation, rounding, clamping [code]
- Clamp the amax from below to avoid a zero or denormal scale (`config.py:10-15`): `e4m3 → 1e-4`; `e4m3 + e4m3-scale → 448·2^-9`; `e2m1 → 6·2^-126`; `e2m1 + e4m3-scale → 6·2^-9`. The value is overridable and tested (`tests/quant/test_per_token_cast.py:164-179`).
- `get_sf_and_inv` (`quant/common.py:251-282`):
  - plain: `sf = amax_c / max_q`, `sf_inv = max_q / amax_c`. The inverse is **computed separately**, not `1/sf`; this matters for bitwise tests (:263-268).
  - `use_e4m3_sf`: the scale is stored as e4m3 and `sf_inv = 1/float(e4m3(sf))` (:264-266; the reference mirrors it at `torch/cast.py:239-243`).
  - `round_sf` (power of two): `exp = ((bits−1) >> 23) + 1 − 127` = **ceil(log2 sf)** (:271-274). The scale is rounded *up* so `|x|/sf ≤ max_q` still holds. The reference is `(bits + 0x7FFFFF) & 0x7F800000` (`torch/cast.py:233-236`).
  - UE8M0 storage = the biased exponent byte `exp+127` (:279-280), packed 4 per int32 on CUDA (`common.py:70-75`). Decode is `uint32(byte) << 23` reinterpreted as f32 (`common.py:303-306`).
- Rounding: the hardware cast does round-to-nearest-even. The FP4 reference implements RTNE explicitly with guard/sticky bits (`torch/cast.py:321-349`). Optional **stochastic rounding** (`round='rs'` with hashed random bits, `per_token_cast_cuda.py:10-25, 162-174`) is for training.
- Saturation: the fp8 reference clamps to ±448 *before* the cast (`torch/cast.py:274`). FP4 saturates by `min(code, 0x7)` (:346). Zero-amax groups get `quant_sf = 0` in the reference (:236, :245).

### 7.3 Granularity [code]
| Kind | Scale block (rows × cols) | Allowed | Where |
|---|---|---|---|
| per-token | (1, num_per_channels) | 16/32/64/128, hidden % 64 == 0 | `per_token_cast_kernel.py:28, 32` |
| per-block | (num_per_tokens, num_per_channels) | {32,128}×{32,128}; tokens and hidden % 128 == 0 | `per_block_cast_cuda.py:27-28`; `per_block_cast_kernel.py:33-34` |
| per-channel | (num_per_tokens, 1) | 32/128, e4m3 only | `per_channel_cast_kernel.py:39-41, 53` |

The per-token kernel: one CTA does an absmax reduction per group (`T.reduce_absmax`) → scale → multiply by `sf_inv` → implicit cast (`per_token_cast_cuda.py:187-216`). Threads and elements per thread switch at 4096 tokens (`:46-49`; `per_token_cast_kernel.py:24, 59`).

### 7.4 Fused SwiGLU + per-token quant [code `quant/swiglu_forward_cuda.py`]
- Reads `x[t, 0:H]` (gate) and `x[t, H:2H]` (up) (:163-180). Optional clamp (:198-211). Computes `silu(g)·u` (×routing weight) (:226-230). Warp-shuffle absmax over the group (:271-280) → `get_sf_and_inv` (:283) → scale, cast, store fp8 plus the scale (:286-297). One kernel, one pass.
- **Why fuse** [inference, arithmetic]: unfused, SwiGLU writes an fp32 H-vector and the quant kernel reads it back. For bf16 input the bytes per token are `4H (read) + 4H (write fp32) + 4H (read) + H (write fp8)` ≈ **13H** unfused vs **5H** fused, about 2.6× less traffic, plus one fewer launch.
- **The "fast certificate"** (:232-269): compute with the fast `__expf`/`__fdividef`, then check whether any result sits close to a rounding-decision boundary. The check uses the low 19 mantissa bits against a margin, plus gate > −16. If any lane of the warp is unsafe, recompute that lane group precisely. The result is **fast math that still matches the precise reference bitwise** (the test asserts bitwise equality, `tests/quant/test_swiglu_forward.py:128-129`). This is a great M6 hook for vectorized `exp` in softmax.

### 7.5 Mapping to the learner's M5 [inference unless cited]
llama2.c's Q8_0 (`runq.c:145-171`: `scale = wmax/127`, `q = (int8_t) round(x/scale)`; export default group 64 with backoff, `export.py:182-195`) is **TileKernels per-token cast with int8 and fp32 scales**. It also quantizes **activations dynamically, per token, before each matmul** (`runq.c:367, 437, 450, 465, 478`). The int32 group dot then scales by `w_s·x_s` (`runq.c:317-342`).

**Worked example (group of 4, exact arithmetic).** x = [63.5, −12.25, 3.25, 0.75], amax = 63.5, scale = 63.5/127 = 0.5 exactly, so x/scale = [127, −24.5, 6.5, 1.5].
- C `round` / Rust `f32::round` (ties away from zero): [127, **−25**, **7**, 2]
- `torch.round` / Rust `f32::round_ties_even` (ties to even): [127, **−24**, **6**, 2]

So `runq.c:167` and `export.py:62` disagree on ties. Pick one, record it in DECISIONS.md, and make the Rust reference identical.
- FP8 analog: `sf = 63.5/448 ≈ 0.1417`, `sf_inv = 448/63.5 ≈ 7.055`, `x·sf_inv` → e4m3 RNE.
- UE8M0: ceil to 2^-2 = 0.25, so max |x|/sf = 254 ≤ 448. For **floating** formats a power-of-two scale only shifts the exponent, so relative precision is unchanged. For **int8** it can waste up to ~1 bit of levels: amax=70 gives exact scale 0.551, but the power-of-two scale is 1.0, so only 70 of 127 codes are used. That is why power-of-two scales suit MX FP8/FP4 but not Q8_0.

**Decisions the learner will face (one at a time, per CLAUDE.md):**
1. **Group size**: 32 (ggml Q8_0 [external]) vs 64 (llama2.c export) vs 128. Overhead with an fp32 scale is 4/32 = +12.5% (1.125 B/weight) vs +6.25% at 64. Error vs overhead: measure drift at each.
2. **Scale dtype**: f32 (simple, exact) vs f16 (ggml [external]) vs power-of-two (a cheap dequant, but it wastes int8 levels, as above).
3. **Symmetric vs asymmetric**: TileKernels and Q8_0 are symmetric (−127..127; −128 unused keeps negation exact). Asymmetric (min + scale) helps skewed tensors such as post-SiLU activations [external/inference].
4. **Divide vs multiply-by-inverse**: `x/scale` (runq.c:166) vs `x*sf_inv` with `sf_inv = 127/amax` (TileKernels style). These give different last bits, so choose one and mirror it in the reference.
5. **Zero-group guard**: runq divides by 0 on an all-zero group, and C's `(int8_t)NaN` is UB. Copy TileKernels' amax clamp or its explicit `sf==0` branch.
6. **Where to quantize activations**: dynamically before each matmul, like runq (the per-token group cast), and eventually **fused into the producer** (rmsnorm→quant, SwiGLU→quant), which is what TileKernels' `norm_forward`/`swiglu_forward` do. On CPU at stories15M sizes the fusion saves an L1-resident pass, likely negligible. Measure it; it matters on GPU and at batch size.
7. **4-bit format later**: int4 (Q4_0-style, scale per 32, codes −8..7) vs **FP4 E2M1** + scale (non-uniform levels 0…6, better for bell-shaped weights). Both dequantize on AVX2 via a 16-entry `_mm256_shuffle_epi8` LUT [external]. E2M1 packing and decoding code to read: `torch/cast.py:278-281, 321-349`, `common.py:319-367`.
8. **Stochastic rounding**: no, for inference. But TileKernels' `check_bias` test (§10) is worth copying to prove RNE has no systematic bias.

---

## 8. MoE routing top-k and RoPE (brief; M8/post-v1)

**MoE top-k** [code]:
- `topk_gate(scores, k)` requires k ≤ 32, fp32, finite inputs (asserted), and returns the **smaller index on ties** (`moe/topk_gate_kernel.py:7-43`, esp. :23). The CUDA kernel is one 32-thread CTA per token. It loads scores into a fragment and repeats k times: block `reduce_max`, min-index reducer among equals, mask to −inf (`moe/topk_gate_cuda.py:12-64`, loop :51-63). The cost is O(k·E), which is fine for E ≤ ~1K and k ≤ 32. This is the "tiny k" strategy, the opposite end from DeepSelect.
- `moe_topk_gate_forward` (DeepSeek routing) computes `sqrt(softplus(logits))` + bias for ranking, top-k by repeated warp-argmax with the lexicographic tie-break `(score >, or == and idx <)` (`moe/moe_topk_gate_forward_cuda.py:224-250`, tie-break :242). Weights are the unbiased scores normalized by their sum ×`routed_scaling_factor` (:262-272). Shared experts are appended and logical→physical replicas are mapped with `(ep_rank + token·23333) % count` (:280-284), a load-balancing hash; see role project 3.
- Reference: `torch/topk.py:18-21` (`stable_topk` = stable sort, then slice) and `:24-154`.

**RoPE** [code]: `apply_rotary(query, cos_sin_cache, key, positions, interleaved, conjugate, seqlen_offset)` works **in place** and supports rot_dim 64/128 only (`transform/rope_kernel.py:9-63`). `interleaved=True` is GPT-J pairing `(2i, 2i+1)`, which is what llama2.c does (`run.c:264-277`); `False` is the NeoX half-split, the HF Llama layout (`rope_kernel.py:31`). The kernel uses a **precomputed fp32 cos/sin cache**, vec=4 per thread and explicit FMAs (`rope_cuda.py:82-114`). llama2.c recomputes `powf`/`cosf`/`sinf` per pair per token (`run.c:266-270`), which is a small M3/M6 decision (table vs recompute; numerics differ slightly, so check against the D1 tolerance). The test tolerance is `calc_diff ≤ 1e-8` because torch does not FMA; see the TODO in `tests/transform/test_rope.py:45-46`. *Rust pitfall [external]:* `f32::mul_add` is a slow software FMA unless the `fma` target feature is enabled.

---

## 9. TileLang in one page (context for M7; the learner writes CUDA C++)

Seen in this repo [code]:
- `@tilelang.jit(pass_configs=…)` wraps a Python function whose static args (hidden, configs) specialize and JIT-compile a `@T.prim_func`, like C++ template instantiation at runtime (`per_token_cast_cuda.py:28-45, 96`). `T.dynamic('num_tokens')` marks symbolic runtime sizes (:84).
- `with T.Kernel(grid, threads=N) as pid:` is the launch (:104).
- Memory scopes: `T.alloc_fragment` (a register tile *distributed across threads*), `T.alloc_shared`, `T.alloc_local` (per-thread registers), `T.alloc_var` (scalar) (:109-114, `swiglu_forward_cuda.py:108-128`).
- `T.copy(global_slice, frag_or_shared, disable_tma=True)` is a tile copy (vectorized loads; TMA on sm_90 if not disabled) (:124).
- `T.Parallel(m, n)` is an elementwise tile loop that the compiler maps to threads by the fragment's layout. `T.annotate_layout({frag: T.Fragment(shape, forward_fn=…)})` gives the explicit (thread, register) mapping (:90-92, 116-121, 176-179).
- `T.reduce_absmax/reduce_max(src, dst, dim)` are tile reductions (:190). `T.shfl_xor`, `T.warp_reduce_sum`, and `T.all_sync` are warp primitives (`swiglu_forward_cuda.py:280, 256`).
- `T.Pipelined(n, num_stages=2)` is a software-pipelined loop that overlaps the next copy with the current compute (`mhc/post_cuda.py:45`). `T.macro` is an inlined helper (`quant/common.py:251`). `T.call_extern`/`T.import_source` inject raw CUDA/PTX (`swiglu_forward_cuda.py:11-16, 91, 189`). `T.pdl_sync/pdl_trigger` is programmatic dependent launch (sm_90).
- `T.gemm` (tensor-core tile matmul) is part of TileLang but **not used by any TileKernels CUDA file** (grep); these kernels are memory-bound [external for the API].

Why DeepSeek uses it [doc README:3 + inference]: one Python source per kernel family, parameterized and JIT-specialized; explicit layouts without C++ template metaprogramming; a second backend (Ascend) behind the same API; and kernels "close to the hardware's compute or memory bandwidth limits". For the learner, each construct has a hand-written CUDA equivalent. A fragment is a per-thread register array with an index map. `T.copy` is `ld.global.v4`. `T.Parallel` is a thread-strided loop. `T.reduce_*` is a shuffle, then smem, then shuffle. `T.Pipelined` is a `cp.async` double buffer.

---

## 10. Testing and benchmarking organization, and lessons for Rust

### 10.1 DeepSelect [code]
- **Property-based correctness** (`tests/test.py:80-147`): NaN rows flagged `0x3F3F3F3F` (:87-90); `0 ≤ idx < len` (:97-101); unique indices (:103-109); `values == input[idx]` **bitwise** (:120-125); **`min(selected) ≥ max(unselected)`** (:127-133); index-sorted / value-sorted when requested (:135-147). All comparisons are exact (`kernelkit/compare.py:16-25`).
- **Adversarial distributions** (`tests/test.py:212-232`, `lib.py:44-187`): normal with shifts and scales (±100, ×1e±2..4); uniform over *bit patterns* up to ±inf; tiny bit ranges `[0,1)`, `[0,0x20)`, `[0,0x1000)` (massive ties, denormals); and a hotspot distribution that plants up to 8 ±NaN values (:230-231). `lib.py:158-178` can also plant a chosen k-th value with up to 1.1·k copies, but the sweep does not use it (pivot=None). Sizes cover V=1, random, 2^23−1, and k up to 4096 (:200-210).
- Random seeds come from a counter (`test.py:25-30`); the stride alignment of test inputs is honored (`lib.py:233-236`).
- **Perf mode** `--perf-only` and `--dtype`; runs to finish with `-rf` (`test.py:177-183, 261-262`; `lib.py:243-245`). The **build-time register spill check** is `setup.py:168-181`.

### 10.2 TileKernels [code]
- **Reference = PyTorch functions in `tile_kernels/torch/`**, written to mirror kernel arithmetic order (e.g. `torch/cast.py:240-243, 257-261` "Bitwise with implementation multiplication order").
- **Checks**:
  - `assert_equal` compares bytes, dtype, shape and **stride** (`testing/numeric.py:5-30`).
  - `calc_diff` = `Σ(x−y)² / Σ(x²+y²)` in fp64, used with a threshold where FMA order differs (RoPE ≤ 1e-8) (:33-38).
  - `check_bias` is a **binomial/CLT test** that RNE or SR errors are not systematically signed: `|P(x<ref)+½P(=) − ½| < 10/√n` (:41-64).
  - `quant_level_rank` checks that stochastic and RNE outputs are adjacent codes (`testing/quant.py:7-27`; used in `test_per_token_cast.py:233-240`).
  - The per-token test asserts data and scales bitwise, bias, a non-contiguous input giving an identical result, and the precomputed-scale and scale-only modes (`test_per_token_cast.py:93-161`).
- **Test levels**: `TK_TEST_LEVEL` 0=core, 1=default, 2=full (`testing/generator.py:16-21`). `generate_samples` returns the **full Cartesian product only at level 2**; otherwise it zips dimensions so every value appears at least once (:24-38). Corner shapes such as 0 or 1 tokens only appear at level 2 (:41-51).
- **Benchmarks**: `@pytest.mark.benchmark` tests are skipped unless `--run-benchmark` (`testing/pytest/benchmark.py:47-53, 140-145`). The timer is CUPTI `do_bench(warmup=0, rep=30)` (:779-802). A record is JSONL `{kernel, operation, params, time_us, bandwidth_gbs, …}` (:691-776). Baselines are sharded per test file, with a regression gate of **>5% slower and ≥0.8 µs** (`tests/conftest.py:26-28`; `benchmark.py:260-298`) and `--refresh-baseline` (:72-77).
- **Determinism**: per-test seed = `--seed` + sha256(nodeid) (`testing/pytest/random.py:15-18`). Also: xdist GPU binding and memory fraction per worker (`tests/conftest.py:53-137`), and precompile passes (`testing/pytest/precompile.py`).

### 10.3 Lessons for the Rust engine [inference]
1. **Every kernel gets a reference function**, a plain scalar Rust port (e.g. `sample_topp` straight from `run.c:624-665`), used as the oracle. Where the arithmetic order can be mirrored, test **bitwise**; where it cannot (SIMD sum order, FMA, GPU), use a **measured tolerance** (the D1 method), never a guessed one.
2. **Selection kernels are tested by properties, not indices** (DeepSelect's five). Write `check_topk_properties(x, idx, vals, k)` once and reuse it for CPU heap, radix, filter, and CUDA variants.
3. **Samplers that change the coin→token map are tested statistically**: draw N samples on a fixed distribution and run a chi-square/G-test vs the target probabilities with a principled threshold (like `check_bias`'s 10/√n). Keep one bit-exact llama2.c-compatible sampler for the M4 checkpoint.
4. **Adversarial generators**: all-equal, one finite and the rest −inf, NaN and ±inf, ±0, denormals, **ascending-sorted rows** (the worst case for a linear-order threshold scan), tiny bit-range ties.
5. **Levels**: fast `cargo test` = core (like level 0); `cargo test -- --ignored` or an env var such as `IE_TEST_LEVEL=2` = the full sweep with zipped vs Cartesian parameter sets.
6. **Benchmark mode separate from tests**: a criterion or `--bench` binary that prints µs and **effective GB/s** (bytes touched ÷ time) and appends JSONL. Compare to a checked-in baseline with a threshold, the TileKernels gate idea. Every number gets its command, per CLAUDE.md.
7. **Seeds from test names** so failures reproduce.

---

## 11. Milestone map

| Milestone | What to take from these repos | Tag |
|---|---|---|
| M1 (kernels) | softmax structure; `distort`/`total_cmp` ordering; bitwise vs tolerance testing styles | CPU |
| M3 (forward/RoPE) | interleaved vs NeoX RoPE; cos/sin cache vs recompute; FMA-order tolerance (`calc_diff`) | CPU |
| **M4 (sampling)** | exact llama2.c sampler (verified §3.3); tie-order decision; property + statistical tests; phase timers + n0 logging | CPU |
| **M5 (quant)** | per-token group cast = runq Q8_0; round-ties decision; zero-amax guard; divide vs sf_inv; scale dtype; group size; FP4 E2M1 as a 4-bit option; `check_bias` | CPU |
| **M6 (fast)** | AVX2 hit-mask filter top-k/top-p; mass-histogram pivot; vectorized exp + "fast certificate"; Amdahl measurement of sampling share | CPU |
| **M7 (CUDA)** | K1–K7 sampling kernels on sm_89 (§6); effective-bandwidth metric; L2-flush timing; CPU oracle validation; no TMA or clusters | sm_89 |
| M8 (real model) | V=32K (TinyLlama) vs ~152K (Qwen-class [external]) changes sampling cost ~5×; logits no longer L2-resident on CPU | CPU+sm_89 |
| post-v1 server | batched sampling (one row per worker/CTA); batch-invariant tie-breaks; MoE routing hash as a load-balancing example (role project 3) | both |

---

## 12. Quiz questions (with expected answers)

1. *Why does DeepSelect visit blocks in random order?* A sorted-ascending input would make every element beat the threshold. Random order makes the expected appends at step i at most L/i, so the total is O(L·log m).
2. *Why is `x > T` (strict) enough for correctness?* T is the current k-th largest, so anything equal is already represented. The final radix select plus the equal quota decides among equal values.
3. *Walk the top-3-of-16 example: what is the pivot, and how many elements does the quota let through at the pivot?* The pivot is 0x7C, gt=2, quota=1.
4. *Why might DeepSelect not return the lowest index among ties, and when does that matter?* Buffer order equals the permuted visit order. It matters for bitwise tests and batch-invariant serving.
5. *What is the "distort" map for a negative float and why?* Flip all bits, because a larger negative magnitude must map to a smaller uint.
6. *Why is the first radix byte weak on fp32 logits?* It is sign + 7 exponent bits, so each bucket spans a factor of 4.
7. *Why is llama2.c's cutoff `(1−p)/(n−1)` safe?* See the proof in §3.3.
8. *Your M6 top-p returns the same distribution but different tokens than llama2.c for the same seed. Bug?* Not necessarily: the inverse CDF depends on order. Test statistically.
9. *What metric fits a top-k kernel and why not FLOPs?* Effective bandwidth, because there is no arithmetic to count. Compare against measured peak.
10. *Why can the batch-6 sampler reach only ~0.1 TB/s when batch 4096 reaches ~2.8?* One CTA per row leaves SMs idle and launch/latency dominates; it is latency-bound, not bandwidth-bound.
11. *Why does DeepSelect not run on the 4070, and what would you replace?* It is compiled for sm_100a/103a only and uses TMA, mbarrier tx, clusters, and 203 KiB smem. Replace with cp.async, a multi-kernel merge, and a smaller B/B2.
12. *Q8_0: what does `round(-24.5)` give in C vs torch?* −25 vs −24. Which one does your reference use?
13. *Why does a power-of-two (UE8M0) scale hurt int8 but not FP8?* FP8 keeps its relative precision under an exponent shift; int8 loses levels.
14. *What does fusing SwiGLU+quant save, in bytes per token?* ~13H → ~5H for a bf16 input with an fp32 intermediate.
15. *How can a kernel use a fast `expf` and still be bitwise-equal to a precise reference?* Certify the distance from the rounding boundary and fall back when too close.
16. *Why does `check_bias` use `10/√n`?* The less-than ratio is ~N(0.5, 1/(4n)), so 10/√n is ~20σ: a principled, non-flaky threshold.

---

## 13. Open questions (to resolve with measurements or later reading)

1. What is n0 (llama2.c's cutoff survivors) and the nucleus size on stories15M at T=1.0/topp=0.9, per step? This decides whether qsort matters at all in M4.
2. What is sampling's share of per-token time before M6, and after the matmuls get AVX2 + 24 threads? Predict first, then measure.
3. Does a reduced DeepSelect config fit sm_89 (≈91 KiB) and beat a simple one-CTA radix select for V=32K at batch 1, given that decode logits are L2-hot?
4. Does fp32 DeepSelect's tie behavior really stay batch-invariant (inference from `api.cpp:196-209`), and bf16's not? This needs Blackwell hardware to test; for the learner, design the CUDA sampler to be batch-invariant by construction (fixed per-row config).
5. What group size and scale dtype minimize quality drift per byte on stories15M (M5), measured with D1-style logit diffs? Should activations use the same group size as the weights, as runq does?
6. For M7: which RNG scheme lets the CPU oracle and the GPU sampler produce the *same* tokens (counter-based per (seed, step, row))? Or is the checkpoint statistical only? This needs a DECISIONS.md entry.
7. ggml's Q8_0/Q4_0 exact layouts (block of 32, fp16 scale) and its AVX2 int8 dot trick: verify in llama.cpp when it is cloned at M5 (not in `~/refs/inference` yet).
