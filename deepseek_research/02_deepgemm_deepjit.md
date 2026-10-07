# DeepGEMM + DeepJIT — tutor's deep notes

**Purpose.** Private reference for the tutor guiding Luigi through `inference-engine` (Rust CPU LLM
inference engine, CUDA backend in M7). These notes translate DeepSeek's production GEMM library into
things Luigi can *use* at M5 (int8/4-bit quantized matmul), M6 (AVX2 + threads + batched prefill) and
M7 (CUDA on sm_89). Every architectural idea is tagged with where it applies.

## Sources and verification

| Repo | Path | Commit | Date |
|---|---|---|---|
| DeepGEMM | `~/refs/inference/deepseek/DeepGEMM` | `057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7` ("Public release 26/09/30 (#462)") | 2026-09-30 |
| DeepJIT | `~/refs/inference/deepseek/DeepJIT` | `3732a3b9a0a2e5126396908a362ba5abdf043b2f` ("Migrate DeepJIT CUDA integration to the PyTorch stable ABI (#11)") | 2026-09-30 |

Submodule pins recorded in DeepGEMM's tree (`git ls-tree HEAD third-party/`):
CUTLASS `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`, DeepJIT `2efdab421e1cfb17fe8bc20e11ca72aa6d0e6c43`
(note: the pinned DeepJIT commit differs from the standalone checkout we read; the standalone one is
newer). Both submodule working directories are **empty** in this checkout, so nothing here is based on
CUTLASS source — only on DeepGEMM's own calls into it.

Notes written 2026-09-30. Every `file:line` below was checked with `grep -n` / `sed -n` at these
commits. Claims that are *not* in the code or docs are marked **[inference]**. Numbers are only quoted
where the repo states them; nothing is estimated. `[external, unverified]` marks things from my own
background knowledge, not from these repos.

**Hard practical constraint for the tutor to state early:** DeepGEMM cannot be built or run on
Luigi's machine. README:31-36 requires "NVIDIA SM90 or SM100 architecture GPU" and "CUDA Toolkit 12.9
or higher"; DeepJIT asserts NVCC ≥ 12.9 at `include/deep_jit/backend/cuda/backend.hpp:72`. Luigi has
sm_89 (Ada) and nvcc 12.0. So this repo is **reading material and an idea source**, never a
dependency, and never a benchmark baseline he can reproduce. For a runnable sm_89 baseline he needs
cuBLAS(Lt) or llama.cpp's CUDA backend instead.

---

## Executive summary (the 15 things that matter for this project)

1. **DeepGEMM is a JIT-first library.** Nothing GPU-side is compiled at install time. Host C++ builds
   a tiny source string that instantiates one kernel template with *all shapes and tile sizes as
   compile-time constants*, then DeepJIT nvcc-compiles it to a CUBIN, caches it on disk by content
   hash, and launches it through the CUDA driver API (`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:29-71`).
2. **Two FP8 scaling "shapes", named in the code.** `Kernel1D1D` = 1-D scales on A *and* B (per token
   / per channel). `Kernel1D2D` = 1-D per-token scales on A, 2-D 128×128 block scales on B. Dispatch
   is literally `gran_n == 1 ? 1d1d : 1d2d` (`csrc/apis/gemm.hpp:120-126`).
3. **Where the scale is applied is the whole design.** On Hopper the tensor core knows nothing about
   scales: WGMMA accumulates a `BLOCK_K = 128` slice in FP32 registers, then CUDA cores multiply by
   `scale_a * scale_b` and add into a *second* FP32 accumulator
   (`deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh:251-254, 329-345`). That "promote every
   128 K-elements" pattern is exactly what Luigi's int8 group-of-32 CPU matmul must do with i32
   accumulators.
4. **On Blackwell the hardware does it**: `tcgen05.mma...kind::mxf8f6f4.block_scale`
   (`deep_gemm/include/deep_gemm/ptx/tcgen05.cuh:59,80`) consumes scale factors staged in tensor
   memory. Neither the instruction nor tensor memory exists on sm_89 — skip entirely.
5. **Scales must be a power of two** in the UE8M0 path, and the library computes the exponent with an
   integer-carry trick, no `log2`/`frexp` (`deep_gemm/include/deep_gemm/common/math.cuh:114-135`).
   Great teaching hook for M5: power-of-two scales make dequant exact and cheap.
6. **Scale-factor memory layout is a first-class artifact**: MN-major, TMA-16-byte-aligned, 4 UE8M0
   exponents packed per `int32`, with a 472-line spec (`docs/scaling-factor-format.md`). The lesson
   for Luigi: quantized formats are *layouts*, not just dtypes, and the layout is chosen by what the
   load instruction wants.
7. **Persistent kernel + software pipeline + warp specialization**: grid = #SMs, each block loops
   over output tiles (`deep_gemm/include/deep_gemm/scheduler/gemm.cuh:186`); one warpgroup only issues
   TMA loads, the others only do MMA; they hand off through mbarriers with `kNumStages` shared-memory
   buffers (`.../impls/sm90_fp8_gemm_1d2d.cuh:165-215`). CPU analog: a producer thread packing panels
   vs worker threads running the microkernel; sm_89 analog: `cp.async` multi-stage pipeline.
8. **Registers are re-partitioned between roles** with `warpgroup_reg_dealloc<40>()` /
   `warpgroup_reg_alloc<248>()` (`.../impls/sm90_fp8_gemm_1d2d.cuh:145-146, 167, 215`) — i.e. PTX
   `setmaxnreg`. Hopper+ only; on sm_89 the register budget is fixed per kernel.
9. **Tile scheduling is rasterized for L2 reuse** with a group size chosen by minimizing bytes
   touched per wave: candidates {8, 16}, cost `candidate*BLOCK_N + ceil(SMs/candidate)*BLOCK_M`
   (`.../scheduler/gemm.cuh:14-26`, applied at `108-144`). Directly portable to sm_89 and conceptually
   to CPU cache blocking.
10. **Block sizes are not powers of two.** SM90 `BLOCK_N` enumerates every multiple of 16 up to
    160/192/256 depending on kernel type (`csrc/jit_kernels/heuristics/sm90.hpp:40-57`), and the WGMMA
    selector supports N = 8, 16, …, 256 in steps of 8 (`deep_gemm/include/deep_gemm/mma/sm90.cuh:36-67`).
    So 112 or 176 are real, chosen options — they exist to divide the problem into a whole number of
    SM waves.
11. **The config chooser is a small analytic performance model** — L1 and L2 byte traffic per tile
    divided by modeled bandwidths, inflated by wave inefficiency
    (`csrc/jit_kernels/heuristics/sm90.hpp:202-239`). This is role project 4 in 40 lines of C++, and
    the best template Luigi has for M0's predicted-vs-measured table.
12. **Correctness is measured with a similarity metric, not max-abs-diff**:
    `calc_diff = 1 - 2·Σxy / Σ(x²+y²)` (`deep_gemm/testing/numeric.py:5-11`), thresholds 0.001 (FP8),
    0.01 (one FP4 operand), 0.02 (FP4×FP4) (`tests/generators.py:69-74`), 1e-5 for BF16
    (`tests/test_bf16.py:46`). Plus **bitwise** determinism checks across 20 reruns
    (`tests/test_fp8_fp4.py:78-82`).
13. **Benchmarks flush L2 (256 MB) before every measured iteration**, warm up, and time with CUDA
    events or with the PyTorch/kineto profiler filtered to one kernel name
    (`deep_gemm/testing/bench.py:9-12, 25-33, 92-117`). Reported as µs | TFLOPS | GB/s | speedup vs
    cuBLAS (`tests/test_fp8_fp4.py:108-111`).
14. **The public API is deep**: ~6 required args (`a, b, d`, plus a recipe), everything else defaulted;
    all tiling/staging/multicast decisions are hidden. The knobs that exist are global
    process-level settings (`csrc/apis/config.hpp:8-48`), not per-call tuning parameters.
15. **DeepJIT is the reusable half**: a header-only C++20 runtime whose cache key is
    `extra_signature + nvcc --version + effective flags + post-hook hash + source-and-tracked-includes
    hash` (`include/deep_jit/runtime/runtime.hpp:46-54, 92-98`), published to disk by *atomic
    directory rename after fsync* (`include/deep_jit/cache/disk.hpp:42-67`). That design is the thing
    to discuss when Luigi decides how M7 gets PTX/CUBIN from Rust.

---

## Part 1 — Repo map

### 1.1 DeepGEMM top level

```
README.md                      213 lines — overview, requirements, API tour, env vars
docs/scaling-factor-format.md  472 lines — the SF contract (shapes, dtypes, strides, transforms)
CMakeLists.txt                 "just for CMake-based IDE indexing, the real compilation is done via JIT" (:1)
setup.py                       builds ONE .so from csrc/python_api.cpp; no device code (see 1.5)
develop.sh / install.sh / build.sh  symlink CUTLASS headers into deep_gemm/include, build, install
csrc/                          host C++ (the whole CPU side)
deep_gemm/include/deep_gemm/   CUDA kernel headers, compiled at runtime
deep_gemm/                     Python package (thin: re-exports _C, plus utils/testing/legacy)
tests/                         14 test files, 4374 lines total — also the benchmark suite
scripts/                       generate_pyi.py, quick_plot_pm.py (NCU metric plots), run_ncu_mega_moe.sh
third-party/                   cutlass, deep_jit (submodules, empty here), tilelang_ops
```

### 1.2 `csrc/` — host side

| Path | Job |
|---|---|
| `python_api.cpp` (49) | The only compiled translation unit. Registers DeepJIT's `get_jit()` plus each API group (`:24-48`). |
| `apis/gemm.hpp` (997) | User-facing GEMM entry points + `register_apis` with all pybind defaults (`:835-997`). |
| `apis/attention.hpp` (478) | MQA-logits (lightning indexer) entry points, paged and sparse variants. |
| `apis/einsum.hpp` (311) | Hard-coded einsum expressions over batched GEMM; permutes operands *and* their SFs (`docs/...md:360-366`). |
| `apis/layout.hpp` (144) | `transform_sf_into_required_layout`, TMA-aligned/packed SF helpers. |
| `apis/mega_moe.hpp` (400) | Fused MoE mega-kernel (dispatch + 2 GEMMs + SwiGLU + combine) over symmetric memory. |
| `apis/mega_gate.hpp`, `apis/mega_mhc.hpp`, `apis/hyperconnection.hpp` | Gating, multi-hyperconnection, TF32 prenorm GEMM. |
| `apis/epilogue_class.hpp` (27) | Pluggable epilogue objects (e.g. `BF16StochasticRounding`, `QuantizeToFP8`). |
| `apis/config.hpp` (50) | The entire global knob surface (see §9). |
| `apis/locality_domain.hpp` (162) | Allocate/localize tensors per device "locality domain"; probe which domain each SM sits in. |
| `jit_kernels/impls/*.hpp` | One class per kernel family: build the instantiation source, make TMA descriptors, launch. |
| `jit_kernels/heuristics/{common,config,runtime,utils,sm90,sm100}.hpp` | The config chooser (§5). |
| `jit_kernels/impls/smxx_cublaslt.hpp` (237) | cuBLASLt wrappers, used as the benchmark baseline. |
| `runtime/jit.hpp` (34) | Constructs the DeepJIT runtime with prefix `"DG"`, signature `"cutlass-<version>"`, include dir, tracked prefix `deep_gemm/`. |
| `runtime/runtime.hpp` (106) | Global runtime state: num_sms, tc_util, cuBLASLt handle/workspace, SM locality domains. |
| `utils/{layout,math,exception,compatibility}.hpp` | Arg validation (`check_sf_layout`), `ceil_div`/`align`, `DG_HOST_ASSERT`. |
| `indexing/main.cu` (36) | Includes every kernel header so IDEs index them; not part of the real build. |

### 1.3 `deep_gemm/include/deep_gemm/` — device side (16,220 lines total)

| Group | Files (lines) |
|---|---|
| **GEMM impls** | `sm90_fp8_gemm_1d2d.cuh` (452), `sm90_fp8_gemm_1d1d.cuh` (362), `sm90_bf16_gemm.cuh` (399), `sm100_fp8_fp4_gemm_1d1d.cuh` (579), `sm100_bf16_gemm.cuh` (432) |
| **MoE mega-kernels** | `sm100_fp8_fp4_mega_moe.cuh` (1528), `sm100_bf16_mega_moe.cuh` (1316) — the two biggest files |
| **Attention / indexer** | `sm90_fp8_mqa_logits.cuh` (324), `sm90_fp8_paged_mqa_logits.cuh` (334), `sm100_mqa_logits.cuh` (609), `sm100_sparse_mqa_logits.cuh` (688) |
| **Other ops** | `sm{90,100}_bmk_bnk_mn.cuh` (einsum), `sm{90,100}_tf32_hc_prenorm_gemm.cuh`, `sm100_mega_gate.cuh`, `sm100_mega_mhc.cuh` (704), `sm100_locality_domain.cuh` (26), `smxx_layout.cuh` (295, the SF transform kernels) |
| **`mma/`** | `sm90.cuh` (293) WGMMA selectors + smem descriptors; `sm100.cuh` (160) UMMA |
| **`ptx/`** | `wgmma.cuh` (25) fence/commit/wait; `tcgen05.cuh` (219) Blackwell MMA; `tma.cuh` (155); `ld_st.cuh` (387) `ldmatrix`/`stmatrix`/vectorized shared loads |
| **`scheduler/`** | `gemm.cuh` (314) the tile scheduler; plus MoE / MQA / paged schedulers |
| **`epilogue/`** | `operators.cuh` (94), `sm100_store_cd.cuh` (220), `sm100_store_cd_swap_ab.cuh` (259), MoE/gate epilogues |
| **`common/`** | `math.cuh` (236), `tma_copy.cuh` (106), `ring_pipeline.cuh` (40), `utils.cuh` (52, `PatternVisitor`), `probe.cuh` (91, per-warp `%clock64` tracing), `types.cuh`, `packing.cuh` |
| **`layout/`, `comm/`** | tensor/SF layout helpers; `barrier.cuh` (94) cluster/multi-GPU barriers |

### 1.4 Kernel inventory (what actually exists)

- **Dense FP8/FP4 GEMM**: `fp8_fp4_gemm_{nt,nn,tn,tt}` (aliases `fp8_gemm_*`, `fp4_gemm_nt`).
  SM90 supports NT only; SM100 all four layouts (README:63).
- **M-grouped contiguous** (MoE prefill/training fwd): `m_grouped_fp8_fp4_gemm_{nt,nn}_contiguous`,
  M-axis grouped only, N and K fixed, each expert segment aligned to the M block size (README:78).
- **M-grouped masked** (MoE decode under CUDA graphs): `m_grouped_fp8_fp4_gemm_nt_masked`, computes
  only the valid rows given a mask (README:82-86).
- **K-grouped contiguous** (MoE weight-gradient): `k_grouped_fp8_gemm_tn_contiguous`,
  `k_grouped_fp8_gemm_nt_contiguous`, `k_grouped_fp4_gemm_nt_contiguous` (README:80, `docs/...md:171-197`).
- **BF16 GEMM**: `bf16_gemm_{nt,nn,tn,tt}`, m-grouped contiguous/masked, k-grouped.
- **Batched**: `GemmType::Batched`, plus `batched_syrk` / `batched_symm` via cuBLASLt
  (shapes documented as "Real Muon parameter shapes", `tests/generators.py:176-186`).
- **MQA logits / lightning indexer (DeepSeek V3.2)**: `fp8_fp4_mqa_logits`,
  `fp8_fp4_paged_mqa_logits`, sparse variants, plus metadata builders
  (`csrc/apis/attention.hpp:446-476`). Semantics in README:88-110: per query token, ReLU the
  per-head dot products, weight them, sum to a scalar logit.
- **Mega MoE**: `fp8_fp4_mega_moe` / `bf16_mega_moe` — fuses EP dispatch, linear1, SwiGLU, linear2,
  EP combine, overlapping NVLink comms with tensor-core work (README:112-144).
- **Mega gate / mHC / HyperConnection / einsum / SF-layout transforms**.
- **cuBLASLt passthroughs** — deliberately present so tests can print a speedup ratio.
- **Legacy Triton kernels for A100** (`deep_gemm/legacy/`, 383 lines, `__init__.py:1`: "All kernels may
  be deprecated in the future (or rewrite in TileLang)"). Worth showing Luigi: a plain Triton
  m-grouped BF16 GEMM (`legacy/m_grouped_gemm.py:12-58`) is the *simplest complete* grouped GEMM in
  the repo and reads like pseudocode.

### 1.5 The build: almost nothing is AOT

`setup.py:32` — `sources = ['csrc/python_api.cpp']`. One CUDAExtension, `-std=c++20 -O3`, include dirs
for `deep_gemm/include`, `third-party/deep_jit/include`, `third-party/cutlass/include` (`:27-45`).
No `.cu` in the build. All device code ships as headers and is compiled by nvcc at first call.
`CMakeLists.txt:1` says so explicitly. This is the single biggest structural difference from
llama.cpp/candle and the thing to discuss in M7.

---

## Part 2 — What the README claims

### 2.1 Design philosophy (quotable)

- README:3 — "a unified, high-performance tensor core kernel library that brings together the key
  computation primitives of modern large language models … into a single, cohesive CUDA codebase. All
  kernels are compiled at runtime through DeepJIT, requiring no CUDA compilation during installation."
- README:5 — "leverages some concepts from CUTLASS and CuTe, but **avoids heavy reliance on their
  templates or algebras**. The library is designed for simplicity, with only a limited number of core
  kernel functions, making it a clean and accessible resource for learning NVIDIA GPU kernel
  optimization techniques."
- README:7 — "Despite its lightweight design, DeepGEMM's performance matches or exceeds expert-tuned
  libraries across various matrix shapes."
- Citation title (README:207) — "DeepGEMM: clean and efficient BLAS kernel library on GPU".

**Reality check on "small".** The claim holds for the *core GEMM*: the Hopper FP8 1D2D kernel is 452
lines including comments. It does not hold library-wide: the FP8 MoE mega-kernel is 1528 lines and the
device tree totals 16,220 lines. Tell Luigi both halves — "small kernel, large library" — because the
lesson he can use is the first one. **[inference]** on the framing; line counts are measured.

### 2.2 Performance claims

**There is no performance table in this README.** The only absolute number anywhere in it is
README:25: "2025.04.18: DeepGEMM now achieves up to **1550 TFLOPS** on H800!" with no shape or dtype
stated in the README (links out to PRs #74/#78/#81/#86 for details). Earlier public versions of this
README carried a shape-by-shape table; this snapshot has one squashed commit
(`git log --follow README.md` → a single commit) so there is no local history to quote.

Do **not** invent DeepGEMM numbers. What the repo *does* give you is a reproducible measurement
recipe, which is the part Luigi needs anyway (§8): every test prints
`us | TFLOPS | GB/s | <x> cuBLAS speedup` per shape, and `tests/generators.py:170-173` prints the
**geometric mean** speedup over cuBLASLt across all shapes in the sweep.

Shapes the library considers representative (`tests/generators.py:127-129`), useful when Luigi asks
"what do production GEMM shapes look like?":
`(n,k)` ∈ {(2112,7168), (576,7168), (24576,1536), (32768,512), (7168,16384), (4096,7168), (7168,2048)}
for BF16 output; {(256,7168), (129280,7168)} for FP32 output; forward M ∈ {1, 128, 4096}, backward
M = 4096. Note M = 1: **decode is a GEMV** and it is in the sweep. That is exactly Luigi's M4 decode
phase, and exactly why the heuristics have special cases for `m <= 16` / `m <= 32`
(`csrc/jit_kernels/heuristics/sm90.hpp:25-26`).

### 2.3 Requirements and supported architectures

README:29-36: SM90 or SM100 GPU; Python ≥ 3.8; C++20 with `<format>`; **CUDA Toolkit ≥ 12.9**;
PyTorch ≥ 2.3; CUTLASS ≥ 4.0. Ada (sm_89) is not supported at all — the kernels are guarded with
`#if __CUDA_ARCH__ >= 900` and trap otherwise (`.../impls/sm90_fp8_gemm_1d2d.cuh:54, 444-447`:
`DG_DEVICE_ASSERT(false and "This kernel only support sm_90a")`).

Naming convention (README:63): `D = C + A @ B`, input layout NT by default, so `fp8_gemm_nt` computes
`D = C + A @ B.T`.

### 2.4 Environment variables (the debugging surface)

README:166-191. Worth mirroring in spirit for M7: `DG_JIT_DEBUG`, `DG_PRINT_CONFIGS` (prints the
chosen config per shape — see §5), `DG_JIT_CACHE_DIR` (colon-separated, first hit wins, misses
compile into the first path), `DG_JIT_PTXAS_VERBOSE`, `DG_JIT_CHECK_NO_SPILLS`,
`DG_JIT_CHECK_NO_LOCAL_MEMORY`, `DG_JIT_DUMP_PTX` / `DUMP_SASS`, `DG_JIT_PRINT_LOAD_TIME`,
`DG_USE_NVIDIA_TOOLS` (skip internal profiling under nsys/ncu/compute-sanitizer).

**Teaching hook:** "assert no register spills" and "assert no local memory" as *build-time* gates is a
performance-engineering habit worth stealing. sm_89 nvcc supports `--ptxas-options=--warn-on-spills`
too, so Luigi can adopt it verbatim in M7.

---

## Part 3 — FP8 scaling, and how it maps onto M5's int8

### 3.1 Granularity: the "recipe"

**What.** `recipe = (gran_m, gran_n, gran_k)` says how many elements one scale covers, for *storage*
(`docs/scaling-factor-format.md:17`). Given `A: [M,K]`, `B: [N,K]`:

```
SFA: [ceil_div(M, gran_m), ceil_div(K, gran_k)]  float32
SFB: [ceil_div(N, gran_n), ceil_div(K, gran_k)]  float32
```
(`docs/...md:22-24`). Supported `gran_k`: 32 or 128 on SM100, **128 only** on SM90 (`docs/...md:30`).

The two production recipes:
- **Activations**: `(1, 128)` — one scale per token per 128-element K group ("per-128-channel
  scaling", asserted in the kernel: `DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel
  FP8 scaling")`, `.../impls/sm90_fp8_gemm_1d2d.cuh:56`).
- **Weights**: `(128, 128)` — one scale per 128×128 block. This is the 1D2D kernel.
- Everything-1-D `(1,1,128)` is the 1D1D kernel, used for weight-gradient GEMMs and on SM100 generally.

**Why blocks for weights and rows for activations?** **[inference]** Activation outliers are per-token
and change every step, so you must quantize on the fly along K; weights are static and can afford a
coarser 2-D grid that is cheaper to store and to broadcast. The dispatch confirms the coupling:
`gran_n == 1 → 1d1d`, else `1d2d` (`csrc/apis/gemm.hpp:120-126`).

**How the scale is computed** (reference, host-side, `deep_gemm/utils/math.py:26-49`):

```python
x_amax = x_view.abs().float().amax(dim=2).clamp(1e-4)   # per (token, 128-group)
sf     = x_amax / 448.0                                  # 448 = max finite e4m3
sf     = ceil_to_ue8m0(sf)                               # round UP to a power of two
x_fp8  = (x_view / sf).to(torch.float8_e4m3fn)
```

`ceil_to_ue8m0` (`math.py:13-16`) bumps the exponent if any mantissa bit is set, then clamps to
[1,254]. The device-side equivalent does it with one integer add and a shift
(`deep_gemm/include/deep_gemm/common/math.cuh:114-135`):

```cpp
const auto rounded_exp = (amax_bits + kMantissaMask - kQuantMaxMantissa) >> kMantissaBits;
return cute::max(rounded_exp, kMinSFExponent + kQuantMaxExponent) - kQuantMaxExponent;
```
with the comment "the carry of the integer addition performs the exponent ceiling" (`:111-113`) and
`kQuantMaxMantissa = mantissa(1.75)`, `kQuantMaxExponent = 8` because `448 = 1.75 · 2^8` (`:123-124`).
The reciprocal is `__uint_as_float((254u - sf_exp) << 23)` — exact, no division (`:140-146`).
FP4 e2m1: max `6 = 1.5 · 2^2`, `sf = amax / 6.0` (`deep_gemm/utils/math.py:109`).

**Applicability.** CPU/Rust **yes** — the power-of-two-scale idea transfers directly to int8 and is a
real design decision for M5 (see 3.5). sm_89 **yes**. The UE8M0 *packing* is a Blackwell requirement,
not a good idea in itself on other targets.

### 3.2 Scale-factor layout: MN-major, TMA-aligned, packed

**What.** A pre-transformed SF tensor for `mn` rows and `k` columns at granularity `(1, gran_k)` must be
(`docs/...md:53-61`):

```
dtype  : int32 (4 UE8M0 exponents packed)
shape  : [mn, ceil_div(k, gran_k*4)]
stride : (1, align(mn, 4))       # MN-major; last-dim stride is a multiple of 16 bytes
```

i.e. memory is `packed_sf_k` consecutive slices, each `align(mn,4)` int32s, padding never read
(`docs/...md:437-447` has the diagram). Packing is little-endian by K index
(`docs/...md:424-432`):

```cpp
packed |= (values[0] >> 23u);   // exp of sf[4k+0] -> bits [7:0]
packed |= (values[1] >> 15u);   // ...
packed |= (values[2] >>  7u);
packed |= (values[3] <<  1u);
```

**Why MN-major?** Because the consumer is a TMA load of a `BLOCK_M`-long column of scales for one K
group: contiguous in MN means one coalesced/bulk transfer per stage. You can see the shape of that
load in the SM90 kernel: the SFA TMA copy is `tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, ...)`
with per-stage smem `BLOCK_M * sizeof(float)` (`.../impls/sm90_fp8_gemm_1d2d.cuh:79, 195-197`).
Why 16-byte alignment: "TMA requires `stride(-1)` to be a multiple of 16 bytes" (`docs/...md:40`).

**Pitfall documented in the repo** (worth repeating to Luigi as a general lesson about layouts):
`docs/...md:371-378` — you must not `view`/`reshape` a transform output into another rank, because the
output is MN-major and reshaping silently makes it contiguous, breaking `stride(-2) == 1` and tripping
a host assert. Layout invariants must be *checked*, and DeepGEMM does check them
(`check_sf_layout`, `csrc/utils/layout.hpp:101`).

**Also documented**: SF values must be exact powers of two, enforced *on device*:
`DG_TRAP_ONLY_DEVICE_ASSERT((value & 0x807fffffu) == 0)` (`docs/...md:33`).

**Applicability.** sm_89 **partially**: no TMA, so no 16-byte-stride requirement from TMA — but the
same "make the scale array contiguous along the dimension the warp reads" argument holds for
`ldg`/`cp.async`. CPU **yes** in spirit: for an int8 group-of-32 matmul, whether scales are stored
row-major-by-group or grouped-by-column changes whether the inner loop streams them.

### 3.3 Where the scales are applied — Hopper (the one to teach)

This is the single most instructive piece of code in the repo for M5/M7. In
`deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh`:

```
:252  constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
:254  float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};
...
:286  float scale_b_0 = ptx::ld_shared(smem_sfb + k_block_idx), scale_b_1;
:301  auto scale_a_0 = do_wgmma_store ? ptx::ld_shared(smem_sfa[stage_idx] + r_0 + m_offset) : 0;
:308  ptx::warpgroup_arrive();
:310  for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) { ... WGMMA::wgmma(a_desc, b_desc, accum, k); }
:315  ptx::warpgroup_commit_batch();
:319  ptx::warpgroup_wait<0>();
:331  float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
:338  for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
:341      shifted_accum[i*4+0] += (predicate ? scale_0_0 : scale_0_1) * accum[i*4+0];
          ...
```

**The pattern, in words.** Two accumulators. `accum[]` is the tensor core's FP32 output for *one*
`BLOCK_K = 128` slice — written by 4 back-to-back WGMMAs of K = 32 each
(`WGMMA::K == 32`, `deep_gemm/include/deep_gemm/mma/sm90.cuh:28`). `final_accum[]` is a separate FP32
array in CUDA-core registers. After each 128-K slice the code multiplies `accum` by
`scale_a * scale_b` (FP32 CUDA-core FFMA) and adds into `final_accum`, then the next slice starts the
tensor-core accumulator from scratch (`scale_d = k` passed to `wgmma`, so the first of the four MMAs
zeroes and the rest accumulate — `mma/sm90.cuh:19` maps it to `ScaleOut::One/Zero`).

**Why: two independent reasons, both worth saying out loud.**
1. **Scaling.** The scale differs per 128-K group, so you must close out the group before you can apply
   its scale. The FP8 WGMMA has no per-group scale operand, so the multiply has to happen on CUDA
   cores. The unit of promotion *is* the quantization group.
2. **Precision.** The tensor core's internal FP32 accumulation is lower precision than a true FP32
   add. Promoting into a clean FP32 register accumulator every 128 elements bounds the error.
   **[inference]** — the repo does not state (2) at this commit; the code comment only says
   "Accumulation for WGMMA or CUDA promotion" (`:251`) and "Promote with scales" (`:329`). Present (1)
   as fact and (2) as the widely-reported Hopper behavior, flagged as such.
   **[external, unverified]** the specific "~FP22 mantissa in WGMMA accumulate" claim circulating in
   the community; do not state a bit count as fact.

**Two micro-optimizations in the same block that teach real lessons:**
- `:330` — "making it as predicates is very important for performance, comparing to two loops". When
  `BLOCK_N` straddles two 128-wide B-scale blocks, the naive fix is two loops; the fast fix is one
  loop with a select. Branch-free inner loops matter as much on AVX2.
- `:274-279, 339-340` — `dispatch_num_former_iters` turns a *runtime* count into a compile-time
  constant by recursive template dispatch, so the predicate folds away. This is JIT specialization
  applied *inside* one kernel. The Rust analogue is a `match` over a small set of const-generic
  instantiations.

**Where scale B comes from is itself interesting** (`:239-249`): the SFB values are loaded from global
memory **by the math warpgroups** (not by TMA), deliberately, "except the first warp, we want to
overlap loading B scales with TMA stores between tasks". Latency hiding by choosing *who* issues a
load.

### 3.4 Where the scales are applied — Blackwell (skip, but know why)

`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`:
- The MMA is `tcgen05.mma.cta_group::{1,2}.kind::mxf8f6f4.block_scale` (`ptx/tcgen05.cuh:59, 80`) —
  the instruction takes scale-factor operands.
- Scales live in **tensor memory (TMEM)**, copied there by `UTCCP`
  (`cute::SM100_UTCCP_4x32dp128bit_{1,2}cta`, `:374-389`), and TMEM columns are budgeted explicitly:
  `kNumAccumTmemCols = UMMA_N * kNumEpilogueStages`, `kNumSFATmemCols = SF_BLOCK_M*SF_BLOCK_K/32`,
  total must fit 512 columns (`:121-133`; the heuristic pre-checks it at
  `csrc/jit_kernels/heuristics/sm100.hpp:126-132`).
- SF blocks are padded to 128 (`kNumUTCCPAlignedElems = 128`, `:92-94`) and a dedicated warp
  transposes them in shared memory for UTCCP (`:460-490`).
- So on Blackwell there is **no CUDA-core promotion loop**: the tensor core applies the scales.

**Applicability: none.** TMEM, `tcgen05`, UTCCP are sm_100+. On sm_89 Luigi is in the Hopper-style
world minus TMA/WGMMA, i.e. the §3.3 pattern implemented with `mma.sync`.

### 3.5 Mapping all of this onto M5 (int8 per-group, Q8_0-style)

Luigi's M5 plan: int8, groups of 32 along K, one FP32 scale per group. Here is the translation table
the tutor should draw.

| DeepGEMM (FP8) | Luigi's M5 (int8) |
|---|---|
| e4m3, max finite 448 | i8, max 127 |
| `sf = amax / 448`, rounded up to 2^k | `scale = amax / 127` (llama.cpp Q8_0 uses `d = amax/127`, stored f16) |
| group = 128 along K | group = 32 along K |
| tensor core FP32 accumulator per group | **i32 accumulator per group** (`_mm256_madd_epi16`/`vpmaddubsw`+`vpmaddwd` chains) |
| promote: `final += (sa*sb) * accum` in FP32 | promote: `acc_f32 += (sa*sb) * (i32 dot)` |
| scale applied once per 128 elements, never inside | scale applied once per 32 elements, never inside |

**The lesson to make him state in his own words:** *the quantization group size is the accumulation
chunk size.* You do an exact integer dot product over one group (no rounding at all inside the group),
then convert to float once and apply the product of the two scales. Getting this wrong — scaling per
element, or accumulating floats inside the group — costs both speed and accuracy.

Worked example to use in the lesson (numbers are illustrative, not from the repo):

```
A row group (32 int8):  a[0..31], scale sa
B col group (32 int8):  b[0..31], scale sb
i32_dot = Σ a[i]*b[i]                      # exact; |i32_dot| ≤ 32*127*127 = 516,128 — fits easily in i32
partial = (sa * sb) * (f32) i32_dot        # one multiply per group, not per element
out    += partial                          # f32 accumulation across groups
```

Then ask: how many groups can you sum in i32 before overflow if you *share* a scale (i.e. if sa,sb are
per-row/per-column rather than per-group)? `2^31 / (127*127) ≈ 133k` products ≈ 4160 groups of 32.
That is the quantitative reason per-tensor int8 can accumulate in i32 across the whole K, and
per-group cannot skip the per-group multiply — a nice M5 quiz.

Second lesson from DeepGEMM worth importing into M5: **power-of-two scales**. If Luigi rounds each
group's scale up to a power of two, dequantization becomes an exponent add (exact, no rounding error
from the scale itself) at the cost of up to 2× lost dynamic range in the mantissa
(`common/math.cuh:111-113` + `deep_gemm/utils/math.py:13-16`). Worth a `DECISIONS.md` entry: Q8_0 uses
a general f16 scale; DeepSeek's UE8M0 path insists on powers of two because Blackwell's SF *format* is
an 8-bit exponent. Different constraints, same knob.

---

## Part 4 — Kernel architecture, technique by technique

Each subsection: **what / where / how / why / numbers / applies to / teaching hook / pitfall**.

### 4.1 Persistent kernel + tile scheduler

**Where.** `deep_gemm/include/deep_gemm/scheduler/gemm.cuh:185-272`; used at
`.../impls/sm90_fp8_gemm_1d2d.cuh:153, 173, 227`; grid is set to exactly `num_sms`
(`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:125`: `.grid_dim = dim3(config.launch_config.num_sms, 1, 1)`).

**How.** One block per SM, looping:
```cpp
const auto next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x;   // :186
if (next_block_idx >= num_blocks) return false;                        // :261-262
get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);       // :269
```
The same `Scheduler` object also resolves grouped-GEMM bookkeeping (masked: walk groups until the
flat index lands inside one, `:188-204`; psum layout: `:205-225`; k-grouped: `:226-246`).

**Why.** Block launch/teardown cost is paid once per SM rather than once per tile; per-block setup
(barrier init, TMA descriptor prefetch, SFB load) amortizes over many tiles; and the scheduler can
*choose* the traversal order (next point).

**Applies to.** sm_89 **yes** — persistent kernels need nothing newer than Kepler; you just need to
know `num_sms` (46 SMs on a 4070 **[external, verify with `cudaDeviceProp.multiProcessorCount`]**).
CPU **yes**: the direct analogue is a work-stealing/atomic-counter loop over output tiles across 24
threads instead of `parallel_for` per tile.

**Teaching hook.** Ask Luigi to write the CPU version first: `while let Some(tile) = counter.next()`
with `tiles = ceil(M/BM) * ceil(N/BN)`. Then show `:186` and note it is the same loop with
`blockIdx.x` as the thread id.

**Pitfall.** With a persistent grid you must handle "fewer tiles than SMs" (→ idle SMs, and
multicast becomes harmful: `csrc/jit_kernels/heuristics/sm90.hpp:234-236` sets cost to `int64 max` if
`num_waves <= 1` and a cluster is requested).

### 4.2 Rasterization for L2 reuse (block swizzling)

**Where.** `scheduler/gemm.cuh:14-26` (group size choice) and `:108-144` (index mapping).

**How.** Blocks are traversed in "groups" of `kNum1DBlocksPerGroup` along the primary axis; within a
group the secondary axis is walked fully. The group size is chosen at compile time by minimizing the
operand bytes a wave touches:

```cpp
for (const auto candidate: {8u, 16u}) {
    const auto usage = kIsMulticastOnA
        ? candidate * BLOCK_N + ceil_div(kNumSMs, candidate) * BLOCK_M   // grouping on N
        : candidate * BLOCK_M + ceil_div(kNumSMs, candidate) * BLOCK_N;  // grouping on M
    ...pick min
}
```

**Why.** Concurrent tiles should share operand rows/columns so they hit in L2 instead of going to HBM.
A pure row-major sweep of tiles makes all concurrent tiles share *one* A panel but *many* B panels;
a squarish group shares both.

**Numbers.** Only the candidate set {8,16} and the cost formula are in the code; no measured L2
hit-rate numbers. There's an extra correctness fix for odd group sizes under multicast, SM90-only
(`:120-133`, guarded by `#if __CUDA_ARCH__ < 1000` because "SM90 can dynamically disable TMA
multicast while SM100 uses 2-CTA, which can not be dynamically disabled").

**Applies to.** sm_89 **yes, directly** — this is pure index arithmetic. CPU **yes**: it is loop
tiling/blocking for the shared L3, and the same formula argues for square-ish tile groups per core
group.

**Teaching hook (tiny worked example).** Assume 46 SMs (the 4070's count — **[external, verify with
`cudaDeviceProp.multiProcessorCount`]**), `BLOCK_M = 128`, `BLOCK_N = 128`.
- group = 8 → per wave you touch `8*128` N-columns + `ceil(46/8)=6` × `128` M-rows = 1024 + 768 = 1792
  "units".
- group = 16 → `16*128 = 2048` + `ceil(46/16)=3` × `128 = 384` = 2432.
So 8 wins here; with `BLOCK_N = 16` (small-N decode shapes) the balance flips. Have Luigi compute both
and say which one the formula picks — it makes "L2 reuse" arithmetic instead of vibes.

### 4.3 Warp specialization (producer/consumer)

**Where.** `.../impls/sm90_fp8_gemm_1d2d.cuh:165-215`.

**How.** `if (warp_idx >= kNumMathThreads / 32)` → the TMA warpgroup; else → math warpgroups.
Inside the TMA warpgroup exactly one warp, and one elected lane, issues all copies:
`if (warp_idx == kNumMathThreads/32 + 2 and cute::elect_one_sync())` (`:171`), with the comment "we
use the third warp, as warp 0/1 may be doing WGMMA with `BLOCK_M == 32`". Another warp initializes the
barriers (`:128-139`). SM100 splits further: warp 0 = TMA, warp 1 = UMMA issuer, warps 2-3 = SF
transposer for UTCCP, remaining 128 threads = epilogue
(`.../impls/sm100_fp8_fp4_gemm_1d1d.cuh:143, 168, 188, 213, 301, 460, 499`).

**Why.** Different roles want different resources (see 4.6) and different instruction mixes; a warp
that only issues async copies never stalls the math pipeline, and mbarriers make the handoff cheap.

**Numbers.** SM90 launch config: `num_tma_threads = 128`, `num_math_threads = block_m <= 64 ? 128 : 256`
(`csrc/jit_kernels/heuristics/sm90.hpp:190-199`). SM100: 256 threads total = 32 TMA + 128 math, with
128 non-epilogue + 128 epilogue (`sm100.hpp:254-260`).

**Applies to.** sm_89 **in modified form**. There is no TMA and no `wgmma`, but the producer/consumer
split still works with `cp.async` (`ldgsts`): dedicate a warp (or have every warp issue its own
`cp.async` — usually simpler and what most sm_80-era kernels do) and synchronize with
`cp.async.commit_group` / `wait_group` instead of mbarriers. `__syncthreads()`-based double buffering
is the simplest version. CPU **yes**: a packing thread producing A/B panels into a ring buffer while
worker threads run the microkernel is the textbook analogue (BLIS packs inline; some libraries thread
it out).

**Teaching hook.** The asymmetry is the point: "the loader warp does not compute; the compute warps do
not load." Ask Luigi where the equivalent boundary is in his M6 design, and whether packing should be
a separate rayon task or inlined.

**Pitfall.** Deadlock-by-mismatched-arrivals. Note `empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32)`
(`:134`) — the arrival count must equal exactly the number of arrivers, and the kernel does an extra
drain loop at the end so barriers can be destroyed safely (`:207-211`).

### 4.4 Multi-stage shared-memory pipeline

**Where.** Stage count comes from the heuristic (`sm90.hpp:148-188`), buffers are laid out by hand
(`.../impls/sm90_fp8_gemm_1d2d.cuh:104-124`), the ring index is advanced by
`advance_pipeline` (`:157-163`) or the reusable `RingPipeline`
(`deep_gemm/include/deep_gemm/common/ring_pipeline.cuh:13-38`).

**How.** `kNumStages` copies of (A tile, B tile, SFA tile), each with a `full` and an `empty`
mbarrier. Producer waits `empty[s]`, issues TMA into stage `s`, arrives on `full[s]` with the expected
byte count (`:203`: `arrive_and_expect_tx(SMEM_A + SMEM_B + SMEM_SFA)`); consumer waits `full[s]`,
does the MMAs, arrives on `empty[s]`. Phase bit flips when the ring wraps (`:161-162`).

`RingPipeline::advance` is a small gem worth showing: it special-cases power-of-two stage counts so
`%` and `/` become bit ops (`ring_pipeline.cuh:26-28`).

**Why.** Hide global-memory latency: while stage `s` is being consumed, stages `s+1..s+k` are in
flight. Stages needed ≈ memory latency / per-stage compute time.

**Numbers.** SM90 cap `kNumMaxStages = 16` (`sm90.hpp:149`), SM100 cap 32 (`sm100.hpp:210`). Stage
count is `min((smem_capacity - smem_extra) / smem_per_stage, cap)` with
`smem_capacity = 232448` bytes (`sm90.hpp:14, 180-182`). The candidate filter *requires* at least 3
stages, or 4 for small tiles: "To hide TMA latency, the stage count should be at least 3; for small
matrices, at least 4" (`sm90.hpp:105-108`).

**Applies to.** sm_89 **yes and this is the main event for M7**: `cp.async` + N-deep shared-memory
staging is the sm_80/sm_89 equivalent, and the arithmetic (`stages = usable_smem / bytes_per_stage`)
is identical — only `smem_capacity` changes (sm_89 allows far less dynamic shared memory per block
than Hopper's 227 KB; check `cudaDevAttrMaxSharedMemoryPerBlockOptin`
**[external, unverified: commonly 100 KB on Ada]**). CPU **analogous**: software prefetch distance and
the depth of the packed-panel ring buffer.

**Teaching hook.** Make him derive the stage count for a plausible sm_89 config before writing code:
`BLOCK_M = BLOCK_N = 128`, `BLOCK_K = 32`, int8 → A tile 128·32 = 4 KB, B tile 4 KB, so 8 KB/stage;
with ~100 KB usable and ~8 KB of other state, stages ≈ 11 → cap it at 4 and explain why more stages
stop helping (latency already hidden; smem pressure hurts occupancy).

### 4.5 TMA and TMA multicast (and the sm_89 substitute)

**Where.** Descriptors built on the host (`csrc/jit_kernels/impls/runtime_utils.hpp:119-200`, note
`CU_TENSOR_MAP_L2_PROMOTION_L2_256B` at `:157`), prefetched in-kernel
(`.../impls/sm90_fp8_gemm_1d2d.cuh:95-101`), issued via `deep_gemm::tma::copy`
(`common/tma_copy.cuh:39-60`), multicast count decided per tile (`.../sm90_fp8_gemm_1d2d.cuh:176-179`).

**How / why.** TMA = one instruction describes a multi-dimensional tile copy with swizzling and
bounds handling; the descriptor is a 128-byte object prefetched into cache. **Multicast** lets one
global read land in the shared memory of both CTAs of a 2-CTA cluster, halving HBM traffic for the
shared operand. DeepGEMM picks which operand to multicast (`kIsTMAMulticastOnA`) and disables it when
the tile pair is invalid (`scheduler/gemm.cuh:275-292`) or when there is only one wave
(`sm90.hpp:234-236`).

**Applies to.** sm_89 **no** for TMA/multicast/clusters (Hopper+). The substitute stack is:
`cp.async.bulk`? no — plain `cp.async` (`ldgsts`) per thread, addresses computed by hand, XOR
swizzling in shared memory done by hand, bounds handled by predication. **[external, unverified]**
`cp.async` with 16-byte per-thread transfers is the standard sm_80+ path and works on sm_89.
CPU: the analogue of multicast is simply that both cores reading the same packed panel hit shared L3 —
you get it for free if the tile scheduler is rasterized (4.2).

**Teaching hook.** Frame TMA as "the DMA engine you always wanted", then say: on Ada you *are* the DMA
engine; every address, swizzle and bound is yours. That is the single biggest reason an sm_89 kernel
is longer than a Hopper one.

### 4.6 Register reallocation (`setmaxnreg`)

**Where.** `.../impls/sm90_fp8_gemm_1d2d.cuh:144-146, 167, 215`:
```cpp
constexpr uint32_t kNumTMARegisters  = 40;
constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 232;
... warpgroup_reg_dealloc<kNumTMARegisters>();   // in the TMA warpgroup
... warpgroup_reg_alloc<kNumMathRegisters>();    // in the math warpgroups
```
Also `sm90_bf16_gemm.cuh:127-128` (48 / 248-or-224) and `sm90_fp8_gemm_1d1d.cuh:153-154`, where the
counts depend on whether the pipeline is unrolled (40/232 vs 24/240) — i.e. they were *tuned*.

**How / why.** CUTLASS's `reg_dealloc/alloc` wrap PTX `setmaxnreg`. The loader warps need almost no
registers; giving their budget to the math warps lets the accumulators (`accum` + `final_accum` can be
hundreds of floats) stay in registers instead of spilling. Note the heuristic already avoids configs
that would spill: block N caps of 160/192/256 are annotated "Register spills" (`sm90.hpp:49-54`) and
"The block sizes cannot be too large (for enough registers), so at least one dim less than 128"
(`sm90.hpp:93-95`).

**Applies to.** sm_89 **no** (`setmaxnreg` is sm_90+). But the *constraint* is the same and even
tighter: 255 registers/thread max, 64 K registers per SM. On sm_89 Luigi controls it with
`__launch_bounds__`, `-maxrregcount`, and by sizing the register tile. **CPU yes, strongly**: 16 YMM
registers is the hard budget that determines the AVX2 microkernel shape (e.g. 6×16 f32 accumulators =
12 YMM + 2 for operands is the classic choice) **[external, unverified: standard BLIS-style
microkernel sizing]**.

**Teaching hook.** "Registers are a budget you allocate, not a resource you're given." Then have him
count YMM registers for a candidate int8 microkernel *before* writing it.

### 4.7 Block sizes, including unaligned ones, and wave quantization

**Where.** Candidates: `sm90.hpp:16-118`; the analytic score: `sm90.hpp:202-239`; SM100 comparator:
`sm100.hpp:276-301`.

**What "unaligned" means here.** `BLOCK_N` candidates are *every* multiple of `lcm(16, user_multiple)`
from 16 (or 24) up to 160 (1D1D) / 192 (1D2D) / 256 (`sm90.hpp:40-57`). The WGMMA selector covers
N = 8,16,…,256 in steps of 8 (`mma/sm90.cuh:36-67`), so 112, 120, 176, 200, 232 are all legal MMA
shapes. `BLOCK_M` candidates: {64, 128}, plus 16 if `m <= 16`, 32 if `m <= 32`, plus 256 for BF16
output (`sm90.hpp:23-30`; the comment says "smaller block M can avoid TMA L2 OOB bound").
`BLOCK_K` is fixed by dtype: `128 * 8 / element_bits` → 128 for FP8, 256 for FP4, 64 for BF16
(`sm90.hpp:60`, `heuristics/utils.hpp:12-19` `get_num_element_bits`).

**Why odd sizes exist: wave quantization.** With `num_blocks = ceil(M/BM)·ceil(N/BN)` tiles over
`num_sms` SMs you run `num_waves = ceil(num_blocks/num_sms)` waves, and the last wave uses only
`num_blocks % num_sms` SMs. The model measures this:
```cpp
num_waves       = ceil_div(num_blocks, desc.num_sms);                   // sm90.hpp:207
last_wave_util  = num_last_blocks == 0 ? num_sms : num_last_blocks;     // :208-209
wave_efficiency = num_blocks / (num_waves * num_sms);                   // :231
num_cycles      = max(l1_cycles, l2_cycles) / wave_efficiency;          // :232
```
A "weird" `BLOCK_N` like 112 can turn 2.1 waves into exactly 2.0, which is worth more than the
slightly worse MMA shape. The SM100 comparator states the preference order explicitly
(`sm100.hpp:276-301`): single wave beats everything → multicast → fewer waves → higher last-wave
utilization → smaller `block_m + block_n` → smaller `block_m * block_n`.

**Applies to.** sm_89 **yes**, with sm_89's `mma.sync` shapes as the granularity instead of WGMMA's
(m16n8k16 for f16/bf16, m16n8k32 for int8/fp8 **[external, unverified — check the PTX ISA table for
sm_89]**). CPU **yes**: "wave quantization" is exactly load imbalance across 24 threads. If Luigi
splits 1000 rows over 24 threads he gets 41.67 rows each → some threads do 42, some 41, and the
critical path is 42; choosing a tile count that is a multiple of 24 (or using dynamic scheduling) is
the same fix.

**Teaching hook (tiny worked example, CPU flavor).** M = 1000 rows, 24 threads.
- Static, 42 rows each: 24·42 = 1008 ≥ 1000, efficiency 1000/1008 = 99.2%, fine.
- Now tile by 128 rows: `ceil(1000/128) = 8` tiles for 24 threads → **16 threads idle**, efficiency
  8/24 = 33%. Same total work, 3× worse wall clock.
This is the CPU version of `num_waves = 1, last_wave_util = 8`, and it is the single most useful
quantitative idea in the heuristics for M6.

### 4.8 Epilogue: shared-memory staging, swizzle, and bulk stores

**Where.** `.../impls/sm90_fp8_gemm_1d2d.cuh:357-441`.

**How.** Accumulators are written to shared memory with `stmatrix`
(`ptx::SM90_U32x2_STSM_N<nv_bfloat162>`, `:414-418`) at a *swizzled* address computed by hand
(`:385-407`, with the comment "think twice before modifying this, as changes may affect the number of
instructions"), then one TMA store per `TMA_D_BLOCK_N` slice writes the tile to global memory
(`:428-440`). Stores are pipelined with `tma_store_wait<0>()` + named barriers (`:371-373`).

**Why.** Direct per-thread global stores of a `BLOCK_M × BLOCK_N` FP32/BF16 tile are badly
uncoalesced because the MMA's register→element mapping is interleaved. Round-tripping through shared
memory lets you (a) transpose/repack into contiguous rows and (b) issue few, wide stores. The XOR
swizzle avoids shared-memory bank conflicts.

A nice cost detail on SM100: `get_num_tma_store_stages` (`sm100.hpp:193-207`) chooses *one* store
stage for k-grouped GEMMs when K is long, with the reasoning spelled out — "this paces the store
traffic and improves the achieved DRAM throughput when C/D flushes dominate DRAM traffic; it also
frees up shared memory for the A/B mainloop and halves the C/D smem read/write footprint".

**Applies to.** sm_89 **yes in principle**: `stmatrix` is sm_90+ **[external, unverified]**, but the
pattern "accumulate in registers → stage through shared memory → coalesced vector stores" is standard
and necessary on Ada too (use `st.shared.v4` / plain stores and `ldmatrix` for loading operands).
CPU **yes**: the analogue is writing the microkernel's register tile into a contiguous C panel buffer
and then copying it out, which is what BLIS does when C is not nicely strided.

### 4.9 Tensor-core utilization throttle (a surprising one)

**Where.** `csrc/runtime/runtime.hpp:93-100` (`set_tc_util`, 0-100, default 100), threaded through
`GemmDesc::tc_util` into the kernel as the template parameter `kTensorCoreUtilControl`
(`csrc/jit_kernels/impls/sm100_bf16_gemm.hpp:70`), used at
`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh:326-343`:

```cpp
// Let tensor cores relax for lower possibility of frequency drop
if constexpr (kTensorCoreUtilControl < 100) {
    ...
    constexpr static uint64_t kNumUMMACycles  = (2ull * UMMA_M * UMMA_N * BLOCK_K) / 8192ull;
    constexpr static uint64_t kNumDummyCycles = (100ull - kTensorCoreUtilControl) * kNumUMMACycles / kTensorCoreUtilControl;
    const auto start_clock = clock64();
    if (cute::elect_one_sync()) while (clock64() - start_clock < kNumDummyCycles) {}
}
```

**Why.** Running tensor cores flat out raises power draw and can trigger clock throttling, hurting
*other* concurrent work (and sometimes total throughput). A deliberate duty cycle trades local
throughput for clock stability. This is a system-level performance argument, not a kernel one.

**Applies to.** sm_89 **conceptually** (Ada throttles too), but nobody should implement this before
measuring. CPU **analogue**: AVX-512 license-based downclocking on older Intel parts — the reason
AVX2 sometimes beat AVX-512 **[external, unverified for current parts]**. Luigi has no AVX-512, so
this is trivia for him, but it's a great story about "the fastest kernel is not always the fastest
program".

**Teaching hook.** Role project 6 (kernel-level latency spikes) and project 3 (load balancing) both
live in this territory: a kernel that maximizes its own throughput can degrade p99 for co-resident
work.

### 4.10 PDL, and other launch-level tricks

- **Programmatic Dependent Launch**: `cudaGridDependencySynchronize()` right before the scheduler
  starts (`.../impls/sm90_fp8_gemm_1d2d.cuh:148-149`, `sm90_bf16_gemm.cuh:131`), enabled via
  `deep_gemm.set_pdl` → `jit->default_launch_options.enable_pdl`
  (`csrc/apis/config.hpp:27-32`) → `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`
  (`DeepJIT include/deep_jit/backend/cuda/kernel.hpp:148-152`). The point: a kernel can start its
  prologue (descriptor prefetch, barrier init) *before* the previous kernel finishes, then wait only
  for the data it needs. **sm_90+** **[external, unverified: PDL requires compute capability 9.0]**.
- **TMA descriptor prefetch** (`:95-100`) — a "warm up your metadata" idea; the sm_89 analogue is
  nothing, since there are no descriptors.
- **L2 promotion hint** on descriptors: `CU_TENSOR_MAP_L2_PROMOTION_L2_256B`
  (`runtime_utils.hpp:157`). sm_89 has `__ldg`/`cp.async` cache hints and the L2 persistence
  window (`cudaAccessPolicyWindow`) as rough equivalents **[external, unverified]**.

### 4.11 SASS / FFMA interleaving — explicitly retired

README:22 — "As NVCC 12.9 will automatically do the FFMA interleaving, all post optimizations will be
no longer supported." Earlier DeepGEMM versions patched the compiled SASS to interleave FFMA
instructions (toggling the yield/reuse bits) so the promotion FFMAs would overlap with MMA issue.
That machinery is gone at this commit; the only remaining hook for that kind of thing is DeepJIT's
`post_hook` (a Python script that may rewrite the CUBIN in place before publication,
`DeepJIT README:255-281`).

**Lesson for Luigi (and it's a good one):** compilers catch up. A hand-tuned assembly trick has a
shelf life; a *measurement* of whether it still helps does not. And note the shape of the
infrastructure they kept: a generic post-compile hook whose *content hash* is part of the cache key
(`DeepJIT README:278`).

**Applies to.** CPU **yes, in kind**: this is why every AVX2 change in M6 must be measured against the
current rustc/LLVM output rather than assumed to help.

### 4.12 Locality domains (brand-new, and skip-worthy)

`csrc/runtime/locality_domain.hpp:16-54` ("Compatability layer for locality domain APIs in CUDA 13.4",
`TODO: cuMemCreate on the location {CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN, ...} with CUDA 13.4`),
`csrc/apis/locality_domain.hpp` (localized allocations; a pointer-chasing latency probe that infers
which locality domain each SM belongs to, then *balances* SMs across domains),
`deep_gemm/include/deep_gemm/impls/sm100_locality_domain.cuh`.

**[inference]** This targets multi-die / partitioned-memory GPUs: allocate a tensor in the memory
domain nearest the SMs that will read it, and spread SMs evenly across domains for load balance. It is
the GPU version of NUMA-aware placement.

**Applies to.** sm_89 **no** (single die, one domain). CPU: the analogue is real and namable — NUMA
placement and `first-touch` allocation. Luigi's WSL2 box is single-socket, so also skip, but it's a
good vocabulary item for the serving work post-v1.

---

## Part 5 — The config heuristic as a quantitative performance model (role project 4)

This is the section to mine hardest for M0.

### 5.1 Structure

`csrc/jit_kernels/heuristics/common.hpp:17-56`:
```cpp
template <typename ArchSpec> static GemmConfig get_best_config(const GemmDesc& desc) {
    desc.check_validity();
    const auto layout_candidates = ArchSpec::get_layout_candidates(desc);   // enumerate legal tilings
    auto layout = layout_candidates[0];
    auto layout_info = ArchSpec::get_layout_info(desc, layout);             // score it
    for (...) if (ArchSpec::compare(candidate_info, layout_info)) ...       // keep the best
    const auto storage_config  = ArchSpec::get_storage_config(desc, layout);
    const auto pipeline_config = ArchSpec::get_pipeline_config(desc, layout, storage_config);
    const auto launch_config   = ArchSpec::get_launch_config(desc, layout);
    ...
}
```
Four separable decisions: **layout** (block sizes + cluster) is *searched*; **storage** (swizzle modes,
load/store block sizes), **pipeline** (stages, smem bytes) and **launch** (threads, grid) are
*derived*. Setting `DG_PRINT_CONFIGS=1` prints the chosen config per distinct shape
(`common.hpp:44-53`).

### 5.2 The SM90 cost model, line by line (`sm90.hpp:202-239`)

```cpp
num_blocks      = ceil_div(M, BM) * ceil_div(N, BN) * num_groups;
num_waves       = ceil_div(num_blocks, num_sms);
last_wave_util  = num_blocks % num_sms == 0 ? num_sms : num_blocks % num_sms;

l2_bandwidth_per_cycle = min(64.0 * num_sms, 8e6 / 1.3e3);    // B/cycle
l1_bandwidth_per_cycle = 128 * num_sms;                        // B/cycle

num_bytes_l2_ab   = K * (BM/cluster_n + BN/cluster_m) * elem_size_ab;   // multicast divides traffic
num_bytes_l1_ab   = K * (BM + BN) * elem_size_ab;
num_bytes_l1_tc   = K * (max(64, BM) + BN) * elem_size_ab + BM*BN*elem_size_cd;
num_bytes_l1_l2_cd= BM*BN*elem_size_cd * (with_accumulation ? 2 : 1);

num_l2_cycles   = (num_bytes_l2_ab + num_bytes_l1_l2_cd) * num_blocks / l2_bandwidth_per_cycle;
num_l1_cycles   = (num_bytes_l1_ab + num_bytes_l1_tc + num_bytes_l1_l2_cd) * num_blocks / l1_bandwidth_per_cycle;
wave_efficiency = num_blocks / (num_waves * num_sms);
num_cycles      = max(num_l1_cycles, num_l2_cycles) / wave_efficiency;
if (cluster > 1 and num_waves <= 1) num_cycles = INT64_MAX;     // multicast is pointless in one wave
```
and the comment that makes it a *model* rather than a guess (`:227-228`): "HBM bandwidth and total
compute (Tensor/CUDA cores) are constant across configs. We only model L1/L2 cycles as they are the
primary variables between configs."

**Read that comment to Luigi twice.** It is the discipline behind role project 4: identify what
actually varies between the options you're choosing among, model only that, and let the constants
cancel. A model that predicts *relative* order can be far simpler than one that predicts absolute
time.

The magic constants (`64 B/cycle/SM` L2, `128 B/cycle/SM` L1, `8e6/1.3e3` as an aggregate L2 ceiling)
are per-architecture peak-throughput numbers with no citation in the repo — treat them as *calibrated
constants*, which is also the honest way for Luigi to present his own.

SM100 does **not** have a cycle model yet: `get_layout_info` returns `num_cycles = 0` with
`// TODO: calculate expected cycles` (`sm100.hpp:271-272`) and `compare` falls back to the lexicographic
preference chain (`:276-301`). Useful to point out: the production library is *inconsistently*
modeled, and the lexicographic heuristic was good enough to ship.

### 5.3 Derived configs

- **Swizzle mode** = the largest of {128, 64, 32, 16} bytes that divides the inner-dim byte size
  (`heuristics/utils.hpp:21-30`), and candidates with a swizzle < 64 B are rejected: "Make sure
  swizzling is large enough (32B's performance is low)" (`sm90.hpp:101-103`).
- **Stages** = `min((232448 - smem_extra) / smem_per_stage, cap)`, where `smem_extra` accounts for the
  epilogue tile, barriers (`kNumMaxStages * 8 * 2`), extra SFB and tensor maps
  (`sm90.hpp:148-188`). Then configs with fewer than 3 (or 4) stages are dropped (`:105-108`).
- **Threads** = 128 TMA + (128 or 256) math (`sm90.hpp:190-199`).

### 5.4 How to use this in M0

Have Luigi write the same four-part shape for his CPU engine, then fill it in:

1. **Enumerate** the legal configurations (tile sizes, thread counts, quantization format).
2. **Model only what varies**: bytes moved per tile from L1/L2/L3/DRAM, and whether the tile count
   divides evenly across 24 threads.
3. **Score** with `cycles = max(byte-limited, flop-limited) / efficiency`.
4. **Predict TTFT and decode tok/s** from that, then measure. For decode the model is almost entirely
   memory-bound: `tok/s ≈ achievable_mem_bw / bytes_per_token_of_weights`. That formula is the one
   number M0 must contain, and the DeepGEMM model is the shape of the justification.

A concrete M0-style worked example (illustrative numbers, not measured):
TinyStories-15M f32 = ~60 MB of weights; if the machine sustains ~20 GB/s single-threaded for a
streaming read, decode is bounded at ~330 tok/s; int8 (15 MB) at ~1300 tok/s. Then the *measured*
number tells him how much he is leaving on the table, and `max(mem, flops)` tells him whether he's even
looking at the right bound. (Have Luigi supply the real bandwidth number with a measurement, e.g. a
simple `sum` over a large array, rather than a spec sheet.)

---

## Part 6 — JIT in DeepGEMM: why compile at runtime

### 6.1 Mechanics

`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:29-71` is the whole idea in 40 lines:

```cpp
static void compile_and_launch(const std::string& tag, const Args& args) {
    const auto kernel = jit->compile(tag, std::format(R"(
#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d2d_impl< ...23 template args... >);
}};
)", /* major_sfb, M, N, K, num_groups, BLOCK_M/N/K, swizzles, stages, threads, cluster, num_sms, gemm_type, cd_dtype */));
    jit->launch(kernel, args.options, args.sfb, args.grouped_layout, m, n, k,
                tensor_map_a, tensor_map_b, tensor_map_d, tensor_map_sfa);
}
```

Note the generated source *does nothing but take the address of a template instantiation* — that's
enough for nvcc to emit the kernel, and `Kernel::load` then asserts the CUBIN contains exactly one
kernel (`DeepJIT include/deep_jit/backend/cuda/kernel.hpp:70-73`).

### 6.2 Why: shapes as compile-time constants

`get_compiled_dim(dim, name, compiled_dims)` returns the real dimension if that letter is in
`compiled_dims`, else `0` (`csrc/jit_kernels/impls/runtime_utils.hpp:31-40`); inside the kernel,
`shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m` (`.../impls/sm90_fp8_gemm_1d2d.cuh:69-72`). So each
dimension is *optionally* burned into the binary. Defaults: `compiled_dims = "nk"` for NT GEMMs,
`"mn"` for TN (`csrc/apis/gemm.hpp:842, 858`) — N and K are the weight dimensions, fixed per model;
M is the token count, which varies per call, so it stays dynamic. `deep_gemm.set_ignore_compile_dims`
turns it all off (fewer compiles, slower kernels).

What constant shapes buy: `ceil_div` by a constant folds, loop trip counts become known (enabling
`#pragma unroll 8` at `:280`), index math strength-reduces, and register allocation can be exact.
Everything about the tiling (`BLOCK_M/N/K`, `kNumStages`, swizzle modes, thread counts, `kNumSMs`) is
also a template parameter, so the *entire* config search result is compile-time.

The cost: a compile per (shape-class × config). DeepGEMM's answer is caching plus deliberately keeping
M dynamic; DeepJIT's answer is a content-hashed disk cache shared across processes and nodes.

### 6.3 NVCC vs NVRTC

DeepJIT uses **NVCC producing a CUBIN**, not NVRTC: `nvcc <src> --cubin --output-file <out>` plus
flags, run through a shell (`DeepJIT include/deep_jit/backend/cuda/backend.hpp:96-115`); "It uses NVCC
to generate a CUBIN" (DeepJIT README:186). **[inference]** the reason is that the kernels include
CUTLASS/CuTe headers and rely on full C++20 host-ish template machinery and nvcc-only flags
(`--expt-relaxed-constexpr`, `--ptxas-options=...`, `--register-usage-level=10`), which NVRTC does not
fully support; nvcc also gives PTXAS diagnostics (spills, local memory) that DeepJIT turns into hard
errors. The price is a process launch and file I/O per compile — mitigated entirely by the cache.

---

## Part 7 — DeepJIT, and what it means for M7's Rust host

### 7.1 Architecture

Header-only C++20, ~2060 lines of headers. Two backends (CUDA, Huawei Ascend) behind one
`Runtime<Backend>` (README:11, 25-28).

```
include/deep_jit/
  runtime/{config,runtime}.hpp     Config + the compile/launch orchestration
  backend/cuda/{backend,device,kernel,options,driver}.hpp
  backend/ascend/...               (same interface, bisheng + ld.lld + ACL)
  cache/{memory,disk}.hpp          process-local map + on-disk directory cache
  utils/{hash,parser,env,lazy,filesystem,command,json,gil,...}.hpp
  python_api.hpp                   registers get_jit() for a consumer's pybind module
```

`Runtime` (`runtime/runtime.hpp:19-99`) owns: config, env, device, default compiler/launch options,
disk cache, backend, include parser, in-memory kernel cache, and a pre-seeded hash.

`compile()` is four lines and reads like a spec (`:56-62`):
```cpp
const auto options = default_compiler_options.override_with(override_options);
const auto key = cache_key(source, options);
return mem_cache.get_or_create(key, [&]{ return Backend::load(compile(name, source, key, options), env); });
```

### 7.2 The cache key

`cache_key` (`runtime/runtime.hpp:92-98`) = `hash_base` (seeded with `extra_signature` and the hash of
the full `nvcc --version` output, `:52-53`) then `options.update_hash` (the *effective flag list*,
`backend/cuda/options.hpp:136-138`) then the post-hook hash then
`parser.parse_into_hash(source)`. DeepJIT README:284-292 lists the five components in order and notes:
"Each component is prefixed by its fixed-width byte length before it is added to the two-state FNV-1a
hash, so boundaries remain unambiguous even for binary strings containing zero bytes. The final digest
is a 32-character hexadecimal string. This is a fast cache checksum, not a cryptographic hash."

The hash itself (`utils/hash.hpp:10-38`): two FNV-1a states with different multipliers, each byte fed
to both, length prefixed (`:21`), finalized with two splitmix64 rounds (`:30-38`).

The **include parser** (`utils/parser.hpp:37-120`) is a deliberately dumb line scanner: it tracks only
`#include <...>` whose filename starts with a configured prefix (DeepGEMM passes `{"deep_gemm/"}`,
`csrc/runtime/jit.hpp:22`), resolves it against the include dirs, recurses, caches per-file digests,
and panics on cycles (`:75-77`) or on non-canonical include forms (`:57-59`). CUTLASS headers are
*not* tracked — their version is represented by `extra_signature = "cutlass-<CUTLASS_VERSION>"`
(`csrc/runtime/jit.hpp:20`). README:306 is refreshingly candid about the limits: `NVCC_PREPEND_FLAGS`,
`CPATH`, host-compiler choice and third-party header changes "are not discovered automatically".

**Teaching hook.** A cache key is a *claim* about what the output depends on. Enumerating that claim,
and writing down what it deliberately excludes, is the engineering. Ask Luigi what his M2 mmap'd
weight loader's "cache key" would be if he memoized anything (file path + mtime + size + format
version?), and what it would silently miss.

### 7.3 Atomic disk publish

`cache/disk.hpp`. Entries live at `<root>/cache/<tag>.<digest>/`, builds happen in
`<root>/tmp/<uuid>` (`:130-132`), and publication is (`:42-67`):

```cpp
write_file_sync(path / ".committed", "");   // marker
fsync_dir(path);                            // durability before visibility
make_dirs(commit_path.parent_path());
std::filesystem::rename(path, commit_path, error_code);   // atomic within a filesystem
if (error_code) safe_remove_all(path);      // someone else won the race: use theirs
```
with comments explaining both the race ("if another rank already created dir_path, rename will fail —
that's fine") and a distributed-FS hazard ("avoid `std::filesystem::remove_all` here — it can segfault
on distributed filesystems, when concurrent processes operate on the same parent directory"). Lookup
checks each root in order for the `.committed` marker and touches its mtime (`:121-127`) — so an LRU
sweeper can use mtimes. Uncommitted temp dirs are removed by the destructor, "including on exception
paths" (`:21-25, 69-73`).

**Why it matters.** N processes on N GPUs (or a whole cluster) start simultaneously, all need the same
kernel, none coordinate. Content-addressed + atomically-published means duplicated *work* but never a
corrupt or partially-visible artifact. README:36-52 documents sharing one cache dir across users and
nodes, and a `personal:shared` root list where misses only write to the first root.

### 7.4 Lazy init, GIL, launch

- **Lazy**: `LazyInit<T>` holds a factory and constructs on first `operator->` (`utils/lazy.hpp:16-40`);
  DeepGEMM's `jit` and `runtime` are both `LazyInit` (`csrc/runtime/jit.hpp:12`,
  `csrc/runtime/runtime.hpp:104`). So `import deep_gemm` touches no GPU and finds no compiler — tested
  explicitly (`DeepJIT tests/test_cuda.py:137` asserts "importing the extension initialized the lazy
  runtime" is false; `tests/test_cuda.py:155, 186` assert `not torch.cuda.is_initialized()`).
- **GIL**: released around compilation and launch (`GilScopedRelease` at
  `backend/cuda/backend.hpp:83`, `kernel.hpp:50, 94`), and there is a test that a Python thread makes
  progress while nvcc holds nothing (`tests/test_cuda.py:282`).
- **Launch** (`backend/cuda/kernel.hpp:91-171`): validates every option, sets
  `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`, builds up to 3 `CUlaunchAttribute`s
  (cooperative, cluster dim, PDL), packs kernel args as `void*`s, calls `cuLaunchKernelEx`. Default
  stream = the current PyTorch stream (`:164-166`). There's an honest comment at `:123-127` about
  deliberately *not* resetting the non-portable-cluster attribute because launch overhead matters more
  than the edge case.
- **Compiler options** (`backend/cuda/options.hpp:39-62`): `-O3`, fast-math **off**,
  `--ptxas-options=--register-usage-level=10`, `-std=c++20`, `--expt-relaxed-constexpr`,
  `--expt-extended-lambda`, arch from the device with an `a`/`f` suffix (`device.hpp:68-80`).
  Note fast-math off by default in a performance library — determinism/accuracy first.

### 7.5 The option space for M7's Rust host (present, do not decide)

Luigi's M7 is "Rust host via `cudarc` + CUDA C++ kernels". There are (at least) four ways to get from
`.cu` to a launchable function. Present all four with tradeoffs; let him choose in `DECISIONS.md`.

**Option A — AOT nvcc in `build.rs`, embed CUBIN/PTX, load via cudarc.**
- How: `build.rs` shells out to `nvcc --cubin -arch=sm_89` (or `--ptx`), `include_bytes!` the result,
  `cudarc`'s module-load from bytes.
- Pros: no runtime compiler dependency; `cargo build` fails on kernel errors (fits the project rule
  that `cargo test` must pass); trivially reproducible; works offline; one artifact to ship.
- Cons: every tile size / dtype / shape specialization must be enumerated at build time (combinatorial
  explosion, or dynamic shapes with runtime loop bounds — losing exactly the specialization DeepGEMM
  is built around); `build.rs` needs nvcc present at build time on every machine; CUBIN is
  arch-locked (PTX is more portable but JIT-compiled by the driver at load, with its own cost).
- Best when: a handful of kernels, shapes stay dynamic, and simplicity wins. **[inference]** This is
  the natural first choice for M7 and matches what most Rust CUDA projects do.

**Option B — NVRTC at runtime via cudarc.**
- How: `cudarc::nvrtc::compile_ptx(src)` with the shape/tile constants string-formatted in, then load
  the PTX.
- Pros: DeepGEMM's specialization benefit without an external process; in-process, no temp files; can
  generate one kernel per encountered shape.
- Cons: NVRTC is a *subset* toolchain — no `--expt-*` niceties in the same way, careful header
  handling, and you must ship the NVRTC shared library; you lose PTXAS diagnostics unless you ask for
  them; first-call latency shows up in TTFT unless you pre-warm; PTX→SASS still happens at load.
- Best when: he wants to *demonstrate* shape specialization (a real Performance Engineer talking
  point) without a disk cache.

**Option C — DeepJIT-style: runtime nvcc + content-hashed disk cache.**
- How: Rust computes a digest of (source + tracked includes + flags + nvcc version), looks in
  `~/.cache/inference-engine/<tag>.<digest>/`, otherwise shells out to nvcc into a temp dir and
  publishes with `std::fs::rename`.
- Pros: full nvcc (all flags, all diagnostics, CUTLASS-style headers if ever needed); compile cost
  paid once per machine, ever; the cache is inspectable (dump PTX/SASS next to the CUBIN);
  parallel-process safe with ~30 lines of Rust; this is the design Luigi can *explain in an interview*
  as "I implemented content-addressed kernel caching the way DeepSeek does".
- Cons: the most machinery to build and test; needs nvcc at runtime on the target machine; cache
  invalidation is a correctness surface (see the honest list of what DeepJIT's key misses,
  README:306); shelling out from a library is a portability/security wrinkle.
- Best when: shape specialization is load-bearing for the benchmark table *and* recompiles are
  frequent during development.

**Option D — hybrid.** AOT-compile a default set in `build.rs` (so `cargo test` always works with no
CUDA toolkit at runtime), and optionally NVRTC-specialize hot shapes at runtime behind a feature flag.
Pros: best of A and B, honest fallback story. Cons: two code paths to keep in sync.

**Cross-cutting things to decide regardless of option:**
- `sm_89` vs PTX-for-forward-compat (CUBIN is faster to load; PTX survives a GPU upgrade).
- Whether `--ptxas-options=--warn-on-spills` (and `--warn-on-local-memory-usage`) become hard build
  errors, as DeepJIT makes them (`backend/cuda/backend.hpp:120-127`). Cheap, high value.
- Where dumped PTX/SASS live, so "measure, don't claim" can cite the actual instruction mix.
- Whether the Rust side owns the config heuristic (it should — it's host logic, and it's role project
  4's evidence).

---

## Part 8 — Testing and benchmarking methodology

### 8.1 The diff metric: `calc_diff`

`deep_gemm/testing/numeric.py:5-11`, verbatim:

```python
def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:    # Which means that all elements in x and y are 0
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim
```

So `diff = 1 - 2⟨x,y⟩/(‖x‖² + ‖y‖²)`. Properties worth stating precisely:
- It is `0` iff `x == y` **[inference: because `1 - 2⟨x,y⟩/(‖x‖²+‖y‖²) = ‖x-y‖²/(‖x‖²+‖y‖²)`]** — and
  that identity is the clean way to explain it: **it is squared relative error, normalized by the
  energy of both tensors**. Write that on the board.
- It is scale-free in the sense that multiplying *both* tensors by `c` leaves it unchanged; it is
  *not* invariant to scaling only one (unlike cosine similarity), so it does catch a systematic gain
  error — exactly the bug class that a wrong scale factor produces.
- It is an *aggregate*: one wild element in a 4096×7168 output barely moves it.

Thresholds actually used:

| Case | Threshold | Where |
|---|---|---|
| FP8 × FP8 | `< 0.001` | `tests/generators.py:74` |
| one FP4 operand | `< 0.01` | `tests/generators.py:72-73` |
| FP4 × FP4 | `< 0.02` | `tests/generators.py:70-71` |
| BF16 GEMM | `< 1e-5` | `tests/test_bf16.py:46` |
| MQA logits, fp8 | `< 1e-3` vs FP32 ref, `< 5e-6` vs *simulated* (dequantized-input) ref | `tests/test_attention.py:74-75, 199-207` |
| MQA logits, fp4 | `< 0.02` / `< 3e-5` | same |
| FP4 vs converted-FP8 equivalence | `< 1e-14` | `tests/test_fp8_fp4.py:99` |

The reference is always FP32 PyTorch on the *unquantized* inputs
(`ref_d = (alpha * (a.float() @ b.float().t()) + c).to(out_dtype)`, `tests/generators.py:363`). So the
measured diff includes the quantization error itself, which is why the tolerance is 1e-3 rather than
1e-6.

**The double-reference trick is the best idea here, and Luigi should copy it.** The attention test
compares against *two* references: the exact FP32 one (`ref_logits`) and a "simulated" one computed by
dequantizing the same quantized inputs back to BF16 and doing the math in FP32
(`tests/test_attention.py:149-157`). The first tolerance (1e-3) absorbs quantization error; the second
(5e-6) is tight and isolates **kernel** error from **format** error. If the tight one fails, the
kernel is wrong; if only the loose one fails, the quantization scheme is the problem.

### 8.2 Contrast with Luigi's D1 (|ours − ref| ≤ 5e-4 absolute, measured)

| | D1: max-abs-diff ≤ 5e-4 | DeepGEMM: `calc_diff < 1e-3` |
|---|---|---|
| Catches one bad element | **Yes** — it's a max | No — averaged away |
| Sensitive to output magnitude | Yes, badly: logits of magnitude 20 vs 0.02 need different bounds | No, self-normalizing |
| Good for "did I port the algorithm right" | **Yes** | Weak |
| Good for "is my quantized format acceptable" | Weak (a few outliers always exceed any tight bound) | **Yes** |
| One number across shapes/dtypes | No | Yes |

**Recommendation for the tutor:** keep D1 for M1-M3 (f32 kernels, exact-ish porting — max-abs-diff is
the right tool and 5e-4 is a sane bound for f32 accumulation order differences), and **add** a
`calc_diff`-style relative metric at M5 when quantization enters, because a max-abs bound on int8
output is either vacuous or unachievable. Log it as a decision (D-something: "two metrics, two
purposes"). Also copy three habits verbatim:
1. **Report both** max-abs-diff and the normalized diff, always, per shape.
2. **Double reference** at M5: ours-vs-f32 (absorbs quantization) and ours-vs-dequantized-inputs-f32
   (isolates kernel bugs). This is a strictly better M5 checkpoint than "quality drift" alone.
3. **Bitwise determinism**: `tests/test_fp8_fp4.py:78-82` runs the same kernel 20 times and asserts
   `torch.equal`. Cheap, and it catches races, uninitialized shared memory, and atomics-order
   nondeterminism — all bugs Luigi *will* hit in M6 (threads) and M7 (CUDA). The failure message
   even locates the first differing byte (`numeric.py:24-44`). There is a
   `use_deterministic_algorithms` switch for when they intentionally relax it
   (`csrc/apis/config.hpp:33-35`).
4. **Accumulation-mode cross-check**: `assert_direct_output_matches_fp32_accumulation`
   (`tests/utils.py:20-38`) asserts that the direct-output path equals the `D = GEMM + 0` accumulate
   path *bitwise* (modulo one FP32→BF16 cast). Two code paths that must agree exactly is a powerful,
   cheap invariant; Luigi's M3 analogue is "prefill of N tokens then decode" vs "decode N tokens one
   at a time" producing identical logits — and that one directly serves his end goal of separating
   prefill from decode.

### 8.3 Benchmarking

`deep_gemm/testing/bench.py:7-33` (`bench`, CUDA events):
1. `torch.cuda.synchronize()`, then **flush L2 by zeroing a 256 MB int tensor** (`:9-12`).
2. `num_warmups = 5` calls (default).
3. Optionally a big FP32 8192³ matmul to "eliminate the CPU launch overhead" (`:18-22`).
4. Time `num_tests = 10` calls between two `torch.cuda.Event(enable_timing=True)`, divide.

`bench_kineto` (`:79-146`) is the one the tests actually use: it profiles with
`torch.profiler` (CUDA activities only, `schedule(wait=0, warmup=1, active=1, repeat=1)`), **flushes
256 MB of L2 before every single iteration** (`:107-108`), optionally inserts `torch.cuda._sleep(2e7)`
(~10 ms) plus a barrier for multi-rank fairness (`:110-113`), then parses the profiler table and
extracts the *per-kernel* average time by name. It asserts the kernel name appears at most once
(`:123`) so you can't accidentally average two kernels. `DG_USE_NVIDIA_TOOLS=1` makes it a no-op so
nsys/ncu/compute-sanitizer runs aren't disturbed (`:89-90`).

Reported metrics (`tests/test_fp8_fp4.py:101-111`):
```
t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)
cublas_times = bench_kineto(cublas_func, ('nvjet','bstensorop','reduce'), with_multiple_kernels=True)
print(f' > Perf (m=..., n=..., k=..., {kernel_opt}, layout=..., {out_opt}, {acc_opt}): '
      f'{t*1e6:6.1f} us | {2*m*n*k/t/1e12:4.0f} TFLOPS | '
      f'{(count_bytes(a,b,d) + count_bytes(c)*int(accumulate))/1e9/t:4.0f} GB/s | '
      f'{cublas_t/t:.2f}x cuBLAS speedup')
```
- TFLOPS = `2·M·N·K / t` (the standard 2 flops per MAC).
- GB/s = actual bytes of the *operands as stored* (`count_bytes`, `numeric.py:14-21`) / t — i.e. the
  achieved traffic assuming perfect reuse, not the measured DRAM traffic. Worth flagging as a
  simplification.
- Baseline is cuBLASLt, summed over its possibly-several kernels, and the sweep's summary is a
  **geometric mean** speedup (`tests/generators.py:170-173`) — the right average for ratios.

**Practices for Luigi to adopt at M4/M6 (all cheap):**
- **Flush the cache before each timed iteration** (his CPU analogue: touch a buffer larger than L3),
  or explicitly say the measurement is warm-cache. This is the difference between a believable and an
  unbelievable tok/s number.
- **Warm up**, then time N iterations, report the mean — and separately report the *first* iteration
  when TTFT is the metric (cold caches are the honest prefill condition).
- **Name the baseline and sum all its kernels.** llama2.c `runq` is his cuBLAS.
- **Geometric mean for ratios**, arithmetic for times.
- **One command, printed output** (project rule) — DeepGEMM's tests *are* the benchmark, which is why
  their evidence is always pasteable.
- `scripts/quick_plot_pm.py` shows the next level: pull selected NCU PM metrics
  (`*.avg.pct_of_peak_sustained_elapsed`) grouped as Overview/SM/L1/L2/DRAM/Interconnect (`:47-60`).
  The CPU analogue for M6 is exactly `perf stat -e` on a curated counter list.

---

## Part 9 — API design through Ousterhout's lens

**Is it a deep module?** Yes, strikingly.

The interface Luigi would call:
```python
deep_gemm.fp8_gemm_nt((a_fp8, sfa), (b_fp8, sfb), d, recipe=(1, 128, 128))
```
Required: three tensors (two of them paired with their scales) and a recipe. Optional, all defaulted:
`c`, `recipe_a`, `recipe_b`, `compiled_dims="nk"`, `disable_ue8m0_cast=False`, `alpha`, `epilogue`
(`csrc/apis/gemm.hpp:838-845`).

Hidden behind that: layout enumeration and scoring, block sizes, cluster size, swizzle modes, stage
count, shared-memory budget, thread counts, TMA descriptor construction, warp specialization,
scale-factor transformation (launched automatically if you pass float32 SFs), kernel source
generation, nvcc invocation, disk caching, driver module loading, and launch attributes. The
interface-to-implementation ratio is enormous — Ousterhout's definition of depth.

**Where complexity leaks out (honest accounting):**
- **The SF contract.** 472 lines of documentation exist because the caller must produce scale factors
  in a specific dtype, shape, stride and *value constraint* (powers of two). This is real leaked
  complexity, and the library says so: "operations like input transposition or FP8 casting must be
  handled separately by the user … these may result in slower performance, but our primary focus is
  on optimizing the GEMM kernels themselves" (README:70). A deliberate scope boundary, not an
  oversight — and a good model for Luigi's own "what does my engine *not* do".
- **Layout naming in the function name** (`_nt`, `_nn`, `_tn`, `_tt`) and `compiled_dims` as a string
  of letters. Compact, but you must know the convention (README:63).
- **Grouped-GEMM alignment**: callers must align expert segments to the M block size and ask the
  library what that is (`get_mk_alignment_for_contiguous_layout()`, README:78).

**The knobs, in full** (`csrc/apis/config.hpp:8-48`) — note they are all *global process state*, not
per-call arguments: `init`, `shutdown`, `set/get_num_sms`, `set/get_tc_util`, `set/get_pdl`,
`use_deterministic_algorithms`, `set_ignore_compile_dims`, `set_block_size_multiple_of`. That's 8
knobs for a library with thousands of possible kernel configurations. Everything else is either
derived or JIT-specialized.

**Lesson for Luigi's Rust API (M1/M3/M7).** His `Tensor`/device abstraction should expose
`matmul(a, b, out)` and hide tile sizes, thread counts, and backend selection. The tempting mistake is
a `MatmulConfig { block_m, block_n, threads, ... }` in the public API "for flexibility"; DeepGEMM's
answer is that the *library* should compute those from the shape and the machine, and expose only a
global override for the rare case (`set_num_sms`, `set_block_size_multiple_of`). Ask him: which of his
planned parameters are genuinely the caller's business, and which are the engine's?

One more Ousterhout-flavored observation: **errors are asserted early and loudly at the boundary.**
`DG_HOST_ASSERT` checks dtype, shape, stride, contiguity and alignment in the API functions before any
kernel work (e.g. `csrc/apis/gemm.hpp:89-110`), and `check_sf_layout` (`csrc/utils/layout.hpp:101`) validates the SF contract. A
deep module with a strict boundary is much easier to trust than a permissive one that fails inside.

---

## Part 10 — A concrete M7 learning path on sm_89

The ladder. Each rung is a *measurable* step with a CPU oracle (which the project already mandates:
the CPU engine is the test oracle). Each rung should produce a row in a benchmark table.

**Rung 0 — plumbing, before any optimization.**
Rust + `cudarc`: allocate, H2D, launch a trivial kernel, D2H, compare against the CPU result with the
D1 tolerance. Decide the compile story (§7.5) here and write it in `DECISIONS.md`. Deliverable: a
`vec_add` or `rmsnorm` matching the CPU bit-for-bit-ish, and the `cargo test` that proves it.
*DeepGEMM idea used:* none yet — but adopt `--warn-on-spills`/`--warn-on-local-memory-usage` as build
errors now (`DeepJIT backend/cuda/backend.hpp:120-127`) and PTX/SASS dumping behind an env var.

**Rung 1 — naive f32 GEMM, one thread per output element.**
Establishes correctness and a baseline. Compute the arithmetic intensity and show it is
memory-bound — the "quantitative model" habit starts here.
*Pitfall:* uncoalesced access to B. Measure it, don't assume it.

**Rung 2 — shared-memory tiling.**
`BLOCK_M × BLOCK_N` output tile, `BLOCK_K` slab staged in `__shared__`, `__syncthreads()` between
load and compute. This is DeepGEMM's mainloop minus the pipeline.
*DeepGEMM ideas used:* the smem budget arithmetic (`stages × bytes_per_stage ≤ capacity`,
`sm90.hpp:180-182`) reduced to `1 × bytes_per_stage`; the "swizzle to avoid bank conflicts" problem
appears the moment he transposes anything (compare `.../sm90_fp8_gemm_1d2d.cuh:385-407`).
*Pitfall:* bank conflicts on the B tile; the classic fix is padding the row stride or XOR swizzling.

**Rung 3 — register tiling (the real win).**
Each thread computes a small `TM × TN` sub-tile in registers (e.g. 4×4 or 8×8), so each shared-memory
load feeds many FMAs. This is where the FLOP/byte ratio finally goes above 1.
*DeepGEMM ideas used:* the register budget as a first-class constraint (§4.6); `accum[]` as an explicit
register array (`.../sm90_fp8_gemm_1d2d.cuh:254`).
*Pitfall:* spills. This is exactly why `--warn-on-spills` is a build error from Rung 0.

**Rung 4 — `cp.async` multi-stage pipeline.**
Double- then N-buffer the shared-memory tiles, issuing `cp.async` for stage `s+1` while computing on
stage `s`; synchronize with `cp.async.commit_group`/`wait_group`. This is the sm_89 stand-in for
TMA + mbarriers.
*DeepGEMM ideas used:* `kNumStages` and the "at least 3, or 4 for small tiles" rule
(`sm90.hpp:105-108`); `RingPipeline`'s power-of-two stage trick (`ring_pipeline.cuh:26-28`);
optionally a producer/consumer warp split (§4.3).
*Pitfall:* too many stages → smem exhaustion → occupancy collapse. Measure the stage count sweep and
put it in the table. This is the single most instructive rung.

**Rung 5 — tensor cores via `mma.sync` + `ldmatrix`.**
Move the inner product to `mma.sync.aligned.m16n8k16` (f16/bf16) or `m16n8k32` (int8/fp8 e4m3 on Ada)
**[external, unverified: confirm the exact sm_89 shapes and the FP8 `mma` availability in the PTX ISA
docs for CUDA 12.x]**, loading operand fragments with `ldmatrix`
(DeepGEMM wraps it at `deep_gemm/include/deep_gemm/ptx/ld_st.cuh:33-50`, so the PTX syntax is right
there).
*DeepGEMM ideas used:* the fragment layout problem, `ldmatrix`/`stmatrix`, and the epilogue
round-trip through shared memory (§4.8).
*Pitfall:* the register→matrix-element mapping for `mma.sync` is fiddly and is where most people's
first tensor-core kernel is wrong. The CPU oracle earns its keep here.
*Note:* Ada has FP8 e4m3/e5m2 tensor-core support via `mma.sync`; it does **not** have block-scaled
MMA. So FP8 on sm_89 means §3.3's CUDA-core promotion, exactly the Hopper pattern.

**Rung 6 — per-group scaling in the mainloop/epilogue.**
Now port §3.3: two accumulators, promote once per quantization group, one scale multiply per group.
For int8: i32 tensor-core accumulate per group → convert → `sa*sb` → f32 accumulate. Validate against
the CPU int8 kernel from M5 with both metrics (§8.2).
*DeepGEMM ideas used:* everything in Part 3; the branch-free predicated promotion
(`.../sm90_fp8_gemm_1d2d.cuh:330-345`); scales staged in shared memory, loaded once per stage.
*Pitfall:* applying the scale inside the group (slow and less accurate), or forgetting that the group
boundary must divide `BLOCK_K`.

**Rung 7 — persistent kernel + rasterized scheduler.**
Grid = #SMs, loop over tiles, swizzled traversal. Now the wave-quantization model from §4.7 becomes
actionable: sweep `BLOCK_N` and show the predicted vs measured wave count.
*DeepGEMM ideas used:* `scheduler/gemm.cuh:14-26, 108-144, 185-272`.
*This rung is the best evidence for role project 4 on the GPU side.*

**Rung 8 — the config heuristic in Rust.**
Port the shape of `get_best_config`: enumerate legal tilings for sm_89, score with an L2/DRAM-traffic
model plus wave efficiency, pick, and print the choice under an env var (DeepGEMM's
`DG_PRINT_CONFIGS`). Then verify the model's ranking against an exhaustive measured sweep. "My model
picked the best config in 8 of 10 shapes" is a genuinely strong portfolio claim.

**Rung 9 — attention on GPU, if time permits.**
The MQA-logits kernels (`sm90_fp8_mqa_logits.cuh`, and the paged variant) are the closest thing in
this repo to what an inference server needs, and the paged one shows how a KV block table enters a
kernel. Read them, don't port them.

### Explicitly skip (Hopper/Blackwell only)

| Feature | Why skip on sm_89 |
|---|---|
| TMA (`cp.async.bulk.tensor`), tensor maps, TMA multicast, `arrive_and_expect_tx` | sm_90+ |
| Thread-block clusters, distributed shared memory, 2-CTA MMA | sm_90+ |
| `wgmma` (warpgroup MMA, `M=64`, async) | sm_90+ |
| `setmaxnreg` register reallocation | sm_90+ |
| PDL / `cudaGridDependencySynchronize` | sm_90+ **[external, unverified]** |
| `stmatrix` | sm_90+ **[external, unverified]** |
| `tcgen05.mma`, tensor memory (TMEM), UTCCP, block-scaled MMA, MXFP4 | sm_100+ |
| Locality domains | sm_100+ / CUDA 13.4 |
| Symmetric memory / multi-GPU mega-kernels | needs multiple GPUs |

### External references (all **[external, unverified]** — I have not re-read them at this commit)

- **CUTLASS documentation and `media/docs/`** — the canonical explanation of GEMM hierarchy
  (thread → warp → CTA tiles), `ldmatrix` fragment layouts, and pipelining. CUTLASS 2.x kernels target
  sm_80 and are the closest *readable* code to what Luigi needs for sm_89.
- **Simon Boehm, "How to Optimize a CUDA Matmul Kernel from Scratch"** (siboehm.com) — a ladder almost
  identical to Rungs 1-5, with measured numbers at each step on an sm_80-class GPU. Probably the single
  best companion to Rung 2-4.
- **NVIDIA PTX ISA documentation** — the authoritative table for which `mma.sync` shapes and dtypes
  exist on sm_89 (check this rather than trusting any blog, including these notes).
- **NVIDIA CUDA C++ Programming Guide**, async-copy and `cuda::pipeline` sections — for `cp.async`
  staging patterns.
- **Nsight Compute** — for the metric list that `scripts/quick_plot_pm.py` curates.
- **CUTLASS CuTe layout algebra** — DeepGEMM deliberately avoids it (README:5); mention it so Luigi
  knows what he is *not* learning, and why that was a defensible choice.
- **llama.cpp `ggml-cuda/mmq.cu`** — quantized matmul on consumer GPUs including Ada, i.e. the closest
  production code to Luigi's exact target. Likely the best single reference for Rungs 5-6, and already
  a planned reference for M5.

---

## Engineering practices worth importing (a checklist)

1. **Build-time gates on generated code**: no register spills, no local memory
   (`DeepJIT backend/cuda/backend.hpp:120-127`). Rust analogue: `cargo clippy -D warnings` plus a
   PTXAS gate in M7.
2. **Assert at the API boundary, loudly, before any work** (`csrc/apis/gemm.hpp:89-110`). In Rust this
   is types plus a validating constructor — cheaper than DeepGEMM has it.
3. **Bitwise determinism as a test** (`tests/test_fp8_fp4.py:78-82`), with a failure message that
   locates the first differing byte (`numeric.py:24-44`).
4. **Two code paths that must agree exactly** (`tests/utils.py:20-38`). Cheap, powerful invariants.
5. **Cache-flushing benchmarks with named baselines and geometric-mean summaries**
   (`bench.py:92-117`, `generators.py:170-173`).
6. **Document the contract that leaks** (`docs/scaling-factor-format.md` exists because the SF layout
   is the caller's job). If Luigi's engine has a layout the user must produce, it gets a doc.
7. **Comments that say *why*, including the negative results**: "making it as predicates is very
   important for performance, comparing to two loops" (`.../sm90_fp8_gemm_1d2d.cuh:330`); "think twice
   before modifying this, as changes may affect the number of instructions" (`:402`); "32B's
   performance is low" (`sm90.hpp:101`); "avoid `std::filesystem::remove_all` here — it can segfault
   on distributed filesystems" (`cache/disk.hpp:57-59`). These are measurements preserved as prose —
   exactly what `DECISIONS.md` is for.
8. **`TODO`s that name the gap**: "TODO: calculate expected cycles" (`sm100.hpp:271`), "TODO: check
   256's performance" (`sm90.hpp:22`). Shipping with known gaps, written down.
9. **Lazy initialization so importing costs nothing** (`utils/lazy.hpp:16-40`, tested at
   `DeepJIT tests/test_cuda.py:137`). Relevant to Luigi's CLI startup time and to TTFT.
10. **An env-var debug surface** (README:166-191) instead of recompiling to investigate.

---

## Milestone map

| Milestone | DeepGEMM/DeepJIT idea to teach | Where in these notes | Source anchor |
|---|---|---|---|
| **M0** plan + perf model | Model only what varies between options; `max(mem_cycles, flop_cycles) / efficiency`; wave/load-balance efficiency as an explicit term | §5, §4.7 | `heuristics/sm90.hpp:202-239` |
| **M0/M9** docs | Document the contract that leaks; decisions-as-comments | §9, practices 6-7 | `docs/scaling-factor-format.md`, README:70 |
| **M1** tensors + kernels | Layout is part of the type; `align`/`ceil_div` discipline; test against hand-computed values then a reference | §3.2, §8 | `csrc/utils/layout.hpp`, `numeric.py:5-11` |
| **M2** weight loading | A "cache key" is a claim about dependencies (what would invalidate your mmap'd parse?) | §7.2 | `DeepJIT runtime.hpp:92-98` |
| **M3** forward + KV cache | Two paths that must agree bitwise (batched prefill vs token-by-token decode) | §8.2 practice 4 | `tests/utils.py:20-38` |
| **M4** sampling + generate | Decode is a GEMV and it is in the shape sweep (M=1); measure with cache flush + warmup; name the baseline | §2.2, §8.3 | `generators.py:129`, `bench.py:9-33` |
| **M5** int8 / 4-bit | **The core lesson**: quantization group = accumulation chunk; scale applied once per group; i32 → f32 promotion; power-of-two scales; two references (f32 and dequantized) | §3.1, §3.3, §3.5, §8.2 | `sm90_fp8_gemm_1d2d.cuh:251-345`, `common/math.cuh:114-146` |
| **M5** SF layout | Scale arrays need a layout chosen by the consumer's load pattern; validate it | §3.2 | `docs/scaling-factor-format.md:53-63` |
| **M6** make it fast | Cache blocking + panel packing = shared-memory tiling; "wave quantization" = load imbalance over 24 threads; branch-free inner loops; producer/consumer packing; measure every change | §4.1-4.8, §4.7 worked example | `scheduler/gemm.cuh:14-26`, `sm90.hpp:202-239` |
| **M6** batched prefill | The matrix×matrix vs matrix×vector split *is* the `m<=16 / m<=32 / else` special-casing in the heuristic | §4.7 | `sm90.hpp:25-26`, `sm100.hpp:63-67` |
| **M7** CUDA backend | The whole ladder: naive → smem → registers → `cp.async` stages → `mma.sync` → per-group scaling → persistent+swizzled → heuristic in Rust | §10 | all of Part 4 |
| **M7** host design | AOT nvcc in build.rs vs NVRTC vs DeepJIT-style cache — tradeoffs, no decision | §7.5 | `DeepJIT runtime.hpp:56-98`, `cache/disk.hpp:42-67` |
| **M7** validation | Every GPU kernel vs the CPU oracle, both metrics, plus bitwise determinism | §8.2 | `tests/test_fp8_fp4.py:54-82` |
| **M8** real model | Grouped GEMM layouts exist because MoE experts have variable token counts; contiguous vs masked = prefill vs decode-under-graph-capture | §1.4 | README:76-86 |
| **post-v1 server** | tc_util throttling and locality domains as system-level perf; PDL as kernel-overlap; masked grouped GEMM as the CUDA-graph-friendly decode shape | §4.9, §4.10, §4.12 | `sm100_bf16_gemm.cuh:326-343`, README:82-86 |

---

## Quiz questions for Luigi

Ordered roughly by milestone. Answers are in the sections named.

**On quantization (M5) — §3**
1. DeepGEMM quantizes activations at `(1, 128)` and weights at `(128, 128)`. Why are the two
   granularities different? What breaks if you use per-tensor scales for activations?
2. In the Hopper FP8 kernel there are two FP32 accumulators, `accum` and `final_accum`. What is each
   one for, and why can't there be just one?
3. You quantize int8 in groups of 32. Where exactly in the dot product does the scale multiply
   happen, and how many multiplies per 32 elements? What would per-element scaling cost you?
4. `i32` accumulation of int8 products: how many products can you add before overflow, assuming worst
   case |a|,|b| = 127? What does that tell you about per-tensor vs per-group scaling?
5. DeepSeek forces every scale to be a power of two. What does that buy, and what does it cost?
6. The SF tensor is "MN-major, TMA-aligned". Restate both properties in terms of *what the consumer
   wants*, without using the words TMA or MN-major.
7. Why does `per_token_cast_to_fp8` clamp amax to `1e-4` before dividing by 448?

**On the performance model (M0, M6) — §5, §4.7**
8. The SM90 cost model ignores HBM bandwidth and total FLOPs entirely. Why is that legitimate?
9. You have 46 SMs (or 24 threads) and 8 tiles of work. What is the efficiency, and which term of
   `num_cycles = max(l1, l2) / wave_efficiency` captures it?
10. Why would a library ever choose `BLOCK_N = 112` over 128 when 128 maps better onto the MMA shape?
11. `wave_efficiency` appears as a *divisor*. Sanity-check the units and explain why dividing is the
    right operation.
12. The model disables multicast when `num_waves <= 1` by setting the cost to `INT64_MAX`. What is
    the physical reason?

**On kernel architecture (M7) — §4**
13. What is a persistent kernel, and what does the grid size become? What does the block get in
    exchange for the extra scheduler code?
14. Explain block swizzling to someone who knows what L2 is but not what a GEMM tile is. Then compute
    which group size (8 or 16) the formula picks for `BLOCK_M = BLOCK_N = 128` on 46 SMs.
15. In a warp-specialized kernel, what does the TMA warpgroup do with its registers, and why does
    that make the math warps faster?
16. You have 100 KB of shared memory, `BLOCK_M = BLOCK_N = 128`, `BLOCK_K = 32`, int8 operands. How
    many pipeline stages fit? Why is "as many as fit" the wrong answer?
17. Name the sm_89 replacement for each of: TMA, mbarrier, `wgmma`, `setmaxnreg`, `stmatrix`.
18. Why does the epilogue write accumulators to shared memory before storing to global memory,
    instead of storing straight from registers?

**On testing and benchmarking (M4-M7) — §8**
19. Write down `calc_diff` and show it equals `‖x-y‖² / (‖x‖² + ‖y‖²)`. What class of bug does it
    catch that a max-abs bound misses, and vice versa?
20. The attention test uses two references with tolerances 1e-3 and 5e-6. What does each one isolate?
    Design the equivalent pair for your M5 int8 matmul.
21. Why flush 256 MB of L2 before *every* timed iteration? What is the CPU equivalent, and when would
    flushing be the *wrong* thing to do?
22. Why is the cross-shape summary a geometric mean of speedups rather than an arithmetic mean?
23. Your int8 matmul passes `calc_diff < 1e-3` but a bitwise-determinism rerun fails. What kinds of
    bug does that combination point at?

**On JIT and API design (M7, M9) — §6, §7, §9**
24. Why does burning `N` and `K` into the compiled kernel help, and why is `M` deliberately left
    dynamic?
25. List the five components of DeepJIT's cache key. Name one thing it deliberately does not cover,
    and what goes wrong if that thing changes.
26. Explain the atomic-rename publish. What exactly could a concurrent reader see if the marker file
    were written *after* the rename instead of before?
27. `fp8_gemm_nt` takes 3 required arguments and hides ~15 tuning decisions. Argue both sides of
    exposing `block_m`/`block_n` in your Rust `matmul` API, then decide.
28. DeepGEMM removed its SASS post-processing because "NVCC 12.9 will automatically do the FFMA
    interleaving". What is the general lesson for your M6 AVX2 work?

---

## Open questions (things I could not resolve from these repos)

1. **Accumulation precision.** The code promotes every 128 K-elements but never says the tensor-core
   accumulator is lower precision than FP32. Is the promotion motivated by precision at all, or
   purely by the scale granularity? At this commit, only the scale motivation is documented. Worth
   resolving before telling Luigi anything definite about Hopper FP8 accumulator width.
2. **Performance table.** The README has no shape-by-shape table at this commit and the repo has a
   single squashed commit, so I cannot quote per-shape TFLOPS. The linked PRs (#74, #78, #81, #86,
   #112, #304, #316, #432, #462) presumably contain them; they need network access to read.
3. **The bandwidth constants** in the SM90 cost model (`64 * num_sms`, `128 * num_sms`, `8e6/1.3e3`)
   are unexplained. Which are peak specs and which are measured/fudge factors? This matters if Luigi
   copies the *method* — he needs to know his own constants' provenance.
4. **`smem_capacity = 232448`** is hard-coded identically for SM90 and SM100 rather than read from
   `jit->device.get_num_smem_bytes()` (which exists, `DeepJIT device.hpp:49`). Deliberate (so the
   heuristic is deterministic across devices) or an oversight?
5. **`kNumUMMACycles = 2*UMMA_M*UMMA_N*BLOCK_K / 8192`** — the 8192 implies a per-cycle MMA
   throughput. Which unit is it (per SM? per SM sub-partition?), and does it hold for BF16 and FP8
   alike?
6. **`get_num_1d_blocks_per_group` candidates are only {8, 16}.** Was 4 or 32 measured and rejected?
7. **sm_89 FP8 `mma.sync`**: I have not verified in the PTX ISA which FP8 `mma` shapes Ada exposes and
   whether the accumulate type is FP32 or FP16. Rung 5 of the learning path depends on this; check the
   ISA docs before committing Luigi to an FP8 path rather than int8 on the GPU.
8. **Locality domains**: my reading (NUMA-like memory domains on a partitioned GPU) is inference from
   the allocator and the SM-probe code. Not needed for this project, but don't assert it.
9. **`tilelang_ops`** appears in `third-party/` but is empty in this checkout; the legacy Triton
   kernels mention a future TileLang rewrite (`deep_gemm/legacy/__init__.py:1`). Unknown scope.
10. **DeepJIT submodule mismatch**: DeepGEMM pins `2efdab42` while the standalone DeepJIT checkout is
    at `3732a3b9` (a PyTorch-stable-ABI migration). Notes about DeepJIT reflect the *newer* standalone
    code; if a detail matters, check which one DeepGEMM actually compiles against.
