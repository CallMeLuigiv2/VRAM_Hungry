# Tolerance: how close is "correct"?

Measured 2026-09-25 for decision **D1** (see `DECISIONS.md`). Scripts: `scripts/tolerance/`.

## What we were testing, and why

Our engine will never produce exactly the same logits as the Python reference. Float addition depends on order: in f32, `(1e8 + 1) - 1e8 = 0` but `(1e8 - 1e8) + 1 = 1`. Two correct engines that add the same numbers in a different order disagree in the last digits. So the correctness test needs a tolerance: a line between **rounding noise** (harmless) and **a bug**.

We didn't want to guess that line, so we measured both sides:

1. **Noise.** How far apart are two *correct* f32 engines on this model?
2. **Bugs.** How far off is an engine with a realistic bug?

The tolerance goes between the two.

## Setup

- **Model:** TinyStories stories15M (dim 288, 6 layers, 6 heads, vocab 32000, max_seq_len 256).
- **"Lab scale" (the truth):** llama2.c `model.py` run in **float64**. `model.py` hard-codes `.float()` in RMSNorm, RoPE and the RoPE table, so `model.to(float64)` alone would still run those parts in f32. `refs.py` builds a copy with all 6 casts changed to `.double()`, and asserts there are exactly 6.
- **"Kitchen scales" (f32 engines):**
  - llama2.c `model.py` as-is in float32. It computes all positions in one pass, prefill-style, with blocked SIMD sums.
  - llama2.c `run.c`, unmodified, driven by `dump.c`. It computes one token at a time through its KV cache, decode-style, with naive loops. We built it three ways:
    - `-O3`: plain loops, no FMA, one long sequential sum per dot product. **Closest to our own first engine**, since Rust never reorders float math.
    - `-O3 -march=native`: the compiler fuses multiply-adds into FMA instructions (51 of them in the binary). Similar to M6's AVX2.
    - `-Ofast`: fast-math, so the compiler may reorder sums.
- **Data:** 20 short prompts (`refs.PROMPTS`). Each one gets BOS prepended and is extended greedily to 256 tokens, then every run sees the same tokens. That's 20 × 256 positions × 32,000 logits ≈ **164 million logits** per comparison.
- **Metric:** the worst |ours − reference| over all of them. We use the worst case, not the average, because one bad logit is enough to change the output.

Before measuring, we estimated the noise from first principles: about 1e-7 relative per f32 op, logits around size 10, and errors partly piling up over 6 layers of 288- to 768-term sums. That gave **1e-5 to 1e-4**.

## Results

### Noise: correct engines vs the f64 truth

| Engine | Worst diff | Average diff | Winner flips |
|---|---|---|---|
| PyTorch f32 | 3.8e-5 | 2.9e-6 | 0 / 5120 |
| run.c `-O3` (naive) | **5.2e-5** | 4.7e-6 | 0 / 5120 |
| run.c `-O3 -march=native` (FMA) | 5.1e-5 | 4.7e-6 | 0 / 5120 |
| run.c `-Ofast` (fast-math) | 3.6e-5 | 2.5e-6 | 0 / 5120 |
| **run.c `-O3` vs PyTorch f32** (two f32 engines, like our M3 test) | **5.5e-5** | | |

The estimate held: the measured worst case, 3.8e-5 to 5.5e-5, is inside 1e-5 to 1e-4.

### Bugs: planted in PyTorch f32, vs the f64 truth (first 5 prompts)

| Bug | Worst diff |
|---|---|
| RMSNorm epsilon 1e-6 instead of 1e-5 | 0.28 |
| Off-by-one: final dot product loops `0..dim-1` | 1.95 |

### Where the line goes

```
0.000055   naive engine's worst noise
0.00011    2× line
0.00055    10× line   ← chosen factor (Luigi, 2026-09-25)
0.0055     100× line
0.28       smallest planted bug
```

Luigi chose **10× the naive engine's worst noise**. That gives room above the noise for engines and prompts we didn't test, while staying far below real bugs. He rounded it to **atol = 5e-4**: about 9× above the noise and 560× below the smallest bug. With that, a winner flip only counts as a "near-tie" when the reference's top-two gap is under 2 × 5e-4 = 1e-3. The closest race in our data had a gap of 1.5e-3.

This tolerance is for **f32 engine vs reference only**. Quantized weights (M5) and GPU kernels (M7) get their own tolerances, measured the same way.

## Findings worth remembering

1. **The average hides the tail.** The worst diff was 13× the average. A tolerance built from the average would have failed correct engines.
2. **Naive loops are about 1.4× noisier than PyTorch** (5.2e-5 vs 3.8e-5). One long sequential sum piles up more rounding than several short ones.
3. **Fast-math was *more* accurate** (3.6e-5). The compiler may split a long sum into parallel partial sums, and shorter sums round less. Expect M6's AVX2 kernels, with 8 partial sums, to lower the noise. Measure it then.
4. **Threads don't have to change results.** PyTorch with 1 thread vs 12 threads gave bit-identical logits, because each thread computes whole dot products, so every sum keeps its order. In M6, split work by output rows and an "engine vs itself" check can stay bit-exact.
5. **Prefill and decode agree.** run.c (one token at a time, KV cache) and PyTorch (all positions at once) match within 5.5e-5, with zero flips. They're the same math, grouped differently.
6. **No growth with position.** run.c's worst diff was about the same at positions 0–63 as at 192–255.
7. **The error is roughly absolute, not relative.** Some logit of size 1 to 1.7 was off by about 3e-5, which is a relative error of 3e-5. If errors grew in proportion to size, a logit of 28.75 would be off by about 9e-4. Yet no logit anywhere was off by more than 5.5e-5. So atol does the work, and rtol adds little.
8. **Check your reference's precision.** `model.py` silently runs parts in f32 even when the model is converted to f64.

## Limits

- One model (stories15M), 20 prompts, one machine. Our Rust engine's own noise gets measured for real in M3.
- The planted bugs are two examples. Real bugs can be smaller, which is why the line sits at 10× and not 100×.

## How to rerun

One-time setup (the large files live outside the repo):

```bash
mkdir -p ~/refs/inference/models && cd ~/refs/inference
git clone https://github.com/karpathy/llama2.c        # measured at 350e04f
curl -L -o models/stories15M.pt  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt
curl -L -o models/stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
python3 -m venv --system-site-packages venv && venv/bin/pip install sentencepiece   # torch from system Python
```

Checkpoint sha256:
- `stories15M.pt`: `3da00c0fef684f3f83b457736837c46ab55e92a26662b61d6104de2d271c708d`
- `stories15M.bin`: `cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a`

The two files hold bit-identical weights (checked on the embeddings, layer-0 `wq`, and layer-0 RMSNorm).

Run (from the repo root, about 7 minutes in total):

```bash
~/refs/inference/venv/bin/python scripts/tolerance/noise_pytorch.py   # ~2 min
~/refs/inference/venv/bin/python scripts/tolerance/planted_bugs.py    # ~1 min
~/refs/inference/venv/bin/python scripts/tolerance/noise_runc.py      # ~4 min, compiles dump.c 3 ways
```

Environment: AMD Ryzen 9 5900X (12 cores / 24 threads), WSL2 Ubuntu, Python 3.12.3, PyTorch 2.12.0 (CPU), sentencepiece 0.2.2, gcc 14.2.0.

<details>
<summary>Raw output (overall sections)</summary>

`noise_pytorch.py`
```
f32 vs f64
  max |diff|               : 3.769e-05
  median of per-prompt max : 2.763e-05
  99.9th pct |diff| (worst prompt): 2.622e-05
  mean |diff|              : 2.907e-06
  max rel diff (|ref|>=1)  : 2.831e-05
  largest |logit|          : 28.75
  max |diff| by position 0-63 / 64-127 / 128-191 / 192-255: 2.79e-05 / 2.91e-05 / 3.77e-05 / 3.52e-05
  argmax flips             : 0 of 5120 positions
  smallest ref top-2 gap   : 1.54e-03

f32 (1 thread) vs f32 (12 threads)
  max |diff|               : 0.000e+00
  mean |diff|              : 0.000e+00
  argmax flips             : 0 of 5120 positions
```

`planted_bugs.py`
```
worst, RMSNorm eps 1e-6 (should be 1e-5): 2.82e-01
worst, off-by-one in classifier dot product: 1.95e+00
```

`noise_runc.py`
```
run.c -O3 (plain, no FMA)
  max |diff| vs f64         : 5.217e-05
  mean |diff| vs f64        : 4.723e-06
  max rel diff (|ref|>=1)   : 3.144e-05
  max |diff| by position 0-63 / 64-127 / 128-191 / 192-255: 5.22e-05 / 4.60e-05 / 4.57e-05 / 4.87e-05
  max |diff| vs PyTorch f32 : 5.531e-05   (two kitchen scales)
  argmax flips vs f64       : 0 of 5120

run.c -O3 -march=native (FMA)
  max |diff| vs f64         : 5.094e-05
  mean |diff| vs f64        : 4.745e-06
  max rel diff (|ref|>=1)   : 3.431e-05
  max |diff| by position 0-63 / 64-127 / 128-191 / 192-255: 5.06e-05 / 4.10e-05 / 4.77e-05 / 5.09e-05
  max |diff| vs PyTorch f32 : 5.150e-05   (two kitchen scales)
  argmax flips vs f64       : 0 of 5120

run.c -Ofast (fast-math)
  max |diff| vs f64         : 3.631e-05
  mean |diff| vs f64        : 2.505e-06
  max rel diff (|ref|>=1)   : 2.078e-05
  max |diff| by position 0-63 / 64-127 / 128-191 / 192-255: 2.96e-05 / 2.77e-05 / 3.63e-05 / 2.98e-05
  max |diff| vs PyTorch f32 : 3.815e-05   (two kitchen scales)
  argmax flips vs f64       : 0 of 5120
```
</details>
