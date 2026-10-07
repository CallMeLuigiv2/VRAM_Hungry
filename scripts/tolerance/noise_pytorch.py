"""Noise floor of PyTorch f32 inference on stories15M: f32 vs f64, and 1 thread vs all threads.

usage: ~/refs/inference/venv/bin/python scripts/tolerance/noise_pytorch.py
"""
import time

import torch

from refs import all_logits, load_both, sequences


def stats(test, ref):
    d = (test - ref).abs()
    big = ref.abs() >= 1.0
    T = ref.shape[0]
    top2 = ref.topk(2, dim=-1).values
    gap = top2[:, 0] - top2[:, 1]
    flips = (test.argmax(-1) != ref.argmax(-1)).nonzero().flatten().tolist()
    return dict(
        max_abs=d.max().item(), mean_abs=d.mean().item(),
        p999_abs=d.flatten().kthvalue(int(0.999 * d.numel())).values.item(),
        max_rel_big=(d[big] / ref[big].abs()).max().item(),
        max_ref=ref.abs().max().item(),
        by_quarter=[d[i * T // 4:(i + 1) * T // 4].max().item() for i in range(4)],
        flips=[gap[p].item() for p in flips], min_gap=gap.min().item(),
    )


def main():
    t0 = time.time()
    f64, f32 = load_both()
    nt = torch.get_num_threads()
    names = ["f32 vs f64", f"f32 (1 thread) vs f32 ({nt} threads)"]
    allr = {n: [] for n in names}
    print(f"max_seq_len={f32.params.max_seq_len}, torch threads={nt}")

    for i, (p, seq) in enumerate(sequences(f32)):
        L64, L32 = all_logits(f64, seq), all_logits(f32, seq)
        torch.set_num_threads(1)
        L32_1 = all_logits(f32, seq)
        torch.set_num_threads(nt)
        a, b = stats(L32, L64), stats(L32_1, L32)
        allr[names[0]].append(a)
        allr[names[1]].append(b)
        print(f"[{i+1:2d}/20] {p[:34]:34s} f32-vs-f64 max={a['max_abs']:.2e} mean={a['mean_abs']:.1e}  "
              f"1t-vs-{nt}t max={b['max_abs']:.2e}  flips={len(a['flips'])}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== overall, 20 prompts x 256 positions x 32000 logits each ===")
    for name, rs in allr.items():
        print(f"\n{name}")
        print(f"  max |diff|               : {max(r['max_abs'] for r in rs):.3e}")
        print(f"  median of per-prompt max : {sorted(r['max_abs'] for r in rs)[10]:.3e}")
        print(f"  99.9th pct |diff| (worst prompt): {max(r['p999_abs'] for r in rs):.3e}")
        print(f"  mean |diff|              : {sum(r['mean_abs'] for r in rs)/len(rs):.3e}")
        print(f"  max rel diff (|ref|>=1)  : {max(r['max_rel_big'] for r in rs):.3e}")
        print(f"  largest |logit|          : {max(r['max_ref'] for r in rs):.2f}")
        qs = [max(r['by_quarter'][k] for r in rs) for k in range(4)]
        print("  max |diff| by position 0-63 / 64-127 / 128-191 / 192-255: " + " / ".join(f"{x:.2e}" for x in qs))
        fl = [g for r in rs for g in r["flips"]]
        print(f"  argmax flips             : {len(fl)} of {20*256} positions"
              + (f"; ref top-2 gaps at flips: {', '.join(f'{g:.1e}' for g in fl)}" if fl else ""))
        print(f"  smallest ref top-2 gap   : {min(r['min_gap'] for r in rs):.2e}")


if __name__ == "__main__":
    main()
