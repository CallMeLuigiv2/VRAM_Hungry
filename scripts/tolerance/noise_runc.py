"""Noise floor of llama2.c run.c (naive C loops, KV cache, one token at a time) vs the f64 lab scale,
for three builds. run.c -O3 is the closest stand-in for our own naive f32 engine.

usage: ~/refs/inference/venv/bin/python scripts/tolerance/noise_runc.py
"""
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from refs import CKPT_BIN, LLAMA2C, all_logits, load_both, sequences

HERE = Path(__file__).resolve().parent
BUILDS = {
    "run.c -O3 (plain, no FMA)": ["-O3"],
    "run.c -O3 -march=native (FMA)": ["-O3", "-march=native"],
    "run.c -Ofast (fast-math)": ["-Ofast"],
}


def build(tmp):
    bins = {}
    for i, (name, flags) in enumerate(BUILDS.items()):
        out = tmp / f"dump{i}"
        cmd = ["gcc", *flags, "-I", str(LLAMA2C), "-o", str(out), str(HERE / "dump.c"), "-lm"]
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)
        bins[name] = out
    return bins


def runc_logits(binary, seq, tmp):
    tok, out = tmp / "tokens.i32", tmp / "out.f32"
    seq.numpy().astype(np.int32).tofile(tok)
    subprocess.run([str(binary), str(CKPT_BIN), str(tok), str(out)], check=True)
    return torch.from_numpy(np.fromfile(out, dtype=np.float32).reshape(seq.shape[1], -1)).double()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="noise_runc_"))
    bins = build(tmp)
    f64, f32 = load_both()
    agg = {name: dict(max64=0.0, maxpt=0.0, mean64=[], q=[0.0] * 4, flips=[], rel=0.0) for name in BUILDS}
    t0 = time.time()
    for i, (p, seq) in enumerate(sequences(f32)):
        L64, L32 = all_logits(f64, seq), all_logits(f32, seq)
        top2 = L64.topk(2, dim=-1).values
        gap = top2[:, 0] - top2[:, 1]
        big = L64.abs() >= 1.0
        line = []
        for name, binary in bins.items():
            R = runc_logits(binary, seq, tmp)
            d = (R - L64).abs()
            a = agg[name]
            a["max64"] = max(a["max64"], d.max().item())
            a["maxpt"] = max(a["maxpt"], (R - L32).abs().max().item())
            a["mean64"].append(d.mean().item())
            a["rel"] = max(a["rel"], (d[big] / L64[big].abs()).max().item())
            for k in range(4):
                a["q"][k] = max(a["q"][k], d[k * 64:(k + 1) * 64].max().item())
            a["flips"] += [gap[j].item() for j in (R.argmax(-1) != L64.argmax(-1)).nonzero().flatten().tolist()]
            line.append(f"{d.max().item():.2e}")
        print(f"[{i+1:2d}/20] {p[:30]:30s} max|diff| vs f64  O3={line[0]}  native={line[1]}  Ofast={line[2]}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== overall: 20 prompts x 256 positions x 32000 logits ===")
    for name, a in agg.items():
        print(f"\n{name}")
        print(f"  max |diff| vs f64         : {a['max64']:.3e}")
        print(f"  mean |diff| vs f64        : {np.mean(a['mean64']):.3e}")
        print(f"  max rel diff (|ref|>=1)   : {a['rel']:.3e}")
        print("  max |diff| by position 0-63 / 64-127 / 128-191 / 192-255: " + " / ".join(f"{x:.2e}" for x in a["q"]))
        print(f"  max |diff| vs PyTorch f32 : {a['maxpt']:.3e}   (two kitchen scales)")
        print(f"  argmax flips vs f64       : {len(a['flips'])} of 5120"
              + (f"; ref top-2 gaps at flips: {', '.join(f'{g:.1e}' for g in a['flips'])}" if a["flips"] else ""))


if __name__ == "__main__":
    main()
