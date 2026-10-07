"""How big is a real bug? Two planted bugs in PyTorch f32, measured against the f64 lab scale
on the first 5 test sequences.

usage: ~/refs/inference/venv/bin/python scripts/tolerance/planted_bugs.py
"""
import itertools

import torch

from refs import all_logits, load, load_both, m32, sequences


def with_wrong_eps():
    """RMSNorm epsilon 1e-6 instead of the 1e-5 the checkpoint was trained with."""
    model = load(m32, torch.float32)
    for mod in model.modules():
        if isinstance(mod, m32.RMSNorm):
            mod.eps = 1e-6
    return lambda seq: all_logits(model, seq)


def with_off_by_one(model):
    """The final classifier dot product loops 0..dim-1, skipping the last term."""
    @torch.inference_mode()
    def run(tokens):
        T = tokens.shape[1]
        h = model.tok_embeddings(tokens)
        for layer in model.layers:
            h = layer(h, model.freqs_cos[:T], model.freqs_sin[:T])
        h = model.norm(h)[0]
        W = model.output.weight
        return (h[:, :-1] @ W[:, :-1].T).double()
    return run


def main():
    f64, f32 = load_both()
    bugs = {"RMSNorm eps 1e-6 (should be 1e-5)": with_wrong_eps(),
            "off-by-one in classifier dot product": with_off_by_one(f32)}
    worst = dict.fromkeys(bugs, 0.0)
    for p, seq in itertools.islice(sequences(f32), 5):
        ref = all_logits(f64, seq)
        for name, run in bugs.items():
            d = (run(seq) - ref).abs()
            worst[name] = max(worst[name], d.max().item())
            print(f"{p[:30]:30s} {name:38s} max |diff| = {d.max().item():.2e}, mean = {d.mean().item():.1e}")
    print()
    for name, w in worst.items():
        print(f"worst, {name}: {w:.2e}")


if __name__ == "__main__":
    main()
