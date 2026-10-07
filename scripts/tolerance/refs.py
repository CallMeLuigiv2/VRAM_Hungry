"""Shared setup for the tolerance measurements (see test_ref/tolerance.md).

"Lab scale":     llama2.c model.py with every .float() cast turned into .double(), run in float64.
"Kitchen scale": llama2.c model.py as-is, run in float32.
"""
import sys
import types
from pathlib import Path

import torch

REFS = Path.home() / "refs" / "inference"
LLAMA2C = REFS / "llama2.c"
CKPT_PT = REFS / "models" / "stories15M.pt"
CKPT_BIN = REFS / "models" / "stories15M.bin"
TOKENIZER = LLAMA2C / "tokenizer.model"
BOS = 1  # llama2.c prepends BOS to every prompt

PROMPTS = [
    "Once upon a time", "One day, a little girl named Lily", "Tom had a red ball",
    "The cat sat on the mat", "There was a big tree in the park",
    "Sam wanted to fly like a bird", "The sun was hot and the sky was blue",
    "Mom said, \"Time for bed.\"", "A little dog found a bone", "Ben and Mia went to the beach",
    "The old man had a secret", "It was raining outside", "The princess lost her crown",
    "A tiny frog lived in a pond", "Once there was a boy who did not like to share",
    "The robot beeped", "Anna baked a cake for her friend", "The bear was very hungry",
    "Lucy found a shiny key", "The wind blew the leaves",
]

sys.path.insert(0, str(LLAMA2C))
import model as m32  # noqa: E402  (llama2.c's model.py, unmodified)


def _model64():
    """model.py hard-codes .float() in RMSNorm, RoPE and the RoPE table, so model.to(float64)
    would still run those parts in f32. Build an all-f64 copy instead of editing the reference."""
    src = (LLAMA2C / "model.py").read_text()
    n = src.count(".float()")
    assert n == 6, f"llama2.c model.py changed: expected 6 .float() casts, found {n}"
    mod = types.ModuleType("model64")
    sys.modules["model64"] = mod  # @dataclass looks its module up here
    exec(compile(src.replace(".float()", ".double()"), str(LLAMA2C / "model.py"), "exec"), mod.__dict__)
    return mod


m64 = _model64()


def load(mod, dtype):
    ck = torch.load(CKPT_PT, map_location="cpu")
    model = mod.Transformer(mod.ModelArgs(**ck["model_args"]))
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.to(dtype)
    assert model.freqs_cos.dtype == dtype
    assert model.layers[0].attention.flash  # SDPA path; the manual path has its own cast
    return model


def load_both():
    return load(m64, torch.float64), load(m32, torch.float32)


def tokenizer():
    import sentencepiece as spm
    return spm.SentencePieceProcessor(model_file=str(TOKENIZER))


@torch.inference_mode()
def all_logits(model, tokens):
    """model.forward only returns the last position; we want every position. Returns (T, vocab) f64."""
    T = tokens.shape[1]
    h = model.tok_embeddings(tokens)
    for layer in model.layers:
        h = layer(h, model.freqs_cos[:T], model.freqs_sin[:T])
    return model.output(model.norm(h))[0].double()


@torch.inference_mode()
def greedy_extend(model, ids, total):
    """Extend a prompt greedily to `total` tokens so every position up to max_seq_len gets tested."""
    x = torch.tensor([ids])
    while x.shape[1] < total:
        nxt = all_logits(model, x)[-1].argmax()
        x = torch.cat([x, nxt.view(1, 1)], dim=1)
    return x


def sequences(f32):
    """The 20 test sequences: BOS + prompt, greedily extended (by PyTorch f32) to max_seq_len."""
    sp = tokenizer()
    total = f32.params.max_seq_len
    for p in PROMPTS:
        yield p, greedy_extend(f32, [BOS] + sp.encode(p), total)
