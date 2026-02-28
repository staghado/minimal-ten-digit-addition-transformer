"""
Train a 200-param Qwen3 to do 10-digit addition.
"""

import random
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import Model, ModelArgs

MODEL_DIM = 3
ATTENTION_HEADS = 2
KEY_VALUE_HEADS = 1
HEAD_DIM = 4
INTERMEDIATE_SIZE = 6
VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1
ROPE_THETA = 3

MAX_STEPS = 15_001
BATCH_SIZE = 128
LR = 0.01

model_args = ModelArgs(
    model_type="qwen3",
    hidden_size=MODEL_DIM,
    num_hidden_layers=1,
    intermediate_size=INTERMEDIATE_SIZE,
    num_attention_heads=ATTENTION_HEADS,
    rms_norm_eps=1e-6,
    vocab_size=VOCAB_SIZE,
    tie_word_embeddings=True,
    num_key_value_heads=KEY_VALUE_HEADS,
    max_position_embeddings=64,
    rope_theta=ROPE_THETA,
    head_dim=HEAD_DIM,
)


def encode(a, b):
    pa, pb = f"{a:010d}", f"{b:010d}"
    return (
        [0]
        + [int(c) for c in reversed(pa)]
        + [0, 0]
        + [int(c) for c in reversed(pb)]
        + [0]
    )


def expected_output(a, b):
    s = str(a + b)[::-1].ljust(OUTPUT_DIGITS, "0")
    return [int(c) for c in s]


def make_batch(batch_size, rng):
    inp, tgt = [], []
    for _ in range(batch_size):
        a, b = rng.randint(0, MAX_ADDEND), rng.randint(0, MAX_ADDEND)
        inp.append(encode(a, b))
        tgt.append(expected_output(a, b))
    return mx.array(inp, dtype=mx.int32), mx.array(tgt, dtype=mx.int32)


def loss_fn(model, x, y):
    loss = mx.zeros(())
    for i in range(OUTPUT_DIGITS):
        logits = model(x)
        loss = loss + nn.losses.cross_entropy(
            logits[:, -1, :], y[:, i], reduction="mean"
        )
        x = mx.concatenate([x, y[:, i : i + 1]], axis=1)
    return loss / OUTPUT_DIGITS


def evaluate(model, n, rng):
    ok_seq, ok_dig, tot = 0, 0, 0
    for _ in range(n):
        a, b = rng.randint(0, MAX_ADDEND), rng.randint(0, MAX_ADDEND)
        exp = expected_output(a, b)
        seq = encode(a, b)
        for _ in range(OUTPUT_DIGITS):
            logits = model(mx.array([seq], dtype=mx.int32))
            seq.append(int(mx.argmax(logits[0, -1, :]).item()))
        pred = seq[-OUTPUT_DIGITS:]
        m = sum(p == e for p, e in zip(pred, exp))
        ok_dig += m
        tot += OUTPUT_DIGITS
        ok_seq += m == OUTPUT_DIGITS
    return ok_seq / n, ok_dig / tot


def count_parameters(model):
    return sum(x.size for _, x in tree_flatten(model.parameters()))


# train
np.random.seed(42)
mx.random.seed(42)
rng = random.Random(42)

model = Model(model_args)
mx.eval(model.parameters())
n_params = count_parameters(model)
print(f"params: {n_params}")
opt = optim.AdamW(learning_rate=LR)
grad_fn = nn.value_and_grad(model, loss_fn)
t0 = time.time()

for step in range(1, MAX_STEPS):
    x, y = make_batch(BATCH_SIZE, rng)
    loss, grads = grad_fn(model, x, y)
    grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state)
    if step % 100 == 0:
        print(f"{step:6d} | loss {loss.item():.4f} | {time.time() - t0:.0f}s")
    if step % 2000 == 0:
        model.eval()
        sa, da = evaluate(model, 200, random.Random(999))
        model.train()
        print(f"       | seq {sa:.3f} dig {da:.3f}")

model.eval()
sa, da = evaluate(model, 1000, random.Random(12345))
print(f"FINAL  | seq {sa:.3f} dig {da:.3f}")
mx.savez(f"checkpoint/best_{n_params}.npz", **dict(tree_flatten(model.parameters())))
