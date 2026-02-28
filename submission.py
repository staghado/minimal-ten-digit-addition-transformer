"""
Submission for Nano Transformer Adder leaderboard.
"""

import os
import mlx.core as mx
from mlx_lm.models.qwen3 import Model, ModelArgs

CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoint", "best.npz")

MODEL_DIM = 4
ATTENTION_HEADS = 2
KEY_VALUE_HEADS = 1
HEAD_DIM = 4
INTERMEDIATE_SIZE = 6
VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1
ROPE_THETA = 3


def _build_model_args():
    return ModelArgs(
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


def _encode(a: int, b: int) -> list[int]:
    pa, pb = f"{a:010d}", f"{b:010d}"
    return ([0] + [int(c) for c in reversed(pa)] + [0] +
            [0] + [int(c) for c in reversed(pb)] + [0])


def build_model():
    model = Model(_build_model_args())
    weights = list(mx.load(CHECKPOINT).items())
    model.load_weights(weights)
    model.eval()
    mx.eval(model.parameters())

    metadata = {
        "name": "228-param Qwen3 Adder",
        "author": "staghado",
        "params": 228,
        "architecture": "1L Qwen3, d=4, 2h/1kv, hd=4, ff=6",
        "tricks": ["Tied embeddings", "RoPE theta=3"]
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    seq = _encode(a, b)
    digits = []
    for _ in range(OUTPUT_DIGITS):
        x = mx.array([seq], dtype=mx.int32)
        logits = model(x)
        d = int(mx.argmax(logits[0, -1, :]).item())
        seq.append(d)
        digits.append(d)
    # Digits are LSD-first, convert to integer
    return int("".join(str(d) for d in reversed(digits)))