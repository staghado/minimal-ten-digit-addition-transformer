"""
Fine-tune with L-BFGS (second-order optimizer).
Loads the AdamW-trained checkpoint and refines it using curvature information.

Usage:
  python finetune.py         # finetune current model (ff from train.py)
  python finetune.py --ff 6  # finetune the 173-param model
"""

import argparse
import random
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import Model, ModelArgs
from scipy.optimize import minimize

from train import make_batch, loss_fn, evaluate, INTERMEDIATE_SIZE

parser = argparse.ArgumentParser()
parser.add_argument("--ff", type=int, default=INTERMEDIATE_SIZE)
args = parser.parse_args()

model_args = ModelArgs(
    model_type="qwen3", hidden_size=3, num_hidden_layers=1,
    intermediate_size=args.ff, num_attention_heads=2, rms_norm_eps=1e-6,
    vocab_size=10, tie_word_embeddings=True, num_key_value_heads=1,
    max_position_embeddings=64, rope_theta=3, head_dim=4,
)

model = Model(model_args)
mx.eval(model.parameters())
template = tree_flatten(model.parameters())
param_shapes = [(name, p.shape) for name, p in template]
n_params = sum(p.size for _, p in template)

checkpoint = f"checkpoint/best_{n_params}.npz"
model.load_weights(list(mx.load(checkpoint).items()))
model.eval()
mx.eval(model.parameters())
template = tree_flatten(model.parameters())

sa, da = evaluate(model, 1000, random.Random(12345))
print(f"Loaded {checkpoint} ({n_params} params)")
print(f"Before: seq_acc={sa:.4f}  dig_acc={da:.4f}")


def vec_to_param_list(vec):
    params = []
    offset = 0
    for name, shape in param_shapes:
        size = int(np.prod(shape))
        params.append((name, mx.array(vec[offset:offset + size].reshape(shape))))
        offset += size
    return params


def loss_on_batch(vec, x, y):
    """MLX autodiff for gradients, returned as numpy for scipy."""
    model.load_weights(vec_to_param_list(vec))
    mx.eval(model.parameters())

    loss_val, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    mx.eval(loss_val, grads)

    grad_flat = np.concatenate([
        np.array(p).reshape(-1) for _, p in tree_flatten(grads)
    ]).astype(np.float64)

    return float(loss_val.item()), grad_flat


# Fixed dataset for L-BFGS: 8 batches x 256 = 2048 examples
rng = random.Random(42)
batches = [make_batch(256, rng) for _ in range(8)]

call_count = 0
best_loss = float('inf')
best_vec = None


def objective(vec):
    """Average loss+grad over all batches — fed to scipy L-BFGS-B."""
    global call_count, best_loss, best_vec
    call_count += 1

    total_loss = 0.0
    total_grad = np.zeros_like(vec)

    for x, y in batches:
        loss, g = loss_on_batch(vec, x, y)
        total_loss += loss
        total_grad += g

    avg_loss = total_loss / len(batches)
    avg_grad = total_grad / len(batches)

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_vec = vec.copy()

    if call_count % 10 == 0:
        print(f"  iter {call_count:4d}  loss {avg_loss:.6f}  best {best_loss:.6f}")

    return avg_loss, avg_grad


x0 = np.concatenate([
    np.array(p).reshape(-1) for _, p in template
]).astype(np.float64)

print(f"\nRunning L-BFGS ({len(batches)} batches of 256)...")
result = minimize(
    objective, x0, method='L-BFGS-B', jac=True,
    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': True},
)

print(f"\n{result.message}")
print(f"Final loss: {result.fun:.6f}  ({result.nfev} evals)")

model.load_weights(vec_to_param_list(best_vec))
mx.eval(model.parameters())
model.eval()

sa, da = evaluate(model, 1000, random.Random(12345))
print(f"After:  seq_acc={sa:.4f}  dig_acc={da:.4f}")

out = f"checkpoint/best_{n_params}_lbfgs.npz"
mx.savez(out, **dict(tree_flatten(model.parameters())))
print(f"Saved {out}")
