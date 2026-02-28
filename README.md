# minimal-ten-digit-addition-transformer

A **228-parameter** Qwen3 transformer that does 10-digit addition at **100% accuracy** on the [AdderBoard](https://github.com/anadim/AdderBoard) 10K test suite. Trained with plain AdamW — no tricks, no curriculum learning, no grokking.

## The only insight: use a tiny RoPE theta

RoPE with `head_dim=4` gives two frequencies: `1/θ^(0/4) = 1.0` and `1/θ^(2/4)=1/sqrt(θ)`. The full sequence is 35 tokens (24 input + 11 output). At θ=10000.0 the slow frequency is `1/100 = 0.01` — over 35 tokens it rotates `0.01 × 34 ≈ 0.34 rad`, so positions are nearly indistinguishable. At θ=3 it's `1/√3 ≈ 0.577`, rotating `0.577 × 34 ≈ 19.6 rad` — every position gets a unique, well-separated signature.

The choice of RoPE θ alone makes or breaks the training at this model param scale.

## Architecture

1-layer Qwen3 — `d=4, 2h/1kv, hd=4, ff=6, vocab=10, rope_theta=3`. Tied embeddings. 228 unique parameters.

## Training

```bash
python train.py
```

Trains for 35k steps with AdamW (lr=3e-3), batch size 128. Weights are saved to `checkpoint/best.npz`.

## Verification

```
$ python verify.py submission.py

Results: 10010/10010 correct (100.00%)
Time: 38.5s (260 additions/sec)
Status: QUALIFIED (threshold: 99%)
```

## Inference

```python
from submission import build_model, add

model, metadata = build_model()
result = add(model, 1234567890, 9876543210)
print(result)  # 11111111100
```

## How it works

Two integers (up to 10 digits each) are encoded digit-by-digit in **least-significant-digit-first** order, separated by zero tokens. The model autoregressively predicts the 11-digit sum in the same reversed format.

```
Input:  [0] d0 d1 ... d9 [0] [0] d0 d1 ... d9 [0]
Output: s0 s1 ... s10  (LSD-first)
```

## Citation

```bibtex
@misc{taghadouini2025tinyrope,
  author       = {Said Taghadouini},
  title        = {minimal-ten-digit-addition-transformer},
  year         = {2026},
  url          = {https://github.com/staghado/minimal-ten-digit-addition-transformer}
}
```
