# AttnOpt Development Log

Running log of all implementations, experiments, and findings. Updated as new runs complete.

---

## Motivation

Inspired by [Attention Residuals](https://arxiv.org/abs/2603.15031) and Karpathy's thought on whether SGD could use attention, the core question is:

> Instead of forcing optimization history through one exponentially decayed running average (Adam's EMA), can an optimizer use attention to selectively attend over recent gradients and decide what matters?

Adam's first moment is:
```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
```
This compresses all gradient history into a single exponentially decaying scalar — an information bottleneck. AttnOpt replaces this with a selective, history-aware mixture.

---

## Model & Training Setup

- **Architecture**: nanoGPT with RMSNorm, RoPE, QK-Norm, ReLU², FlashAttention, embedding skip connections. No biases anywhere. ~85M parameters.
- **Dataset**: HuggingFace FineWeb, 10 shards (~1B tokens)
- **Budget**: 4,096 steps × 262,144 tokens/step ≈ 1.07B tokens per run
- **Batch**: micro_batch=16, grad_accum=16, seq_len=1024
- **LR**: cosine decay, warmup 200 steps, min_lr = 0.1 × max_lr
- **Seed**: 42 (fixed across all runs for reproducibility)
- **Embedding table**: excluded from all history-based methods (too sparse, too memory-costly). Routed to AdamW instead.

---

## Experiment 1 — Baselines (2026-03-30)

### Runs: BASE-SGD, BASE-ADAM, BASE-MUON

Establish the performance floor and ceiling before testing any history-based methods.

**Note on Muon**: BASE-MUON was run with all parameters passed to Muon, including the embedding table. This is incorrect — Muon is designed for weight matrices only and degrades on sparse/embedding parameters. Will be rerun as BASE-MUON (fixed) with proper embedding split.

| Run | Final Loss |
|---|---|
| BASE-SGD | 6.351 |
| BASE-ADAM | 3.757 |
| BASE-MUON* | 4.142 |

*Muon result is unfair due to missing embedding split.

**Takeaway**: Adam is the target to beat at 3.757.

---

## Experiment 2 — Uniform History Average vs Attention (2026-03-30)

### Runs: AVG-8, AVG-8R, ATTNRAW-8, ATTNRAW-8R

The first real test. Two methods, two second-moment variants each.

### AVG-8 / AVG-8R — Sliding Average Baseline

Replaces Adam's EMA first moment with a uniform average of the last 8 gradients mixed with the current gradient:

```
m_tilde = (1 - mix_beta) * g_t + mix_beta * mean(g_{t-1}, ..., g_{t-8})
```

- `mix_beta = 0.9` (90% past average, 10% current)
- **AVG-8**: second moment tracks `m_tilde²`
- **AVG-8R**: second moment tracks raw `g_t²` (decoupled from the smoothed first moment)

### ATTNRAW-8 / ATTNRAW-8R — Parameter-Free Cosine Attention

Replaces the uniform average with cosine-similarity attention. Each parameter tensor attends over its own gradient history — one attention distribution per tensor:

```
scores_i = cos(g_t, g_{t-i})
alpha = softmax(scores)
m_tilde = (1 - mix_beta) * g_t + mix_beta * (alpha @ history)
```

Zero learned parameters. Cosine similarity decides which past gradients are most relevant to the current one.

**Design note**: Per-tensor (not per-scalar) attention was chosen deliberately. Per-row attention would be more expressive but memory-prohibitive at 85M params.

**Second moment variants**: The `R` suffix means `raw_second_moment=True` — the second moment EMA tracks `g_t²` instead of `m_tilde²`. This tests whether decoupling the variance estimate from the history mixture helps.

| Run | Final Loss | Notes |
|---|---|---|
| AVG-8 | 4.188 | Uniform average, m̃² variance |
| AVG-8R | 3.957 | Uniform average, raw g_t² variance |
| ATTNRAW-8 | 3.791 | Cosine attention, m̃² variance |
| ATTNRAW-8R | 3.855 | Cosine attention, raw g_t² variance |

**Key findings**:
1. **Attention beats uniform average decisively** — ATTNRAW-8 (3.791) vs AVG-8 (4.188). The selective weighting matters; not all past gradients are equally useful.
2. **ATTNRAW-8 nearly matches Adam** — gap of only 0.034. Parameter-free cosine attention in raw gradient space is surprisingly competitive.
3. **Raw second moment helps AVG but hurts ATTNRAW** — for the uniform average, decoupling variance estimation from the noisy mixture helps. For attention, the smoothed m_tilde² is a better variance signal.
4. **Uniform average is significantly worse** — confirms the hypothesis that selective attention over gradient history is meaningfully different from just averaging.

---

## Experiment 3 — EMA on Top of Attention (2026-03-30)

### Runs: ATTNRAW-V2-8, ATTNRAW-V3-8

ATTNRAW-8 replaces Adam's EMA entirely with a one-shot attention mixture. A natural extension is to keep EMA but feed it a smarter instantaneous signal:

**V2 — g_t inside the window, EMA on top**:
```
m_tilde = cosine-attention([g_t, g_{t-1}, ..., g_{t-7}])  # g_t included
m_t = beta1 * m_{t-1} + (1 - beta1) * m_tilde
v_t = beta2 * v_{t-1} + (1 - beta2) * m_tilde²
```
`mix_beta` disappears — `beta1` does the long-term smoothing. Attention can decide how much weight to give the current gradient vs history.

**V3 — past-only attention + residual + EMA**:
```
a_t = cosine-attention(query=g_t, values=[g_{t-1}, ..., g_{t-8}])  # g_t is query only
u_t = (1 - mix_beta) * g_t + mix_beta * a_t
m_t = beta1 * m_{t-1} + (1 - beta1) * u_t
v_t = beta2 * v_{t-1} + (1 - beta2) * u_t²
```
Hybrid of V1's mixing formula with EMA stacked on top.

| Run | Final Loss | Notes |
|---|---|---|
| ATTNRAW-V2-8 | 4.669 | Worse than V1 |
| ATTNRAW-V3-8 | 5.081 | Significantly worse |

**Post-run analysis — V3 bug**: With `mix_beta=0.9`, the current gradient's net contribution to `m_t` is `(1 - 0.9) * (1 - 0.9) = 0.01` — nearly nothing. The EMA double-dilutes the current gradient. **Fix**: set `mix_beta=0.1` so `u_t = 0.9*g_t + 0.1*a_t`, keeping g_t's net contribution at ~0.09 (close to Adam's 0.1).

**Why V2 underperforms**: Including `g_t` inside the attention window means the query and one key are identical — the model attends maximally to itself at every step, partially collapsing to a weighted average dominated by the current gradient. The EMA then further dilutes the attention signal.

**Status**: V3 will be rerun with corrected `mix_beta=0.1`. V2 may need architectural rethinking.

---

## Cumulative Results (2026-03-30)

| Run | Final Loss | vs Adam |
|---|---|---|
| BASE-ADAM | 3.757 | — |
| ATTNRAW-8 | 3.791 | +0.034 |
| ATTNRAW-8R | 3.855 | +0.098 |
| AVG-8R | 3.957 | +0.200 |
| BASE-MUON* | 4.142 | +0.385 |
| AVG-8 | 4.188 | +0.431 |
| ATTNRAW-V2-8 | 4.669 | +0.912 |
| ATTNRAW-V3-8 | 5.081 | +1.324 |
| BASE-SGD | 6.351 | +2.594 |

---

## Pending / Upcoming Runs

| Run | Description | Status |
|---|---|---|
| BASE-MUON (fixed) | Muon with proper embedding split → AdamW | Queued |
| BASE-SGD (rerun) | Clean rerun with seed 42 | Queued |
| ATTNRAW-V3-8 (fixed) | V3 with mix_beta=0.1 | Queued |
| ATTNRAW-V2-8 (debug) | Investigate self-attention collapse | Pending |
| ATTNOPT-B-8 | Learned W_Q/W_K, next-gradient prediction | Not started |
| ATTNOPT-A-8 | Learned W_Q/W_K, differentiable val step | Not started |

---

## Design Notes & Open Questions

### Why parameter-free attention works at all
Cosine similarity in raw gradient space is a reasonable proxy for "is this past gradient pointing in a similar direction to the current one?" Steps where the gradient pointed similarly are likely in the same loss landscape region and should be upweighted. The softmax sharpens this into a selective retrieval.

### Why uniform average is worse
Averaging gradients from 8 different model states implicitly mixes signal from different parts of the loss landscape. If the model has moved significantly, early gradients in the window may be pointing in misleading directions. Attention downweights those; uniform average doesn't.

### The EMA double-dilution problem (V2/V3)
When stacking attention + EMA, the current gradient contribution compounds: if attention gives g_t weight `(1-β_attn)` and EMA gives the mixture weight `(1-β₁)`, then g_t's net contribution is `(1-β_attn)(1-β₁)`. At both = 0.9, this is 0.01 — too small. Future EMA-on-top designs need to account for this.

### Learned projections (AttnOptA/B)
The self-referential problem with learned W_Q/W_K: the projections shape which gradients get upweighted, those gradients update θ, and new gradients from updated θ are used to train W_Q/W_K. Two approaches being implemented:
- **AttnOptB**: next-gradient prediction — train W_Q/W_K to make m_tilde align with g_{t+1} (self-supervised, no val split)
- **AttnOptA**: differentiable val step — backprop val loss through a virtual optimizer step to get gradient on W_Q/W_K (MAML/DARTS-style)

See `notes/type2_design.md` for full design discussion.
