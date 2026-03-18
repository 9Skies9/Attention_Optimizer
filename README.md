## Motivation

Two things collided to start this project.

First, the Kimi team published [**Attention Residuals**](https://arxiv.org/abs/2603.15031) (arXiv 2603.15031), showing that replacing fixed residual connections with attention-based aggregation improves both training stability and downstream performance at scale.

Second, Andrej Karpathy asked whether **stochastic gradient descent could be replicated by attention**:

![Karpathy tweet](assets/kaparthy.png)

When looking at Adam more carefully, I thought the **first moment EMA** is structurally identical to the bottleneck in a sequential modeling network. It compresses the entire gradient history into a single exponentially decayed running average, kinda like a hidden state of sequential networks.

So, instead of forcing optimization history through a single EMA bottleneck, can we let the optimizer use attention to attend over recent gradient history and decide what matters?

---

## AttnOpt: Attention as a First Moment

### The Idea

Adam's update rule uses an EMA of gradients as its first moment:

$$m_t = \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t$$

This is a fixed exponential decay — every past gradient contributes, weighted only by how long ago it arrived. AttnOpt replaces this with an attention-weighted sum over a sliding window of recent gradients, letting the optimizer *selectively* decide which past steps are informative for the current update.

### Mechanism

```
Step t gradient:  g_t  →  ĝ_t = g_t / RMS(g_t)        (RMS normalize)

Gradient stats:   s_t = [mean(ĝ_t),  E[ĝ_t²],  E[|ĝ_t|]]  ∈ ℝ³

Key/Query input:  x_i = [ s_i ‖ pos_i ]   ∈ ℝ^(3 + d_pos)

Attention:        q = x_t W_Q,   K = [x_t; x_{t-1}; … ; x_{t-K+1}] W_K

Weights:          α = softmax( q Kᵀ / √d_head )   ∈ ℝ^K

Attended moment:  m_attn = Σᵢ αᵢ ĝ_{t-i}
```

The second moment (variance estimate) from Adam is kept unchanged.

### Update Rules

**Pure** — attention fully replaces the EMA:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{m_{\text{attn}}}{\sqrt{\hat{v}_t} + \varepsilon}$$

**Gated** — attention blends with the EMA:

$$\tilde{m} = (1 - \lambda)\, m_{\text{EMA}} + \lambda\, m_{\text{attn}}$$

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\tilde{m}}{\sqrt{\hat{v}_t} + \varepsilon}$$

### Flow

```
g_t ──► RMS-norm ──► stats s_t ──┐
                                  ├──► Q, K projections
g_{t-1} … g_{t-K} ── stats ──────┘
                                  │
                           softmax attention
                                  │
                            α-weighted sum
                                  │
                            m_attn (first moment)
                                  │
                    ┌─────────────┴─────────────┐
                 pure mode                  gated mode
                 m̃ = m_attn           m̃ = (1-λ)m_EMA + λ m_attn
                                  │
                         θ ← θ - lr · m̃ / (√v̂ + ε)
```

---

## Test Bed

The model under test is **Karpathy's nanoGPT** (GPT-2), extended with the incremental architecture and training improvements documented in the [nanoGPT community discussion #481](https://github.com/karpathy/nanochat/discussions/481). Pre-training runs on HuggingFace's **[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)** dataset.

The goal is to see whether AttnOpt can match or beat Adam/AdamW/Muon on validation loss at a fixed token budget.

---

## Run Matrix

Training budget: ~`1.07B` tokens per run (`4,096` steps × `262,144` tokens/step).

| ID | Optimizer |
|---|---|
| `BASE-SGD` | SGD + momentum |
| `BASE-ADAM` | Adam |
| `BASE-ADAMW` | AdamW |
| `BASE-MUON` | Muon |
| `ATTN-PURE-8-TRAIN` | attention replaces EMA, context window 8 |
| `ATTN-PURE-16-TRAIN` | attention replaces EMA, context window 16 |
| `ATTN-GATED-8-TRAIN` | `0.5 × EMA + 0.5 × attention`, context window 8 |
| `ATTN-GATED-16-TRAIN` | `0.5 × EMA + 0.5 × attention`, context window 16 |

---

## Results

Live runs tracked on Weights & Biases (`attn-optimizer`). Loss curves compared in `analysis/results.ipynb`.
