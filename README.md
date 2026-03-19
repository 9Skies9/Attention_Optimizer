## Motivation

[Attention Residuals](https://arxiv.org/abs/2603.15031) showed that replacing fixed residual connections with attention-based ones can improve performance.

- <img src="assets/residuals.png" width="200"/>

Andrej Karpathy's followed up with a thought whether stochastic gradient descent could also use attention in it:

- <img src="assets/kaparthy.png" width="400"/>

That made me look at Adam’s first-moment EMA differently: it compresses gradient history into a single exponentially decayed running average, much like a hidden state bottleneck in early sequential models.

So the question becomes: instead of forcing optimization history through one EMA, can an optimizer use attention to attend over recent gradients and decide what matters?

---

## AttnOpt: Attention as a First Moment

Adam's update rule uses an EMA of gradients as its first moment:

$$m_t = \beta_1 \ m_{t-1} + (1 - \beta_1) \, g_t$$

AttnOpt replaces that fixed decay with a learned, selective attention over a sliding window of the last $L$ gradients:

$$m_t=\sum_{i=0}^{L-1}\alpha_i\cdot g_{t-i}$$

$$q_t=x_tW_Q,\qquad k_{t-i}=x_{t-i}W_K,\qquad i\in{0,\dots,L-1}$$

$$\alpha=\mathrm{softmax}\!\left(\left[\frac{q_tk_t^\top}{\sqrt{d}},\frac{q_tk_{t-1}^\top}{\sqrt{d}},\dots,\frac{q_tk_{t-L+1}^\top}{\sqrt{d}}\right]\right)$$


## Testing

- Architecture:  Karpathy's nanoGPT architecture (with the improvements documented in [here](https://github.com/karpathy/nanochat/discussions/481))
- Pre-training dataset: HuggingFace's [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset

The goal is to see whether AttnOpt can match or beat Adam/AdamW/Muon on validation loss at a fixed token budget.

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
