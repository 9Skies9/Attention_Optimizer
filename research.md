Below is the revised version, changed to reflect:

* only **nanoGPT 44M** for now
* full sweeps over **history length** (L \in {4,8,16})
* full sweeps over **temperature** with one value (<1), one at (=1), and one (>1)
* **AttnRaw-v1/v2/v3** and **AttnPrec-v1/v2/v3**
* **AttnPrec uses (v_{t-1}) for the current gradient and (v_{t-i}) for historical gradients**
* a note that future work may explore **projected attention scores with learned (W_Q, W_K)** / meta-learned query-key projections. 

---

# AttnPrec: Metric-Aware Cosine Attention over Gradient History

## Scope and Constraints

**Paper Type:** Short paper
**Target Venues:** NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
**Contribution Type:** Method (with diagnostic analysis)

## Introduction

### Context and Motivation

Modern large-scale model training is highly sensitive to the choice of optimizer and its hyperparameters. Despite many variants, Adam/AdamW remain default choices for training transformer language models because they provide good convergence and stability with minimal overhead.

A common design pattern in these optimizers is to compress optimization history into a small state per parameter, typically exponential moving averages (EMAs) of the first and second moments of the gradient. This compression is computationally convenient, but it can be an information bottleneck: when gradient directions change non-stationarily (e.g., across warmup, curriculum effects, data-mixture shifts, or phase transitions during pretraining), a fixed EMA may lag or over-smooth.

Recent work has successfully replaced other fixed “history mixing” rules in neural networks with content-dependent attention. This raises the natural question of whether optimization history itself can be mixed in a content-dependent way, rather than through a fixed exponential decay.

### The Problem

Existing attention-over-history optimizer ideas typically focus on replacing only AdamW’s first-moment EMA with an attention-weighted mixture of recent gradients. A strong parameter-free baseline is **AttnRaw**, which computes attention scores using cosine similarity between the current gradient and each past gradient in raw space.

However, this leaves several open questions unresolved:

1. Is raw cosine similarity the right metric for retrieving useful gradient history?
2. Is attention over history only useful as a replacement for the first moment, or can it also replace the second-moment recurrence?
3. Is optimizer state such as (m_{t-1}) and (v_{t-1}) fundamentally necessary, or can explicit attention over a short history window replace part or all of that implicit memory?

These questions are especially relevant because gradients in transformer pretraining are often heteroskedastic and heavy-tailed. Raw cosine similarity may therefore be a noisy measure of directional agreement, while the second-moment recurrence may itself be replaceable if explicit history retrieval is sufficiently informative.

### Key Insight and Hypothesis

Key insight: cosine similarity in raw gradient space can be a low signal-to-noise measure of directional agreement because gradients are often heteroskedastic (coordinate-wise variance differs widely) and heavy-tailed (occasional very large coordinates). A small subset of high-variance coordinates can therefore dominate dot products and cosine scores, making the resulting attention weights noisy or high-entropy.

Hypothesis: if we compute attention weights in a preconditioned gradient space, where each gradient is normalized by its corresponding second-moment statistics, then the attention distribution will become more informative and the retrieved gradient mixture will improve optimization.

A second hypothesis is structural: explicit attention over recent gradients may replace not only Adam-style first-moment state, but also some or all of the second-moment recurrence. This motivates testing progressively less stateful variants.

This could still fail for several reasons:

* preconditioned similarity may introduce a metric-mismatch problem across time;
* explicit history retrieval may not outperform a well-tuned EMA;
* removing (m_{t-1}) and (v_{t-1}) may destabilize training even if the attention weights are informative.

## Proposed Approach

### Overview

We propose two attention-based optimizer families that operate independently per tensor / layer:

* **AttnRaw**: attention scores computed in raw gradient space
* **AttnPrec**: attention scores computed in preconditioned gradient space

For each family, we define three variants:

* **v1**: keep both (m_{t-1}) and (v_{t-1})
* **v2**: remove (m_{t-1}), keep (v_{t-1})
* **v3**: remove both (m_{t-1}) and (v_{t-1})

This gives a controlled way to test both the role of similarity geometry and the role of retained optimizer state.

### Method Details

Let (g_t^{(\ell)}) denote the current gradient tensor for layer (\ell). Maintain a history buffer
[
{g_t^{(\ell)}, g_{t-1}^{(\ell)}, \dots, g_{t-(L-1)}^{(\ell)}}.
]

For **AttnRaw**, define scores
[
s_i^{(\ell)}=\cos!\big(g_t^{(\ell)},g_{t-i}^{(\ell)}\big), \qquad i=1,\dots,L-1.
]

For **AttnPrec**, define preconditioned gradients using causal second-moment statistics:
[
\tilde g_t^{(\ell)}=\frac{g_t^{(\ell)}}{\sqrt{v_{t-1}^{(\ell)}}+\epsilon},
\qquad
\tilde g_{t-i}^{(\ell)}=\frac{g_{t-i}^{(\ell)}}{\sqrt{v_{t-i}^{(\ell)}}+\epsilon}.
]

Then define scores
[
s_i^{(\ell)}=\cos!\big(\tilde g_t^{(\ell)},\tilde g_{t-i}^{(\ell)}\big).
]

Convert scores to attention weights using temperature (\tau):
[
\alpha_i^{(\ell)}=
\frac{\exp(s_i^{(\ell)}/\tau)}
{\sum_{j=1}^{L-1}\exp(s_j^{(\ell)}/\tau)}.
]

Define the attention-weighted gradient mixture
[
\bar g_t^{(\ell)}
=================

\beta^{(\ell)} g_t^{(\ell)}
+
\big(1-\beta^{(\ell)}\big)\sum_{i=1}^{L-1}\alpha_i^{(\ell)}g_{t-i}^{(\ell)}.
]

#### Variant v1: keep (m_{t-1}) and (v_{t-1})

For both AttnRaw-v1 and AttnPrec-v1,
[
m_t^{(\ell)}
============

\beta_1 m_{t-1}^{(\ell)}+(1-\beta_1)\bar g_t^{(\ell)},
]
[
v_t^{(\ell)}
============

\beta_2 v_{t-1}^{(\ell)}+(1-\beta_2)\big(\bar g_t^{(\ell)}\big)^2.
]

#### Variant v2: remove (m_{t-1}), keep (v_{t-1})

For both AttnRaw-v2 and AttnPrec-v2,
[
m_t^{(\ell)}=\bar g_t^{(\ell)},
]
[
v_t^{(\ell)}
============

\beta_2 v_{t-1}^{(\ell)}+(1-\beta_2)\big(m_t^{(\ell)}\big)^2.
]

#### Variant v3: remove both (m_{t-1}) and (v_{t-1})

For both AttnRaw-v3 and AttnPrec-v3,
[
m_t^{(\ell)}=\bar g_t^{(\ell)},
]
[
v_t^{(\ell)}
============

\beta^{(\ell)}\big(g_t^{(\ell)}\big)^2
+
\big(1-\beta^{(\ell)}\big)\sum_{i=1}^{L-1}\alpha_i^{(\ell)}\big(g_{t-i}^{(\ell)}\big)^2.
]

All variants update parameters using RMS-style normalization:
[
\theta_{t+1}^{(\ell)}
=====================

\theta_t^{(\ell)}
-\eta
\frac{m_t^{(\ell)}}{\sqrt{v_t^{(\ell)}}+\epsilon}.
]

### Key Innovations

1. **Attention over gradient history instead of fixed EMA compression**
   The optimizer retrieves history using similarity-based attention rather than a predetermined exponential kernel.

2. **Metric-aware similarity for retrieval**
   AttnPrec compares gradients in preconditioned space rather than raw space.

3. **State-removal study through v1/v2/v3**
   The proposal tests whether attention can replace part or all of Adam-style state.

4. **Future extension toward projected similarity / meta-learned attention**
   A natural follow-up is to replace direct cosine similarity with projected query-key similarity, e.g.
   [
   q_t = W_Q \phi(g_t), \qquad k_{t-i}=W_K \phi(g_{t-i}),
   ]
   where (W_Q, W_K) may be learned or meta-learned. This would allow the optimizer to learn a more task-relevant similarity geometry than raw or diagonally preconditioned cosine alone. This is not part of the main method in the current paper, but it is explicitly identified as a future research direction.

## Experiments

### Experimental Setup

Primary setting:

* **Codebase:** nanoGPT-style training harness
* **Task:** decoder-only language model pretraining
* **Model:** **nanoGPT 44M only** for the present study
* **Training budget:** fixed token budget per run
* **Methods compared:**

  * AdamW baseline
  * AttnRaw-v1 / v2 / v3
  * AttnPrec-v1 / v2 / v3

### Sweep Design

For all attention-based variants, sweep both history length and temperature.

#### History-length sweep

[
L \in {4,8,16}
]

#### Temperature sweep

Use one temperature below 1, one equal to 1, and one above 1:
[
\tau \in {0.5,1.0,2.0}.
]

This yields the full grid
[
(L,\tau) \in {4,8,16}\times{0.5,1.0,2.0}.
]

This sweep is run for each of:

* AttnRaw-v1
* AttnRaw-v2
* AttnRaw-v3
* AttnPrec-v1
* AttnPrec-v2
* AttnPrec-v3

The purpose of the (L) sweep is to test how much explicit history is useful. The purpose of the (\tau) sweep is to test the sharpness of the retrieval distribution, ranging from sharper-than-cosine default ((\tau<1)) to flatter averaging ((\tau>1)).

### Benchmarks and Metrics

Benchmark:

* FineWeb-style pretraining subset or equivalent web-scale text subset used in the project

Primary metric:

* validation loss / perplexity at fixed token budget

Secondary metrics:

* step time
* tokens/sec
* peak GPU memory
* attention entropy
* lag-wise attention mass
* next-gradient predictiveness
* update RMS

### Experimental Rigor

Run all main comparisons with at least 3 seeds and report mean ± standard deviation.

Compute a minimum detectable difference estimate for the primary metric after collecting seed statistics, and only claim wins when the mean gap exceeds the noise band.

Keep training budget fixed across methods so that results reflect optimizer quality rather than extra token exposure.

## Analysis

The main diagnostic analyses directly test the mechanism claims:

* **Attention entropy:** does AttnPrec produce lower-entropy, more selective attention than AttnRaw?
* **Attention mass by lag:** does the method collapse to recency or use deeper history meaningfully?
* **Predictiveness:** does (m_t) better align with (g_{t+1}) than AdamW momentum?
* **Overhead:** do gains persist after accounting for compute and memory cost?

## Success Criteria

The method is successful if one or more AttnPrec variants outperform both the AdamW baseline and the corresponding AttnRaw variants by a statistically meaningful margin at fixed token budget, while keeping compute overhead reasonable.

More specifically:

* **Geometry question:** AttnPrec should outperform AttnRaw if metric-aware similarity helps retrieval.
* **State question:** v2 or v3 should remain competitive with v1 if explicit history can replace part of Adam-style state.
* **Practicality question:** improvements should not come with prohibitive memory or wall-clock overhead.

## Impact Statement

If successful, this work would show that optimizer state need not be restricted to fixed exponential recurrences. Instead, a short explicit gradient history combined with attention-based retrieval may provide a more adaptive and interpretable alternative. If preconditioned similarity proves effective, it would further suggest that optimizer geometry is useful not only for scaling updates, but also for deciding which parts of optimization history are worth retrieving. A successful result would also motivate future work on learned query-key projections and meta-learned similarity functions for optimization history. 

If you want, I can next compress this into a more polished conference-style abstract and introduction pair.

---

## Additional Part: Uniform-Average Baseline

We introduce a non-attention control baseline that replaces attention weights with a uniform average over the same history window.

This baseline uses:

\[
\alpha_i^{(\ell)} = \frac{1}{L-1}, \quad i = 1, \dots, L-1
\]

and defines:

\[
\bar g_t^{(\ell)} =
\beta^{(\ell)} g_t^{(\ell)} +
(1 - \beta^{(\ell)}) \frac{1}{L-1} \sum_{i=1}^{L-1} g_{t-i}^{(\ell)}.
\]

This produces a pure sliding-window average of gradients, without any similarity-based weighting.

We define three variants for this baseline, mirroring AttnRaw and AttnPrec:

---

### Avg-v1 (EMA + uniform history)

\[
m_t^{(\ell)} =
\beta_1 m_{t-1}^{(\ell)} +
(1 - \beta_1)\bar g_t^{(\ell)}
\]

\[
v_t^{(\ell)} =
\beta_2 v_{t-1}^{(\ell)} +
(1 - \beta_2)\left(\bar g_t^{(\ell)}\right)^2
\]

---

### Avg-v2 (no first-moment EMA)

\[
m_t^{(\ell)} = \bar g_t^{(\ell)}
\]

\[
v_t^{(\ell)} =
\beta_2 v_{t-1}^{(\ell)} +
(1 - \beta_2)\left(m_t^{(\ell)}\right)^2
\]

---

### Avg-v3 (fully stateless)

\[
m_t^{(\ell)} = \bar g_t^{(\ell)}
\]

\[
v_t^{(\ell)} =
\beta^{(\ell)} \left(g_t^{(\ell)}\right)^2 +
(1 - \beta^{(\ell)}) \frac{1}{L-1} \sum_{i=1}^{L-1} \left(g_{t-i}^{(\ell)}\right)^2
\]