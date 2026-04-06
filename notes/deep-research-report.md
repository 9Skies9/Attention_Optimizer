# AttnOpt Type 2: Learning \(W_Q\) and \(W_K\) for Attention Over Gradient History

## Executive Summary
AttnOpt Type 2 replaces “raw cosine attention” over gradient history with **learned projections** \(W_Q, W_K\) so that similarity is computed in a learned embedding space rather than the original gradient space. The provided repo’s README already formalizes both the **parameter-free baseline** (AttnRaw) and the **learned** version (AttnOpt) at a per-tensor/per-layer level, including the softmax attention and the convex mix with the current gradient. citeturn4view0turn5view1 The hard part is not computing attention; it is **training \(W_Q/W_K\) without instability or degeneracy** caused by circular dependence between (a) how \(W\) shapes the update and (b) how updates change future gradients. This circularity is naturally framed as **bilevel optimization / meta-learning**, like the “one-step validation” gradient used in DARTS and the differentiable inner-loop idea used in MAML. citeturn8view2turn3view3 The most practical path is to start with **low-parameter, low-memory** projection schemes (shared/compressed features, symmetric metrics, strong normalization) and use **first-order meta-gradients or proxy objectives** before attempting full second-order unrolling.

## Problem Setup and Mathematical Formulation
The repo defines a tensorwise (layerwise) “history mixing” view of Adam’s first moment: Adam uses an EMA
\[
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,
\]
while AttnOpt replaces fixed decay with a **selective attention** over the last \(L\) gradients for each layer/tensor \(\ell\). citeturn4view0turn5view1 The README writes the mixed first moment as
\[
\tilde m_t^{(\ell)} \;=\; \beta^{(\ell)} g_t^{(\ell)} \;+\; \bigl(1-\beta^{(\ell)}\bigr)\sum_{i=1}^{L-1}\alpha_i^{(\ell)} g_{t-i}^{(\ell)},
\quad
\alpha^{(\ell)}=\mathrm{softmax}\!\bigl([s_1^{(\ell)},\dots,s_{L-1}^{(\ell)}]\bigr).
\] citeturn4view0turn5view1

### AttnRaw Type 1 (parameter-free baseline)
AttnRaw uses raw cosine similarity scores over the flattened tensor gradients:
\[
s_i^{(\ell)} \;=\; \cos\!\bigl(g_t^{(\ell)},\, g_{t-i}^{(\ell)}\bigr),
\quad i\in\{1,\dots,L-1\}.
\] citeturn4view0turn5view1  
The repo’s implementation matches this: it normalizes dot-products by norms to form cosine scores, softmaxes them, and uses the weighted sum of history in a convex mixture with the current gradient. citeturn6view0turn5view1

### AttnOpt Type 2 (learned projections)
Type 2 replaces cosine with learned query/key projections:
\[
q_t^{(\ell)} = g_t^{(\ell)}W_Q^{(\ell)}, \qquad
k_{t-i}^{(\ell)} = g_{t-i}^{(\ell)}W_K^{(\ell)},
\]
\[
s_i^{(\ell)}=\frac{q_t^{(\ell)} \, {k_{t-i}^{(\ell)}}^\top}{\sqrt{d_\ell}},
\qquad
\alpha^{(\ell)}=\mathrm{softmax}\!\bigl([s_1^{(\ell)},\dots,s_{L-1}^{(\ell)}]\bigr).
\] citeturn4view0turn5view1  
Plugging scores into the same mixing rule yields the update direction \(\tilde m_t\); the rest of the optimizer can remain “Adam-like” (e.g., maintain a second moment \(v_t\) on \(\tilde m_t\) or on \(g_t\), as your experiment matrix distinguishes with “R” variants). citeturn5view1turn6view3

### A useful reinterpretation: learning a low-rank similarity metric
If \(g\in\mathbb{R}^{d_\ell}\) and \(W_Q,W_K\in\mathbb{R}^{d_\ell\times d_{\text{attn}}}\), then
\[
s_i \;=\; \frac{(g_t W_Q)\,(g_{t-i} W_K)^\top}{\sqrt{d_{\text{attn}}}}
\;=\; \frac{g_t \,A\, g_{t-i}^\top}{\sqrt{d_{\text{attn}}}},
\quad A := W_Q W_K^\top.
\]
So Type 2 is *metric learning* (a bilinear similarity) with a rank-\(\le d_{\text{attn}}\) factorization. This makes two design levers obvious: constrain \(A\) to avoid degeneracy, and keep parameter count feasible by making the effective dimensionality small.

## The Self-Referential Loop as Bilevel Optimization
Your “circular dependency” is real: \(W\) changes \(\tilde m_t\), \(\tilde m_t\) changes \(\theta\), and \(\theta\) changes future gradients that train \(W\). The clean way to see this is as a **bilevel** / nested optimization problem, like DARTS explicitly formulates (outer variable \(\alpha\), inner weights \(w\)): minimize a validation loss subject to weights optimized on training loss. citeturn8view2

### Bilevel formulation for AttnOpt Type 2
Let \(\theta\) be model weights and \(\phi\) collect optimizer-parameters (here \(\phi=\{W_Q,W_K,\beta,\tau,\dots\}\)). Define:
- Train-step update: \(\theta' = \theta - \eta\, U(\theta;\phi,\mathcal{H}_t)\), where \(U\) is the AttnOpt update computed from gradients and history \(\mathcal{H}_t\).
- Outer objective: validation loss after the update.

A one-step bilevel objective is:
\[
\min_{\phi}\; \mathbb{E}_{B_{\text{tr}},B_{\text{val}}}\Big[L_{\text{val}}\big(\theta - \eta\,U(\theta;\phi,\mathcal{H}_t(B_{\text{tr}}));\,B_{\text{val}}\big)\Big].
\]
This is structurally the same “differentiate through an update step” idea used in MAML (differentiate through one or more gradient steps) and DARTS (differentiate validation loss through a one-step inner update). citeturn3view3turn8view0

### Why “naively update \(W\) from the same gradient it processes” can be unstable
If you update \(\phi\) using the exact same signal that \(\phi\) just shaped, you’re effectively doing **simultaneous optimization on coupled dynamics**. Learned optimizer work emphasizes that learned update rules can improve outcomes but often incur compute/memory overhead and can generalize poorly without careful design—meaning stability and trade-offs are central, not incidental. citeturn3view6turn10view0

Practically, degeneracies you already anticipated are common:
- **Collapse to uniform attention**: \(W\to 0\) makes all scores similar \(\Rightarrow\) softmax ~ uniform \(\Rightarrow\) behaves like mean-history.
- **Hard one-hot attention**: large score magnitudes saturate softmax \(\Rightarrow\) brittle selection of a single past gradient.
- **Ignore-history solution**: learn to push \(\beta\to 1\) (or equivalent gating) so \(\tilde m_t \approx g_t\), reducing to baseline.

All three are *valid optima* unless your outer objective rewards real improvements vs the baselines under fair compute/time.

## Making Type 2 Feasible: Parameterizations That Avoid Huge \(W_Q/W_K\)
If you literally instantiate \(W_Q^{(\ell)}\in\mathbb{R}^{d_\ell\times d_{\text{attn}}}\) per tensor, the parameter count explodes: total parameters in \(W_Q,W_K\) are \(\approx 2d_{\text{attn}}\sum_{\ell} d_\ell\), i.e., \(2d_{\text{attn}}\) times (most of) the model parameter count. For an \(\sim 85\)M model and \(d_{\text{attn}}=32\), that’s \(\sim 5.4\)B learned parameters just for \(W\) (before optimizer states). This is why the repo currently focuses on Type 1 tensorwise cosine attention and excludes large/sparse embeddings from history-based methods due to memory cost. citeturn5view2turn6view0

Below are workable alternatives that preserve the “learned similarity space” idea without per-parameter full matrices.

### Shared projection on compressed gradient features (recommended starting point)
Define a **fixed compression** \(C_\ell:\mathbb{R}^{d_\ell}\to\mathbb{R}^{p}\) (with small \(p\), e.g. 64–512), then learn shared \(W_Q,W_K\in\mathbb{R}^{p\times d_{\text{attn}}}\):
\[
\bar g_t^{(\ell)} = C_\ell(g_t^{(\ell)})\in\mathbb{R}^p,\quad
q_t^{(\ell)}=\bar g_t^{(\ell)}W_Q,\quad
k_{t-i}^{(\ell)}=\bar g_{t-i}^{(\ell)}W_K.
\]
Compression \(C_\ell\) can be:
- random projection (seeded per tensor),
- sketching (CountSketch / hashing),
- pooled stats + a small fixed basis (RMS, mean, quantiles, sign counts),
- low-rank adapters that are fixed or lightly trained.

This reduces learnable parameters from \(O(d_\ell d_{\text{attn}})\) per tensor to \(O(p d_{\text{attn}})\) global. It also makes “shared vs per-layer” viable because dimensions match.

### Symmetric metric: set \(W_Q=W_K=W\)
Instead of learning two matrices, enforce:
\[
q_t = \bar g_t W,\quad k_{t-i}=\bar g_{t-i} W,\quad
s_i = \frac{\langle q_t, k_{t-i}\rangle}{\sqrt{d_{\text{attn}}}}.
\]
This is equivalent to learning a symmetric PSD similarity \(A=W W^\top\) (a Mahalanobis-like metric) in the compressed space. It halves parameters and removes an entire class of asymmetric degeneracies.

### Blockwise or channelwise \(W\)
If a tensor is a matrix \(G\in\mathbb{R}^{o\times i}\), you can compute per-row pooled features (size \(p\ll i\)) and apply a shared projection. This gives “per-row attention” without storing/querying full per-row history. Per-row attention skyrockets memory if you store full row gradients for \(L\) steps, so pooling is key.

### Concrete cost table
Assume:
- history length \(L\),
- tensor gradient dimension \(d_\ell\),
- compressed dimension \(p\),
- attention dim \(d_a\).

| Scheme | Learnable params | Stored history / tensor | Score compute / step / tensor |
|---|---:|---:|---:|
| Full \(W_Q,W_K\in\mathbb{R}^{d_\ell\times d_a}\) | \(2d_\ell d_a\) | \(L d_\ell\) | \(O(L\,d_a + L\,d_\ell)\) (projection + dot) |
| Shared on compressed \(\bar g\in\mathbb{R}^p\) | \(2 p d_a\) (global) | \(L p\) (or store raw + recompute) | \(O(L\,p d_a)\) |
| Symmetric \(W_Q=W_K=W\) | \(p d_a\) | \(L p\) | \(O(L\,p d_a)\) |
| No learned \(W\) (AttnRaw cosine) | 0 | \(L d_\ell\) | \(O(L\,d_\ell)\) |

For large models, “shared on compressed features” is the option that makes Type 2 testable without turning the optimizer into a second model.

## Training \(W_Q/W_K\): Algorithms, Signals, and Stability Controls
You proposed Option A (differentiable step, MAML/DARTS-like) and Option B (frozen meta-update). Both are valid, but they differ in what gradient signal they provide and what second-order terms they require.

### Option A: Differentiable optimizer step with train/val split
This is the cleanest meta-learning signal: update \(W\) based on whether the step it produced improves validation loss. It is the same computational pattern: “differentiate through one gradient step” as in MAML, and “outer val objective through inner update” as in DARTS bilevel optimization. citeturn3view3turn8view2

#### Second-order vs first-order meta-gradients
Let:
\[
\theta'(\phi)=\theta - \eta\,U(\theta;\phi),\quad
\mathcal{L}_{\text{meta}}(\phi)=L_{\text{val}}(\theta'(\phi)).
\]
Then:
\[
\nabla_\phi \mathcal{L}_{\text{meta}}
= \nabla_{\theta'}L_{\text{val}}(\theta') \cdot \frac{\partial \theta'}{\partial \phi}
= -\eta\, \nabla_{\theta'}L_{\text{val}}(\theta')\cdot \frac{\partial U}{\partial \phi}
\;-\;\eta\,\nabla_{\theta'}L_{\text{val}}(\theta')\cdot \frac{\partial U}{\partial \theta}\frac{\partial \theta}{\partial \phi}.
\]
The “hard” part is that \(U\) depends on \(\theta\) through \(g_t=\nabla_\theta L_{\text{tr}}(\theta)\), which introduces Hessian terms. DARTS explicitly distinguishes a “first-order approximation” where certain second-order derivatives disappear (their \(\xi=0\) setting), trading faithfulness for speed. citeturn8view0turn8view1

**Recommended practical compromise for AttnOpt Type 2:** compute \(g_t\) with `create_graph=False` and **detach it** when forming \(U\). Then \(U\) is treated as a function of \(\phi\) given observed gradients, so the meta-gradient avoids Hessian-through-training-gradient terms. This is analogous in spirit to first-order approximations used to keep bilevel training tractable. citeturn8view0turn11view2

#### Implementation scaffolding (PyTorch)
To implement differentiable steps, you typically need functional/stateless parameter updates and differentiable optimizers. Two widely used toolkits:
- `higher`: supports unrolled first-order optimization loops, tracking parameter changes and enabling gradients through intermediate parameters; it provides differentiable optimizers. citeturn3view1  
- TorchOpt: provides functional optimizers and differentiable optimization support, including explicit/implicit modes. citeturn3view2  

**Pseudocode (Option A, first-order meta-gradient; one-step):**
```python
# theta: model params
# phi: {W_Q, W_K, maybe beta, tau}  (optimizer params)
# Btr, Bval: train/val microbatches split from a larger batch

# 1) Train gradient (no second-order graph)
loss_tr = L(theta, Btr)
g_tr = grad(loss_tr, theta, create_graph=False)  # detach grads

# 2) Build history features and compute AttnOpt update U using phi
U = attn_update(g_tr, history, phi)  # differentiable in phi

# 3) Virtual differentiable step
theta_prime = theta - lr * U   # functional update (no in-place)

# 4) Validation loss and meta-gradient
loss_val = L(theta_prime, Bval)
dphi = grad(loss_val, phi)     # updates W_Q/W_K

# 5) Update phi with a separate optimizer (small LR)
phi = phi - lr_phi * dphi
```

#### Memory and compute notes
- You pay at least one extra forward/backward pass on \(B_{\text{val}}\) per meta-update step.
- If you include the full second-order path (i.e., allow gradients through \(g_t(\theta)\)), memory/compute rises sharply; learned optimizer work emphasizes these trade-offs and shows performance improvements often come with overhead. citeturn10view0turn3view6  
- For large LLMs, start with “first-order meta” (detach \(g_t\)) and meta-update \(W\) only periodically (every \(N\) steps).

### Option B: Frozen meta-updates (make it non-degenerate)
As written, “freeze \(\theta\), compute loss, backprop to \(W\)” yields zero gradient because \(W\) doesn’t enter the forward pass. To make Option B meaningful, it must include either:
1) a **virtual update** using \(W\) (and differentiate val loss through that update), or  
2) a **proxy objective** defined purely on gradients (next-gradient prediction, alignment with descent, etc.).

#### Option B1: Frozen-\(\theta\) + virtual step on val loss (recommended “Option B”)
This is essentially Option A’s first-order version, but done in periodic chunks with \(\theta\) held constant to reduce noise:
\[
\theta'=\theta - \eta U(g_{\text{tr}};\phi),\quad
\min_\phi L_{\text{val}}(\theta').
\]
Because \(\theta\) is frozen, you can meta-update \(\phi\) for \(M\) iterations on multiple \(B_{\text{tr}},B_{\text{val}}\) pairs before resuming normal training. This fits within the “approximate bilevel” perspective in bilevel programming work (solve an approximate version using finite optimization dynamics). citeturn11view0turn11view2

**Pseudocode (Frozen meta-loop):**
```python
# Every N training steps:
theta_snap = theta.detach_copy()

for j in range(M):
    Btr, Bval = sample_split_batch()
    g_tr = grad(L(theta_snap, Btr), theta_snap, create_graph=False)
    U = attn_update(g_tr, history_features(theta_snap), phi)
    theta_prime = theta_snap - lr * U
    loss_val = L(theta_prime, Bval)
    dphi = grad(loss_val, phi)
    phi = opt_phi.step(phi, dphi)  # e.g., Adam/Muon at tiny lr

# restore theta and continue regular training of theta
```

#### Option B2: Next-gradient prediction (self-supervised, no val split)
Define a target based on observed future gradients:
- Compute an attended mixture \(\tilde m_t(\phi)\) from \(\{g_t,\dots,g_{t-L+1}\}\).
- After applying your base update to get \(\theta_{t+1}\), observe \(g_{t+1}\).
- Train \(\phi\) to make \(\tilde m_t\) predict \(g_{t+1}\), e.g.
\[
\mathcal{L}_\phi
= 1 - \cos\!\bigl(\tilde m_t(\phi),\, g_{t+1}\bigr)
\quad\text{or}\quad
\|\mathrm{norm}(\tilde m_t(\phi))-\mathrm{norm}(g_{t+1})\|^2.
\]
This avoids differentiating through the step entirely, but it optimizes a proxy (“predict gradient”), not the actual objective (“reduce val loss”). Learned optimizers literature highlights that picking the objective matters and affects generalization and stability. citeturn3view6turn10view0

### Hypergradient-style learning for \(\beta\) and \(\tau\)
Instead of learning full \(W\), you can also learn a small set of coefficients online (mix \(\beta\), temperature \(\tau\), decay) using hypergradient descent. This is a proven strategy for tuning optimizer hyperparameters like learning rates, and recent work (e.g., MADA) uses hyper-gradients to adapt optimizer coefficients during training. citeturn3view8turn12view0

### Stability checklist (practical mitigations)
A concise set of best practices that map directly to your failure modes:

- Normalize inputs to attention:
  \[
  \bar g = \frac{g}{\|g\|+\epsilon}
  \quad\Rightarrow\quad
  s_i=\frac{(\bar g_tW_Q)(\bar g_{t-i}W_K)^\top}{\sqrt{d_a}}.
  \]
- Add score stabilization: subtract max score before softmax; clamp scores to \([-c,c]\).
- Use a residual path (you already do): \(\tilde m_t = \beta\,(\text{attn history}) + (1-\beta)g_t\). citeturn5view1turn6view0
- Two-timescale updates: \(\eta_\phi \ll \eta_\theta\), and update \(\phi\) every \(N\) steps.
- Stop-gradient choices:
  - Detach \(g_t\) when computing meta-gradients to avoid Hessians (first-order meta).
  - Optionally detach attention weights when updating \(\theta\) (reduce feedback loops).
- Clip \(\|\tilde m_t\|\) or clip parameter update magnitude; your training loop already clips gradients. citeturn5view2

## Experimental Design for Type 2 in Your Repo’s Setting
Your repo already defines the baseline matrix and the training budget, including fixed lookback \(L=8\) and a token-fixed comparison across optimizers. citeturn5view1turn6view3 That’s a good “outer shell” for fairness, but Type 2 adds extra compute for learning \(W\), so you’ll need additional fairness constraints on wall-time and meta-update frequency.

### Baselines and ablations to include (explicit table)
Use your existing baselines plus these Type-2-specific ablations:

| Method ID | Attention scores | Learnable params | Meta-signal | Notes |
|---|---|---:|---|---|
| ATTNRAW-8 | cosine in gradient space | 0 | none | already defined citeturn5view1turn6view3 |
| ATTNOPT-COMP | learned \(W\) on compressed features | \(O(p d_a)\) | val-step (first-order) | recommended starting point |
| ATTNOPT-SYM | \(W_Q=W_K\) | \(O(p d_a)\) | val-step (first-order) | better stability |
| ATTNOPT-NGP | learned \(W\) | \(O(p d_a)\) | next-gradient prediction | cheapest meta loop |
| ATTNOPT-FULL | per-layer full \(d_\ell\times d_a\) | \(O(d_\ell d_a)\) | val-step (2nd order) | likely infeasible at scale |

### Fairness checks
Keep at least these fairness constraints:

- **Compute-matched reporting:** report (a) loss vs step, and (b) loss vs wall-clock time / tokens-per-second (your training loop logs tokens/sec). citeturn5view2turn6view3  
- **Meta-overhead accounting:** if Type 2 uses extra forward/backward (val split), report effective token budget spent on training vs meta.
- **History budget parity:** if Type 2 uses \(L\) history, keep \(L\) equal across Avg, AttnRaw, and AttnOpt variants. citeturn5view1turn6view3
- **Embeddings excluded consistently:** since you exclude embeddings from history-based methods for memory reasons, keep that consistent for all history-based variants. citeturn5view2turn6view0

### Concrete experimental recipe (fits your repo structure)
Use your existing run matrix style and add runs like:
- \(L=8\) fixed.
- \(p\in\{64,128,256\}\), \(d_a\in\{8,16,32\}\).
- meta frequency \(N\in\{20,50,100\}\) steps and meta-iterations \(M\in\{1,5,10\}\).
- \(\eta_\phi \in \{10^{-5}, 3\cdot 10^{-5},10^{-4}\}\) with weight decay \(10^{-3}\).
- keep \(\eta_\theta\) and all other training settings identical to ATTNRAW-8. citeturn6view3turn5view2

## Implementation Roadmap and Mermaid Diagrams

### Minimal viable Type 2 (what to build first)
Build “ATTNOPT-COMP-SYM”:
1) Compress each tensor gradient to \(p\) features.
2) Use a single shared \(W\in\mathbb{R}^{p\times d_a}\) (symmetric metric).
3) Train \(W\) by first-order val-step meta-gradient every \(N\) steps (detach train grads).

This tests whether “learned similarity” beats cosine without the infeasible parameter explosion.

### Publishable Type 2
Add:
- learned \(\beta\) (or gate) updated via hypergradients,
- controlled temperature schedule,
- ablations showing where gains come from.

Hypergradient methods provide precedent for learning optimizer hyperparameters online (learning rates, coefficients) without a huge outer loop. citeturn3view8turn12view0

### Ambitious Type 2
Only after the above works:
- try second-order bilevel updates (include Hessian terms) using unrolling or implicit differentiation approximations,
- evaluate on multiple tasks to measure generalization (learned optimizers often don’t generalize “for free”). citeturn3view6turn11view1

### Mermaid architecture diagram
```mermaid
flowchart LR
  Gt[Tensor gradient g_t^ℓ] --> C[Compress C_ℓ or normalize]
  H[History {g_{t-1}^ℓ..g_{t-L+1}^ℓ}] --> C2[Compress/normalize history]
  C --> Q[q_t = ḡ_t W_Q]
  C2 --> K[k_i = ḡ_{t-i} W_K]
  Q --> S[s_i = q_t k_i^T / sqrt(d_attn)]
  K --> S
  S --> A[alpha = softmax(s)]
  A --> M[m_hist = sum_i alpha_i g_{t-i}]
  Gt --> MIX[m̃ = (1-β) g_t + β m_hist]
  M --> MIX
  MIX --> UPD[θ ← θ - η * precondition(m̃)]
```

### Mermaid training timeline diagram
```mermaid
flowchart TB
  T0[Normal training steps on θ] -->|every N steps| META[Meta-update W_Q/W_K]
  META -->|Option A| A1[Split batch: train/val]
  A1 --> A2[Compute g_train; build m̃(W)]
  A2 --> A3[Virtual θ' = θ - η m̃]
  A3 --> A4[Compute loss_val(θ'); backprop to W]
  META -->|Option B2| B2[Next-gradient prediction loss on W]
  META --> T0
```

## Recommended Next Experiments
Start with the smallest experiment that answers the central question: “can learned projections beat cosine attention under fair overhead?”

- Implement **ATTNOPT-COMP-SYM** with \(p=128\), \(d_a=16\), \(L=8\), meta every \(N=50\) steps, \(M=1\), and detach \(g_{\text{train}}\) for first-order meta-gradients.
- Compare against ATTNRAW-8 and AVG-8, and include an ablation where \(W\) is random and frozen to separate “projection helps” from “learning helps.” citeturn5view1turn6view3
- Produce three required plots: (i) val loss vs steps, (ii) val loss vs wall-clock time / tokens/sec, (iii) ablation table (random \(W\), symmetric vs asymmetric, different \(p,d_a\)).

If you see any win at all (even small), then invest in Option A second-order or implicit-gradient variants, leveraging bilevel optimization tools (iterative vs implicit differentiation) surveyed in the bilevel/hyperparameter optimization literature. citeturn11view2turn11view1turn11view0