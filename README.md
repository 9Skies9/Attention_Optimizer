# Attention Optimizer

## Motivation

Two things collided to start this project.

First, Kimi's recent paper on **attention residues** — the idea that attention heads form persistent, structured residual signals across layers — got me thinking about how selective memory over sequences is a more general computational primitive than it often gets credit for.

Second, Andrej Karpathy asked (half seriously) whether **stochastic gradient descent could be replicated by attention** — whether the iterative, history-aware nature of gradient-based optimization could be expressed as some form of learned sequence modeling over gradient updates.

That question stuck. And when I looked at Adam more carefully, something jumped out: the **first moment (EMA)** is structurally a lot like a bottleneck in a sequential modeling network. It compresses the entire gradient history into a single exponentially decayed running average — a fixed, non-selective summary. That is exactly the kind of bottleneck that attention was designed to get around in sequence modeling.

So the question became: **what if we replaced Adam's EMA with attention over the current gradient and a short window of past gradients?** Instead of a fixed exponential decay blending everything equally (weighted by recency), attention would selectively weight which past gradient steps are most relevant to the current one — the same way sequence models use attention to selectively pull from context rather than compressing it all into a hidden state.

This repo runs that experiment.

---

## Core Question

Can **attention over recent gradient history** replace or augment Adam's first-moment EMA on GPT pretraining, and if so, does selective gradient aggregation outperform fixed exponential averaging?

---

## Test Bed

The model under test is **Karpathy's nanoGPT** (GPT-2), extended with the incremental architecture and training improvements documented in the [nanoGPT community discussion #481](https://github.com/karpathy/nanochat/discussions/481). Pre-training runs on HuggingFace's **[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)** dataset.

The goal is to see whether AttnOpt can match or beat Adam/AdamW/Muon on validation loss at a fixed token budget.

---

## Run Matrix

The current default training budget is ~`1.07B` tokens per run (`4,096` steps at `262,144` tokens/step).

| ID | Optimizer |
|---|---|
| `BASE-SGD` | SGD + momentum |
| `BASE-ADAM` | Adam |
| `BASE-ADAMW` | AdamW |
| `BASE-MUON` | Muon |
| `ATTN-PURE-8-TRAIN` | attention replaces EMA, context 8 |
| `ATTN-PURE-16-TRAIN` | attention replaces EMA, context 16 |
| `ATTN-GATED-8-TRAIN` | `0.5 * EMA + 0.5 * attention`, context 8 |
| `ATTN-GATED-16-TRAIN` | `0.5 * EMA + 0.5 * attention`, context 16 |

---

## AttnOpt Design

- **Values** are the RMS-normalized raw gradients themselves.
- **History window** is the last `8` or `16` optimizer steps.
- **Keys and queries** come from low-dimensional gradient statistics (mean, second moment, L1 norm) plus recency positional embeddings.
- `W_q`, `W_k`, and recency embeddings are updated online in the trainable variants using a next-gradient prediction surrogate (cosine similarity loss).
- Adam's **second moment** (variance estimate) is kept — only the first moment is replaced.

This is not full meta-learning. It is a cheap, online learned-attention approximation designed for a first experimental pass at the hypothesis.

---

## Project Structure

```text
attn-optimizer/
├── model/
│   └── gpt.py
├── optimizers/
│   ├── attnopt.py
│   └── muon.py
├── data/
│   └── fineweb.py
├── configs/
│   └── runs.py
├── train.py
├── preflight.py
├── launchers/
│   ├── launch_local.sh
│   ├── launch_single.sh
│   └── launch_vast.sh
└── analysis/
    ├── attnopt_toy.py
    ├── attnopt_compare.py
    └── results.ipynb
```

---

## Setup

```bash
pip install -r requirements.txt
wandb login
```

Run the synthetic smoke test before any paid job:

```bash
python preflight.py --tiny
```

---

## Local Reproduction

```bash
git clone <your-repo-url> attn-optimizer
cd attn-optimizer

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

wandb login
export WANDB_ENTITY="your_team_name"
export WANDB_PROJECT="attn-optimizer"
export FINEWEB_MAX_SHARDS=10   # or smaller for a cheaper test
```

Verify and run:

```bash
python preflight.py --tiny
bash launchers/launch_single.sh BASE-ADAMW
# or full sweep
bash launchers/launch_local.sh
```

Inspect results:

```bash
jupyter notebook analysis/results.ipynb
```

---

## Running

Single run:

```bash
bash launchers/launch_single.sh BASE-ADAMW
```

Sequential sweep on one GPU:

```bash
bash launchers/launch_local.sh
```

Vast.ai:

```bash
ssh root@<vast-instance>
cd /workspace/attn-optimizer
bash launchers/launch_vast.sh
```

Optional FineWeb shard cap:

```bash
FINEWEB_MAX_SHARDS=2 bash launchers/launch_single.sh BASE-ADAMW
```

---

## Toy CPU Diagnostics

```bash
python analysis/attnopt_toy.py --steps 30
python analysis/attnopt_compare.py --steps 120
```

---

## Results

Track live runs on Weights & Biases with project `attn-optimizer`, then compare loss curves in `analysis/results.ipynb`.
