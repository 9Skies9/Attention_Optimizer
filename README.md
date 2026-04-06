## AttnPrec: Metric-Aware Cosine Attention over Gradient History

### Motivation

Modern large-scale model training is highly sensitive to the choice of optimizer. Adam/AdamW remain default choices because they provide good convergence with exponential moving averages (EMAs) of gradient moments. However, this compression can be an information bottleneck when gradient directions change non-stationarily.

This project explores whether optimization history can be mixed in a **content-dependent way** using attention, rather than through a fixed exponential decay.

### Method

We propose two attention-based optimizer families that operate independently per tensor/layer:

**AttnRaw**: attention scores computed in raw gradient space
**AttnPrec**: attention scores computed in preconditioned gradient space (using second-moment normalization)

For each family, we define three variants to test the role of retained optimizer state:

| Variant | Keeps m_{t-1}? | Keeps v_{t-1}? |
|---------|-----------------|-----------------|
| v1      | Yes             | Yes             |
| v2      | No              | Yes             |
| v3      | No              | No              |

We also include a **SimpleAvg** baseline that replaces attention with uniform averaging over the same history window.

### Experiment Design

Following the research.md specification:

- **Model**: nanoGPT ~44M (~46M params: 6 layers, 8 heads, 512 dim)
- **Training budget**: ~4.2B tokens per run (16,000 steps × 262,144 tokens/step)
- **History length sweep**: L ∈ {4, 8, 16}
- **Temperature sweep**: τ ∈ {0.5, 1.0, 2.0} (for attention-based methods)

### Optimizer Variants

| Optimizer      | Similarity Metric     | State Retention | Formula Type                |
|----------------|----------------------|-----------------|-----------------------------|
| AdamW          | N/A                  | m + v           | Standard AdamW              |
| SGD            | N/A                  | None            | Vanilla SGD                 |
| AttnRaw-v1/v2/v3 | Cosine (raw)      | v1: m+v, v2: v, v3: none | Attention-weighted mixture |
| AttnPrec-v1/v2/v3 | Cosine (preconditioned) | v1: m+v, v2: v, v3: none | Attention-weighted mixture |
| SimpleAvg-v1/v2/v3 | Uniform average   | v1: m+v, v2: v, v3: none | Simple average mixture    |

### Key Formulas

**AttnPrec** uses preconditioned gradients for similarity:
```
g_t~ = g_t / (sqrt(v_{t-1}) + eps)
g_{t-i}~ = g_{t-i} / (sqrt(v_{t-i}) + eps)
s_i = cos(g_t~, g_{t-i}~)
```

**Gradient mixture** (all variants):
```
g_bar = beta * g_t + (1 - beta) * sum_i(alpha_i * g_{t-i})
```

### Running Experiments

**Single run:**
```bash
python train.py --run_id ATTNRAW-V1-L8-T1.0
```

**Distributed across 4 GPUs (recommended):**
```bash
# Launch all 4 GPUs - they coordinate via shared state file
./launch_distributed.sh

# Monitor progress
cat experiment_state.json

# Or watch live
watch -n 5 'cat experiment_state.json'

# Kill all
pkill -f run_distributed.py
```

**How it works:** 4 processes share a JSON state file (`experiment_state.json`). When a GPU finishes a run, it atomically claims the next available run. This means GPUs stay balanced - if one is faster it just picks up more work.

### Available Runs

- **Baselines**: SGD, AdamW
- **AttnRaw-v1/v2/v3**: 3 variants × 3 L values × 3 τ values = 27 runs
- **AttnPrec-v1/v2/v3**: 3 variants × 3 L values × 3 τ values = 27 runs
- **SimpleAvg-v1/v2/v3**: 3 variants × 3 L values = 9 runs

**Total: 65 runs** at ~4.2B tokens each (~2-3 hours per run on A100)

### Success Criteria

- AttnPrec variants should outperform AttnRaw if metric-aware similarity helps retrieval
- v2 or v3 should remain competitive with v1 if explicit history can replace Adam-style state
- Improvements should not come with prohibitive compute/memory overhead
