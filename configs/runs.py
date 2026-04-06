# configs/runs.py
# Experiment matrix following research.md style:
# - nanoGPT 44M
# - Sweep over history length L ∈ {4, 8, 16}
# - Sweep over temperature τ ∈ {0.5, 1.0, 2.0} for attention-based methods
# - AttnRaw-v1/v2/v3, AttnPrec-v1/v2/v3 with L×τ sweep
# - SimpleAvg-v1/v2/v3 with only L sweep (no temperature)
# - SGD and AdamW baselines

import itertools

HISTORY_LENGTHS = [4, 8, 16]
TEMPERATURES = [0.5, 1.0, 2.0]

RUNS = {}

# --- Baselines ---
RUNS["SGD"] = {
    "optimizer": "sgd",
    "lr": 1e-2,
}

RUNS["ADAMW"] = {
    "optimizer": "adamw",
    "lr": 3e-4,
    "weight_decay": 0.1,
}

# --- AttnRaw-v1: keep both m_{t-1} and v_{t-1} ---
for L, tau in itertools.product(HISTORY_LENGTHS, TEMPERATURES):
    key = f"ATTNRAW-V1-L{L}-T{tau}"
    RUNS[key] = {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "attn_config": {
            "context_length": L,
            "mix_beta": 0.9,
            "temperature": tau,
        },
    }

# --- AttnRaw-v2: remove m_{t-1}, keep v_{t-1} ---
for L, tau in itertools.product(HISTORY_LENGTHS, TEMPERATURES):
    key = f"ATTNRAW-V2-L{L}-T{tau}"
    RUNS[key] = {
        "optimizer": "attnraw_v2",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "attn_config": {
            "context_length": L,
            "mix_beta": 0.9,
            "temperature": tau,
        },
    }

# --- AttnRaw-v3: remove both m_{t-1} and v_{t-1} ---
for L, tau in itertools.product(HISTORY_LENGTHS, TEMPERATURES):
    key = f"ATTNRAW-V3-L{L}-T{tau}"
    RUNS[key] = {
        "optimizer": "attnraw_v3",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "attn_config": {
            "context_length": L,
            "mix_beta": 0.9,
            "temperature": tau,
        },
    }

# --- AttnPrec-v1: keep both m_{t-1} and v_{t-1} ---
for L, tau in itertools.product(HISTORY_LENGTHS, TEMPERATURES):
    key = f"ATTPREC-V1-L{L}-T{tau}"
    RUNS[key] = {
        "optimizer": "attnprec_v1",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "attn_config": {
            "context_length": L,
            "mix_beta": 0.9,
            "temperature": tau,
        },
    }

# --- AttnPrec-v2: remove m_{t-1}, keep v_{t-1} ---
for L, tau in itertools.product(HISTORY_LENGTHS, TEMPERATURES):
    key = f"ATTPREC-V2-L{L}-T{tau}"
    RUNS[key] = {
        "optimizer": "attnprec_v2",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "attn_config": {
            "context_length": L,
            "mix_beta": 0.9,
            "temperature": tau,
        },
    }

# --- AttnPrec-v3: remove both m_{t-1} and v_{t-1} ---
for L, tau in itertools.product(HISTORY_LENGTHS, TEMPERATURES):
    key = f"ATTPREC-V3-L{L}-T{tau}"
    RUNS[key] = {
        "optimizer": "attnprec_v3",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "attn_config": {
            "context_length": L,
            "mix_beta": 0.9,
            "temperature": tau,
        },
    }

# --- SimpleAvg-v1: keep both m_{t-1} and v_{t-1} ---
for L in HISTORY_LENGTHS:
    key = f"AVG-V1-L{L}"
    RUNS[key] = {
        "optimizer": "simpleavg_v1",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "avg_config": {
            "context_length": L,
            "mix_beta": 0.9,
        },
    }

# --- SimpleAvg-v2: remove m_{t-1}, keep v_{t-1} ---
for L in HISTORY_LENGTHS:
    key = f"AVG-V2-L{L}"
    RUNS[key] = {
        "optimizer": "simpleavg_v2",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "avg_config": {
            "context_length": L,
            "mix_beta": 0.9,
        },
    }

# --- SimpleAvg-v3: remove both m_{t-1} and v_{t-1} ---
for L in HISTORY_LENGTHS:
    key = f"AVG-V3-L{L}"
    RUNS[key] = {
        "optimizer": "simpleavg_v3",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "avg_config": {
            "context_length": L,
            "mix_beta": 0.9,
        },
    }

TRAIN_CONFIG = {
    "max_steps": 16_000,
    "warmup_steps": 500,
    "min_lr_ratio": 0.1,
    "micro_batch_size": 16,
    "grad_accum_steps": 16,
    "seq_len": 1024,
    "grad_clip": 1.0,
    "log_interval": 100,
    "seed": 42,
}

MODEL_CONFIG = {
    "n_layer": 6,
    "n_head": 8,
    "n_embd": 512,
    "vocab_size": 50304,
    "block_size": 1024,
}
