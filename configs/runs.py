# configs/runs.py
# Unified gradient optimizer experiment matrix:
# - SimpleAvg: true uniform average over K+1 gradient window
# - AttnRaw: cosine attention over gradient window
#   - include_g_t=True: g_t is part of attention window (G variants)
#   - include_g_t=False: g_t blended separately via mix_beta (MIX variants)

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

RUNS["MUON"] = {
    "optimizer": "muon",
    "lr": 3e-4,
}

# --- SimpleAvg: true uniform average over K+1 gradient window (no EMA on numerator) ---
for L in HISTORY_LENGTHS:
    key = f"SIMPLEAVG-L{L}"
    RUNS[key] = {
        "optimizer": "simpleavg",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": L,
        },
    }

# --- AttnRaw: g_t IN attention window (softmax over K+1 items) ---
for L, tau in itertools.product(HISTORY_LENGTHS, TEMPERATURES):
    key = f"ATTNRAW-G-L{L}-T{tau}"
    RUNS[key] = {
        "optimizer": "attnraw",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": L,
            "include_g_t": True,
            "temperature": tau,
        },
    }

# --- AttnRaw: g_t NOT in attention window (forced blend) ---
for L, tau in itertools.product(HISTORY_LENGTHS, TEMPERATURES):
    key = f"ATTNRAW-L{L}-T{tau}"
    RUNS[key] = {
        "optimizer": "attnraw",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": L,
            "include_g_t": False,
            "temperature": tau,
            "mix_beta": 0.9,
        },
    }

# --- AttnRaw: g_t NOT in window, mix_beta sweep (L=4, T=1.0) ---
MIX_BETAS = [0.9, 0.75, 0.5, 0.25, 0.1]
for mix_beta in MIX_BETAS:
    key = f"ATTNRAW-MIX{int(mix_beta * 100):02d}-L4-T1.0"
    RUNS[key] = {
        "optimizer": "attnraw",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": 4,
            "include_g_t": False,
            "temperature": 1.0,
            "mix_beta": mix_beta,
        },
    }

TRAIN_CONFIG = {
    "max_tokens": 2_000_000_000,
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
