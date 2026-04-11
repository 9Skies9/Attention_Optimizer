# configs/runs.py
# Optimizer experiment matrix:
# - SimpleAvg-V1: true uniform average over K+1 gradient window
# - AttnRaw-V1: past-only attention + mix_beta blend
# - AttnRaw-V1-G: g_t in window + mix_beta blend (G variant)
# - AttnRaw-V2: g_t in window + EMA on first moment
# - AttnRaw-V3: past-only attention + residual + EMA on top
# - Muon: Newton-Schulz iteration on non-embedding params

RUNS = {
    # --- Baselines ---
    "MUON": {
        "optimizer": "muon",
        "lr": 3e-4,
    },
    # --- SimpleAvg: true uniform average over K+1 gradient window (no EMA on numerator) ---
    "SIMPLEAVG-L4": {
        "optimizer": "simpleavg_v1",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": 4,
        },
    },
    # --- AttnRaw V1: past-only attention + mix_beta blend ---
    "ATTNRAW-V1-L4": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": 4,
            "mix_beta": 0.9,
        },
    },
    "ATTNRAW-V1-L4-MIX10": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": 4,
            "mix_beta": 0.1,
        },
    },
    # --- AttnRaw V1-G: g_t in window + mix_beta blend ---
    "ATTNRAW-V1-G-L4": {
        "optimizer": "attnraw_v1_g",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": 4,
            "temperature": 1.0,
            "mix_beta": 0.9,
        },
    },
    "ATTNRAW-V1-G-L4-T0.5": {
        "optimizer": "attnraw_v1_g",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": 4,
            "temperature": 0.5,
            "mix_beta": 0.9,
        },
    },
    "ATTNRAW-V1-G-L4-T2.0": {
        "optimizer": "attnraw_v1_g",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": 4,
            "temperature": 2.0,
            "mix_beta": 0.9,
        },
    },
    # --- MIX variants: g_t blended separately via mix_beta ---
    "ATTNRAW-MIX90-L4-T1.0": {
        "optimizer": "attnraw_v1_g",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": 4,
            "temperature": 1.0,
            "mix_beta": 0.9,
        },
    },
    "ATTNRAW-MIX75-L4-T1.0": {
        "optimizer": "attnraw_v1_g",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": 4,
            "temperature": 1.0,
            "mix_beta": 0.75,
        },
    },
    "ATTNRAW-MIX50-L4-T1.0": {
        "optimizer": "attnraw_v1_g",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": 4,
            "temperature": 1.0,
            "mix_beta": 0.5,
        },
    },
    "ATTNRAW-MIX25-L4-T1.0": {
        "optimizer": "attnraw_v1_g",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": 4,
            "temperature": 1.0,
            "mix_beta": 0.25,
        },
    },
    # --- AttnRaw V2: g_t in window + EMA on first moment ---
    "ATTNRAW-V2-L4": {
        "optimizer": "attnraw_v2",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": 4,
        },
    },
    # --- AttnRaw V3: past-only + residual + EMA on top ---
    "ATTNRAW-V3-L4": {
        "optimizer": "attnraw_v3",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": 4,
            "mix_beta": 0.1,
        },
    },
    "ATTNRAW-V3-L4-MIX10": {
        "optimizer": "attnraw_v3",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_opt_config": {
            "context_length": 4,
            "mix_beta": 0.1,
        },
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
