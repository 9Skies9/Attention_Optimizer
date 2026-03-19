# configs/runs.py
# Optimizer-only experiment matrix for the first AttnOpt study.

RUNS = {
    "BASE-SGD": {
        "optimizer": "sgd",
        "lr": 1e-2,
    },
    "BASE-ADAM": {
        "optimizer": "adam",
        "lr": 3e-4,
    },
    "BASE-ADAMW": {
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 0.1,
    },
    "BASE-MUON": {
        "optimizer": "muon",
        "lr": 3e-4,
    },
    "ATTN-PURE-8-TRAIN": {
        "optimizer": "attnopt",
        "lr": 3e-4,
        "weight_decay": 0.1,
        "attnopt_config": {
            "moment_mode": "pure",
            "context_length": 8,
            "trainable_attn": True,
            "gate_value": 1.0,
        },
    },
    "ATTN-GATED-8-TRAIN": {
        "optimizer": "attnopt",
        "lr": 3e-4,
        "weight_decay": 0.1,
        "attnopt_config": {
            "moment_mode": "gated",
            "context_length": 8,
            "trainable_attn": True,
            "gate_value": 0.5,
        },
    },
}

TRAIN_CONFIG = {
    "max_steps": 4_096,  # ~1.07B tokens at 262,144 tokens/step
    "warmup_steps": 200,
    "min_lr_ratio": 0.1,
    "micro_batch_size": 16,
    "grad_accum_steps": 16,
    "seq_len": 1024,
    "grad_clip": 1.0,
    "log_interval": 10,
    "seed": 42,
}

MODEL_CONFIG = {
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "vocab_size": 50304,
    "block_size": 1024,
}
