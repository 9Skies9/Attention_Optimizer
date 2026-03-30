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
    "BASE-MUON": {
        "optimizer": "muon",
        "lr": 3e-4,
    },
    "AVG-8": {
        "optimizer": "avg",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "avg_config": {
            "context_length": 8,
            "mix_beta": 0.9,
            "raw_second_moment": False,
        },
    },
    "AVG-8R": {
        "optimizer": "avg",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "avg_config": {
            "context_length": 8,
            "mix_beta": 0.9,
            "raw_second_moment": True,
        },
    },
    "ATTNRAW-8": {
        "optimizer": "attnraw",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "attnraw_config": {
            "context_length": 8,
            "mix_beta": 0.9,
            "raw_second_moment": False,
        },
    },
    "ATTNRAW-8R": {
        "optimizer": "attnraw",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "attnraw_config": {
            "context_length": 8,
            "mix_beta": 0.9,
            "raw_second_moment": True,
        },
    },
    "ATTNRAW-V2-8": {
        "optimizer": "attnraw_v2",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "attnema_config": {
            "context_length": 8,
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
