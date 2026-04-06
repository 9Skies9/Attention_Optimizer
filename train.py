# train.py
#
# Platform-agnostic training entry point.
# Usage: python train.py --run_id ATTNRAW-8
#
# Works on Vast.ai, Lambda Labs, RunPod, or any local GPU.

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from configs.runs import RUNS, TRAIN_CONFIG, MODEL_CONFIG
from data.fineweb import get_dataloader
from model.gpt import GPT, GPTConfig
from optimizers.muon import Muon
from optimizers.attnraw import AttnRaw
from optimizers.avg import Avg
from optimizers.attnraw_v2 import AttnRawV2
from optimizers.attnraw_v3 import AttnRawV3
from optimizers.attnopt_b import AttnOptB
from optimizers.attnopt_a import AttnOptA
from optimizers.attnraw_v1_new import AttnRawV1 as AttnRawV1New
from optimizers.attnraw_v2_new import AttnRawV2 as AttnRawV2New
from optimizers.attnraw_v3_new import AttnRawV3 as AttnRawV3New
from optimizers.attnprec_v1 import AttnPrecV1
from optimizers.attnprec_v2 import AttnPrecV2
from optimizers.attnprec_v3 import AttnPrecV3
from optimizers.simpleavg_v1 import SimpleAvgV1
from optimizers.simpleavg_v2 import SimpleAvgV2
from optimizers.simpleavg_v3 import SimpleAvgV3


# ------------------------------------------------------------------ #
# Combined optimizer wrapper                                          #
# ------------------------------------------------------------------ #


class CombinedOptimizer:
    """Wraps two optimizers so train.py can call .step() / .zero_grad() once."""

    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def zero_grad(self, set_to_none=False):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for opt in self.optimizers:
            opt.step()


# ------------------------------------------------------------------ #
# Utilities                                                           #
# ------------------------------------------------------------------ #


def cosine_schedule(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_model(run_cfg, model_config_override=None):
    model_cfg = dict(MODEL_CONFIG)
    if model_config_override:
        model_cfg.update(model_config_override)
    cfg = GPTConfig(**model_cfg)
    return GPT(cfg)


def build_optimizer(model, run_cfg):
    opt_name = run_cfg["optimizer"]
    lr = run_cfg["lr"]
    wd = run_cfg.get("weight_decay", 0.0)

    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    elif opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    elif opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd
        )

    elif opt_name == "muon":
        embed_ids = {id(model.wte.weight)}
        embed_params, other_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in embed_ids:
                embed_params.append(p)
            else:
                other_params.append(p)
        muon_opt = Muon(other_params, lr=lr, weight_decay=wd)
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([muon_opt, embed_opt])

    elif opt_name == "avg":
        scfg = run_cfg["avg_config"]

        # Embeddings are excluded from history-based methods: too sparse and
        # memory-costly to store gradient history for a 50k-vocab table.
        embed_ids = {id(model.wte.weight)}
        embed_params, other_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in embed_ids:
                embed_params.append(p)
            else:
                other_params.append(p)

        avg_opt = Avg(
            other_params,
            lr=lr,
            weight_decay=wd,
            context_length=scfg["context_length"],
            mix_beta=scfg.get("mix_beta", 0.9),
            raw_second_moment=scfg.get("raw_second_moment", False),
        )
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([avg_opt, embed_opt])

    elif opt_name == "attnraw":
        acfg = run_cfg["attnraw_config"]

        # Split parameters: embedding → Adam (sparse, huge),
        # everything else → AttnOpt (tensorwise Type 1 attention).
        # Handle weight tying: wte.weight == lm_head.weight, deduplicate by id.
        embed_ids = {id(model.wte.weight)}
        embed_params, other_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in embed_ids:
                embed_params.append(p)
            else:
                other_params.append(p)

        attn_opt = AttnRaw(
            other_params,
            lr=lr,
            weight_decay=wd,
            context_length=acfg["context_length"],
            mix_beta=acfg.get("mix_beta", 0.9),
            raw_second_moment=acfg.get("raw_second_moment", False),
        )
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([attn_opt, embed_opt])

    elif opt_name == "attnraw_v2":
        ecfg = run_cfg["attnema_config"]

        embed_ids = {id(model.wte.weight)}
        embed_params, other_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in embed_ids:
                embed_params.append(p)
            else:
                other_params.append(p)

        attnema_opt = AttnRawV2(
            other_params,
            lr=lr,
            weight_decay=wd,
            context_length=ecfg["context_length"],
        )
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([attnema_opt, embed_opt])

    elif opt_name == "attnraw_v3":
        vcfg = run_cfg["attnema_config"]

        embed_ids = {id(model.wte.weight)}
        embed_params, other_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in embed_ids:
                embed_params.append(p)
            else:
                other_params.append(p)

        attnema_opt = AttnRawV3(
            other_params,
            lr=lr,
            weight_decay=wd,
            context_length=vcfg["context_length"],
            mix_beta=vcfg.get("mix_beta", 0.9),
        )
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([attnema_opt, embed_opt])

    elif opt_name in ("attnopt_b", "attnopt_a"):
        acfg = run_cfg["attnopt_config"]
        OptClass = AttnOptB if opt_name == "attnopt_b" else AttnOptA

        embed_ids = {id(model.wte.weight)}
        embed_params, other_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in embed_ids:
                embed_params.append(p)
            else:
                other_params.append(p)

        attn_opt = OptClass(
            other_params,
            lr=lr,
            lr_meta=acfg.get("lr_meta", 1e-4),
            weight_decay=wd,
            context_length=acfg["context_length"],
            d_attn=acfg.get("d_attn", 64),
            mix_beta=acfg.get("mix_beta", 0.9),
            **({"meta_every": acfg["meta_every"]} if opt_name == "attnopt_a" else {}),
        )
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([attn_opt, embed_opt])

    elif opt_name == "attnraw_v1":
        acfg = run_cfg["attn_config"]
        embed_ids = {id(model.wte.weight)}
        embed_params, other_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in embed_ids:
                embed_params.append(p)
            else:
                other_params.append(p)
        attn_opt = AttnRawV1New(
            other_params,
            lr=lr,
            weight_decay=wd,
            context_length=acfg["context_length"],
            mix_beta=acfg.get("mix_beta", 0.9),
            temperature=acfg.get("temperature", 1.0),
        )
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([attn_opt, embed_opt])

    elif opt_name == "attnraw_v2":
        acfg = run_cfg["attn_config"]
        embed_ids = {id(model.wte.weight)}
        embed_params, other_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in embed_ids:
                embed_params.append(p)
            else:
                other_params.append(p)
        attn_opt = AttnRawV2New(
            other_params,
            lr=lr,
            weight_decay=wd,
            context_length=acfg["context_length"],
            mix_beta=acfg.get("mix_beta", 0.9),
            temperature=acfg.get("temperature", 1.0),
        )
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([attn_opt, embed_opt])

    elif opt_name == "attnraw_v3":
        acfg = run_cfg["attn_config"]
        embed_ids = {id(model.wte.weight)}
        embed_params, other_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in embed_ids:
                embed_params.append(p)
            else:
                other_params.append(p)
        attn_opt = AttnRawV3New(
            other_params,
            lr=lr,
            weight_decay=wd,
            context_length=acfg["context_length"],
            mix_beta=acfg.get("mix_beta", 0.9),
            temperature=acfg.get("temperature", 1.0),
        )
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([attn_opt, embed_opt])

    elif opt_name == "attnprec_v1":
        acfg = run_cfg["attn_config"]
        embed_ids = {id(model.wte.weight)}
        embed_params, other_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in embed_ids:
                embed_params.append(p)
            else:
                other_params.append(p)
        attn_opt = AttnPrecV1(
            other_params,
            lr=lr,
            weight_decay=wd,
            context_length=acfg["context_length"],
            mix_beta=acfg.get("mix_beta", 0.9),
            temperature=acfg.get("temperature", 1.0),
        )
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([attn_opt, embed_opt])

    elif opt_name == "attnprec_v2":
        acfg = run_cfg["attn_config"]
        embed_ids = {id(model.wte.weight)}
        embed_params, other_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in embed_ids:
                embed_params.append(p)
            else:
                other_params.append(p)
        attn_opt = AttnPrecV2(
            other_params,
            lr=lr,
            weight_decay=wd,
            context_length=acfg["context_length"],
            mix_beta=acfg.get("mix_beta", 0.9),
            temperature=acfg.get("temperature", 1.0),
        )
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([attn_opt, embed_opt])

    elif opt_name == "attnprec_v3":
        acfg = run_cfg["attn_config"]
        embed_ids = {id(model.wte.weight)}
        embed_params, other_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in embed_ids:
                embed_params.append(p)
            else:
                other_params.append(p)
        attn_opt = AttnPrecV3(
            other_params,
            lr=lr,
            weight_decay=wd,
            context_length=acfg["context_length"],
            mix_beta=acfg.get("mix_beta", 0.9),
            temperature=acfg.get("temperature", 1.0),
        )
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([attn_opt, embed_opt])

    elif opt_name == "simpleavg_v1":
        acfg = run_cfg["avg_config"]
        embed_ids = {id(model.wte.weight)}
        embed_params, other_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in embed_ids:
                embed_params.append(p)
            else:
                other_params.append(p)
        avg_opt = SimpleAvgV1(
            other_params,
            lr=lr,
            weight_decay=wd,
            context_length=acfg["context_length"],
            mix_beta=acfg.get("mix_beta", 0.9),
        )
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([avg_opt, embed_opt])

    elif opt_name == "simpleavg_v2":
        acfg = run_cfg["avg_config"]
        embed_ids = {id(model.wte.weight)}
        embed_params, other_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in embed_ids:
                embed_params.append(p)
            else:
                other_params.append(p)
        avg_opt = SimpleAvgV2(
            other_params,
            lr=lr,
            weight_decay=wd,
            context_length=acfg["context_length"],
            mix_beta=acfg.get("mix_beta", 0.9),
        )
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([avg_opt, embed_opt])

    elif opt_name == "simpleavg_v3":
        acfg = run_cfg["avg_config"]
        embed_ids = {id(model.wte.weight)}
        embed_params, other_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in embed_ids:
                embed_params.append(p)
            else:
                other_params.append(p)
        avg_opt = SimpleAvgV3(
            other_params,
            lr=lr,
            weight_decay=wd,
            context_length=acfg["context_length"],
            mix_beta=acfg.get("mix_beta", 0.9),
        )
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([avg_opt, embed_opt])

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


# ------------------------------------------------------------------ #
# Main training loop                                                  #
# ------------------------------------------------------------------ #


def train(run_id: str):
    run_cfg = RUNS[run_id]
    tcfg = TRAIN_CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ.setdefault("FINEWEB_MAX_SHARDS", "10")

    seed = tcfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ---- Model ----
    model = build_model(run_cfg)
    model = model.to(device)
    model = torch.compile(model)
    print(f"[{run_id}] params: {model.get_num_params() / 1e6:.1f}M")

    # ---- Optimizer ----
    optimizer = build_optimizer(model, run_cfg)

    # ---- Data ----
    loader = get_dataloader(
        seq_len=tcfg["seq_len"],
        micro_batch_size=tcfg["micro_batch_size"],
        seed=tcfg["seed"],
        cache_dir=os.environ.get("DATA_CACHE_DIR", None),
    )
    data_iter = iter(loader)

    # ---- Logging setup ----
    log_dir = os.path.join(os.environ.get("LOG_DIR", "logs"), run_id)
    ckpt_dir = os.path.join(os.environ.get("CKPT_DIR", "checkpoints"), run_id)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "metrics.jsonl")
    log_file = open(log_path, "w")

    # ---- Training ----
    max_steps = tcfg["max_steps"]
    warmup_steps = tcfg["warmup_steps"]
    grad_accum = tcfg["grad_accum_steps"]
    max_lr = run_cfg["lr"]
    min_lr = max_lr * tcfg["min_lr_ratio"]
    grad_clip = tcfg["grad_clip"]
    log_interval = tcfg["log_interval"]

    model.train()
    step = 0
    t0 = time.time()

    pbar = tqdm(total=max_steps, desc=run_id, unit="step")
    while step < max_steps:
        # ---- LR schedule ----
        lr = cosine_schedule(step, warmup_steps, max_steps, max_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- Gradient accumulation ----
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / grad_accum

            loss.backward()
            accum_loss += loss.item()

        # ---- Gradient clip + step ----
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # ---- AttnOptA meta-step (val split for W_Q/W_K update) ----
        attnopt_a_inner = None
        if isinstance(optimizer, CombinedOptimizer):
            for opt in optimizer.optimizers:
                if isinstance(opt, AttnOptA):
                    attnopt_a_inner = opt
                    break
        elif isinstance(optimizer, AttnOptA):
            attnopt_a_inner = optimizer

        if attnopt_a_inner is not None:
            meta_every = run_cfg["attnopt_config"].get("meta_every", 10)
            if step % meta_every == 0 and step > 0:
                try:
                    val_x, val_y = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    val_x, val_y = next(data_iter)
                val_x, val_y = val_x.to(device), val_y.to(device)
                raw_model = getattr(model, "_orig_mod", model)
                attnopt_a_inner.meta_step(raw_model, val_x, val_y)

        # ---- Logging ----
        pbar.update(1)
        if step % log_interval == 0 and step > 0:
            dt = time.time() - t0
            tokens_per_sec = (
                tcfg["micro_batch_size"]
                * tcfg["seq_len"]
                * grad_accum
                * log_interval
                / dt
            )
            log = {
                "step": step,
                "loss": accum_loss,
                "lr": lr,
                "tokens_per_sec": int(tokens_per_sec),
            }
            log_file.write(json.dumps(log) + "\n")
            log_file.flush()
            pbar.set_postfix(
                loss=f"{accum_loss:.4f}",
                lr=f"{lr:.2e}",
                tok_s=f"{tokens_per_sec / 1e3:.1f}k",
            )
            t0 = time.time()

        step += 1

    pbar.close()

    # ---- Save checkpoint ----
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "ckpt_final.pt")
    # Unwrap torch.compile wrapper so state_dict keys don't have "_orig_mod." prefix
    raw_model = getattr(model, "_orig_mod", model)
    torch.save(
        {"model_state": raw_model.state_dict(), "run_cfg": run_cfg, "step": step},
        ckpt_path,
    )
    print(f"[{run_id}] Saved checkpoint -> {ckpt_path}")

    log_file.close()


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run ID from configs/runs.py"
    )
    args = parser.parse_args()

    if args.run_id not in RUNS:
        raise ValueError(
            f"Unknown run_id '{args.run_id}'. Available: {list(RUNS.keys())}"
        )

    train(args.run_id)
