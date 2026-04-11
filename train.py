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
from optimizers.attnraw_v1 import AttnRawV1
from optimizers.attnraw_v1_g import AttnRawV1G
from optimizers.attnraw_v2 import AttnRawV2
from optimizers.attnraw_v3 import AttnRawV3
from optimizers.simpleavg_v1 import SimpleAvgV1


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

    def state_dict(self):
        return {
            "optimizers": [opt.state_dict() for opt in self.optimizers],
        }

    def load_state_dict(self, state_dict):
        opt_states = state_dict.get("optimizers", [])
        if len(opt_states) != len(self.optimizers):
            raise ValueError(
                "Optimizer count mismatch when loading CombinedOptimizer state."
            )
        for opt, opt_state in zip(self.optimizers, opt_states):
            opt.load_state_dict(opt_state)


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


def _parse_count(raw: str) -> int:
    try:
        return int(raw)
    except ValueError:
        return int(float(raw))


def _tokens_per_step(tcfg):
    return tcfg["micro_batch_size"] * tcfg["seq_len"] * tcfg["grad_accum_steps"]


def _atomic_torch_save(payload, path: str):
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def build_model(run_cfg, model_config_override=None):
    model_cfg = dict(MODEL_CONFIG)
    if model_config_override:
        model_cfg.update(model_config_override)
    cfg = GPTConfig(**model_cfg)
    return GPT(cfg)


def _split_embed_params(model):
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
    return embed_params, other_params


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
        embed_params, other_params = _split_embed_params(model)
        muon_opt = Muon(other_params, lr=lr, weight_decay=wd)
        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([muon_opt, embed_opt])

    elif opt_name in (
        "simpleavg_v1",
        "attnraw_v1",
        "attnraw_v1_g",
        "attnraw_v2",
        "attnraw_v3",
    ):
        cfg = run_cfg.get("grad_opt_config", {})
        embed_params, other_params = _split_embed_params(model)

        if opt_name == "simpleavg_v1":
            grad_opt = SimpleAvgV1(
                other_params,
                lr=lr,
                weight_decay=wd,
                context_length=cfg.get("context_length", 4),
            )
        elif opt_name == "attnraw_v1":
            grad_opt = AttnRawV1(
                other_params,
                lr=lr,
                weight_decay=wd,
                context_length=cfg.get("context_length", 4),
                mix_beta=cfg.get("mix_beta", 0.9),
            )
        elif opt_name == "attnraw_v1_g":
            grad_opt = AttnRawV1G(
                other_params,
                lr=lr,
                weight_decay=wd,
                context_length=cfg.get("context_length", 4),
                temperature=cfg.get("temperature", 1.0),
                mix_beta=cfg.get("mix_beta", 0.9),
            )
        elif opt_name == "attnraw_v2":
            grad_opt = AttnRawV2(
                other_params,
                lr=lr,
                weight_decay=wd,
                context_length=cfg.get("context_length", 4),
            )
        else:  # attnraw_v3
            grad_opt = AttnRawV3(
                other_params,
                lr=lr,
                weight_decay=wd,
                context_length=cfg.get("context_length", 4),
                mix_beta=cfg.get("mix_beta", 0.1),
            )

        embed_opt = torch.optim.Adam(
            embed_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        return CombinedOptimizer([grad_opt, embed_opt])

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


# ------------------------------------------------------------------ #
# Main training loop                                                  #
# ------------------------------------------------------------------ #


def train(
    run_id: str,
    max_steps_override: int | None = None,
    max_tokens: int | None = None,
    checkpoint_every: int | None = None,
    resume_from: str | None = None,
):
    run_cfg = RUNS[run_id]
    tcfg = dict(TRAIN_CONFIG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ.setdefault("FINEWEB_MAX_SHARDS", "20")

    env_max_steps = os.environ.get("MAX_STEPS")
    env_max_tokens = os.environ.get("MAX_TOKENS")
    env_checkpoint_every = os.environ.get("CHECKPOINT_EVERY")
    env_resume_from = os.environ.get("RESUME_FROM")

    if max_steps_override is None and env_max_steps:
        max_steps_override = _parse_count(env_max_steps)
    if max_tokens is None and env_max_tokens:
        max_tokens = _parse_count(env_max_tokens)
    if checkpoint_every is None and env_checkpoint_every:
        checkpoint_every = _parse_count(env_checkpoint_every)
    if resume_from is None and env_resume_from:
        resume_from = env_resume_from

    if max_steps_override is not None and max_tokens is not None:
        raise ValueError("Use only one of max_steps or max_tokens.")

    tokens_per_step = _tokens_per_step(tcfg)
    if max_tokens is None:
        if "max_tokens" in tcfg:
            max_tokens = tcfg["max_tokens"]
        elif "max_steps" in tcfg:
            max_steps_override = max_steps_override or tcfg.get("max_steps")
    if max_tokens is not None:
        max_steps = math.ceil(max_tokens / tokens_per_step)
    elif max_steps_override is not None:
        max_steps = max_steps_override
    else:
        raise ValueError("Either max_tokens or max_steps must be set.")

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    tcfg["max_steps"] = max_steps
    if checkpoint_every is None:
        checkpoint_every = 500
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be >= 0.")

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
    raw_model = getattr(model, "_orig_mod", model)
    print(f"[{run_id}] params: {raw_model.get_num_params() / 1e6:.1f}M")
    target_tokens = max_steps * tokens_per_step
    print(f"[{run_id}] steps: {max_steps} (~{target_tokens:,} tokens)")

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

    # ---- Logging + Checkpoint setup ----
    log_dir = os.path.join(os.environ.get("LOG_DIR", "logs"), run_id)
    ckpt_dir = os.path.join(os.environ.get("CKPT_DIR", "checkpoints"), run_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "metrics.jsonl")
    ckpt_latest_path = os.path.join(ckpt_dir, "ckpt_latest.pt")
    ckpt_final_path = os.path.join(ckpt_dir, "ckpt_final.pt")

    def save_checkpoint(step, path):
        payload = {
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "run_cfg": run_cfg,
            "train_config": tcfg,
            "step": step,
        }
        _atomic_torch_save(payload, path)

    if resume_from == "latest":
        resume_from = ckpt_latest_path
    if resume_from:
        if not os.path.isfile(resume_from):
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        raw_model.load_state_dict(checkpoint["model_state"])
        opt_state = checkpoint.get("optimizer_state")
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        start_step = int(checkpoint.get("step", 0))
        print(f"[{run_id}] Resumed from {resume_from} at step {start_step}")
    else:
        start_step = 0

    log_mode = "a" if resume_from else "w"
    log_file = open(log_path, log_mode)

    # ---- Training ----
    warmup_steps = tcfg["warmup_steps"]
    grad_accum = tcfg["grad_accum_steps"]
    max_lr = run_cfg["lr"]
    min_lr = max_lr * tcfg["min_lr_ratio"]
    grad_clip = tcfg["grad_clip"]
    log_interval = tcfg["log_interval"]

    model.train()
    step = start_step
    t0 = time.time()

    pbar = tqdm(total=max_steps, desc=run_id, unit="step", initial=step)
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
        if checkpoint_every and step % checkpoint_every == 0:
            save_checkpoint(step, ckpt_latest_path)

    pbar.close()

    # ---- Save checkpoint ----
    save_checkpoint(step, ckpt_latest_path)
    save_checkpoint(step, ckpt_final_path)
    print(f"[{run_id}] Saved checkpoint -> {ckpt_final_path}")

    log_file.close()


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run ID from configs/runs.py"
    )
    parser.add_argument("--max_steps", type=str, default=None)
    parser.add_argument("--max_tokens", type=str, default=None)
    parser.add_argument("--checkpoint_every", type=str, default=None)
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint, or 'latest' for checkpoints/<run_id>/ckpt_latest.pt",
    )
    args = parser.parse_args()

    if args.run_id not in RUNS:
        raise ValueError(
            f"Unknown run_id '{args.run_id}'. Available: {list(RUNS.keys())}"
        )

    max_steps = _parse_count(args.max_steps) if args.max_steps else None
    max_tokens = _parse_count(args.max_tokens) if args.max_tokens else None
    checkpoint_every = (
        _parse_count(args.checkpoint_every) if args.checkpoint_every else None
    )
    train(
        args.run_id,
        max_steps_override=max_steps,
        max_tokens=max_tokens,
        checkpoint_every=checkpoint_every,
        resume_from=args.resume_from,
    )
