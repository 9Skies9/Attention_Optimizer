import argparse
import copy
import os
import random
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPT, GPTConfig
from optimizers.attnraw import AttnRaw


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def build_model(vocab_size: int, seq_len: int):
    return GPT(
        GPTConfig(
            n_layer=2,
            n_head=2,
            n_embd=32,
            vocab_size=vocab_size,
            block_size=seq_len,
        )
    )


def build_dataset(num_examples: int, seq_len: int, vocab_size: int):
    xs = []
    ys = []
    for i in range(num_examples):
        base = torch.arange(seq_len)
        pattern = (base * ((i % 5) + 1) + i * 7) % vocab_size
        x = pattern.long()
        y = torch.roll(x, shifts=-1)
        y[-1] = (x[0] + 3) % vocab_size
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def instrument_attnraw(optimizer: AttnRaw):
    original = optimizer._compute_past_mix

    def wrapped(self, g_flat, history):
        current_norm = g_flat.norm().clamp(min=1e-8)
        history_norms = history.norm(dim=1).clamp(min=1e-8)
        scores = (history @ g_flat) / (history_norms * current_norm)
        alpha = torch.softmax(scores, dim=0)
        entropy = -(alpha.clamp_min(1e-9) * alpha.clamp_min(1e-9).log()).sum()
        self._last_attn_stats = {
            "slot_count": int(history.shape[0]),
            "alpha": alpha.detach().cpu(),
            "entropy": float(entropy.detach().cpu()),
            "max_weight": float(alpha.max().detach().cpu()),
        }
        return original(g_flat, history)

    optimizer._compute_past_mix = types.MethodType(wrapped, optimizer)


def train(model, optimizer, x_all, y_all, steps: int, batch_size: int):
    losses = []
    attn_stats = []
    num_examples = x_all.shape[0]

    for step in range(steps):
        start = (step * batch_size) % num_examples
        end = start + batch_size
        if end <= num_examples:
            x = x_all[start:end]
            y = y_all[start:end]
        else:
            wrap = end - num_examples
            x = torch.cat([x_all[start:], x_all[:wrap]], dim=0)
            y = torch.cat([y_all[start:], y_all[:wrap]], dim=0)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(float(loss.detach()))
        if hasattr(optimizer, "_last_attn_stats"):
            attn_stats.append(optimizer._last_attn_stats)

    return losses, attn_stats


def summarize_attn_stats(stats):
    if not stats:
        return None
    last = stats[-1]
    mean_entropy = sum(s["entropy"] for s in stats) / len(stats)
    mean_max_weight = sum(s["max_weight"] for s in stats) / len(stats)
    return {
        "last_alpha": [round(float(v), 4) for v in last["alpha"].tolist()],
        "last_slot_count": last["slot_count"],
        "mean_entropy": mean_entropy,
        "mean_max_weight": mean_max_weight,
    }


def main():
    parser = argparse.ArgumentParser(description="Tiny CPU AdamW vs AttnRaw comparison.")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    set_seed(123)
    vocab_size = 96
    seq_len = 16
    x_all, y_all = build_dataset(num_examples=32, seq_len=seq_len, vocab_size=vocab_size)

    base_model = build_model(vocab_size=vocab_size, seq_len=seq_len)
    adam_model = copy.deepcopy(base_model)
    attnopt_model = copy.deepcopy(base_model)

    adam = torch.optim.AdamW(adam_model.parameters(), lr=1e-3)
    attnopt = AttnRaw(
        attnopt_model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
        context_length=8,
        mix_beta=0.6,
    )
    instrument_attnraw(attnopt)

    adam_losses, _ = train(adam_model, adam, x_all, y_all, steps=args.steps, batch_size=args.batch_size)
    attnopt_losses, attn_stats = train(
        attnopt_model, attnopt, x_all, y_all, steps=args.steps, batch_size=args.batch_size
    )

    print("Tiny CPU Compare")
    print(
        f"adamw   start={adam_losses[0]:.4f} mid={adam_losses[len(adam_losses)//2]:.4f} "
        f"end={adam_losses[-1]:.4f}"
    )
    print(
        f"attnopt start={attnopt_losses[0]:.4f} mid={attnopt_losses[len(attnopt_losses)//2]:.4f} "
        f"end={attnopt_losses[-1]:.4f}"
    )

    summary = summarize_attn_stats(attn_stats)
    if summary is not None:
        print("\nAttnRaw Attention Summary")
        print(f"mean_entropy={summary['mean_entropy']:.4f}")
        print(f"mean_max_weight={summary['mean_max_weight']:.4f}")
        print(f"last_slot_count={summary['last_slot_count']}")
        print(f"last_alpha={summary['last_alpha']}")

    print("\nStep Samples")
    sample_steps = [0, 9, 19, 39, 79, args.steps - 1]
    seen = set()
    for idx in sample_steps:
        if idx < 0 or idx >= args.steps or idx in seen:
            continue
        seen.add(idx)
        print(
            f"step={idx+1:3d} adamw={adam_losses[idx]:.4f} attnopt={attnopt_losses[idx]:.4f}"
        )


if __name__ == "__main__":
    main()
