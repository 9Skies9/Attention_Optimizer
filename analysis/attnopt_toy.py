import argparse
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


def build_toy_batch(batch_size: int, seq_len: int, vocab_size: int):
    base = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    offsets = torch.arange(batch_size).unsqueeze(1) * 5
    x = (base + offsets) % vocab_size
    y = (x + 1) % vocab_size
    return x.long(), y.long()


def build_tiny_gpt(vocab_size: int = 64, seq_len: int = 16):
    return GPT(
        GPTConfig(
            n_layer=2,
            n_head=2,
            n_embd=32,
            vocab_size=vocab_size,
            block_size=seq_len,
        )
    )


def instrument_attnraw(optimizer: AttnRaw):
    original = optimizer._compute_past_mix

    def wrapped(self, g_flat, history):
        current_norm = g_flat.norm().clamp(min=1e-8)
        history_norms = history.norm(dim=1).clamp(min=1e-8)
        scores = (history @ g_flat) / (history_norms * current_norm)
        alpha = torch.softmax(scores, dim=0)
        self._last_alpha = alpha.detach().cpu()
        self._last_query = g_flat.detach().cpu()
        self._last_slot_count = int(history.shape[0])
        return original(g_flat, history)

    optimizer._compute_past_mix = types.MethodType(wrapped, optimizer)


def run_history_probe():
    set_seed(13)
    p = torch.nn.Parameter(torch.zeros(4, 4))
    opt = AttnRaw([p], lr=1e-1, context_length=4, mix_beta=0.6)
    instrument_attnraw(opt)

    grads = [
        torch.full_like(p, 1.0),
        torch.full_like(p, 1.0),
        torch.full_like(p, -1.0),
        torch.full_like(p, -1.0),
        torch.eye(4),
        -torch.eye(4),
    ]

    rows = []
    for step, grad in enumerate(grads, start=1):
        p.grad = grad.clone()
        opt.step()
        opt.zero_grad(set_to_none=True)
        alpha = getattr(opt, "_last_alpha", None)
        rows.append(
            {
                "step": step,
                "slot_count": getattr(opt, "_last_slot_count", 1),
                "alpha": None if alpha is None else [round(float(v), 4) for v in alpha.tolist()],
            }
        )
    return rows


def train_model(model, optimizer, x, y, steps: int):
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def run_overfit(steps: int):
    set_seed(21)
    vocab_size = 64
    seq_len = 16
    x, y = build_toy_batch(batch_size=8, seq_len=seq_len, vocab_size=vocab_size)

    adam_model = build_tiny_gpt(vocab_size=vocab_size, seq_len=seq_len)
    attnopt_model = build_tiny_gpt(vocab_size=vocab_size, seq_len=seq_len)

    adam = torch.optim.AdamW(adam_model.parameters(), lr=1e-3)
    attnopt = AttnRaw(
        attnopt_model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
        context_length=8,
        mix_beta=0.6,
    )
    instrument_attnraw(attnopt)

    adam_losses = train_model(adam_model, adam, x, y, steps=steps)
    attnopt_losses = train_model(attnopt_model, attnopt, x, y, steps=steps)
    final_alpha = getattr(attnopt, "_last_alpha", None)
    return adam_losses, attnopt_losses, final_alpha


def main():
    parser = argparse.ArgumentParser(description="Toy CPU diagnostics for AttnRaw.")
    parser.add_argument("--steps", type=int, default=30, help="Toy overfit steps per optimizer.")
    args = parser.parse_args()

    print("AttnRaw History Probe")
    for row in run_history_probe():
        print(
            f"step={row['step']} slots={row['slot_count']} "
            f"alpha={row['alpha']}"
        )

    adam_losses, attnopt_losses, final_alpha = run_overfit(args.steps)
    print("\nToy Overfit")
    print(
        f"adamw   start={adam_losses[0]:.4f} end={adam_losses[-1]:.4f} "
        f"delta={adam_losses[-1] - adam_losses[0]:.4f}"
    )
    print(
        f"attnopt start={attnopt_losses[0]:.4f} end={attnopt_losses[-1]:.4f} "
        f"delta={attnopt_losses[-1] - attnopt_losses[0]:.4f}"
    )
    if final_alpha is not None:
        print(f"final_attnopt_alpha={[round(float(v), 4) for v in final_alpha.tolist()]}")


if __name__ == "__main__":
    main()
