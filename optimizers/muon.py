# optimizers/muon.py
# Muon: MomentUm Orthogonalized by Newton-schulz
# Applies to 2D+ matrix parameters; uses Adam for embeddings/scalars.
# Reference: https://github.com/KellerJordan/modded-nanogpt

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G, steps=5):
    """Orthogonalize G via 5 Newton-Schulz iterations."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon optimizer.
    - Matrix params (ndim >= 2): Nesterov momentum + Newton-Schulz orthogonalization.
    - All other params (embeddings, norms): Adam.
    """

    def __init__(self, params, lr=3e-4, momentum=0.95,
                 adam_lr=3e-4, adam_betas=(0.9, 0.999), adam_eps=1e-8,
                 weight_decay=0.0, ns_steps=5):
        defaults = dict(
            lr=lr, momentum=momentum,
            adam_lr=adam_lr, adam_betas=adam_betas, adam_eps=adam_eps,
            weight_decay=weight_decay, ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            adam_lr = group["adam_lr"]
            b1, b2 = group["adam_betas"]
            eps = group["adam_eps"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # Route embedding-like matrices (vocab_size >> n_embd, ratio ≥ 32)
                # through Adam instead of Newton-Schulz to avoid ~8x scale amplification.
                _is_muon_param = (
                    p.ndim >= 2
                    and max(p.shape) / min(p.shape) < 32
                )

                if _is_muon_param:
                    # --- Muon update for matrix params ---
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(g)
                        state["step"] = 0

                    state["step"] += 1
                    buf = state["momentum_buffer"]

                    # Nesterov momentum
                    buf.mul_(momentum).add_(g)
                    g_nesterov = g + momentum * buf

                    # Orthogonalize
                    g_orth = zeropower_via_newtonschulz5(g_nesterov, steps=ns_steps)

                    # Scale to match RMS of original gradient
                    scale = max(1, g.size(0) / g.size(1)) ** 0.5
                    g_orth = g_orth * scale

                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(g_orth, alpha=-lr)

                else:
                    # --- Adam update for non-matrix params and embedding matrices ---
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    t = state["step"]
                    m = state["exp_avg"]
                    v = state["exp_avg_sq"]

                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)

                    m_hat = m / (1 - b1 ** t)
                    v_hat = v / (1 - b2 ** t)

                    if wd != 0:
                        p.mul_(1 - adam_lr * wd)
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-adam_lr)

        return loss
