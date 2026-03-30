#
# AttnRaw V2: Attention over gradient window + EMA on top.
#
# Instead of replacing Adam's EMA entirely, attention produces a smarter
# instantaneous signal m_tilde, and EMA smooths over it:
#
#   m_tilde = cosine-attention([g_t, g_{t-1}, ..., g_{t-L+1}])
#   m_t = beta1 * m_{t-1} + (1 - beta1) * m_tilde
#   v_t = beta2 * v_{t-1} + (1 - beta2) * m_tilde^2
#   theta -= lr * (m_t / (1 - beta1^t)) / (sqrt(v_t / (1 - beta2^t)) + eps)
#
# g_t is included in the attention window so attention can decide how much
# weight to give the current gradient vs history. mix_beta disappears —
# beta1 does the long-term smoothing job.

import torch
from torch.optim import Optimizer


class AttnRawV2(Optimizer):
    """
    AttnRaw V2: cosine-attention over gradient window feeding into Adam-style EMA.

    Args:
        params:         model parameters
        lr:             learning rate
        betas:          (beta1, beta2) — EMA decay for first and second moment
        eps:            numerical stability term
        weight_decay:   decoupled weight decay
        context_length: gradient history window K (g_t included)
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        context_length=8,
    ):
        if context_length < 1:
            raise ValueError("context_length must be >= 1")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            context_length=context_length,
        )
        super().__init__(params, defaults)

    def _init_param_state(self, p: torch.Tensor):
        state = self.state[p]
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["grad_history"] = []

    def _compute_past_mix(self, g_flat: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Cosine-attention weighted sum over history (which includes g_flat)."""
        # history: (K, d) — rows are gradients, history[0] = g_t
        norms = history.norm(dim=1).clamp(min=1e-8)
        query_norm = g_flat.norm().clamp(min=1e-8)
        scores = history @ g_flat / (norms * query_norm)
        alpha = torch.softmax(scores, dim=0)
        return alpha @ history

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            K = group["context_length"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    self._init_param_state(p)

                g = p.grad.float()
                g_flat = g.flatten()

                state["step"] += 1
                t = state["step"]

                # Prepend current gradient to history, keep last K
                state["grad_history"] = (
                    [g_flat.detach().clone()] + state["grad_history"]
                )[:K]

                # Attention over full window (including g_t)
                history = torch.stack(state["grad_history"], dim=0)
                m_tilde_flat = self._compute_past_mix(g_flat, history)
                m_tilde = m_tilde_flat.reshape_as(p)

                # EMA first moment on m_tilde
                m = state["exp_avg"]
                m.mul_(beta1).add_(m_tilde, alpha=1.0 - beta1)
                m_hat = m / (1.0 - beta1 ** t)

                # EMA second moment on m_tilde^2
                v = state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(m_tilde, m_tilde, value=1.0 - beta2)
                v_hat = v / (1.0 - beta2 ** t)

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Parameter update
                p.addcdiv_(
                    m_hat.to(p.dtype),
                    v_hat.sqrt().add_(eps),
                    value=-lr,
                )

        return loss
