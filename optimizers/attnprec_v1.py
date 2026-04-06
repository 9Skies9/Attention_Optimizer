#
# AttnPrec V1: Preconditioned cosine attention over gradient window.
#
# Uses preconditioned gradient space for attention similarity:
#   g_t~ = g_t / (sqrt(v_{t-1}) + eps)
#   g_{t-i}~ = g_{t-i} / (sqrt(v_{t-i}) + eps)
#
# Attention scores computed on preconditioned space, then applied to raw gradients.
# Both m_{t-1} and v_{t-1} are retained.

import torch
from torch.optim import Optimizer


class AttnPrecV1(Optimizer):
    """
    AttnPrec V1: preconditioned cosine attention, retaining full Adam state.

    Args:
        params:         model parameters
        lr:             learning rate
        betas:          (beta1, beta2) — EMA decay for first and second moment
        eps:            numerical stability term
        weight_decay:   decoupled weight decay
        context_length: number of past gradients to attend over
        mix_beta:       weight assigned to attended past gradients
        temperature:    softmax temperature for attention scores
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        context_length=8,
        mix_beta=0.9,
        temperature=1.0,
    ):
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        if not 0.0 < mix_beta <= 1.0:
            raise ValueError("mix_beta must be in (0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            context_length=context_length,
            mix_beta=mix_beta,
            temperature=temperature,
        )
        super().__init__(params, defaults)

    def _init_param_state(self, p: torch.Tensor):
        state = self.state[p]
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["grad_history"] = []
        state["v_history"] = []

    def _compute_past_mix(
        self,
        g_flat: torch.Tensor,
        v_t: torch.Tensor,
        history: torch.Tensor,
        v_history: list,
        temperature: float,
    ) -> torch.Tensor:
        """Cosine attention in preconditioned space, applied to raw gradients."""
        query = g_flat / (torch.sqrt(v_t) + 1e-8)

        past_tilde = []
        for g_hist, v_hist in zip(history, v_history):
            past_tilde.append(g_hist / (torch.sqrt(v_hist) + 1e-8))
        past_tilde = torch.stack(past_tilde, dim=0)

        query_norm = query.norm().clamp(min=1e-8)
        past_norms = past_tilde.norm(dim=1).clamp(min=1e-8)
        scores = past_tilde @ query / (past_norms * query_norm)
        alpha = torch.softmax(scores / temperature, dim=0)
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
            mix_beta = group["mix_beta"]
            temperature = group["temperature"]

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

                past = state["grad_history"][:K]
                v_past = state["v_history"][:K]

                if past:
                    v_t = state["exp_avg_sq"].flatten()
                    v_past_flat = [v.flatten() for v in v_past]
                    m_past = self._compute_past_mix(
                        g_flat, v_t, torch.stack(past, dim=0), v_past_flat, temperature
                    )
                    g_bar_flat = (1.0 - mix_beta) * g_flat + mix_beta * m_past
                else:
                    g_bar_flat = g_flat

                g_bar = g_bar_flat.reshape_as(p)

                m = state["exp_avg"]
                m.mul_(beta1).add_(g_bar, alpha=1.0 - beta1)
                m_hat = m / (1.0 - beta1**t)

                v = state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(g_bar, g_bar, value=1.0 - beta2)
                v_hat = v / (1.0 - beta2**t)

                state["grad_history"] = (
                    [g_flat.detach().clone()] + state["grad_history"]
                )[:K]
                state["v_history"] = ([v.detach().clone()] + state["v_history"])[:K]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.addcdiv_(
                    m_hat.to(p.dtype),
                    v_hat.sqrt().add_(eps),
                    value=-lr,
                )

        return loss
