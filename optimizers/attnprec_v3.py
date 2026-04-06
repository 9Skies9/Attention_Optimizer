#
# AttnPrec V3: Preconditioned cosine attention, no momentum or second-moment EMA.
#
# Uses preconditioned gradient space for attention similarity:
#   g_t~ = g_t / (sqrt(v_{t-1}) + eps)
#   g_{t-i}~ = g_{t-i} / (sqrt(v_{t-i}) + eps)
#
# Attention scores computed on preconditioned space, then applied to raw gradients.
# Neither m_{t-1} nor v_{t-1} is retained; second moment is computed from history.

import torch
from torch.optim import Optimizer


class AttnPrecV3(Optimizer):
    """
    AttnPrec V3: preconditioned cosine attention, no EMA state.

    Args:
        params:         model parameters
        lr:             learning rate
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
        state["grad_history"] = []
        state["v_history"] = []

    def _compute_past_mix(
        self,
        g_flat: torch.Tensor,
        v_t: torch.Tensor,
        history: torch.Tensor,
        v_history: list,
        temperature: float,
    ) -> tuple:
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
        return alpha @ history, alpha

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
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
                    v_t = torch.zeros_like(g_flat, dtype=torch.float32)
                    if state["v_history"]:
                        v_t = state["v_history"][0].flatten()
                    v_past_flat = [v.flatten() for v in v_past]
                    history = torch.stack(past, dim=0)
                    m_past, alpha = self._compute_past_mix(
                        g_flat, v_t, history, v_past_flat, temperature
                    )
                    g_bar_flat = (1.0 - mix_beta) * g_flat + mix_beta * m_past

                    g_sq = g_flat.pow(2)
                    past_sq = history.pow(2)
                    v_t_next_flat = mix_beta * g_sq + (1.0 - mix_beta) * (
                        alpha @ past_sq
                    )
                else:
                    g_bar_flat = g_flat
                    v_t_next_flat = g_flat.pow(2)

                g_bar = g_bar_flat.reshape_as(p)
                v_t_next = v_t_next_flat.reshape_as(p)
                v_hat = v_t_next.sqrt().add_(eps)

                state["grad_history"] = (
                    [g_flat.detach().clone()] + state["grad_history"]
                )[:K]
                state["v_history"] = ([v_t_next.detach().clone()] + state["v_history"])[
                    :K
                ]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.addcdiv_(
                    g_bar.to(p.dtype),
                    v_hat.to(p.dtype),
                    value=-lr,
                )

        return loss
