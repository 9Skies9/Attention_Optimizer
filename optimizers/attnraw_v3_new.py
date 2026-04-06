#
# AttnRaw V3: Raw cosine attention, no momentum or second-moment EMA.
#
# Uses cosine similarity in raw gradient space for attention:
#   s_i = cos(g_t, g_{t-i})
#
# Attention scores computed in raw space, then applied to gradients.
# Neither m_{t-1} nor v_{t-1} is retained; second moment is computed from history.

import torch
from torch.optim import Optimizer


class AttnRawV3(Optimizer):
    """
    AttnRaw V3: raw cosine attention, no EMA state.

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

    def _compute_past_mix(
        self,
        g_flat: torch.Tensor,
        history: torch.Tensor,
        temperature: float,
    ) -> tuple:
        """Cosine attention in raw space, returns both mixture and attention weights."""
        current_norm = g_flat.norm().clamp(min=1e-8)
        history_norms = history.norm(dim=1).clamp(min=1e-8)
        scores = history @ g_flat
        scores = scores / (history_norms * current_norm)
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
                if past:
                    history = torch.stack(past, dim=0)
                    m_past, alpha = self._compute_past_mix(g_flat, history, temperature)
                    m_t_flat = mix_beta * g_flat + (1.0 - mix_beta) * m_past

                    g_sq = g_flat.pow(2)
                    past_sq = history.pow(2)
                    v_t_flat = (1.0 - mix_beta) * (alpha @ past_sq) + mix_beta * g_sq
                else:
                    m_t_flat = g_flat
                    v_t_flat = g_flat.pow(2)

                m_t = m_t_flat.reshape_as(p)
                v_t = v_t_flat.reshape_as(p)
                v_hat = v_t.sqrt().add_(eps)

                state["grad_history"] = (
                    [g_flat.detach().clone()] + state["grad_history"]
                )[:K]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.addcdiv_(
                    m_t.to(p.dtype),
                    v_hat.to(p.dtype),
                    value=-lr,
                )

        return loss
