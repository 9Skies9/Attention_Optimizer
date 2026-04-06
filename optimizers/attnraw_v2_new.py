#
# AttnRaw V2: Raw cosine attention, no first-moment EMA.
#
# Uses cosine similarity in raw gradient space for attention:
#   s_i = cos(g_t, g_{t-i})
#
# Attention scores computed in raw space, then applied to gradients.
# Only v_{t-1} is retained; m_{t-1} is replaced by the attention mixture.

import torch
from torch.optim import Optimizer


class AttnRawV2(Optimizer):
    """
    AttnRaw V2: raw cosine attention, no first-moment EMA.

    Args:
        params:         model parameters
        lr:             learning rate
        betas:          (beta2,) — EMA decay for second moment only
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
        betas=(0.999,),
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
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["grad_history"] = []

    def _compute_past_mix(
        self,
        g_flat: torch.Tensor,
        history: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Cosine attention in raw space."""
        current_norm = g_flat.norm().clamp(min=1e-8)
        history_norms = history.norm(dim=1).clamp(min=1e-8)
        scores = history @ g_flat
        scores = scores / (history_norms * current_norm)
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
            beta2 = group["betas"][0]
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

                history_list = state["grad_history"][:K]
                if history_list:
                    history = torch.stack(history_list, dim=0)
                    m_past = self._compute_past_mix(g_flat, history, temperature)
                    m_t_flat = mix_beta * g_flat + (1.0 - mix_beta) * m_past
                else:
                    m_t_flat = g_flat

                m_t = m_t_flat.reshape_as(p)

                v = state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(m_t, m_t, value=1.0 - beta2)
                v_hat = v / (1.0 - beta2**t)

                state["grad_history"] = (
                    [g_flat.detach().clone()] + state["grad_history"]
                )[:K]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.addcdiv_(
                    m_t.to(p.dtype),
                    v_hat.sqrt().add_(eps),
                    value=-lr,
                )

        return loss
