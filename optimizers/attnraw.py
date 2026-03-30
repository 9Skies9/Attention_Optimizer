#
# Tensorwise Type 1 attention optimizer.
#
# Each parameter tensor keeps a short history of its past gradients. The current
# flattened tensor gradient attends over that history via cosine similarity, then
# mixes the retrieved past update with the current gradient in a convex
# combination.

import torch
from torch.optim import Optimizer


class AttnRaw(Optimizer):
    """
    Parameter-free tensorwise attention optimizer.

    Type 1 uses one attention distribution per parameter tensor rather than per
    scalar. Query/key projections are removed entirely: cosine similarity
    between the current flattened tensor gradient and each past flattened tensor
    gradient decides which history entries matter.

    Args:
        params:         model parameters
        lr:             learning rate
        betas:          Adam-style betas for second moment normalization
        eps:            numerical stability term
        weight_decay:   decoupled weight decay
        context_length: gradient history window K
        mix_beta:       weight assigned to attended past gradients
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        context_length=8,
        mix_beta=0.5,
    ):
        if context_length < 1:
            raise ValueError("context_length must be >= 1")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            context_length=context_length,
            mix_beta=mix_beta,
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
    ) -> torch.Tensor:
        """Return the attention-weighted sum of past tensor gradients."""
        current_norm = g_flat.norm().clamp(min=1e-8)
        history_norms = history.norm(dim=1).clamp(min=1e-8)
        scores = history @ g_flat
        scores = scores / (history_norms * current_norm)
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
            mix_beta = group["mix_beta"]

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
                    m_past = self._compute_past_mix(g_flat, history)
                    m_type1 = mix_beta * m_past + (1.0 - mix_beta) * g_flat
                else:
                    m_type1 = g_flat

                m_tilde = m_type1.reshape_as(p)

                v = state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(m_tilde, m_tilde, value=1.0 - beta2)
                v_hat = v / (1.0 - beta2 ** t)

                state["grad_history"] = (
                    [g_flat.detach().clone()] + state["grad_history"]
                )[:K]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.addcdiv_(
                    m_tilde.to(p.dtype),
                    v_hat.sqrt().add_(eps),
                    value=-lr,
                )

        return loss
