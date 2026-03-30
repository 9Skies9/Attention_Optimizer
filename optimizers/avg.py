#
# Sliding-window average optimizer.
#
# Replaces Adam's first-moment EMA with a weighted mix of the current gradient
# and the uniform average of the most recent K past gradients:
#
#   m_t = (1 - mix_beta) * g_t + mix_beta * mean(g_{t-1}, ..., g_{t-K})

import torch
from torch.optim import Optimizer


class Avg(Optimizer):
    """
    Adam-like optimizer with a weighted history first moment.

        m_t = (1 - mix_beta) * g_t + mix_beta * mean(g_{t-1}, ..., g_{t-K})

    The second moment tracks the variance of m_t, not the raw gradient.
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
        raw_second_moment=False,
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
            raw_second_moment=raw_second_moment,
        )
        super().__init__(params, defaults)

    def _init_param_state(self, p: torch.Tensor):
        state = self.state[p]
        state["step"] = 0
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["grad_history"] = []

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            _, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            K = group["context_length"]
            mix_beta = group["mix_beta"]
            raw_v = group["raw_second_moment"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    self._init_param_state(p)

                g = p.grad.float()
                state["step"] += 1
                t = state["step"]

                past = state["grad_history"][:K]
                if past:
                    m_past = torch.stack(past, dim=0).mean(dim=0)
                    m_tilde = (1.0 - mix_beta) * g + mix_beta * m_past
                else:
                    m_tilde = g

                v = state["exp_avg_sq"]
                v_input = g if raw_v else m_tilde
                v.mul_(beta2).addcmul_(v_input, v_input, value=1.0 - beta2)
                v_hat = v / (1.0 - beta2 ** t)

                state["grad_history"] = (
                    [g.detach().clone()] + state["grad_history"]
                )[:K]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.addcdiv_(
                    m_tilde.to(p.dtype),
                    v_hat.sqrt().add_(eps),
                    value=-lr,
                )

        return loss
