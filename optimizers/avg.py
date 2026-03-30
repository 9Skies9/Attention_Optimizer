#
# Sliding-window average optimizer.
#
# Replaces Adam's first-moment EMA with a literal average over the most recent
# K gradients, while keeping Adam's second moment, epsilon, and decoupled weight
# decay behavior.

import torch
from torch.optim import Optimizer


class Avg(Optimizer):
    """
    Adam-like optimizer with a sliding-window first moment.

    The first-moment term is the uniform average of the current gradient and the
    most recent K-1 stored gradients for each parameter tensor:

        m_t = (1 / L) * sum_{i=0}^{L-1} g_{t-i}

    where L is the number of available history entries up to context_length.
    The second-moment path remains identical to Adam.
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.95),
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

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    self._init_param_state(p)

                g = p.grad.float()
                state["step"] += 1
                t = state["step"]

                history = [g] + state["grad_history"][: K - 1]
                m_tilde = torch.stack(history, dim=0).mean(dim=0)

                v = state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(m_tilde, m_tilde, value=1.0 - beta2)
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
