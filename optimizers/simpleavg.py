#
# SimpleAvg: True uniform average over gradient window.
#
# g_bar = mean([g_t, g_{t-1}, ..., g_{t-K}])
# Second moment uses EMA for normalization.
# No EMA on first moment — g_bar is used directly.
#

import torch
from torch.optim import Optimizer


class SimpleAvg(Optimizer):
    """
    True uniform average over gradient window (including current gradient).

    g_bar = mean([g_t, g_{t-1}, ..., g_{t-K}])
    v = EMA(g_bar^2)
    p += -lr * g_bar / sqrt(v_hat)

    Args:
        params:         model parameters
        lr:             learning rate
        betas:          (beta2,) — EMA decay for second moment only
        eps:            numerical stability term
        weight_decay:   decoupled weight decay
        context_length: number of PAST gradients to store (K)
                         Window includes g_t, total K+1 gradients
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.999,),
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
            beta2 = group["betas"][0]
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

                past = state["grad_history"][:K]
                all_grads = [g_flat] + past
                g_bar_flat = torch.stack(all_grads, dim=0).mean(dim=0)

                g_bar = g_bar_flat.reshape_as(p)

                v = state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(g_bar, g_bar, value=1.0 - beta2)
                v_hat = v / (1.0 - beta2**t)

                state["grad_history"] = (
                    [g_flat.detach().clone()] + state["grad_history"]
                )[:K]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.addcdiv_(
                    g_bar.to(p.dtype),
                    v_hat.sqrt().add_(eps),
                    value=-lr,
                )

        return loss
