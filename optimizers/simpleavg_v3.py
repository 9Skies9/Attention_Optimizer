#
# SimpleAvg V3: Simple average over gradient history, no momentum or second-moment EMA.
#
# Uses simple averaging (not attention) to mix current gradient with history.
# Neither m_{t-1} nor v_{t-1} is retained; second moment is computed from history.

import torch
from torch.optim import Optimizer


class SimpleAvgV3(Optimizer):
    """
    SimpleAvg V3: simple averaging, no EMA state.

    Args:
        params:         model parameters
        lr:             learning rate
        eps:            numerical stability term
        weight_decay:   decoupled weight decay
        context_length: number of past gradients to average over
        mix_beta:       weight assigned to past gradient average
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        eps=1e-8,
        weight_decay=0.0,
        context_length=8,
        mix_beta=0.9,
    ):
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        if not 0.0 < mix_beta <= 1.0:
            raise ValueError("mix_beta must be in (0, 1]")

        defaults = dict(
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            context_length=context_length,
            mix_beta=mix_beta,
        )
        super().__init__(params, defaults)

    def _init_param_state(self, p: torch.Tensor):
        state = self.state[p]
        state["step"] = 0
        state["grad_history"] = []

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
                    m_past = history.mean(dim=0)
                    g_bar_flat = mix_beta * g_flat + (1.0 - mix_beta) * m_past

                    g_sq = g_flat.pow(2)
                    past_sq = history.pow(2).mean(dim=0)
                    v_t_flat = mix_beta * g_sq + (1.0 - mix_beta) * past_sq
                else:
                    g_bar_flat = g_flat
                    v_t_flat = g_flat.pow(2)

                g_bar = g_bar_flat.reshape_as(p)
                v_t = v_t_flat.reshape_as(p)
                v_hat = v_t.sqrt().add_(eps)

                state["grad_history"] = (
                    [g_flat.detach().clone()] + state["grad_history"]
                )[:K]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.addcdiv_(
                    g_bar.to(p.dtype),
                    v_hat.to(p.dtype),
                    value=-lr,
                )

        return loss
