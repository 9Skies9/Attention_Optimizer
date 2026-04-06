#
# AttnOptB: Learned W_Q/W_K with next-gradient prediction training signal.
#
# W_Q and W_K are per-tensor learned projections. After each step, the
# optimizer observes g_{t+1} and trains W_Q/W_K so that the previous step's
# m_tilde aligns with the newly observed gradient.
#
# Architecture:
#   q_t   = g_t @ W_Q                     (d_attn,)
#   k_i   = g_{t-i} @ W_K                (d_attn,)
#   s_i   = dot(q_t, k_i) / sqrt(d_attn)
#   alpha = softmax(s)
#   m_past = alpha @ history
#   m_tilde = (1 - mix_beta) * g_t + mix_beta * m_past
#
# Meta-update (next-gradient prediction):
#   loss_phi = 1 - cos(m_tilde_prev, g_t)
#   W_Q, W_K updated via gradient of loss_phi at lr_meta

import math
import torch
from torch.optim import Optimizer


class AttnOptB(Optimizer):
    """
    Per-tensor learned attention optimizer trained via next-gradient prediction.

    Args:
        params:         model parameters
        lr:             learning rate for model parameter updates
        lr_meta:        learning rate for W_Q/W_K meta-updates
        betas:          (beta1, beta2) for Adam-style first/second moments
        eps:            numerical stability term
        weight_decay:   decoupled weight decay
        context_length: gradient history window K
        d_attn:         attention projection dimension
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        lr_meta=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        context_length=8,
        d_attn=64,
        mix_beta=0.9,
    ):
        defaults = dict(
            lr=lr,
            lr_meta=lr_meta,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            context_length=context_length,
            d_attn=d_attn,
            mix_beta=mix_beta,
        )
        super().__init__(params, defaults)

    def _init_param_state(self, p: torch.Tensor, d_attn: int):
        state = self.state[p]
        d = p.numel()

        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["grad_history"] = []

        # Learned projections — requires_grad so meta-loss can flow back
        state["W_Q"] = torch.nn.init.orthogonal_(
            torch.empty(d, d_attn, dtype=torch.float32, device=p.device)
        ).requires_grad_(True)
        state["W_K"] = torch.nn.init.orthogonal_(
            torch.empty(d, d_attn, dtype=torch.float32, device=p.device)
        ).requires_grad_(True)

        # Adam states for W_Q/W_K meta-optimizer
        state["wq_exp_avg"] = torch.zeros(d, d_attn, dtype=torch.float32, device=p.device)
        state["wq_exp_avg_sq"] = torch.zeros(d, d_attn, dtype=torch.float32, device=p.device)
        state["wk_exp_avg"] = torch.zeros(d, d_attn, dtype=torch.float32, device=p.device)
        state["wk_exp_avg_sq"] = torch.zeros(d, d_attn, dtype=torch.float32, device=p.device)

        # Previous step's m_tilde (for next-gradient prediction)
        state["m_tilde_prev"] = None

    def _attend(
        self,
        g_flat: torch.Tensor,
        history: torch.Tensor,
        W_Q: torch.Tensor,
        W_K: torch.Tensor,
        d_attn: int,
    ) -> torch.Tensor:
        """Project current and past gradients into d_attn, then retrieve past values."""
        q = g_flat @ W_Q
        ks = history @ W_K
        scores = ks @ q / math.sqrt(d_attn)
        alpha = torch.softmax(scores, dim=0)
        return alpha @ history

    def _adam_update(self, param, grad, exp_avg, exp_avg_sq, t, lr_meta, beta1=0.9, beta2=0.999, eps=1e-8):
        """In-place Adam update for a meta-parameter."""
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        m_hat = exp_avg / (1.0 - beta1 ** t)
        v_hat = exp_avg_sq / (1.0 - beta2 ** t)
        param.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr_meta)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lr_meta = group["lr_meta"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            K = group["context_length"]
            d_attn = group["d_attn"]
            mix_beta = group["mix_beta"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    self._init_param_state(p, d_attn)

                g = p.grad.float()
                g_flat = g.flatten()

                state["step"] += 1
                t = state["step"]

                W_Q = state["W_Q"]
                W_K = state["W_K"]

                # ---- Meta-update: next-gradient prediction ----
                # m_tilde from previous step should align with current g_t
                if state["m_tilde_prev"] is not None:
                    with torch.enable_grad():
                        m_prev = state["m_tilde_prev"]  # (d,) with grad
                        m_norm = m_prev / (m_prev.norm().clamp(min=1e-8))
                        g_norm = g_flat / (g_flat.norm().clamp(min=1e-8))
                        meta_loss = 1.0 - (m_norm * g_norm).sum()
                        meta_loss.backward()

                    if W_Q.grad is not None:
                        self._adam_update(
                            W_Q, W_Q.grad,
                            state["wq_exp_avg"], state["wq_exp_avg_sq"],
                            t, lr_meta,
                        )
                        W_Q.grad.zero_()
                    if W_K.grad is not None:
                        self._adam_update(
                            W_K, W_K.grad,
                            state["wk_exp_avg"], state["wk_exp_avg_sq"],
                            t, lr_meta,
                        )
                        W_K.grad.zero_()

                # ---- Compute m_tilde from past-only history for next step ----
                history_list = state["grad_history"][:K]
                with torch.enable_grad():
                    if history_list:
                        history = torch.stack(history_list, dim=0)
                        m_past = self._attend(g_flat, history, W_Q, W_K, d_attn)
                        m_tilde_flat = (1.0 - mix_beta) * g_flat + mix_beta * m_past
                    else:
                        m_tilde_flat = g_flat
                    state["m_tilde_prev"] = m_tilde_flat  # keep graph for next step

                m_tilde = m_tilde_flat.detach().reshape_as(p)

                state["grad_history"] = (
                    [g_flat.detach().clone()] + state["grad_history"]
                )[:K]

                # ---- Adam-style update on model params ----
                m = state["exp_avg"]
                m.mul_(beta1).add_(m_tilde, alpha=1.0 - beta1)
                m_hat = m / (1.0 - beta1 ** t)

                v = state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(m_tilde, m_tilde, value=1.0 - beta2)
                v_hat = v / (1.0 - beta2 ** t)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.addcdiv_(
                    m_hat.to(p.dtype),
                    v_hat.sqrt().add_(eps),
                    value=-lr,
                )

        return loss
