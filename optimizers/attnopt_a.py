#
# AttnOptA: Learned W_Q/W_K with differentiable step + held-out batch.
#
# W_Q/W_K are trained by: take a virtual differentiable step on the train
# batch using the current attention-weighted m_tilde, then measure val loss
# on the updated model. Backprop val loss through the virtual step to get
# gradients on W_Q/W_K.
#
# The optimizer exposes two methods:
#   .step(closure)                 — normal model update (no_grad)
#   .meta_step(model, val_x, val_y) — updates W_Q/W_K via val loss
#
# train.py calls meta_step every N steps after the normal parameter update.

import math
import torch
from torch.optim import Optimizer
from torch.func import functional_call


class AttnOptA(Optimizer):
    """
    Per-tensor learned attention optimizer trained via differentiable val step.

    Args:
        params:         model parameters
        lr:             learning rate for model parameter updates
        lr_meta:        learning rate for W_Q/W_K meta-updates
        betas:          (beta1, beta2) for Adam-style first/second moments
        eps:            numerical stability
        weight_decay:   decoupled weight decay
        context_length: gradient history window K
        d_attn:         attention projection dimension
        meta_every:     update W_Q/W_K every N optimizer steps
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
        meta_every=10,
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
            meta_every=meta_every,
        )
        super().__init__(params, defaults)

    def _init_param_state(self, p: torch.Tensor, d_attn: int):
        state = self.state[p]
        d = p.numel()

        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["grad_history"] = []

        state["W_Q"] = torch.nn.init.orthogonal_(
            torch.empty(d, d_attn, dtype=torch.float32, device=p.device)
        ).requires_grad_(True)
        state["W_K"] = torch.nn.init.orthogonal_(
            torch.empty(d, d_attn, dtype=torch.float32, device=p.device)
        ).requires_grad_(True)

        state["wq_exp_avg"] = torch.zeros(d, d_attn, dtype=torch.float32, device=p.device)
        state["wq_exp_avg_sq"] = torch.zeros(d, d_attn, dtype=torch.float32, device=p.device)
        state["wk_exp_avg"] = torch.zeros(d, d_attn, dtype=torch.float32, device=p.device)
        state["wk_exp_avg_sq"] = torch.zeros(d, d_attn, dtype=torch.float32, device=p.device)

        # Snapshots needed to rebuild the exact current step differentiably.
        state["param_before"] = None
        state["exp_avg_before"] = None
        state["exp_avg_sq_before"] = None
        state["last_m_tilde"] = None

    def _attend(self, g_flat, history, W_Q, W_K, d_attn):
        q = g_flat @ W_Q
        ks = history @ W_K
        scores = ks @ q / math.sqrt(d_attn)
        alpha = torch.softmax(scores, dim=0)
        return alpha @ history

    def _adam_update(self, param, grad, exp_avg, exp_avg_sq, t, lr_meta, beta1=0.9, beta2=0.999, eps=1e-8):
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

                state["param_before"] = p.detach().clone()
                state["exp_avg_before"] = state["exp_avg"].detach().clone()
                state["exp_avg_sq_before"] = state["exp_avg_sq"].detach().clone()

                history_list = state["grad_history"][:K]

                # Compute m_tilde from past-only history. Keep the graph alive so
                # meta_step can differentiate the virtual update w.r.t. W_Q/W_K.
                with torch.enable_grad():
                    if history_list:
                        history = torch.stack(history_list, dim=0)
                        m_past = self._attend(g_flat, history, W_Q, W_K, d_attn)
                        m_tilde_flat = (1.0 - mix_beta) * g_flat + mix_beta * m_past
                    else:
                        m_tilde_flat = g_flat

                m_tilde = m_tilde_flat.detach().reshape_as(p)

                m = state["exp_avg"]
                m.mul_(beta1).add_(m_tilde, alpha=1.0 - beta1)
                m_hat = m / (1.0 - beta1 ** t)

                v = state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(m_tilde, m_tilde, value=1.0 - beta2)
                v_hat = v / (1.0 - beta2 ** t)

                state["last_m_tilde"] = m_tilde_flat

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.addcdiv_(
                    m_hat.to(p.dtype),
                    v_hat.sqrt().add_(eps),
                    value=-lr,
                )

                state["grad_history"] = (
                    [g_flat.detach().clone()] + state["grad_history"]
                )[:K]

        return loss

    def meta_step(self, model, val_x, val_y):
        """
        Update W_Q/W_K using val loss after a differentiable reconstruction of
        the train-step update. This is the real bilevel path: val loss is
        evaluated at theta' built from pre-step theta/state plus last_m_tilde.
        """
        name_to_param = dict(model.named_parameters())
        param_to_name = {id(param): name for name, param in name_to_param.items()}
        buffers = dict(model.named_buffers())
        virtual_params = {}

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                state = self.state[p]
                if (
                    len(state) == 0
                    or state["last_m_tilde"] is None
                    or state["param_before"] is None
                ):
                    continue

                t = state["step"]
                m_tilde_flat = state["last_m_tilde"]
                m_tilde = m_tilde_flat.reshape_as(p)

                m_prev = state["exp_avg_before"]
                v_prev = state["exp_avg_sq_before"]
                p_before = state["param_before"]

                m_new = beta1 * m_prev + (1.0 - beta1) * m_tilde
                m_hat = m_new / (1.0 - beta1 ** t)
                v_new = beta2 * v_prev + (1.0 - beta2) * (m_tilde * m_tilde)
                v_hat = v_new / (1.0 - beta2 ** t)

                update = m_hat / (v_hat.sqrt().add(eps))
                if wd != 0.0:
                    p_base = p_before * (1.0 - lr * wd)
                else:
                    p_base = p_before
                p_prime = p_base - lr * update.to(p.dtype)

                param_name = param_to_name.get(id(p))
                if param_name is not None:
                    virtual_params[param_name] = p_prime

        if not virtual_params:
            return

        with torch.enable_grad():
            full_param_dict = {}
            for name, param in name_to_param.items():
                full_param_dict[name] = virtual_params.get(name, param)
            _, val_loss = functional_call(model, (full_param_dict, buffers), (val_x, val_y))
            val_loss.backward()

        # Update W_Q/W_K
        for group in self.param_groups:
            lr_meta = group["lr_meta"]
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    continue
                t = state["step"]
                W_Q, W_K = state["W_Q"], state["W_K"]
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
