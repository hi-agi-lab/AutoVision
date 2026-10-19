# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""LARS optimizer (You et al., 2017, "Large Batch Training of Convolutional
Networks").

Layer-wise adaptive rate scaling: each layer's update is scaled by a local
trust ratio ``|w| / |g + wd*w|`` times the trust coefficient. Self-contained
(no third-party optimizer dependency).
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

__all__ = ["LARS"]


class LARS(Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening = 0")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            trust_coefficient=trust_coefficient,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            trust_coefficient = group["trust_coefficient"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LARS does not support sparse gradients")

                # Local trust ratio: |w| / (|g| + wd*|w| + eps), gated on
                # both norms being non-zero (identity update otherwise).
                p_norm = torch.norm(p)
                g_norm = torch.norm(grad)
                if p_norm != 0 and g_norm != 0:
                    update = grad.add(p, alpha=weight_decay)
                    local_lr = trust_coefficient * p_norm / (
                        update.norm() + eps
                    )
                    update.mul_(local_lr)
                else:
                    update = grad.clone()

                if momentum != 0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.clone(update).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(update, alpha=1 - dampening)
                    if nesterov:
                        update = update.add(buf, alpha=momentum)
                    else:
                        update = buf

                p.add_(update, alpha=-group["lr"])
        return loss
