# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pretraining method base class.

A method owns the self-supervised loss: it declares its args class, its
transform, the modules to optimize, and implements ``training_step_impl``.
The base class provides the LR scaling + scheduler wiring, gradient
clipping, and checkpoint hooks. The engine (not Lightning) drives the loop.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from autovision.augment.method_transform import MethodTransform
from autovision.optim import OptimizationTargets, get_optimizer
from autovision.optim.schedulers import CosineWarmupScheduler

__all__ = ["PretextRecipe", "StepOutput", "prepare_teacher"]


@dataclass
class StepOutput:
    """Result of one training step: the summed loss plus log values.

    Multi-term methods sum their loss components inside
    ``training_step_impl``; components are reported via ``log_dict`` only.
    """

    loss: Tensor
    log_dict: Mapping[str, float] | None = None


def prepare_teacher(module: nn.Module) -> nn.Module:
    """Freeze an EMA teacher: eval mode, no gradients, no stochastic dropout.

    ``make_teacher`` (when the backbone wrapper provides it) zeroes
    dropout/drop-path in place: ``eval()`` alone only keeps them dormant,
    and any later ``train()`` call would silently re-enable them. Works on
    ``FeatureExtractor`` instances (delegates through ``_inner``), bare
    backbone wrappers, and plain heads (no-op for the dropout part).
    """
    module.eval()
    owner = getattr(module, "_inner", module)
    make_teacher = getattr(owner, "make_teacher", None)
    if make_teacher is not None:
        make_teacher()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


class PretextRecipe(nn.Module):
    """Base class for all pretraining methods."""

    # Methods with unstable gradients set a norm value; None disables clipping.
    gradient_clip_val: float | None = None

    # Updated by the engine: the global optimizer step and the total step
    # count of the run. Methods use them for schedules (e.g. DINO momentum).
    global_step: int = 0
    total_steps: int = 0

    def __init__(
        self,
        method_args: Any,
        optimizer_args: Any,
        embedding_model: Any,
        global_batch_size: int,
        num_input_channels: int,
    ) -> None:
        super().__init__()
        self.method_args: Any = None  # subclasses assign the typed args
        self.optimizer_args = optimizer_args
        self.embedding_model = embedding_model
        self.global_batch_size = global_batch_size
        self.num_input_channels = num_input_channels
        self._last_log_dict: Mapping[str, float] | None = None

    # ── hooks subclasses must implement ─────────────────────────────────
    @staticmethod
    def method_args_cls() -> type:
        raise NotImplementedError

    @staticmethod
    def optimizer_args_cls(optim_type: str) -> type:
        """Optimizer args class for the given optimizer type ("auto" allowed)."""
        from autovision.optim import AdamWArgs, get_optimizer_args_cls

        if optim_type == "auto":
            return AdamWArgs
        return get_optimizer_args_cls(optim_type)

    @staticmethod
    def transform_cls() -> type[MethodTransform]:
        raise NotImplementedError

    def trainable_modules(self) -> OptimizationTargets:
        raise NotImplementedError

    def training_step_impl(self, batch: dict, batch_idx: int) -> StepOutput:
        raise NotImplementedError

    # ── concrete base behavior ──────────────────────────────────────────
    def forward(self, batch: dict, batch_idx: int = 0) -> Tensor:
        """Alias of :meth:`training_step` so ``DistributedDataParallel`` can
        wrap the method directly (DDP synchronizes gradients on ``forward``).
        """
        return self.training_step(batch, batch_idx)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        result = self.training_step_impl(batch, batch_idx)
        self._last_log_dict = result.log_dict
        return result.loss

    def consume_log_dict(self) -> Mapping[str, float] | None:
        """Return and clear the log dict of the most recent training step."""
        log_dict = self._last_log_dict
        self._last_log_dict = None
        return log_dict

    def configure_optimizers(
        self, total_steps: int, num_epochs: int
    ) -> tuple[torch.optim.Optimizer, CosineWarmupScheduler]:
        """LR scaled by global batch size; linear warmup + cosine decay."""
        lr_scale: float = self.global_batch_size / self.method_args.reference_batch_size
        if self.method_args.lr_scale_method == "sqrt":
            lr_scale = math.sqrt(lr_scale)
        optimizer = get_optimizer(
            optim_args=self.optimizer_args,
            trainable_modules=self.trainable_modules(),
            lr_scale=lr_scale,
        )
        max_epochs = max(1, num_epochs)
        # Warm up for 10 epochs, or 10% of the total if fewer than 100 epochs.
        warmup_epochs = min(10, max_epochs / 10)
        warmup_steps = min(
            int(total_steps), int(total_steps / max_epochs * warmup_epochs)
        )
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer, warmup_epochs=warmup_steps, max_epochs=int(total_steps)
        )
        return optimizer, scheduler

    def clip_gradients(self, optimizer: torch.optim.Optimizer) -> float | None:
        """Clip by norm; returns the pre-clip total norm (None if disabled)."""
        if self.gradient_clip_val is None:
            return None
        params = [p for group in optimizer.param_groups for p in group["params"]]
        return float(
            torch.nn.utils.clip_grad_norm_(params, self.gradient_clip_val)
        )

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Per-step hook before ``optimizer.step()`` (e.g. weight-decay
        schedules or temporary LR freezes); default is a no-op."""

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Mutate the checkpoint dict before saving (e.g. strip teachers)."""

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Mutate the checkpoint dict after loading (e.g. restore teachers)."""
