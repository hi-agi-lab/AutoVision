# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""TrainModel base class: training wrapper around a TaskNetwork."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
from torch import nn

from autovision.config.base import PydanticConfig

__all__ = ["TaskStepResult", "TrainModel", "TrainModelArgs"]


class TrainModelArgs(PydanticConfig):
    """Base for per-model training args: class-level defaults drive the
    ``steps="auto"``/``batch_size="auto"`` resolution."""

    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = 10_000

    model_config = {"extra": "forbid"}

    def resolve_auto_fields(self, **kwargs: Any) -> None:
        """Resolve any auto fields."""


@dataclass
class TaskStepResult:
    """Result of one training/validation step.

    ``log_dict`` values may be float or detached tensors: the engine only
    converts them to floats at logging time (every ``log_every_n_steps``),
    so per-step GPU->CPU syncs stay off the hot path.
    """

    loss: torch.Tensor
    log_dict: dict[str, float | torch.Tensor] = field(default_factory=dict)


class TrainModel(nn.Module):
    """Wraps a TaskNetwork with criterion, metrics, and parameter groups.

    PretextRecipe names intentionally match the Lightning convention so a future
    Fabric-compatible engine can dispatch transparently. The engine drives
    backward/optimizer; ``training_step`` returns the loss.

    ``forward`` aliases :meth:`training_step` so ``DistributedDataParallel``
    can wrap the TrainModel directly (DDP synchronizes gradients through the
    forward call).

    Class variables:

    - ``watch_metric``: validation metric key that drives best-checkpoint
      selection (engine falls back to ``val_metric/map``, then ``val_loss``).
    """

    task: ClassVar[str] = ""
    train_model_args_cls: ClassVar[type[TrainModelArgs]] = TrainModelArgs
    watch_metric: ClassVar[str | None] = None

    def __init__(self, task_model: TaskModelProtocol) -> None:
        super().__init__()
        self.model = task_model

    def get_task_model(self):
        return self.model

    def _apply(self, fn, recurse: bool = True):  # type: ignore[override]
        # Keep the EMA copy on the same device as the training model.
        if getattr(self, "ema_model", None) is not None:
            self.ema_model.model._apply(fn, recurse=recurse)
        return super()._apply(fn, recurse=recurse)

    def forward(self, batch: dict, step: int = 0) -> TaskStepResult:
        """Alias of :meth:`training_step` (DDP entry point)."""
        return self.training_step(batch, step)

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        raise NotImplementedError

    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        raise NotImplementedError

    def get_optimizer(
        self, total_steps: int, global_batch_size: int
    ) -> tuple[torch.optim.Optimizer, Any]:
        raise NotImplementedError

    def get_export_state_dict(self) -> dict[str, Any]:
        """State dict for the exported model: EMA weights when present."""
        state = self.state_dict()
        ema_keys = [key for key in state if key.startswith("ema_model.")]
        if ema_keys:
            return {key: state[key] for key in ema_keys}
        return state

    def clip_gradients(self, optimizer) -> float | None:
        return None  # no clipping by default

    def on_train_batch_end(self) -> None:
        """Per-step hook (EMA update)."""

    def enable_gradient_checkpointing(self) -> int:
        """Trade compute for activation memory; return patched module count.

        HF backbones use their native implementation; custom ViT/GTR
        backbones have their ``blocks``/``layers`` wrapped. Safe to call
        twice (patched blocks are tracked).
        """
        from autovision.tasks.shared.gradient_checkpointing import (
            enable_gradient_checkpointing,
        )

        return enable_gradient_checkpointing(self)

    def update_metrics(self, outputs: Any, batch: dict, split: str) -> None:
        """Feed predictions/targets to the metric collection."""

    def compute_metrics(self, split: str) -> dict[str, float]:
        return {}

    def reset_metrics(self, split: str) -> None:
        pass

    # ── distributed validation support ──────────────────────────────────
    def export_metric_state(self, split: str) -> dict | None:
        """Snapshot the accumulated metric state for cross-rank gathering."""
        return None

    def merge_metric_state(self, split: str, state: dict | None) -> None:
        """Merge a gathered metric state snapshot into this rank's collector."""


class TaskModelProtocol:
    """Structural placeholder for typing: a TaskNetwork instance."""
