# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Task transform base: step hooks and the "auto" resolution contract.

Mirrors the baseline ``TaskTransform``/``TaskTransformArgs`` pair. Step
awareness enables the scheduled augmentation compose; dataloader
reinitialization propagates step-state into already-spawned workers.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from autovision.config.base import PydanticConfig

__all__ = ["TaskTransform", "TaskTransformArgs", "ActivationPolicyArgs"]


class TaskTransformArgs(PydanticConfig):
    """Base args for task transforms."""

    def resolve_auto_fields(self, **kwargs: Any) -> None:
        """Resolve any ``"auto"`` fields (model/geometry dependent)."""

    def resolve_step_schedule(
        self, total_steps: int, train_num_batches: int, gradient_accumulation_steps: int
    ) -> None:
        """Resolve the auto step windows from the training schedule."""

    def resolve_incompatible(self) -> None:
        """Disable transforms that conflict with the resolved config."""


@runtime_checkable
class TaskTransform(Protocol):
    """Callable item transform with a step hook."""

    transform_args: TaskTransformArgs

    def set_step(self, step: int) -> None: ...

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]: ...


class ActivationPolicyArgs(PydanticConfig):
    """Active-in-window policy: ``[step_start, step_stop)``."""

    step_start: int | str = 0
    step_stop: int | None | str = None

    def is_active(self, step: int) -> bool:
        if self.step_start == "auto":
            return False
        if step < int(self.step_start):
            return False
        if self.step_stop is None or self.step_stop == "auto":
            return True
        return step < int(self.step_stop)
