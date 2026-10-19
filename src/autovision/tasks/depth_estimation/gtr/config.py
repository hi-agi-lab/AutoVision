# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Training arguments for the GTR metric-depth family (doc 31 §11 S4)."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from autovision.augment.transform_args import NormalizeArgs
from autovision.tasks.train_model import TrainModelArgs

__all__ = ["GTRDepthEstimationTrainArgs"]


class GTRDepthEstimationTrainArgs(TrainModelArgs):
    """Metric-depth training args (official pretrain recipe defaults).

    Loss: scale-invariant log (``silog_lambd`` 0.5, valid mask by depth
    range); lr: 2e-4 with backbone at half (official 1e-4 vs 2e-4).
    """

    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = 100_000

    variant: Literal["s", "m", "l", "x"] = "s"
    image_size: tuple[int, int] = (640, 640)
    features: int = 128
    checkpoint: str | None = None
    load_checkpoint: bool = True
    min_depth: float = 0.001
    max_depth: float = 10.0
    silog_lambd: float = 0.5
    lr: float = 2e-4
    backbone_lr_factor: float = 0.5
    optimizer_betas: tuple[float, float] = Field(default=(0.9, 0.999), strict=False)
    weight_decay: float = 0.0001
    scheduler: Literal["cosine"] = "cosine"
    lr_warmup_steps: int | Literal["auto"] = "auto"
    horizontal_flip_prob: float = 0.5
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    fused_optimizer: bool = False

    model_config = {"extra": "forbid"}

    def resolve_auto_fields(self, **kwargs) -> None:
        total_steps = kwargs.get("total_steps", 0)
        if self.lr_warmup_steps == "auto":
            self.lr_warmup_steps = (
                max(10, int(total_steps * 0.02)) if total_steps else 10
            )
