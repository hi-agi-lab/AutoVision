# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Arguments for linear semantic segmentation training."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from autovision.augment.transform_args import NormalizeArgs
from autovision.tasks.train_model import TrainModelArgs
from autovision.types import ImageSizeTuple

__all__ = ["LinearSemanticSegmentationTrainArgs"]


class LinearSemanticSegmentationTrainArgs(TrainModelArgs):
    """Training arguments for the linear segmentation task.

    The backbone stays frozen (linear probing on pretrained features); only
    the 1x1 classification head trains.
    """

    default_batch_size: ClassVar[int] = 32
    default_steps: ClassVar[int] = 20_000

    num_classes: int
    backbone: str = "torchvision/resnet18"
    image_size: ImageSizeTuple = (512, 512)
    ignore_index: int = 255
    lr: float = 0.001
    optimizer_betas: tuple[float, float] = Field(default=(0.9, 0.999), strict=False)
    weight_decay: float = 0.0001
    scheduler: Literal["cosine"] = "cosine"
    lr_warmup_steps: int | Literal["auto"] = "auto"
    horizontal_flip_prob: float = 0.5
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)

    model_config = {"extra": "forbid"}

    def resolve_auto_fields(self, **kwargs) -> None:
        total_steps = kwargs.get("total_steps", 0)
        if self.lr_warmup_steps == "auto":
            self.lr_warmup_steps = max(10, int(total_steps * 0.05)) if total_steps else 10
