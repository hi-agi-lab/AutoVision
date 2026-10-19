# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Arguments for image classification training."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field

from autovision.augment.transform_args import NormalizeArgs
from autovision.tasks.train_model import TrainModelArgs
from autovision.types import ImageSizeTuple

__all__ = ["ImageClassificationTrainArgs"]


class ImageClassificationTrainArgs(TrainModelArgs):
    """Training arguments for the image classification task.

    Defaults follow the linear-probing recipe: frozen backbone, AdamW on the
    classification head. Set ``freeze_backbone=False`` to finetune the whole
    network (the backbone then trains with ``backbone_lr_factor`` × lr).
    """

    default_batch_size: ClassVar[int] = 128
    default_steps: ClassVar[int] = 10_000

    num_classes: int
    backbone: str = "torchvision/resnet18"
    image_size: ImageSizeTuple = (224, 224)
    freeze_backbone: bool = True
    lr: float = 0.001
    optimizer_betas: tuple[float, float] = Field(default=(0.9, 0.999), strict=False)
    weight_decay: float = 0.0001
    backbone_lr_factor: float = 0.1
    scheduler: Literal["cosine", "flat-cosine"] = "cosine"
    lr_warmup_steps: int | Literal["auto"] = "auto"
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)

    model_config = {"extra": "forbid"}

    def resolve_auto_fields(self, **kwargs: Any) -> None:
        """Resolve any auto fields."""
        total_steps = kwargs.get("total_steps", 0)
        if self.lr_warmup_steps == "auto":
            # Warm up for at most 5% of the run, at least 10 steps.
            self.lr_warmup_steps = max(10, int(total_steps * 0.05)) if total_steps else 10
