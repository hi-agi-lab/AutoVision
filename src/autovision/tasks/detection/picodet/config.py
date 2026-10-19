# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""PicoDet object detection: model configuration."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from autovision.config.base import PydanticConfig
from autovision.tasks.train_model import TrainModelArgs
from autovision.types import ImageSizeTuple

__all__ = ["PicoDetObjectDetectionTrainArgs"]


class PicoDetObjectDetectionTrainArgs(TrainModelArgs, PydanticConfig):
    """Training/architecture args for the PicoDet detection family."""

    model_config = {"extra": "forbid"}

    default_batch_size: ClassVar[int] = 32
    default_steps: ClassVar[int] = 12_000

    # Width scale alias (xs/s/m/l or float string), resolved by
    # ``resolve_backbone_scale``; carried over from "picodet/<name>".
    backbone: str = "s"
    num_classes: int = 80
    image_size: ImageSizeTuple = (640, 640)
    neck_channels: int = 96
    reg_max: int = 7

    # Dynamic label assignment + GFL-family loss weights (port defaults).
    assigner: Literal["simota", "taskaligned"] = "simota"
    assigner_topk: int = 10
    loss_cls_weight: float = 1.0
    loss_giou_weight: float = 2.0
    loss_dfl_weight: float = 0.25

    # Eval decode: top-k candidates per image, then class-offset NMS.
    num_top_queries: int = 300
    nms_iou_threshold: float = 0.6

    # optimizer / schedule — SGD recipe (linear lr scaling with the global
    # batch size; 0.2 at the framework's default_batch_size=32).
    lr: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    gradient_clip_val: float = 10.0
    lr_warmup_steps: int | Literal["auto"] = "auto"
    scheduler_flat_steps: int | Literal["auto"] = "auto"
    scheduler_no_aug_steps: int | Literal["auto"] = "auto"

    # EMA
    use_ema_model: bool = True
    ema_momentum: float = 0.9998
    ema_warmup_steps: int = 2000

    def resolve_auto_fields(
        self, total_steps: int, train_num_batches: int, gradient_accumulation_steps: int
    ) -> None:
        from autovision.tasks.shared.flat_cosine import (
            resolve_flat_cosine_windows,
        )

        warmup, flat, no_aug = resolve_flat_cosine_windows(
            total_steps, train_num_batches, gradient_accumulation_steps
        )
        if self.lr_warmup_steps == "auto":
            self.lr_warmup_steps = warmup
        if self.scheduler_flat_steps == "auto":
            self.scheduler_flat_steps = flat
        if self.scheduler_no_aug_steps == "auto":
            self.scheduler_no_aug_steps = no_aug

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
