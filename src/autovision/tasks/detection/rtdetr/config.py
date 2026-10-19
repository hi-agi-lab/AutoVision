# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RT-DETR object detection: model configuration."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from autovision.config.base import PydanticConfig
from autovision.tasks.train_model import TrainModelArgs
from autovision.types import ImageSizeTuple

__all__ = ["RTDETRObjectDetectionTrainArgs"]


class RTDETRObjectDetectionTrainArgs(TrainModelArgs, PydanticConfig):
    """Training/architecture args for the RT-DETR detection family."""

    model_config = {"extra": "forbid"}

    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = 12_000

    backbone: str = "torchvision/resnet18"
    backbone_pretrained: bool = False
    num_classes: int = 80
    image_size: ImageSizeTuple = (640, 640)
    hidden_dim: int = 256
    num_queries: int = 100
    dec_layers: int = 3
    num_heads: int = 8
    # Eval decode keeps the top-300 candidates per image (baseline decode
    # width); 100 caps recall on dense datasets like VisDrone.
    num_top_queries: int = 300
    # COCO eval per-image detection cap. Dense datasets (VisDrone) want 500
    # — parity with the detr/rfdetr families' eval_max_dets.
    eval_max_dets: int = 100

    # Contrastive denoising (CDN): noised GT queries accelerate early box
    # learning. Counts/factors mirror the baseline RT-DETRv2 recipe.
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0

    # optimizer / schedule — lr follows the baseline RT-DETRv2 recipe
    # (5e-4 at the framework's default_batch_size=16, sqrt-scaled with the
    # global batch size).
    lr: float = 5e-4
    backbone_lr_factor: float = 0.05
    weight_decay: float = 1e-4
    gradient_clip_val: float = 0.1
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    lr_warmup_steps: int | Literal["auto"] = "auto"
    scheduler_flat_steps: int | Literal["auto"] = "auto"
    scheduler_no_aug_steps: int | Literal["auto"] = "auto"

    # EMA
    use_ema_model: bool = True
    ema_momentum: float = 0.9999
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
