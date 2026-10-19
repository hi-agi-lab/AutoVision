# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Arguments for GTR semantic segmentation training (roadmap doc 31 §11)."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from autovision.augment.transform_args import NormalizeArgs
from autovision.tasks.train_model import TrainModelArgs
from autovision.types import ImageSizeTuple

__all__ = ["GTRSemSegTrainArgs"]


class GTRSemSegTrainArgs(TrainModelArgs):
    """Training arguments for the GTR semantic-segmentation family.

    Architecture mirrors the official configs (``configs/semseg/
    cityscapes_finetune/gtrsemseg_{s,m,l,x}.yml`` + the shared base): the
    GTR ViT backbone taps → single-level (stride-8) ``GTREncoder`` → FCN
    head, loss = per-pixel CE with ``ignore_index``. The official recipe
    fine-tunes the backbone at ``lr * backbone_lr_factor`` (their
    ``1.5e-5 / 5e-4 = 0.03``).
    """

    default_batch_size: ClassVar[int] = 8
    default_steps: ClassVar[int] = 20_000

    num_classes: int = 19
    #: Backbone scale: official GTR-S/M/L/X widths.
    variant: Literal["s", "m", "l", "x"] = "s"
    image_size: ImageSizeTuple = (1024, 1024)
    ignore_index: int = 255
    dropout: float = 0.1
    #: Official MIT checkpoint (registered asset name or a file path);
    #: EMA weights are used (GTR README benchmark口径).
    checkpoint: str | None = None
    #: Skip the backbone-weights load at construction (tests / from scratch).
    load_checkpoint: bool = True
    lr: float = 5e-4
    backbone_lr_factor: float = 0.03
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
            # Official: warmup_iter 2000 within 30 epochs (110k iters).
            self.lr_warmup_steps = (
                max(10, int(total_steps * 0.02)) if total_steps else 10
            )
