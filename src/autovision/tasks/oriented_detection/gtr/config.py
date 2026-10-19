# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Training arguments for the GTR oriented-detection family (doc 31 §11 S3b)."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from autovision.augment.transform_args import NormalizeArgs
from autovision.tasks.train_model import TrainModelArgs

__all__ = ["GTRROBBTrainArgs"]


class GTRROBBTrainArgs(TrainModelArgs):
    """Official OBB finetune defaults (``gtrobb_base`` + S config).

    Losses ``['mal', 'boxes']`` at the official weights; matcher costs
    class 2 / chamfer 5 / KLD 2; lr 5e-4 with the backbone at 0.36x
    (official 1.8e-4 / 5e-4).
    """

    default_batch_size: ClassVar[int] = 8
    default_steps: ClassVar[int] = 20_000

    num_classes: int = 15  # DOTA-v1.0
    variant: Literal["s", "x"] = "s"
    image_size: tuple[int, int] = (1024, 1024)  # official OBB canvas
    num_queries: int = 300
    num_denoising: int = 100
    checkpoint: str | None = None
    load_checkpoint: bool = True
    losses: list[str] = Field(default_factory=lambda: ["mal", "boxes"])
    loss_weight_dict: dict[str, float] = Field(
        default_factory=lambda: {
            "loss_mal": 1.0,
            "loss_bbox": 5.0,
            "loss_kld": 5.0,
            "loss_fgl": 0.15,
        }
    )
    cost_class: float = 2.0
    cost_chamfer: float = 5.0
    cost_kld: float = 2.0
    lr: float = 5e-4
    backbone_lr_factor: float = 0.36
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
                max(10, int(total_steps * 0.1)) if total_steps else 10
            )
