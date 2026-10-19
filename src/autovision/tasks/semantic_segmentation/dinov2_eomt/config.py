# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Arguments for EOMT semantic segmentation training (DINOv2 variant)."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from autovision.augment.transform_args import NormalizeArgs
from autovision.tasks.train_model import TrainModelArgs

__all__ = ["Dinov2EomtSemanticSegmentationTrainArgs"]


class Dinov2EomtSemanticSegmentationTrainArgs(TrainModelArgs):
    """Training arguments for the EOMT semantic segmentation task.

    The whole encoder-only mask transformer trains end to end (EOMT has no
    frozen backbone); the head/layout defaults follow the DINOv2-small
    ADE20k recipe (100 queries, 4 interaction blocks, layer decay 0.9).
    """

    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = 160_000

    num_classes: int
    image_size: int = 512
    num_queries: int = 100
    num_blocks: int = 4
    num_upscale_blocks: int = 2
    hidden_size: int = 384
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    patch_size: int = 16
    mlp_ratio: int = 4
    use_swiglu_ffn: bool = False
    drop_path_rate: float = 0.0
    load_weights: bool = False
    weight_source: str | None = None

    ignore_index: int = 255
    lr: float = 1e-4
    optimizer_betas: tuple[float, float] = Field(default=(0.9, 0.999), strict=False)
    weight_decay: float = 0.01
    layer_decay: float = 0.9
    lr_warmup_steps: int | Literal["auto"] = "auto"
    horizontal_flip_prob: float = 0.5
    scale_jitter: tuple[float, float] = Field(default=(0.5, 2.0), strict=False)
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)

    model_config = {"extra": "forbid"}

    def resolve_auto_fields(self, **kwargs) -> None:
        total_steps = kwargs.get("total_steps", 0)
        if self.lr_warmup_steps == "auto":
            self.lr_warmup_steps = max(10, int(total_steps * 0.05)) if total_steps else 10
