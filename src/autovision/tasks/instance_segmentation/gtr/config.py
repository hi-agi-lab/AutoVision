# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Arguments for GTR instance-segmentation training (roadmap doc 31 §11)."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from autovision.augment.transform_args import NormalizeArgs
from autovision.tasks.detection.detr.config import DETRObjectDetectionTrainArgs

__all__ = ["GTRSegTrainArgs"]


class GTRSegTrainArgs(DETRObjectDetectionTrainArgs):
    """DETR train args specialised for the GTR seg lineage.

    Mask supervision is on by default (official ``gtrseg_base`` recipe:
    point-sampled CE/Dice losses at 5.0/5.0, matcher costs 5.0/5.0, ratio
    16 — inherited field defaults). ``checkpoint`` loads the official MIT
    weights (registered asset name or path; EMA口径).
    """

    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = 30_000

    variant: Literal["s", "m", "l", "x"] = "s"
    checkpoint: str | None = None
    load_checkpoint: bool = True
    mask_downsample_ratio: int = 4
    #: Override the base default: the seg family trains masks.
    mask_losses: bool = True
    #: The instance-seg CLI's transform builder reads these (ImageNet
    #: defaults + flip); the DET-family base args do not carry them.
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    horizontal_flip_prob: float = 0.5
