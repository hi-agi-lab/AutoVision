# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""D-FINE object detection: model configuration."""

from __future__ import annotations

from autovision.tasks.detection.rtdetr.config import (
    RTDETRObjectDetectionTrainArgs,
)

__all__ = ["DFINEObjectDetectionTrainArgs"]


class DFINEObjectDetectionTrainArgs(RTDETRObjectDetectionTrainArgs):
    """RT-DETR family args + the D-FINE distributional refinement knobs.

    ``fgl_loss_weight=0.15`` / ``ddf_loss_weight=1.5`` follow the D-FINE
    paper's loss weighting; ``reg_max`` sets the per-side bin resolution
    (``2 × reg_max + 1`` bins over ``[-1, 1]`` reference-dimension units).
    """

    reg_max: int = 32
    fgl_loss_weight: float = 0.15
    ddf_loss_weight: float = 1.5
