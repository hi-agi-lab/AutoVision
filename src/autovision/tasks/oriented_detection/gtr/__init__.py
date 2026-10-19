# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR oriented-detection task model package (roadmap doc 31 §11 S3).

Inference/load slice: :class:`GTRROBBOrientedDetection` loads the official
MIT ``gtrobb_*_dota`` checkpoints and predicts rotated polygons. Training
(oriented criterion/matcher/oriented-CDN) is roadmap §11 **S3b**.
"""

from autovision.tasks.oriented_detection.gtr.config import GTRROBBTrainArgs
from autovision.tasks.oriented_detection.gtr.task_model import (
    GTR_OBB_VARIANTS,
    GTRROBBOrientedDetection,
)
from autovision.tasks.oriented_detection.gtr.train_model import GTRROBBTrain

__all__ = [
    "GTRROBBOrientedDetection",
    "GTRROBBTrain",
    "GTRROBBTrainArgs",
    "GTR_OBB_VARIANTS",
]
