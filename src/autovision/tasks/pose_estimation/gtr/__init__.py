# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR pose-estimation task model package (roadmap doc 31 §11 S5).

Load + inference (:class:`GTRPoseEstimation`) and the training surface
(:class:`GTRPoseEstimationTrain`, §11 S5b: official criterion/matcher/
CDN, mean-OKS validation). The COCO keypoint dataset surface and its
official AP metric remain data-gated follow-ups.
"""

from autovision.tasks.pose_estimation.gtr.config import (
    GTRPoseEstimationTrainArgs,
)
from autovision.tasks.pose_estimation.gtr.task_model import (
    GTR_POSE_VARIANTS,
    GTRPoseEstimation,
)
from autovision.tasks.pose_estimation.gtr.train_model import (
    GTRPoseEstimationTrain,
)

__all__ = [
    "GTRPoseEstimation",
    "GTRPoseEstimationTrain",
    "GTRPoseEstimationTrainArgs",
    "GTR_POSE_VARIANTS",
]
