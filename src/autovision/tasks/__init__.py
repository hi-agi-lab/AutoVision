# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Task model layer: bases, per-task family packages, architecture cores.

First level = task type (detection, oriented_detection, segmentation
tasks, classification, depth/pose scaffolds); second level = model family.
``eomt/`` and ``shared/`` hold architecture cores / training infrastructure
spanning multiple tasks (design doc 30).
"""

from autovision.tasks.detection.rtdetr import (
    RTDETRObjectDetection,
    RTDETRObjectDetectionTrain,
    RTDETRObjectDetectionTrainArgs,
)
from autovision.tasks.task_model import TaskNetwork
from autovision.tasks.task_model_io import (
    BaseModelOutput,
    ModelInputSpec,
    TensorSpec,
)
from autovision.tasks.train_model import TaskStepResult, TrainModel

__all__ = [
    "BaseModelOutput",
    "ModelInputSpec",
    "RTDETRObjectDetection",
    "RTDETRObjectDetectionTrain",
    "RTDETRObjectDetectionTrainArgs",
    "TaskNetwork",
    "TaskStepResult",
    "TensorSpec",
    "TrainModel",
]
