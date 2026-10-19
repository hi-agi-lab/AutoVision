# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Image classification task model package."""

from autovision.tasks.classification.config import (
    ImageClassificationTrainArgs,
)
from autovision.tasks.classification.task_model import (
    ImageClassification,
)
from autovision.tasks.classification.train_model import (
    ImageClassificationTrain,
)

__all__ = [
    "ImageClassification",
    "ImageClassificationTrain",
    "ImageClassificationTrainArgs",
]
