# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DETR object detection family (SSL-backbone detection finetuning)."""

from autovision.tasks.detection.detr.config import (
    DETR_MODEL_REGISTRY,
    DETRDensityArgs,
    DETRDistillArgs,
    DETRObjectDetectionTrainArgs,
    DETROpenVocabArgs,
    DETRSSLArgs,
    DETRTokenDropArgs,
    DETRVariantSpec,
    detr_default_batch_size,
    list_detr_model_names,
    resolve_detr_variant,
)
from autovision.tasks.detection.detr.dome_decoder import (
    DomeDFINEv2Decoder,
)
from autovision.tasks.detection.detr.dome_encoder import (
    DomeHybridEncoder,
)
from autovision.tasks.detection.detr.task_model import (
    DETRObjectDetection,
)
from autovision.tasks.detection.detr.train_model import (
    DETRObjectDetectionTrain,
)

__all__ = [
    "DETR_MODEL_REGISTRY",
    "DETRDensityArgs",
    "DETRDistillArgs",
    "DomeDFINEv2Decoder",
    "DomeHybridEncoder",
    "DETRObjectDetection",
    "DETRObjectDetectionTrain",
    "DETRObjectDetectionTrainArgs",
    "DETROpenVocabArgs",
    "DETRSSLArgs",
    "DETRTokenDropArgs",
    "DETRVariantSpec",
    "list_detr_model_names",
    "detr_default_batch_size",
    "resolve_detr_variant",
]
