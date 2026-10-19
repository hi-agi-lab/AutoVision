# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Object detection architecture core shared by detector families.

RT-DETR/DFINE lineage: encoder, decoders, criteria, matcher, denoising,
distillation/SSL heads. Pure box math lives in :mod:`autovision.geometry`;
cross-task training infrastructure in :mod:`autovision.tasks.shared`.
"""

from autovision.tasks.detection.shared.hybrid_encoder import (
    HybridEncoder,
)
from autovision.tasks.detection.shared.matcher import (
    HungarianMatcher,
)
from autovision.tasks.detection.shared.rtdetrv2_criterion import (
    RTDETRCriterionV2,
)
from autovision.tasks.detection.shared.rtdetrv2_decoder import (
    RTDETRv2Decoder,
    inverse_sigmoid,
)

__all__ = [
    "HungarianMatcher",
    "HybridEncoder",
    "RTDETRCriterionV2",
    "RTDETRv2Decoder",
    "inverse_sigmoid",
]
