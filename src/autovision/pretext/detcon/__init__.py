# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DetCon: masked region-contrastive pretraining (S = SupCon / B = BYOL)."""

from autovision.pretext.detcon.detcon import (
    DetConB,
    DetConBArgs,
    DetConBLoss,
    DetConBSGDArgs,
    DetConS,
    DetConSArgs,
    DetConSGDArgs,
    DetConSLoss,
)

__all__ = [
    "DetConB",
    "DetConBArgs",
    "DetConBLoss",
    "DetConBSGDArgs",
    "DetConS",
    "DetConSArgs",
    "DetConSGDArgs",
    "DetConSLoss",
]
