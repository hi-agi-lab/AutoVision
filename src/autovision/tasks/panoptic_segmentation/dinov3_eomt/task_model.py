# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT panoptic segmentation task model, DINOv3 variant."""

from __future__ import annotations

from typing import Any

import torch

from autovision.tasks.panoptic_segmentation.dinov2_eomt.task_model import (
    Dinov2EomtPanopticSegmentation,
)
from autovision.tasks.semantic_segmentation.dinov3_eomt.task_model import (
    Dinov3EomtSemanticSegmentation,
    _Dinov3EomtCheckpointMixin,
)

__all__ = ["Dinov3EomtPanopticSegmentation"]


class Dinov3EomtPanopticSegmentation(
    _Dinov3EomtCheckpointMixin, Dinov2EomtPanopticSegmentation
):
    """DINOv3 backbone/head with the panoptic decode: queries become a
    per-pixel (class, segment) map plus per-segment scores."""

    def __init__(
        self,
        *args: Any,
        num_thing_classes: int,
        threshold: float = 0.8,
        mask_threshold: float = 0.5,
        mask_overlap_threshold: float = 0.8,
        **kwargs: Any,
    ) -> None:
        # Route the model build through the DINOv3 semantic base; calling
        # super().__init__ would chain into the DINOv2 task-model classes.
        Dinov3EomtSemanticSegmentation.__init__(self, *args, **kwargs)
        self.num_thing_classes = int(num_thing_classes)
        self.threshold = float(threshold)
        self.mask_threshold = float(mask_threshold)
        self.mask_overlap_threshold = float(mask_overlap_threshold)
        self._init_args.update(
            {
                "num_thing_classes": self.num_thing_classes,
                "threshold": self.threshold,
                "mask_threshold": self.mask_threshold,
                "mask_overlap_threshold": self.mask_overlap_threshold,
            }
        )
        # Internal class ids are contiguous (things first, then stuffs);
        # the ignore slot (== num_classes) surfaces as the last user class.
        self.register_buffer(
            "internal_class_to_class",
            torch.tensor(
                list(range(self.num_classes)) + [self.num_classes - 1],
                dtype=torch.long,
            ),
            persistent=False,
        )
