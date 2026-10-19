# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT panoptic segmentation task model (inference/export facing)."""

from __future__ import annotations

from typing import Any

import torch

from autovision.tasks.semantic_segmentation.dinov2_eomt.task_model import (
    Dinov2EomtSemanticSegmentation,
)
from autovision.tasks.shared.eomt.postprocess import panoptic_segmentation_decode

__all__ = ["Dinov2EomtPanopticSegmentation"]


class Dinov2EomtPanopticSegmentation(Dinov2EomtSemanticSegmentation):
    """Same EOMT backbone/head as the semantic variant; the panoptic decode
    turns queries into a per-pixel (class, segment) map plus per-segment
    scores. Unassigned pixels surface as the last user class (the null
    class slot remaps there), matching the baseline convention."""

    def __init__(
        self,
        *args: Any,
        num_thing_classes: int,
        threshold: float = 0.8,
        mask_threshold: float = 0.5,
        mask_overlap_threshold: float = 0.8,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
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

    def postprocess(
        self,
        raw_outputs: Any,
        metadata: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        del kwargs
        if isinstance(raw_outputs, tuple):
            masks_logits, class_logits = raw_outputs
        else:
            masks_logits = raw_outputs.masks_queries_logits
            class_logits = raw_outputs.class_queries_logits
        results = panoptic_segmentation_decode(
            masks_logits,
            class_logits,
            metadata,
            num_labels=self.num_classes,
            num_thing_classes=self.num_thing_classes,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            mask_overlap_threshold=self.mask_overlap_threshold,
        )
        for result in results:
            result["masks"][..., 0] = self.internal_class_to_class[
                result["masks"][..., 0]
            ]
        return results
