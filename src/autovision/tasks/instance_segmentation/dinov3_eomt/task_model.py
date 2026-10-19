# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT instance segmentation task model, DINOv3 variant."""

from __future__ import annotations

from typing import Any

from autovision.tasks.instance_segmentation.dinov2_eomt.task_model import (
    Dinov2EomtInstanceSegmentation,
)
from autovision.tasks.semantic_segmentation.dinov3_eomt.task_model import (
    Dinov3EomtSemanticSegmentation,
    _Dinov3EomtCheckpointMixin,
)

__all__ = ["Dinov3EomtInstanceSegmentation"]


class Dinov3EomtInstanceSegmentation(
    _Dinov3EomtCheckpointMixin, Dinov2EomtInstanceSegmentation
):
    """DINOv3 backbone/head with the instance decode: queries become
    (label, score, binary mask) triplets."""

    def __init__(
        self,
        *args: Any,
        score_threshold: float = 0.05,
        mask_threshold: float = 0.5,
        num_top_queries: int = 100,
        **kwargs: Any,
    ) -> None:
        # Route the model build through the DINOv3 semantic base; calling
        # super().__init__ would chain into the DINOv2 task-model classes.
        Dinov3EomtSemanticSegmentation.__init__(self, *args, **kwargs)
        self.score_threshold = float(score_threshold)
        self.mask_threshold = float(mask_threshold)
        self.num_top_queries = int(num_top_queries)
        self._init_args.update(
            {
                "score_threshold": self.score_threshold,
                "mask_threshold": self.mask_threshold,
                "num_top_queries": self.num_top_queries,
            }
        )
