# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT instance segmentation task model (inference/export facing)."""

from __future__ import annotations

from typing import Any

from autovision.tasks.semantic_segmentation.dinov2_eomt.task_model import (
    Dinov2EomtSemanticSegmentation,
)
from autovision.tasks.shared.eomt.postprocess import instance_segmentation_decode

__all__ = ["Dinov2EomtInstanceSegmentation"]


class Dinov2EomtInstanceSegmentation(Dinov2EomtSemanticSegmentation):
    """Same EOMT backbone/head as the semantic variant; instance decode
    turns queries into (label, score, binary mask) triplets."""

    def __init__(self, *args: Any, score_threshold: float = 0.05,
                 mask_threshold: float = 0.5, num_top_queries: int = 100, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
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
        return instance_segmentation_decode(
            masks_logits,
            class_logits,
            metadata,
            score_threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            num_top_queries=self.num_top_queries,
        )
