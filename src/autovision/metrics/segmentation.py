# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Semantic segmentation metrics: confusion-matrix mIoU and pixel accuracy."""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["SegmentationMetricCollector"]


class SegmentationMetricCollector:
    """mIoU + pixel accuracy over an accumulated confusion matrix.

    Label 255 marks ignored pixels (cityscapes convention) and never counts
    towards either metric.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch: class-index maps of shape (B, H, W)."""
        predictions = predictions.detach().cpu().long().flatten()
        targets = targets.detach().cpu().long().flatten()
        valid = targets != self.ignore_index
        predictions = predictions[valid]
        targets = targets[valid]
        # Drop predictions of out-of-range classes defensively.
        in_range = (predictions >= 0) & (predictions < self.num_classes)
        predictions = predictions[in_range]
        targets = targets[in_range]
        indices = targets * self.num_classes + predictions
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        self._matrix += counts.reshape(self.num_classes, self.num_classes)

    def reset(self) -> None:
        self._matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )

    def compute(self) -> dict[str, float]:
        matrix = self._matrix.float()
        intersection = matrix.diag()
        union = matrix.sum(dim=0) + matrix.sum(dim=1) - intersection
        present = union > 0
        if not present.any():
            return {"miou": 0.0, "pixel_accuracy": 0.0}
        miou = float((intersection[present] / union[present]).mean())
        total = matrix.sum()
        pixel_accuracy = float(intersection.sum() / total) if total > 0 else 0.0
        return {"miou": miou, "pixel_accuracy": pixel_accuracy}

    # ── distributed validation support ──────────────────────────────────
    def export_state(self) -> dict[str, Any]:
        return {"matrix": self._matrix.clone()}

    def merge_state(self, state: dict[str, Any]) -> None:
        self._matrix += state["matrix"].to(self._matrix.dtype)
