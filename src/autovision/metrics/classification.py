# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Classification metrics: top-k accuracy over accumulated predictions."""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["ClassificationMetricCollector"]


class ClassificationMetricCollector:
    """Top-1/top-5 accuracy over accumulated logits and labels.

    Logits arrive per validation batch on any device and are accumulated as
    full float32 CPU copies — classification validation sets are small, and
    this keeps ``export_state``/``merge_state`` (distributed evaluation)
    trivially picklable.
    """

    def __init__(self, top_k: tuple[int, ...] = (1, 5)) -> None:
        self.top_k = top_k
        self._logits: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        self._logits.append(logits.detach().float().cpu())
        self._labels.append(labels.detach().cpu())

    def reset(self) -> None:
        self._logits = []
        self._labels = []

    def compute(self) -> dict[str, float]:
        if not self._logits:
            return {f"accuracy_top{k}": 0.0 for k in self.top_k}
        logits = torch.cat(self._logits)
        labels = torch.cat(self._labels)
        max_k = min(max(self.top_k), logits.shape[1])
        _, topk_indices = logits.topk(max_k, dim=1)
        correct = topk_indices.eq(labels.unsqueeze(1))  # (N, max_k)
        result: dict[str, float] = {}
        for k in self.top_k:
            if k < 1:
                continue
            # Fewer classes than k: top-k degenerates to "always correct"
            # (torchmetrics convention).
            effective_k = min(k, max_k)
            result[f"accuracy_top{k}"] = float(
                correct[:, :effective_k].any(dim=1).float().mean()
            )
        return result

    # ── distributed validation support ──────────────────────────────────
    def export_state(self) -> dict[str, Any]:
        return {"logits": list(self._logits), "labels": list(self._labels)}

    def merge_state(self, state: dict[str, Any]) -> None:
        self._logits.extend(state["logits"])
        self._labels.extend(state["labels"])
