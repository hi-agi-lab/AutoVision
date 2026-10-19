# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Monocular metric-depth metrics (AbsRel / RMSE / delta<1.25).

Standard Eigen-split protocol metrics over (prediction, ground-truth)
depth pairs in meters; ``alignment`` optionally removes per-image scale
bias before scoring:

- ``none``: raw comparison;
- ``median``: scale by median(gt / pred) (scale-invariant depth);
- ``log-affine``: per-image least-squares fit in log space
  (``min ||a log(pred) + b - log(gt)||`` — the Depth-Anything-V2 /
  GTR NYU protocol).

Ignore pixels outside ``[min_depth, max_depth]`` (padded / invalid
ground truth), mirroring the GTR depth criterion's valid mask.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

logger = logging.getLogger("autovision")

__all__ = ["DepthMetricCollector"]


class DepthMetricCollector:
    """Accumulate per-image depth maps and compute the metric set."""

    def __init__(
        self,
        min_depth: float = 0.001,
        max_depth: float = 10.0,
        alignment: Literal["none", "median", "log-affine"] = "log-affine",
    ) -> None:
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.alignment = alignment
        self._pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """``predictions``/``targets``: (B, H, W) or (H, W) depth in meters."""
        if predictions.ndim == 2:
            predictions = predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)
        for pred, gt in zip(predictions, targets):
            self._pairs.append(
                (pred.detach().float().cpu(), gt.detach().float().cpu())
            )

    def reset(self) -> None:
        self._pairs = []

    def export_state(self) -> dict:
        return {"pairs": list(self._pairs)}

    def merge_state(self, state: dict) -> None:
        self._pairs.extend(state.get("pairs", []))

    def _align(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if self.alignment == "none":
            return pred
        if self.alignment == "median":
            scale = torch.median(gt) / torch.median(pred).clamp_min(1e-6)
            return pred * scale
        # log-affine: solve min ||a * log(pred) + b - log(gt)|| in closed form.
        log_pred = pred.clamp_min(1e-6).log()
        log_gt = gt.clamp_min(1e-6).log()
        mean_pred = log_pred.mean()
        mean_gt = log_gt.mean()
        variance = ((log_pred - mean_pred) ** 2).sum().clamp_min(1e-12)
        slope = ((log_pred - mean_pred) * (log_gt - mean_gt)).sum() / variance
        intercept = mean_gt - slope * mean_pred
        return (slope * log_pred + intercept).exp()

    def compute(self) -> dict[str, float]:
        if not self._pairs:
            return {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}
        abs_rel_sum = 0.0
        squared_error_sum = 0.0
        delta_counts = [0, 0, 0]
        total = 0
        for pred, gt in self._pairs:
            support = (
                (gt >= self.min_depth)
                & (gt <= self.max_depth)
                & torch.isfinite(pred)
                & (pred > 0)
            )
            if not bool(support.any()):
                continue
            pred_v = self._align(pred[support], gt[support]).clamp_min(1e-6)
            gt_v = gt[support]
            abs_rel_sum += (pred_v - gt_v).abs().div(gt_v).sum().item()
            squared_error_sum += ((pred_v - gt_v) ** 2).sum().item()
            ratio = torch.maximum(pred_v / gt_v, gt_v / pred_v)
            for index, threshold in enumerate((1.25, 1.25**2, 1.25**3)):
                delta_counts[index] += int((ratio < threshold).sum().item())
            total += pred_v.numel()
        if total == 0:
            return {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}
        return {
            "abs_rel": abs_rel_sum / total,
            "rmse": (squared_error_sum / total) ** 0.5,
            "delta1": delta_counts[0] / total,
            "delta2": delta_counts[1] / total,
            "delta3": delta_counts[2] / total,
        }
