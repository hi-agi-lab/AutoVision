# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Panoptic Quality (PQ/SQ/RQ) over accumulated (class, segment) maps.

Self-contained implementation of the panoptic segmentation metric
(Kirillov et al., 2019). The baseline delegates to ``torchmetrics``
``PanopticQuality``; torchmetrics is not a dependency here, so the
algorithm is re-implemented with the same void semantics:

- Void pixels (target pixels that belong to no valid segment: ignore
  label, crowd label -1, or segment id -1) are removed from both the
  ground-truth and the predicted segment before computing IoU.
- Predicted segments whose pixels are entirely void are silently dropped
  (they never count as false positives).
- A ground-truth segment matches a predicted segment of the same class
  when their IoU is strictly greater than 0.5 (this guarantees a unique
  match per segment on both sides).
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["PanopticMetricCollector"]


class PanopticMetricCollector:
    """Accumulates per-class TP/FP/FN/IoU for PQ, SQ and RQ.

    Predictions/targets are (B, H, W, 2) long tensors where the last
    dimension holds ``(class_id, segment_id)`` with internal class ids.
    Things come first (``things``), then stuff classes (``stuffs``).
    """

    def __init__(
        self,
        things: list[int],
        stuffs: list[int],
        class_names: dict[int, str] | None = None,
    ) -> None:
        self.class_ids = list(things) + list(stuffs)
        self._class_to_index = {c: i for i, c in enumerate(self.class_ids)}
        self.class_names = class_names or {}
        num = len(self.class_ids)
        self._iou_sum = torch.zeros(num, dtype=torch.double)
        self._tp = torch.zeros(num, dtype=torch.double)
        self._fp = torch.zeros(num, dtype=torch.double)
        self._fn = torch.zeros(num, dtype=torch.double)

    # ── accumulation ────────────────────────────────────────────────────
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch of (B, H, W, 2) class/segment maps."""
        predictions = predictions.detach().cpu().long()
        targets = targets.detach().cpu().long()
        for pred, target in zip(predictions, targets):
            self._update_single(pred, target)

    @torch.no_grad()
    def _update_single(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        offset = int(max(target.max(), pred.max()).item()) + 2
        valid = torch.zeros(offset, dtype=torch.bool)
        for class_id in self._class_to_index:
            if class_id < offset:
                valid[class_id] = True
        t_label, t_segment = target[..., 0], target[..., 1]
        p_label, p_segment = pred[..., 0], pred[..., 1]
        # Void: labels outside the declared classes, crowd (-1) or no segment.
        t_valid = (t_label >= 0) & valid[t_label.clamp(min=0)] & (t_segment >= 0)
        p_valid = (p_label >= 0) & valid[p_label.clamp(min=0)] & (p_segment >= 0)

        t_combined = t_label * offset + t_segment
        p_combined = p_label * offset + p_segment

        gt_ids, gt_area = t_combined[t_valid].unique(return_counts=True)
        # Predicted segments restricted to non-void pixels (void removal).
        pred_ids, pred_area = p_combined[t_valid & p_valid].unique(return_counts=True)
        num_gt, num_pred = gt_ids.numel(), pred_ids.numel()
        if num_gt == 0 and num_pred == 0:
            return

        gt_class = (gt_ids // offset).long()
        pred_class = (pred_ids // offset).long()
        gt_matched = torch.zeros(num_gt, dtype=torch.bool)
        pred_matched = torch.zeros(num_pred, dtype=torch.bool)

        # IoU per overlapping (gt, pred) segment pair, void pixels removed
        # from both sides (the overlap region is exactly the non-void part).
        overlap = t_valid & p_valid
        if overlap.any():
            g_idx = torch.searchsorted(gt_ids, t_combined[overlap])
            p_idx = torch.searchsorted(pred_ids, p_combined[overlap])
            inter = torch.bincount(
                g_idx * num_pred + p_idx, minlength=num_gt * num_pred
            ).double().reshape(num_gt, num_pred)
            pair_g, pair_p = inter.nonzero(as_tuple=True)
            intersection = inter[pair_g, pair_p]
            union = (
                gt_area[pair_g].double() + pred_area[pair_p].double() - intersection
            )
            iou = intersection / union
            keep = (gt_class[pair_g] == pred_class[pair_p]) & (iou > 0.5)
            if keep.any():
                match_g, match_p, match_iou = (
                    pair_g[keep],
                    pair_p[keep],
                    iou[keep],
                )
                gt_matched[match_g] = True
                pred_matched[match_p] = True
                rows = gt_class[match_g]
                self._tp.index_add_(0, rows, torch.ones_like(match_iou))
                self._iou_sum.index_add_(0, rows, match_iou)

        # Unmatched ground truth -> FN; unmatched predictions -> FP.
        ones = torch.ones(1, dtype=torch.double)
        self._fn.index_add_(0, gt_class[~gt_matched], ones.expand((~gt_matched).sum()))
        self._fp.index_add_(0, pred_class[~pred_matched], ones.expand((~pred_matched).sum()))

    def reset(self) -> None:
        num = len(self.class_ids)
        self._iou_sum = torch.zeros(num, dtype=torch.double)
        self._tp = torch.zeros(num, dtype=torch.double)
        self._fp = torch.zeros(num, dtype=torch.double)
        self._fn = torch.zeros(num, dtype=torch.double)

    # ── results ─────────────────────────────────────────────────────────
    def compute(self) -> dict[str, float]:
        denominator = self._tp + 0.5 * self._fp + 0.5 * self._fn
        present = denominator > 0
        zeros = denominator.new_zeros(())
        pq = torch.where(denominator > 0, self._iou_sum / denominator.clamp(min=1e-12), zeros)
        sq = torch.where(self._tp > 0, self._iou_sum / self._tp.clamp(min=1e-12), zeros)
        rq = torch.where(denominator > 0, self._tp / denominator.clamp(min=1e-12), zeros)
        result = {
            name: float(values[present].mean()) if present.any() else 0.0
            for name, values in (("pq", pq), ("sq", sq), ("rq", rq))
        }
        for index, class_id in enumerate(self.class_ids):
            name = self.class_names.get(class_id, str(class_id))
            result[f"pq_{name}"] = float(pq[index])
            result[f"sq_{name}"] = float(sq[index])
            result[f"rq_{name}"] = float(rq[index])
        return result

    # ── distributed validation support ──────────────────────────────────
    def export_state(self) -> dict[str, Any]:
        return {
            "iou_sum": self._iou_sum.clone(),
            "tp": self._tp.clone(),
            "fp": self._fp.clone(),
            "fn": self._fn.clone(),
        }

    def merge_state(self, state: dict[str, Any]) -> None:
        self._iou_sum += state["iou_sum"].to(self._iou_sum.dtype)
        self._tp += state["tp"].to(self._tp.dtype)
        self._fp += state["fp"].to(self._fp.dtype)
        self._fn += state["fn"].to(self._fn.dtype)
