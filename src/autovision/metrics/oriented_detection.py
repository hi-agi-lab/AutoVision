# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Oriented detection mAP (self-contained rotated-box COCO-style metric).

pycocotools cannot represent rotated boxes, so this evaluator runs the
matching itself: per class and per IoU threshold (0.50:0.05:0.95), a greedy
score-descending assignment on the exact rotated IoU
(:func:`rotated_iou_matrix`), then all-point-interpolated AP. ``map`` is
the macro average over classes that have ground truth across all ten
thresholds; ``map_50`` is the same average at IoU 0.50.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from autovision.geometry.obb import (
    rotated_iou_matrix,
    xyxyxyxy_to_xywhr,
)

__all__ = ["OrientedDetectionMetricCollector"]

_IOU_THRESHOLDS = tuple(0.5 + 0.05 * i for i in range(10))


def _average_precision(scores_tps: list[tuple[float, int]], num_gt: int) -> float:
    """All-point (VOC-style envelope) AP from score-ordered TP flags."""
    if num_gt == 0:
        return 0.0
    scores_tps = sorted(scores_tps, key=lambda entry: -entry[0])
    tp_cum = 0
    fp_cum = 0
    recalls_precisions: list[tuple[float, float]] = []
    for _, tp in scores_tps:
        tp_cum += tp
        fp_cum += 1 - tp
        recalls_precisions.append(
            (tp_cum / num_gt, tp_cum / (tp_cum + fp_cum))
        )
    # Monotone precision envelope from the right, then integrate over the
    # strictly increasing recall steps.
    ap = 0.0
    prev_recall = 0.0
    best_precision = 0.0
    for recall, precision in reversed(recalls_precisions):
        best_precision = max(best_precision, precision)
        if recall > prev_recall:
            ap += (recall - prev_recall) * best_precision
            prev_recall = recall
    return float(ap)


class OrientedDetectionMetricCollector:
    """Accumulates oriented detections + ground truth and computes mAP.

    Predictions arrive as per-image ``{"labels", "polygons", "scores"}``
    with polygons in absolute original-frame pixels (the OBB decode
    output). Targets arrive per image as normalized corner arrays
    (``corners``: (n, 8)) that are rescaled to original-frame pixels here
    — the oriented analogue of the COCO evaluator's normalized-cxcywh
    rescaling.
    """

    def __init__(self, categories: dict[int, str]) -> None:
        self.categories = categories  # internal id -> name
        self._predictions: list[dict] = []
        self._ground_truth: list[dict] = []

    def update(self, predictions: list[dict[str, torch.Tensor]]) -> None:
        self._predictions.extend(
            {
                "labels": p["labels"].detach().cpu(),
                "polygons": p["polygons"].detach().cpu(),
                "scores": p["scores"].detach().cpu(),
            }
            for p in predictions
        )

    def update_targets(
        self,
        targets: list[dict[str, torch.Tensor]],
        original_sizes: list[tuple[int, int]],
    ) -> None:
        for index, target in enumerate(targets):
            corners = (
                target["corners"].detach().cpu()
                if torch.is_tensor(target["corners"])
                else torch.as_tensor(target["corners"])
            ).reshape(-1, 8)
            labels = (
                target["labels"].detach().cpu()
                if torch.is_tensor(target["labels"])
                else torch.as_tensor(target["labels"])
            )
            orig_w, orig_h = original_sizes[index]
            if corners.numel():
                corners = corners.clone()
                corners[:, 0::2] *= orig_w
                corners[:, 1::2] *= orig_h
            self._ground_truth.append({"corners": corners, "labels": labels})

    def reset(self) -> None:
        self._predictions = []
        self._ground_truth = []

    def export_state(self) -> dict:
        """Snapshot accumulated predictions/targets (CPU tensors) for
        cross-rank gathering in distributed validation."""
        return {
            "predictions": list(self._predictions),
            "ground_truth": list(self._ground_truth),
        }

    def merge_state(self, state: dict) -> None:
        """Merge a snapshot produced by :meth:`export_state` (e.g. gathered
        from other ranks under DDP)."""
        self._predictions.extend(state["predictions"])
        self._ground_truth.extend(state["ground_truth"])

    def compute(self) -> dict[str, float]:
        if not self._ground_truth:
            return {"map": 0.0, "map_50": 0.0}

        # Per threshold -> per class -> score-ordered (score, tp) stream.
        streams: dict[float, dict[int, list[tuple[float, int]]]] = {
            threshold: defaultdict(list) for threshold in _IOU_THRESHOLDS
        }
        num_gt_per_class: dict[int, int] = defaultdict(int)

        for gt, pred in zip(self._ground_truth, self._predictions):
            gt_xywhr = (
                xyxyxyxy_to_xywhr(gt["corners"])
                if gt["corners"].numel()
                else torch.zeros(0, 5)
            )
            gt_labels = gt["labels"]
            pred_xywhr = (
                xyxyxyxy_to_xywhr(pred["polygons"].reshape(-1, 8))
                if pred["polygons"].numel()
                else torch.zeros(0, 5)
            )
            pred_labels = pred["labels"]
            pred_scores = pred["scores"]

            for class_id in gt_labels.unique().tolist():
                class_id = int(class_id)
                gt_class = gt_xywhr[gt_labels == class_id]
                num_gt_per_class[class_id] += int(gt_class.shape[0])
                class_mask = pred_labels == class_id
                pred_class = pred_xywhr[class_mask]
                scores_class = pred_scores[class_mask]
                if pred_class.shape[0] == 0:
                    continue
                order = torch.argsort(scores_class, descending=True)
                pred_class = pred_class[order]
                scores_class = scores_class[order]
                iou = rotated_iou_matrix(pred_class, gt_class)
                for threshold in _IOU_THRESHOLDS:
                    matched: set[int] = set()
                    stream = streams[threshold][class_id]
                    for k in range(pred_class.shape[0]):
                        best_j = -1
                        best_iou = -1.0
                        for j in range(gt_class.shape[0]):
                            if j in matched:
                                continue
                            value = float(iou[k, j])
                            if value >= threshold and value > best_iou:
                                best_iou = value
                                best_j = j
                        if best_j >= 0:
                            matched.add(best_j)
                            stream.append((float(scores_class[k]), 1))
                        else:
                            stream.append((float(scores_class[k]), 0))

        classes_with_gt = sorted(num_gt_per_class)
        if not classes_with_gt:
            return {"map": 0.0, "map_50": 0.0}

        def class_mean_at(threshold: float) -> float:
            aps = [
                _average_precision(
                    streams[threshold].get(class_id, []), num_gt_per_class[class_id]
                )
                for class_id in classes_with_gt
            ]
            return sum(aps) / len(aps)

        return {
            "map": sum(class_mean_at(t) for t in _IOU_THRESHOLDS) / len(_IOU_THRESHOLDS),
            "map_50": class_mean_at(0.5),
        }
