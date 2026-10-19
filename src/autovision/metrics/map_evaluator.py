# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""COCO-style evaluation for object detection (pycocotools backend)."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch

__all__ = ["COCOMAPEvaluator", "predictions_to_coco"]


def predictions_to_coco(
    predictions: Sequence[dict],
    categories: dict[int, str],
    original_sizes: Sequence[tuple[int, int]],
) -> tuple[dict, list[dict]]:
    """Build (coco_gt_dict, coco_dt_list) from in-memory results.

    ``predictions[i]``: {"boxes": (N,4) xyxy absolute pixels,
    "scores": (N,), "labels": (N,) internal ids}. Internal ids map back to
    user categories via insertion order in ``categories``.
    """
    internal_to_user = {index: class_id for index, class_id in enumerate(sorted(categories))}
    gt = {
        "images": [
            {"id": index, "width": size[0], "height": size[1]}
            for index, size in enumerate(original_sizes)
        ],
        "annotations": [],  # filled by compute() from real targets
        "categories": [
            {"id": internal_to_user[index], "name": name}
            for index, name in enumerate(sorted(categories))
        ],
    }
    dt = []
    for index, prediction in enumerate(predictions):
        boxes = prediction.get("bboxes", prediction.get("boxes"))
        scores = prediction["scores"]
        labels = prediction["labels"]
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(v) for v in box]
            dt.append(
                {
                    "image_id": index,
                    "category_id": internal_to_user[int(label)],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return gt, dt


class COCOMAPEvaluator:
    """COCO mAP over accumulated predictions (single class or multi-class).

    Ground truth is provided once via :meth:`set_ground_truth`; predictions
    accumulate per validation round and ``compute()`` runs ``COCOeval``.
    """

    def __init__(self, categories: dict[int, str], max_dets: int = 100) -> None:
        """``max_dets`` caps detections per image in the COCO eval.

        Default 100 matches pycocotools. Dense benchmarks evaluate higher
        (the VisDrone official protocol uses 500: crowded frames hold
        hundreds of objects, and the cap directly bounds recall).
        """
        self.categories = categories  # internal id -> name
        self.max_dets = max_dets
        self._predictions: list[dict] = []
        self._ground_truth: list[dict] = []  # {"boxes","labels","size"} per image
        #: Per-class rows from the last :meth:`compute` (name, ap, ar, f1,
        #: precision, recall) — consumed by the validation report renderer.
        self.last_per_class: list[dict[str, float]] = []

    def set_ground_truth(self, targets: Sequence[dict]) -> None:
        self._ground_truth = [
            {
                "boxes": t["boxes"].detach().cpu() if torch.is_tensor(t["boxes"]) else torch.as_tensor(t["boxes"]),
                "labels": t["labels"].detach().cpu() if torch.is_tensor(t["labels"]) else torch.as_tensor(t["labels"]),
            }
            for t in targets
        ]

    def update(self, predictions: Sequence[dict]) -> None:
        self._predictions.extend(predictions)

    def update_targets(
        self,
        targets: Sequence[dict],
        original_sizes: Sequence[tuple[int, int]],
    ) -> None:
        """Accumulate ground truth for the current validation round.

        Target boxes arrive as normalized cxcywh (invariant under the val
        resize); ``original_sizes`` (w, h) per item maps them to absolute
        xyxy on the original frame (matching predictions). Optional
        ``"iscrowds"`` per target carries the COCO crowd flag (1 = ignore
        region under the official protocol; missing = all zeros).
        """
        for index, target in enumerate(targets):
            boxes = (
                target["boxes"].detach().cpu()
                if torch.is_tensor(target["boxes"]) else torch.as_tensor(target["boxes"])
            )
            labels = (
                target["labels"].detach().cpu()
                if torch.is_tensor(target["labels"]) else torch.as_tensor(target["labels"])
            )
            crowds = target.get("iscrowds")
            if crowds is None:
                crowds = torch.zeros(labels.shape[0], dtype=torch.long)
            else:
                crowds = (
                    crowds.detach().cpu()
                    if torch.is_tensor(crowds) else torch.as_tensor(crowds)
                )
            orig_w, orig_h = original_sizes[index]
            if boxes.numel():
                cx, cy, w, h = boxes.unbind(-1)
                boxes = torch.stack(
                    [(cx - w / 2) * orig_w, (cy - h / 2) * orig_h,
                     (cx + w / 2) * orig_w, (cy + h / 2) * orig_h], dim=-1,
                )
            self._ground_truth.append(
                {
                    "boxes": boxes,
                    "labels": labels,
                    "iscrowds": crowds,
                    "size": (int(orig_w), int(orig_h)),
                }
            )

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
        from pycocotools.cocoeval import COCOeval

        self.last_per_class = []
        if not self._ground_truth or not self._predictions:
            return {
                "map": 0.0,
                "map_50": 0.0,
                "map_75": 0.0,
                "map_small": 0.0,
                "map_medium": 0.0,
                "map_large": 0.0,
                "mar": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }
        # True original sizes when recorded by update_targets; the fallback
        # derives a lower bound from the GT extent (set_ground_truth does not
        # carry image sizes).
        original_sizes: list[tuple[int, int]] = []
        for target in self._ground_truth:
            size = target.get("size")
            if size is not None:
                original_sizes.append((int(size[0]), int(size[1])))
                continue
            boxes = target["boxes"]
            if boxes.numel() > 0:
                width = float(boxes[:, 2].max())
                height = float(boxes[:, 3].max())
            else:
                width, height = 1.0, 1.0
            original_sizes.append((int(width) + 1, int(height) + 1))
        gt_dict, dt_list = predictions_to_coco(
            self._predictions, self.categories, original_sizes
        )
        # Replace dummy full-image GT annotations with real boxes.
        gt_annotations = []
        annotation_id = 1
        internal_to_user = {
            index: class_id for index, class_id in enumerate(sorted(self.categories))
        }
        for index, target in enumerate(self._ground_truth):
            crowds = target.get("iscrowds")
            for position, (box, label) in enumerate(
                zip(target["boxes"], target["labels"])
            ):
                x1, y1, x2, y2 = [float(v) for v in box]
                gt_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": index,
                        "category_id": internal_to_user[int(label)],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": max((x2 - x1) * (y2 - y1), 1.0),
                        # Crowd regions are ignore regions in the official
                        # COCO protocol (no recall requirement, overlapping
                        # detections not penalized); earlier revisions wrote
                        # 0 unconditionally, which counted the 446 val2017
                        # crowd boxes as regular GT.
                        "iscrowd": int(crowds[position]) if crowds is not None else 0,
                    }
                )
                annotation_id += 1
        gt_dict["annotations"] = gt_annotations

        with tempfile.TemporaryDirectory() as tmp:
            gt_file = Path(tmp) / "gt.json"
            gt_file.write_text(json.dumps(gt_dict))
            from pycocotools.coco import COCO

            coco_gt = COCO(str(gt_file))
        if not dt_list:
            return {
                "map": 0.0,
                "map_50": 0.0,
                "map_75": 0.0,
                "map_small": 0.0,
                "map_medium": 0.0,
                "map_large": 0.0,
                "mar": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }
        coco_dt = coco_gt.loadRes(dt_list)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        # Three-element ladder per pycocotools convention; [1, 10, 100]
        # equals the library default, so the default path is numerically
        # unchanged.
        evaluator.params.maxDets = [1, 10, self.max_dets]
        evaluator.evaluate()
        evaluator.accumulate()
        # summarize() is deliberately skipped: its headline stats[0]/stats[1]
        # are pinned to maxDets=100 in pycocotools (the bare ``_summarize(1)``
        # call), returning -1 under any other ladder — and printing a table
        # of the wrong cap next to the right one invites misreading. Every
        # headline number is computed directly from the precision/recall
        # arrays, numerically identical to the official stats when the cap
        # is 100.
        import numpy as np

        params = evaluator.params
        tind = int(np.where(params.iouThrs == 0.5)[0][0])
        tind75 = int(np.where(params.iouThrs == 0.75)[0][0])
        aind = params.areaRngLbl.index("all")
        mind = len(params.maxDets) - 1
        precision = evaluator.eval["precision"]  # [T, R, K, A, M]
        recall = evaluator.eval["recall"]  # [T, K, A, M] — no score axis

        def _mean(values) -> float:
            valid = values[values > -1]
            return float(valid.mean()) if valid.size else -1.0

        metrics = {
            "map": _mean(precision[tind:, :, :, aind, mind]),
            "map_50": _mean(precision[tind, :, :, aind, mind]),
            "map_75": _mean(precision[tind75, :, :, aind, mind]),
            "mar": _mean(recall[:, :, aind, mind]),
        }
        # Per-area slices (COCO protocol: small < 32², medium 32²–96²,
        # large > 96² px²) — the areaRng ladder pycocotools always builds,
        # previously pinned to "all"; additive keys, headline numbers are
        # numerically unchanged.
        for area_label in ("small", "medium", "large"):
            area_index = params.areaRngLbl.index(area_label)
            metrics[f"map_{area_label}"] = _mean(
                precision[tind:, :, :, area_index, mind]
            )

        # F1 sweep at IoU 0.5 across pycocotools' operating-point axis: the
        # precision array varies along it, recall does not (it is fixed by
        # the maxDets cutoff), so each point pairs precision@point with the
        # single recall value — the standard PR curve. Headline
        # precision/recall/F1 are macro-averaged over classes with ground
        # truth at the F1-optimal point.
        p50 = precision[tind, :, :, aind, mind]  # [R, K]
        r50 = recall[tind, :, aind, mind]  # [K]
        valid_class = (p50 > -1).any(axis=0)
        r_values = r50[valid_class]
        r_values = r_values[r_values > -1]
        r_scalar = float(r_values.mean()) if r_values.size else -1.0
        p_curve = np.array(
            [
                (
                    p50[t][valid_class][p50[t][valid_class] > -1].mean()
                    if (p50[t][valid_class] > -1).any() else -1.0
                )
                for t in range(p50.shape[0])
            ]
        )
        if p_curve.max() > -1 and r_scalar > 0:
            f1_curve = np.where(
                p_curve > -1,
                2 * p_curve * r_scalar / np.maximum(p_curve + r_scalar, 1e-12),
                -1.0,
            )
            best_t = int(np.argmax(f1_curve))
            metrics["f1"] = float(f1_curve[best_t])
            metrics["precision"] = float(p_curve[best_t])
            metrics["recall"] = float(r_scalar)
        else:
            metrics["f1"] = 0.0
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0

        # Per-class detail for the validation report (category order);
        # classes without ground truth are skipped.
        class_names = [name for _, name in sorted(self.categories.items())]
        self.last_per_class = []
        for k, name in enumerate(class_names):
            if k >= p50.shape[1] or not (p50[:, k] > -1).any():
                continue
            r_k = float(r50[k]) if r50[k] > -1 else -1.0
            p_k = p50[:, k]
            if p_k.max() > -1 and r_k > 0:
                f1_k = np.where(
                    p_k > -1,
                    2 * p_k * r_k / np.maximum(p_k + r_k, 1e-12),
                    -1.0,
                )
                best_k = int(np.argmax(f1_k))
                f1_best = float(f1_k[best_k])
                p_best = float(p_k[best_k])
            else:
                f1_best = 0.0
                p_best = 0.0
            self.last_per_class.append(
                {
                    "name": name,
                    "ap": _mean(precision[tind:, :, k, aind, mind]),
                    "ar": _mean(recall[:, k, aind, mind]),
                    "f1": f1_best,
                    "precision": p_best,
                    "recall": max(r_k, 0.0),
                }
            )
        return metrics
