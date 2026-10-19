# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""COCO mask AP evaluation for instance segmentation (pycocotools segm)."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

__all__ = ["COCOMaskAPEvaluator", "resize_masks"]


def resize_masks(masks, height: int, width: int):
    """Resize ``(N, H, W)`` instance masks to the requested resolution.

    Validation decodes predictions back to the original frame while the
    transformed GT masks still live on the network canvas. COCO evaluation
    requires both to share the declared image size, so the GT side is
    resampled with nearest-neighbour (binary masks must stay binary).
    """
    import torch.nn.functional as F

    if masks.numel() == 0 or tuple(masks.shape[-2:]) == (height, width):
        return masks
    return F.interpolate(
        masks.unsqueeze(1).float(), size=(height, width), mode="nearest"
    ).squeeze(1)


def _to_rle(mask: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    # COCO json convention: counts as str (loadRes decodes it back).
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _mask_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


class COCOMaskAPEvaluator:
    """COCO mask AP over accumulated instance predictions.

    Targets/predictions carry per-instance binary masks at the original
    frame resolution; ``compute()`` runs ``COCOeval`` with ``iouType="segm"``.
    """

    def __init__(self, categories: dict[int, str]) -> None:
        self.categories = categories  # internal id -> name
        self._predictions: list[dict] = []
        self._targets: list[dict] = []

    def update(self, predictions: Sequence[dict]) -> None:
        self._predictions.extend(predictions)

    def update_targets(self, targets: Sequence[dict]) -> None:
        """``targets[i]``: {"masks": (K,H,W) bool, "labels": (K,), "size": (h, w)}."""
        self._targets.extend(targets)

    def reset(self) -> None:
        self._predictions = []
        self._targets = []

    def export_state(self) -> dict:
        return {
            "predictions": list(self._predictions),
            "targets": list(self._targets),
        }

    def merge_state(self, state: dict) -> None:
        self._predictions.extend(state["predictions"])
        self._targets.extend(state["targets"])

    def compute(self) -> dict[str, float]:
        from pycocotools.cocoeval import COCOeval

        if not self._targets or not self._predictions:
            return {"mask_map": 0.0, "mask_map_50": 0.0}
        internal_to_user = {
            index: class_id for index, class_id in enumerate(sorted(self.categories))
        }
        gt = {
            "images": [
                {"id": i, "width": int(t["size"][1]), "height": int(t["size"][0])}
                for i, t in enumerate(self._targets)
            ],
            "categories": [
                {"id": internal_to_user[index], "name": name}
                for index, name in enumerate(sorted(self.categories))
            ],
            "annotations": [],
        }
        annotation_id = 1
        for index, target in enumerate(self._targets):
            for mask, label in zip(target["masks"], target["labels"]):
                mask_np = mask.detach().cpu().numpy().astype(bool)
                if not mask_np.any():
                    continue
                gt["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": index,
                        "category_id": internal_to_user[int(label)],
                        "segmentation": _to_rle(mask_np),
                        "bbox": _mask_bbox(mask_np),
                        "area": int(mask_np.sum()),
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        dt = []
        for index, prediction in enumerate(self._predictions):
            for mask, label, score in zip(
                prediction["masks"], prediction["labels"], prediction["scores"]
            ):
                mask_np = mask.detach().cpu().numpy().astype(bool)
                if not mask_np.any():
                    continue
                rle = _to_rle(mask_np)
                dt.append(
                    {
                        "image_id": index,
                        "category_id": internal_to_user[int(label)],
                        "segmentation": rle,
                        "bbox": [float(v) for v in np.atleast_1d(_rle_to_bbox(rle))],
                        "score": float(score),
                    }
                )
        if not gt["annotations"] or not dt:
            return {"mask_map": 0.0, "mask_map_50": 0.0}
        with tempfile.TemporaryDirectory() as tmp:
            gt_file = Path(tmp) / "gt.json"
            gt_file.write_text(json.dumps(gt))
            from pycocotools.coco import COCO

            coco_gt = COCO(str(gt_file))
        coco_dt = coco_gt.loadRes(dt)
        evaluator = COCOeval(coco_gt, coco_dt, "segm")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return {
            "mask_map": float(evaluator.stats[0]),
            "mask_map_50": float(evaluator.stats[1]),
        }


def _rle_to_bbox(rle: dict):
    from pycocotools import mask as mask_utils

    return mask_utils.toBbox(rle)
