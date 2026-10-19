# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""COCO keypoint AP (OKS) evaluation for the GTR pose family (doc 31 §11 S5b).

Official protocol: ``COCOeval(iouType="keypoints")`` over the **raw**
``person_keypoints`` annotation file — pycocotools itself applies the
keypoint ignore rules (``num_keypoints == 0`` and ``iscrowd`` persons are
ignored) and the OKS sigmas. Detections are ``{"keypoints": [x, y, v] *
17, "category_id", "score"}`` per image, exactly the official
``prepare_for_coco_keypoint`` layout; ``category_id`` is passed through
from the model label (the pose lineage keeps COCO category ids — person
== 1), so non-person labels simply never match the GT category and drop
out of the pycocotools per-category gather.

Headline numbers are computed directly from the precision/recall arrays
(same approach as :class:`~autovision.metrics.map_evaluator.COCOMAPEvaluator`,
per-keypoint ``maxDets = [20]`` — pycocotools' keypoint ladder).
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from autovision.types import PathLike

__all__ = ["COCOKeypointAPEvaluator", "predictions_to_coco_keypoints"]

_EMPTY_METRICS: dict[str, float] = {
    "kp_map": 0.0,
    "kp_map_50": 0.0,
    "kp_map_75": 0.0,
    "kp_map_m": 0.0,
    "kp_map_l": 0.0,
    "kp_mar": 0.0,
}


def predictions_to_coco_keypoints(
    predictions: Sequence[dict],
    image_ids: Sequence[int] | None = None,
) -> list[dict]:
    """COCO keypoint detection entries from decoded pose outputs.

    ``predictions[i]``: {"keypoints": (N, 3K) absolute pixels as
    ``[x, y, v]`` triplets, "scores": (N,), "labels": (N,)}. ``image_ids``
    defaults to the enumeration position (which is what the evaluator's
    ground-truth remap uses).
    """
    results: list[dict] = []
    for index, prediction in enumerate(predictions):
        keypoints = prediction["keypoints"]
        scores = prediction["scores"]
        labels = prediction["labels"]
        image_id = int(image_ids[index]) if image_ids is not None else index
        keypoints = (
            keypoints.detach().cpu() if torch.is_tensor(keypoints) else torch.as_tensor(keypoints)
        )
        if keypoints.numel() == 0:
            continue
        flat = keypoints.reshape(keypoints.shape[0], -1).tolist()
        for row, score, label in zip(flat, scores, labels):
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "keypoints": [float(value) for value in row],
                    "score": float(score),
                }
            )
    return results


class COCOKeypointAPEvaluator:
    """COCO keypoint AP over accumulated predictions (single category).

    Ground truth comes from the raw annotation file (loaded once); the
    file's image ids are remapped to their sorted-enumeration index so
    predictions indexed by dataset position line up regardless of the raw
    COCO ids.
    """

    def __init__(
        self,
        annotations_path: PathLike,
        categories: dict[int, str] | None = None,
        image_ids: Sequence[int] | None = None,
    ) -> None:
        self.annotations_path = Path(annotations_path)
        with open(self.annotations_path) as handle:
            raw: dict[str, Any] = json.load(handle)
        self._raw = raw
        self._subset: list[int] | None = (
            sorted(int(image_id) for image_id in image_ids)
            if image_ids is not None
            else None
        )
        categories_from_file = {
            int(category["id"]): str(category["name"])
            for category in raw.get("categories", [])
        }
        self.categories = dict(categories) if categories else categories_from_file
        self._predictions: list[dict] = []
        self._image_id_count = 0

    # ── ground truth ────────────────────────────────────────────────────
    def _ground_truth(self) -> dict[str, Any]:
        raw = self._raw
        sorted_ids = sorted(int(image["id"]) for image in raw.get("images", []))
        id_map = {original: index for index, original in enumerate(sorted_ids)}
        annotations = []
        for position, annotation in enumerate(raw.get("annotations", [])):
            entry = dict(annotation)
            entry["image_id"] = id_map[int(annotation["image_id"])]
            entry.setdefault("id", position + 1)
            keypoints = entry.get("keypoints") or []
            if "num_keypoints" not in entry:
                entry["num_keypoints"] = sum(
                    1 for value in keypoints[2::3] if value > 0
                )
            if "iscrowd" not in entry:
                entry["iscrowd"] = 0
            if "area" not in entry:
                bbox = entry.get("bbox") or [0, 0, 0, 0]
                entry["area"] = float(bbox[2]) * float(bbox[3])
            annotations.append(entry)
        return {
            "images": [
                {
                    "id": id_map[int(image["id"])],
                    "width": int(image.get("width", 0)),
                    "height": int(image.get("height", 0)),
                }
                for image in raw.get("images", [])
            ],
            "annotations": annotations,
            "categories": raw.get("categories", []),
        }

    # ── accumulation ────────────────────────────────────────────────────
    def update(
        self,
        predictions: Sequence[dict],
        image_ids: Sequence[int] | None = None,
    ) -> None:
        if image_ids is None:
            image_ids = list(range(self._image_id_count, self._image_id_count + len(predictions)))
        self._image_id_count = max(
            [self._image_id_count]
            + [int(image_id) + 1 for image_id in image_ids]
        )
        self._predictions.extend(
            predictions_to_coco_keypoints(predictions, image_ids)
        )

    def reset(self) -> None:
        self._predictions = []
        self._image_id_count = 0

    def export_state(self) -> dict:
        return {"predictions": list(self._predictions)}

    def merge_state(self, state: dict) -> None:
        self._predictions.extend(state["predictions"])

    # ── metric ──────────────────────────────────────────────────────────
    def compute(self) -> dict[str, float]:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        if not self._predictions:
            return dict(_EMPTY_METRICS)
        gt_dict = self._ground_truth()
        if not gt_dict["annotations"] or not gt_dict["images"]:
            return dict(_EMPTY_METRICS)
        with tempfile.TemporaryDirectory() as tmp:
            gt_file = Path(tmp) / "gt.json"
            gt_file.write_text(json.dumps(gt_dict))
            coco_gt = COCO(str(gt_file))
        coco_dt = coco_gt.loadRes(self._predictions)
        evaluator = COCOeval(coco_gt, coco_dt, "keypoints")
        if self._subset is not None:
            # Evaluate only the requested images (e.g. a val subset run);
            # GT outside the subset would otherwise count as missed.
            evaluator.params.imgIds = list(self._subset)
        evaluator.evaluate()
        evaluator.accumulate()

        import numpy as np

        params = evaluator.params
        tind = int(np.where(params.iouThrs == 0.5)[0][0])
        tind75 = int(np.where(params.iouThrs == 0.75)[0][0])
        aind = params.areaRngLbl.index("all")
        mind = 0  # keypoints ladder is a single maxDets entry ([20])
        precision = evaluator.eval["precision"]  # [T, R, K, A, M]
        recall = evaluator.eval["recall"]  # [T, K, A, M]

        def _mean(values) -> float:
            valid = values[values > -1]
            return float(valid.mean()) if valid.size else -1.0

        return {
            "kp_map": _mean(precision[tind:, :, :, aind, mind]),
            "kp_map_50": _mean(precision[tind, :, :, aind, mind]),
            "kp_map_75": _mean(precision[tind75, :, :, aind, mind]),
            "kp_map_m": _mean(precision[tind:, :, :, params.areaRngLbl.index("medium"), mind]),
            "kp_map_l": _mean(precision[tind:, :, :, params.areaRngLbl.index("large"), mind]),
            "kp_mar": _mean(recall[:, :, aind, mind]),
        }
