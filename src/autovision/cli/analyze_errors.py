# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Detection error analysis command (roadmap C2).

Two input modes:

- ``model= + data=``: run the model over a labeled split (the same
  prediction/target contract as ``autovision-cli val``) and analyze it;
- ``predictions= + annotations=``: analyze an offline predictions json
  (``predict_to_files`` output) against a COCO annotations json.

Both feed :func:`autovision.metrics.error_analysis.analyze_detection_errors`
and write ``report.json`` + ``report.md`` + the top-FP/FN gallery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from autovision.errors import AutoVisionError
from autovision.metrics.error_analysis import (
    ErrorAnalysisReport,
    analyze_detection_errors,
)
from autovision.tasks.task_model_helpers import load_task_model
from autovision.types import PathLike

__all__ = ["analyze_errors"]


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as handle:
        width, height = handle.size
    return int(width), int(height)


def _absolute_targets(
    boxes_norm: list[list[float]],
    labels: list[int],
    iscrowds: list[int] | None,
    size: tuple[int, int],
) -> dict[str, Any]:
    """Normalized cxcywh targets -> absolute xyxy (the evaluator contract)."""
    width, height = size
    boxes = []
    for cx, cy, w, h in boxes_norm:
        boxes.append(
            [
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ]
        )
    target: dict[str, Any] = {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if iscrowds is not None:
        target["iscrowds"] = torch.tensor(iscrowds, dtype=torch.long)
    return target


def _from_model(
    model: PathLike | torch.nn.Module,
    data: PathLike | dict[str, Any],
    *,
    split: str,
    threshold: float,
    max_images: int | None,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, str], list[Path]]:
    from autovision.cli.model_handle import _resolve_data_args

    network = (
        load_task_model(model, device=device)
        if isinstance(model, (str, Path))
        else model
    )
    data_args = _resolve_data_args(data)
    predictions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    for index, item in enumerate(data_args.list_image_info(split)):
        if max_images is not None and index >= max_images:
            break
        image_path = Path(item["image_path"])
        size = _image_size(image_path)
        prediction = network.predict(image_path, threshold=threshold)
        predictions.append(prediction)
        targets.append(
            _absolute_targets(
                json.loads(item["bboxes"]),
                json.loads(item["class_labels"]),
                json.loads(item["iscrowds"]) if "iscrowds" in item else None,
                size,
            )
        )
        image_paths.append(image_path)
    if not predictions:
        raise AutoVisionError(f"The '{split}' split produced no images to analyze.")
    return predictions, targets, dict(data_args.classes()), image_paths


def _from_offline(
    predictions: PathLike,
    annotations: PathLike,
    images: PathLike | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, str], list[Path]]:
    records = json.loads(Path(predictions).read_text())
    payload = json.loads(Path(annotations).read_text())
    if not isinstance(records, list) or not records:
        raise AutoVisionError(f"predictions json is empty: '{predictions}'.")

    category_pairs = sorted(
        (int(c["id"]), str(c["name"])) for c in payload.get("categories", [])
    )
    categories = {category_id: name for category_id, name in category_pairs}
    internal_of = {category_id: index for index, (category_id, _) in enumerate(category_pairs)}
    annotations_root = Path(images) if images is not None else Path(annotations).parent

    by_path: dict[str, dict[str, Any]] = {}
    by_stem: dict[str, list[dict[str, Any]]] = {}
    for image in payload.get("images", []):
        name = str(image["file_name"])
        resolved = Path(name) if Path(name).is_absolute() else annotations_root / name
        by_path[resolved.resolve().as_posix()] = image
        by_stem.setdefault(Path(name).stem, []).append(image)
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in payload.get("annotations", []):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    def _targets_for(image: dict[str, Any]) -> dict[str, Any]:
        boxes, labels, crowds = [], [], []
        for annotation in annotations_by_image.get(int(image["id"]), []):
            category_id = int(annotation["category_id"])
            if category_id not in internal_of:
                continue
            x, y, w, h = (float(v) for v in annotation["bbox"])
            boxes.append([x, y, x + w, y + h])
            labels.append(internal_of[category_id])
            crowds.append(int(annotation.get("iscrowd", 0)))
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "iscrowds": torch.tensor(crowds, dtype=torch.long),
        }

    predictions_out: list[dict[str, Any]] = []
    targets_out: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    for record in records:
        path = Path(str(record["image"]))
        image = by_path.get(path.resolve().as_posix())
        if image is None:
            candidates = by_stem.get(path.stem, [])
            if len(candidates) > 1:
                raise AutoVisionError(
                    f"prediction record '{record['image']}' matches "
                    f"{len(candidates)} images with the stem '{path.stem}' — "
                    "match by full path instead of stem."
                )
            image = candidates[0] if candidates else None
        if image is None:
            raise AutoVisionError(
                f"prediction record '{record['image']}' matches no image in "
                f"'{annotations}'."
            )
        predictions_out.append(
            {
                "bboxes": torch.tensor(record["bboxes"], dtype=torch.float32).reshape(-1, 4),
                "scores": torch.tensor(record["scores"], dtype=torch.float32).reshape(-1),
                "labels": torch.tensor(record["labels"], dtype=torch.long).reshape(-1),
            }
        )
        targets_out.append(_targets_for(image))
        image_paths.append(path)
    return predictions_out, targets_out, categories, image_paths


def analyze_errors(
    model: PathLike | torch.nn.Module | None = None,
    data: PathLike | dict[str, Any] | None = None,
    out: PathLike = "error_report",
    *,
    split: str = "val",
    threshold: float = 0.0,
    iou: float = 0.5,
    top_k: int = 5,
    max_images: int | None = None,
    predictions: PathLike | None = None,
    annotations: PathLike | None = None,
    images: PathLike | None = None,
    device: str = "cpu",
) -> ErrorAnalysisReport:
    """Produce the error report + top-FP/FN gallery for one detection model
    or one predictions json (see the module docstring for the two modes)."""
    if predictions is not None or annotations is not None:
        if predictions is None or annotations is None:
            raise AutoVisionError(
                "the offline mode needs both predictions= and annotations=."
            )
        gathered = _from_offline(predictions, annotations, images)
    else:
        if model is None or data is None:
            raise AutoVisionError("pass model= + data= (or predictions= + annotations=).")
        gathered = _from_model(
            model, data, split=split, threshold=threshold, max_images=max_images,
            device=device,
        )
    predictions_list, targets_list, categories, image_paths = gathered
    return analyze_detection_errors(
        predictions_list,
        targets_list,
        categories,
        image_paths,
        out,
        iou_threshold=iou,
        score_threshold=threshold,
        top_k=top_k,
        split=split,
    )
