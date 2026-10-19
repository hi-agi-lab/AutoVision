# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pseudo-label generation for detection self-training (roadmap C1, §2.3).

Runs a detection task model over an unlabeled image directory and writes a
COCO instance-annotation file that the training data plane consumes
directly (``data={"format": "coco", "train": {"annotations": <json>,
"images": <dir>}}``). Optional SAHI slicing for small-object domains.

Contract notes:

- Predictions are the model's normal postprocess output (original-pixel
  xyxy) — the same contract the mAP evaluator consumes, so every
  self-training round stays measurable end to end.
- Category mapping follows ``predictions_to_coco``: internal label ids
  index ``sorted(categories)`` and the emitted ``category_id`` is the
  original key. Without ``categories`` the ids are ``label + 1`` (the
  position-based fallback, documented on the generated file).
- ``class_thresholds`` overrides ``threshold`` per class (keys: class
  names, or the **category ids** written into the generated json — NOT the
  model's internal label indices) — the long-tail knob: keep rare classes
  only at higher confidence.
- ``file_name`` stays relative to the images directory when possible
  (absolute otherwise), so the split root can be the images directory.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "PseudoLabelResult",
    "generate_pseudo_labels",
    "list_images",
]

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(images: Path | str) -> list[Path]:
    """Image files at ``images`` (a directory or a single file), sorted.

    Directories are walked RECURSIVELY — the training data plane lists
    nested pools the same way, so a subdir-organized pool must not be
    silently truncated to its top level.
    """
    root = Path(images)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Images path does not exist: '{root}'.")
    paths = sorted(
        path for path in root.rglob("*") if path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No images found at '{root}'.")
    return paths


@dataclass(frozen=True)
class PseudoLabelResult:
    """What one pseudo-labeling pass produced (for the self-training log)."""

    out_json: Path
    images: int
    annotations: int
    per_class: dict[str, int] = field(default_factory=dict)
    score_min: float = 0.0
    score_mean: float = 0.0
    score_max: float = 0.0
    threshold: float = 0.0
    sahi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_json": str(self.out_json),
            "images": self.images,
            "annotations": self.annotations,
            "per_class": dict(self.per_class),
            "score_min": self.score_min,
            "score_mean": self.score_mean,
            "score_max": self.score_max,
            "threshold": self.threshold,
            "sahi": self.sahi,
        }


def _batched(items: Sequence[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _category_pairs(
    categories: Mapping[int, str] | Sequence[str] | None,
) -> list[tuple[int, str]]:
    """Category (id, name) pairs by convention.

    With a mapping: sorted keys are the internal label order and the keys
    double as category ids (the ``predictions_to_coco`` convention). With
    a sequence: ids are 1..N by position (the ``convert.py`` convention).
    Without one: ids are ``label + 1`` over the labels actually seen.
    """
    if categories is None:
        return []
    if isinstance(categories, Mapping):
        return sorted((int(k), str(v)) for k, v in categories.items())
    return [(index + 1, str(name)) for index, name in enumerate(categories)]


def _resolve_class_thresholds(
    class_thresholds: Mapping[str | int, float] | None,
    pairs: list[tuple[int, str]],
    threshold: float,
) -> dict[int, float]:
    """Category id -> threshold (default = ``threshold`` for every class)."""
    resolved = {category_id: float(threshold) for category_id, _ in pairs}
    if not class_thresholds:
        return resolved
    by_name = {name: category_id for category_id, name in pairs}
    known_ids = set(resolved)
    for key, value in class_thresholds.items():
        if isinstance(key, int) and key in known_ids:
            resolved[key] = float(value)
        elif isinstance(key, str) and key in by_name:
            resolved[by_name[key]] = float(value)
        else:
            raise ValueError(
                f"class_thresholds key '{key}' matches no category "
                f"(known names: {sorted(by_name)}; known ids: {sorted(known_ids)})."
            )
    return resolved


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as handle:
        width, height = handle.size
    return int(width), int(height)


def _relative_name(path: Path, root: Path) -> str:
    """``file_name`` relative to the images root when inside it, else absolute."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def generate_pseudo_labels(
    model: Any,
    images: Path | str,
    out_json: Path | str,
    *,
    threshold: float = 0.5,
    class_thresholds: Mapping[str | int, float] | None = None,
    categories: Mapping[int, str] | Sequence[str] | None = None,
    sahi: bool = False,
    tile_size: int | None = None,
    overlap: float = 0.2,
    nms_iou_threshold: float = 0.3,
    global_local_iou_threshold: float = 0.1,
    batch_size: int = 8,
    max_images: int | None = None,
    image_paths: Sequence[Path | str] | None = None,
) -> PseudoLabelResult:
    """Write COCO instance annotations for ``images`` from ``model``.

    ``model`` is a detection task model (``load_task_model`` /
    ``autovision.detection(...)`` / an exported ``.pt``); it is set to
    eval and never mutated beyond that. ``image_paths`` bypasses directory
    listing (the caller keeps its own ordering). SAHI runs one tile pass
    per image through the model's ``predict_sahi``.
    """
    root = Path(images)
    if image_paths is not None:
        paths = [Path(path) for path in image_paths]
    else:
        paths = list_images(root)
    if max_images is not None:
        paths = paths[: int(max_images)]
    if not paths:
        raise ValueError("No images to pseudo-label.")

    pairs = _category_pairs(categories)
    per_class_thresholds = _resolve_class_thresholds(
        class_thresholds, pairs, threshold
    )
    # SAHI filters internally with one global threshold; pass the LOWEST
    # active threshold so nothing a per-class override would keep is dropped
    # before the per-class filter below runs.
    sahi_threshold = min(
        [float(threshold), *per_class_thresholds.values()]
    )

    coco_images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    per_class_counts: dict[str, int] = {}
    scores_kept: list[float] = []
    annotation_id = 1
    seen_labels: set[int] = set()

    def _emit(image_id: int, result: dict[str, Any]) -> None:
        nonlocal annotation_id
        boxes = result.get("bboxes", result.get("boxes"))
        scores = result.get("scores")
        labels = result.get("labels")
        if boxes is None or scores is None or labels is None:
            raise TypeError(
                "generate_pseudo_labels needs a detection task model "
                "(postprocess output with bboxes/scores/labels tensors)."
            )
        for box, score, label in zip(boxes, scores, labels):
            internal = int(label)
            seen_labels.add(internal)
            category_id = _category_id(internal, pairs)
            name = _category_name(internal, pairs)
            score_value = float(score)
            if score_value < per_class_thresholds.get(category_id, float(threshold)):
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            width_ = max(x2 - x1, 1.0)
            height_ = max(y2 - y1, 1.0)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, width_, height_],
                    "area": width_ * height_,
                    "iscrowd": 0,
                    "score": score_value,
                }
            )
            annotation_id += 1
            per_class_counts[name] = per_class_counts.get(name, 0) + 1
            scores_kept.append(score_value)

    batch_size = max(1, int(batch_size))
    for chunk in _batched(paths, batch_size):
        if sahi:
            results = [
                model.predict_sahi(
                    path,
                    tile_size=tile_size,
                    overlap=overlap,
                    threshold=sahi_threshold,
                    nms_iou_threshold=nms_iou_threshold,
                    global_local_iou_threshold=global_local_iou_threshold,
                )
                for path in chunk
            ]
        else:
            results = model.predict_batch([str(path) for path in chunk], threshold=0.0)
        for path, result in zip(chunk, results):
            image_id = len(coco_images) + 1
            width, height = _image_size(path)
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": _relative_name(path, root),
                    "width": width,
                    "height": height,
                }
            )
            _emit(image_id, result)

    if pairs:
        category_rows = [
            {"id": category_id, "name": name} for category_id, name in pairs
        ]
    else:
        category_rows = [
            {"id": internal + 1, "name": f"class_{internal}"}
            for internal in sorted(seen_labels)
        ]
    payload = {
        "images": coco_images,
        "annotations": annotations,
        "categories": category_rows,
        "info": {
            "source": "autovision.pseudo_label",
            "threshold": float(threshold),
            "sahi": bool(sahi),
            "category_mapping": (
                "sorted(categories) internal order" if pairs else "label + 1"
            ),
            **(
                {"class_thresholds": dict(class_thresholds)}
                if class_thresholds
                else {}
            ),
        },
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload))

    return PseudoLabelResult(
        out_json=out_json,
        images=len(coco_images),
        annotations=len(annotations),
        per_class=per_class_counts,
        score_min=min(scores_kept) if scores_kept else 0.0,
        score_mean=(
            sum(scores_kept) / len(scores_kept) if scores_kept else 0.0
        ),
        score_max=max(scores_kept) if scores_kept else 0.0,
        threshold=float(threshold),
        sahi=bool(sahi),
    )


def _category_id(internal: int, pairs: list[tuple[int, str]]) -> int:
    if pairs:
        if not 0 <= internal < len(pairs):
            raise ValueError(
                f"Prediction label id {internal} is outside the declared "
                f"category list (0..{len(pairs) - 1})."
            )
        return pairs[internal][0]
    return internal + 1


def _category_name(internal: int, pairs: list[tuple[int, str]]) -> str:
    if pairs:
        return pairs[internal][1]
    return f"class_{internal}"
