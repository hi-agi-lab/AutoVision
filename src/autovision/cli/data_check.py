# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""``autovision-cli data-check`` (roadmap 90d, phase 3).

Builds audit records from the same detection data config the trainers use,
optionally enriching them with the raw COCO annotation file (original
category ids and absolute xywh boxes), then writes the report via
``autovision.data.quality``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autovision.data.coco_object_detection_dataset import (
    COCOObjectDetectionDataArgs,
)
from autovision.data.quality import (
    DataAuditReport,
    ImageRecord,
    audit_detection_dataset,
)
from autovision.errors import AutoVisionError

__all__ = ["data_check"]


def _split_names(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("train", "val")
    if isinstance(value, str):
        names = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, (list, tuple)):
        names = tuple(str(part).strip() for part in value if str(part).strip())
    else:
        raise AutoVisionError("splits must be a comma-separated string or a list.")
    if not names:
        raise AutoVisionError("splits must not be empty.")
    return names


def _json_boxes(payload: str) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row)
        for row in json.loads(payload)
    )


def _raw_coco_index(
    annotations_path: Path, images_dir: Path
) -> tuple[dict[str, tuple[int | None, int | None, list[dict[str, Any]]]], dict[int, str]]:
    payload = json.loads(annotations_path.read_text())
    categories = {
        int(category["id"]): str(category["name"])
        for category in payload.get("categories", [])
    }
    by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in payload.get("annotations", []):
        by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    index: dict[str, tuple[int | None, int | None, list[dict[str, Any]]]] = {}
    for image in payload.get("images", []):
        path = (images_dir / str(image["file_name"])).resolve()
        index[str(path)] = (
            int(image["width"]) if image.get("width") is not None else None,
            int(image["height"]) if image.get("height") is not None else None,
            by_image.get(int(image["id"]), []),
        )
    return index, categories


def data_check(
    data: str | Path | dict[str, Any],
    out: str | Path,
    *,
    splits: Any = None,
    near_duplicate: bool = False,
    max_examples: int = 20,
) -> DataAuditReport:
    """Audit a detection dataset and write ``out/report.{json,md}``.

    ``data`` is the same COCO/YOLO config accepted by
    ``train_object_detection`` (dict, YAML path, or ``"coco8"``).
    """
    from autovision.cli.train_task.object_detection import (
        _DATA_FORMATS,
        _load_data_config,
    )
    from autovision.config.validate import pydantic_model_validate

    payload = _load_data_config(data)
    data_args = pydantic_model_validate(_DATA_FORMATS[payload["format"]], payload)
    base_dir = Path(data).parent if isinstance(data, (str, Path)) else None
    data_args.resolve_data_paths(base_dir)
    # Audit includes empty images; the training loader may skip them.
    if hasattr(data_args, "skip_if_annotations_missing"):
        data_args.skip_if_annotations_missing = False
    categories = data_args.classes()
    names = _split_names(splits)
    for split in names:
        if not hasattr(data_args, split):
            raise AutoVisionError(
                f"data config has no '{split}' split; available: "
                f"{[name for name in ('train', 'val') if hasattr(data_args, name)]}."
            )

    records: list[ImageRecord] = []
    raw_categories: dict[int, str] = {}
    for split in names:
        raw_index: dict[str, tuple[int | None, int | None, list[dict[str, Any]]]] = {}
        if isinstance(data_args, COCOObjectDetectionDataArgs):
            split_args = getattr(data_args, split)
            images_dir = Path(
                split_args.images or Path(split_args.annotations).parent
            )
            raw_index, split_categories = _raw_coco_index(
                Path(split_args.annotations), images_dir
            )
            for category_id, name in split_categories.items():
                raw_categories.setdefault(category_id, name)
        for row in data_args.list_image_info(split):
            path = Path(row["image_path"]).resolve()
            raw = raw_index.get(str(path))
            raw_category_ids = None
            raw_bboxes = None
            width = height = None
            if raw is not None:
                width, height, raw_annotations = raw
                raw_category_ids = tuple(
                    int(annotation["category_id"]) for annotation in raw_annotations
                )
                raw_bboxes = tuple(
                    tuple(float(value) for value in annotation["bbox"])
                    for annotation in raw_annotations
                )
            records.append(
                ImageRecord(
                    split=split,
                    path=path,
                    width=width,
                    height=height,
                    boxes=_json_boxes(row["bboxes"]),
                    labels=tuple(int(value) for value in json.loads(row["class_labels"])),
                    iscrowds=tuple(
                        int(value) for value in json.loads(row.get("iscrowds", "[]"))
                    ),
                    raw_category_ids=raw_category_ids,
                    raw_bboxes=raw_bboxes,
                )
            )

    report = audit_detection_dataset(
        records,
        categories=categories,
        raw_categories=raw_categories or None,
        near_duplicate=bool(near_duplicate),
        max_examples=int(max_examples),
    )
    return report.save(out)
