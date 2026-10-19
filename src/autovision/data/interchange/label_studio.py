# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Label Studio JSON export -> COCO (rectangle / polygon / keypoint)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from autovision.data.interchange.common import CocoBuilder, polygon_area
from autovision.errors import AutoVisionError

__all__ = ["label_studio_to_coco"]


def _image_name(task: dict[str, Any]) -> str:
    data = task.get("data") or {}
    raw = data.get("image") or data.get("img") or data.get("image_url")
    if not raw:
        raise AutoVisionError("Label Studio task has no data.image path.")
    name = str(raw)
    parsed = urlparse(name)
    if parsed.query:
        query = parse_qs(parsed.query)
        if "d" in query:
            name = query["d"][0]
    return unquote(name)


def _dimensions(
    result: dict[str, Any], image_name: str, images_dir: Path | None
) -> tuple[int, int]:
    width = result.get("original_width") or result.get("value", {}).get("original_width")
    height = result.get("original_height") or result.get("value", {}).get("original_height")
    if width and height:
        return int(width), int(height)
    path = Path(image_name)
    if images_dir is not None and not path.is_absolute():
        path = images_dir / path
    try:
        from PIL import Image

        with Image.open(path) as handle:
            return int(handle.width), int(handle.height)
    except Exception as error:  # noqa: BLE001 - LS should carry dimensions
        raise AutoVisionError(
            f"Cannot determine dimensions for Label Studio image "
            f"'{image_name}': {error}."
        ) from error


def _percent_box(value: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float]:
    x = float(value["x"]) / 100.0 * width
    y = float(value["y"]) / 100.0 * height
    box_width = float(value["width"]) / 100.0 * width
    box_height = float(value["height"]) / 100.0 * height
    return x, y, box_width, box_height


def _point(value: dict[str, Any], width: int, height: int) -> tuple[float, float]:
    return float(value["x"]) / 100.0 * width, float(value["y"]) / 100.0 * height


def label_studio_to_coco(
    json_path: str | Path,
    out_json: str | Path,
    *,
    images_dir: str | Path | None = None,
    keypoint_radius: float = 2.0,
) -> Path:
    """Convert a Label Studio JSON export to COCO.

    Rectangle labels use Label Studio's percentage coordinates and are
    converted with each result's ``original_width``/``original_height``.
    Polygons keep COCO segmentation. Keypoints become a small box centered
    on the point (``keypoint_radius`` pixels) with the raw point in
    ``keypoint``; detection training can consume them, pose training should
    use a native keypoint format.
    """
    source = Path(json_path)
    try:
        payload = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AutoVisionError(
            f"Cannot read Label Studio JSON '{source}': {error}"
        ) from error
    if isinstance(payload, dict) and "tasks" in payload:
        tasks = payload["tasks"]
    else:
        tasks = payload
    if not isinstance(tasks, list):
        raise AutoVisionError("Label Studio export must be a task list.")
    if keypoint_radius <= 0:
        raise AutoVisionError("keypoint_radius must be > 0.")

    root_dir = Path(images_dir) if images_dir is not None else None
    builder = CocoBuilder(images_dir=root_dir)
    for task in tasks:
        if not isinstance(task, dict):
            raise AutoVisionError("Label Studio task entries must be objects.")
        image_name = _image_name(task)
        annotations = task.get("annotations") or task.get("predictions") or []
        results: list[dict[str, Any]] = []
        for annotation in annotations:
            results.extend(annotation.get("result", []))
        if results:
            width, height = _dimensions(results[0], image_name, root_dir)
        else:
            width, height = _dimensions({}, image_name, root_dir)
        image_id = builder.add_image(image_name, width, height)

        for result in results:
            result_type = result.get("type")
            value = result.get("value") or {}
            if result_type == "rectanglelabels":
                labels = value.get("rectanglelabels") or []
                if not labels:
                    continue
                x, y, box_width, box_height = _percent_box(value, width, height)
                extra: dict[str, Any] = {}
                rotation = float(value.get("rotation", 0.0))
                if rotation:
                    extra["rotation"] = rotation
                for label in labels:
                    builder.add_annotation(
                        image_id, str(label), (x, y, box_width, box_height),
                        extra=extra,
                    )
            elif result_type == "polygonlabels":
                labels = value.get("polygonlabels") or []
                points = value.get("points") or []
                if not labels or len(points) < 3:
                    continue
                pixels = [
                    (
                        float(point[0]) / 100.0 * width,
                        float(point[1]) / 100.0 * height,
                    )
                    for point in points
                ]
                xs = [point[0] for point in pixels]
                ys = [point[1] for point in pixels]
                flat = [coordinate for point in pixels for coordinate in point]
                for label in labels:
                    builder.add_annotation(
                        image_id,
                        str(label),
                        (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)),
                        segmentation=[flat],
                        area=polygon_area(pixels),
                    )
            elif result_type == "keypointlabels":
                labels = value.get("keypointlabels") or []
                if not labels:
                    continue
                x, y = _point(value, width, height)
                x1 = max(0.0, x - keypoint_radius)
                y1 = max(0.0, y - keypoint_radius)
                x2 = min(float(width), x + keypoint_radius)
                y2 = min(float(height), y + keypoint_radius)
                if x2 <= x1 or y2 <= y1:
                    continue
                for label in labels:
                    builder.add_annotation(
                        image_id,
                        str(label),
                        (x1, y1, x2 - x1, y2 - y1),
                        extra={"keypoint": [x, y]},
                    )
            else:
                warnings.warn(
                    f"Label Studio result type '{result_type}' is not "
                    "converted (rectanglelabels/polygonlabels/keypointlabels only).",
                    stacklevel=2,
                )
    return builder.write(out_json)
