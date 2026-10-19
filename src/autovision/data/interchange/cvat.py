# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""CVAT for Images 1.1 XML -> COCO (bbox + polygon)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from defusedxml import ElementTree as ET

from autovision.data.interchange.common import CocoBuilder, polygon_area
from autovision.errors import AutoVisionError

__all__ = ["cvat_to_coco"]


def _points(raw: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split(",")
        if len(parts) != 2:
            raise AutoVisionError(f"Malformed CVAT polygon point: '{pair}'.")
        points.append((float(parts[0]), float(parts[1])))
    if len(points) < 3:
        raise AutoVisionError("CVAT polygon needs at least three points.")
    return points


def _image_size(
    image: ET.Element, file_name: str, images_dir: Path | None
) -> tuple[int, int]:
    width, height = image.get("width"), image.get("height")
    if width is not None and height is not None:
        return int(float(width)), int(float(height))
    if images_dir is None:
        raise AutoVisionError(
            f"CVAT image '{file_name}' has no width/height and no images_dir "
            "was supplied; refusing to open an arbitrary path from the XML."
        )
    base = Path(images_dir).resolve()
    path = Path(file_name)
    path = path if path.is_absolute() else base / path
    resolved = path.resolve()
    if resolved != base and base not in resolved.parents:
        raise AutoVisionError(
            f"CVAT image path '{file_name}' escapes images_dir '{base}'."
        )
    try:
        from PIL import Image

        with Image.open(resolved) as handle:
            return int(handle.width), int(handle.height)
    except Exception as error:  # noqa: BLE001 - CVAT must provide a size
        raise AutoVisionError(
            f"Cannot determine size for image '{file_name}' (CVAT width/height "
            f"missing and the file is unreadable: {error})."
        ) from error


def cvat_to_coco(
    xml_path: str | Path,
    out_json: str | Path,
    *,
    images_dir: str | Path | None = None,
) -> Path:
    """Convert a CVAT ``annotations.xml`` to COCO detection/segmentation.

    Supports ``<box>`` (absolute pixel xywh) and ``<polygon>`` (COCO
    segmentation + derived bbox/area). Other shapes are ignored; the image
    still appears in the COCO ``images`` list.
    """
    source = Path(xml_path)
    try:
        root = ET.parse(source).getroot()
    except (OSError, ET.ParseError) as error:
        raise AutoVisionError(f"Cannot parse CVAT XML '{source}': {error}") from error
    root_dir = Path(images_dir) if images_dir is not None else None
    builder = CocoBuilder(images_dir=root_dir)

    labels = {
        str(label.get("name"))
        for label in root.findall("./meta/task/labels/label")
        if label.get("name")
    }
    for element in root.findall("image"):
        file_name = element.get("name")
        if not file_name:
            raise AutoVisionError("CVAT image element is missing a 'name'.")
        width, height = _image_size(element, file_name, root_dir)
        image_id = builder.add_image(file_name, width, height)
        for box in element.findall("box"):
            label = box.get("label")
            if not label:
                raise AutoVisionError(f"CVAT box in '{file_name}' has no label.")
            try:
                xtl = float(box.get("xtl", "nan"))
                ytl = float(box.get("ytl", "nan"))
                xbr = float(box.get("xbr", "nan"))
                ybr = float(box.get("ybr", "nan"))
            except ValueError as error:
                raise AutoVisionError(
                    f"CVAT box in '{file_name}' has malformed coordinates."
                ) from error
            extra: dict[str, Any] = {}
            if box.get("occluded") is not None:
                extra["occluded"] = int(float(box.get("occluded", "0")))
            builder.add_annotation(
                image_id,
                label,
                (xtl, ytl, xbr - xtl, ybr - ytl),
                extra=extra,
            )
            labels.add(label)
        for polygon in element.findall("polygon"):
            label = polygon.get("label")
            if not label:
                raise AutoVisionError(
                    f"CVAT polygon in '{file_name}' has no label."
                )
            points = _points(polygon.get("points", ""))
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            builder.add_annotation(
                image_id,
                label,
                (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)),
                segmentation=[[coordinate for point in points for coordinate in point]],
                area=polygon_area(points),
            )
            labels.add(label)

    # Keep the meta label table in the builder's category set even when an
    # image has no instances: an empty class is still a dataset class.
    for label in labels:
        builder.add_category(label)
    return builder.write(out_json)
