# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Annotation conversion between on-disk formats (filesystem-only).

Directions: YOLO detection <-> COCO, plus the C3/C4 extensions of the
data loop — PASCAL VOC detection -> COCO, YOLO-pose -> COCO keypoints,
YOLO-seg polygons -> COCO instance segmentation, and COCO merging (for
labeled + pseudo-labeled self-training rounds).

All directions keep images in place; only annotation artifacts are
written. COCO category ids are arbitrary; the YOLO side uses 0-based
contiguous ids assigned by sorted COCO category id (the same convention
the data layer's class mapping documents).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger("autovision")

__all__ = [
    "coco_to_yolo",
    "merge_coco",
    "voc_to_coco",
    "yolo_pose_to_coco",
    "yolo_seg_to_coco",
    "yolo_to_coco",
]

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sorted_categories(
    categories: Sequence[Mapping[str, Any]] | Mapping[int, str] | Sequence[str],
) -> list[tuple[int, str]]:
    """Normalize the accepted ``names`` shapes to sorted (id, name) pairs."""
    if isinstance(categories, Mapping):
        return sorted((int(k), str(v)) for k, v in categories.items())
    return [(index + 1, str(name)) for index, name in enumerate(categories)]


def _categories_from_payload(payload: Mapping[str, Any]) -> list[tuple[int, str]]:
    """Read a COCO json ``categories`` list into sorted (id, name) pairs."""
    return sorted(
        (int(c["id"]), str(c["name"])) for c in payload.get("categories", [])
    )


def yolo_to_coco(
    images_dir: Path | str,
    names: Mapping[int, str] | Sequence[str],
    out_json: Path | str,
    labels_dir: Path | str | None = None,
) -> Path:
    """Convert a YOLO layout (images + per-image label txts) to COCO json.

    ``names`` maps YOLO class indices to names (dict) or is a list whose
    positions are the indices. Label lines are ``cls cx cy w h`` with
    boxes normalized to [0, 1]; missing label files yield images with no
    annotations (valid for a background-heavy split).
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir is not None else images_dir.parent / "labels"
    category_pairs = _sorted_categories(names)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
    )
    for image_id, image_path in enumerate(image_paths, start=1):
        from PIL import Image

        with Image.open(image_path) as handle:
            width, height = handle.size
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": int(width),
                "height": int(height),
            }
        )
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"'{label_path}': expected 'cls cx cy w h', got '{line}'"
                )
            class_index = int(parts[0])
            if not 0 <= class_index < len(category_pairs):
                raise ValueError(
                    f"'{label_path}': class index {class_index} is outside "
                    f"'names' ({len(category_pairs)} entries)."
                )
            cx, cy, w, h = (float(v) for v in parts[1:])
            x = (cx - w / 2) * width
            y = (cy - h / 2) * height
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_pairs[class_index][0],
                    "bbox": [x, y, w * width, h * height],
                    "area": w * width * h * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name} for category_id, name in category_pairs
        ],
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload))
    return out_json


def merge_coco(
    annotations_jsons: Sequence[Path | str],
    out_json: Path | str,
) -> Path:
    """Merge COCO detection jsons into one (self-training: labeled + pseudo).

    Image ids and annotation ids are renumbered to stay unique; ``file_name``
    values are carried over verbatim, so every input must be resolvable from
    one shared images root (use absolute ``file_name``s when the inputs live
    in different roots — the training split takes one images directory).
    Categories are unioned by NAME: the first file's id wins per name, and
    ids that conflict with an earlier file's different name are reassigned
    to the next free id. Returns the written path.
    """
    inputs = [Path(path) for path in annotations_jsons]
    if not inputs:
        raise ValueError("merge_coco needs at least one annotations json.")

    name_to_id: dict[str, int] = {}
    id_to_name: dict[int, str] = {}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1

    for path in inputs:
        payload = json.loads(path.read_text())
        # Union categories by name (ids are arbitrary in COCO).
        id_map: dict[int, int] = {}
        for category in payload.get("categories", []):
            source_id = int(category["id"])
            name = str(category["name"])
            if name not in name_to_id:
                candidate = source_id
                while candidate in id_to_name and id_to_name[candidate] != name:
                    candidate += 1
                name_to_id[name] = candidate
                id_to_name[candidate] = name
            id_map[source_id] = name_to_id[name]
        image_map: dict[int, int] = {}
        for image in payload.get("images", []):
            source_image_id = int(image["id"])
            image_map[source_image_id] = next_image_id
            row = dict(image)
            row["id"] = next_image_id
            images.append(row)
            next_image_id += 1
        dropped = 0
        for annotation in payload.get("annotations", []):
            source_image_id = int(annotation["image_id"])
            if source_image_id not in image_map:
                dropped += 1
                continue
            source_category_id = int(annotation["category_id"])
            if source_category_id not in id_map:
                dropped += 1
                continue
            row = dict(annotation)
            row["id"] = next_annotation_id
            row["image_id"] = image_map[source_image_id]
            row["category_id"] = id_map[source_category_id]
            annotations.append(row)
            next_annotation_id += 1
        if dropped:
            logger.warning(
                "merge_coco: '%s' contributed %d annotation(s) whose image_id "
                "or category_id is undeclared in that file — dropped.",
                path,
                dropped,
            )

    merged = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": id_to_name[category_id]}
            for category_id in sorted(id_to_name)
        ],
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(merged))
    return out_json


def coco_to_yolo(
    annotations_json: Path | str,
    out_labels_dir: Path | str,
    names_out: Path | str | None = None,
) -> Path:
    """Convert a COCO detection json to YOLO label txts.

    Writes one ``<image_stem>.txt`` per annotated image with
    ``cls cx cy w h`` lines (normalized, YOLO class index = position of
    the COCO category id in sorted order). ``names_out`` (if given)
    receives one class name per line in that same order. Image ids with
    no annotations produce no file, mirroring ultralytics-style layouts.
    """
    payload = json.loads(Path(annotations_json).read_text())
    category_pairs = _categories_from_payload(payload)
    id_to_index = {category_id: index for index, (category_id, _) in enumerate(category_pairs)}

    sizes = {
        int(image["id"]): (int(image["width"]), int(image["height"]))
        for image in payload.get("images", [])
    }
    stems = {
        int(image["id"]): Path(image["file_name"]).stem
        for image in payload.get("images", [])
    }
    lines_by_image: dict[int, list[str]] = {}
    for annotation in payload.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id not in sizes:
            continue
        x, y, w, h = (float(v) for v in annotation["bbox"])
        width, height = sizes[image_id]
        cx = (x + w / 2) / width
        cy = (y + h / 2) / height
        class_index = id_to_index[int(annotation["category_id"])]
        lines_by_image.setdefault(image_id, []).append(
            f"{class_index} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}"
        )

    out_labels_dir = Path(out_labels_dir)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    for image_id, lines in lines_by_image.items():
        (out_labels_dir / f"{stems[image_id]}.txt").write_text(
            "\n".join(lines) + "\n"
        )
    if names_out is not None:
        Path(names_out).write_text(
            "\n".join(name for _, name in category_pairs) + "\n"
        )
    return out_labels_dir


def _read_image_list(split_file: Path) -> list[str]:
    """Image stems from a VOC ``ImageSets/Main/*.txt`` style list file."""
    stems = []
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if line:
            stems.append(line.split()[0])
    return stems


def _image_file_for(images_dir: Path, stem: str) -> Path:
    for suffix in sorted(_IMAGE_SUFFIXES):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image for '{stem}' under '{images_dir}'.")


def voc_to_coco(
    voc_root: Path | str,
    out_json: Path | str,
    *,
    images_dir: Path | str | None = None,
    annotations_dir: Path | str | None = None,
    imagesets: str | None = None,
    split_file: Path | str | None = None,
    class_names: Sequence[str] | None = None,
) -> Path:
    """Convert a PASCAL VOC detection tree to COCO json.

    ``voc_root`` is the year directory (``VOC2007/``): XML lives in
    ``Annotations/``, images in ``JPEGImages/``, and the split lists in
    ``ImageSets/Main/<name>.txt`` (pass ``imagesets="train"`` — or an
    explicit ``split_file`` — to restrict to a split; without one every
    XML is converted, ordered by stem).

    Conventions:

    - VOC boxes are 1-based inclusive (``xmin``/``ymin``/``xmax``/``ymax``);
      the COCO ``bbox`` is ``[xmin-1, ymin-1, xmax-xmin+1, ymax-ymin+1]``
      (the standard lossless integer convention). Objects with a
      non-positive width/height are dropped.
    - ``difficult`` is carried as ``iscrowd=1`` — the official COCO
      protocol treats crowd boxes as ignore regions (no recall
      requirement), which is the closest semantic match; ``difficult`` /
      ``truncated`` flags also stay on the annotation for provenance.
    - Category ids are ``1..N`` over sorted class names, or over
      ``class_names`` when given (VOC2007 has 20 classes).
    """
    voc_root = Path(voc_root)
    annotations_dir = (
        Path(annotations_dir) if annotations_dir is not None else voc_root / "Annotations"
    )
    images_dir = Path(images_dir) if images_dir is not None else voc_root / "JPEGImages"
    if split_file is None and imagesets is not None:
        split_file = voc_root / "ImageSets" / "Main" / f"{imagesets}.txt"

    stems: list[str] | None = None
    if split_file is not None:
        stems = _read_image_list(Path(split_file))
    xmls = sorted(annotations_dir.glob("*.xml"))
    if stems is not None:
        by_stem = {path.stem: path for path in xmls}
        xmls = [by_stem[stem] for stem in stems if stem in by_stem]
    if not xmls:
        raise FileNotFoundError(
            f"No VOC annotations found under '{annotations_dir}'"
            + (f" for split '{split_file}'." if split_file is not None else ".")
        )

    from defusedxml import ElementTree as ET

    def _local(tag: str) -> str:
        """Element local name (tolerates default/xsi namespaces)."""
        return tag.rsplit("}", 1)[-1]

    def _find(parent: Any, name: str) -> Any:
        for child in parent:
            if _local(child.tag) == name:
                return child
        return None

    def _findall(parent: Any, name: str) -> list[Any]:
        return [child for child in parent if _local(child.tag) == name]

    def _text(parent: Any, name: str, default: str | None = None) -> str | None:
        node = _find(parent, name)
        return node.text if node is not None and node.text is not None else default

    parsed: list[tuple[str, int, int, list[dict[str, Any]]]] = []
    seen_names: set[str] = set()
    for xml in xmls:
        root = ET.parse(xml).getroot()
        size = _find(root, "size")
        if size is None:
            raise ValueError(f"'{xml}' has no <size> element.")
        width = int(float(_text(size, "width")))
        height = int(float(_text(size, "height")))
        objects = []
        for obj in _findall(root, "object"):
            name = (_text(obj, "name") or "").strip()
            if not name:
                continue
            seen_names.add(name)
            # VOC spells the box element <bndbox>; <bbox> is a
            # tolerated variant of sloppy exporters.
            box = _find(obj, "bndbox")
            if box is None:
                box = _find(obj, "bbox")
            if box is None:
                continue
            xmin = float(_text(box, "xmin"))
            ymin = float(_text(box, "ymin"))
            xmax = float(_text(box, "xmax"))
            ymax = float(_text(box, "ymax"))
            w = xmax - xmin + 1.0
            h = ymax - ymin + 1.0
            if w <= 0 or h <= 0:
                continue
            difficult = int(float(_text(obj, "difficult") or "0")) == 1
            objects.append(
                {
                    "name": name,
                    "bbox": [xmin - 1.0, ymin - 1.0, w, h],
                    "difficult": int(difficult),
                    "truncated": int(float(_text(obj, "truncated") or "0") == 1),
                }
            )
        parsed.append((xml.stem, width, height, objects))

    if class_names is not None:
        pairs = [(index + 1, str(name)) for index, name in enumerate(class_names)]
    else:
        pairs = [(index + 1, name) for index, name in enumerate(sorted(seen_names))]
    name_to_id = {name: category_id for category_id, name in pairs}

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, (stem, width, height, objects) in enumerate(parsed, start=1):
        image_path = _image_file_for(images_dir, stem)
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        for obj in objects:
            if obj["name"] not in name_to_id:
                continue
            x, y, w, h = obj["bbox"]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": name_to_id[obj["name"]],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": obj["difficult"],
                    "difficult": obj["difficult"],
                    "truncated": obj["truncated"],
                }
            )
            annotation_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name} for category_id, name in pairs
        ],
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload))
    return out_json


def _shoelace_area(points: Sequence[float]) -> float:
    """Absolute polygon area (shoelace) from flat ``x0 y0 x1 y1 ...``."""
    xs, ys = points[0::2], points[1::2]
    total = 0.0
    count = len(xs)
    for index in range(count):
        nxt = (index + 1) % count
        total += xs[index] * ys[nxt] - xs[nxt] * ys[index]
    return abs(total) / 2.0


def _yolo_rows(images_dir: Path, labels_dir: Path, expected_extra: int | None = None):
    """Yield ``(image_path, width, height, class_index, values)`` per label row.

    ``expected_extra`` pins the per-row value count when the format fixes
    it (pose: ``4 + 3K``); segmentation rows vary (2 * num_points).
    """
    paths = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES
    )
    rows = []
    for image_path in paths:
        from PIL import Image

        with Image.open(image_path) as handle:
            width, height = handle.size
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            class_index = int(parts[0])
            values = [float(value) for value in parts[1:]]
            if expected_extra is not None and len(values) != expected_extra:
                raise ValueError(
                    f"'{label_path}' line {line_number}: expected "
                    f"{expected_extra} values after the class index, got {len(values)}."
                )
            rows.append((image_path, width, height, class_index, values))
    return rows


def yolo_pose_to_coco(
    images_dir: Path | str,
    names: Mapping[int, str] | Sequence[str],
    out_json: Path | str,
    *,
    num_keypoints: int = 17,
    labels_dir: Path | str | None = None,
) -> Path:
    """Convert YOLO-pose labels to a COCO keypoints json.

    Row format (Ultralytics): ``cls cx cy w h px1 py1 pv1 … pxK pyK pvK``,
    everything normalized, ``K == num_keypoints``. ``pv == 0`` marks an
    unlabeled point (coordinates zeroed in COCO); the annotation's
    ``num_keypoints`` counts visible points so the keypoint dataset's
    official filter (``num_keypoints > 0``) applies.
    """
    images_dir = Path(images_dir)
    labels_dir = (
        Path(labels_dir) if labels_dir is not None else images_dir.parent / "labels"
    )
    category_pairs = _sorted_categories(names)
    rows = _yolo_rows(images_dir, labels_dir, expected_extra=4 + 3 * int(num_keypoints))

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_ids: dict[str, int] = {}
    annotation_id = 1
    for image_path, width, height, class_index, values in rows:
        if not 0 <= class_index < len(category_pairs):
            raise ValueError(
                f"class index {class_index} is outside 'names' "
                f"({len(category_pairs)} entries)."
            )
        image_id = image_ids.get(image_path.name)
        if image_id is None:
            image_id = len(image_ids) + 1
            image_ids[image_path.name] = image_id
            images.append(
                {
                    "id": image_id,
                    "file_name": image_path.name,
                    "width": int(width),
                    "height": int(height),
                }
            )
        cx, cy, w, h = (value * size for value, size in zip(values[:4], (width, height, width, height)))
        keypoint_values: list[float] = []
        visible = 0
        for kp in range(int(num_keypoints)):
            px = values[4 + 3 * kp] * width
            py = values[4 + 3 * kp + 1] * height
            pv = int(values[4 + 3 * kp + 2])
            if pv > 0:
                visible += 1
                keypoint_values.extend([px, py, float(pv)])
            else:
                keypoint_values.extend([0.0, 0.0, 0.0])
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_pairs[class_index][0],
                "bbox": [cx - w / 2, cy - h / 2, w, h],
                "area": w * h,
                "iscrowd": 0,
                "num_keypoints": visible,
                "keypoints": keypoint_values,
            }
        )
        annotation_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name} for category_id, name in category_pairs
        ],
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload))
    return out_json


def yolo_seg_to_coco(
    images_dir: Path | str,
    names: Mapping[int, str] | Sequence[str],
    out_json: Path | str,
    *,
    labels_dir: Path | str | None = None,
) -> Path:
    """Convert YOLO-seg polygon labels to a COCO instance-segmentation json.

    Row format (Ultralytics): ``cls x1 y1 x2 y2 … xn yn`` (normalized,
    one polygon per row, ``n >= 3``). The COCO side gets absolute polygon
    ``segmentation``, a bbox from the polygon bounds, and the shoelace
    polygon ``area`` (the COCO convention).
    """
    images_dir = Path(images_dir)
    labels_dir = (
        Path(labels_dir) if labels_dir is not None else images_dir.parent / "labels"
    )
    category_pairs = _sorted_categories(names)
    rows = _yolo_rows(images_dir, labels_dir)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_ids: dict[str, int] = {}
    annotation_id = 1
    for image_path, width, height, class_index, values in rows:
        if not 0 <= class_index < len(category_pairs):
            raise ValueError(
                f"class index {class_index} is outside 'names' "
                f"({len(category_pairs)} entries)."
            )
        if len(values) % 2 != 0 or len(values) < 6:
            raise ValueError(
                f"'{image_path.stem}.txt': segmentation rows need >= 3 "
                f"normalized (x, y) points, got {len(values)} values."
            )
        image_id = image_ids.get(image_path.name)
        if image_id is None:
            image_id = len(image_ids) + 1
            image_ids[image_path.name] = image_id
            images.append(
                {
                    "id": image_id,
                    "file_name": image_path.name,
                    "width": int(width),
                    "height": int(height),
                }
            )
        polygon: list[float] = []
        for value, size in zip(values, (width, height) * (len(values) // 2)):
            polygon.append(float(value) * float(size))
        xs, ys = polygon[0::2], polygon[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_pairs[class_index][0],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": _shoelace_area(polygon),
                "iscrowd": 0,
                "segmentation": [polygon],
            }
        )
        annotation_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name} for category_id, name in category_pairs
        ],
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload))
    return out_json
