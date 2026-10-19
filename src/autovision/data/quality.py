# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Dataset quality audit (roadmap 90d, phase 3).

The audit consumes the same normalized rows the training loaders yield
(``image_path`` + JSON ``bboxes``/``class_labels``/``iscrowds``), plus the
raw COCO annotations when available for category-id and absolute-box
checks. It is deliberately format-agnostic at the core: the CLI builds the
records, this module only judges them.

Checks are split into errors (the dataset is unsafe to train/evaluate) and
warnings (likely mistakes / distribution risks). The report is written as
``report.json`` + ``report.md`` + ``manifest.json``; the CLI controls the
exit code via ``fail_on``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from PIL import Image

__all__ = [
    "AuditIssue",
    "DataAuditReport",
    "ImageRecord",
    "audit_detection_dataset",
]

Severity = Literal["error", "warning", "info"]
_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
_COCO_TOL = 1e-3


@dataclass(frozen=True)
class ImageRecord:
    """One image's boxes/labels in the loaders' normalized contract."""

    split: str
    path: Path
    width: int | None = None
    height: int | None = None
    boxes: tuple[tuple[float, float, float, float], ...] = ()
    labels: tuple[int, ...] = ()
    iscrowds: tuple[int, ...] = ()
    #: Original COCO category ids / absolute xywh boxes, when the CLI had
    #: the raw annotation file. Empty/None means "not available".
    raw_category_ids: tuple[int, ...] | None = None
    raw_bboxes: tuple[tuple[float, float, float, float], ...] | None = None


@dataclass(frozen=True)
class AuditIssue:
    split: str
    path: str
    kind: str
    severity: Severity
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "split": self.split,
            "path": self.path,
            "kind": self.kind,
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass
class DataAuditReport:
    """Structured audit result; paths are filled by :meth:`save`."""

    out_dir: Path
    summary: dict[str, Any] = field(default_factory=dict)
    issues: list[AuditIssue] = field(default_factory=list)
    per_class: dict[str, dict[str, int]] = field(default_factory=dict)
    per_split: dict[str, dict[str, int]] = field(default_factory=dict)
    report_json: Path | None = None
    report_md: Path | None = None
    manifest_json: Path | None = None
    max_examples: int = 20

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.severity == "warning" for issue in self.issues)

    def summary_text(self) -> str:
        return (
            f"images={self.summary.get('images', 0)} "
            f"boxes={self.summary.get('boxes', 0)} "
            f"errors={self.summary.get('errors', 0)} "
            f"warnings={self.summary.get('warnings', 0)} "
            f"splits={','.join(sorted(self.per_split))}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "summary": dict(self.summary),
            "per_split": {
                split: dict(values) for split, values in sorted(self.per_split.items())
            },
            "per_class": {
                name: dict(values) for name, values in sorted(self.per_class.items())
            },
            "issues": [issue.to_dict() for issue in self._sorted_issues()],
        }

    def save(self, out_dir: str | Path | None = None) -> "DataAuditReport":
        directory = Path(out_dir) if out_dir is not None else self.out_dir
        directory.mkdir(parents=True, exist_ok=True)
        self.out_dir = directory
        self.report_json = directory / "report.json"
        self.report_md = directory / "report.md"
        self.manifest_json = directory / "manifest.json"
        payload = self.to_dict()
        self.report_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        self.manifest_json.write_text(
            json.dumps(
                {"schema_version": 1, "issues": payload["issues"]},
                indent=2,
                sort_keys=True,
            )
        )
        self.report_md.write_text(self._render_markdown())
        return self

    def _sorted_issues(self) -> list[AuditIssue]:
        return sorted(
            self.issues,
            key=lambda issue: (
                _SEVERITY_ORDER.get(issue.severity, 9),
                issue.split,
                issue.path,
                issue.kind,
                issue.detail,
            ),
        )

    def _render_markdown(self) -> str:
        lines = ["# 数据体检报告\n"]
        lines.append(
            f"- 图像 {self.summary.get('images', 0)}；标注框 "
            f"{self.summary.get('boxes', 0)}；错误 "
            f"{self.summary.get('errors', 0)}；警告 "
            f"{self.summary.get('warnings', 0)}\n"
        )
        lines.append("## 分片\n")
        lines.append("| split | images | boxes | errors | warnings |")
        lines.append("|---|---|---|---|---|")
        for split, values in sorted(self.per_split.items()):
            lines.append(
                f"| {split} | {values.get('images', 0)} | "
                f"{values.get('boxes', 0)} | {values.get('errors', 0)} | "
                f"{values.get('warnings', 0)} |"
            )
        lines.append("\n## 类别分布\n")
        lines.append("| 类别 | " + " | ".join(sorted(self.per_split)) + " |")
        lines.append("|---|" + "---|" * len(self.per_split))
        for name, values in sorted(self.per_class.items()):
            lines.append(
                f"| {name} | "
                + " | ".join(str(values.get(split, 0)) for split in sorted(self.per_split))
                + " |"
            )
        issues = self._sorted_issues()
        lines.append("\n## 问题（按严重级别）\n")
        if not issues:
            lines.append("无。\n")
        else:
            lines.append("| severity | split | path | kind | detail |")
            lines.append("|---|---|---|---|---|")
            for issue in issues[: self.max_examples]:
                path = issue.path.replace("|", "\\|")
                detail = issue.detail.replace("|", "\\|")
                lines.append(
                    f"| {issue.severity} | {issue.split} | {path} | "
                    f"{issue.kind} | {detail} |"
                )
            if len(issues) > self.max_examples:
                lines.append(
                    f"\n_仅显示前 {self.max_examples} 条；完整清单见 manifest.json。_\n"
                )
        return "\n".join(lines)


def _iou(box: tuple[float, ...], other: tuple[float, ...]) -> float:
    x1 = max(box[0], other[0])
    y1 = max(box[1], other[1])
    x2 = min(box[2], other[2])
    y2 = min(box[3], other[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = max(0.0, other[2] - other[0]) * max(0.0, other[3] - other[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _average_hash(image: Image.Image, size: int = 8) -> int:
    gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    pixels = list(
        gray.get_flattened_data()
        if hasattr(gray, "get_flattened_data")
        else gray.getdata()
    )
    average = sum(pixels) / len(pixels)
    bits = 0
    for value in pixels:
        bits = (bits << 1) | (1 if value >= average else 0)
    return bits


def _hamming(left: int, right: int) -> int:
    return int(left ^ right).bit_count()


def _norm_box_xyxy(
    box: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    cx, cy, width, height = box
    return (
        cx - width / 2.0,
        cy - height / 2.0,
        cx + width / 2.0,
        cy + height / 2.0,
    )


def _add_issue(
    issues: list[AuditIssue],
    per_split: dict[str, dict[str, int]],
    *,
    split: str,
    path: Path,
    kind: str,
    severity: Severity,
    detail: str,
) -> None:
    issues.append(
        AuditIssue(
            split=split,
            path=str(path),
            kind=kind,
            severity=severity,
            detail=detail,
        )
    )
    key = f"{severity}s"
    if key in per_split[split]:
        per_split[split][key] += 1


def audit_detection_dataset(
    records: Sequence[ImageRecord],
    *,
    categories: Mapping[int, str],
    raw_categories: Mapping[int, str] | None = None,
    near_duplicate: bool = False,
    near_duplicate_limit: int = 2000,
    max_examples: int = 20,
) -> DataAuditReport:
    """Audit normalized detection records and write-ready report data.

    ``categories`` is the internal contiguous label map used by the
    loaders; ``raw_categories`` (COCO original ids) is optional and enables
    original category-id checks. Near-duplicate scanning is O(n²) and is
    skipped (with an info issue) above ``near_duplicate_limit``.
    """
    split_names = sorted({record.split for record in records})
    per_split = {
        split: {"images": 0, "boxes": 0, "errors": 0, "warnings": 0, "infos": 0}
        for split in split_names
    }
    per_class: dict[str, dict[str, int]] = {}
    issues: list[AuditIssue] = []
    total_boxes = 0
    digests: dict[str, list[tuple[str, Path]]] = {}
    hashes: list[tuple[str, Path, int]] = []
    image_sizes: list[tuple[int, int]] = []

    for record in records:
        split = record.split
        path = Path(record.path)
        per_split.setdefault(
            split, {"images": 0, "boxes": 0, "errors": 0, "warnings": 0, "infos": 0}
        )
        per_split[split]["images"] += 1
        width, height = record.width, record.height
        try:
            with Image.open(path) as image:
                image.load()
                actual_width, actual_height = image.size
                image_sizes.append((actual_width, actual_height))
                if width is None or height is None:
                    width, height = actual_width, actual_height
                elif (width, height) != (actual_width, actual_height):
                    _add_issue(
                        issues, per_split, split=split, path=path,
                        kind="size_mismatch", severity="warning",
                        detail=(
                            f"annotation says {width}x{height}, file is "
                            f"{actual_width}x{actual_height}"
                        ),
                    )
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                digests.setdefault(digest, []).append((split, path))
                if near_duplicate:
                    hashes.append((split, path, _average_hash(image)))
        except Exception as error:  # noqa: BLE001 - unreadable is the finding
            _add_issue(
                issues, per_split, split=split, path=path,
                kind="unreadable", severity="error",
                detail=f"{type(error).__name__}: {error}",
            )
            continue

        if len(record.labels) != len(record.boxes):
            _add_issue(
                issues, per_split, split=split, path=path,
                kind="label_count_mismatch", severity="error",
                detail=f"{len(record.boxes)} boxes vs {len(record.labels)} labels",
            )
        if record.iscrowds and len(record.iscrowds) != len(record.boxes):
            _add_issue(
                issues, per_split, split=split, path=path,
                kind="crowd_count_mismatch", severity="error",
                detail=f"{len(record.boxes)} boxes vs {len(record.iscrowds)} crowd flags",
            )

        if record.raw_category_ids is not None and raw_categories is not None:
            for category_id in record.raw_category_ids:
                if int(category_id) not in raw_categories:
                    _add_issue(
                        issues, per_split, split=split, path=path,
                        kind="unknown_category", severity="error",
                        detail=f"category_id {category_id} is not in the dataset categories",
                    )

        if record.raw_bboxes is not None and width and height:
            for index, raw_box in enumerate(record.raw_bboxes):
                x, y, box_width, box_height = (float(v) for v in raw_box)
                if (
                    not all(math.isfinite(v) for v in (x, y, box_width, box_height))
                    or box_width <= 0
                    or box_height <= 0
                    or x < -_COCO_TOL
                    or y < -_COCO_TOL
                    or x + box_width > width + _COCO_TOL
                    or y + box_height > height + _COCO_TOL
                ):
                    _add_issue(
                        issues, per_split, split=split, path=path,
                        kind="raw_box_out_of_bounds", severity="error",
                        detail=(
                            f"annotation {index}: xywh=({x:g},{y:g},"
                            f"{box_width:g},{box_height:g}) on {width}x{height}"
                        ),
                    )

        image_boxes: list[tuple[float, float, float, float]] = []
        for index, box in enumerate(record.boxes):
            values = tuple(float(v) for v in box)
            if len(values) != 4 or not all(math.isfinite(v) for v in values):
                _add_issue(
                    issues, per_split, split=split, path=path,
                    kind="invalid_box", severity="error",
                    detail=f"box {index} is not four finite numbers: {box!r}",
                )
                continue
            cx, cy, box_width, box_height = values
            if box_width <= 0 or box_height <= 0:
                _add_issue(
                    issues, per_split, split=split, path=path,
                    kind="invalid_box", severity="error",
                    detail=f"box {index} has non-positive size: {values}",
                )
                continue
            xyxy = _norm_box_xyxy(values)
            if (
                min(xyxy[0], xyxy[1]) < -_COCO_TOL
                or max(xyxy[2], xyxy[3]) > 1.0 + _COCO_TOL
            ):
                _add_issue(
                    issues, per_split, split=split, path=path,
                    kind="box_out_of_bounds", severity="error",
                    detail=f"box {index} normalized xyxy={xyxy}",
                )
                continue
            image_boxes.append(xyxy)
            if (
                record.labels
                and (index >= len(record.labels) or not (0 <= int(record.labels[index]) < len(categories)))
            ):
                _add_issue(
                    issues, per_split, split=split, path=path,
                    kind="class_label_out_of_range", severity="error",
                    detail=f"box {index} label is not in 0..{len(categories) - 1}",
                )
            label = int(record.labels[index]) if index < len(record.labels) else -1
            name = categories.get(label, f"label:{label}")
            per_class.setdefault(name, {name_split: 0 for name_split in split_names})
            per_class[name][split] = per_class[name].get(split, 0) + 1
            total_boxes += 1
            per_split[split]["boxes"] += 1

        # Duplicate boxes of the same class inside one image are a common
        # annotation-tool artifact; warn, do not fail the dataset.
        if record.labels:
            for left in range(len(image_boxes)):
                for right in range(left + 1, len(image_boxes)):
                    if (
                        left < len(record.labels)
                        and right < len(record.labels)
                        and int(record.labels[left]) == int(record.labels[right])
                        and _iou(image_boxes[left], image_boxes[right]) > 0.95
                    ):
                        _add_issue(
                            issues, per_split, split=split, path=path,
                            kind="duplicate_box", severity="warning",
                            detail=f"boxes {left} and {right} overlap (IoU > 0.95)",
                        )
                        break
        if not image_boxes:
            _add_issue(
                issues, per_split, split=split, path=path,
                kind="empty_image", severity="info",
                detail="no usable boxes",
            )

    for digest, entries in sorted(digests.items()):
        if len(entries) < 2:
            continue
        group_splits = {split for split, _ in entries}
        first_split = entries[0][0]
        for split, path in entries[1:]:
            # A duplicate inside one split is an annotation/dataset hygiene
            # warning; the same bytes crossing splits is a leakage error.
            cross_split = len(group_splits) > 1 and split != first_split
            _add_issue(
                issues, per_split, split=split, path=path,
                kind="split_leak" if cross_split else "duplicate_image",
                severity="error" if cross_split else "warning",
                detail=f"same bytes as {entries[0][1]} (sha256 {digest[:12]}...)",
            )

    if near_duplicate:
        if len(hashes) > near_duplicate_limit:
            split = split_names[0] if split_names else "train"
            _add_issue(
                issues, per_split, split=split, path=Path("."),
                kind="near_duplicate_skipped", severity="info",
                detail=(
                    f"{len(hashes)} images exceed the O(n^2) scan limit "
                    f"({near_duplicate_limit}); exact duplicates were still checked"
                ),
            )
        else:
            for left in range(len(hashes)):
                left_split, left_path, left_hash = hashes[left]
                for right in range(left + 1, len(hashes)):
                    right_split, right_path, right_hash = hashes[right]
                    if _hamming(left_hash, right_hash) <= 4:
                        cross_split = left_split != right_split
                        _add_issue(
                            issues, per_split, split=right_split, path=right_path,
                            kind="near_split_leak" if cross_split else "near_duplicate",
                            severity="error" if cross_split else "warning",
                            detail=f"perceptual hash close to {left_path} ({left_split})",
                        )

    class_totals = {
        name: sum(values.values()) for name, values in per_class.items()
    }
    positive = [count for count in class_totals.values() if count > 0]
    if len(positive) > 1 and max(positive) / min(positive) >= 10:
        split = split_names[0] if split_names else "train"
        _add_issue(
            issues, per_split, split=split, path=Path("."),
            kind="class_imbalance", severity="warning",
            detail=f"max/min class count = {max(positive) / min(positive):.1f}",
        )
    if "train" in per_split and "val" in per_split:
        for name, values in sorted(per_class.items()):
            if values.get("train", 0) > 0 and values.get("val", 0) == 0:
                _add_issue(
                    issues, per_split, split="val", path=Path("."),
                    kind="missing_in_val", severity="warning",
                    detail=f"class '{name}' has train boxes but no val boxes",
                )

    severity_counts = {
        severity: sum(1 for issue in issues if issue.severity == severity)
        for severity in ("error", "warning", "info")
    }
    summary = {
        "images": len(records),
        "boxes": total_boxes,
        "errors": severity_counts["error"],
        "warnings": severity_counts["warning"],
        "infos": severity_counts["info"],
        "exact_duplicate_images": sum(
            len(entries) - 1 for entries in digests.values() if len(entries) > 1
        ),
        "image_size_min": (
            [min(width for width, _ in image_sizes), min(height for _, height in image_sizes)]
            if image_sizes else None
        ),
        "image_size_max": (
            [max(width for width, _ in image_sizes), max(height for _, height in image_sizes)]
            if image_sizes else None
        ),
        "near_duplicate_checked": bool(near_duplicate),
    }
    return DataAuditReport(
        out_dir=Path("."),
        summary=summary,
        issues=issues,
        per_class=per_class,
        per_split=per_split,
        max_examples=max_examples,
    )
