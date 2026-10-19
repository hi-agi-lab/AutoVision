# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""End-to-end detection parity (roadmap 90d, phase 2).

Tensor-level gates answer "did the graph stay numerically close?"; this
module answers the deployment question: **are the final boxes, scores and
labels the same?** It compares two engines' records after their own
pre/post-processing and greedy-matches detections by label + IoU, so a
backend that shifts every box by a few pixels cannot hide behind a loose
tensor tolerance.

No import of the inference layer here (layering: export must not depend on
inference). Any object with ``run_batch(images, threshold=...)`` works.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from autovision.errors import AutoVisionError

__all__ = [
    "ParityReport",
    "compare_detection_records",
    "verify_end_to_end_parity",
]


def _array(value: Any, *, dtype: Any = np.float32) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, np.ndarray]]:
    normalized = []
    for record in records:
        if "bboxes" not in record:
            raise AutoVisionError(
                "End-to-end parity needs detection records with "
                "'bboxes'/'scores'/'labels'."
            )
        boxes = _array(record["bboxes"]).reshape(-1, 4)
        scores = _array(record.get("scores", np.ones(len(boxes)))).reshape(-1)
        labels = _array(record.get("labels", np.zeros(len(boxes))), dtype=np.int64).reshape(-1)
        if not (len(boxes) == len(scores) == len(labels)):
            raise AutoVisionError(
                "Detection record has mismatched boxes/scores/labels lengths."
            )
        normalized.append({"bboxes": boxes, "scores": scores, "labels": labels})
    return normalized


def _iou(box: np.ndarray, other: np.ndarray) -> float:
    x1 = max(box[0], other[0])
    y1 = max(box[1], other[1])
    x2 = min(box[2], other[2])
    y2 = min(box[3], other[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = max(0.0, other[2] - other[0]) * max(0.0, other[3] - other[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class ParityReport:
    """One comparison's verdict (JSON-serializable via :meth:`to_dict`)."""

    images: int
    matched: int
    missing: int = 0
    extra: int = 0
    label_mismatches: int = 0
    max_box_diff_px: float = 0.0
    max_score_diff: float = 0.0
    box_tolerance_px: float = 1.0
    score_tolerance: float = 1e-3
    min_iou: float = 0.5
    details: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.missing == 0
            and self.extra == 0
            and self.label_mismatches == 0
            and self.max_box_diff_px <= self.box_tolerance_px
            and self.max_score_diff <= self.score_tolerance
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "images": self.images,
            "matched": self.matched,
            "missing": self.missing,
            "extra": self.extra,
            "label_mismatches": self.label_mismatches,
            "max_box_diff_px": self.max_box_diff_px,
            "max_score_diff": self.max_score_diff,
            "box_tolerance_px": self.box_tolerance_px,
            "score_tolerance": self.score_tolerance,
            "min_iou": self.min_iou,
            "details": list(self.details),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path


def compare_detection_records(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    box_tolerance_px: float = 1.0,
    score_tolerance: float = 1e-3,
    min_iou: float = 0.5,
    max_details: int = 20,
) -> ParityReport:
    """Compare final detections image-by-image.

    Greedy candidate-first matching on (label, IoU). ``min_iou`` is a
    permissive *matching* gate (default 0.5); the verdict is the box
    tolerance, so a small shift still matches and is then measured. A
    matched pair with the largest box deviation sets
    ``max_box_diff_px``. Unmatched reference boxes are ``missing``;
    unmatched candidate boxes are ``extra``.
    """
    if len(reference) != len(candidate):
        raise AutoVisionError(
            f"Record count mismatch: {len(reference)} reference vs "
            f"{len(candidate)} candidate images."
        )
    ref_records = _records(reference)
    cand_records = _records(candidate)
    report = ParityReport(
        images=len(ref_records),
        matched=0,
        box_tolerance_px=box_tolerance_px,
        score_tolerance=score_tolerance,
        min_iou=min_iou,
    )
    for image_index, (ref, cand) in enumerate(zip(ref_records, cand_records)):
        used_reference: set[int] = set()
        used_candidate: set[int] = set()
        # Highest-scoring candidates claim their best reference first.
        for candidate_index in np.argsort(-cand["scores"], kind="stable"):
            candidate_label = int(cand["labels"][candidate_index])
            best_index: int | None = None
            best_iou = 0.0
            for reference_index in range(len(ref["bboxes"])):
                if reference_index in used_reference:
                    continue
                if int(ref["labels"][reference_index]) != candidate_label:
                    continue
                iou = _iou(
                    cand["bboxes"][candidate_index],
                    ref["bboxes"][reference_index],
                )
                if iou > best_iou:
                    best_iou = iou
                    best_index = reference_index
            if best_index is None or best_iou < min_iou:
                report.extra += 1
                if len(report.details) < max_details:
                    report.details.append(
                        f"image {image_index}: extra detection label="
                        f"{candidate_label} score={cand['scores'][candidate_index]:.4f}"
                    )
                continue
            used_reference.add(best_index)
            used_candidate.add(int(candidate_index))
            report.matched += 1
            box_diff = float(
                np.abs(
                    cand["bboxes"][candidate_index] - ref["bboxes"][best_index]
                ).max()
            )
            score_diff = float(
                abs(cand["scores"][candidate_index] - ref["scores"][best_index])
            )
            report.max_box_diff_px = max(report.max_box_diff_px, box_diff)
            report.max_score_diff = max(report.max_score_diff, score_diff)
        unmatched = len(ref["bboxes"]) - len(used_reference)
        if unmatched:
            report.missing += unmatched
            if len(report.details) < max_details:
                report.details.append(
                    f"image {image_index}: {unmatched} reference detection(s) missing"
                )
        # A label swap shows up as missing+extra; count it explicitly for
        # the report so operators do not have to infer it.
        if any(
            int(ref["labels"][index]) not in set(cand["labels"].tolist())
            for index in range(len(ref["bboxes"]))
            if index not in used_reference
        ):
            report.label_mismatches += 1
    return report


def verify_end_to_end_parity(
    reference_engine: Any,
    candidate_engine: Any,
    images: Sequence[Any],
    *,
    threshold: float = 0.5,
    box_tolerance_px: float = 1.0,
    score_tolerance: float = 1e-3,
    min_iou: float = 0.5,
    report_path: str | Path | None = None,
) -> ParityReport:
    """Run two engines on the same images and compare final detections.

    Engines must expose ``run_batch(images, threshold=...)`` (the
    ``InferenceEngine`` contract). With ``report_path`` the verdict is
    written as JSON for CI/release artifacts.
    """
    for name, engine in (("reference", reference_engine), ("candidate", candidate_engine)):
        if not callable(getattr(engine, "run_batch", None)):
            raise AutoVisionError(
                f"{name} engine has no run_batch(images, threshold=...) method."
            )
    reference = reference_engine.run_batch(images, threshold=threshold)
    candidate = candidate_engine.run_batch(images, threshold=threshold)
    report = compare_detection_records(
        reference,
        candidate,
        box_tolerance_px=box_tolerance_px,
        score_tolerance=score_tolerance,
        min_iou=min_iou,
    )
    if report_path is not None:
        report.save(report_path)
    return report
