# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Detection error analysis: TP/FP/FN matching, slices, and galleries (C2).

Takes in-memory detections in the SAME contract as the mAP evaluator
(``predictions[i]``: ``bboxes`` xyxy original-pixel pixels, ``scores``,
``labels``; ``targets[i]``: ``boxes`` xyxy original pixels, ``labels``,
optional ``iscrowds``) and produces:

- greedy per-class, score-ordered matching at an IoU threshold
  (COCO-style: each prediction takes the best-IoU unmatched GT box);
- false-positive attribution — ``duplicate`` / ``classification`` /
  ``localization`` / ``background`` (TIDE-style, mutually exclusive in
  that order), plus an ``ignored`` bucket: a detection whose best match is
  a crowd (``iscrowd == 1``) GT at IoU >= the threshold is an ignore-region
  hit under the official protocol (pycocotools ``dtIg``) — neither TP nor
  FP, counted separately and excluded from precision/recall;
- slices by class and by COCO area range (small / medium / large);
- a report (``report.json`` + ``report.md``) and a **top-FP / top-FN
  gallery**: rendered overlays (TP green, FP red with reason + score, FN
  orange "MISS") plus per-class worst cases.

Labels are internal ids indexing ``sorted(categories)`` — the exact
convention of :class:`~autovision.metrics.map_evaluator.COCOMAPEvaluator`.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from autovision.metrics.confusion import ConfusionMatrix

__all__ = [
    "COCO_AREA_RANGES",
    "ConfusionMatrix",
    "ErrorAnalysisReport",
    "Match",
    "analyze_detection_errors",
    "match_detections",
    "summarize_matches",
]

#: COCO area ranges in absolute pixels² (the pycocotools protocol).
COCO_AREA_RANGES: dict[str, tuple[float, float]] = {
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}

#: FP attribution tags, mutually exclusive in this priority order.
FP_REASONS = ("duplicate", "classification", "localization", "background")
#: Same-class IoU floor for the ``localization`` tag (below it: background).
LOCALIZATION_IOU_FLOOR = 0.1


@dataclass(frozen=True)
class Match:
    """One matched/missed box row (TP, FP, FN, or an ``ignored`` crowd hit)."""

    image_index: int
    kind: str  # "tp" | "fp" | "fn" | "ignored" (crowd-region hit)
    class_id: int
    class_name: str
    score: float
    box: tuple[float, float, float, float]  # xyxy original pixels
    area: float  # box area in pixels²
    iou: float = 0.0
    fp_reason: str | None = None
    #: GT class behind a ``classification`` FP (``None`` for other reasons).
    other_class_id: int | None = None


@dataclass
class ErrorAnalysisReport:
    """Result of :func:`analyze_detection_errors`."""

    out_dir: Path
    report_json: Path
    report_md: Path
    gallery: list[Path] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    hard_examples_json: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "report_json": str(self.report_json),
            "report_md": str(self.report_md),
            "hard_examples_json": (
                str(self.hard_examples_json)
                if self.hard_examples_json is not None
                else None
            ),
            "gallery": [str(path) for path in self.gallery],
            "summary": dict(self.summary),
        }


def _boxes(entry: Mapping[str, Any]) -> torch.Tensor:
    boxes = entry["boxes"] if "boxes" in entry else entry["bboxes"]
    return torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)


def _scores(entry: Mapping[str, Any], count: int) -> torch.Tensor:
    if entry.get("scores") is None:
        return torch.ones(count, dtype=torch.float32)
    return torch.as_tensor(entry["scores"], dtype=torch.float32).reshape(-1)


def _labels(entry: Mapping[str, Any], count: int) -> torch.Tensor:
    if entry.get("labels") is None:
        return torch.zeros(count, dtype=torch.long)
    return torch.as_tensor(entry["labels"], dtype=torch.long).reshape(-1)


def _area_of(box: Sequence[float]) -> float:
    return max((float(box[2]) - float(box[0])), 0.0) * max(
        (float(box[3]) - float(box[1])), 0.0
    )


def _area_bucket(area: float, ranges: Mapping[str, tuple[float, float]]) -> str:
    """First matching ``[low, high)`` bucket; at/above the last bucket's
    lower bound it belongs to the last bucket; anything else (a gap in a
    caller-supplied mapping) fails loud instead of mis-bucketing."""
    items = list(ranges.items())
    for name, (low, high) in items:
        if low <= area < high:
            return name
    if items and area >= items[-1][1][0]:
        return items[-1][0]
    raise ValueError(
        f"area {area} falls outside the configured area ranges "
        f"{[name for name, _ in items]}."
    )


def _box_tuple(values: Any) -> tuple[float, float, float, float]:
    """Four-corner box as a fixed-length tuple (mypy-friendly)."""
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def match_detections(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    class_names: Mapping[int, str] | None = None,
) -> list[Match]:
    """Greedy per-class matching of ``predictions`` against ``targets``.

    Per image and class, predictions are consumed in score order and each
    takes the best-IoU unmatched GT box; a prediction below
    ``iou_threshold`` is a false positive, an unconsumed GT is a false
    negative. Crowd GT rows (``iscrowds == 1``) are excluded from matching:
    under the official COCO protocol they are ignore regions (they neither
    require recall nor penalize overlapping detections).
    """
    from torchvision.ops import box_iou

    names = dict(class_names or {})
    matches: list[Match] = []

    for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
        boxes_p = _boxes(prediction)
        scores_p = _scores(prediction, boxes_p.shape[0])
        labels_p = _labels(prediction, boxes_p.shape[0])
        boxes_g = _boxes(target)
        labels_g = _labels(target, boxes_g.shape[0])
        crowds_g = target.get("iscrowds")
        if crowds_g is not None:
            crowds_g = torch.as_tensor(crowds_g, dtype=torch.long).reshape(-1)
        else:
            crowds_g = torch.zeros(boxes_g.shape[0], dtype=torch.long)

        keep_p = scores_p >= float(score_threshold)
        boxes_p, scores_p, labels_p = (
            boxes_p[keep_p],
            scores_p[keep_p],
            labels_p[keep_p],
        )
        regular_g = crowds_g == 0

        matched_gt: set[int] = set()
        for class_id in sorted(set(labels_p.tolist()) | set(labels_g.tolist())):
            class_name = names.get(class_id, str(class_id))
            pred_index = torch.nonzero(labels_p == class_id).flatten()
            order = pred_index[scores_p[pred_index].argsort(descending=True)]
            gt_index = torch.nonzero(
                (labels_g == class_id) & regular_g
            ).flatten()
            ious = (
                box_iou(boxes_p[pred_index], boxes_g[gt_index])
                if pred_index.numel() and gt_index.numel()
                else None
            )
            for position, p in enumerate(order.tolist()):
                box = _box_tuple(boxes_p[p])
                best_iou, best_gt = 0.0, None
                if ious is not None:
                    row = ious[position]
                    for local_gt, gt in enumerate(gt_index.tolist()):
                        if gt in matched_gt:
                            continue
                        value = float(row[local_gt])
                        if value > best_iou:
                            best_iou, best_gt = value, gt
                if best_gt is not None and best_iou >= float(iou_threshold):
                    matched_gt.add(best_gt)
                    matches.append(
                        Match(
                            image_index=image_index,
                            kind="tp",
                            class_id=class_id,
                            class_name=class_name,
                            score=float(scores_p[p]),
                            box=box,
                            area=_area_of(box),
                            iou=best_iou,
                        )
                    )
                else:
                    reason, other_class_id = _classify_fp(
                        box,
                        boxes_g,
                        labels_g,
                        crowds_g,
                        matched_gt,
                        iou_threshold,
                        class_id,
                    )
                    matches.append(
                        Match(
                            image_index=image_index,
                            kind="ignored" if reason == "ignored" else "fp",
                            class_id=class_id,
                            class_name=class_name,
                            score=float(scores_p[p]),
                            box=box,
                            area=_area_of(box),
                            iou=best_iou,
                            fp_reason=None if reason == "ignored" else reason,
                            other_class_id=(
                                None if reason == "ignored" else other_class_id
                            ),
                        )
                    )
            for gt in gt_index.tolist():
                if gt in matched_gt:
                    continue
                box = _box_tuple(boxes_g[gt])
                matches.append(
                    Match(
                        image_index=image_index,
                        kind="fn",
                        class_id=class_id,
                        class_name=class_name,
                        score=0.0,
                        box=box,
                        area=_area_of(box),
                    )
                )
    return matches


def _classify_fp(
    box: tuple[float, float, float, float],
    boxes_g: torch.Tensor,
    labels_g: torch.Tensor,
    crowds_g: torch.Tensor,
    matched_gt: set[int],
    iou_threshold: float,
    class_id: int,
) -> tuple[str, int | None]:
    """TIDE-style attribution for one false positive.

    A detection overlapping a CROWD gt (``iscrowd == 1``) at IoU >= the
    threshold is an **ignore** hit (pycocotools ``dtIg`` semantics: the
    official protocol neither counts it as a true nor as a false positive),
    returned as ``"ignored"`` for the caller to record separately.
    Otherwise, in priority order: ``duplicate`` (same-class IoU >= the
    threshold but the GT is taken), ``classification`` (IoU >= threshold
    with another class's GT), ``localization`` (same-class IoU >= 0.1),
    ``background``. The second value is the GT class for a
    ``classification`` hit, else ``None``.
    """
    from torchvision.ops import box_iou

    if boxes_g.shape[0] == 0:
        return "background", None
    query = torch.tensor([box], dtype=torch.float32, device=boxes_g.device)
    ious = box_iou(query, boxes_g)[0]
    # Closest-to-pyocotools rule (``dtIg``): pick the BEST-IoU GT among the
    # crowd boxes and the UNMATCHED regular GT; only when that best match is
    # the crowd box is the detection an ignore-region hit. (pycocotools
    # additionally skips already-matched regular GT in its argmax; here a
    # strong overlap on a taken GT reads as ``duplicate`` — the TIDE
    # attribution — which is the more informative answer.)
    best_overall, best_overall_is_crowd = 0.0, False
    for gt in range(boxes_g.shape[0]):
        value = float(ious[gt])
        if value > best_overall:
            best_overall, best_overall_is_crowd = value, int(crowds_g[gt]) != 0
    if best_overall_is_crowd and best_overall >= float(iou_threshold):
        return "ignored", None
    best_any, best_any_label = 0.0, -1
    best_same, same_matched = 0.0, False
    for gt in range(boxes_g.shape[0]):
        if int(crowds_g[gt]) != 0:
            continue
        value = float(ious[gt])
        if value > best_any:
            best_any, best_any_label = value, int(labels_g[gt])
        if int(labels_g[gt]) == class_id:
            if value > best_same:
                best_same, same_matched = value, gt in matched_gt
    if best_same >= float(iou_threshold) and same_matched:
        return "duplicate", None
    if best_any >= float(iou_threshold) and best_any_label >= 0 and best_any_label != class_id:
        return "classification", best_any_label
    if best_same >= LOCALIZATION_IOU_FLOOR:
        return "localization", None
    return "background", None


def summarize_matches(
    matches: Sequence[Match],
    class_names: Mapping[int, str] | None = None,
    *,
    area_ranges: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Counts and ratios: totals, per class, per area bucket, FP reasons."""
    ranges = area_ranges or COCO_AREA_RANGES
    totals = Counter(match.kind for match in matches)
    totals.setdefault("ignored", 0)
    per_class: dict[str, dict[str, Any]] = {}
    per_area: dict[str, dict[str, Any]] = {
        name: {"tp": 0, "fp": 0, "fn": 0, "ignored": 0} for name in ranges
    }
    for match in matches:
        row = per_class.setdefault(
            match.class_name,
            {
                "gt": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "ignored": 0,
                **{f"fp_{reason}": 0 for reason in FP_REASONS},
            },
        )
        if match.kind == "fn":
            row["gt"] += 1
        if match.kind == "tp":
            row["gt"] += 1
        row[match.kind] += 1
        if match.kind == "fp" and match.fp_reason:
            row[f"fp_{match.fp_reason}"] += 1
        bucket = _area_bucket(match.area, ranges)
        per_area[bucket][match.kind] += 1
    for row in per_class.values():
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        row["precision"] = tp / (tp + fp) if tp + fp else 0.0
        row["recall"] = tp / (tp + fn) if tp + fn else 0.0
    for counts in per_area.values():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        counts["precision"] = tp / (tp + fp) if tp + fp else 0.0
        counts["recall"] = tp / (tp + fn) if tp + fn else 0.0
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    return {
        "totals": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "ignored": totals["ignored"],
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        },
        "per_class": dict(sorted(per_class.items())),
        "per_area": {name: dict(counts) for name, counts in per_area.items()},
        "fp_reasons": {
            reason: sum(row[f"fp_{reason}"] for row in per_class.values())
            for reason in FP_REASONS
        },
    }


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in name)


def _render_overlay(
    image_path: Path,
    image_matches: Sequence[Match],
    class_names: Mapping[int, str],
    max_boxes: int = 60,
) -> Any:
    """TP (green) / FP (red, reason + score) / FN (orange MISS) / ignored
    crowd hits (grey IGNORE) overlay."""
    from PIL import Image, ImageDraw

    from autovision.data.file_helpers import open_image

    canvas = Image.fromarray(open_image(image_path).permute(1, 2, 0).numpy())
    draw = ImageDraw.Draw(canvas)
    for match in list(image_matches)[:max_boxes]:
        x1, y1, x2, y2 = match.box
        if match.kind == "tp":
            draw.rectangle([x1, y1, x2, y2], outline=(0, 190, 0), width=2)
            draw.text((x1 + 2, y1 + 2), f"TP {match.class_name} {match.score:.2f}",
                      fill=(0, 190, 0))
        elif match.kind == "fp":
            draw.rectangle([x1, y1, x2, y2], outline=(220, 0, 0), width=2)
            draw.text(
                (x1 + 2, y1 + 2),
                f"FP {match.class_name} {match.score:.2f} [{match.fp_reason}]",
                fill=(220, 0, 0),
            )
        elif match.kind == "ignored":
            # Crowd-region hit: drawn for context, never counted as an error.
            draw.rectangle([x1, y1, x2, y2], outline=(130, 130, 130), width=1)
            draw.text((x1 + 2, y1 + 2), f"IGNORE {match.class_name}", fill=(130, 130, 130))
        else:  # fn
            draw.rectangle([x1, y1, x2, y2], outline=(235, 130, 0), width=3)
            draw.text((x1 + 2, min(y1 + 12, max(y2 - 12, 0))), f"MISS {match.class_name}",
                      fill=(235, 130, 0))
    return canvas


def _rank_images(
    matches: Sequence[Match], kind: str, class_id: int | None
) -> list[tuple[int, float, float]]:
    """(image_index, count, tie-break) ranked worst-first for a kind/class."""
    counts: dict[int, list[float]] = {}
    for match in matches:
        if match.kind != kind:
            continue
        if class_id is not None and match.class_id != class_id:
            continue
        row = counts.setdefault(match.image_index, [0.0, 0.0])
        row[0] += 1.0
        # Tie-break: FP -> highest score, FN -> largest missed area.
        weight = match.score if kind == "fp" else match.area
        row[1] = max(row[1], weight)
    ranked = [
        (image_index, row[0], row[1]) for image_index, row in counts.items()
    ]
    ranked.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return ranked


def analyze_detection_errors(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    categories: Mapping[int, str],
    image_paths: Sequence[Path | str],
    out_dir: Path | str,
    *,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    top_k: int = 5,
    max_boxes_per_image: int = 60,
    area_ranges: Mapping[str, tuple[float, float]] | None = None,
    split: str = "train",
) -> ErrorAnalysisReport:
    """Full error report: matching, slices, top-FP/FN gallery, md + json.

    Gallery selection (documented, deterministic):

    - global FP / FN: ``top_k`` images by count (FP tie-break: highest
      score; FN tie-break: largest missed area);
    - per-class FP / FN: each class's single worst image (same ranking).

    Files: ``report.json``, ``report.md``, ``hard_examples.json``,
    ``gallery/{fp,fn}_all/`` and ``gallery/{fp,fn}_<class>/`` overlays.
    The hard-example manifest records ``split``; the weighted sampler only
    consumes ``split="train"`` entries, so a val audit cannot leak into
    training.
    """
    # Internal label ids are the rank in ``sorted(categories)`` BY KEY —
    # the convention of predictions_to_coco / the data loaders / the
    # evaluator's per-class report. Sorting by name would mislabel every
    # class whenever names do not sort in id order (standard COCO).
    class_names = {
        index: name for index, (_, name) in enumerate(sorted(categories.items()))
    }
    if len(predictions) != len(targets) or len(targets) != len(image_paths):
        raise ValueError(
            "predictions, targets and image_paths must have the same length "
            f"({len(predictions)} / {len(targets)} / {len(image_paths)})."
        )
    ranges = area_ranges or COCO_AREA_RANGES
    matches = match_detections(
        predictions,
        targets,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_names=class_names,
    )
    summary = summarize_matches(matches, class_names, area_ranges=ranges)
    summary["iou_threshold"] = float(iou_threshold)
    summary["score_threshold"] = float(score_threshold)
    summary["images"] = len(targets)
    confusion = ConfusionMatrix.from_matches(
        matches,
        class_names=class_names,
        class_ids=sorted(class_names),
    )
    summary["confusion"] = confusion.to_dict()

    out_dir = Path(out_dir)
    gallery_dir = out_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    by_image: dict[int, list[Match]] = {}
    for match in matches:
        by_image.setdefault(match.image_index, []).append(match)

    written: list[Path] = []

    def _write_gallery(
        kind: str,
        class_id: int | None,
        class_label: str,
        ranked: list[tuple[int, float, float]],
    ) -> None:
        folder = gallery_dir / (f"{kind}_all" if class_id is None else f"{kind}_{class_label}")
        folder.mkdir(parents=True, exist_ok=True)
        for rank, (image_index, count, tie) in enumerate(ranked):
            path = Path(image_paths[image_index])
            canvas = _render_overlay(
                path, by_image.get(image_index, ()), class_names, max_boxes_per_image
            )
            out_path = folder / f"rank{rank:02d}_{count:.0f}_{path.stem}.jpg"
            canvas.save(str(out_path))
            written.append(out_path)

    # Global galleries (top_k images) + per-class worst image.
    global_fp = _rank_images(matches, "fp", None)[:top_k]
    global_fn = _rank_images(matches, "fn", None)[:top_k]
    _write_gallery("fp", None, "", global_fp)
    _write_gallery("fn", None, "", global_fn)
    for index, name in class_names.items():
        row = summary["per_class"].get(name)
        if row is None:
            continue
        if row["fp"]:
            _write_gallery("fp", index, _safe_name(name), _rank_images(matches, "fp", index)[:1])
        if row["fn"]:
            _write_gallery("fn", index, _safe_name(name), _rank_images(matches, "fn", index)[:1])

    # ── report.md ───────────────────────────────────────────────────────
    def _fmt(value: float) -> str:
        return f"{value:.4f}"

    md: list[str] = []
    md.append("# 检测误差分析报告\n")
    md.append(
        f"- 图像数 {len(targets)}；IoU 阈值 {iou_threshold}；分数下限 {score_threshold}；"
        f"面积分档 small/medium/large（COCO：<32² / 32²–96² / >96² px²）\n"
    )
    totals = summary["totals"]
    md.append("## 汇总\n")
    md.append("| TP | FP | FN | precision | recall |")
    md.append("|---|---|---|---|---|")
    md.append(
        f"| {totals['tp']} | {totals['fp']} | {totals['fn']} "
        f"| {_fmt(totals['precision'])} | {_fmt(totals['recall'])} |\n"
    )
    md.append("FP 归因：" + "；".join(
        f"{reason} {summary['fp_reasons'][reason]}" for reason in FP_REASONS
    ) + f"；ignore 区命中（crowd）{totals['ignored']}\n")
    md.append("_（precision/recall 不计 ignore 区命中；分档读数为 -1.0 表示该档无 GT——"
              "pycocotools 约定）_\n")
    md.append("## 按类别\n")
    md.append("| 类别 | GT | TP | FP | FN | precision | recall | FP 归因 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for name, row in summary["per_class"].items():
        reasons = " / ".join(
            f"{reason} {row[f'fp_{reason}']}" for reason in FP_REASONS
            if row[f"fp_{reason}"]
        ) or "-"
        md.append(
            f"| {name} | {row['gt']} | {row['tp']} | {row['fp']} | {row['fn']} "
            f"| {_fmt(row['precision'])} | {_fmt(row['recall'])} | {reasons} |"
        )
    md.append("")
    md.append("## 混淆矩阵\n")
    md.append(confusion.to_markdown())
    md.append("")
    md.append("## 按目标尺寸\n")
    md.append("| 分档 | TP | FP | FN | precision | recall |")
    md.append("|---|---|---|---|---|---|")
    for name, counts in summary["per_area"].items():
        md.append(
            f"| {name} | {counts['tp']} | {counts['fp']} | {counts['fn']} "
            f"| {_fmt(counts['precision'])} | {_fmt(counts['recall'])} |"
        )
    md.append("")

    def _gallery_section(title: str, kind: str, ranked: list[tuple[int, float, float]]):
        md.append(f"## {title}\n")
        if not ranked:
            md.append("_（无）_\n")
            return
        for rank, (image_index, count, tie) in enumerate(ranked):
            path = Path(image_paths[image_index])
            rel = f"gallery/{kind}_all/rank{rank:02d}_{count:.0f}_{path.stem}.jpg"
            md.append(f"![{kind} rank{rank}]({rel})\n")

    _gallery_section(f"Top-FP 画廊（按 FP 数排序，取前 {len(global_fp)}）", "fp", global_fp)
    _gallery_section(f"Top-FN 画廊（按漏检数排序，取前 {len(global_fn)}）", "fn", global_fn)
    md.append("## 逐类最差样本\n")
    for index, name in class_names.items():
        row = summary["per_class"].get(name)
        if row is None or not (row["fp"] or row["fn"]):
            continue
        md.append(f"### {name}\n")
        if row["fp"]:
            ranked = _rank_images(matches, "fp", index)[:1]
            image_index, count, tie = ranked[0]
            rel = (
                f"gallery/fp_{_safe_name(name)}/"
                f"rank00_{count:.0f}_{Path(image_paths[image_index]).stem}.jpg"
            )
            md.append(f"FP {row['fp']}：![{name} fp]({rel})\n")
        if row["fn"]:
            ranked = _rank_images(matches, "fn", index)[:1]
            image_index, count, tie = ranked[0]
            rel = (
                f"gallery/fn_{_safe_name(name)}/"
                f"rank00_{count:.0f}_{Path(image_paths[image_index]).stem}.jpg"
            )
            md.append(f"FN {row['fn']}：![{name} fn]({rel})\n")

    report_md = out_dir / "report.md"
    report_md.write_text("\n".join(md))
    report_json = out_dir / "report.json"
    report_json.write_text(
        json.dumps(
            {
                "summary": summary,
                "confusion": confusion.to_dict(),
                "matches": [asdict(match) for match in matches],
                "gallery": [str(path.relative_to(out_dir)) for path in written],
                "images": [str(path) for path in image_paths],
            },
            indent=2,
        )
    )
    hard_examples: list[dict[str, Any]] = []
    for image_index, image_path in enumerate(image_paths):
        image_matches = by_image.get(image_index, [])
        tp = sum(1 for match in image_matches if match.kind == "tp")
        fp = sum(1 for match in image_matches if match.kind == "fp")
        fn = sum(1 for match in image_matches if match.kind == "fn")
        if not (fp or fn):
            continue
        hard_examples.append(
            {
                "image": str(Path(image_path).resolve()),
                "split": split,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "fp_reasons": dict(
                    Counter(
                        match.fp_reason
                        for match in image_matches
                        if match.kind == "fp" and match.fp_reason
                    )
                ),
                "max_fn_area": max(
                    (
                        match.area
                        for match in image_matches
                        if match.kind == "fn"
                    ),
                    default=0.0,
                ),
                "classes_fn": sorted(
                    {
                        match.class_id
                        for match in image_matches
                        if match.kind == "fn"
                    }
                ),
            }
        )
    hard_examples_json = out_dir / "hard_examples.json"
    hard_examples_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_run": str(out_dir),
                "split": split,
                "iou_threshold": float(iou_threshold),
                "score_threshold": float(score_threshold),
                "images": hard_examples,
            },
            indent=2,
        )
    )
    return ErrorAnalysisReport(
        out_dir=out_dir,
        report_json=report_json,
        report_md=report_md,
        gallery=written,
        summary=summary,
        hard_examples_json=hard_examples_json,
    )
