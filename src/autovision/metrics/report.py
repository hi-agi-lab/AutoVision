# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Human-readable metric tables for the training log.

Renders the validation report in the fixed-width box style of the
reference detection training logs: an overall headline table
(AP 50:95 / 50 / 75, mAR, F1-sweep precision/recall) and a per-class
breakdown. Dependency-free — the tables are plain Unicode-drawn text so
they survive tee, journalctl, and CI logs alike.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["format_metric_table", "format_detection_validation_report"]


def _box(widths: Sequence[int]) -> str:
    return "┏" + "┳".join("━" * (w + 2) for w in widths) + "┓"


def _mid(widths: Sequence[int]) -> str:
    return "┡" + "╇".join("━" * (w + 2) for w in widths) + "┩"


def _sep(widths: Sequence[int]) -> str:
    return "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"


def _bottom(widths: Sequence[int]) -> str:
    return "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"


def _row(values: Sequence[str], widths: Sequence[int]) -> str:
    return "┃ " + " ┃ ".join(
        str(v).rjust(w) if i else str(v).ljust(w)
        for i, (v, w) in enumerate(zip(values, widths))
    ) + " ┃"


def format_metric_table(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    """Render a titled box table; the first column is left-aligned."""
    widths = [
        max(len(str(headers[i])), *(len(str(r[i])) for r in rows)) if rows
        else len(str(headers[i]))
        for i in range(len(headers))
    ]
    lines = [f" {title} ", _box(widths), _row(headers, widths), _mid(widths)]
    lines.extend(_row([str(v) for v in row], widths) for row in rows)
    lines.append(_bottom(widths))
    return "\n".join(lines)


def format_detection_validation_report(
    heading: str,
    metrics: dict[str, float],
    per_class: Sequence[dict[str, float]],
    max_dets: int,
    footer: str | None = None,
) -> str:
    """Overall + per-class validation tables in the reference log style.

    ``metrics`` is a ``compute()`` result (map/map_50/map_75/mar/f1/
    precision/recall); ``per_class`` the evaluator's ``last_per_class``
    rows. ``footer`` (e.g. a val-loss scope caveat) is appended verbatim.
    """
    overall = format_metric_table(
        heading,
        ("mAP 50:95", "mAP 50", "mAP 75", f"mAR @{max_dets}", "F1", "Prec", "Recall"),
        [[
            f"{metrics.get('map', -1):.4f}",
            f"{metrics.get('map_50', -1):.4f}",
            f"{metrics.get('map_75', -1):.4f}",
            f"{metrics.get('mar', -1):.4f}",
            f"{metrics.get('f1', -1):.4f}",
            f"{metrics.get('precision', -1):.4f}",
            f"{metrics.get('recall', -1):.4f}",
        ]],
    )
    class_rows = [
        [
            row["name"],
            f"{row['ap']:.4f}",
            f"{row['ar']:.4f}",
            f"{row['f1']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
        ]
        for row in per_class
    ]
    per_class_table = format_metric_table(
        f"{heading} — Per-class",
        ("Class", "AP 50:95", "AR", "F1", "Precision", "Recall"),
        class_rows,
    )
    parts = [overall, per_class_table]
    if footer:
        parts.append(footer)
    return "\n".join(parts)
