# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pseudo-label command (roadmap C1): teacher model -> COCO annotations.

Thin CLI-side wrapper over
:func:`autovision.tasks.detection.shared.pseudo_label.generate_pseudo_labels`;
``data=`` (optional) provides the category list (read from a data config's
annotations json) so the emitted ``category_id``s are the labeled set's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autovision.tasks.detection.shared.pseudo_label import (
    PseudoLabelResult,
    generate_pseudo_labels,
)
from autovision.tasks.task_model_helpers import load_task_model
from autovision.types import PathLike

__all__ = ["pseudo_label"]


def _categories_from_data(
    data: PathLike | dict[str, Any], split: str
) -> dict[int, str] | None:
    """Category list from a data config's annotations json.

    ``data`` is OPTIONAL for the caller (without it the positional
    fallback applies), but when it IS given and carries no readable
    annotations the call fails loud — silently degrading to
    ``class_N`` names would mislabel every emitted category.
    """
    import yaml

    from autovision.errors import AutoVisionError

    if isinstance(data, dict):
        payload = data
    else:
        text = Path(data).read_text()
        payload = (
            json.loads(text) if text.lstrip().startswith("{") else yaml.safe_load(text)
        )
    entry = payload.get(split) or payload.get("train") or payload.get("val")
    if not isinstance(entry, dict) or "annotations" not in entry:
        raise AutoVisionError(
            f"data= was given but carries no '{split}'/'train'/'val' entry "
            "with annotations; omit data= to use the positional class naming."
        )
    annotations = json.loads(Path(entry["annotations"]).read_text())
    return {int(c["id"]): str(c["name"]) for c in annotations.get("categories", [])}


def pseudo_label(
    model: PathLike,
    images: PathLike,
    out: PathLike,
    *,
    data: PathLike | dict[str, Any] | None = None,
    split: str = "train",
    threshold: float = 0.5,
    class_thresholds: dict[str, float] | None = None,
    sahi: bool = False,
    tile_size: int | None = None,
    overlap: float = 0.2,
    batch_size: int = 8,
    max_images: int | None = None,
    device: str = "cpu",
) -> PseudoLabelResult:
    """Run ``model`` over ``images`` and write a COCO annotations json at ``out``."""
    network = load_task_model(model, device=device)
    categories = (
        _categories_from_data(data, split) if data is not None else None
    )
    return generate_pseudo_labels(
        network,
        images,
        out,
        threshold=threshold,
        class_thresholds=class_thresholds,
        categories=categories,
        sahi=sahi,
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
        max_images=max_images,
    )
