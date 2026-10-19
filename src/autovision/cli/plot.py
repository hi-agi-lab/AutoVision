# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Debug visualization for pretraining augmentations."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

__all__ = ["plot_example_augmentations", "plot_training_curves"]


def plot_example_augmentations(
    train_batch: dict,
    max_examples: int = 10,
) -> Image.Image:
    """Grid of the first examples across all views of a pretraining batch.

    ``train_batch["views"]`` holds one (B, C, H, W) tensor per view.
    """
    views = train_batch["views"]
    num_views = len(views)
    num_examples = min(max_examples, views[0].shape[0])
    cell = 128
    grid = Image.new(
        "RGB",
        (num_views * cell, num_examples * cell),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(grid)
    for view_index, view in enumerate(views):
        for example_index in range(num_examples):
            tensor = view[example_index, :3].float()
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-9)
            array = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
            tile = Image.fromarray(array).resize((cell, cell))
            grid.paste(tile, (view_index * cell, example_index * cell))
    for view_index in range(num_views):
        draw.text((view_index * cell + 4, 2), f"view {view_index}", fill=(200, 0, 0))
    for example_index in range(num_examples):
        draw.text((2, example_index * cell + 2), f"#{example_index}", fill=(200, 0, 0))
    return grid


def save_example_augmentations(
    train_batch: dict, out_path: Path | str, max_examples: int = 10
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = plot_example_augmentations(train_batch, max_examples=max_examples)
    grid.save(out_path)
    return out_path


_DEFAULT_CURVE_KEYS = ("train_loss", "val_metric/map", "val_metric/map_50")


def plot_training_curves(
    metrics: Path | str,
    out_path: Path | str | None = None,
    keys: list[str] | None = None,
) -> Path:
    """Plot training curves from a run's ``metrics.jsonl``.

    ``keys`` defaults to ``train_loss`` plus the val mAP series when
    present. matplotlib is an optional dependency — the raised error says
    so explicitly instead of failing deep inside an import chain.
    Returns the written PNG path (default:
    ``<metrics dir>/training_curves.png``).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on env
        from autovision.errors import AutoVisionError

        raise AutoVisionError(
            "plot_training_curves requires matplotlib (optional "
            "dependency): pip install matplotlib"
        ) from error

    metrics = Path(metrics)
    series: dict[str, list[float]] = {}
    for line in metrics.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for key in keys or _DEFAULT_CURVE_KEYS:
            value = record.get(key)
            if isinstance(value, (int, float)):
                series.setdefault(key, []).append(float(value))
    if not series:
        from autovision.errors import AutoVisionError

        raise AutoVisionError(
            f"No plottable series {keys or _DEFAULT_CURVE_KEYS} in "
            f"'{metrics}'. Pass keys=[...] explicitly."
        )

    out_path = (
        Path(out_path)
        if out_path is not None
        else metrics.parent / "training_curves.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8, 4))
    for key, values in series.items():
        axes.plot(range(1, len(values) + 1), values, marker=".", label=key)
    axes.set_xlabel("record")
    axes.legend()
    axes.set_title(metrics.parent.name)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path
