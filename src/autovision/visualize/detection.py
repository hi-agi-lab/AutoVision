# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Validation-batch visualization for object detection (GT + predictions)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw

from autovision.augment.transform_args import NormalizeArgs

__all__ = ["draw_prediction", "save_detection_examples"]


def _cxcywh_norm_to_xyxy(boxes, width: int, height: int):
    """Normalized cxcywh -> absolute xyxy (skips degenerate rows)."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    cx, cy, w, h = boxes.reshape(-1, 4).unbind(-1)
    return torch.stack(
        [(cx - w / 2) * width, (cy - h / 2) * height,
         (cx + w / 2) * width, (cy + h / 2) * height], dim=-1,
    )


def save_detection_examples(
    out_dir: Path | str,
    batch: dict,
    step: int,
    predictions: Sequence[dict] | None = None,
    max_examples: int = 3,
    normalize_args: Any = None,
) -> list[Path]:
    """Render the first N images of a batch with GT boxes (green, normalized
    cxcywh on the input grid) and, when provided, prediction boxes (red, xyxy
    absolute pixels); mirrors the baseline image_examples."""
    out_dir = Path(out_dir) / "image_examples" / f"step_{step:07d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    images = batch["image"].detach().cpu()
    written = []
    num = min(max_examples, images.shape[0])
    for index in range(num):
        image = images[index]
        _, height, width = image.shape
        display = image.float()
        if display.min() < 0:
            # Normalized input (negative values): un-normalize for display.
            # This check must come first — normalized bright pixels reach
            # ~2.6 and would trip a ">1.5 means raw 0-255" heuristic,
            # flattening the whole image to the mean color.
            args = normalize_args or {}
            mean = args.get("mean", NormalizeArgs().mean)
            std = args.get("std", NormalizeArgs().std)
            mean = torch.tensor(mean).view(-1, 1, 1)
            std = torch.tensor(std).view(-1, 1, 1)
            display = display * std + mean
        elif display.max() > 1.5:  # raw 0-255 input
            display = display / 255.0
        grid = Image.fromarray(
            (display.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
        )
        draw = ImageDraw.Draw(grid)
        gt_boxes = batch["bboxes"][index] if isinstance(batch["bboxes"], list) else batch["bboxes"]
        gt_labels = batch["classes"][index] if isinstance(batch["classes"], list) else batch["classes"]
        gt_boxes = gt_boxes.detach().cpu()
        for box, label in zip(
            _cxcywh_norm_to_xyxy(gt_boxes, width, height),
            gt_labels.reshape(-1),
        ):
            x1, y1, x2, y2 = [float(v) for v in box]
            if x2 < x1 or y2 < y1:
                continue
            draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=2)
            draw.text((x1 + 2, y1 + 2), str(int(label)), fill=(0, 200, 0))
        if predictions is not None and index < len(predictions):
            # Predictions must be in the SAME frame as the drawn image
            # (input-grid xyxy) — original-frame boxes need rescaling first.
            original_size = batch.get("original_size")
            if original_size is not None and index < len(original_size):
                orig = original_size[index]
            else:
                orig = (width, height)  # already input-frame
            scale_x = width / float(orig[0])
            scale_y = height / float(orig[1])
            for box in predictions[index]["bboxes"]:
                x1, y1, x2, y2 = [float(v) for v in box]
                x2, y2 = x1 + (x2 - x1) * scale_x, y1 + (y2 - y1) * scale_y
                x1, y1 = x1 * scale_x, y1 * scale_y
                if x2 < x1 or y2 < y1:
                    continue
                draw.rectangle([x1, y1, x2, y2], outline=(220, 0, 0), width=2)
        path = out_dir / f"example_{index}.jpg"
        grid.save(path)
        written.append(path)
    return written


def draw_prediction(
    image: Any,
    result: Any,
    class_names: dict[int, str] | None = None,
    color: tuple[int, int, int] = (220, 0, 0),
    min_score: float = 0.0,
) -> Image.Image:
    """Render one task result as an overlay on the source image.

    ``result`` is a task-model record (``bboxes``/``scores``/``labels``,
    plus optional masks, polygons, keypoints, semantic mask or depth) or an
    :class:`~autovision.results.InferenceResult`. The renderer
    dispatches on the populated fields, so pose, OBB, classification and
    segmentation records no longer raise on the missing ``bboxes`` key.

    ``image`` accepts a path, a CHW tensor (0-1 or 0-255), an HWC NumPy
    array, or a PIL image; the input is never mutated.
    """
    import pathlib

    import numpy as np

    from autovision.data.file_helpers import open_image
    from autovision.visualize.annotators import annotate_result

    if isinstance(image, (str, pathlib.Path)):
        canvas = Image.fromarray(open_image(image).permute(1, 2, 0).numpy()).convert("RGB")
    elif isinstance(image, torch.Tensor):
        tensor = image.detach().cpu().float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor[:3]
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3].permute(2, 0, 1)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
        array = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
        canvas = Image.fromarray(array).convert("RGB")
    elif isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        elif array.ndim == 3 and array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=2)
        elif array.ndim == 3 and array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            if array.max() > 1.5:
                array = array / 255.0
            else:
                array = np.clip(array, 0, 1)
            array = (array * 255).astype(np.uint8)
        canvas = Image.fromarray(array).convert("RGB")
    else:
        canvas = image.convert("RGB")

    return annotate_result(
        canvas,
        result,
        class_names=class_names,
        min_score=min_score,
        box_color=color,
    )
