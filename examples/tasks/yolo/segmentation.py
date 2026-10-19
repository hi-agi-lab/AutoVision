#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""YOLO instance segmentation inference on a bundled image.

Requires the AGPL ``ultralytics`` extra:

    pip install -e ".[ultralytics]"
    python examples/tasks/yolo/segmentation.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from _common import build_parser, load_model, predict, save_and_report
from PIL import Image

from autovision.results import InferenceResult


def main() -> None:
    args = build_parser(
        "YOLO segmentation inference (Ultralytics)",
        default_out="runs/examples/yolo_segmentation",
        default_model="yolo11n-seg.pt",
    ).parse_args()
    model = load_model(args)
    result = predict(model, args)
    boxes = result.boxes
    masks = result.masks.data.cpu() if result.masks is not None else None
    if masks is not None:
        with Image.open(args.image) as image:
            width, height = image.size
        if tuple(masks.shape[-2:]) != (height, width):
            masks = (
                F.interpolate(
                    masks.float()[None], size=(height, width), mode="nearest"
                )[0]
                > 0.5
            )
        masks = masks.bool()
    detections = InferenceResult(
        bboxes=boxes.xyxy.cpu() if boxes is not None else torch.empty(0, 4),
        scores=boxes.conf.cpu() if boxes is not None else torch.empty(0),
        labels=(
            boxes.cls.long().cpu() if boxes is not None else torch.empty(0, dtype=torch.long)
        ),
        masks=masks,
    )
    save_and_report(detections, args, names=result.names, title="yolo_segmentation")


if __name__ == "__main__":
    main()
