#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""YOLO detection inference on a bundled image.

Requires the AGPL ``ultralytics`` extra:

    pip install -e ".[ultralytics]"
    python examples/tasks/yolo/detection.py
"""

from __future__ import annotations

import torch
from _common import build_parser, load_model, predict, save_and_report

from autovision.results import InferenceResult


def main() -> None:
    args = build_parser(
        "YOLO detection inference (Ultralytics)",
        default_out="runs/examples/yolo_detection",
        default_model="yolo11n.pt",
    ).parse_args()
    model = load_model(args)
    result = predict(model, args)
    boxes = result.boxes
    detections = InferenceResult(
        bboxes=boxes.xyxy.cpu() if boxes is not None else torch.empty(0, 4),
        scores=boxes.conf.cpu() if boxes is not None else torch.empty(0),
        labels=(
            boxes.cls.long().cpu() if boxes is not None else torch.empty(0, dtype=torch.long)
        ),
    )
    save_and_report(detections, args, names=result.names, title="yolo_detection")


if __name__ == "__main__":
    main()
