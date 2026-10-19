#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""YOLO oriented-box (OBB) inference on the bundled airport image.

Requires the AGPL ``ultralytics`` extra:

    pip install -e ".[ultralytics]"
    python examples/tasks/yolo/obb.py
"""

from __future__ import annotations

import torch
from _common import AIRPORT, build_parser, load_model, predict, save_and_report

from autovision.results import InferenceResult


def main() -> None:
    args = build_parser(
        "YOLO oriented-box inference (Ultralytics)",
        default_out="runs/examples/yolo_obb",
        default_model="yolo11n-obb.pt",
        default_image=AIRPORT,
    ).parse_args()
    model = load_model(args)
    result = predict(model, args)
    obb = result.obb
    detections = InferenceResult(
        polygons=obb.xyxyxyxy.cpu() if obb is not None else torch.empty(0, 4, 2),
        scores=obb.conf.cpu() if obb is not None else torch.empty(0),
        labels=(
            obb.cls.long().cpu() if obb is not None else torch.empty(0, dtype=torch.long)
        ),
    )
    save_and_report(detections, args, names=result.names, title="yolo_obb")


if __name__ == "__main__":
    main()
