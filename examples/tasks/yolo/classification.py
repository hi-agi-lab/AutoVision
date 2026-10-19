#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""YOLO classification inference on a bundled image.

Requires the AGPL ``ultralytics`` extra:

    pip install -e ".[ultralytics]"
    python examples/tasks/yolo/classification.py
"""

from __future__ import annotations

import torch
from _common import build_parser, load_model, predict, save_and_report

from autovision.results import InferenceResult


def main() -> None:
    args = build_parser(
        "YOLO classification inference (Ultralytics)",
        default_out="runs/examples/yolo_classification",
        default_model="yolo11n-cls.pt",
    ).parse_args()
    model = load_model(args)
    result = predict(model, args)
    probabilities = result.probs
    detections = InferenceResult(
        scores=torch.tensor([float(probabilities.top1conf)], dtype=torch.float32),
        labels=torch.tensor([int(probabilities.top1)], dtype=torch.long),
    )
    save_and_report(
        detections, args, names=result.names, title="yolo_classification"
    )


if __name__ == "__main__":
    main()
