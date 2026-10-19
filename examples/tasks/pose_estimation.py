#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pose estimation inference: keypoints -> COCO skeleton overlay.

No training. By default this loads the official MIT ``gtrpose_s_coco``
checkpoint and runs it on the bundled street image. Pass
``--weights path/to/exported.pt`` to use a model from
``pose_estimation_train.py``.
"""

from __future__ import annotations

from _common import (
    build_inference_parser,
    gtr_requirement,
    load_class_names,
    predict_and_annotate,
    resolve_device,
)

import autovision


def main() -> None:
    parser = build_inference_parser(
        "Pose estimation inference (GTR pose / exported .pt)",
        default_out="runs/examples/pose_estimation",
    )
    parser.add_argument("--checkpoint", default="gtrpose_s_coco")
    parser.add_argument("--variant", choices=("s", "m", "l", "x"), default="s")
    args = parser.parse_args()

    if args.weights is not None:
        model = autovision.load_model(args.weights, device=resolve_device(args.device))
        names = load_class_names(args.class_names) or {0: "person", 1: "person"}
    else:
        gtr_requirement()
        from autovision.tasks.pose_estimation.gtr import GTRPoseEstimation

        model = GTRPoseEstimation(
            num_classes=2,
            checkpoint=args.checkpoint,
            variant=args.variant,
            image_size=(640, 640),
        )
        model.eval().to(resolve_device(args.device))
        names = {0: "person", 1: "person"}
    predict_and_annotate(model, args, names, title="pose_estimation")


if __name__ == "__main__":
    main()
