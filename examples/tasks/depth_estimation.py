#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Depth estimation inference: metric depth map -> colorized overlay.

No training. By default this loads the official MIT ``gtrdepth_s``
checkpoint and runs it on the bundled street image. Pass
``--weights path/to/exported.pt`` to use a model from
``depth_estimation_train.py``.
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
        "Depth estimation inference (GTR depth / exported .pt)",
        default_out="runs/examples/depth_estimation",
    )
    parser.add_argument("--checkpoint", default="gtrdepth_s")
    parser.add_argument("--variant", choices=("s", "m", "l", "x"), default="s")
    args = parser.parse_args()

    if args.weights is not None:
        model = autovision.load_model(args.weights, device=resolve_device(args.device))
        names = load_class_names(args.class_names)
    else:
        gtr_requirement()
        from autovision.tasks.depth_estimation.gtr import GTRDepthEstimation

        model = GTRDepthEstimation(
            checkpoint=args.checkpoint,
            variant=args.variant,
            image_size=(640, 640),
        )
        model.eval().to(resolve_device(args.device))
        names = None
    predict_and_annotate(model, args, names, title="depth_estimation")


if __name__ == "__main__":
    main()
