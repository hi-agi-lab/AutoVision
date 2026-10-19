#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Semantic segmentation inference: label map -> colorized overlay.

No training. By default this loads the official MIT
``gtrsemseg_s_cityscapes`` checkpoint and runs it on the bundled street
image. Pass ``--weights path/to/exported.pt`` to use a model from
``semantic_segmentation_train.py`` (EOMT or linear head).
"""

from __future__ import annotations

from _common import (
    _cityscapes_names,
    build_inference_parser,
    gtr_requirement,
    load_class_names,
    predict_and_annotate,
    resolve_device,
)

import autovision


def main() -> None:
    parser = build_inference_parser(
        "Semantic segmentation inference (GTR semseg / exported .pt)",
        default_out="runs/examples/semantic_segmentation",
    )
    parser.add_argument("--checkpoint", default="gtrsemseg_s_cityscapes")
    parser.add_argument("--variant", choices=("s", "m", "l", "x"), default="s")
    args = parser.parse_args()

    if args.weights is not None:
        model = autovision.load_model(args.weights, device=resolve_device(args.device))
        names = load_class_names(args.class_names) or _cityscapes_names()
    else:
        gtr_requirement()
        from autovision.tasks.semantic_segmentation.gtr import (
            GTRSemanticSegmentation,
        )

        model = GTRSemanticSegmentation(
            num_classes=19,
            checkpoint=args.checkpoint,
            variant=args.variant,
            image_size=(1024, 1024),
        )
        model.eval().to(resolve_device(args.device))
        names = _cityscapes_names()
    predict_and_annotate(model, args, names, title="semantic_segmentation")


if __name__ == "__main__":
    main()
