#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Instance segmentation inference: boxes + masks -> annotate.

No training. By default this loads the official MIT ``gtrseg_s_coco``
checkpoint and runs it on the bundled street image. Pass
``--weights path/to/exported.pt`` to use a model from
``instance_segmentation_train.py`` (including EOMT checkpoints).
"""

from __future__ import annotations

from _common import (
    _coco_names,
    build_inference_parser,
    gtr_requirement,
    load_class_names,
    predict_and_annotate,
    resolve_device,
)

import autovision


def main() -> None:
    parser = build_inference_parser(
        "Instance segmentation inference (GTR seg / exported .pt)",
        default_out="runs/examples/instance_segmentation",
    )
    parser.add_argument("--checkpoint", default="gtrseg_s_coco")
    parser.add_argument("--variant", choices=("s", "m", "l", "x"), default="s")
    args = parser.parse_args()

    if args.weights is not None:
        model = autovision.load_model(args.weights, device=resolve_device(args.device))
        names = load_class_names(args.class_names) or _coco_names()
    else:
        gtr_requirement()
        from autovision.tasks.instance_segmentation.gtr import GTRSegObjectDetection

        model = GTRSegObjectDetection(
            num_classes=80,
            checkpoint=args.checkpoint,
            variant=args.variant,
            image_size=(640, 640),
        )
        model.eval().to(resolve_device(args.device))
        names = _coco_names()
    predict_and_annotate(model, args, names, title="instance_segmentation")


if __name__ == "__main__":
    main()
