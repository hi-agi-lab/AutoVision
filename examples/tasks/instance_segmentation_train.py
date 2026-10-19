#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Instance segmentation: COCO instances -> masks + boxes -> predict.

The default demo writes polygon annotations for two categories. Requires the
``transformers`` extra (EOMT). Real data: COCO instances JSON + images:

    data = {"train": {"annotations": "instances.json", "images": "images"},
            "val":   {"annotations": "instances_val.json", "images": "images"}}
"""

from __future__ import annotations

from _common import (
    EOMT_MODEL_ARGS,
    build_parser,
    demo_root,
    dry_run,
    finish,
    instance_demo,
)

import autovision


def main() -> None:
    args = build_parser(
        "Instance segmentation example (COCO polygons by default)",
        default_out="runs/examples/instance_segmentation_train",
    ).parse_args()

    data = args.data or instance_demo(demo_root(args, "instance_segmentation"))
    if args.dry_run:
        dry_run("instance_segmentation", data)
        return

    result = autovision.train_instance_segmentation(
        out=args.out,
        data=data,
        model="dinov2/eomt",
        num_classes=args.num_classes or 2,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        overwrite=args.overwrite,
        model_args={**EOMT_MODEL_ARGS, "lr_warmup_steps": 2},
        trainer_args={"val_every_num_steps": max(1, args.steps)},
    )
    finish(result, args)


if __name__ == "__main__":
    main()
