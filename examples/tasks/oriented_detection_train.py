#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Oriented object detection: YOLO-OBB -> predict.

The default demo writes normalized 8-point labels for two classes. Real data
is a YOLO-OBB ``data.yaml``:

    format: yolo-obb
    train: images/train
    val: images/val
    names: {0: plane, 1: ship}
"""

from __future__ import annotations

from _common import (
    OBB_TRANSFORM_ARGS,
    RTDETR_MODEL_ARGS,
    build_parser,
    demo_root,
    dry_run,
    finish,
    obb_demo,
)

import autovision


def main() -> None:
    args = build_parser(
        "Oriented object detection example (YOLO-OBB by default)",
        default_out="runs/examples/oriented_detection_train",
    ).parse_args()

    data = args.data or obb_demo(demo_root(args, "oriented_detection"))
    if args.dry_run:
        dry_run("oriented_object_detection", data)
        return

    result = autovision.train_oriented_object_detection(
        out=args.out,
        data=data,
        model="rtdetr-obb/resnet18",
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        overwrite=args.overwrite,
        model_args=RTDETR_MODEL_ARGS,
        transform_args=OBB_TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": max(1, args.steps)},
    )
    finish(result, args)


if __name__ == "__main__":
    main()
