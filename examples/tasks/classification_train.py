#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Image classification: ImageFolder train/val -> predict.

The default demo writes ``<out>/_demo_data/classification/{train,val}/`` with
two classes. Real data is the same ImageFolder layout:

    data = {"train": "data/train", "val": "data/val"}
"""

from __future__ import annotations

from _common import build_parser, classification_demo, demo_root, dry_run, finish

import autovision


def main() -> None:
    args = build_parser(
        "Image classification example (ImageFolder by default)",
        default_out="runs/examples/classification_train",
    ).parse_args()

    data = args.data or classification_demo(demo_root(args, "classification"))
    if args.dry_run:
        dry_run("classification", data)
        return

    result = autovision.train_image_classification(
        out=args.out,
        data=data,
        model="torchvision/resnet18",
        num_classes=args.num_classes or 2,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        overwrite=args.overwrite,
        model_args={"image_size": (64, 64), "lr_warmup_steps": 2},
        trainer_args={"val_every_num_steps": max(1, args.steps)},
    )
    finish(result, args)


if __name__ == "__main__":
    main()
