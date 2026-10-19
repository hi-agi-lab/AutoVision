#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Semantic segmentation: paired images/masks -> predict.

Masks are PNG class-index maps; label 255 is ignored. The default demo writes
three 64x64 images and masks under ``<out>/_demo_data/semantic``.
"""

from __future__ import annotations

from _common import build_parser, demo_root, dry_run, finish, semantic_demo

import autovision


def main() -> None:
    args = build_parser(
        "Semantic segmentation example (paired images/masks by default)",
        default_out="runs/examples/semantic_segmentation_train",
    ).parse_args()

    data = args.data or semantic_demo(demo_root(args, "semantic_segmentation"))
    if args.dry_run:
        dry_run("semantic_segmentation", data)
        return

    result = autovision.train_semantic_segmentation(
        out=args.out,
        data=data,
        model="torchvision/resnet18",
        num_classes=args.num_classes or 3,
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
