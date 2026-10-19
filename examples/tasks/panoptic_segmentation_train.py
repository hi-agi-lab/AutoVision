#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Panoptic segmentation: masks PNG + segments JSON -> predict.

The default demo has one thing class and one stuff class. Requires the
``transformers`` extra (EOMT). Real data:

    {"train": {"images": dir, "masks": dir, "segments": "panoptic.json"},
     "val":   {...}}
"""

from __future__ import annotations

from _common import (
    EOMT_MODEL_ARGS,
    build_parser,
    demo_root,
    dry_run,
    finish,
    panoptic_demo,
)

import autovision


def main() -> None:
    args = build_parser(
        "Panoptic segmentation example (masks + segments JSON by default)",
        default_out="runs/examples/panoptic_segmentation_train",
    ).parse_args()

    data = args.data or panoptic_demo(demo_root(args, "panoptic_segmentation"))
    if args.dry_run:
        dry_run("panoptic_segmentation", data)
        return

    result = autovision.train_panoptic_segmentation(
        out=args.out,
        data=data,
        model="dinov2/eomt",
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
