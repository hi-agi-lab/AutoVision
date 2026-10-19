#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Depth estimation: images + depth maps -> depth -> predict.

The default demo writes float32 ``.npy`` depth maps in metres. Training the
GTR family needs ``pip install -e '.[gtr]'`` and a CUDA machine (Triton
kernels). Use ``--dry-run`` to inspect the data contract without training.
"""

from __future__ import annotations

from _common import (
    build_parser,
    demo_root,
    depth_demo,
    dry_run,
    finish,
    gtr_requirement,
)

import autovision


def main() -> None:
    args = build_parser(
        "Depth estimation example (images + .npy depth by default)",
        default_out="runs/examples/depth_estimation_train",
    ).parse_args()

    data = args.data or depth_demo(demo_root(args, "depth_estimation"))
    if args.dry_run:
        dry_run("depth_estimation", data)
        return
    gtr_requirement()

    result = autovision.train_depth_estimation(
        out=args.out,
        data=data,
        model="gtr/s",
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        overwrite=args.overwrite,
        model_args={
            "image_size": (128, 128),
            "load_checkpoint": False,
            "lr_warmup_steps": 2,
        },
        trainer_args={"val_every_num_steps": max(1, args.steps)},
    )
    finish(result, args)


if __name__ == "__main__":
    main()
