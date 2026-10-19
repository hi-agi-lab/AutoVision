#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pose estimation: COCO person_keypoints -> keypoints -> predict.

The default demo writes a two-image COCO keypoint dataset. Training the GTR
family needs ``pip install -e '.[gtr]'`` and a CUDA machine (Triton kernels).
Use ``--dry-run`` to inspect the data contract without training.
"""

from __future__ import annotations

from _common import build_parser, demo_root, dry_run, finish, gtr_requirement, pose_demo

import autovision


def main() -> None:
    args = build_parser(
        "Pose estimation example (COCO person_keypoints by default)",
        default_out="runs/examples/pose_estimation_train",
    ).parse_args()

    data = args.data or pose_demo(demo_root(args, "pose_estimation"))
    if args.dry_run:
        dry_run("pose_estimation", data)
        return
    gtr_requirement()

    result = autovision.train_pose_estimation(
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
            "num_queries": 16,
            "load_checkpoint": False,
            "lr_warmup_steps": 2,
        },
        trainer_args={"val_every_num_steps": max(1, args.steps)},
    )
    finish(result, args)


if __name__ == "__main__":
    main()
