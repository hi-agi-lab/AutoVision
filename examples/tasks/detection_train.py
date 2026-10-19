#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Object detection training: train -> validate -> predict.

Default data is the bundled ``coco8`` smoke set, so this runs offline:

    python examples/tasks/detection_train.py --steps 2
    python examples/tasks/detection_train.py --steps 2 \
        --predict-image examples/tasks/assets/airport_aerial.jpg

For inference with released weights (no training), use
``examples/tasks/detection.py``. Real data: pass a COCO/YOLO ``data.yaml``
(see docs/index.md).
"""

from __future__ import annotations

from _common import (
    RTDETR_MODEL_ARGS,
    RTDETR_TRANSFORM_ARGS,
    build_parser,
    dry_run,
    finish,
)

import autovision


def main() -> None:
    args = build_parser(
        "Object detection training example (coco8 by default)",
        default_out="runs/examples/detection_train",
    ).parse_args()

    data = args.data or autovision.ensure_coco8()
    if args.dry_run:
        dry_run("detection", data)
        return

    result = autovision.train_object_detection(
        out=args.out,
        data=data,
        model="rtdetr/resnet18",
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        overwrite=args.overwrite,
        model_args=RTDETR_MODEL_ARGS,
        transform_args=RTDETR_TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": max(1, args.steps)},
    )
    finish(result, args)


if __name__ == "__main__":
    main()
