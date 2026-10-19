#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Train one public detection spectrum tier (Phase 0 skeleton).

This is the reproducible entry point referenced by the spectrum model cards.
It wraps :func:`autovision.train_object_detection` with the tier's model key
and the recipe hooks that exist today:

- ``detr/n``: distillation-first (the published recipe is 12ep on COCO);
  pass ``--teacher`` to enable cross-model distillation.
- ``detr/s``: D-FINE-S assembly warm start; pass ``--warm-start``.
- ``m/l/x``: planned; refused until the family lands.

``--smoke`` runs 3 steps on the synthetic coco8 set at 64px, so the wiring
can be checked without COCO. A full COCO run is not part of CI. For a
12ep COCO run, pass ``--steps`` computed as
``epochs * train_images // batch_size`` (118,287 images in train2017).

Usage::

    python scripts/train_spectrum.py --tier n --smoke --out runs/n-smoke
    python scripts/train_spectrum.py --tier n --data coco.yaml \
        --batch-size 16 --steps 88715 --teacher rfdetr/rf-detr-nano \
        --out runs/n-12ep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import autovision  # noqa: E402
from autovision.tasks.detection.detr.variants import (  # noqa: E402
    resolve_spectrum_tier,
)

_SMOKE_TRANSFORMS = {
    "scale_jitter": None,
    "photometric_distort_prob": 0.5,
    "random_zoom_out_prob": None,
    "random_iou_crop_prob": None,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tier", required=True, choices=("n", "s", "m", "l", "x"))
    parser.add_argument(
        "--data",
        default="coco8",
        help="COCO/YOLO data spec; 'coco8' uses the synthetic smoke set",
    )
    parser.add_argument("--out", required=True, help="run directory")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument(
        "--teacher",
        default=None,
        help="teacher model key or exported .pt (recommended for detr/n)",
    )
    parser.add_argument(
        "--warm-start",
        default=None,
        help="AVWarmStart .pt path (recommended for detr/s)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="3 steps on coco8 at 64px (wiring check, not a benchmark)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        data: Any = autovision.ensure_coco8()
        steps = 3
        batch_size: Any = 2
        image_size = 64
        num_workers: Any = 0
        transform_args: dict[str, Any] | None = dict(_SMOKE_TRANSFORMS)
    else:
        data = (
            autovision.ensure_coco8()
            if args.data == "coco8"
            else args.data
        )
        steps = args.steps
        batch_size = args.batch_size
        image_size = args.image_size
        num_workers = args.num_workers
        transform_args = None

    model_args: dict[str, Any] = {"image_size": (image_size, image_size)}
    if args.smoke:
        model_args.update(
            num_queries=50,
            num_top_queries=50,
            num_denoising=10,
        )
    if args.warm_start:
        model_args["warm_start"] = str(Path(args.warm_start))
    if args.teacher:
        teacher_path = Path(args.teacher)
        model_args["distill"] = (
            {"teacher_checkpoint": str(teacher_path)}
            if teacher_path.is_file()
            else {"teacher_model": args.teacher}
        )
    elif args.tier == "n" and not args.smoke:
        print(
            "warning: detr/n is distillation-first; pass --teacher for the "
            "published recipe (see docs/benchmarks.md §1.0)",
            file=sys.stderr,
        )
    elif args.tier == "s" and not args.smoke and not args.warm_start:
        print(
            "warning: detr/s published recipe uses the D-FINE-S assembly "
            "warm start; pass --warm-start <AVWarmStart.pt>",
            file=sys.stderr,
        )

    return {
        "model": f"detr/{args.tier}",
        "status": resolve_spectrum_tier(args.tier).status,
        "data": data if isinstance(data, dict) else str(data),
        "out": str(args.out),
        "steps": steps,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": args.seed,
        "precision": args.precision,
        "devices": args.devices,
        "model_args": model_args,
        "transform_args": transform_args,
        "trainer_args": {"val_every_num_steps": 2} if args.smoke else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    tier = resolve_spectrum_tier(args.tier)
    if tier.status == "planned":
        parser.error(
            f"detr/{args.tier} is planned; the product family has not landed "
            "(see docs/model_zoo.md §3)"
        )

    config = _resolve_config(args)
    if args.dry_run:
        print(json.dumps(config, indent=2, default=str))
        return 0

    result = autovision.train_object_detection(
        out=config["out"],
        data=config["data"],
        model=config["model"],
        steps=config["steps"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        devices=config["devices"],
        precision=config["precision"],
        seed=config["seed"],
        overwrite=args.overwrite,
        model_args=config["model_args"],
        transform_args=config["transform_args"],
        trainer_args=config["trainer_args"],
    )
    report = {
        "model": config["model"],
        "steps_run": result.steps_run,
        "checkpoint": str(result.checkpoint_path) if result.checkpoint_path else None,
        "best_checkpoint": (
            str(result.best_checkpoint_path) if result.best_checkpoint_path else None
        ),
        "exported_model": (
            str(result.exported_model_path) if result.exported_model_path else None
        ),
        "best_exported_model": (
            str(result.best_exported_model_path)
            if result.best_exported_model_path
            else None
        ),
        "best_metrics": result.best_metrics,
    }
    out_dir = Path(config["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spectrum_train.json").write_text(
        json.dumps({"config": config, "result": report}, indent=2, default=str)
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
