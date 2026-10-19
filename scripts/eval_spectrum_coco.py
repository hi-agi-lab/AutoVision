#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Evaluate an exported spectrum model on a COCO-style split (Phase 0 skeleton).

Thin, reproducible wrapper around
:func:`autovision.evaluate_object_detection`: the exported task model runs
its own preprocess/forward/postprocess and pycocotools computes the AP.
The script writes the metrics as JSON and is the form the model card's
``metrics`` field expects for a released tier.

Usage::

    python scripts/eval_spectrum_coco.py \
        --weights runs/n-12ep/exported_models/exported_best.pt \
        --data coco.yaml --split val --out runs/n-12ep/eval_val2017.json \
        --max-images 5000

``--dry-run`` prints the resolved call without loading weights.
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="exported task model (*.pt), e.g. exported_best.pt",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="COCO/YOLO data spec; 'coco8' uses the synthetic smoke set",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="optional JSON report path")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data: Any = (
        autovision.ensure_coco8() if args.data == "coco8" else args.data
    )
    config = {
        "weights": str(args.weights),
        "data": data if isinstance(data, dict) else str(data),
        "split": args.split,
        "threshold": args.threshold,
        "max_images": args.max_images,
        "device": args.device,
    }
    if args.dry_run:
        print(json.dumps(config, indent=2, default=str))
        return 0

    network = autovision.load_task_model(args.weights, device=args.device)
    metrics = autovision.evaluate_object_detection(
        network,
        data=data,
        split=args.split,
        threshold=args.threshold,
        max_images=args.max_images,
    )
    report = {**config, "metrics": metrics}
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
