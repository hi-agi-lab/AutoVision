#!/usr/bin/env python3
"""Official-protocol COCO keypoint AP for the GTR pose family (benchmarks.md §1).

Reproduces the ``gtrpose/<s|m|l|x>`` rows of docs/benchmarks.md §1: the
registered official EMA checkpoints (``gtrpose_{s,m,l,x}_coco``) through
this framework's full chain — COCO ``person_keypoints`` data plane -> val
transform -> network -> official postprocess (plain top-k, no score gate)
-> pycocotools keypoint evaluator. This is the same chain the training val
loop uses (``GTRPoseEstimationTrain.validation_step``).

Requires COCO val2017 images + ``person_keypoints_val2017.json`` on disk.

Usage::

    python scripts/eval_gtr_pose_ap.py --variant s \\
        --data '{"val": {"annotations": ".../person_keypoints_val2017.json",
                         "images": ".../val2017"}}'
    python scripts/eval_gtr_pose_ap.py --variant m --data coco_pose_val.yaml
    # smoke: a subset
    python scripts/eval_gtr_pose_ap.py --variant s --data coco_pose_val.yaml --max-images 500

The ``--data`` payload may inline the checkpoint: ``"checkpoint":
"gtrpose_s_coco"`` (registered asset name) or a path; the CLI flag wins.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402


def _load_payload(data: str) -> dict[str, Any]:
    import yaml

    path = Path(data)
    if path.is_file():
        loaded = yaml.safe_load(path.read_text())
    else:
        loaded = json.loads(data)
    if not isinstance(loaded, dict) or "val" not in loaded:
        raise SystemExit("--data must provide a 'val' entry with 'annotations' and 'images'.")
    return loaded


def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Same batch assembly as the training val loop."""
    return {
        "image": torch.stack([item["image"] for item in items]),
        "original_size": [item["original_size"] for item in items],
        "index": [int(item["index"]) for item in items],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="s", choices=["s", "m", "l", "x"])
    parser.add_argument("--data", required=True, help="YAML file or inline JSON with 'val'")
    parser.add_argument("--checkpoint", default=None, help="registered asset name or path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from torch.utils.data import DataLoader, Subset

    from autovision.config.validate import pydantic_model_validate
    from autovision.data.coco_keypoint_dataset import CocoKeypointDataset
    from autovision.engine.engine import resolve_device
    from autovision.metrics.pose_estimation import COCOKeypointAPEvaluator
    from autovision.tasks.pose_estimation.gtr.config import (
        GTRPoseEstimationTrainArgs,
    )
    from autovision.tasks.pose_estimation.gtr.task_model import GTRPoseEstimation
    from autovision.tasks.pose_estimation.gtr.transforms import build_pose_transform

    payload = _load_payload(args.data)
    entry = payload["val"]
    checkpoint = args.checkpoint or payload.get("checkpoint") or f"gtrpose_{args.variant}_coco"

    model_args = pydantic_model_validate(
        GTRPoseEstimationTrainArgs,
        {"variant": args.variant, "checkpoint": checkpoint, "load_checkpoint": True},
    )

    dataset = CocoKeypointDataset(
        entry["annotations"],
        entry["images"],
        transform=build_pose_transform(model_args, train=False),
        train=False,
        num_body_points=model_args.num_body_points,
    )
    if args.max_images is not None:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    print(f"# val images: {len(dataset)} | checkpoint: {checkpoint}")

    device = resolve_device(args.device)
    torch.manual_seed(model_args.seed if hasattr(model_args, "seed") else 0)
    network = GTRPoseEstimation(
        variant=model_args.variant,
        image_size=model_args.image_size,
        num_queries=model_args.num_queries,
        num_decoder_layers=model_args.num_decoder_layers,
        dim_feedforward=model_args.dim_feedforward,
        num_body_points=model_args.num_body_points,
        checkpoint=checkpoint,
    ).to(device)
    network.eval()

    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    evaluator = COCOKeypointAPEvaluator(entry["annotations"], categories=base.categories)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_collate,
    )
    with torch.no_grad():
        for batch in loader:
            out = network.forward_backend(batch["image"].to(device))
            decoded = network.postprocess(
                out,
                [
                    {"orig_w": width, "orig_h": height}
                    for width, height in batch["original_size"]
                ],
            )
            evaluator.update(decoded, [int(index) for index in batch["index"]])
    metrics = evaluator.compute()
    print(json.dumps({key: round(float(value), 4) for key, value in metrics.items()}))
    print(
        f"RESULT variant={args.variant} checkpoint={checkpoint} "
        f"images={len(dataset)} kp_map={metrics['kp_map']:.2f} "
        f"kp_map_50={metrics['kp_map_50']:.2f} kp_mar={metrics['kp_mar']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
