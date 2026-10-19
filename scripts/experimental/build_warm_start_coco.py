#!/usr/bin/env python3
"""Build the self-produced COCO warm-start checkpoint (design doc 28, §6.2).

Replaces the retired reference-system S3 warm-start assets with a licensed,
self-produced equivalent (ledger P3-1). Pipeline:

    1. init : decoder/encoder from the Apache-2.0 D-FINE official COCO
       components (registered assets ``d-fine/dfine_{s,m,l}_coco``);
       backbone from a license-clean line — timm/torchvision pretrained
       weights, or ``dinov2/*-notpretrained`` initialized from this
       framework's own self-supervised pretraining output (the CC-BY-NC
       HF checkpoints are gated and NOT part of the clean path);
    2. assemble : the target detection network (family per --family);
    3. finetune : short standard COCO recipe through the regular
       detection training pipeline;
    4. export : AVWarmStart v1 checkpoint (registry/warm_start.py) with
       mandatory license metadata.

STATUS: skeleton. The ECViT assembly loader was removed together with the
ECViT backbone line (commit 588530d), so step 2 needs a new width-recipe
assembly for the current backbone lines (DINOv2/DINOv3/timm/torchvision)
before step 3's COCO finetune can run. The save/export path below is
final and already tested.

Usage (once unblocked):
    python scripts/experimental/build_warm_start_coco.py \
        --family detr --backbone dinov2/vitl14 --classes 80 \
        --out weights/detr_warm_start_coco.pt \
        --spec-note "D-FINE decoder init + own-SSL backbone init"
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autovision.registry.assets import LicenseRef  # noqa: E402
from autovision.registry.warm_start import (  # noqa: E402
    WarmStartError,
    save_warm_start,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default="detr", help="detector family")
    parser.add_argument(
        "--backbone",
        default="dinov2/vitl14",
        help="backbone for the target network; the license-clean routes "
        "are timm/torchvision pretrained weights or a dinov2/-notpretrained "
        "build initialized from own-SSL weights. DINOv3 variants "
        "additionally require AUTOVISION_DINOV3_WEIGHTS=1 (decision D1-a), "
        "and CC-BY-NC DINOv2 HF checkpoints additionally require "
        "AUTOVISION_DINOV2_WEIGHTS=1 — neither is part of the clean path",
    )
    parser.add_argument("--classes", type=int, default=80, help="COCO class count")
    parser.add_argument(
        "--out",
        default="weights/av_warm_start_coco.pt",
        help="output AVWarmStart v1 path",
    )
    parser.add_argument("--spec-note", default="", help="free-form provenance note")
    return parser.parse_args()


def build(family: str, backbone: str, classes: int, out: Path, note: str) -> Path:
    # TODO(Phase 4, ledger P3-1): blocked on the detr-family assembly step
    # for the current backbone lines (the ECViT-specific assembler went
    # away with that backbone line) plus a COCO finetune run. The export
    # path is final.
    raise NotImplementedError(
        "COCO warm-start build needs a detr-family assembly step for the "
        "current backbone lines (DINOv2/DINOv3/timm/torchvision; the "
        "ECViT assembler was removed with that line) and a COCO finetune "
        "run; see the provenance plan section 6.2."
    )
    # state_dict = assemble_target_network(family=family, backbone=backbone)  # noqa: E800
    # state_dict = finetune_on_coco(state_dict, classes=classes)  # noqa: E800
    return save_warm_start(
        out,
        model={},  # replaced by the finetuned state dict
        spec={
            "family": family,
            "backbone": backbone,
            "classes": classes,
            "created": _dt.date.today().isoformat(),
            "note": note,
        },
        license=LicenseRef(id="Apache-2.0", origin="self-trained (this run)"),
    )


def main() -> int:
    args = parse_args()
    try:
        out_path = build(
            family=args.family,
            backbone=args.backbone,
            classes=args.classes,
            out=Path(args.out),
            note=args.spec_note,
        )
    except NotImplementedError as error:
        print(f"BLOCKED: {error}")
        return 2
    except WarmStartError as error:
        print(f"ERROR: {error}")
        return 1
    print(f"warm-start written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
