# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pretrained detection builders (roadmap 90d, phase 1).

A user-facing handle can load a released detector without training:

    model = autovision.detection("rfdetr/rf-detr-small", pretrained=True)
    model.predict("image.jpg")

The builder is intentionally separate from the training entry: exported
artifacts are still rebuilt hermetically (``load_weights=False`` + state
dict), while this module is the only place that downloads and loads a
task-level checkpoint. Model families without registered task weights fail
with guidance instead of silently building a random head.

GTR needs the flash-linear-attention extra and CUDA (the GLA kernel has no
CPU fallback). Its loader assembles the official checkpoint through the
same warm-start + backbone path the training tests use; it is gated so a
CPU-only machine gets a readable error, never a half-initialized model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch

from autovision.errors import AutoVisionError
from autovision.registry.catalog import get_model_entry

__all__ = [
    "COCO_CLASSES",
    "build_pretrained_detection",
    "pretrained_class_names",
]


#: COCO 2017 80-class names in category-id order (the order every DETR
#: checkpoint in this framework was trained against). GTR/DETR task heads use
#: a contiguous 0..79 label space; RF-DETR keeps the original COCO
#: category-id head (0=N/A, 1..90 with gaps), see
#: :data:`COCO_CATEGORY_NAMES` / :data:`COCO_RFDETR_NUM_CLASSES`.
COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

#: COCO category ids for :data:`COCO_CLASSES` (the official ids skip 12, 26,
#: 29, 30, 45, 66, 68, 69, 71 and 83).
COCO_CATEGORY_IDS: tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
)

#: ``category_id -> name`` for the official COCO id space.
COCO_CATEGORY_NAMES: dict[int, str] = dict(zip(COCO_CATEGORY_IDS, COCO_CLASSES))

#: RF-DETR's released head keeps ``category_id`` labels plus the N/A slot 0.
COCO_RFDETR_NUM_CLASSES: int = max(COCO_CATEGORY_IDS) + 1


def pretrained_class_names(model_key: str) -> dict[int, str]:
    """Class-id → name map for a released detector.

    RF-DETR checkpoints keep COCO category ids (1..90, 0 = N/A); GTR/DETR
    checkpoints use the contiguous 0..79 order of :data:`COCO_CLASSES`.
    A custom fine-tune records its own classes in the exported payload and
    must not use this helper.
    """
    entry = get_model_entry(model_key)  # validates the key early.
    if entry.family == "rfdetr":
        return {
            index: COCO_CATEGORY_NAMES.get(index, "N/A")
            for index in range(COCO_RFDETR_NUM_CLASSES)
        }
    return dict(enumerate(COCO_CLASSES))


def _require_full_weights(model_key: str) -> Any:
    try:
        entry = get_model_entry(model_key)
    except KeyError as error:
        raise AutoVisionError(str(error)) from error
    if entry.pretrained != "full":
        raise AutoVisionError(
            f"'{model_key}' has no registered pretrained task weights "
            f"(status={entry.status}, pretrained={entry.pretrained}). "
            "Train it first (`.train(...)`) or choose a released model: "
            "rfdetr/rf-detr-{nano,small,medium,large} or "
            "detr/gtrvit-{s,m,l,x}."
        )
    return entry


def _build_rfdetr(model_key: str, device: str | torch.device) -> torch.nn.Module:
    from autovision.tasks.detection.rfdetr.config import (
        resolve_rfdetr_variant,
    )
    from autovision.tasks.detection.rfdetr.task_model import (
        RFDETRObjectDetection,
    )

    variant = model_key.split("/", 1)[1]
    if variant.endswith("-notpretrained"):
        raise AutoVisionError(
            f"'{model_key}' is the offline random-build twin; use a released "
            "variant or pass load_weights=True through the training args."
        )
    info = resolve_rfdetr_variant(variant)
    resolution = int(info["resolution"])
    network = RFDETRObjectDetection(
        num_classes=COCO_RFDETR_NUM_CLASSES,
        model=variant,
        image_size=(resolution, resolution),
        load_weights=True,
    )
    network.eval()
    return network.to(device)


def _build_gtr(model_key: str, device: str | torch.device) -> torch.nn.Module:
    # Check the GLA stack BEFORE importing the assembly modules: a CPU-only
    # or fla-less install should get the actionable message, not an
    # incidental import error from a transitive dependency.
    try:
        from autovision.tasks.detection.shared.gtr import gtr_available
    except Exception:  # noqa: BLE001 - packaging failure is unavailability
        gtr_available = None

    if gtr_available is None or not gtr_available():
        raise AutoVisionError(
            "GTR inference needs the 'gtr' extra (flash-linear-attention) and "
            "a CUDA device for the GLA kernel. Install "
            "`pip install 'autovision-framework[gtr]'` on a CUDA machine, or "
            "use RF-DETR for CPU/portable inference."
        )
    if not torch.cuda.is_available():
        raise AutoVisionError(
            "GTR GLA inference requires CUDA; this machine has no CUDA device. "
            "Use RF-DETR for CPU/portable inference."
        )

    from autovision.registry.gtr_assets import resolve_gtr_asset
    from autovision.tasks.detection.detr import (
        DETRObjectDetectionTrain,
        DETRObjectDetectionTrainArgs,
    )
    from autovision.tasks.detection.detr.dfine_official_warm_start import (
        build_dfine_official_warm_start,
    )
    from autovision.tasks.detection.detr.gtr_teacher import (
        extract_gtr_backbone,
        load_gtr_teacher,
    )

    tier = model_key.rsplit("-", 1)[1]
    if tier not in {"s", "m", "l", "x"}:
        raise AutoVisionError(f"Unknown GTR detection tier in '{model_key}'.")
    asset = resolve_gtr_asset(f"gtr_{tier}_coco")
    variant = f"gtrvit-{tier}"
    with tempfile.TemporaryDirectory(prefix="autovision-gtr-") as tmp:
        tmp_path = Path(tmp)
        warm_start = tmp_path / "warm_start.pt"
        build_dfine_official_warm_start(
            variant,
            out=warm_start,
            checkpoint=asset.path,
            backbone_pretrained=False,
            license_id=asset.license,
            origin=asset.source,
            source_format="gtr",
            num_classes=len(COCO_CLASSES),
        )
        backbone_weights = tmp_path / "backbone.pth"
        torch.save(
            extract_gtr_backbone(load_gtr_teacher(asset.path, asset=asset)),
            backbone_weights,
        )
        args = DETRObjectDetectionTrainArgs(
            backbone=variant,
            num_classes=len(COCO_CLASSES),
            image_size=(640, 640),
            warm_start=warm_start,
            backbone_weights=backbone_weights,
            backbone_pretrained=False,
        )
        network = DETRObjectDetectionTrain(args).model
    network.eval()
    return network.to(device)


def build_pretrained_detection(
    model_key: str, *, device: str | torch.device = "cpu"
) -> torch.nn.Module:
    """Build a released detector with its official COCO weights.

    Dispatches by family; raises :class:`AutoVisionError` for models whose
    catalog row has no full task weights. The returned network is in eval
    mode and moved to ``device``.
    """
    entry = _require_full_weights(model_key)
    if entry.family == "rfdetr":
        return _build_rfdetr(model_key, device)
    if entry.family == "gtr":
        return _build_gtr(model_key, device)
    raise AutoVisionError(
        f"No pretrained builder for family '{entry.family}' (model "
        f"'{model_key}'). Add a builder in tasks/detection/pretrained.py."
    )
