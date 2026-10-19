#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Object detection inference: load weights -> predict -> annotate.

No training. By default this loads the released ``rfdetr/rf-detr-nano``
checkpoint and runs it on the bundled CC0 airport image:

    python examples/tasks/detection.py
    python examples/tasks/detection.py --image path/to/image.jpg --threshold 0.5
    python examples/tasks/detection.py --effect blur

``--effect`` renders through the annotators migrated from supervision
(``autovision.visualize.effects``): ``corner`` / ``round`` replace the plain
box with corner brackets or rounded boxes, while ``blur`` / ``pixelate``
apply privacy filtering to the detections before boxes and labels are drawn.

Pass ``--weights path/to/exported.pt`` to use a model produced by
``train_object_detection`` instead of the released checkpoint. For a full
training walkthrough, see ``examples/tasks/detection_train.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

import autovision
from autovision.results import InferenceResult
from autovision.tasks.detection.pretrained import pretrained_class_names
from autovision.visualize import (
    BlurAnnotator,
    BoxCornerAnnotator,
    LabelAnnotator,
    PixelateAnnotator,
    RoundBoxAnnotator,
    annotate_result,
    draw_prediction,
)

ASSET = Path(__file__).resolve().parent / "assets" / "airport_aerial.jpg"

#: Rendering styles backed by ``autovision.visualize.effects``.
EFFECTS = ("none", "corner", "round", "blur", "pixelate")


def _tolist(value):
    return value.detach().cpu().tolist() if hasattr(value, "detach") else list(value)


def _load_rgb(path: Path) -> Image.Image:
    """Decode ``path`` to an RGB Pillow canvas (same reader as draw_prediction)."""
    from autovision.data.file_helpers import open_image

    return Image.fromarray(open_image(path).permute(1, 2, 0).numpy()).convert("RGB")


def _render_effect(
    image: Path,
    result: InferenceResult,
    effect: str,
    class_names: dict[int, str] | None,
) -> Image.Image:
    """Render ``result`` with one supervision-derived effect."""
    canvas = _load_rgb(image)
    if effect == "corner":
        canvas = BoxCornerAnnotator().annotate(canvas, result)
        return LabelAnnotator().annotate(canvas, result, class_names=class_names)
    if effect == "round":
        canvas = RoundBoxAnnotator().annotate(canvas, result)
        return LabelAnnotator().annotate(canvas, result, class_names=class_names)
    privacy = BlurAnnotator() if effect == "blur" else PixelateAnnotator()
    canvas = privacy.annotate(canvas, result)
    return annotate_result(canvas, result, class_names=class_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ASSET)
    parser.add_argument(
        "--model",
        default="rfdetr/rf-detr-nano",
        help="released registry key (ignored when --weights is given)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="exported .pt from training; overrides --model",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--effect",
        choices=EFFECTS,
        default="none",
        help=(
            "supervision-derived rendering style: corner/round boxes or "
            "blur/pixelate privacy (default: none)"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto = CUDA when available, else CPU",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/examples/detection")
    )
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    if args.weights is not None:
        model = autovision.load_model(args.weights, device=device)
        names = None  # custom checkpoints carry their own class ids
    else:
        names = pretrained_class_names(args.model)
        model = autovision.detection(args.model, pretrained=True)
        model.to(device)

    prediction = model.predict(args.image, threshold=args.threshold)
    labels = _tolist(prediction["labels"])
    scores = _tolist(prediction["scores"])
    boxes = _tolist(prediction["bboxes"])
    class_names = [
        names[int(label)] if names is not None else str(int(label))
        for label in labels
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    if args.effect == "none":
        rendered = draw_prediction(
            args.image,
            prediction,
            class_names=names,
            min_score=args.threshold,
        )
    else:
        result = InferenceResult.from_record(prediction)
        if args.threshold > 0:
            result = result.filter_scores(args.threshold)
        rendered = _render_effect(args.image, result, args.effect, names)
    rendered.save(args.out / "prediction.jpg")
    records = [
        {"name": name, "score": float(score), "box": box}
        for name, score, box in zip(class_names, scores, boxes)
    ]
    (args.out / "detections.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False)
    )

    print(f"image       = {args.image} ({rendered.width}x{rendered.height})")
    print(f"weights     = {args.weights or args.model} @ threshold {args.threshold}")
    print(f"device      = {device}")
    print(f"effect      = {args.effect}")
    print(f"detections  = {len(records)}")
    print(f"classes     = {dict(Counter(class_names).most_common())}")
    if not records:
        print("note: no detections above the threshold; lower --threshold to inspect.")
    print(f"prediction  = {args.out / 'prediction.jpg'}")
    print(f"json        = {args.out / 'detections.json'}")


if __name__ == "__main__":
    main()
