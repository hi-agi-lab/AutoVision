# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Shared helpers for the Ultralytics YOLO examples.

These examples use the external ``ultralytics`` package (AGPL-3.0), which is
an optional extra of AutoVision:

    pip install -e ".[ultralytics]"

The dependency is imported lazily so ``--help`` works without it. Weights are
cached under ``$AUTOVISION_CACHE_DIR/yolo`` (default
``~/.cache/autovision/yolo``) instead of the repository root.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

ASSETS = Path(__file__).resolve().parent.parent / "assets"
#: Default CC0 demo image for all five YOLO examples.
AIRPORT = ASSETS / "airport_aerial.jpg"
DEFAULT_IMAGE = AIRPORT
YOLO_CACHE = (
    Path(os.environ.get("AUTOVISION_CACHE_DIR", Path.home() / ".cache" / "autovision"))
    / "yolo"
)
MODELSCOPE_BASE = (
    "https://www.modelscope.cn/models/Ultralytics/YOLO11/resolve/master"
)
#: ``Ultralytics/YOLO11`` on ModelScope ships detection / segmentation / pose
#: weights; OBB and classification are not in that repository.
MODELSCOPE_WEIGHTS = frozenset(
    [f"yolo11{size}.pt" for size in "nsmlx"]
    + [f"yolo11{size}-{task}.pt" for size in "nsmlx" for task in ("seg", "pose")]
)


def build_parser(
    description: str,
    *,
    default_out: str,
    default_model: str,
    default_image: Path = DEFAULT_IMAGE,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=Path, default=default_image)
    parser.add_argument(
        "--model",
        default=default_model,
        help="Ultralytics weight name (downloaded to the cache) or local path",
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto = let Ultralytics choose",
    )
    parser.add_argument("--out", type=Path, default=Path(default_out))
    return parser


def _model_path(name: str) -> Path:
    path = Path(name)
    if path.parent == Path(".") and path.suffix.lower() == ".pt":
        YOLO_CACHE.mkdir(parents=True, exist_ok=True)
        return YOLO_CACHE / path.name
    return path


def _download_modelscope(path: Path) -> None:
    """Download one official YOLO11 weight from ModelScope atomically."""
    import requests

    url = f"{MODELSCOPE_BASE}/{path.name}"
    print(f"downloading = {url}")
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response.raw, handle)
        if temporary.stat().st_size < 1_000_000:
            raise ValueError("downloaded payload is unexpectedly small")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_weight(path: Path) -> None:
    """Fetch detection/seg/pose weights from ModelScope when available."""
    if path.exists() or path.parent != YOLO_CACHE:
        return
    if path.name not in MODELSCOPE_WEIGHTS:
        return
    try:
        _download_modelscope(path)
    except Exception as error:  # noqa: BLE001 - fall back to Ultralytics
        print(
            f"ModelScope download failed ({error}); falling back to the "
            "Ultralytics default source."
        )


def load_model(args: argparse.Namespace) -> Any:
    """Import Ultralytics lazily, then load/download the YOLO checkpoint."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "The YOLO examples need the optional AGPL package:\n"
            "  pip install -e '.[ultralytics]'\n"
            "The rest of AutoVision remains Apache-2.0."
        ) from error
    path = _model_path(args.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_weight(path)
    print(f"weights     = {path}")
    return YOLO(str(path))


def predict(model: Any, args: argparse.Namespace) -> Any:
    device = None if args.device == "auto" else args.device
    results = model(
        str(args.image), conf=args.threshold, device=device, verbose=False
    )
    return results[0]


def _tolist(value: Any) -> list[Any]:
    return value.detach().cpu().tolist() if hasattr(value, "detach") else list(value)


def save_and_report(
    result: Any,
    args: argparse.Namespace,
    *,
    names: dict[int, str] | None,
    title: str,
) -> None:
    """Save overlay + JSON and print a compact summary for one YOLO result."""
    from autovision.visualize import draw_prediction

    args.out.mkdir(parents=True, exist_ok=True)
    rendered = draw_prediction(
        args.image, result, class_names=names, min_score=args.threshold
    )
    rendered.save(args.out / "prediction.jpg")
    payload = result.to_json(masks="rle")
    (args.out / "detections.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )

    labels = result.labels
    if labels is not None:
        values = [int(value) for value in _tolist(labels)]
        named = [
            names.get(value, str(value)) if names else str(value)
            for value in values
        ]
        print(f"detections  = {len(values)}")
        print(f"classes     = {dict(Counter(named).most_common())}")
        if not values:
            print(
                "note: no detections above the threshold; lower --threshold "
                "to inspect."
            )
    else:
        print("detections  = n/a (image-level output)")
    print(f"title       = {title}")
    print(f"prediction  = {args.out / 'prediction.jpg'}")
    print(f"json        = {args.out / 'detections.json'}")
