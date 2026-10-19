#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""int8 vs fp32 COCO AP report (roadmap 90d, phase 2).

Exports fp32 and int8 artifacts for one detection checkpoint, evaluates both
on the same split with the same preprocessing/evaluator, and writes a
verdict (AP drop threshold). int8 calibration must come from a set disjoint
from the evaluation split; pass ``--calibration-images``.

ONNX Runtime path:

    python scripts/measure_int8_accuracy.py --model best.pt --data coco.yaml \
        --calibration-images calib/ --out bench/int8 --max-images 5000

TensorRT path (requires CUDA + tensorrt):

    python scripts/measure_int8_accuracy.py --model best.pt --data coco.yaml \
        --backend tensorrt --calibration-images calib/ --out bench/int8-trt
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import autovision  # noqa: E402
from autovision.cli.train_task.object_detection import (  # noqa: E402
    _DATA_FORMATS,
    _load_data_config,
)
from autovision.config.validate import pydantic_model_validate  # noqa: E402
from autovision.export.int8_report import (  # noqa: E402
    summarize_int8_accuracy,
)
from autovision.inference import InferenceEngine  # noqa: E402
from autovision.metrics.map_evaluator import COCOMAPEvaluator  # noqa: E402


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as handle:
        return int(handle.width), int(handle.height)


def _evaluate_engine(
    engine: InferenceEngine,
    data_args: Any,
    *,
    split: str,
    max_images: int | None,
    threshold: float,
    batch_size: int,
) -> tuple[dict[str, float], int]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(data_args.list_image_info(split)):
        if max_images is not None and index >= max_images:
            break
        rows.append(row)
    if not rows:
        raise SystemExit(f"The '{split}' split produced no images to evaluate.")
    evaluator = COCOMAPEvaluator(categories=dict(data_args.classes()))
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        paths: list[Path] = []
        targets: list[dict[str, Any]] = []
        sizes: list[tuple[int, int]] = []
        for row in chunk:
            path = Path(row["image_path"])
            width, height = _image_size(path)
            target: dict[str, Any] = {
                "boxes": torch.tensor(
                    json.loads(row["bboxes"]), dtype=torch.float32
                ).reshape(-1, 4),
                "labels": torch.tensor(
                    json.loads(row["class_labels"]), dtype=torch.long
                ),
            }
            if "iscrowds" in row:
                target["iscrowds"] = torch.tensor(
                    json.loads(row["iscrowds"]), dtype=torch.long
                )
            paths.append(path)
            targets.append(target)
            sizes.append((width, height))
        predictions = engine.run_batch(paths, threshold=threshold)
        evaluator.update(predictions)
        evaluator.update_targets(targets, sizes)
    return evaluator.compute(), len(rows)


def _export(
    network: Any,
    backend: str,
    out_dir: Path,
    precision: str,
    calibration_images: str | None,
) -> Path:
    if backend == "onnx":
        path = out_dir / f"{precision}.onnx"
        if precision == "fp32":
            network.export_onnx(path, verify=True)
        else:
            network.export_onnx_int8(
                path, calibration_data=calibration_images, verify=True
            )
        return path
    path = out_dir / f"{precision}.engine"
    network.export_tensorrt(
        path,
        precision=precision,
        calibration_data=calibration_images if precision == "int8" else None,
    )
    return path


def _hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        import onnxruntime

        info["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass
    try:
        import tensorrt

        info["tensorrt"] = tensorrt.__version__
    except ImportError:
        pass
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="exported detection .pt")
    parser.add_argument("--data", required=True, help="COCO/YOLO data config")
    parser.add_argument("--out", default="bench/int8", help="report directory")
    parser.add_argument("--backend", choices=("onnx", "tensorrt"), default="onnx")
    parser.add_argument(
        "--calibration-images", default=None,
        help="calibration image file/dir (disjoint from the evaluation split)",
    )
    parser.add_argument("--fp32-artifact", default=None, help="reuse an existing fp32 artifact")
    parser.add_argument("--int8-artifact", default=None, help="reuse an existing int8 artifact")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-ap-drop", type=float, default=1.0)
    parser.add_argument("--no-fail", action="store_true", help="do not exit 1 on a failed verdict")
    parser.add_argument("--device", default=None, help="torch device for TensorRT (cuda)")
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        raise SystemExit("--max-images must be >= 1.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1.")
    if args.int8_artifact is None and not args.calibration_images:
        raise SystemExit("int8 export needs --calibration-images (or --int8-artifact).")

    payload = _load_data_config(args.data)
    data_args = pydantic_model_validate(_DATA_FORMATS[payload["format"]], payload)
    base_dir = Path(args.data).parent if isinstance(args.data, str) and Path(args.data).is_file() else None
    data_args.resolve_data_paths(base_dir)
    if not hasattr(data_args, args.split):
        raise SystemExit(f"data config has no '{args.split}' split.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    network = autovision.load_task_model(args.model, device=args.device or "cpu")
    started = time.perf_counter()

    fp32_artifact = (
        Path(args.fp32_artifact)
        if args.fp32_artifact
        else _export(network, args.backend, out_dir, "fp32", None)
    )
    int8_artifact = (
        Path(args.int8_artifact)
        if args.int8_artifact
        else _export(
            network, args.backend, out_dir, "int8", args.calibration_images
        )
    )

    fp32_engine = InferenceEngine(
        network, backend=args.backend, artifact=fp32_artifact, device=args.device
    )
    int8_engine = InferenceEngine(
        network, backend=args.backend, artifact=int8_artifact, device=args.device
    )
    fp32_metrics, images = _evaluate_engine(
        fp32_engine,
        data_args,
        split=args.split,
        max_images=args.max_images,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    int8_metrics, _ = _evaluate_engine(
        int8_engine,
        data_args,
        split=args.split,
        max_images=args.max_images,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    report = summarize_int8_accuracy(
        fp32_metrics,
        int8_metrics,
        max_ap_drop=args.max_ap_drop,
        model=args.model,
        backend=args.backend,
        calibration=args.calibration_images or "prebuilt",
        images=images,
        hardware={**_hardware(), "elapsed_s": round(time.perf_counter() - started, 2)},
    ).save(out_dir)
    print(report.summary_text())
    print(f"report: {report.report_md}")
    print(f"json:   {report.report_json}")
    if not report.passed and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
