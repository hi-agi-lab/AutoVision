#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Reproduce the docs/benchmarks.md §2.2 numbers (P1 deployment breadth).

Runs ``autovision.benchmark`` over the new/existing backends with the parity
gate on: torch (eager), ONNX (CPU EP), OpenVINO (CPU), and optionally a
TensorRT int8 engine built from the given calibration images.

Usage::

    # 64px rows (torch / onnx / openvino)
    python scripts/measure_new_backends.py \
        --model runs/d3/exported_models/exported_best.pt \
        --out bench_64 --backends torch onnx openvino

    # 640px int8 row (needs tensorrt + CUDA; build is slow). ``--model`` also
    # accepts a benchmark spec like ``rtdetr/resnet18`` (random weights,
    # speed only); ``--quantization-exclude auto`` derives the first-layer
    # fp32 recipe from the model's own fp32 ONNX export.
    python scripts/measure_new_backends.py \
        --model runs/d4/exported_models/exported_best.pt --image-size 640 \
        --calibration-images calib/ --out bench_640 --backends torch tensorrt-int8 \
        --quantization-exclude auto

The int8 exclusion recipe (``--quantization-exclude``) keeps the listed fp32
graph nodes in fp32; ``auto`` uses ``suggest_first_layer_nodes`` on a fresh
fp32 export (the leading Conv nodes — TensorRT has no int8 tactic for the
stem Conv+Relu+MaxPool fusion). Enumerate names yourself with
``onnx.load(fp32.onnx).graph.node``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import autovision  # noqa: E402
from autovision.tasks.task_model_helpers import load_task_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="exported .pt checkpoint")
    parser.add_argument("--out", required=True, help="benchmark output directory")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["torch", "onnx", "openvino"],
        choices=["torch", "onnx", "openvino", "tensorrt", "tensorrt-int8"],
    )
    parser.add_argument("--image-size", type=int, default=None,
                        help="square spatial size; for a checkpooint this must equal the "
                             "checkpoint's own spec size (the ONNX/TRT graphs are built at it)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--calibration-images", default=None,
                        help="image dir/file for the int8 calibration")
    parser.add_argument("--num-classes", type=int, default=10,
                        help="class count when --model is a benchmark spec string")
    parser.add_argument("--quantization-exclude", default="",
                        help="comma-separated fp32 node names to keep in fp32 (int8 recipe), "
                             "or 'auto' for the leading-Conv suggestion")
    args = parser.parse_args()

    if Path(args.model).exists():
        net = load_task_model(args.model, device="cpu")
    elif "/" in args.model and not args.model.startswith("/"):
        # Benchmark spec (e.g. "rtdetr/resnet18"): random weights, speed only.
        net = args.model
    else:
        print(
            f"--model '{args.model}' is neither an existing checkpoint nor a "
            "benchmark spec of the form '<family>/<backbone>' (e.g. rtdetr/resnet18)",
            file=sys.stderr,
        )
        return 2
    backends: dict = {}
    for name in args.backends:
        if name == "torch":
            backends["torch"] = {"precision": "fp32"}
        elif name == "onnx":
            backends["onnx"] = {"provider": "cpu"}
        elif name == "openvino":
            backends["openvino"] = {"ov_device": "CPU"}
        elif name == "tensorrt":
            backends["tensorrt"] = {"precision": "fp16"}
        else:  # tensorrt-int8
            if not args.calibration_images:
                print("tensorrt-int8 needs --calibration-images", file=sys.stderr)
                return 2
            specification: dict = {"precision": "int8"}
            if args.quantization_exclude:
                specifications = [
                    name.strip()
                    for name in args.quantization_exclude.split(",")
                    if name.strip()
                ]
                if args.quantization_exclude.strip() == "auto":
                    from autovision.export.benchmark import _resolve_benchmark_model
                    from autovision.export.quantization import suggest_first_layer_nodes

                    probe = net
                    if isinstance(probe, str):
                        probe = _resolve_benchmark_model(
                            probe,
                            num_classes=args.num_classes,
                            image_size=(args.image_size, args.image_size)
                            if args.image_size
                            else None,
                        )
                    else:
                        # Never deploy the object that gets benchmarked:
                        # benchmark() snapshots params/FLOPs BEFORE the
                        # backends deploy, and the probe export deploys.
                        probe = copy.deepcopy(probe)
                    probe_dir = Path(args.out)
                    probe_dir.mkdir(parents=True, exist_ok=True)
                    fp32_probe = probe.export_onnx(
                        probe_dir / "stem_probe_fp32.onnx", verify=False
                    )
                    try:
                        specifications = suggest_first_layer_nodes(fp32_probe)
                        print(f"auto first-layer fp32 recipe: {specifications}")
                    finally:
                        # The probe graph must not masquerade as a benchmark
                        # artifact (dynamo may add a 100MB-class .data file).
                        for candidate in (
                            fp32_probe,
                            fp32_probe.with_suffix(fp32_probe.suffix + ".data"),
                        ):
                            Path(candidate).unlink(missing_ok=True)
                if specifications:
                    specification["quantization"] = {
                        "nodes_to_exclude": specifications
                    }
            backends["tensorrt"] = specification
    if args.quantization_exclude.strip() == "auto" and "tensorrt-int8" not in args.backends:
        print("note: --quantization-exclude auto applies only to tensorrt-int8",
              file=sys.stderr)

    results = autovision.benchmark(
        net,
        out=args.out,
        num_classes=args.num_classes,
        backends=backends,
        verify_parity=True,
        calibration_data=args.calibration_images,
        batch_size=args.batch_size,
        runs=args.runs,
        warmup=args.warmup,
        image_size=(args.image_size, args.image_size) if args.image_size else None,
    )
    for key, result in results.items():
        print(json.dumps({
            "backend": key,
            "precision": result.extra.get("precision"),
            "p50_ms": round(result.latency_ms_p50, 3),
            "p90_ms": round(result.latency_ms_p90, 3),
            "images_per_sec": round(result.images_per_sec, 1),
        }))
    print(f"summary: {Path(args.out) / 'benchmark.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
