#!/usr/bin/env python3
"""Reproduce the spectrum cost/latency numbers in docs/benchmarks.md §2 / §2.1.

Cost (parameters + FLOPs) and latency for the P1 spectrum models
(``detr/n``, ``detr/s``, ``detr/gtrvit-s``) plus the
``rtdetr/resnet18`` reference. Model weights are NOT required: cost and
latency are architecture-level (random init), matching the §2 table's
"装配变体" rows.

Protocols (as recorded in the table):

- cost        : 640x640 spec input, batch 1, **before** ``deploy()`` — the
                published convention counts the trained checkpoint's
                parameters. FLOPs follow the torch flop-counter convention
                (multiply-add = 2 FLOPs), the convention the GTR paper's
                table uses (cross-check: ``detr/gtrvit-s`` reads 33.8
                GFLOPs, the official GTR-S number).
- torch       : eager ``forward_backend`` of the **undeployed** model,
                p50 over 50 iters, warmup 20 (§2 protocol).
- tensorrt    : fp16 engine over the batch profile 1..4; two readings —
                ``engine`` (engine-only forward) and ``pipeline``
                (resize+normalize preprocessing -> H2D -> engine, no
                postprocess) — p50 over ``--steps`` iters, warmup
                ``--warmup`` (§2.1 protocol).

Usage::

    python scripts/measure_spectrum_benchmarks.py                     # cost + torch latency
    python scripts/measure_spectrum_benchmarks.py --backends torch tensorrt
    python scripts/measure_spectrum_benchmarks.py --what cost --models detr/n rtdetr/resnet18
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from autovision.errors import AutoVisionError  # noqa: E402
from autovision.export import measure_model_cost  # noqa: E402

IMAGE_SIZE = (640, 640)
NUM_CLASSES = 80


def _detr(name: str, device: str):
    from autovision.tasks.detection.detr.task_model import DETRObjectDetection

    return DETRObjectDetection(
        num_classes=NUM_CLASSES,
        model_name=name,
        image_size=IMAGE_SIZE,
        decoder="dfine_official",
        load_weights=False,
        backbone_pretrained=False,
    ).to(device)


def _builders() -> dict[str, tuple[str, Callable[[str], Any]]]:
    def resnet18(device: str):
        from autovision.tasks import RTDETRObjectDetection

        return RTDETRObjectDetection(
            num_classes=NUM_CLASSES,
            backbone="torchvision/resnet18",
            image_size=IMAGE_SIZE,
        ).to(device)

    return {
        "detr/n": ("cuda", lambda device: _detr("n", device)),
        "detr/s": ("cuda", lambda device: _detr("s", device)),
        # GLA backbone: forward needs CUDA (Triton); export is blocked (G2).
        "detr/gtrvit-s": ("cuda", lambda device: _detr("gtrvit-s", device)),
        "rtdetr/resnet18": ("cuda", resnet18),
    }


def _p50(call: Callable[[], Any], *, warmup: int, steps: int) -> float:
    on_cuda = torch.cuda.is_available()
    with torch.no_grad():
        for _ in range(warmup):
            call()
        if on_cuda:
            torch.cuda.synchronize()
        latencies = []
        for _ in range(steps):
            start = time.perf_counter()
            call()
            if on_cuda:
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)
    latencies.sort()
    return latencies[len(latencies) // 2]


def _source_image() -> torch.Tensor:
    """Deterministic 1280x720 RGB frame (uint8 CHW) stand-in for a decode."""
    import numpy as np

    frame = (np.random.default_rng(0).random((720, 1280, 3)) * 255).astype("uint8")
    return torch.from_numpy(frame).permute(2, 0, 1)


def run_cost(models: dict, selected: list[str], batch: int) -> None:
    print("## cost (params pre-deploy, FLOPs = 2 x MACs, torch flop counter)")
    print("| model | params (M) | GFLOPs | MACs (G) | input |")
    print("|---|---|---|---|---|")
    for label in selected:
        default_device, build = models[label]
        model = build(default_device).eval()
        cost = measure_model_cost(model, batch_size=batch, device=default_device)
        print(
            f"| {label} | {cost.parameters / 1e6:.2f} | {cost.flops / 1e9:.1f} "
            f"| {cost.macs / 1e9:.1f} | {tuple(cost.input_shape)} |"
        )


def run_latency(
    models: dict,
    selected: list[str],
    backends: list[str],
    batches: list[int],
    torch_warmup: int,
    torch_steps: int,
    trt_warmup: int,
    trt_steps: int,
    engine_dir: Path | None,
) -> None:
    for label in selected:
        default_device, build = models[label]
        device = default_device
        # torch: undeployed eager forward (§2 protocol).
        if "torch" in backends:
            model = build(device).eval()
            for batch in batches:
                tensor = torch.randn(batch, 3, *IMAGE_SIZE, device=device)

                def forward(module=model, x=tensor):
                    return module.forward_backend(x)

                latency = _p50(forward, warmup=torch_warmup, steps=torch_steps)
                print(
                    f"torch\t{label}\tfp32\tb{batch}\tp50={latency:.1f}ms", flush=True
                )
        if "tensorrt" not in backends:
            continue
        try:
            from autovision.export import TensorRTEngineRunner

            workdir = engine_dir or Path(tempfile.mkdtemp())
            workdir.mkdir(parents=True, exist_ok=True)
            engine_path = workdir / f"{label.replace('/', '_')}.fp16.engine"
            model = build(device).eval()  # preprocessing source
            if not engine_path.exists():
                print(f"# building TensorRT engine: {engine_path.name}", flush=True)
                model.export_tensorrt(
                    engine_path,
                    precision="fp16",
                    min_batch_size=min(batches),
                    opt_batch_size=max(batches),
                    max_batch_size=max(batches),
                )
            runner = TensorRTEngineRunner(engine_path, device=device)
            source = _source_image()
            for batch in batches:
                tensor = torch.randn(batch, 3, *IMAGE_SIZE, device=device)

                def engine_only(x=tensor):
                    return runner.run(x)

                def pipeline(b=batch):
                    image, _ = model.preprocess_image(source)
                    stacked = model.preprocess_batch(torch.stack([image] * b))
                    return runner.run(stacked)

                engine_latency = _p50(
                    engine_only, warmup=trt_warmup, steps=trt_steps
                )
                pipeline_latency = _p50(
                    pipeline, warmup=trt_warmup, steps=trt_steps
                )
                print(
                    f"tensorrt-fp16\t{label}\tb{batch}\tengine={engine_latency:.2f}ms "
                    f"pipeline={pipeline_latency:.2f}ms",
                    flush=True,
                )
        except AutoVisionError as error:
            print(f"tensorrt-fp16\t{label}\tBLOCKED: {error}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", default="cost,latency", help="cost,latency")
    parser.add_argument("--backends", nargs="+", default=["torch"])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--torch-warmup", type=int, default=20)
    parser.add_argument("--torch-steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--engine-dir", type=Path, default=None)
    args = parser.parse_args()

    models = _builders()
    selected = args.models or list(models)
    unknown = [label for label in selected if label not in models]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}; known: {list(models)}")
    what = set(args.what.split(","))
    if "cost" in what:
        run_cost(models, selected, batch=1)
    if "latency" in what:
        run_latency(
            models,
            selected,
            backends=args.backends,
            batches=args.batch,
            torch_warmup=args.torch_warmup,
            torch_steps=args.torch_steps,
            trt_warmup=args.warmup,
            trt_steps=args.steps,
            engine_dir=args.engine_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
