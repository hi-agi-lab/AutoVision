#!/usr/bin/env python
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Verify a serialized TensorRT engine against the eager task model.

Run this on the target GPU host (TensorRT + CUDA required) after exporting:

    python scripts/verify_tensorrt_engine.py \
        --model runs/det/exported_models/exported_best.pt \
        --engine runs/det/model.engine \
        --image samples/frame.jpg \
        --atol 5e-3 --rtol 1e-2

The check compares the engine's final tensors with ``forward_backend`` on a
seeded random input of the exported shape. Exit code 1 means the engine is
not safe to publish.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from autovision.export.tensorrt_helpers import verify_engine_outputs
from autovision.tasks.task_model_helpers import load_task_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="exported task model .pt")
    parser.add_argument("--engine", required=True, help="serialized TensorRT engine")
    parser.add_argument(
        "--image",
        help="optional input image; default: seeded random at the exported shape",
    )
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is required for TensorRT verification.", file=sys.stderr)
        return 2
    if not Path(args.engine).is_file():
        print(f"Engine not found: {args.engine}", file=sys.stderr)
        return 2

    model = load_task_model(args.model, device=args.device).eval()
    if args.image:
        tensor, _ = model.preprocess_image(args.image)
        tensor = model.preprocess_batch(tensor).unsqueeze(0)
    else:
        spec = model.model_input_spec
        sample = next(
            iter(spec.example_inputs(batch_size=1, device="cpu").values())
        )
        generator = torch.Generator().manual_seed(0)
        tensor = torch.rand(
            sample.shape, generator=generator, dtype=sample.dtype
        )
    tensor = tensor.to(args.device)
    with torch.no_grad():
        expected = model.forward_backend(tensor)
    try:
        verify_engine_outputs(
            args.engine,
            expected,
            tensor.cpu(),
            device=args.device,
            atol=args.atol,
            rtol=args.rtol,
        )
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    print(
        f"OK: {args.engine} matches {args.model} "
        f"(atol={args.atol}, rtol={args.rtol})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
