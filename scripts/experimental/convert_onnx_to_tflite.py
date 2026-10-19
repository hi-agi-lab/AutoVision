#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""ONNX -> TFLite (LiteRT) conversion for the edge deployment slot (D3).

STATUS (2026-10-16): the TFLite/LiteRT conversion is the chosen edge
direction, but it is **not verified for this framework's detection graphs**
and it is not part of the first-class backends (the D5 policy: non-primary
backends go through the ONNX hub plus conversion scripts/docs). Two
recorded blockers:

1. The LiteRT conversion toolchain cannot coexist with the framework
   environment: the converter stack pins ``typing-extensions<4.13``
   (``xdsl``, pulled in by ``ai-edge-litert``/``litert-converter``) while
   the framework requires >= 4.13 — installing it breaks ``import
   autovision`` with ``ImportError: cannot import name 'Sentinel'``
   (pydantic_core imports it). Run this script with an ISOLATED
   environment (see ``--onnx2tf`` below).
2. ``onnx2tf`` (the ONNX-route converter) fails on this family's graph:
   ``ERROR: node_upsample_nearest2d`` (the FPN's nearest-neighbour Resize)
   — a documented onnx2tf parameter-replacement case (its suggestions:
   static shapes via ``-b``/``-ois``, ``-kat``, or op replacements). The
   ``saved_model`` output is empty after the failure.

Usage (isolated env)::

    uv venv /tmp/edge_env --python 3.10
    uv pip install --python /tmp/edge_env/bin/python onnx2tf ai-edge-litert
    python scripts/experimental/convert_onnx_to_tflite.py model.onnx --onnx2tf \
        /tmp/edge_env/bin/python --out-dir tflite/

Parity on the target: run the produced ``.tflite`` through
``ai_edge_litert.interpreter.Interpreter`` and compare against ONNX Runtime
on the same input (the same torch-vs-backend comparison the in-repo export
gates use) — a device-side gate belongs to the target platform bring-up.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("onnx", help="the framework's exported .onnx (fp32)")
    parser.add_argument("--out-dir", default="tflite", help="output directory")
    parser.add_argument(
        "--onnx2tf",
        default=None,
        help="python interpreter that has onnx2tf (isolated env); default: PATH",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"ONNX not found: '{onnx_path}'", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.onnx2tf:
        command = [args.onnx2tf, "-m", "onnx2tf", "-i", str(onnx_path), "-o", str(out_dir / "saved_model")]
    else:
        executable = shutil.which("onnx2tf")
        if executable is None:
            print(
                "onnx2tf not found. Install it in an ISOLATED environment "
                "(the LiteRT converter stack pins typing-extensions<4.13 via "
                "xdsl, which conflicts with the framework's >=4.13) and pass "
                "--onnx2tf <python>.",
                file=sys.stderr,
            )
            return 2
        command = [executable, "-i", str(onnx_path), "-o", str(out_dir / "saved_model")]

    print("Running:", " ".join(command))
    result = subprocess.run(command)
    produced = sorted(out_dir.rglob("*.tflite"))
    if result.returncode != 0 or not produced:
        print(
            "\nConversion did not produce a .tflite. Known blocker for the DETR "
            "family: onnx2tf fails on the FPN's nearest Resize "
            "('node_upsample_nearest2d'); try '-b 1'/'--overwrite_input_shape' "
            "for static shapes or the documented parameter replacements, and "
            "record the outcome — the edge slot ships as this script plus the "
            "conversion notes until the target platform's parity gate exists.",
            file=sys.stderr,
        )
        return 1
    print("Produced:", *[str(path) for path in produced], sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
