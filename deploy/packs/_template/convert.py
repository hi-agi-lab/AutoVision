#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Hardware-pack conversion template.

Validates the ONNX + postprocess inputs, copies the decode contract into the
pack, writes ``pack.json``, and (optionally) runs the vendor SDK command.
The vendor SDK is supplied by the integrator; nothing is vendored here.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", required=True, help="exported ONNX file")
    parser.add_argument(
        "--postprocess", required=True, help="<artifact>.postprocess.json"
    )
    parser.add_argument("--out", required=True, help="pack output directory")
    parser.add_argument("--vendor", required=True, help="vendor / chip name")
    parser.add_argument(
        "--vendor-command", default=None,
        help="command executed with cwd=<out> (vendor SDK conversion)",
    )
    args = parser.parse_args()

    onnx = Path(args.onnx)
    spec_path = Path(args.postprocess)
    if not onnx.is_file() or onnx.stat().st_size == 0:
        parser.error(f"ONNX file missing or empty: {onnx}")
    try:
        spec = json.loads(spec_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot read postprocess spec: {error}")
    if spec.get("schema_version") != 1:
        parser.error("postprocess schema_version must be 1")
    if spec.get("family") not in {"detr", "picodet"}:
        parser.error("postprocess family must be detr or picodet")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec_path, out / "postprocess.json")
    pack = {
        "schema_version": 1,
        "vendor": args.vendor,
        "onnx": str(onnx),
        "family": spec.get("family"),
        "input_size": spec.get("input_size"),
        "nms": spec.get("nms"),
        "vendor_command": args.vendor_command,
    }
    (out / "pack.json").write_text(json.dumps(pack, indent=2, sort_keys=True))
    if args.vendor_command:
        subprocess.run(shlex.split(args.vendor_command), cwd=out, check=True)
        print(f"vendor command completed in {out}")
    else:
        print(f"pack prepared in {out}; run the vendor SDK conversion manually")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
