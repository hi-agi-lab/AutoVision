# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Hardware-pack template: conversion prep + target parity harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TEMPLATE = ROOT / "deploy" / "packs" / "_template"


def test_convert_prepares_pack_and_runs_vendor_command(tmp_path):
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    out = tmp_path / "pack"
    result = subprocess.run(
        [
            sys.executable, str(TEMPLATE / "convert.py"),
            "--onnx", str(onnx),
            "--postprocess", str(TEMPLATE / "postprocess.json"),
            "--out", str(out),
            "--vendor", "demo",
            "--vendor-command", "touch vendor-ran.txt",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "postprocess.json").is_file()
    assert (out / "pack.json").is_file()
    assert (out / "vendor-ran.txt").is_file()
    payload = json.loads((out / "pack.json").read_text())
    assert payload["vendor"] == "demo"
    assert payload["family"] == "detr"


def test_convert_rejects_bad_inputs(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(TEMPLATE / "convert.py"),
            "--onnx", str(tmp_path / "missing.onnx"),
            "--postprocess", str(TEMPLATE / "postprocess.json"),
            "--out", str(tmp_path / "out"), "--vendor", "x",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0


def test_parity_harness(tmp_path):
    record = {
        "predictions": {
            "bboxes": [[1.0, 2.0, 10.0, 12.0]],
            "scores": [0.9],
            "labels": [0],
        }
    }
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    reference.write_text(json.dumps(record) + "\n")
    candidate.write_text(json.dumps(record) + "\n")
    result = subprocess.run(
        [
            sys.executable, str(TEMPLATE / "parity.py"),
            "--reference", str(reference), "--candidate", str(candidate),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "parity ok" in result.stdout

    shifted = json.loads(json.dumps(record))
    shifted["predictions"]["bboxes"] = [[6.0, 2.0, 10.0, 12.0]]
    candidate.write_text(json.dumps(shifted) + "\n")
    result = subprocess.run(
        [
            sys.executable, str(TEMPLATE / "parity.py"),
            "--reference", str(reference), "--candidate", str(candidate),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_template_files_exist():
    assert (ROOT / "deploy" / "packs" / "README.md").is_file()
    assert (TEMPLATE / "postprocess.json").is_file()
    assert (TEMPLATE / "README.md").is_file()
