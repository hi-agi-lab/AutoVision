# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""int8 vs fp32 report: verdict, markdown, script wiring."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from autovision.errors import AutoVisionError
from autovision.export.int8_report import summarize_int8_accuracy

ROOT = Path(__file__).parent.parent


def test_summary_pass_and_fail():
    fp32 = {"map": 50.0, "map_50": 70.0, "map_75": 55.0}
    int8 = {"map": 49.2, "map_50": 69.5, "map_75": 54.0}
    report = summarize_int8_accuracy(
        fp32, int8, max_ap_drop=1.0, model="best.pt", backend="onnx"
    )
    assert report.passed
    assert report.deltas["map"] == pytest.approx(-0.8)
    assert "PASS" in report.summary_text()

    failed = summarize_int8_accuracy(fp32, {"map": 48.0}, max_ap_drop=1.0)
    assert not failed.passed
    assert failed.deltas["map"] == pytest.approx(-2.0)


def test_report_save_and_markdown(tmp_path):
    report = summarize_int8_accuracy(
        {"map": 50.0, "map_50": 70.0},
        {"map": 49.5, "map_50": 69.0},
        max_ap_drop=1.0,
        hardware={"gpu": "test"},
    ).save(tmp_path / "int8")
    assert report.report_json.is_file()
    assert report.report_md.is_file()
    payload = json.loads(report.report_json.read_text())
    assert payload["schema_version"] == 1
    assert payload["passed"] is True
    assert "| map | 50.0000 | 49.5000 | -0.5000 |" in report.report_md.read_text()
    assert "gpu: test" in report.report_md.read_text()


def test_missing_map_is_rejected():
    with pytest.raises(AutoVisionError, match="'map'"):
        summarize_int8_accuracy({"map_50": 1.0}, {"map": 1.0})


def test_script_help_lists_inputs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "measure_int8_accuracy.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "--calibration-images" in result.stdout
    assert "--max-ap-drop" in result.stdout


def test_evaluate_engine_with_fake_engine(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "av_measure_int8", ROOT / "scripts" / "measure_int8_accuracy.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    image = tmp_path / "a.png"
    Image.new("RGB", (64, 48), (120, 80, 40)).save(image)

    class DataArgs:
        def classes(self):
            return {0: "cat"}

        def list_image_info(self, split):
            assert split == "val"
            yield {
                "image_path": str(image),
                "bboxes": json.dumps([[20 / 64, 20 / 48, 20 / 64, 20 / 48]]),
                "class_labels": json.dumps([0]),
            }

    class Engine:
        def run_batch(self, paths, *, threshold=0.5):
            assert len(paths) == 1
            return [
                {
                    "bboxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                }
            ]

    metrics, images = module._evaluate_engine(
        Engine(), DataArgs(), split="val", max_images=None,
        threshold=0.0, batch_size=2,
    )
    assert images == 1
    assert "map" in metrics and "map_50" in metrics
