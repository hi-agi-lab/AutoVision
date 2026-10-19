# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Spectrum training/evaluation script skeletons.

The scripts are the reproducible entry points a released tier's model card
points at. These tests exercise the resolved config (``--dry-run``) and the
planned-tier refusal; actual training stays in the API-level closure tests.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAIN = ROOT / "scripts" / "train_spectrum.py"
EVAL = ROOT / "scripts" / "eval_spectrum_coco.py"


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


class TestTrainSpectrumScript:
    def test_spec_frozen_tier_dry_run(self):
        result = _run(TRAIN, "--tier", "n", "--dry-run", "--out", "runs/n")
        assert result.returncode == 0, result.stderr
        config = json.loads(result.stdout)
        assert config["model"] == "detr/n"
        assert config["status"] == "spec-frozen"
        assert config["steps"] == 2000
        assert config["model_args"]["image_size"] == [640, 640]

    def test_teacher_wires_distillation(self):
        result = _run(
            TRAIN,
            "--tier", "n",
            "--teacher", "rfdetr/rf-detr-nano",
            "--dry-run",
            "--out", "runs/n",
        )
        assert result.returncode == 0, result.stderr
        config = json.loads(result.stdout)
        assert config["model_args"]["distill"] == {
            "teacher_model": "rfdetr/rf-detr-nano"
        }

    def test_warm_start_is_forwarded(self):
        result = _run(
            TRAIN,
            "--tier", "s",
            "--warm-start", "ws_detr_s.pt",
            "--dry-run",
            "--out", "runs/s",
        )
        assert result.returncode == 0, result.stderr
        config = json.loads(result.stdout)
        assert config["model_args"]["warm_start"] == "ws_detr_s.pt"

    def test_planned_tier_is_refused(self):
        result = _run(TRAIN, "--tier", "m", "--dry-run", "--out", "runs/m")
        assert result.returncode == 2
        assert "planned" in result.stderr

    def test_smoke_config_uses_coco8(self):
        result = _run(
            TRAIN, "--tier", "n", "--smoke", "--dry-run", "--out", "runs/n-smoke"
        )
        assert result.returncode == 0, result.stderr
        config = json.loads(result.stdout)
        assert config["steps"] == 3
        assert config["batch_size"] == 2
        assert config["model_args"]["image_size"] == [64, 64]
        assert config["trainer_args"] == {"val_every_num_steps": 2}


class TestEvalSpectrumScript:
    def test_dry_run_does_not_require_weights(self):
        result = _run(
            EVAL,
            "--weights", "runs/n/exported_models/exported_best.pt",
            "--data", "coco8",
            "--dry-run",
        )
        assert result.returncode == 0, result.stderr
        config = json.loads(result.stdout)
        assert config["split"] == "val"
        assert config["weights"].endswith("exported_best.pt")

    def test_help_lists_the_public_entry_points(self):
        result = _run(EVAL, "--help")
        assert result.returncode == 0
        assert "exported task model" in result.stdout
