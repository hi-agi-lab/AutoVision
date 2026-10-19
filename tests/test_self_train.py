# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""End-to-end self-training gate: the script's loop produces an evaluable
model (roadmap C1; the P1 exit criterion).

Runs the REAL ``scripts/self_train_detect.py``: teacher training ->
pseudo-labeling pass -> student training on the pseudo labels -> val read.
The teacher is trained once per module and reused across walks.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from tests.conftest import create_image

import autovision
from autovision import train_object_detection
from autovision.tasks.detection.shared.pseudo_label import generate_pseudo_labels
from autovision.tasks.task_model_helpers import load_task_model

ROOT = Path(__file__).parent.parent
CATEGORIES = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
MODEL_ARGS = {
    "backbone": "torchvision/resnet18",
    "image_size": (64, 64),
    "hidden_dim": 32,
    "num_queries": 10,
    "dec_layers": 2,
    "num_heads": 4,
    "num_top_queries": 10,
}
TRANSFORM_ARGS = {
    "scale_jitter": None,
    "photometric_distort_prob": 0.5,
    "random_zoom_out_prob": None,
    "random_iou_crop_prob": None,
}


def _write_coco(tmp_path: Path) -> tuple[Path, Path, Path]:
    images_dir = tmp_path / "images"
    rows, annotations = [], []
    for index in range(4):
        path = create_image(images_dir / f"img_{index:03d}.jpg", size=(64, 64), seed=index)
        rows.append({"id": index + 1, "file_name": path.name, "width": 64, "height": 64})
        annotations.append(
            {
                "id": index + 1,
                "image_id": index + 1,
                "category_id": 1 + (index % 2),
                "bbox": [8.0, 8.0, 32.0, 32.0],
                "area": 1024.0,
                "iscrowd": 0,
            }
        )
    payload = {"images": rows, "annotations": annotations, "categories": CATEGORIES}
    train_json = tmp_path / "train.json"
    train_json.write_text(json.dumps(payload))
    val_json = tmp_path / "val.json"
    val_json.write_text(json.dumps(payload))
    return train_json, val_json, images_dir


def _data_config(train_json: Path, val_json: Path, images_dir: Path) -> dict:
    return {
        "format": "coco",
        "train": {"annotations": str(train_json), "images": str(images_dir)},
        "val": {"annotations": str(val_json), "images": str(images_dir)},
    }


@pytest.fixture(scope="module")
def labeled(tmp_path_factory):
    return _write_coco(tmp_path_factory.mktemp("labeled"))


@pytest.fixture(scope="module")
def teacher(labeled, tmp_path_factory):
    train_json, val_json, images_dir = labeled
    result = train_object_detection(
        out=tmp_path_factory.mktemp("teacher_run") / "run",
        data=_data_config(train_json, val_json, images_dir),
        steps=4,
        batch_size=2,
        devices="cpu",
        num_workers=0,
        model_args=MODEL_ARGS,
        transform_args=TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": 4},
    )
    return result.best_exported_model_path


class TestSelfTrainingLoop:
    def test_library_loop_produces_evaluable_model(self, labeled, teacher, tmp_path):
        """teacher -> pseudo labels -> student -> val metrics (C1 plumbing)."""
        train_json, val_json, images_dir = labeled
        network = load_task_model(teacher, device="cpu")
        # threshold 0.0 keeps every predicted box: the loop is exercised
        # regardless of how good the 4-step teacher is.
        pseudo = generate_pseudo_labels(
            network,
            images_dir,
            tmp_path / "pseudo.json",
            threshold=0.0,
            categories={1: "cat", 2: "dog"},
            batch_size=2,
        )
        assert pseudo.annotations > 0
        pseudo_spec = {
            "format": "coco",
            "train": {"annotations": str(pseudo.out_json), "images": str(images_dir)},
            "val": {"annotations": str(val_json), "images": str(images_dir)},
        }
        result = train_object_detection(
            out=tmp_path / "student",
            data=pseudo_spec,
            steps=2,
            batch_size=2,
            devices="cpu",
            num_workers=0,
            model_args=MODEL_ARGS,
            transform_args=TRANSFORM_ARGS,
            trainer_args={"val_every_num_steps": 2},
        )
        student = autovision.load_model(result.best_exported_model_path, device="cpu")
        metrics = student.val(
            data=_data_config(train_json, val_json, images_dir), split="val"
        )
        assert "map" in metrics and "map_50" in metrics
        assert all(isinstance(value, float) for value in metrics.values())

    def test_script_end_to_end(self, labeled, teacher, tmp_path):
        """The gate artifact itself: the script's report carries the read."""
        train_json, val_json, images_dir = labeled
        data_path = tmp_path / "data.json"
        data_path.write_text(json.dumps(_data_config(train_json, val_json, images_dir)))
        out_dir = tmp_path / "round"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "self_train_detect.py"),
                "--teacher", str(teacher),
                "--images", str(images_dir),
                "--val-data", str(data_path),
                "--out", str(out_dir),
                "--student", "rtdetr/resnet18",
                "--steps", "2",
                "--batch-size", "2",
                "--devices", "cpu",
                "--threshold", "0.0",
                "--pseudo-batch-size", "2",
                "--model-args", json.dumps(MODEL_ARGS),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads((out_dir / "report.json").read_text())
        assert report["pseudo"]["annotations"] > 0
        assert report["val_metrics"]["map"] == report["val_metrics"]["map"]  # not NaN
        assert Path(report["exported_model"]).exists()
        assert (out_dir / "pseudo" / "train.json").exists()
        assert "[model] 可评测模型:" in result.stdout

    def test_script_multi_round_summary(self, labeled, teacher, tmp_path):
        """Two rounds: per-round artifacts, promotion, summary + stop reason."""
        train_json, val_json, images_dir = labeled
        data_path = tmp_path / "data.json"
        data_path.write_text(json.dumps(_data_config(train_json, val_json, images_dir)))
        out_dir = tmp_path / "multi"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "self_train_detect.py"),
                "--teacher", str(teacher),
                "--images", str(images_dir),
                "--val-data", str(data_path),
                "--out", str(out_dir),
                "--student", "rtdetr/resnet18",
                "--steps", "1",
                "--batch-size", "2",
                "--devices", "cpu",
                "--threshold", "0.0",
                "--pseudo-batch-size", "2",
                "--rounds", "2",
                "--model-args", json.dumps(MODEL_ARGS),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads((out_dir / "summary.json").read_text())
        assert len(summary["rounds"]) == 2
        assert summary["stop_reason"] in {"completed", "no_improvement"}
        assert summary["best_model"] and Path(summary["best_model"]).exists()
        for index in range(2):
            round_dir = out_dir / "rounds" / f"r{index}"
            assert (round_dir / "report.json").is_file()
            assert (round_dir / "pseudo" / "train.json").is_file()
        assert (out_dir / "summary.md").is_file()
        assert "[promote] next teacher:" in result.stdout
