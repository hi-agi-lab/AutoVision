# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Command-level end-to-end smoke for the SSL branch (design doc 14 §7).

Runs the real ``train_object_detection`` assembly (two-view dataset wrap,
collate wiring, step hooks, loggers, checkpoint/export) against a tiny COCO
dataset — the layer the unit tests deliberately do not cover.
"""

import json
from pathlib import Path

import pytest
import torch
from tests.test_train_object_detection import data_config, make_coco_dataset

from autovision import load_task_model as load_model
from autovision import train_object_detection

MODEL = "detr/dinov2-vittest-notpretrained"

DETR_ARGS = {
    "image_size": (224, 224),
    "num_queries": 10,
    "num_top_queries": 10,
    "decoder": "dfine",
    "ema_warmup_steps": 0,
}

TRANSFORM_ARGS = {"scale_jitter": None, "mixup_prob": None}


def _read_metric_keys(out_dir: Path) -> set[str]:
    path = Path(out_dir) / "metrics.jsonl"
    keys: set[str] = set()
    for line in path.read_text().splitlines():
        keys.update(json.loads(line).keys())
    return keys


@pytest.fixture(scope="module")
def coco(tmp_path_factory):
    return make_coco_dataset(tmp_path_factory.mktemp("ssl_coco"), num_images=8)


def test_joint_ssl_two_crop_end_to_end(tmp_path, coco):
    train_json, val_json, images_dir = coco
    result = train_object_detection(
        out=tmp_path / "joint",
        data=data_config(train_json, val_json, images_dir),
        model=MODEL,
        steps=3,
        batch_size=4,
        num_workers=0,
        model_args={
            **DETR_ARGS,
            "ssl": {"mode": "unsup", "view": "two_crop", "loss_coef": 0.1},
        },
        transform_args=TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": 2},
        seed=0,
    )
    assert result.steps_run == 3
    keys = _read_metric_keys(tmp_path / "joint")
    assert {"loss_ssl", "ssl_pool_size", "ssl_paired_ratio"} <= keys
    assert "val_metric/map" in result.best_metrics
    assert result.checkpoint_path.exists() and result.exported_model_path.exists()
    # Export round trip: ssl rides the init-args snapshot (as a plain dict).
    model = load_model(result.exported_model_path, device="cpu")
    prediction = model.predict(
        torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8), threshold=0.0
    )
    assert set(prediction) == {"labels", "bboxes", "scores"}


def test_batch_matched_baseline_has_no_ssl_keys(tmp_path, coco):
    train_json, val_json, images_dir = coco
    result = train_object_detection(
        out=tmp_path / "baseline",
        data=data_config(train_json, val_json, images_dir),
        model=MODEL,
        steps=2,
        batch_size=4,
        num_workers=0,
        model_args={**DETR_ARGS, "ssl": {"mode": "off", "view": "two_crop"}},
        transform_args=TRANSFORM_ARGS,
        seed=0,
    )
    assert result.steps_run == 2
    assert "loss_ssl" not in _read_metric_keys(tmp_path / "baseline")


def test_route_b_ssl_only_end_to_end(tmp_path, coco):
    train_json, val_json, images_dir = coco
    result = train_object_detection(
        out=tmp_path / "routeb",
        data=data_config(train_json, val_json, images_dir),
        model=MODEL,
        steps=3,
        batch_size=4,
        num_workers=0,
        model_args={
            **DETR_ARGS,
            "ssl": {
                "mode": "unsup",
                "view": "dropout",
                "loss_coef": 1.0,
                "instance_source": "encoder_candidates",
            },
        },
        transform_args=TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": 2},
        seed=0,
    )
    assert result.steps_run == 3
    keys = _read_metric_keys(tmp_path / "routeb")
    assert "loss_ssl" in keys
    assert "loss_vfl" not in keys  # detection losses are skipped entirely
    assert "val_loss" in result.best_metrics  # stage-1 selection signal
