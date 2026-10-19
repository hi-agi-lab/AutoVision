# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Hard-example manifests: error-analysis output, weights, training hook."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from autovision import ensure_coco8, train_object_detection
from autovision.data.hard_examples import (
    HardExampleManifest,
    build_weighted_sampler,
)
from autovision.errors import AutoVisionError
from autovision.metrics.error_analysis import analyze_detection_errors


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), (120, 120, 120)).save(path)
    return path


def _manifest(tmp_path: Path, hard: Path, easy: Path) -> Path:
    path = tmp_path / "hard_examples.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_run": "test",
                "iou_threshold": 0.5,
                "score_threshold": 0.0,
                "images": [
                    {
                        "image": str(hard),
                        "split": "train",
                        "tp": 1,
                        "fp": 2,
                        "fn": 3,
                        "fp_reasons": {"background": 2},
                        "max_fn_area": 256.0,
                        "classes_fn": [0],
                    },
                    {
                        "image": str(easy),
                        "split": "train",
                        "tp": 3,
                        "fp": 0,
                        "fn": 0,
                    },
                ],
            }
        )
    )
    return path


def test_weights_and_sampler_are_deterministic(tmp_path):
    hard = tmp_path / "hard.png"
    easy = tmp_path / "easy.png"
    hard.write_bytes(b"x")
    easy.write_bytes(b"y")
    manifest = HardExampleManifest.load(_manifest(tmp_path, hard, easy))
    weights = manifest.weights_for([hard, easy, tmp_path / "absent.png"])
    assert weights[0] > weights[1] == 1.0
    assert weights[2] == 1.0

    first = list(build_weighted_sampler(manifest, [hard, easy], seed=0))
    second = list(build_weighted_sampler(manifest, [hard, easy], seed=0))
    assert first == second
    assert all(index in (0, 1) for index in first)


def test_split_filter_prevents_validation_leakage(tmp_path):
    val_image = tmp_path / "val.png"
    val_image.write_bytes(b"x")
    path = tmp_path / "val.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "images": [
                    {"image": str(val_image), "split": "val", "fp": 9, "fn": 9}
                ],
            }
        )
    )
    manifest = HardExampleManifest.load(path)
    assert manifest.weights_for([val_image], split="train").tolist() == [1.0]
    assert manifest.weights_for([val_image], split="val")[0] > 1.0


def test_invalid_manifest_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 2, "images": []}))
    with pytest.raises(AutoVisionError, match="schema_version"):
        HardExampleManifest.load(path)
    path.write_text("[]")
    with pytest.raises(AutoVisionError, match="JSON object"):
        HardExampleManifest.load(path)


def test_analyze_errors_writes_hard_examples(tmp_path):
    first = _image(tmp_path / "first.png")
    second = _image(tmp_path / "second.png")
    predictions = [
        {
            "bboxes": torch.tensor([[10, 10, 30, 30], [40, 5, 55, 20]]),
            "scores": torch.tensor([0.9, 0.4]),
            "labels": torch.tensor([0, 0]),
        },
        {"bboxes": torch.empty(0, 4), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)},
    ]
    targets = [
        {
            "boxes": torch.tensor([[10, 10, 30, 30], [40, 30, 55, 45]]),
            "labels": torch.tensor([0, 0]),
        },
        {"boxes": torch.tensor([[10, 10, 30, 30]]), "labels": torch.tensor([0])},
    ]
    report = analyze_detection_errors(
        predictions,
        targets,
        {0: "cat"},
        [first, second],
        tmp_path / "errors",
        iou_threshold=0.5,
        split="train",
    )
    assert report.hard_examples_json is not None
    payload = json.loads(report.hard_examples_json.read_text())
    assert payload["schema_version"] == 1
    assert payload["split"] == "train"
    assert {entry["image"] for entry in payload["images"]} == {str(first), str(second)}
    assert report.to_dict()["hard_examples_json"] == str(report.hard_examples_json)


def test_training_entry_accepts_hard_examples(tmp_path):
    from autovision.cli.train_task.object_detection import (
        _DATA_FORMATS,
        _load_data_config,
    )
    from autovision.config.validate import pydantic_model_validate

    data = ensure_coco8()
    payload = _load_data_config(data)
    data_args = pydantic_model_validate(_DATA_FORMATS[payload["format"]], payload)
    data_args.resolve_data_paths(None)
    rows = list(data_args.list_image_info("train"))
    manifest_path = tmp_path / "hard.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "images": [
                    {
                        "image": str(Path(rows[0]["image_path"]).resolve()),
                        "split": "train",
                        "tp": 0,
                        "fp": 2,
                        "fn": 3,
                    }
                ],
            }
        )
    )
    result = train_object_detection(
        out=tmp_path / "run",
        data=data,
        model="rtdetr/resnet18",
        steps=1,
        batch_size=1,
        num_workers=0,
        devices=1,
        precision="32-true",
        hard_examples=manifest_path,
    )
    assert result.exported_model_path.exists()
