# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Versioned postprocess spec: NumPy reference decode + export sidecar."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from autovision.errors import AutoVisionError
from autovision.export.postprocess import (
    PostprocessSpec,
    build_postprocess_spec,
    decode_detections_numpy,
    load_postprocess_spec,
    write_postprocess_spec,
)
from autovision.tasks import RTDETRObjectDetection
from autovision.tasks.detection.detr.task_model import DETRObjectDetection
from autovision.tasks.detection.picodet.task_model import PicoDetObjectDetection
from autovision.tasks.detection.rfdetr.task_model import RFDETRObjectDetection

_export_installed = (
    importlib.util.find_spec("onnxruntime") is not None
    and importlib.util.find_spec("onnxslim") is not None
    and importlib.util.find_spec("onnx") is not None
)


def _onnx_stack_importable() -> bool:
    """A package can be installed but unusable (numpy/onnx ABI drift)."""
    try:
        import onnx  # noqa: F401
    except Exception:  # noqa: BLE001 - environment probe, not a test failure
        return False
    return True


requires_export = pytest.mark.skipif(
    not (_export_installed and _onnx_stack_importable()),
    reason="onnx stack not importable (pip install -e '.[export]' in a compatible env)",
)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    model = RTDETRObjectDetection(
        num_classes=2, backbone="torchvision/resnet18",
        image_size=(64, 64), hidden_dim=32, num_queries=10,
        dec_layers=1, num_heads=4, num_top_queries=10,
    )
    model.eval()
    return model


@pytest.fixture()
def raw_outputs():
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(2, 10, 3, generator=generator)
    boxes = 0.25 + 0.5 * torch.rand(2, 10, 4, generator=generator)
    metadata = [{"orig_w": 100, "orig_h": 80}, {"orig_w": 64, "orig_h": 64}]
    return logits, boxes, metadata


def test_detr_task_models_declare_the_sidecar_family():
    for cls in (RTDETRObjectDetection, DETRObjectDetection, RFDETRObjectDetection):
        assert cls.postprocess_spec_family == "detr"


def test_spec_fields_and_json_round_trip(tiny_model, tmp_path):
    spec = build_postprocess_spec(tiny_model, score_threshold=0.2)
    assert spec.schema_version == 1
    assert spec.family == "detr"
    assert spec.input_size == (64, 64)
    assert spec.num_classes == 2
    assert spec.num_queries == 10
    assert spec.top_k == 10
    assert spec.score_threshold == 0.2
    assert spec.preprocess is not None
    assert len(spec.preprocess.mean) == 3

    path = spec.save(tmp_path / "spec.json")
    loaded = load_postprocess_spec(path)
    assert loaded == spec
    assert loaded.to_dict()["preprocess"]["resize"] == "resize"


def test_reference_decode_matches_torch_postprocess(tiny_model, raw_outputs):
    logits, boxes, metadata = raw_outputs
    spec = build_postprocess_spec(tiny_model, score_threshold=0.2)

    torch_records = tiny_model.postprocess(
        (logits, boxes), metadata, threshold=0.2
    )
    numpy_records = decode_detections_numpy(
        (logits.numpy(), boxes.numpy()), metadata, spec
    )

    for expected, actual in zip(torch_records, numpy_records):
        assert np.array_equal(actual["labels"], expected["labels"].numpy())
        np.testing.assert_allclose(
            actual["bboxes"], expected["bboxes"].numpy(), atol=1e-4, rtol=0
        )
        np.testing.assert_allclose(
            actual["scores"], expected["scores"].numpy(), atol=1e-6, rtol=0
        )


def test_reference_decode_top_k_and_threshold(tiny_model, raw_outputs):
    logits, boxes, metadata = raw_outputs
    spec = build_postprocess_spec(tiny_model)
    limited = replace(spec, top_k=3)
    records = decode_detections_numpy((logits.numpy(), boxes.numpy()), metadata, limited)
    assert all(len(record["labels"]) <= 3 for record in records)
    # Scores are strictly descending within an image.
    for record in records:
        scores = record["scores"]
        assert np.all(scores[:-1] >= scores[1:])

    empty = decode_detections_numpy(
        (logits.numpy(), boxes.numpy()), metadata, spec, threshold=2.0
    )
    assert all(len(record["labels"]) == 0 for record in empty)


def test_unsupported_family_is_rejected(tiny_model):
    with pytest.raises(NotImplementedError, match="family"):
        build_postprocess_spec(tiny_model, family="segmentation")
    spec = PostprocessSpec(
        schema_version=1, family="picodet", input_size=(64, 64),
        num_classes=2, num_queries=10, top_k=10,
    )
    with pytest.raises(AutoVisionError, match="NMS config"):
        decode_detections_numpy((np.zeros((1, 10, 3)), np.zeros((1, 10, 4))), [{}], spec)


def test_picodet_spec_and_numpy_nms_match_torch(tmp_path):
    torch.manual_seed(0)
    model = PicoDetObjectDetection(
        num_classes=2, backbone="s", image_size=(64, 64),
        neck_channels=8, reg_max=7, num_top_queries=10, nms_iou_threshold=0.6,
    )
    model.eval()
    metadata = [{"orig_w": 100, "orig_h": 80}]
    logits = torch.full((1, 6, 2), -10.0)
    logits[0, 0, 0] = 3.0
    logits[0, 1, 0] = 2.9  # duplicate of box 0, suppressed by NMS
    logits[0, 2, 1] = 2.8
    logits[0, 3, 0] = 2.7
    logits[0, 4, 1] = 2.6
    logits[0, 5, 0] = 2.5
    boxes = torch.tensor(
        [[
            [0.50, 0.50, 0.20, 0.20],
            [0.50, 0.50, 0.20, 0.20],
            [0.10, 0.10, 0.10, 0.10],
            [0.80, 0.80, 0.10, 0.10],
            [0.20, 0.80, 0.10, 0.10],
            [0.80, 0.20, 0.10, 0.10],
        ]],
        dtype=torch.float32,
    )
    spec = build_postprocess_spec(model, family="picodet", score_threshold=0.1)
    assert spec.family == "picodet"
    assert spec.nms is not None
    assert spec.nms.iou_threshold == 0.6
    assert spec.nms.class_aware is True

    torch_records = model.postprocess((logits, boxes), metadata, threshold=0.1)
    numpy_records = decode_detections_numpy(
        (logits.numpy(), boxes.numpy()), metadata, spec
    )
    for expected, actual in zip(torch_records, numpy_records):
        assert np.array_equal(actual["labels"], expected["labels"].numpy())
        np.testing.assert_allclose(
            actual["bboxes"], expected["bboxes"].numpy(), atol=1e-4, rtol=0
        )
        np.testing.assert_allclose(
            actual["scores"], expected["scores"].numpy(), atol=1e-6, rtol=0
        )
    assert len(torch_records[0]["labels"]) == 5  # one duplicate suppressed

    sidecar = write_postprocess_spec(model, tmp_path / "pico.onnx")
    loaded = load_postprocess_spec(sidecar)
    assert loaded.family == "picodet"
    assert loaded.nms is not None


def test_sidecar_is_written_next_to_the_artifact(tiny_model, tmp_path):
    artifact = tmp_path / "model.onnx"
    sidecar = write_postprocess_spec(tiny_model, artifact)
    assert sidecar == tmp_path / "model.onnx.postprocess.json"
    loaded = load_postprocess_spec(sidecar)
    assert loaded.input_size == (64, 64)


@requires_export
def test_onnx_export_writes_a_valid_sidecar(tiny_model, tmp_path):
    out = tiny_model.export_onnx(tmp_path / "model.onnx", verify=True)
    sidecar = Path(str(out) + ".postprocess.json")
    assert sidecar.is_file()
    spec = load_postprocess_spec(sidecar)
    assert spec.box_format == "cxcywh_normalized"
    assert spec.outputs == ("logits", "boxes")
