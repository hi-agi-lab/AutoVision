# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""CLI ``export`` command: exported ``.pt`` -> ONNX / TensorRT dispatch.

Mirrors the deployment-level export tests: a tiny exported payload goes
through the real command (same validation as the Python API) and lands as
an ONNX graph that ONNX Runtime can open — the numerical parity gate runs
inside the export pipeline by default.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from autovision.cli._app import app
from autovision.tasks import (
    RTDETRObjectDetection,
    RTDETRObjectDetectionTrain,
    RTDETRObjectDetectionTrainArgs,
)

_export_extra_installed = (
    importlib.util.find_spec("onnxruntime") is not None
    and importlib.util.find_spec("onnxslim") is not None
)
requires_export = pytest.mark.skipif(
    not _export_extra_installed,
    reason="onnxruntime/onnxslim not installed (pip install -e '.[export]')",
)


def _write_exported_pt(tmp_path: Path) -> Path:
    """Minimal exported ``.pt`` payload (the ``load_task_model`` format)."""
    torch.manual_seed(0)
    model = RTDETRObjectDetection(
        num_classes=2, backbone="torchvision/resnet18",
        image_size=(64, 64), hidden_dim=32, num_queries=10,
        dec_layers=1, num_heads=4, num_top_queries=10,
    )
    train_model = RTDETRObjectDetectionTrain(
        RTDETRObjectDetectionTrainArgs(
            num_classes=2, image_size=(64, 64), hidden_dim=32,
            num_queries=10, dec_layers=1, num_heads=4,
            backbone="torchvision/resnet18", num_top_queries=10,
        )
    )
    train_model.model = model
    train_model.on_train_batch_end()
    out = tmp_path / "exported_last.pt"
    torch.save(
        {
            "schema_version": 1,
            "kind": "task_model",
            "model_class_path": model.class_path,
            "model_init_args": {**model.init_args, "model_name": "rtdetr/test"},
            "train_model": train_model.get_export_state_dict(),
        },
        out,
    )
    return out


def test_new_commands_registered():
    names = {command.name for command in app.registered_commands}
    assert {
        "export",
        "train-oriented-object-detection",
        "train-instance-segmentation",
        "train-panoptic-segmentation",
    } <= names


@requires_export
def test_export_command_onnx_with_gate(tmp_path):
    artifact = _write_exported_pt(tmp_path)
    onnx_path = tmp_path / "model.onnx"
    result = CliRunner().invoke(
        app, ["export", f"model={artifact}", f"out={onnx_path}", "precision=fp32"]
    )
    assert result.exit_code == 0, result.output
    assert onnx_path.exists()
    # The verification gate ran (default verify=true) and the graph opens.
    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    assert session.get_inputs()[0].shape[1:] == [3, 64, 64]


@requires_export
def test_export_command_rejects_unknown_suffix(tmp_path):
    artifact = _write_exported_pt(tmp_path)
    result = CliRunner().invoke(
        app, ["export", f"model={artifact}", f"out={tmp_path / 'model.xyz'}"]
    )
    assert result.exit_code != 0
    assert "Unsupported export target" in str(result.exception)
