# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""ONNX export must not destroy a previous good artifact on failure."""

import errno

import pytest

from autovision.export.export_onnx import ONNXExportMixin


class _DummyExport(ONNXExportMixin):
    pass


def test_failed_export_preserves_the_previous_artifact(tmp_path, monkeypatch):
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous-good-graph")
    dummy = _DummyExport()

    def fail(*args, **kwargs):
        raise RuntimeError("export failed")

    monkeypatch.setattr(dummy, "_export_onnx_to_path", fail)
    with pytest.raises(RuntimeError, match="export failed"):
        dummy.export_onnx(target, write_spec=False)
    assert target.read_bytes() == b"previous-good-graph"
    assert not any(p.name.endswith(".tmp.onnx") for p in tmp_path.iterdir())


def test_enospc_export_preserves_the_previous_artifact(tmp_path, monkeypatch):
    """Disk full during export keeps the old graph and sweeps the temp."""
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous-good-graph")
    dummy = _DummyExport()

    def fail(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(dummy, "_export_onnx_to_path", fail)
    with pytest.raises(OSError, match="No space left"):
        dummy.export_onnx(target, write_spec=False)
    assert target.read_bytes() == b"previous-good-graph"
    assert not any(p.name.endswith(".tmp.onnx") for p in tmp_path.iterdir())


def test_successful_export_replaces_the_target_atomically(tmp_path, monkeypatch):
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous")
    dummy = _DummyExport()

    def succeed(out, **kwargs):
        with open(out, "wb") as handle:
            handle.write(b"new-graph")
        return out

    monkeypatch.setattr(dummy, "_export_onnx_to_path", succeed)
    result = dummy.export_onnx(target, write_spec=False)
    assert result == target
    assert target.read_bytes() == b"new-graph"
    assert not any(p.name.endswith(".tmp.onnx") for p in tmp_path.iterdir())
