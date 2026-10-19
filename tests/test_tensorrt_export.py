# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the TensorRT export path.

Full engine builds need the ``tensorrt`` package plus a GPU; the tests below
cover everything reachable without it: argument validation (which happens
before the TensorRT import by design), the graceful degradation message, and
the builder-API version tolerance (TensorRT 10.x vs 11.x shapes, exercised
against fake modules).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from autovision.export.tensorrt_helpers import (
    _enable_half_precision,
    _network_creation_flags,
    export_onnx_to_tensorrt,
)


def _noop_export_fn(**kwargs):  # pragma: no cover - only reached with trt installed
    raise AssertionError("export_onnx_fn must not run when validation fails")


class _FakeFlag:
    """Mimics a pybind enum member: int(flag) yields its bit."""

    def __init__(self, bit: int) -> None:
        self.bit = bit

    def __int__(self) -> int:
        return self.bit


class _RecordingBuildConfig:
    def __init__(self) -> None:
        self.flags: list[object] = []

    def set_flag(self, flag) -> None:
        self.flags.append(flag)


class TestValidation:
    def test_batch_size_ordering_enforced(self, tmp_path):
        with pytest.raises(ValueError, match="min_batch_size <= opt_batch_size"):
            export_onnx_to_tensorrt(
                export_onnx_fn=_noop_export_fn,
                out=tmp_path / "model.engine",
                min_batch_size=4,
                opt_batch_size=2,
                max_batch_size=8,
            )

    def test_strongly_typed_excludes_fp32_attention(self, tmp_path):
        with pytest.raises(ValueError, match="mutually exclusive"):
            export_onnx_to_tensorrt(
                export_onnx_fn=_noop_export_fn,
                out=tmp_path / "model.engine",
                strongly_typed=True,
                fp32_attention_scores=True,
            )


class TestBuilderApiVersionTolerance:
    """The builder API moved between TensorRT 10.x and 11.x: EXPLICIT_BATCH
    and the precision flags left the enums (explicit batch became the only
    mode; precision moved to the graph dtype)."""

    def test_creation_flags_with_explicit_batch(self):
        tensorrt = SimpleNamespace(
            NetworkDefinitionCreationFlag=SimpleNamespace(
                EXPLICIT_BATCH=_FakeFlag(0), STRONGLY_TYPED=_FakeFlag(1)
            )
        )
        assert _network_creation_flags(tensorrt, strongly_typed=False) == 1
        assert _network_creation_flags(tensorrt, strongly_typed=True) == 1 | 2

    def test_creation_flags_without_explicit_batch(self):
        tensorrt = SimpleNamespace(
            NetworkDefinitionCreationFlag=SimpleNamespace(
                STRONGLY_TYPED=_FakeFlag(1)
            )
        )
        assert _network_creation_flags(tensorrt, strongly_typed=False) == 0
        assert _network_creation_flags(tensorrt, strongly_typed=True) == 2

    def test_half_precision_sets_flag_when_available(self):
        fp16, constraints = _FakeFlag(0), _FakeFlag(1)
        tensorrt = SimpleNamespace(
            BuilderFlag=SimpleNamespace(
                FP16=fp16, OBEY_PRECISION_CONSTRAINTS=constraints
            )
        )
        config = _RecordingBuildConfig()
        builder = SimpleNamespace(platform_has_fast_fp16=True)
        _enable_half_precision(tensorrt, builder, config)
        assert config.flags == [fp16, constraints]

    def test_half_precision_without_platform_query(self):
        """TensorRT 11 removed the platform query; unknown != unsupported."""
        fp16 = _FakeFlag(0)
        tensorrt = SimpleNamespace(BuilderFlag=SimpleNamespace(FP16=fp16))
        config = _RecordingBuildConfig()
        _enable_half_precision(tensorrt, SimpleNamespace(), config)
        assert config.flags == [fp16]

    def test_half_precision_without_fp16_flag_is_noop(self):
        """TensorRT 11 removed the FP16 flag entirely: nothing to set."""
        tensorrt = SimpleNamespace(BuilderFlag=SimpleNamespace(TF32=_FakeFlag(0)))
        config = _RecordingBuildConfig()
        _enable_half_precision(tensorrt, SimpleNamespace(), config)
        assert config.flags == []


class TestGracefulDegradation:
    def test_missing_tensorrt_gives_install_guidance(self, tmp_path):
        try:
            import tensorrt  # noqa: F401
        except ModuleNotFoundError:
            pass
        else:
            pytest.skip("tensorrt is installed; degradation path not applicable")
        with pytest.raises(ModuleNotFoundError, match="pip install tensorrt"):
            export_onnx_to_tensorrt(
                export_onnx_fn=_noop_export_fn,
                out=tmp_path / "model.engine",
            )

    def test_task_model_method_exposed(self):
        from autovision.tasks import RTDETRObjectDetection

        assert hasattr(RTDETRObjectDetection, "export_tensorrt")


class TestEngineParity:
    def test_flatten_tensors_is_deterministic(self):
        from autovision.export.tensorrt_helpers import _flatten_tensors

        value = {
            "b": [torch.ones(1)],
            "a": (torch.zeros(1), {"c": torch.full((1,), 2.0)}),
        }
        flattened = _flatten_tensors(value)
        # Insertion order mirrors the forward/export output order.
        assert [float(t.item()) for t in flattened] == [1.0, 0.0, 2.0]

    def test_verify_engine_outputs_passes_within_tolerance(self, monkeypatch, tmp_path):
        import autovision.export.tensorrt_helpers as helpers

        inputs = torch.ones(1, 3)

        class _FakeRunner:
            def __init__(self, engine_path, *, device="cuda"):
                del engine_path, device

            def run(self, images):
                return [images * 2.0]

        monkeypatch.setattr(helpers, "TensorRTEngineRunner", _FakeRunner)
        helpers.verify_engine_outputs(
            tmp_path / "unused.engine", [inputs * 2.0], inputs
        )

    def test_verify_engine_outputs_fails_closed(self, monkeypatch, tmp_path):
        import autovision.export.tensorrt_helpers as helpers

        inputs = torch.ones(1, 3)

        class _WrongRunner:
            def __init__(self, engine_path, *, device="cuda"):
                del engine_path, device

            def run(self, images):
                return [images * 3.0]

        monkeypatch.setattr(helpers, "TensorRTEngineRunner", _WrongRunner)
        with pytest.raises(RuntimeError, match="parity check failed"):
            helpers.verify_engine_outputs(
                tmp_path / "unused.engine", [inputs * 2.0], inputs
            )

    def test_verify_engine_outputs_rejects_shape_mismatch(self, monkeypatch, tmp_path):
        import autovision.export.tensorrt_helpers as helpers

        inputs = torch.ones(1, 3)

        class _ShapeRunner:
            def __init__(self, engine_path, *, device="cuda"):
                del engine_path, device

            def run(self, images):
                return [torch.ones(1, 4)]

        monkeypatch.setattr(helpers, "TensorRTEngineRunner", _ShapeRunner)
        with pytest.raises(RuntimeError, match="shape mismatch"):
            helpers.verify_engine_outputs(
                tmp_path / "unused.engine", [torch.ones(1, 3)], inputs
            )
