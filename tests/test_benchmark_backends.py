# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for multi-backend inference benchmarking (torch / ONNX / TensorRT)."""

import json

import pytest
import torch

pytest.importorskip("onnx", reason="onnx not installed (pip install -e '.[export]')")
onnxruntime = pytest.importorskip("onnxruntime", reason="onnxruntime not installed (pip install -e '.[export]')")
pytest.importorskip("onnxslim", reason="onnxslim not installed (pip install -e '.[export]')")

from autovision.export import benchmark  # noqa: E402
from autovision.export.benchmark import (  # noqa: E402
    ONNXBackendArgs,
    TorchBackendArgs,
    _normalize_backends,
)
from autovision.tasks import RTDETRObjectDetection  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return RTDETRObjectDetection(
        num_classes=2, backbone="torchvision/resnet18",
        image_size=(64, 64), hidden_dim=32, num_queries=10,
        dec_layers=1, num_heads=4, num_top_queries=10,
    )


def _torch_only() -> dict:
    return {"torch": TorchBackendArgs()}


class TestNormalizeBackends:
    def test_default_is_torch_plus_onnx(self):
        normalized = _normalize_backends(None)
        assert set(normalized) == {"torch", "onnx"}

    def test_dicts_are_coerced(self):
        normalized = _normalize_backends({"torch": {"compile": True}})
        assert normalized["torch"].compile is True

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="Unknown backends"):
            _normalize_backends({"tensorquant": {}})


class TestBenchmarkCommand:
    def test_torch_backend(self, tiny_model, tmp_path):
        results = benchmark(
            tiny_model,
            tmp_path / "bench",
            batch_size=2,
            runs=2,
            warmup=1,
            backends={"torch": TorchBackendArgs()},
        )
        assert set(results) == {"torch"}
        torch_result = results["torch"]
        assert torch_result.images_per_sec > 0
        assert torch_result.extra["backend"] == "torch"
        assert (tmp_path / "bench" / "benchmark_torch.json").exists()
        assert (tmp_path / "bench" / "benchmark.md").exists()

    def test_onnx_backend_exports_and_times_session(self, tiny_model, tmp_path):
        results = benchmark(
            tiny_model,
            tmp_path / "bench",
            batch_size=2,
            runs=2,
            warmup=1,
            backends={"onnx": ONNXBackendArgs(provider="cpu")},
        )
        assert "onnx" in results
        assert results["onnx"].extra["backend"] == "onnx"
        assert (tmp_path / "bench" / "model.onnx").exists()
        payload = json.loads(
            (tmp_path / "bench" / "benchmark_onnx.json").read_text()
        )
        assert payload["images_per_sec"] > 0

    def test_parity_gate_torch_vs_onnx(self, tiny_model, tmp_path):
        results = benchmark(
            tiny_model,
            tmp_path / "bench",
            batch_size=2,
            runs=2,
            warmup=1,
            backends={"torch": TorchBackendArgs(), "onnx": ONNXBackendArgs()},
            verify_parity=True,
        )
        assert set(results) == {"torch", "onnx"}

    def test_model_spec_string_resolves(self, tmp_path):
        results = benchmark(
            "rtdetr/resnet18",
            tmp_path / "bench",
            num_classes=2,
            image_size=(64, 64),
            batch_size=1,
            runs=1,
            warmup=1,
            backends={"torch": TorchBackendArgs()},
        )
        assert "torch" in results

    def test_unknown_model_spec_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported benchmark model"):
            benchmark(
                "yolo/test",
                tmp_path / "bench",
                backends={"torch": TorchBackendArgs()},
            )

    def test_tensorrt_backend_requires_cuda(self, tiny_model, tmp_path):
        with pytest.raises(ValueError, match="CUDA device"):
            benchmark(
                tiny_model,
                tmp_path / "bench",
                runs=1,
                warmup=1,
                device="cpu",
                backends={"tensorrt": {}},
            )
