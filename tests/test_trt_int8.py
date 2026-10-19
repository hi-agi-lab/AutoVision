# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""TensorRT int8 chain (roadmap D4): QDQ sanitizer, symmetric activations,
the QDQ parity gate, and a full engine build behind an env gate.

The engine build is slow (TensorRT compiles at a production spatial size)
and needs the ``tensorrt`` package plus a CUDA device: set
``AUTOVISION_TRT_TESTS=1`` to exercise it. Everything else runs CPU-only.
"""

import json
import os

import numpy as np
import onnx
import pytest
import torch
from onnx import numpy_helper
from tests.conftest import create_image

import autovision
from autovision.export.quantization import (
    Int8QuantizationConfig,
    suggest_first_layer_nodes,
)
from autovision.tasks.task_model_helpers import load_task_model


def _requires_cuda():
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


MODEL_ARGS = {"image_size": (64, 64), "num_queries": 10, "num_top_queries": 10}


def _tiny_dataset(tmp_path):
    images_dir = tmp_path / "images"
    rows, annotations = [], []
    for index in range(4):
        path = create_image(images_dir / f"img_{index:03d}.jpg", size=(64, 64), seed=index)
        rows.append({"id": index + 1, "file_name": path.name, "width": 64, "height": 64})
        annotations.append(
            {
                "id": index + 1,
                "image_id": index + 1,
                "category_id": 1,
                "bbox": [8.0, 8.0, 32.0, 32.0],
                "area": 1024.0,
                "iscrowd": 0,
            }
        )
    payload = {"images": rows, "annotations": annotations,
               "categories": [{"id": 1, "name": "cat"}]}
    train_json = tmp_path / "train.json"
    train_json.write_text(json.dumps(payload))
    return {
        "format": "coco",
        "train": {"annotations": str(train_json), "images": str(images_dir)},
        "val": {"annotations": str(train_json), "images": str(images_dir)},
    }


class TestQDQSanitizer:
    """The ORT-QDQ -> TensorRT interop fold (int32 DequantizeLinear constants)."""

    def _graph_with_int32_dequantize(self, tmp_path, *, per_channel: bool, axis: int = 0):
        quantized = numpy_helper.from_array(
            np.array([-64, 0, 64], dtype=np.int32), name="q_bias"
        )
        scale = numpy_helper.from_array(
            np.array([0.5, 0.5, 0.5], dtype=np.float32)
            if per_channel
            else np.array(0.5, dtype=np.float32),
            name="scale",
        )
        zero = numpy_helper.from_array(
            np.array(0, dtype=np.int32), name="zero_point"
        )
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["q_bias", "scale", "zero_point"],
            outputs=["bias_fp32"],
            name="bias_DQ",
            axis=axis,
        )
        consumer = onnx.helper.make_node(
            "Add", inputs=["bias_fp32", "bias_fp32"], outputs=["out"], name="add"
        )
        graph = onnx.helper.make_graph(
            [node, consumer],
            "g",
            inputs=[],
            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [3])],
            initializer=[quantized, scale, zero],
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
        path = tmp_path / "qdq.onnx"
        onnx.save(model, str(path))
        return path

    def test_int32_dequantize_folds_to_fp32_constant(self, tmp_path):
        from autovision.export.tensorrt_helpers import _sanitize_qdq_for_tensorrt

        path = self._graph_with_int32_dequantize(tmp_path, per_channel=False)
        folded = _sanitize_qdq_for_tensorrt(path)
        assert folded == 1
        model = onnx.load(str(path))
        assert all(node.op_type != "DequantizeLinear" for node in model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        values = numpy_helper.to_array(initializers["bias_fp32"])
        assert values.dtype == np.float32
        assert values.tolist() == pytest.approx([-32.0, 0.0, 32.0])

    def test_fold_drops_the_now_unused_quantized_initializers(self, tmp_path):
        from autovision.export.tensorrt_helpers import _sanitize_qdq_for_tensorrt

        path = self._graph_with_int32_dequantize(tmp_path, per_channel=False)
        _sanitize_qdq_for_tensorrt(path)
        names = {init.name for init in onnx.load(str(path)).graph.initializer}
        # q_bias / scale / zero_point are all unreferenced now; the folded
        # fp32 constant replaced them.
        assert "bias_fp32" in names
        assert names.isdisjoint({"q_bias", "scale", "zero_point"})

    def test_per_axis_scale_broadcasts(self, tmp_path):
        from autovision.export.tensorrt_helpers import _sanitize_qdq_for_tensorrt

        quantized = numpy_helper.from_array(
            np.array([[1, 2], [3, 4]], dtype=np.int32), name="q_w"
        )
        scale = numpy_helper.from_array(np.array([0.5, 2.0], dtype=np.float32), name="s")
        zero = numpy_helper.from_array(np.array(0, dtype=np.int32), name="z")
        node = onnx.helper.make_node(
            "DequantizeLinear", inputs=["q_w", "s", "z"], outputs=["w_fp32"],
            name="w_DQ", axis=1,
        )
        graph = onnx.helper.make_graph(
            [node], "g", inputs=[],
            outputs=[onnx.helper.make_tensor_value_info("w_fp32", onnx.TensorProto.FLOAT, [2, 2])],
            initializer=[quantized, scale, zero],
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
        path = tmp_path / "axis.onnx"
        onnx.save(model, str(path))

        assert _sanitize_qdq_for_tensorrt(path) == 1
        values = numpy_helper.to_array(
            {init.name: init for init in onnx.load(str(path)).graph.initializer}["w_fp32"]
        )
        assert np.asarray(values).ravel().tolist() == pytest.approx([0.5, 4.0, 1.5, 8.0])


class TestSymmetricActivations:
    def test_symmetric_default_reaches_quantize_static(self, monkeypatch, tmp_path):
        """ActivationSymmetric=True by default (TRT rejects non-zero zero points)."""
        from autovision.export import quantization as Q

        captured: dict = {}

        class _FakeQuantType:
            QInt8 = "QInt8"
            QUInt8 = "QUInt8"

        def fake_resolve():
            return (
                lambda **kwargs: captured.update(kwargs),
                "QDQ",
                {"QInt8": "QInt8", "QUInt8": "QUInt8"},
                {"MinMax": "MinMax"},
            )

        monkeypatch.setattr(Q, "_resolve_ort_quantization", fake_resolve)
        monkeypatch.setattr(onnx, "checker", type("C", (), {"check_model": staticmethod(lambda *a, **k: None)}))
        Q.quantize_onnx_int8(
            tmp_path / "in.onnx", tmp_path / "out.onnx",
            calibration_data_reader=object(),
            config=Int8QuantizationConfig(),
        )
        assert captured["extra_options"]["ActivationSymmetric"] is True
        config = Int8QuantizationConfig(activation_symmetric=False)
        assert config.to_dict()["activation_symmetric"] is False
        assert Int8QuantizationConfig().to_dict()["activation_symmetric"] is True


class TestInt8OnnxGate:
    def test_qdq_export_runs_the_parity_gate(self, tmp_path):
        pytest.importorskip("onnxruntime")
        result = autovision.train_object_detection(
            out=tmp_path / "run", data=_tiny_dataset(tmp_path), model="rtdetr/resnet18",
            steps=2, batch_size=2, devices="cpu", precision="32-true", num_workers=0,
            model_args=MODEL_ARGS,
        )
        net = load_task_model(result.best_exported_model_path, device="cpu")
        images = sorted((tmp_path / "images").glob("*.jpg"))
        samples = [
            {"images": net.preprocess_batch(net.preprocess_image(str(path))[0].unsqueeze(0))}
            for path in images
        ]
        out = net.export_onnx_int8(tmp_path / "model.int8.onnx", calibration_data=samples, verify=True)
        import onnxruntime as ort

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        outputs = session.run(None, {"images": samples[0]["images"].numpy()})
        assert outputs and all(np.isfinite(np.asarray(o)).all() for o in outputs)

        # An image PATH is accepted too (the documented CLI recipe): the
        # resolver preprocesses through the model's own pipeline.
        from_path = net.export_onnx_int8(
            tmp_path / "model_path.int8.onnx",
            calibration_data=str(tmp_path / "images"),
            verify=True,
        )
        assert from_path.exists()


_engine_gate = pytest.mark.skipif(
    os.environ.get("AUTOVISION_TRT_TESTS") != "1"
    or not torch.cuda.is_available()
    or __import__("importlib").util.find_spec("tensorrt") is None,
    reason="set AUTOVISION_TRT_TESTS=1 (needs tensorrt + CUDA); builds are slow",
)


@_engine_gate
class TestInt8EngineBuild:
    def test_engine_build_and_parity_at_production_size(self, tmp_path):
        """Full D4 path: QDQ graph (stem in fp32) -> strongly-typed engine ->
        torch-vs-engine parity within the int8 tolerances."""
        from autovision.export.tensorrt_helpers import TensorRTEngineRunner

        images_dir = tmp_path / "images"
        rows, annotations = [], []
        for index in range(4):
            path = create_image(images_dir / f"img_{index:03d}.jpg", size=(640, 640), seed=index)
            rows.append({"id": index + 1, "file_name": path.name, "width": 640, "height": 640})
            annotations.append(
                {"id": index + 1, "image_id": index + 1, "category_id": 1,
                 "bbox": [80.0, 80.0, 320.0, 320.0], "area": 57600.0, "iscrowd": 0}
            )
        payload = {"images": rows, "annotations": annotations,
                   "categories": [{"id": 1, "name": "cat"}]}
        train_json = tmp_path / "train.json"
        train_json.write_text(json.dumps(payload))
        data = {
            "format": "coco",
            "train": {"annotations": str(train_json), "images": str(images_dir)},
            "val": {"annotations": str(train_json), "images": str(images_dir)},
        }
        result = autovision.train_object_detection(
            out=tmp_path / "run", data=data, model="rtdetr/resnet18",
            steps=2, batch_size=1, devices="cpu", precision="32-true", num_workers=0,
            model_args={"image_size": (640, 640), "num_queries": 100, "num_top_queries": 100},
        )
        net = load_task_model(result.best_exported_model_path, device="cpu")
        # First-layers-fp32 recipe: exclude the stem Convs from quantization
        # (TensorRT has no int8 tactic for the stem Conv+Relu+MaxPool fusion).
        fp32_onnx = net.export_onnx(tmp_path / "fp32.onnx", verify=False)
        stem_convs = suggest_first_layer_nodes(fp32_onnx)
        assert stem_convs, "expected Conv nodes in the fp32 graph"
        engine = net.export_tensorrt(
            tmp_path / "model_int8.engine",
            precision="int8",
            calibration_data=[
                {"images": net.preprocess_batch(net.preprocess_image(str(path))[0].unsqueeze(0))}
                for path in sorted(images_dir.glob("*.jpg"))
            ],
            quantization_config=Int8QuantizationConfig(nodes_to_exclude=tuple(stem_convs)),
        )
        assert engine.exists()

        runner = TensorRTEngineRunner(engine, device="cuda")
        example = net.preprocess_batch(
            net.preprocess_image(str(sorted(images_dir.glob("*.jpg"))[0]))[0].unsqueeze(0)
        )
        torch_outputs = net.forward_backend(example)
        engine_outputs = runner.run(example)
        for torch_out, engine_out in zip(torch_outputs, engine_outputs):
            torch.testing.assert_close(
                torch_out.float().sum(dim=1),
                torch.as_tensor(engine_out).float().sum(dim=1),
                atol=2.0,
                rtol=0.5,
                check_device=False,
                check_dtype=False,
            )
