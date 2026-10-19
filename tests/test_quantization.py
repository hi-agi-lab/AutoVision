# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for int8 post-training quantization (PTQ) of exported ONNX models."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from autovision.export import (
    Int8CalibrationReader,
    Int8QuantizationConfig,
    image_folder_calibration_reader,
    quantize_onnx_int8,
    suggest_first_layer_nodes,
)
from autovision.tasks import (
    RTDETRObjectDetection,
)

onnxruntime = pytest.importorskip("onnxruntime", reason="onnxruntime not installed (pip install -e '.[export]')")
pytest.importorskip("onnxslim", reason="onnxslim not installed (pip install -e '.[export]')")

OPSET = 18


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return RTDETRObjectDetection(
        num_classes=2, backbone="torchvision/resnet18",
        image_size=(64, 64), hidden_dim=32, num_queries=10,
        dec_layers=1, num_heads=4, num_top_queries=10,
    )


@pytest.fixture(scope="module")
def calibration_samples():
    # API contract: samples are dicts of input_name -> Tensor.
    return [{"images": torch.randn(1, 3, 64, 64)} for _ in range(4)]


@pytest.fixture(scope="module")
def fp32_onnx(tiny_model, tmp_path_factory):
    out = tmp_path_factory.mktemp("fp32") / "model_fp32.onnx"
    tiny_model.export_onnx(out, format_args={"opset_version": OPSET})
    return out


def _count_qdq(model) -> tuple[int, int]:
    import onnx

    qdq = sum(
        1 for n in model.graph.node
        if n.op_type in ("QuantizeLinear", "DequantizeLinear")
    )
    int8_inits = sum(1 for i in model.graph.initializer if i.data_type == onnx.TensorProto.INT8)
    return qdq, int8_inits


class TestInt8CalibrationReader:
    def test_numpy_conversion_and_termination(self, calibration_samples):
        reader = Int8CalibrationReader(calibration_samples)
        first = reader.get_next()
        assert set(first) == {"images"}
        assert isinstance(first["images"], np.ndarray)
        assert first["images"].shape == (1, 3, 64, 64)
        for _ in range(3):
            assert reader.get_next() is not None
        assert reader.get_next() is None

    def test_name_mismatch_raises(self, calibration_samples):
        reader = Int8CalibrationReader(
            calibration_samples, expected_input_names=["pixels"]
        )
        with pytest.raises(ValueError, match="do not match"):
            reader.get_next()

    def test_non_dict_sample_raises(self):
        reader = Int8CalibrationReader([torch.randn(1, 3, 8, 8)])
        with pytest.raises(TypeError, match="dicts"):
            reader.get_next()

    def test_rewind_replays_samples(self, calibration_samples):
        reader = Int8CalibrationReader(calibration_samples)
        first_pass = [reader.get_next()["images"] for _ in range(2)]
        reader.rewind()
        assert reader.get_next()["images"] is not None
        # First sample after rewind equals the original first sample.
        np.testing.assert_array_equal(first_pass[0], calibration_samples[0]["images"].numpy())

    def test_num_samples_counts(self, calibration_samples):
        reader = Int8CalibrationReader(calibration_samples)
        while reader.get_next() is not None:
            pass
        assert reader.num_samples == len(calibration_samples)


class TestImageFolderCalibrationReader:
    @pytest.fixture()
    def image_dir(self, tmp_path):
        rng = np.random.default_rng(0)
        for i in range(5):
            array = rng.integers(0, 256, (48, 52, 3), dtype=np.uint8)
            Image.fromarray(array).save(tmp_path / f"cal_{i}.jpg")
        return tmp_path

    def test_shapes_align_with_model_spec(self, tiny_model, image_dir):
        reader = image_folder_calibration_reader(tiny_model, image_dir, batch_size=2)
        samples = []
        while (s := reader.get_next()) is not None:
            samples.append(s)
        assert len(samples) == 3  # ceil(5/2)
        assert samples[0]["images"].shape == (2, 3, 64, 64)
        assert samples[-1]["images"].shape == (1, 3, 64, 64)  # remainder batch
        assert reader.num_samples == 3

    def test_max_samples(self, tiny_model, image_dir):
        reader = image_folder_calibration_reader(tiny_model, image_dir, max_samples=1)
        assert reader.get_next() is not None
        assert reader.get_next() is None

    def test_bad_batch_size(self, tiny_model, image_dir):
        with pytest.raises(ValueError, match="batch_size"):
            image_folder_calibration_reader(tiny_model, image_dir, batch_size=0)

    def test_empty_dir_raises(self, tiny_model, tmp_path):
        with pytest.raises(ValueError, match="No images"):
            image_folder_calibration_reader(tiny_model, tmp_path)


class TestInt8QuantizationConfig:
    def test_bad_calibrate_method(self):
        with pytest.raises(ValueError, match="calibrate_method"):
            Int8QuantizationConfig(calibrate_method="MSE")._validate()

    def test_bad_quant_types(self):
        with pytest.raises(ValueError, match="activation_type"):
            Int8QuantizationConfig(activation_type="QInt4")._validate()
        with pytest.raises(ValueError, match="weight_type"):
            Int8QuantizationConfig(weight_type="QInt4")._validate()

    def test_to_dict_snapshot(self):
        config = Int8QuantizationConfig(nodes_to_exclude=("n1",))
        payload = config.to_dict()
        assert payload["per_channel"] is True
        assert payload["op_types_to_quantize"] == ["Conv", "Gemm", "MatMul"]
        assert payload["nodes_to_exclude"] == ["n1"]
        # JSON-serializable (goes into ONNX metadata verbatim).
        assert json.loads(json.dumps(payload)) == payload

    def test_validation_precedes_ort_import(self, fp32_onnx, tmp_path):
        # Invalid config fails before touching onnxruntime internals.
        reader = Int8CalibrationReader([torch.randn(1, 3, 64, 64)])
        with pytest.raises(ValueError, match="calibrate_method"):
            quantize_onnx_int8(
                fp32_onnx, tmp_path / "q.onnx", reader,
                Int8QuantizationConfig(calibrate_method="nope"),
            )


class TestSuggestFirstLayerNodes:
    def test_lists_leading_convs_in_graph_order(self, tmp_path):
        import onnx
        from onnx import TensorProto, helper

        inputs = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])
        outputs = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 6, 6])
        weight1 = helper.make_tensor("w1", TensorProto.FLOAT, [4, 3, 3, 3], [0.1] * 108)
        weight2 = helper.make_tensor("w2", TensorProto.FLOAT, [4, 4, 3, 3], [0.1] * 144)
        nodes = [
            helper.make_node("Conv", ["images", "w1"], ["c1"], name="/stem/conv1/Conv"),
            helper.make_node("Relu", ["c1"], ["r1"], name="/stem/relu/Relu"),
            helper.make_node("Conv", ["r1", "w2"], ["c2"], name="/stem/conv2/Conv"),
        ]
        graph = helper.make_graph(nodes, "stem", [inputs], [outputs], [weight1, weight2])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        path = tmp_path / "stem.onnx"
        onnx.save(model, path)

        assert suggest_first_layer_nodes(path) == [
            "/stem/conv1/Conv", "/stem/conv2/Conv",
        ]
        assert suggest_first_layer_nodes(path, limit=1) == ["/stem/conv1/Conv"]
        # The recipe is exactly the config shape the failure hint suggests.
        config = Int8QuantizationConfig(nodes_to_exclude=tuple(suggest_first_layer_nodes(path)))
        assert config.to_dict()["nodes_to_exclude"] == ["/stem/conv1/Conv", "/stem/conv2/Conv"]

    def test_best_effort_never_raises(self, tmp_path):
        assert suggest_first_layer_nodes(tmp_path / "missing.onnx") == []
        corrupt = tmp_path / "corrupt.onnx"
        corrupt.write_bytes(b"not an onnx graph")
        assert suggest_first_layer_nodes(corrupt) == []
        assert suggest_first_layer_nodes("/dev/null", limit=0) == []


class TestQuantizeOnnxInt8:
    def test_qdq_graph_and_parity(self, fp32_onnx, calibration_samples, tmp_path):
        import onnx
        import onnxruntime as ort

        out = tmp_path / "int8.onnx"
        reader = Int8CalibrationReader(calibration_samples)
        quantize_onnx_int8(fp32_onnx, out, reader)
        assert out.exists()
        assert out.stat().st_size < fp32_onnx.stat().st_size  # int8 shrinks the model

        model = onnx.load(str(out))
        qdq, int8_inits = _count_qdq(model)
        assert qdq > 0
        assert int8_inits > 0

        sample = {"images": torch.randn(1, 3, 64, 64).numpy()}
        fp32_out = ort.InferenceSession(
            str(fp32_onnx), providers=["CPUExecutionProvider"]
        ).run(None, sample)
        int8_out = ort.InferenceSession(
            str(out), providers=["CPUExecutionProvider"]
        ).run(None, sample)
        for fp32_o, int8_o in zip(fp32_out, int8_out):
            assert fp32_o.shape == int8_o.shape

    def test_per_channel_false(self, fp32_onnx, calibration_samples, tmp_path):
        out = tmp_path / "int8_tensorwise.onnx"
        quantize_onnx_int8(
            fp32_onnx, out, Int8CalibrationReader(calibration_samples),
            Int8QuantizationConfig(per_channel=False),
        )
        assert out.exists()

    def test_nodes_to_exclude_reduces_quantization(
        self, fp32_onnx, calibration_samples, tmp_path
    ):
        import onnx

        fp32_model = onnx.load(str(fp32_onnx))
        conv_names = [
            n.name for n in fp32_model.graph.node if n.op_type == "Conv" and n.name
        ]
        assert conv_names, "expected named Conv nodes in the FP32 graph"
        exclude = tuple(conv_names[:2])

        out_all = tmp_path / "all.onnx"
        out_excluded = tmp_path / "excluded.onnx"
        reader = Int8CalibrationReader(calibration_samples)
        quantize_onnx_int8(fp32_onnx, out_all, reader)
        reader.rewind()
        quantize_onnx_int8(
            fp32_onnx, out_excluded, reader,
            Int8QuantizationConfig(nodes_to_exclude=exclude),
        )
        qdq_all, _ = _count_qdq(onnx.load(str(out_all)))
        qdq_excluded, _ = _count_qdq(onnx.load(str(out_excluded)))
        assert qdq_excluded < qdq_all

    def test_missing_ort_raises_clear_error(
        self, fp32_onnx, calibration_samples, tmp_path, monkeypatch
    ):
        import sys

        monkeypatch.setitem(
            sys.modules, "onnxruntime.quantization", None
        )
        monkeypatch.delitem(sys.modules, "onnxruntime.quantization", raising=False)
        # Block re-import.
        import builtins

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name.startswith("onnxruntime"):
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        with pytest.raises(ImportError, match="onnxruntime is required"):
            quantize_onnx_int8(
                fp32_onnx, tmp_path / "q.onnx", Int8CalibrationReader(calibration_samples)
            )


class TestExportOnnxInt8:
    def test_e2e_export_with_metadata_and_dynamic_batch(
        self, tiny_model, calibration_samples, tmp_path
    ):
        import onnx
        import onnxruntime as ort

        out = tmp_path / "model_int8.onnx"
        tiny_model.export_onnx_int8(
            out, calibration_data=calibration_samples, opset_version=OPSET
        )
        assert out.exists()
        assert not out.parent.joinpath("model_int8.fp32.onnx").exists()  # default: deleted

        model = onnx.load(str(out))
        qdq, int8_inits = _count_qdq(model)
        assert qdq > 0 and int8_inits > 0

        metadata = {p.key: p.value for p in model.metadata_props}
        payload = json.loads(metadata["int8_quantization"])
        assert payload["per_channel"] is True
        assert payload["calibrate_method"] == "MinMax"
        assert payload["num_calibration_samples"] == len(calibration_samples)
        assert "onnxruntime" in payload["calibration"]
        assert float(metadata["int8_parity_atol"]) > 0

        # Dynamic batch survives quantization.
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        outputs = session.run(None, {"images": torch.randn(3, 3, 64, 64).numpy()})
        assert outputs[0].shape[0] == 3

    def test_keep_fp32_reference(self, tiny_model, calibration_samples, tmp_path):
        out = tmp_path / "kept_int8.onnx"
        tiny_model.export_onnx_int8(
            out, calibration_data=calibration_samples,
            opset_version=OPSET, keep_fp32_reference=True,
        )
        assert out.parent.joinpath("kept_int8.fp32.onnx").exists()

    def test_missing_calibration_data_raises(self, tiny_model, tmp_path):
        with pytest.raises(ValueError, match="calibration"):
            tiny_model.export_onnx_int8(tmp_path / "x.onnx", opset_version=OPSET)

    def test_parity_gate_failure_raises(
        self, tiny_model, calibration_samples, tmp_path
    ):
        # Zero tolerance: int8 quantization error is guaranteed to exceed it.
        with pytest.raises(AssertionError):
            tiny_model.export_onnx_int8(
                tmp_path / "bad.onnx", calibration_data=calibration_samples,
                opset_version=OPSET, parity_atol=0.0, parity_rtol=0.0,
            )

    def test_verify_false_skips_gate(self, tiny_model, calibration_samples, tmp_path):
        out = tiny_model.export_onnx_int8(
            tmp_path / "noverify.onnx", calibration_data=calibration_samples,
            opset_version=OPSET, verify=False,
        )
        assert out.exists()

    def test_custom_calibration_reader(self, tiny_model, calibration_samples, tmp_path):
        out = tmp_path / "custom_reader.onnx"
        reader = Int8CalibrationReader(calibration_samples)
        tiny_model.export_onnx_int8(
            out, calibration_reader=reader, opset_version=OPSET
        )
        assert out.exists()

    def test_image_folder_reader_e2e(self, tiny_model, tmp_path):
        rng = np.random.default_rng(1)
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        for i in range(3):
            Image.fromarray(
                rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            ).save(image_dir / f"i_{i}.jpg")
        out = tmp_path / "folder_int8.onnx"
        tiny_model.export_onnx_int8(
            out,
            calibration_reader=image_folder_calibration_reader(
                tiny_model, image_dir, batch_size=2
            ),
            opset_version=OPSET,
        )
        assert out.exists()


class TestExportContract:
    def test_export_onnx_int8_is_separate_chain(self, tiny_model, tmp_path):
        # The scalar precision switch of export_onnx only controls FP32/FP16
        # graph conversion; int8 is a dedicated PTQ chain (calibrated).
        with pytest.raises(ValueError, match="precision"):
            tiny_model.export_onnx(tmp_path / "x.onnx", precision="int8")
