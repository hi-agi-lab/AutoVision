# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""OpenVINO backend (roadmap D3): IR export, parity gate, benchmark backend."""

import json

import pytest
from tests.conftest import create_image
from typer.testing import CliRunner

import autovision
from autovision.cli._app import app
from autovision.tasks.task_model_helpers import load_task_model

pytest.importorskip("openvino")

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


@pytest.fixture(scope="module")
def tiny_export(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ov")
    result = autovision.train_object_detection(
        out=tmp / "run", data=_tiny_dataset(tmp), model="rtdetr/resnet18",
        steps=2, batch_size=2, devices="cpu", precision="32-true", num_workers=0,
        model_args=MODEL_ARGS,
    )
    return result.best_exported_model_path


class TestOpenVINOExport:
    def test_ir_export_with_parity_gate(self, tiny_export, tmp_path):
        net = load_task_model(tiny_export, device="cpu")
        xml = net.export_openvino(tmp_path / "ov" / "model.xml", verify=True)
        assert xml.exists() and xml.suffix == ".xml"
        assert xml.with_suffix(".bin").exists()  # IR weights
        metadata = json.loads(xml.with_suffix(".json").read_text())
        assert metadata["backend"] == "openvino"
        assert metadata["precision"] == "fp32"
        assert metadata["device"] == "CPU"
        assert not xml.with_suffix(".onnx").exists()  # intermediate cleaned up

    def test_runtime_matches_torch(self, tiny_export, tmp_path):
        from autovision.export.openvino_helpers import OpenVINOEngineRunner

        net = load_task_model(tiny_export, device="cpu")
        xml = net.export_openvino(tmp_path / "ov2" / "model.xml", verify=False)
        runner = OpenVINOEngineRunner(xml, device="CPU")
        example = net.model_input_spec.example_inputs(batch_size=2)
        images = next(iter(example.values())).float()
        torch_outputs = net.forward_backend(images)
        ov_outputs = runner.run(images)
        net.verify_onnx_export_outputs(torch_outputs, ov_outputs)
        assert len(ov_outputs) == len(torch_outputs)

    def test_onnx_suffixed_target_is_rejected(self, tiny_export, tmp_path):
        """A *.onnx target is not a directory: refuse instead of creating
        ``model.onnx/model.xml``."""
        net = load_task_model(tiny_export, device="cpu")
        with pytest.raises(ValueError, match="neither a .xml file"):
            net.export_openvino(tmp_path / "not_a_dir.onnx", verify=False)

    def test_missing_openvino_extra_message(self, monkeypatch, tiny_export, tmp_path):
        """The import guard names the extra instead of a bare ImportError."""
        import builtins

        from autovision.errors import AutoVisionError
        from autovision.export import openvino_helpers

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "openvino":
                raise ImportError("blocked for the test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(AutoVisionError, match="autovision\\[openvino\\]"):
            openvino_helpers._import_openvino()


class TestOpenVINOBenchmark:
    def test_benchmark_with_parity_and_numbers(self, tiny_export, tmp_path):
        net = load_task_model(tiny_export, device="cpu")
        results = autovision.benchmark(
            net,
            out=tmp_path / "bench",
            verify_parity=True,
            backends={"torch": {"precision": "fp32"}, "openvino": {"ov_device": "CPU"}},
            runs=5,
            warmup=2,
            batch_size=1,
        )
        assert set(results) == {"torch", "openvino"}
        openvino_result = results["openvino"]
        assert openvino_result.device == "openvino-compiled"
        assert openvino_result.latency_ms_p50 > 0
        assert openvino_result.extra["openvino_device"] == "CPU"
        assert openvino_result.extra["ir"].endswith("model.xml")
        summary = (tmp_path / "bench" / "benchmark.md").read_text()
        assert "openvino" in summary
        assert (tmp_path / "bench" / "benchmark_openvino.json").exists()


class TestExportCli:
    def test_xml_suffix_dispatches_to_openvino(self, tiny_export, tmp_path):
        result = CliRunner().invoke(
            app,
            ["export", f"model={tiny_export}", f"out={tmp_path / 'cli_model.xml'}"],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "cli_model.xml").exists()
