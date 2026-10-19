# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Dynamic H/W ONNX export (roadmap D2): multi-size execution + parity gate.

The export gate runs a SECOND spatial size (torch vs ONNX Runtime at the
same size); these tests additionally run the graph at sizes the export
never saw and compare against torch, and lock down the fail-closed rule
that dynamic spatial export requires the dynamo exporter (the legacy
TorchScript path bakes the traced size into the graph).
"""

import importlib.util
import json
import logging

import pytest
import torch
from tests.conftest import create_image

import autovision
from autovision.export.export_onnx import _dim_from_marker, _spec_dynamic_axes
from autovision.tasks.task_model_helpers import load_task_model

MODEL_ARGS = {
    "image_size": (64, 64),
    "num_queries": 10,
    "num_top_queries": 10,
}


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
def tiny_network(tmp_path_factory):
    pytest.importorskip("onnxruntime")
    tmp = tmp_path_factory.mktemp("d2")
    result = autovision.train_object_detection(
        out=tmp / "run",
        data=_tiny_dataset(tmp),
        model="rtdetr/resnet18",
        steps=2,
        batch_size=2,
        devices="cpu",
        precision="32-true",
        num_workers=0,
        model_args=MODEL_ARGS,
    )
    return load_task_model(result.best_exported_model_path, device="cpu")


class TestDynamicSpatialExport:
    def test_export_runs_at_sizes_it_never_saw(self, tiny_network, tmp_path):
        import onnxruntime as ort

        out = tiny_network.export_onnx(
            tmp_path / "dyn.onnx", dynamic_spatial=True, verify=True
        )
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        # Dynamic axes are recorded on the graph input (names, not sizes).
        dim_names = [str(dim) for dim in input_meta.shape]
        assert "height" in dim_names and "width" in dim_names

        for size in (48, 96, 160):
            outputs = session.run(
                None, {"images": torch.rand(1, 3, size, size).numpy()}
            )
            assert outputs and all(o.shape[0] == 1 for o in outputs)

    def test_parity_at_an_unseen_spatial_size(self, tiny_network, tmp_path):
        """torch vs ORT at a size the export never traced."""
        import onnxruntime as ort

        out = tiny_network.export_onnx(
            tmp_path / "dyn2.onnx", dynamic_spatial=True, verify=False
        )
        reference = load_task_model(tiny_network._exported_path, device="cpu") \
            if hasattr(tiny_network, "_exported_path") else None  # noqa: SLF001
        net = reference or tiny_network
        net = net.to("cpu").float().eval()
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        probe = torch.rand(1, 3, 128, 128)
        torch_outputs = net.forward_backend(probe)
        if not isinstance(torch_outputs, (list, tuple)):
            import dataclasses

            torch_outputs = [
                getattr(torch_outputs, f.name)
                for f in dataclasses.fields(torch_outputs)
            ]
        onnx_outputs = session.run(None, {"images": probe.numpy()})
        tiny_network.verify_onnx_export_outputs(torch_outputs, onnx_outputs)

    def test_spec_markers_also_require_the_dynamo_exporter(self, tiny_network, tmp_path, monkeypatch):
        """A dynamic-axis MARKER (not the flag) must fail closed too: the
        legacy exporter would emit a graph that claims dynamic H/W but bakes
        the traced sizes."""
        import torch.onnx

        original = torch.onnx.export

        def flaky_export(*args, **kwargs):
            if kwargs.get("dynamo"):
                raise RuntimeError("simulated dynamo failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(torch.onnx, "export", flaky_export)
        # model_input_spec is a property (fresh object per access): patch the
        # class property so the export sees the marker-declaring spec.
        mutated = tiny_network.model_input_spec
        # Bounds must admit the example AND the model's internal size guard
        # (feature pyramids need size // divisor >= 1); a too-low min fails
        # the strict trace loudly (documented on _dim_from_marker).
        mutated.input_dynamic_shapes = {"images": (None, None, (64, 8192), (64, 8192))}
        monkeypatch.setattr(
            type(tiny_network), "model_input_spec", property(lambda _self: mutated)
        )
        with pytest.raises(autovision.AutoVisionError, match="dynamo"):
            tiny_network.export_onnx(tmp_path / "marker.onnx", verify=False)

    def test_dynamic_spatial_requires_the_dynamo_exporter(self, tiny_network, tmp_path, monkeypatch):
        import torch.onnx

        original = torch.onnx.export

        def flaky_export(*args, **kwargs):
            if kwargs.get("dynamo"):
                raise RuntimeError("simulated dynamo failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(torch.onnx, "export", flaky_export)
        with pytest.raises(autovision.AutoVisionError, match="dynamic_spatial"):
            tiny_network.export_onnx(
                tmp_path / "fails.onnx", dynamic_spatial=True, verify=False
            )


class TestDynamicSpatialAcrossFamilies:
    """D2 coverage across backbone families (roadmap §3.6 遗留 ③)."""

    @pytest.mark.skipif(
        importlib.util.find_spec("timm") is None,
        reason="HGNetV2 backbone needs the timm extra (pip install -e '.[timm]')",
    )
    def test_hgnetv2_cnn_exports_and_runs_at_unseen_sizes(self, tmp_path):
        """The timm CNN line (HGNetV2 spectrum backbones) is size-flexible:
        dynamic H/W export + parity at two sizes the export never traced."""
        import onnxruntime as ort
        import torch

        from autovision.tasks.detection.detr import DETRObjectDetection

        torch.manual_seed(0)
        net = DETRObjectDetection(
            num_classes=2, model_name="hgnetv2-b0", image_size=(256, 256),
            num_top_queries=20, load_weights=False,  # offline: random init
        )
        net.eval()
        out = net.export_onnx(tmp_path / "hgnet.onnx", dynamic_spatial=True, verify=True)
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        # ``export_onnx`` deployed the model; it is the parity reference.
        for size in (192, 320):  # multiples of the 2 x patch divisor (64)
            probe = torch.rand(1, 3, size, size)
            torch_outputs = net.forward_backend(probe)
            import dataclasses

            if not isinstance(torch_outputs, (list, tuple)) and dataclasses.is_dataclass(torch_outputs):
                torch_outputs = [
                    getattr(torch_outputs, f.name) for f in dataclasses.fields(torch_outputs)
                ]
            onnx_outputs = session.run(None, {"images": probe.numpy()})
            net.verify_onnx_export_outputs(torch_outputs, onnx_outputs)

    def test_hf_vit_dynamic_export_is_either_working_or_actionable(self, tmp_path):
        """The HF-backed ViT family branches in Python on the input size
        (transformers' patch-count branch), which strict tracing cannot turn
        symbolic — the failure must be ACTIONABLE, and if a future
        transformers stops branching, the export is expected to work."""
        import torch

        from autovision.tasks.detection.detr import DETRObjectDetection

        torch.manual_seed(0)
        net = DETRObjectDetection(
            num_classes=2, model_name="dinov2-vittest-notpretrained",
            image_size=(112, 112), num_top_queries=20, load_weights=False,
        )
        net.eval()
        out = tmp_path / "vit_dyn.onnx"
        try:
            net.export_onnx(out, dynamic_spatial=True, verify=False)
        except autovision.AutoVisionError as exc:
            message = str(exc)
            assert "dynamic_spatial=False" in message
            assert "PYTHON on the INPUT SIZE" in message or "dynamo" in message
        else:
            assert out.exists()


class TestSpecDynamicAxes:
    def test_marker_forms(self):
        from torch.export import Dim

        assert _dim_from_marker(Dim("h", min=8), 2) is not None
        assert _spec_dynamic_axes(
            type("S", (), {"input_dynamic_shapes": {"images": (None, None, "height", (32, 512))}})(),
            "images",
        ) == {2: "height", 3: (32, 512)}
        assert _spec_dynamic_axes(
            type("S", (), {"input_dynamic_shapes": {}})(), "images"
        ) == {}

    def test_unknown_marker_fails_loud(self):
        with pytest.raises(ValueError, match="marker"):
            _dim_from_marker(object(), 2)


class TestExportTraceFallbacks:
    def test_first_call_setup_is_warmed_before_tracing(
        self, tiny_network, tmp_path, monkeypatch, caplog
    ):
        """Regression: one-time untraceable setup on the FIRST forward.

        The transformers ViT backbones install output-capturing hooks under
        a ``threading.Lock`` on their first eager call; dynamo cannot trace
        ``with <lock>``, so without a warm-up every HF-backed export failed
        at the strict trace.
        """
        import threading

        original = tiny_network.forward_backend
        lock = threading.Lock()
        calls = {"count": 0}

        def first_call_takes_a_lock(x):
            if calls["count"] == 0:
                calls["count"] += 1
                with lock:  # untraceable by dynamo — must be consumed by warm-up
                    return original(x)
            return original(x)

        monkeypatch.setattr(tiny_network, "forward_backend", first_call_takes_a_lock)
        with caplog.at_level(logging.WARNING, logger="autovision"):
            out = tiny_network.export_onnx(tmp_path / "warm.onnx", verify=False)
        assert out.exists()
        assert calls["count"] >= 1
        # Pin the path: a trace that hit the lock would fail and silently
        # fall back to the legacy exporter, which ALSO produces an artifact —
        # so "an artifact exists" alone cannot distinguish the warm-up from
        # its absence (the fallback masks it). The dynamo path must complete.
        assert "legacy TorchScript" not in caplog.text

    def test_strict_trace_failure_falls_back_to_legacy_for_static_axes(
        self, tiny_network, tmp_path, monkeypatch
    ):
        """A strict-trace failure with STATIC axes is not fatal: the legacy
        TorchScript exporter is the documented fallback."""
        import onnxruntime as ort
        import torch.export

        def broken_export(*args, **kwargs):
            raise RuntimeError("simulated strict-trace failure")

        monkeypatch.setattr(torch.export, "export", broken_export)
        out = tiny_network.export_onnx(tmp_path / "legacy.onnx", verify=False)
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        outputs = session.run(
            None, {"images": torch.rand(1, 3, 64, 64).numpy()}
        )
        assert outputs and all(o.shape[0] == 1 for o in outputs)

    def test_strict_trace_failure_is_fatal_for_dynamic_axes(
        self, tiny_network, tmp_path, monkeypatch
    ):
        import torch.export

        def broken_export(*args, **kwargs):
            raise RuntimeError("Constraints violated (height, width)!")

        monkeypatch.setattr(torch.export, "export", broken_export)
        with pytest.raises(autovision.AutoVisionError) as excinfo:
            tiny_network.export_onnx(
                tmp_path / "no_dyn.onnx", dynamic_spatial=True, verify=False
            )
        message = str(excinfo.value)
        # Actionable: names the remedy AND the model-side branch cause.
        assert "dynamic_spatial=False" in message
        assert "PYTHON on the INPUT SIZE" in message
