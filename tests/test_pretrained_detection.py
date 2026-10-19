# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pretrained detection handles (roadmap phase 1).

Real RF-DETR downloads stay behind ``AUTOVISION_HF_TESTS``; the CPU tests
exercise the dispatch, lazy-load and error contracts with a fake network so
a missing extra or a typo cannot silently produce a random model.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

from autovision.cli.model_handle import DetectionModel, detection
from autovision.errors import AutoVisionError
from autovision.tasks.detection.pretrained import (
    COCO_CLASSES,
    build_pretrained_detection,
    pretrained_class_names,
)


class FakeNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls: list[tuple[str, dict]] = []

    def predict(self, image, **kwargs):
        self.calls.append(("predict", kwargs))
        return {
            "bboxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }

    def export_openvino(self, path, **kwargs):
        self.calls.append(("openvino", kwargs))
        return path

    def export_onnx(self, path, **kwargs):
        self.calls.append(("onnx", kwargs))
        return path


def test_coco_class_table_is_complete():
    assert len(COCO_CLASSES) == 80
    assert COCO_CLASSES[0] == "person"
    assert COCO_CLASSES[-1] == "toothbrush"
    # RF-DETR keeps the COCO category-id head: 0=N/A, 1=person, 67=dining table.
    names = pretrained_class_names("rfdetr/rf-detr-small")
    assert len(names) == 91
    assert names[0] == "N/A"
    assert names[1] == "person"
    assert names[67] == "dining table"
    # GTR/DETR use the contiguous 0..79 head.
    gtr_names = pretrained_class_names("detr/gtrvit-s")
    assert gtr_names[0] == "person" and gtr_names[79] == "toothbrush"


def test_model_without_task_weights_is_rejected():
    with pytest.raises(AutoVisionError, match="no registered pretrained task weights"):
        build_pretrained_detection("rtdetr/resnet18")


def test_unknown_model_is_rejected():
    with pytest.raises(AutoVisionError, match="Unknown task model"):
        build_pretrained_detection("not/a-model")


def test_rfdetr_builder_uses_native_resolution_and_loads_weights(monkeypatch):
    import autovision.tasks.detection.rfdetr.task_model as rfdetr_task

    seen: dict[str, object] = {}

    class FakeRFDetr(FakeNetwork):
        def __init__(self, **kwargs):
            super().__init__()
            seen.update(kwargs)

    monkeypatch.setattr(rfdetr_task, "RFDETRObjectDetection", FakeRFDetr)
    network = build_pretrained_detection("rfdetr/rf-detr-small")
    assert isinstance(network, FakeRFDetr)
    assert seen["model"] == "rf-detr-small"
    assert seen["num_classes"] == 91  # COCO category-id head (0=N/A, 1..90)
    assert seen["image_size"] == (512, 512)
    assert seen["load_weights"] is True
    assert network.training is False


def test_gtr_builder_fails_readably_without_the_gla_stack(monkeypatch):
    try:
        import autovision.tasks.detection.shared.gtr as gtr_module
    except Exception:  # noqa: BLE001 - no gtr extra at all
        gtr_module = None
    if gtr_module is not None:
        monkeypatch.setattr(gtr_module, "gtr_available", lambda: False)
    with pytest.raises(AutoVisionError, match="gtr"):
        build_pretrained_detection("detr/gtrvit-s")


def test_detection_handle_lazy_load_and_predict(monkeypatch):
    import autovision.tasks.detection.pretrained as pretrained_module

    fake = FakeNetwork()
    monkeypatch.setattr(
        pretrained_module, "build_pretrained_detection", lambda key, device="cpu": fake
    )
    monkeypatch.setattr(
        pretrained_module,
        "pretrained_class_names",
        lambda key: {0: "person"},
    )

    handle = detection("rfdetr/rf-detr-small", pretrained=True)
    assert handle._network is None  # lazy: no load at construction
    result = handle.predict("image.jpg")
    assert torch.equal(result["labels"], torch.tensor([0]))
    assert handle._network is fake
    assert handle._classes == {0: "person"}


def test_from_pretrained_loads_eagerly(monkeypatch):
    import autovision.tasks.detection.pretrained as pretrained_module

    fake = FakeNetwork()
    monkeypatch.setattr(
        pretrained_module, "build_pretrained_detection", lambda key, device="cpu": fake
    )
    monkeypatch.setattr(
        pretrained_module, "pretrained_class_names", lambda key: {0: "person"}
    )
    handle = DetectionModel.from_pretrained("rfdetr/rf-detr-small")
    assert handle._network is fake


def test_plain_handle_still_requires_training():
    handle = detection("rtdetr/resnet18")
    with pytest.raises(AutoVisionError, match="No trained/exported model"):
        handle.predict("image.jpg")


def test_pretrained_handle_exports_openvino(monkeypatch, tmp_path):
    import autovision.tasks.detection.pretrained as pretrained_module

    fake = FakeNetwork()
    monkeypatch.setattr(
        pretrained_module, "build_pretrained_detection", lambda key, device="cpu": fake
    )
    monkeypatch.setattr(
        pretrained_module, "pretrained_class_names", lambda key: {0: "person"}
    )
    handle = detection("rfdetr/rf-detr-small", pretrained=True)
    out = handle.export(tmp_path / "model.xml", precision="fp32")
    assert out == tmp_path / "model.xml"
    assert fake.calls[-1][0] == "openvino"


@pytest.mark.skipif(
    os.environ.get("AUTOVISION_HF_TESTS", "").strip().lower() not in {"1", "true", "yes", "on"},
    reason="real RF-DETR download (set AUTOVISION_HF_TESTS=1)",
)
def test_real_rfdetr_nano_pretrained_smoke():
    """Real weights smoke; needs the rfdetr extra and network access."""
    network = build_pretrained_detection("rfdetr/rf-detr-nano")
    assert network.training is False
    assert int(network.num_classes) == 80
