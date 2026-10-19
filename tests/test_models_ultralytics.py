# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the ultralytics (YOLO/RT-DETR) package.

Primary coverage runs against the **real** ``ultralytics`` package
(architecture-only builds from ``.yaml`` names — no weight downloads).
A fake module (``tests/helpers_ultralytics.py``) covers version gating and
the legacy export path that the installed version cannot exercise.
"""

import importlib.metadata
import sys

import pytest
import torch

import autovision.encoders.ultralytics.ultralytics_source as ultralytics_package_module
from autovision.encoders import FeatureExtractor, get_source, get_wrapped_model
from autovision.encoders.model_wrapper import EncoderProtocol
from autovision.encoders.package_helpers import (
    list_sources,
    source_for_model,
)
from autovision.encoders.ultralytics import ULTRALYTICS_SOURCE, UltralyticsModelWrapper

ultralytics = pytest.importorskip(
    "ultralytics", reason="ultralytics package not installed (pip install -e '.[ultralytics]')"
)

_REAL_VERSION = ultralytics_package_module._ultralytics_version()


def make_wrapper(name: str = "yolov8n.yaml") -> UltralyticsModelWrapper:
    model = ULTRALYTICS_SOURCE.build(name, load_weights=False)
    return ULTRALYTICS_SOURCE.as_encoder(model)


@pytest.fixture
def fake_ultralytics(monkeypatch):
    """Install the fake ultralytics module and force version resolution
    through its ``__version__`` (metadata lookup disabled)."""
    from tests.helpers_ultralytics import install_fake_ultralytics

    fake = install_fake_ultralytics(monkeypatch)

    def _raise(name, package=None):
        raise importlib.metadata.PackageNotFoundError()

    monkeypatch.setattr(
        ultralytics_package_module.importlib.metadata, "version", _raise
    )
    return fake


class TestRegistration:
    def test_registered_before_custom(self):
        names = [p.name for p in list_sources()]
        assert "ultralytics" in names
        assert names.index("ultralytics") < names.index("custom")

    def test_get_source(self):
        assert get_source("ultralytics") is ULTRALYTICS_SOURCE

    def test_graceful_without_dependency(self, monkeypatch):
        import importlib

        monkeypatch.delitem(sys.modules, "ultralytics", raising=False)

        def _raise(name, package=None):
            raise ImportError(name)

        monkeypatch.setattr(importlib, "import_module", _raise)
        assert ULTRALYTICS_SOURCE.catalog_names() == []


class TestListModelNamesReal:
    def test_names_sorted_and_prefixed(self):
        names = ULTRALYTICS_SOURCE.catalog_names()
        assert names
        assert names == sorted(names)
        assert all(name.startswith("ultralytics/") for name in names)

    def test_families_present(self):
        names = ULTRALYTICS_SOURCE.catalog_names()
        assert "ultralytics/yolov8n.pt" in names
        assert "ultralytics/yolov8n.yaml" in names
        assert "ultralytics/yolo11n.pt" in names
        assert "ultralytics/rtdetr-l.pt" in names
        if _REAL_VERSION >= (8, 3, 78):
            assert "ultralytics/yolo12n.pt" in names
        if _REAL_VERSION >= (8, 4, 0):
            assert "ultralytics/yolo26n.pt" in names


class TestGetModelReal:
    def test_three_channels_required(self):
        with pytest.raises(ValueError, match="3 input channels"):
            ULTRALYTICS_SOURCE.build("yolov8n.yaml", num_input_channels=1)

    def test_load_weights_false_uses_yaml(self, monkeypatch):
        constructed = []

        class SpyYOLO(ultralytics.YOLO):
            def __init__(self, model="yolov8n.yaml", **kwargs):
                constructed.append(model)
                super().__init__(model=model, **kwargs)

        monkeypatch.setattr(ultralytics, "YOLO", SpyYOLO)
        ULTRALYTICS_SOURCE.build("yolov8n.pt", load_weights=False)
        assert constructed[-1] == "yolov8n.yaml"

    def test_detr_routing(self, monkeypatch):
        routed = []

        class SpyRTDETR(ultralytics.RTDETR):
            def __init__(self, model="rtdetr-l.yaml", **kwargs):
                routed.append(model)
                super().__init__(model=model, **kwargs)

        monkeypatch.setattr(ultralytics, "RTDETR", SpyRTDETR)
        ULTRALYTICS_SOURCE.build("rtdetr-l.yaml", load_weights=False)
        assert routed[-1] == "rtdetr-l.yaml"


class TestWrapperReal:
    def test_is_model_wrapper(self):
        assert isinstance(make_wrapper(), EncoderProtocol)

    def test_backbone_end_v8(self):
        wrapper = make_wrapper("yolov8n.yaml")
        # [Conv, Conv, C2f, Conv, C2f, Conv, C2f, Conv, C2f, SPPF, ...]
        assert wrapper._backbone_end == 9
        assert wrapper.encoder_dim() == 256

    def test_forward_shapes_v8(self):
        wrapper = make_wrapper("yolov8n.yaml")
        features = wrapper.feature_maps(torch.randn(2, 3, 64, 64))
        assert features["features"].shape == (2, 256, 2, 2)  # stride 32
        pooled = wrapper.pool_features(features)
        assert pooled["pooled_features"].shape == (2, 256, 1, 1)

    def test_backbone_end_v11_ends_at_c2psa(self):
        wrapper = make_wrapper("yolo11n.yaml")
        assert wrapper._backbone_end == 10  # C2PSA closes the backbone

    def test_backbone_end_rtdetr(self):
        wrapper = make_wrapper("rtdetr-l.yaml")
        assert wrapper._backbone_end >= 1
        features = wrapper.feature_maps(torch.randn(1, 3, 64, 64))
        assert features["features"].shape[1] == wrapper.encoder_dim()

    def test_arch_traits(self):
        assert make_wrapper().arch_traits() == {
            "model_type": "convolutional",
            "norm_type": "batchnorm",
        }

    def test_gradients_enabled(self):
        wrapper = make_wrapper()
        assert all(p.requires_grad for p in wrapper.parameters())

    def test_freeze_honored(self):
        model = ULTRALYTICS_SOURCE.build("yolov8n.yaml", load_weights=False)
        model.model.args["freeze"] = 2  # freeze the first two layers
        wrapper = ULTRALYTICS_SOURCE.as_encoder(model)
        layer_params = list(wrapper.inner_model.model[:2].parameters())
        assert all(not p.requires_grad for p in layer_params)
        rest = list(wrapper.inner_model.model[2:].parameters())
        assert all(p.requires_grad for p in rest)

    def test_train_eval_delegation(self):
        wrapper = make_wrapper()
        wrapper.eval()
        assert not wrapper.inner_model.training
        wrapper.train()
        assert wrapper.inner_model.training

    def test_device_apply_reaches_inner_params(self):
        wrapper = make_wrapper()
        seen = []
        wrapper._apply(lambda tensor: seen.append(1) or tensor)
        # Every parameter (and more: buffers) was traversed via the wrapper.
        assert len(seen) >= len(list(wrapper.parameters())) > 0

    def test_state_dict_roundtrip_strict(self):
        wrapper = make_wrapper()
        state = wrapper.state_dict()
        assert len(state) > 0
        wrapper.load_state_dict(state, strict=True)

    def test_embedding_model_roundtrip(self):
        wrapper = make_wrapper()
        embedding_model = FeatureExtractor(wrapped_model=wrapper, embed_dim=64)
        state = embedding_model.state_dict()
        assert any(k.startswith("_inner.") for k in state)
        assert any(k.startswith("_embed.") for k in state)
        # No duplicated backbone keys under a second name.
        assert not any("_backbone" in k for k in state)
        embedding_model.load_state_dict(state, strict=True)
        out = embedding_model(torch.randn(1, 3, 64, 64), pool=True)
        assert out.shape == (1, 64, 1, 1)

    def test_recognizes(self):
        wrapper = make_wrapper()
        assert ULTRALYTICS_SOURCE.recognizes(wrapper)
        assert not ULTRALYTICS_SOURCE.recognizes(wrapper._model[0])
        assert source_for_model(wrapper) is ULTRALYTICS_SOURCE

    def test_get_wrapped_model_by_name(self):
        wrapper = get_wrapped_model("ultralytics/yolov8n.yaml")
        assert isinstance(wrapper, UltralyticsModelWrapper)


class TestExportReal:
    def test_modern_save(self, tmp_path):
        wrapper = make_wrapper()
        out = tmp_path / "exported" / "model.pt"
        ULTRALYTICS_SOURCE.export_weights(wrapper, out, log_example=False)
        assert out.exists()
        assert out.stat().st_size > 1000


# ── fake-module tests: version gating and legacy export ─────────────────


class TestVersionGatingFake:
    def test_default_version_8_3_50(self, fake_ultralytics):
        assert ultralytics_package_module._ultralytics_version() == (8, 3, 50)
        names = ULTRALYTICS_SOURCE.catalog_names()
        assert "ultralytics/yolov8n.pt" in names
        assert "ultralytics/yolo11n.pt" in names
        assert "ultralytics/rtdetr-l.pt" in names
        assert "ultralytics/yolo12n.pt" not in names  # needs >= 8.3.78
        assert "ultralytics/yolo26n.pt" not in names  # needs >= 8.4.0
        assert not any("v8unmatched" in n for n in names)

    def test_older_version_gates_families(self, monkeypatch):
        from tests.helpers_ultralytics import install_fake_ultralytics

        install_fake_ultralytics(monkeypatch, version="8.2.0")

        def _raise(name, package=None):
            raise importlib.metadata.PackageNotFoundError()

        monkeypatch.setattr(
            ultralytics_package_module.importlib.metadata, "version", _raise
        )
        names = ULTRALYTICS_SOURCE.catalog_names()
        assert "ultralytics/yolov8n.pt" in names
        assert "ultralytics/yolo11n.pt" not in names  # 8.2.0 < 8.3.0
        assert "ultralytics/rtdetr-l.pt" in names  # 8.2.0 >= 8.0.140

    def test_yaml_validation_skips_missing(self, fake_ultralytics):
        # yolo26 is version-gated off; its yamls (incl. -p2/-p6) must not appear.
        names = ULTRALYTICS_SOURCE.catalog_names()
        assert not any("yolo26" in name for name in names)


class TestGetModelFake:
    def test_model_args_forwarded(self, fake_ultralytics):
        model = ULTRALYTICS_SOURCE.build("yolov8n.pt", model_args={"freeze": 2})
        assert model.model.args["freeze"] == 2

    def test_detr_routing(self, fake_ultralytics):
        ULTRALYTICS_SOURCE.build("rtdetr-l.pt")
        assert fake_ultralytics._constructed_names[-1] == "rtdetr-l.pt"

    def test_load_weights_false_converts_pt_to_yaml(self, fake_ultralytics):
        ULTRALYTICS_SOURCE.build("yolov8n.pt", load_weights=False)
        assert fake_ultralytics._constructed_names[-1] == "yolov8n.yaml"

    def test_fake_wrapper_backbone_patterns(self, fake_ultralytics):
        # Fake v8 stack: [Conv, Conv, C2f, SPPF, Upsample, Conv]
        wrapper = get_wrapped_model("ultralytics/yolov8n.pt")
        assert wrapper._backbone_end == 3  # SPPF closes the backbone
        assert wrapper.encoder_dim() == 32
        features = wrapper.feature_maps(torch.randn(2, 3, 64, 64))
        assert features["features"].shape == (2, 32, 16, 16)  # stride 4
        # Fake v11 stack ends at the first pattern element (C2PSA).
        wrapper11 = get_wrapped_model("ultralytics/yolo11n.yaml")
        assert wrapper11._backbone_end == 4

    def test_unsupported_architecture_raises(self, fake_ultralytics):
        from tests.helpers_ultralytics import Conv, FakeDetectionModel

        yolo = fake_ultralytics.YOLO("yolov8n.pt")
        yolo.model = FakeDetectionModel([Conv(3, 8), Conv(8, 8)])
        with pytest.raises(ValueError, match="backbone end"):
            UltralyticsModelWrapper(yolo)

    def test_gradients_reenabled_after_pt_load(self, fake_ultralytics):
        # The fake .pt build disables gradients; the wrapper re-enables them.
        wrapper = get_wrapped_model("ultralytics/yolov8n.pt")
        assert all(p.requires_grad for p in wrapper.parameters())


class TestExportFake:
    def test_legacy_fallback(self, monkeypatch, tmp_path):
        from tests.helpers_ultralytics import install_fake_ultralytics

        install_fake_ultralytics(monkeypatch, version="8.2.9")

        def _raise(name, package=None):
            raise importlib.metadata.PackageNotFoundError()

        monkeypatch.setattr(
            ultralytics_package_module.importlib.metadata, "version", _raise
        )
        wrapper = get_wrapped_model("ultralytics/yolov8n.pt")
        del wrapper._model[0].save  # instance attr: force the pre-8.3 path
        out = tmp_path / "legacy.pt"
        ULTRALYTICS_SOURCE.export_weights(wrapper, out, log_example=False)
        payload = torch.load(out, weights_only=False)
        assert "model" in payload


class TestFakeResolution:
    def test_fake_module_is_used_while_installed(self, fake_ultralytics):
        assert ultralytics_package_module._get_ultralytics() is fake_ultralytics
