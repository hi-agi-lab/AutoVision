# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the timm model package (optional extra)."""

from __future__ import annotations

import os

import pytest
import torch

timm = pytest.importorskip("timm", reason="timm not installed")

from autovision.encoders.package_helpers import (  # noqa: E402
    get_source,
    get_wrapped_model,
)
from autovision.encoders.timm import TIMM_SOURCE  # noqa: E402
from autovision.encoders.timm.timm_wrapper import (  # noqa: E402
    TIMMModelWrapper,
    arch_traits_for_name,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class TestPackage:
    def test_registered(self):
        assert get_source("timm") is TIMM_SOURCE

    def test_catalog_names_nonempty(self):
        names = TIMM_SOURCE.catalog_names()
        assert any(name.startswith("timm/") for name in names)

    def test_get_model_untrained_by_default(self):
        model = TIMM_SOURCE.build("resnet18", load_weights=False)
        assert model.__module__.split(".")[0] == "timm"

    def test_get_model_nondefault_channels(self):
        model = TIMM_SOURCE.build("resnet18", num_input_channels=5, load_weights=False)
        assert model.conv1.in_channels == 5

    def test_wrapper_roundtrip(self):
        model = TIMM_SOURCE.build("resnet18", load_weights=False)
        wrapper = TIMM_SOURCE.as_encoder(model)
        assert isinstance(wrapper, TIMMModelWrapper)
        assert wrapper.raw_module() is model
        assert model.num_features == wrapper.encoder_dim()

    def test_via_public_resolver(self):
        wrapper = get_wrapped_model("timm/resnet18", load_weights=False)
        assert isinstance(wrapper, TIMMModelWrapper)

    def test_pretrained_requires_explicit_args(self):
        # Pretrained weights need an explicit opt-in; real downloads are
        # gated behind AUTOVISION_HF_TESTS=1 (see README) so restricted
        # networks don't hang the suite.
        if not os.environ.get("AUTOVISION_HF_TESTS"):
            pytest.skip("set AUTOVISION_HF_TESTS=1 to test real weight downloads")
        model = TIMM_SOURCE.build("resnet18", model_args={"pretrained": True})
        assert model.num_features > 0

    def test_export_weights(self, tmp_path):
        model = TIMM_SOURCE.build("resnet18", load_weights=False)
        out = tmp_path / "timm_model.pt"
        TIMM_SOURCE.export_weights(model, out, log_example=False)
        assert out.exists()
        reference = TIMM_SOURCE.build("resnet18", load_weights=False)
        reference.load_state_dict(torch.load(out, weights_only=True))


class TestWrapperProtocol:
    def _wrapper(self) -> TIMMModelWrapper:
        model = timm.create_model("resnet18", pretrained=False)
        return TIMMModelWrapper(model)

    def test_feature_maps_nchw(self):
        wrapper = self._wrapper()
        output = wrapper.feature_maps(torch.randn(2, 3, 32, 32))
        features = output["features"]
        assert features.ndim == 4
        assert features.shape[0] == 2
        assert features.shape[1] == wrapper.encoder_dim()

    def test_pool_features_shape(self):
        wrapper = self._wrapper()
        output = wrapper.feature_maps(torch.randn(2, 3, 32, 32))
        pooled = wrapper.pool_features(output)["pooled_features"]
        assert pooled.shape[:2] == (2, wrapper.encoder_dim())

    def test_vit_features_drop_class_token(self):
        model = timm.create_model(
            "vit_tiny_patch16_224", pretrained=False, dynamic_img_size=True
        )
        wrapper = TIMMModelWrapper(model)
        output = wrapper.feature_maps(torch.randn(1, 3, 32, 32))
        features = output["features"]
        assert features.ndim == 4  # tokens reshaped to a grid, class token gone
        grid_tokens = features.shape[2] * features.shape[3]
        sequence_tokens = 1 + (32 // 16) ** 2  # cls + patches
        assert grid_tokens == sequence_tokens - 1


class TestArchTraits:
    def test_prefix_table(self):
        assert arch_traits_for_name("resnet18") == {
            "model_type": "convolutional", "norm_type": "batchnorm",
        }
        assert arch_traits_for_name("vit_base_patch16_224") == {
            "model_type": "transformer", "norm_type": "layernorm",
        }
        assert arch_traits_for_name("convnext_tiny") == {
            "model_type": "convolutional", "norm_type": "layernorm",
        }
        assert arch_traits_for_name("maxvit_tiny_224") == {
            "model_type": "hybrid", "norm_type": "layernorm",
        }

    def test_wrapper_arch_traits(self):
        model = timm.create_model("resnet18", pretrained=False)
        wrapper = TIMMModelWrapper(model)
        assert wrapper.arch_traits()["norm_type"] == "batchnorm"
