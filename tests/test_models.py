# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
import pytest
import torch
from torch import nn

from autovision.encoders import (
    EncoderProtocol,
    FeatureExtractor,
    catalog_names,
    get_source,
    get_wrapped_model,
    missing_encoder_members,
    parse_model_name,
    register_source,
    source_for_model,
)
from autovision.encoders.model_wrapper import ArchTraitsSource
from autovision.encoders.package_helpers import list_catalog_sources, list_sources
from autovision.errors import UnknownModelError


class TinyWrapper(nn.Module):
    """Minimal EncoderProtocol implementation used across tests."""

    def __init__(self, dim: int = 8, spatial: int = 2, model_type: str = "convolutional"):
        super().__init__()
        self.dim = dim
        self.spatial = spatial
        self._model_type = model_type
        self.conv = nn.Conv2d(3, dim, 3, padding=1)

    def feature_maps(self, x):
        return {"features": self.conv(x), "cls_token": self.conv(x).mean(dim=(2, 3))}

    def pool_features(self, x):
        pooled = torch.nn.functional.adaptive_avg_pool2d(x["features"], 1)
        return {"pooled_features": pooled}

    def encoder_dim(self) -> int:
        return self.dim

    def raw_module(self):
        return self.conv

    def arch_traits(self):
        return {"model_type": self._model_type, "norm_type": "batchnorm"}


class TestProtocols:
    def test_tiny_wrapper_is_model_wrapper(self):
        assert isinstance(TinyWrapper(), EncoderProtocol)
        assert isinstance(TinyWrapper(), ArchTraitsSource)

    def test_missing_attrs_diagnostic(self):
        class Incomplete(nn.Module):
            def feature_maps(self, x): ...

        missing = missing_encoder_members(Incomplete(), exclude_module_attrs=True)
        assert "encoder_dim" in missing
        assert "pool_features" in missing
        assert "raw_module" in missing
        assert "feature_maps" not in missing  # implemented

    def test_module_attrs_not_reported(self):
        assert missing_encoder_members(TinyWrapper()) == []


class TestParseModelName:
    def test_simple(self):
        assert parse_model_name("torchvision/resnet18") == ("torchvision", "resnet18")

    def test_nested_name_keeps_rest(self):
        assert parse_model_name("pkg/a/b/c") == ("pkg", "a/b/c")

    def test_missing_slash_raises(self):
        with pytest.raises(ValueError, match="must have the format"):
            parse_model_name("resnet18")


class TestPackageRegistry:
    def test_torchvision_registered(self):
        names = [p.name for p in list_sources()]
        assert "torchvision" in names
        assert names[-1] == "custom"  # custom always last

    def test_get_source(self):
        package = get_source("torchvision")
        assert package.name == "torchvision"

    def test_get_unknown_package(self):
        with pytest.raises(ValueError, match="Unknown package"):
            get_source("nonexistent")

    def test_list_catalog_sources_excludes_custom(self):
        assert "custom" not in [p.name for p in list_catalog_sources()]

    def test_catalog_names_sorted(self):
        names = catalog_names()
        assert names == sorted(names)
        assert "torchvision/resnet18" in names
        assert "torchvision/convnext_tiny" in names

    def test_duplicate_registration_raises(self):
        from autovision.encoders.package import ModelSource

        class Duplicate(ModelSource):
            name = "torchvision"

            @classmethod
            def export_weights(cls, model, out, log_example=True): ...

            @classmethod
            def recognizes(cls, model):
                return False

        with pytest.raises(ValueError, match="already registered"):
            register_source(Duplicate())


class TestTorchvisionSource:
    def test_build_resnet18(self):
        wrapper = get_wrapped_model("torchvision/resnet18")
        assert isinstance(wrapper, EncoderProtocol)
        model = wrapper.raw_module()
        assert model.conv1.in_channels == 3

    def test_build_adapted_channels(self):
        wrapper = get_wrapped_model("torchvision/resnet18", num_input_channels=1)
        assert wrapper.raw_module().conv1.in_channels == 1
        wrapper4 = get_wrapped_model("torchvision/resnet18", num_input_channels=5)
        assert wrapper4.raw_module().conv1.in_channels == 5

    def test_feature_maps_shapes(self):
        wrapper = get_wrapped_model("torchvision/resnet18")
        x = torch.randn(2, 3, 64, 64)
        out = wrapper.feature_maps(x)
        assert out["features"].shape == (2, 512, 2, 2)
        pooled = wrapper.pool_features(out)
        assert pooled["pooled_features"].shape == (2, 512, 1, 1)
        assert wrapper.encoder_dim() == 512

    def test_arch_traits(self):
        wrapper = get_wrapped_model("torchvision/resnet18")
        info = wrapper.arch_traits()
        assert info == {"model_type": "convolutional", "norm_type": "batchnorm"}

    def test_convnext_wrapper(self):
        wrapper = get_wrapped_model("torchvision/convnext_tiny")
        assert wrapper.arch_traits() == {
            "model_type": "convolutional",
            "norm_type": "layernorm",
        }
        out = wrapper.feature_maps(torch.randn(1, 3, 64, 64))
        assert out["features"].shape == (1, 768, 2, 2)

    def test_unsupported_model_name(self):
        with pytest.raises(ValueError):
            get_wrapped_model("torchvision/vit_h_14_does_not_exist_pattern")

    def test_export_weights(self, tmp_path):
        wrapper = get_wrapped_model("torchvision/resnet18")
        out = tmp_path / "exported" / "model.pt"
        from autovision.encoders.torchvision.torchvision_source import (
            TORCHVISION_SOURCE,
        )

        TORCHVISION_SOURCE.export_weights(wrapper, out)
        assert out.exists()
        state = torch.load(out, weights_only=True)
        assert "conv1.weight" in state


class TestWrappedModelResolution:
    def test_instance_passthrough(self):
        wrapper = TinyWrapper()
        assert get_wrapped_model(wrapper) is wrapper

    def test_nn_module_attributed(self):
        from autovision.encoders.torchvision.torchvision_source import (
            TORCHVISION_SOURCE,
        )

        raw = TORCHVISION_SOURCE.build("resnet18")
        package = source_for_model(raw, include_custom=False)
        assert package.name == "torchvision"

    def test_unknown_module_raises_with_diagnostic(self):
        with pytest.raises(UnknownModelError, match="Missing"):
            source_for_model(nn.Sequential(nn.Linear(2, 2)), include_custom=False, fallback_custom=False)

    def test_incomplete_wrapper_raises_with_missing_list(self):
        class Incomplete(nn.Module):
            pass

        with pytest.raises(UnknownModelError, match="implement the EncoderProtocol"):
            get_wrapped_model(Incomplete())


class TestEmbeddingModel:
    def test_identity_head(self):
        wrapper = TinyWrapper(dim=8)
        embedding_model = FeatureExtractor(wrapped_model=wrapper)
        assert embedding_model.embed_dim == 8
        x = torch.randn(2, 3, 16, 16)
        pooled = embedding_model(x, pool=True)
        assert pooled.shape == (2, 8, 1, 1)
        features = embedding_model(x, pool=False)
        assert features.shape == (2, 8, 16, 16)

    def test_projection_head(self):
        wrapper = TinyWrapper(dim=8)
        embedding_model = FeatureExtractor(wrapped_model=wrapper, embed_dim=16)
        assert embedding_model.embed_dim == 16
        out = embedding_model(torch.randn(2, 3, 8, 8), pool=True)
        assert out.shape == (2, 16, 1, 1)
