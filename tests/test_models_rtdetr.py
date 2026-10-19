# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the RT-DETR raw-module wrapper and the custom package export."""

import pytest
import torch
from tests.helpers_models import TinyWrapper
from torch import nn

from autovision.encoders.custom import CustomSource
from autovision.encoders.rtdetr import RTDETRModelWrapper


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out_channels = [64, 128]
        self.stage1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.stage2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

    def forward(self, x):
        features = self.stage1(x)
        return [features, self.stage2(features)]


class FakeRTDETRModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = FakeBackbone()


class TestRTDETRModelWrapper:
    def test_requires_backbone(self):
        with pytest.raises(ValueError, match=".backbone"):
            RTDETRModelWrapper(nn.Linear(2, 2))

    def test_requires_out_channels(self):
        class NoChannels(nn.Module):
            def __init__(self):
                super().__init__()
                # nn.Sequential does not expose an out_channels attribute.
                self.backbone = nn.Sequential(nn.Conv2d(3, 8, 3))

        with pytest.raises(ValueError, match="out_channels"):
            RTDETRModelWrapper(NoChannels())

    def test_feature_maps_last_scale(self):
        wrapper = RTDETRModelWrapper(FakeRTDETRModel())
        out = wrapper.feature_maps(torch.randn(2, 3, 64, 64))
        assert out["features"].shape == (2, 128, 16, 16)

    def test_feature_dim(self):
        assert RTDETRModelWrapper(FakeRTDETRModel()).encoder_dim() == 128

    def test_pool(self):
        wrapper = RTDETRModelWrapper(FakeRTDETRModel())
        out = wrapper.feature_maps(torch.randn(1, 3, 64, 64))
        pooled = wrapper.pool_features(out)
        assert pooled["pooled_features"].shape == (1, 128, 1, 1)

    def test_arch_traits(self):
        assert RTDETRModelWrapper(FakeRTDETRModel()).arch_traits() == {
            "model_type": "convolutional",
            "norm_type": "batchnorm",
        }

    def test_get_model(self):
        model = FakeRTDETRModel()
        assert RTDETRModelWrapper(model).raw_module() is model

    def test_public_module(self):
        import autovision.encoders.rtdetr as package

        assert package.RTDETRModelWrapper is RTDETRModelWrapper


class TestCustomSourceExport:
    def test_export_wrapper_state_dict(self, tmp_path):
        wrapper = TinyWrapper()
        out = tmp_path / "exported" / "custom.pt"
        CustomSource.export_weights(wrapper, out, log_example=False)
        state = torch.load(out, weights_only=True)
        assert "conv.weight" in state  # TinyWrapper's conv

    def test_export_via_get_model(self, tmp_path):
        class Holder(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = nn.Linear(4, 4)

            def raw_module(self):
                return self.inner

        out = tmp_path / "custom.pt"
        CustomSource.export_weights(Holder(), out, log_example=False)
        assert "inner.weight" in torch.load(out, weights_only=True)
