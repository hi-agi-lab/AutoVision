# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the transformers-gated DINOv3 package.

The installed transformers does not ship DINOv3 classes yet, so the package
logic is exercised by injecting a fake ``transformers`` module (replacing
``sys.modules["transformers"]``) that provides the expected DINOv3 classes.
When a future transformers version ships real DINOv3 support, the package
auto-enables against it.
"""

import hashlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from autovision.encoders import get_wrapped_model
from autovision.encoders.dinov3 import (
    DINOV3_SOURCE,
    TransformersDinov3ConvNextWrapper,
    TransformersDinov3ViTWrapper,
)

# ── current environment: transformers without DINOv3 ────────────────────


class TestGating:
    def test_support_matches_installed_transformers(self):
        # The real transformers may or may not ship DINOv3 classes; the
        # package must agree with whatever is installed.
        from autovision.encoders.dinov3.dinov3_source import _dinov3_classes

        has_support = _dinov3_classes() is not None
        names = DINOV3_SOURCE.catalog_names()
        assert bool(names) is has_support
        if has_support:
            assert "dinov3/vits16" in names
            assert "dinov3/convnext-tiny" in names

    def test_no_support_raises_on_get_model(self):
        from autovision.encoders.dinov3.dinov3_source import _dinov3_classes

        if _dinov3_classes() is not None:
            pytest.skip("installed transformers ships DINOv3")
        with pytest.raises(ImportError, match="DINOv3 support"):
            DINOV3_SOURCE.build("vits16", load_weights=False)

    def test_graceful_without_transformers(self, monkeypatch):
        import importlib

        monkeypatch.delitem(sys.modules, "transformers", raising=False)

        def _raise(name, package=None):
            raise ImportError(name)

        monkeypatch.setattr(importlib, "import_module", _raise)
        assert DINOV3_SOURCE.catalog_names() == []


# ── fake transformers with DINOv3 classes ───────────────────────────────


class FakeDinov3ViTConfig:
    def __init__(self, **kwargs) -> None:
        self.model_type = "dinov3_vit"
        self.num_channels = 3
        defaults = dict(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            image_size=224,
            patch_size=16,
            num_register_tokens=4,
        )
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)


class FakeDinov3ViTModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embeddings = SimpleNamespace(
            patch_embeddings=SimpleNamespace(
                projection=nn.Conv2d(
                    config.num_channels, config.hidden_size, config.patch_size, config.patch_size
                )
            )
        )

    def forward(self, pixel_values, **kwargs):
        patches = self.embeddings.patch_embeddings.projection(pixel_values)
        b, d, h, w = patches.shape
        seq = patches.flatten(2).transpose(1, 2)
        prefix = 1 + getattr(self.config, "num_register_tokens", 0)
        hidden = torch.cat(
            [torch.ones(b, prefix, d, device=seq.device), seq], dim=1
        )
        return SimpleNamespace(last_hidden_state=hidden)


class FakeDinov3ConvNextConfig:
    def __init__(self, **kwargs) -> None:
        self.model_type = "dinov3_convnext"
        self.num_channels = 3
        defaults = dict(
            hidden_sizes=[96, 192, 384, 768],
            image_size=224,
            patch_size=16,
        )
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)


class FakeDinov3ConvNextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embeddings = SimpleNamespace(
            # Stem projects to the LAST stage channels: the wrapper's
            # encoder_dim is hidden_sizes[-1].
            patch_embeddings=nn.Conv2d(
                config.num_channels, config.hidden_sizes[-1], config.patch_size, config.patch_size
            )
        )

    def forward(self, pixel_values, **kwargs):
        return SimpleNamespace(
            last_hidden_state=self.embeddings.patch_embeddings(pixel_values)
        )


@pytest.fixture
def fake_dinov3_transformers(monkeypatch):
    """Replace sys.modules['transformers'] with a stub exposing DINOv3."""
    fake = types.ModuleType("transformers")
    fake.Dinov3ViTConfig = FakeDinov3ViTConfig
    fake.Dinov3ViTModel = FakeDinov3ViTModel
    fake.Dinov3ConvNextConfig = FakeDinov3ConvNextConfig
    fake.Dinov3ConvNextModel = FakeDinov3ConvNextModel
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return fake


class TestWithInjectedSupport:
    def test_catalog_names(self, fake_dinov3_transformers):
        names = DINOV3_SOURCE.catalog_names()
        # 12 full pairs (pretrained + -notpretrained) + the two distilled
        # tiny models, which only resolve as -notpretrained (no public HF
        # checkpoint).
        assert len(names) == 26
        assert names == sorted(names)
        assert "dinov3/vits16" in names
        assert "dinov3/vits16-notpretrained" in names
        assert "dinov3/convnext-tiny" in names
        assert "dinov3/vitl16-sat493m" in names
        assert "dinov3/vitt16-notpretrained" in names
        assert "dinov3/vitt16plus-notpretrained" in names
        assert "dinov3/vitt16" not in names
        assert "dinov3/vitt16plus" not in names

    def test_vitt16_no_public_checkpoint_raises(self, fake_dinov3_transformers):
        with pytest.raises(ValueError, match="vitt16-notpretrained"):
            DINOV3_SOURCE.build("vitt16", load_weights=True)

    def test_vitt16_notpretrained_build(self, fake_dinov3_transformers):
        model = DINOV3_SOURCE.build(
            "vitt16-notpretrained", load_weights=False
        )
        assert isinstance(model, FakeDinov3ViTModel)
        assert model.config.hidden_size == 192
        assert model.config.num_register_tokens == 4
        model = DINOV3_SOURCE.build(
            "vitt16plus-notpretrained", load_weights=False
        )
        assert model.config.hidden_size == 192
        assert model.config.intermediate_size == 1152

    def test_registration_before_custom(self, fake_dinov3_transformers):
        from autovision.encoders.package_helpers import list_sources

        names = [p.name for p in list_sources()]
        assert names.index("dinov3") < names.index("custom")

    def test_vit_notpretrained_build(self, fake_dinov3_transformers):
        model = DINOV3_SOURCE.build(
            "vits16-notpretrained", load_weights=False
        )
        assert isinstance(model, FakeDinov3ViTModel)
        assert model.config.num_register_tokens == 4
        wrapper = DINOV3_SOURCE.as_encoder(model)
        assert isinstance(wrapper, TransformersDinov3ViTWrapper)
        out = wrapper.feature_maps(torch.randn(1, 3, 224, 224))
        assert out["features"].shape == (1, 384, 14, 14)
        assert out["cls_token"].shape == (1, 384)
        pooled = wrapper.pool_features(out)
        assert pooled["pooled_features"].shape == (1, 384, 1, 1)
        assert wrapper.encoder_dim() == 384
        assert wrapper.patch_size() == 16
        assert wrapper.arch_traits() == {
            "model_type": "transformer",
            "norm_type": "layernorm",
        }

    def test_convnext_notpretrained_build(self, fake_dinov3_transformers):
        model = DINOV3_SOURCE.build(
            "convnext-tiny-notpretrained", load_weights=False
        )
        assert isinstance(model, FakeDinov3ConvNextModel)
        wrapper = DINOV3_SOURCE.as_encoder(model)
        assert isinstance(wrapper, TransformersDinov3ConvNextWrapper)
        out = wrapper.feature_maps(torch.randn(1, 3, 224, 224))
        assert out["features"].shape == (1, 768, 14, 14)
        assert wrapper.encoder_dim() == 768
        assert wrapper.arch_traits() == {
            "model_type": "convolutional",
            "norm_type": "layernorm",
        }

    def test_vit_input_channels(self, fake_dinov3_transformers):
        model = DINOV3_SOURCE.build(
            "vits16-notpretrained", num_input_channels=2, load_weights=False
        )
        assert (
            model.embeddings.patch_embeddings.projection.in_channels == 2
        )
        wrapper = DINOV3_SOURCE.as_encoder(model)
        out = wrapper.feature_maps(torch.randn(1, 2, 224, 224))
        assert out["features"].shape == (1, 384, 14, 14)

    def test_convnext_input_channels(self, fake_dinov3_transformers):
        model = DINOV3_SOURCE.build(
            "convnext-tiny-notpretrained", num_input_channels=1, load_weights=False
        )
        assert model.embeddings.patch_embeddings.in_channels == 1
        wrapper = DINOV3_SOURCE.as_encoder(model)
        out = wrapper.feature_maps(torch.randn(1, 1, 224, 224))
        assert out["features"].shape == (1, 768, 14, 14)

    def test_unknown_model(self, fake_dinov3_transformers):
        with pytest.raises(ValueError, match="Unknown dinov3 model"):
            DINOV3_SOURCE.build("vitm16", load_weights=False)

    def test_pretrained_uses_from_pretrained(
        self, fake_dinov3_transformers, tmp_path, monkeypatch
    ):
        # The verified-loading flow (behind the D1-a opt-in): the repo id
        # goes to the download layer, and from_pretrained receives the
        # verified local snapshot dir.
        monkeypatch.setenv("AUTOVISION_DINOV3_WEIGHTS", "1")
        requested = []
        downloaded = []

        class Spy(FakeDinov3ViTModel):
            @classmethod
            def from_pretrained(cls, name, **kwargs):
                requested.append(name)
                return cls(FakeDinov3ViTConfig(hidden_size=768))

        fake_dinov3_transformers.Dinov3ViTModel = Spy
        snapshot = tmp_path / "snap"
        snapshot.mkdir()
        (snapshot / "model.safetensors").write_bytes(b"weights")
        # Force the lfs-metadata verification tier: an unpinned manifest
        # entry, independent of what the real manifest has pinned.
        from autovision.encoders.weight_verification import WeightArtifact

        monkeypatch.setattr(
            "autovision.encoders.weight_verification.get_weight_artifact",
            lambda repo_id: WeightArtifact(repo_id=repo_id, license="apache-2.0"),
        )
        monkeypatch.setattr(
            "autovision.encoders.weight_verification._download_snapshot",
            lambda repo_id, revision, *a, **k: downloaded.append(repo_id)
            or snapshot,
        )
        monkeypatch.setattr(
            "autovision.encoders.weight_verification._lfs_sha256",
            lambda repo_id, revision: {
                "model.safetensors": hashlib.sha256(b"weights").hexdigest()
            },
        )
        DINOV3_SOURCE.build("vitb16")  # pretrained by default
        assert downloaded == ["facebook/dinov3-vitb16-pretrain-lvd1689m"]
        assert requested == [snapshot]

    def test_recognizes(self, fake_dinov3_transformers):
        wrapper = DINOV3_SOURCE.as_encoder(
            DINOV3_SOURCE.build("vits16-notpretrained", load_weights=False)
        )
        assert DINOV3_SOURCE.recognizes(wrapper)
        by_name = get_wrapped_model("dinov3/vits16-notpretrained")
        assert isinstance(by_name, TransformersDinov3ViTWrapper)

    def test_export(self, tmp_path, fake_dinov3_transformers):
        wrapper = DINOV3_SOURCE.as_encoder(
            DINOV3_SOURCE.build("vits16-notpretrained", load_weights=False)
        )
        out = tmp_path / "exported" / "model.pt"
        DINOV3_SOURCE.export_weights(wrapper, out, log_example=False)
        assert out.exists()


# ── D1-a: user-held DINOv3 weights gate ─────────────────────────────────


class TestD1aWeightsGate:
    """Pretrained DINOv3 weights are user-held (design doc 28 §6.1, D1-a):
    nothing is fetched without an explicit per-environment opt-in."""

    def test_pretrained_refused_without_opt_in(
        self, fake_dinov3_transformers, monkeypatch
    ):
        from autovision.encoders.dinov3 import dinov3_source

        monkeypatch.delenv(dinov3_source.DINOV3_WEIGHTS_ENV, raising=False)

        def _must_not_run(*args, **kwargs):
            raise AssertionError("weight download attempted without opt-in")

        monkeypatch.setattr(dinov3_source, "load_pretrained_verified", _must_not_run)
        with pytest.raises(PermissionError, match="AUTOVISION_DINOV3_WEIGHTS"):
            DINOV3_SOURCE.build("vits16", load_weights=True)

    def test_opt_in_routes_to_verified_loader(
        self, fake_dinov3_transformers, monkeypatch
    ):
        from autovision.encoders.dinov3 import dinov3_source

        monkeypatch.setenv(dinov3_source.DINOV3_WEIGHTS_ENV, "1")
        sentinel = object()
        seen: dict[str, str] = {}

        def _fake_load(model_cls, repo_id, config_kwargs):
            seen["repo"] = repo_id
            return sentinel

        monkeypatch.setattr(dinov3_source, "load_pretrained_verified", _fake_load)
        assert DINOV3_SOURCE.build("vits16", load_weights=True) is sentinel
        assert seen["repo"] == "facebook/dinov3-vits16-pretrain-lvd1689m"

    def test_opt_in_flag_parsing(self, monkeypatch):
        from autovision.encoders.dinov3.dinov3_source import (
            DINOV3_WEIGHTS_ENV,
            dinov3_weights_opt_in,
        )

        for value in ("1", "yes", "TRUE", " on "):
            monkeypatch.setenv(DINOV3_WEIGHTS_ENV, value)
            assert dinov3_weights_opt_in(), value
        for value in ("", "0", "no", "false", "  "):
            monkeypatch.setenv(DINOV3_WEIGHTS_ENV, value)
            assert not dinov3_weights_opt_in(), value
        monkeypatch.delenv(DINOV3_WEIGHTS_ENV)
        assert not dinov3_weights_opt_in()

    def test_notpretrained_never_gated(self, fake_dinov3_transformers, monkeypatch):
        from autovision.encoders.dinov3 import dinov3_source

        monkeypatch.delenv(dinov3_source.DINOV3_WEIGHTS_ENV, raising=False)
        model = DINOV3_SOURCE.build("vits16-notpretrained", load_weights=False)
        assert isinstance(model, FakeDinov3ViTModel)
