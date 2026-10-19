# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""P1 spectrum: HGNetV2 width recipes, tier registry (D10 metadata),
D-FINE assembly -> AVWarmStart, warm-start consumption, model cards,
offline hub bridge.

The dfine_s-dependent tests auto-skip when the cached Apache checkpoint
is absent (fresh checkouts without the weight cache stay green).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autovision.data.coco8 import ensure_coco8
from autovision.registry.model_card import (
    LatencyRecord,
    ModelCard,
    ModelCardError,
    load_model_card,
)
from autovision.tasks.detection.detr.backbone_wrappers import TimmCNNBackbone
from autovision.tasks.detection.detr.variants import (
    DETR_MODEL_REGISTRY,
    SPECTRUM,
    detr_default_batch_size,
    list_detr_model_names,
    resolve_detr_variant,
    resolve_spectrum_tier,
)

DFINE_S_CANDIDATES = (
    Path.home() / ".cache/autovision/d-fine_dfine_s_coco.pth",
    Path.home() / ".cache/autovision/downloadable_models/d-fine/dfine_s_coco.pt",
)

TINY_TRANSFORM_ARGS = {
    "scale_jitter": None,
    "photometric_distort_prob": 0.5,
    "random_zoom_out_prob": None,
    "random_iou_crop_prob": None,
}


@pytest.fixture(scope="module")
def dfine_s_checkpoint() -> Path:
    for candidate in DFINE_S_CANDIDATES:
        if candidate.exists():
            return candidate
    pytest.skip("d-fine/dfine_s_coco not in the local weight cache")


@pytest.fixture(scope="module")
def coco8(tmp_path_factory):
    return ensure_coco8(tmp_path_factory.mktemp("coco8"))


@pytest.fixture(scope="module")
def warm_start_asset(tmp_path_factory, dfine_s_checkpoint) -> Path:
    from autovision.tasks.detection.detr.dfine_official_warm_start import (
        build_dfine_official_warm_start,
    )

    out = tmp_path_factory.mktemp("ws") / "detr_s_hgnetv2b1_dfine_s.pt"
    report = build_dfine_official_warm_start(
        "hgnetv2-b1",
        out=out,
        checkpoint=dfine_s_checkpoint,
        backbone_pretrained=False,
    )
    assert report["decoder_coverage"] >= 0.99
    return out


class TestSpectrumRegistry:
    def test_n_and_s_are_spec_frozen_with_hgnetv2_recipes(self):
        tier_n = resolve_spectrum_tier("n")
        tier_s = resolve_spectrum_tier("s")
        assert tier_n.status == "spec-frozen" and tier_n.route == "distill"
        assert tier_s.status == "spec-frozen" and tier_s.route == "assembly-finetune"
        _, spec_n = resolve_detr_variant("detr/n")
        _, spec_s = resolve_detr_variant("detr/s")
        for spec in (spec_n, spec_s):
            assert spec.family == "timm_cnn"
            assert spec.backbone_package == "timm"
            assert spec.hidden_dim == 256  # dfine_s-assembly width
            assert spec.dec_layers == 3
            assert spec.patch_size == 32
        assert spec_n.backbone_name == "hgnetv2_b0"
        assert spec_s.backbone_name == "hgnetv2_b1"

    def test_status_vocabulary_is_tri_state(self):
        statuses = {entry.status for entry in SPECTRUM.values()}
        assert statuses <= {"planned", "spec-frozen", "released"}
        assert "frozen" not in statuses

    def test_spectrum_keys_are_public_model_names(self):
        names = set(list_detr_model_names())
        assert {"detr/n", "detr/s", "detr/hgnetv2-b0", "detr/hgnetv2-b1"} <= names

    def test_planned_tiers_declare_metadata_only(self):
        for tier in ("m", "l", "x"):
            entry = resolve_spectrum_tier(tier)
            assert entry.status == "planned"
            assert entry.variant_key == ""  # D10: no component, no promise
            assert entry.tier not in DETR_MODEL_REGISTRY

    def test_auto_batch_table_covers_spectrum(self):
        assert detr_default_batch_size("n") == 32
        assert detr_default_batch_size("s") == 32

    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError, match="Unknown spectrum tier"):
            resolve_spectrum_tier("z")

    def test_d10_metadata_in_spec_frozen_notes(self):
        assert "D1b" in SPECTRUM["n"].note
        assert "dfine_s" in SPECTRUM["s"].note


class TestTimmCNNBackbone:
    def test_pyramid_shapes_offline(self):
        backbone = TimmCNNBackbone("hgnetv2_b0", pretrained=False)
        assert len(backbone.in_channels) == 3
        features = backbone(torch.randn(1, 3, 64, 64))
        assert len(features) == 3
        for feature, channels in zip(features, backbone.in_channels):
            assert feature.shape[1] == channels
        # strides 8/16/32 at 64px input
        assert [f.shape[-1] for f in features] == [8, 4, 2]

    def test_rejects_non_three_pyramid(self):
        with pytest.raises(ValueError, match="num_stages must be 3"):
            TimmCNNBackbone("hgnetv2_b0", pretrained=False, num_stages=4)


class TestAssembly:
    def test_real_dfine_s_assembly(self, warm_start_asset):
        from autovision.registry.warm_start import load_warm_start

        payload = load_warm_start(warm_start_asset)
        assert payload.license.id == "Apache-2.0"
        assert payload.spec["decoder_coverage"] >= 0.99
        assert len(payload.model) >= 168  # LEDGER-verified direct-match count

    def test_missing_checkpoint_refused(self, tmp_path):
        from autovision.tasks.detection.detr.dfine_official_warm_start import (
            WarmStartAssemblyError,
            build_dfine_official_warm_start,
        )

        with pytest.raises(WarmStartAssemblyError, match="not found"):
            build_dfine_official_warm_start(
                "hgnetv2-b1", out=tmp_path / "x.pt",
                checkpoint=tmp_path / "missing.pth",
                backbone_pretrained=False,
            )


class TestWarmStartConsumption:
    def test_load_into_network(self, warm_start_asset):
        from autovision.tasks.detection.detr.task_model import (
            DETRObjectDetection,
            load_warm_start_into_network,
        )

        network = DETRObjectDetection(
            num_classes=80,
            model_name="hgnetv2-b1",
            image_size=(640, 640),
            load_weights=False,
            decoder="dfine_official",
        )
        report = load_warm_start_into_network(network, warm_start_asset)
        assert report["decoder_coverage"] == 1.0
        assert report["transferred"] >= 168

    def test_license_is_hard_requirement(self, warm_start_asset, tmp_path):
        from autovision.registry.warm_start import load_warm_start
        from autovision.tasks.detection.detr.task_model import (
            DETRObjectDetection,
            load_warm_start_into_network,
        )

        payload = load_warm_start(warm_start_asset)
        tampered = tmp_path / "unlicensed.pt"
        torch.save(
            {
                "format": "av_warm_start_v1",
                "model": payload.model,
                "spec": payload.spec,
                # license stripped
            },
            tampered,
        )
        network = DETRObjectDetection(
            num_classes=80,
            model_name="hgnetv2-b1",
            image_size=(640, 640),
            load_weights=False,
            decoder="dfine_official",
        )
        with pytest.raises(ValueError, match="license"):
            load_warm_start_into_network(network, tampered)


class TestModelCard:
    def _card(self, **overrides):
        fields = dict(
            model="detr/s",
            name="DETR-S (HGNetV2-B1 + D-FINE-S decoder)",
            train_dataset="COCO 2017 train2017",
            schedule="6ep assembly finetune",
            seed=0,
            metrics={"coco_map": 0.0},
            license="Apache-2.0",
            weights_sha256={"detr_s.pt": "a" * 64},
        )
        fields.update(overrides)
        return ModelCard(**fields)

    def test_valid_card_roundtrip(self, tmp_path):
        card = self._card(
            latency=[
                LatencyRecord(
                    backend="tensorrt", precision="fp16",
                    hardware="NVIDIA T4", batch_size=1, p50_ms=1.2,
                    script="scripts/train_spectrum.py",
                )
            ]
        )
        card.save(tmp_path / "detr_s.pt")
        loaded = load_model_card(tmp_path / "detr_s.pt")
        assert loaded.license == "Apache-2.0"
        assert loaded.latency[0].precision == "fp16"

    def test_license_required(self):
        with pytest.raises(ModelCardError, match="license is required"):
            self._card(license="").validate()

    def test_sha256_format_enforced(self):
        with pytest.raises(ModelCardError, match="64-hex"):
            self._card(weights_sha256={"w.pt": "deadbeef"}).validate()

    def test_latency_precision_whitelist(self):
        card = self._card(
            latency=[
                LatencyRecord(
                    backend="tensorrt", precision="bf16",
                    hardware="T4", batch_size=1, p50_ms=1.0, script="s",
                )
            ]
        )
        with pytest.raises(ModelCardError, match="fp32/fp16/int8"):
            card.validate()


class TestOfflineBridge:
    def test_offline_maps_to_hf_hub_offline(self, monkeypatch):
        from autovision.encoders.weight_verification import bridge_hub_env

        monkeypatch.setenv("AUTOVISION_OFFLINE", "1")
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        bridge_hub_env()
        import os

        assert os.environ["HF_HUB_OFFLINE"] == "1"

    def test_no_offline_no_touch(self, monkeypatch):
        import os

        from autovision.encoders.weight_verification import bridge_hub_env

        monkeypatch.setenv("AUTOVISION_OFFLINE", "0")
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        bridge_hub_env()
        assert "HF_HUB_OFFLINE" not in os.environ


class TestSpectrumTrainClosure:
    """detr/{n,s} 加载闭环: tiny coco8 train -> artifacts (roadmap J1)."""

    def test_train_detr_n(self, coco8, tmp_path):
        from autovision import train_object_detection

        result = train_object_detection(
            out=tmp_path / "run_n",
            data=coco8,
            model="detr/n",
            steps=3,
            batch_size=2,
            num_workers=0,
            model_args={
                "image_size": (64, 64),
                "num_queries": 50,
                "num_top_queries": 50,
                "num_denoising": 10,
            },
            transform_args=TINY_TRANSFORM_ARGS,
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.steps_run == 3
        assert result.exported_model_path and result.exported_model_path.exists()

    def test_train_detr_s_with_warm_start(
        self, coco8, warm_start_asset, tmp_path
    ):
        from autovision import train_object_detection

        result = train_object_detection(
            out=tmp_path / "run_s",
            data=coco8,
            model="detr/s",
            steps=3,
            batch_size=2,
            num_workers=0,
            model_args={
                "image_size": (64, 64),
                "num_queries": 50,
                "num_top_queries": 50,
                "num_denoising": 10,
                "warm_start": warm_start_asset,
            },
            transform_args=TINY_TRANSFORM_ARGS,
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.steps_run == 3
        assert result.exported_model_path and result.exported_model_path.exists()
