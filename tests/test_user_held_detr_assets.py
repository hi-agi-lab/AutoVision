# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""User-held DINOv3 detector assets: registry ↔ store consistency (D1-a).

Registry shape and build-contract tests run everywhere. Store-dependent
tests require both a local store and ``AUTOVISION_USER_HELD_TESTS=1``;
they are private-asset integration checks, not part of the default release
suite. The full ~4 GB sha sweep uses the same opt-in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from autovision.registry.user_held_assets import (
    USER_HELD_ASSETS,
    USER_HELD_LOADABLE_ASSETS,
    USER_HELD_ROOT,
    USER_HELD_UNSUPPORTED_ASSETS,
    UserHeldAssetError,
    resolve_avwarm_start_source,
    resolve_user_held_asset,
    user_held_model_args,
    verify_user_held_asset,
)

_MANIFEST = USER_HELD_ROOT / "manifest.json"
if not _MANIFEST.is_file():
    # Task-split store layout: the detector category lives in detection/.
    _MANIFEST = USER_HELD_ROOT / "detection" / "manifest.json"
_store_present = pytest.mark.skipif(
    not _MANIFEST.is_file(), reason="user-held store not present"
)

_CONVNEXT_NAMES = (
    "dinov3-convnext-tiny",
    "dinov3-convnext-small",
    "dinov3-convnext-base",
    "dinov3-convnext-large",
)
_VIT_NAMES = ("dinov3-vitt16", "dinov3-vitt16plus", "dinov3-vits16")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and set(value) <= set("0123456789abcdef")


class TestRegistryShape:
    def test_seven_assets_with_pinned_identity(self):
        assert sorted(USER_HELD_ASSETS) == sorted(_VIT_NAMES + _CONVNEXT_NAMES)
        assert sorted(USER_HELD_LOADABLE_ASSETS) == sorted(_CONVNEXT_NAMES)
        assert sorted(USER_HELD_UNSUPPORTED_ASSETS) == sorted(_VIT_NAMES)
        assert not set(USER_HELD_LOADABLE_ASSETS) & set(USER_HELD_UNSUPPORTED_ASSETS)
        for name, asset in USER_HELD_ASSETS.items():
            assert asset.name == name
            assert asset.model_name.startswith("detr/")
            for digest in (
                asset.source_sha256,
                asset.weights_sha256,
                asset.warm_start_sha256,
            ):
                assert _is_sha256(digest), (name, digest)
            assert asset.num_classes == 80
            assert asset.image_size == (640, 640)
            assert asset.license_id == "DINOv3-License+NOASSERTION"
            assert asset.redistributable is False
            assert asset.origin.endswith(".pt")
            assert asset.requires_removed_sta is (name in _VIT_NAMES), name

    def test_model_names_resolve_in_detr_registry(self):
        from autovision.tasks.detection.detr.variants import resolve_detr_variant

        for name, asset in USER_HELD_ASSETS.items():
            key, _spec = resolve_detr_variant(asset.model_name)
            # ConvNeXt store names double as the registry key; the ViT
            # entries load into the ``-notpretrained`` (offline) variants.
            assert key in (name, f"{name}-notpretrained")

    def test_convnext_requires_six_layer_256_decoder(self):
        """The published ConvNeXt checkpoints carry dec_layers=6 at the
        fixed 256 decoder width — NOT the variant spec defaults (3/4
        layers, 384 width), so the overrides are part of the contract."""
        for name in _CONVNEXT_NAMES:
            asset = USER_HELD_ASSETS[name]
            assert asset.dec_layers == 6, name
            assert asset.decoder_hidden_dim == 256, name
        for name in _VIT_NAMES:
            asset = USER_HELD_ASSETS[name]
            assert asset.dec_layers is None, name
            assert asset.decoder_hidden_dim is None, name

    def test_required_model_args_shape(self):
        args = user_held_model_args("dinov3-convnext-base")
        assert args["dec_layers"] == 6
        assert args["decoder_hidden_dim"] == 256
        assert args["backbone_pretrained"] is False
        assert args["image_size"] == (640, 640)
        # The registered name, not a path: resolution happens at build
        # time against the (overridable) store root.
        assert args["detector_checkpoint"] == "dinov3-convnext-base"

        for name in _VIT_NAMES:
            with pytest.raises(UserHeldAssetError, match="spatial prior"):
                user_held_model_args(name)

    def test_unknown_name_error_lists_registered(self):
        with pytest.raises(UserHeldAssetError) as excinfo:
            resolve_user_held_asset("dinov3-nope")
        message = str(excinfo.value)
        assert "dinov3-vitt16" in message
        assert "D1-a" in message


class TestStaRemovalGuards:
    """The DEIMv2-derived spatial prior was removed for licensing; every
    entry point that would load an STA-era ViT checkpoint refuses it."""

    def test_registry_entry_points_reject_vit_assets(self):
        from autovision.registry.downloadable import DownloadableModelError
        from autovision.tasks.detection.detr.coco_checkpoint import (
            resolve_detector_checkpoint,
        )

        for name in _VIT_NAMES:
            with pytest.raises(UserHeldAssetError, match="spatial prior"):
                resolve_user_held_asset(name)
            with pytest.raises(UserHeldAssetError, match="spatial prior"):
                user_held_model_args(name)
            with pytest.raises(UserHeldAssetError, match="spatial prior"):
                resolve_avwarm_start_source(name)
            with pytest.raises(DownloadableModelError, match="spatial prior"):
                resolve_detector_checkpoint(name)

    def test_convnext_assets_are_not_blocked(self):
        for name in _CONVNEXT_NAMES:
            assert user_held_model_args(name)["detector_checkpoint"] == name
            assert USER_HELD_ASSETS[name].requires_removed_sta is False
            assert USER_HELD_ASSETS[name].unsupported_reason is None

    def test_resolve_rejects_sta_before_verification(self, monkeypatch):
        """Fail-fast: no 40 MB–920 MB hash runs for an unsupported asset."""
        import autovision.registry.user_held_assets as held

        calls: list[object] = []
        monkeypatch.setattr(
            held,
            "verify_user_held_asset",
            lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        for name in _VIT_NAMES:
            with pytest.raises(UserHeldAssetError, match="spatial prior"):
                resolve_user_held_asset(name, verify=True)
        assert calls == []

    def test_allow_unsupported_is_audit_only(self):
        for name in _VIT_NAMES:
            asset = resolve_user_held_asset(name, allow_unsupported=True)
            assert asset.name == name
            assert asset.unsupported_reason is not None

    def test_unsupported_reason_names_convnext_alternatives(self):
        for name in _VIT_NAMES:
            reason = USER_HELD_ASSETS[name].unsupported_reason
            assert reason is not None
            assert "backbone.sta.*" in reason
            assert "spatial prior" in reason
            for alternative in _CONVNEXT_NAMES:
                assert alternative in reason

    def test_detector_checkpoint_unknown_error_separates_unsupported(self):
        from autovision.registry.downloadable import DownloadableModelError
        from autovision.tasks.detection.detr.coco_checkpoint import (
            resolve_detector_checkpoint,
        )

        with pytest.raises(DownloadableModelError) as excinfo:
            resolve_detector_checkpoint("dinov3-nope")
        message = str(excinfo.value)
        assert "loadable names" in message
        assert "registered but unsupported" in message
        for name in _CONVNEXT_NAMES + _VIT_NAMES:
            assert name in message

    def test_baseline_loader_rejects_sta_tensors(self, tmp_path):
        from autovision.tasks.detection.detr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
        )

        path = tmp_path / "sta_era.pt"
        torch.save(
            {
                "train_model": {
                    "ema_model.model.backbone.dinov3.patch_embed.proj.weight":
                        torch.zeros(1),
                    "ema_model.model.backbone.sta.stem.0.weight": torch.zeros(1),
                }
            },
            path,
        )
        # Fails before touching the target model, with the licensing reason.
        with pytest.raises(ValueError, match="spatial prior"):
            load_baseline_coco_checkpoint(object(), path)

    def test_baseline_loader_sta_guard_precedes_partition(self, tmp_path):
        from autovision.tasks.detection.detr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
        )

        path = tmp_path / "sta_only.pt"
        torch.save(
            {
                "train_model": {
                    "ema_model.model.backbone.sta.stem.0.weight": torch.zeros(1),
                }
            },
            path,
        )
        # A partition-first implementation would report "no recognizable
        # raw-backbone keys"; the licensing error must win.
        with pytest.raises(ValueError, match="spatial prior"):
            load_baseline_coco_checkpoint(object(), path)

    def test_warm_start_loader_rejects_sta_tensors(self, tmp_path):
        from autovision.registry.assets import LicenseRef
        from autovision.registry.warm_start import save_warm_start
        from autovision.tasks.detection.detr.task_model import (
            load_warm_start_into_network,
        )

        path = save_warm_start(
            tmp_path / "sta_warm_start.pt",
            {"backbone.sta.stem.0.weight": torch.zeros(1)},
            {},
            LicenseRef(id="test", origin="unit test"),
        )
        # No network access is needed: the STA guard precedes the transfer.
        with pytest.raises(ValueError, match="spatial prior"):
            load_warm_start_into_network(object(), path)


class TestConfigAcceptsNames:
    def test_registered_names_pass_strict_validation(self):
        """``detector_checkpoint`` / ``warm_start`` accept registered names
        (strings) next to Path — same convention as the GTR families'
        ``checkpoint`` field. Resolution happens at build time."""
        from autovision.tasks.detection.detr import DETRObjectDetectionTrainArgs

        args = DETRObjectDetectionTrainArgs(
            backbone="dinov3-vitt16-notpretrained",
            num_classes=80,
            detector_checkpoint="dinov3-vitt16",
            warm_start="dinov3-vitt16",
        )
        assert args.detector_checkpoint == "dinov3-vitt16"
        assert args.warm_start == "dinov3-vitt16"


@pytest.mark.skipif(
    os.environ.get("AUTOVISION_USER_HELD_TESTS") != "1",
    reason="set AUTOVISION_USER_HELD_TESTS=1 to run private-store integration tests",
)
class TestStoreConsistency:
    """Store-dependent: opt-in on machines that hold the private store."""

    @_store_present
    def test_manifest_matches_registry(self):
        manifest = json.loads(_MANIFEST.read_text())
        by_name = {a["variant"]: a for a in manifest["assets"]}
        assert set(by_name) == set(USER_HELD_ASSETS)
        for name, entry in by_name.items():
            asset = USER_HELD_ASSETS[name]
            assert entry["dir"] == asset.name
            assert entry["model_name"] == asset.model_name
            assert entry["files"]["weights.pt"] == asset.weights_sha256
            assert entry["files"]["warm_start.pt"] == asset.warm_start_sha256
            assert entry["license"]["redistributable"] is False

    @_store_present
    def test_vitt16_weights_sha256_spot_check(self):
        verify_user_held_asset(
            resolve_user_held_asset("dinov3-vitt16", allow_unsupported=True),
            files=("weights.pt",),
        )

    @_store_present
    def test_resolve_detector_checkpoint_accepts_registered_name(self):
        from autovision.tasks.detection.detr.coco_checkpoint import (
            resolve_detector_checkpoint,
        )

        name = "dinov3-convnext-tiny"
        resolved = resolve_detector_checkpoint(name)
        assert resolved == USER_HELD_ASSETS[name].weights_path
        assert resolved.is_file()

    @_store_present
    def test_warm_start_name_resolution(self):
        name = "dinov3-convnext-tiny"
        resolved = resolve_avwarm_start_source(name)
        assert resolved == USER_HELD_ASSETS[name].warm_start_path
        assert resolved.is_file()

    def test_warm_start_unknown_name_passes_through(self):
        passthrough = resolve_avwarm_start_source("some/other/path.pt")
        assert passthrough == Path("some/other/path.pt")

    @_store_present
    def test_convnext_real_load_matches_recorded_summary(self):
        """End-to-end: build through the train model (name resolution +
        load) and match the summary the packing session recorded."""
        from autovision.tasks.detection.detr import (
            DETRObjectDetectionTrain,
            DETRObjectDetectionTrainArgs,
        )
        from autovision.tasks.detection.detr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
            resolve_detector_checkpoint,
        )

        name = "dinov3-convnext-tiny"
        args = DETRObjectDetectionTrainArgs(
            backbone=name,
            num_classes=80,
            **user_held_model_args(name),
        )
        model = DETRObjectDetectionTrain(args)
        recorded = json.loads(
            (USER_HELD_ASSETS[name].directory / "ASSET.json").read_text()
        )["loader_summary"]
        path = resolve_detector_checkpoint(args.detector_checkpoint)
        actual = load_baseline_coco_checkpoint(model.model, path)
        for key, value in recorded.items():
            assert actual[key] == value, key

    @_store_present
    def test_convnext_without_overrides_is_rejected(self):
        """The store's ConvNeXt checkpoints carry dec_layers=6; building at
        the variant defaults must fail loudly rather than mis-load."""
        from autovision.tasks.detection.detr import (
            DETRObjectDetectionTrain,
            DETRObjectDetectionTrainArgs,
        )

        args = DETRObjectDetectionTrainArgs(
            backbone="dinov3-convnext-tiny",
            num_classes=80,
            detector_checkpoint="dinov3-convnext-tiny",
            backbone_pretrained=False,
            image_size=(640, 640),
        )
        with pytest.raises(ValueError):
            DETRObjectDetectionTrain(args)

    @_store_present
    def test_convnext_warm_start_name_loads_into_network(self):
        from autovision.tasks.detection.detr import (
            DETRObjectDetectionTrain,
            DETRObjectDetectionTrainArgs,
        )
        from autovision.tasks.detection.detr.task_model import (
            load_warm_start_into_network,
        )

        name = "dinov3-convnext-tiny"
        args = DETRObjectDetectionTrainArgs(
            backbone=name, num_classes=80, **user_held_model_args(name)
        )
        model = DETRObjectDetectionTrain(args)
        summary = load_warm_start_into_network(model.model, name)
        assert summary["transferred"] > 0

    @pytest.mark.skipif(
        os.environ.get("AUTOVISION_USER_HELD_TESTS") != "1",
        reason="set AUTOVISION_USER_HELD_TESTS=1 to hash all store containers",
    )
    def test_full_sha256_verification_all_assets(self):
        for asset in USER_HELD_ASSETS.values():
            verify_user_held_asset(asset)


def test_verify_user_held_asset_detects_tampering_and_caches(tmp_path, monkeypatch):
    import hashlib

    import autovision.registry.user_held_assets as held

    held_dir = tmp_path / "unit-test"
    held_dir.mkdir()
    weights = held_dir / "weights.pt"
    weights.write_bytes(b"real-weights")
    good = hashlib.sha256(b"real-weights").hexdigest()
    monkeypatch.setattr(held, "USER_HELD_ROOT", tmp_path)
    asset = held.UserHeldAsset(
        name="unit-test",
        model_name="unit/test",
        origin="unit-test",
        source_sha256=good,
        weights_sha256=good,
        warm_start_sha256=good,
        license_id="DINOv3 License",
        redistributable=False,
    )
    held.verify_user_held_asset(asset, files=("weights.pt",))  # cached success
    weights.write_bytes(b"tampered")
    with pytest.raises(held.UserHeldAssetError, match="sha256 mismatch"):
        held.verify_user_held_asset(asset, files=("weights.pt",))
