# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR detection variant coverage: ViT M/L/X specs, 365-class assembly,
and the removed timm ``gtr-*`` targets.

The GTR ViT M/L/X checkpoints share a 256-wide encoder/decoder stack but
keep 256/384/384-wide ViTs; ``backbone_embed_dim`` separates the widths.
The det-obj365 checkpoints ship 365 dataset classes + a no-object slot, so
the assembly target must use ``GTRAsset.model_num_classes`` (366).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autovision.registry.gtr_assets import GTR_ASSETS, resolve_gtr_asset
from autovision.tasks.detection.detr import (
    DETRObjectDetectionTrain,
    DETRObjectDetectionTrainArgs,
)
from autovision.tasks.detection.detr.dfine_official_warm_start import (
    build_dfine_official_warm_start,
)
from autovision.tasks.detection.detr.variants import resolve_detr_variant

_fla = True
try:
    from autovision.tasks.detection.shared.gtr import gtr_available as _gtr_available

    _fla = _gtr_available()
except Exception:  # noqa: BLE001 - fla missing entirely counts as unavailable
    _fla = False

requires_gla = pytest.mark.skipif(
    not (_fla and torch.cuda.is_available()),
    reason="GTR GLA assembly needs flash-linear-attention + CUDA",
)


def _spec(name: str):
    _, spec = resolve_detr_variant(name)
    return spec


class TestVariantSpecs:
    def test_gtrvit_m_l_x_specs(self):
        rows = (
            ("m", 256, 4.0, 1024),
            ("l", 384, 4.0, 1024),
            ("x", 384, 6.0, 2048),
        )
        for name, embed, ffn, dec_ffn in rows:
            spec = _spec(f"gtrvit-{name}")
            assert spec.family == "gtr"
            assert spec.hidden_dim == 256  # encoder + decoder
            assert spec.backbone_embed_dim == embed
            assert spec.backbone_ffn_ratio == ffn
            assert spec.dec_dim_feedforward == dec_ffn
            assert spec.dec_layers == 4
            assert spec.dec_num_points == (3, 6, 3)

    def test_obj365_model_num_classes(self):
        assert GTR_ASSETS["gtr_s_obj365"].num_classes == 365
        assert GTR_ASSETS["gtr_s_obj365"].model_num_classes == 366
        assert GTR_ASSETS["gtr_m_obj365"].model_num_classes == 366
        assert GTR_ASSETS["gtr_s_coco"].model_num_classes == 80

    @pytest.mark.parametrize(
        "name", ["gtr-s", "gtr-m", "gtr-l", "gtr-x", "detr/gtr-s", "detr/gtr-m", "detr/gtr-l", "detr/gtr-x"]
    )
    def test_removed_or_unknown_gtr_variants(self, name):
        # The timm decoder-only ``gtr-*`` targets are removed; only the
        # full GTR ViT ``gtrvit-*`` route remains.
        with pytest.raises(ValueError):
            resolve_detr_variant(name)


class TestAssembly:
    @requires_gla
    def test_gtrvit_m_assembly_loads(self, tmp_path: Path):
        try:
            asset = resolve_gtr_asset("gtr_m_coco")
        except Exception:  # noqa: BLE001 - store absent
            pytest.skip("local GTR det weights not present")
        from autovision.tasks.detection.detr.gtr_teacher import (
            extract_gtr_backbone,
            load_gtr_teacher,
        )

        ws = tmp_path / "ws_gtrvit_m.pt"
        bb = tmp_path / "gtr_m_backbone.pth"
        report = build_dfine_official_warm_start(
            "gtrvit-m",
            out=ws,
            checkpoint=asset.path,
            backbone_pretrained=False,
            license_id="MIT",
            origin=asset.source,
            source_format="gtr",
            num_classes=80,
        )
        assert report["decoder_coverage"] == 1.0
        assert report["skipped_shape"] == 0
        torch.save(extract_gtr_backbone(load_gtr_teacher(asset.path)), bb)
        args = DETRObjectDetectionTrainArgs(
            backbone="gtrvit-m",
            num_classes=80,
            warm_start=ws,
            backbone_weights=bb,
            backbone_pretrained=False,
            image_size=(640, 640),
        )
        train = DETRObjectDetectionTrain(args)
        assert train.model.variant_spec.backbone_embed_dim == 256

    @requires_gla
    def test_obj365_s_assembly_uses_366_classes(self, tmp_path: Path):
        try:
            asset = resolve_gtr_asset("gtr_s_obj365")
        except Exception:  # noqa: BLE001 - store absent
            pytest.skip("local GTR obj365 weights not present")
        ws = tmp_path / "ws_gtrvit_s_obj365.pt"
        report = build_dfine_official_warm_start(
            "gtrvit-s",
            out=ws,
            checkpoint=asset.path,
            backbone_pretrained=False,
            license_id="MIT",
            origin=asset.source,
            source_format="gtr",
            num_classes=asset.model_num_classes,
        )
        assert report["decoder_coverage"] == 1.0
        assert report["skipped_shape"] == 0
