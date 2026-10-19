# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR integration (roadmap design doc 31): G0 teacher foundation +
GTR decoder-lineage checks. The former timm decoder-only GTR route is removed.

Checkpoint-dependent tests auto-skip when the local GTR weights are
absent (trees without the user-held store stay green).
"""

from __future__ import annotations

import pytest
import torch

from autovision.errors import AutoVisionError
from autovision.registry.gtr_assets import (
    GTR_ASSETS,
    resolve_gtr_asset,
    verify_gtr_asset,
)


def _gtr_s_checkpoint():
    try:
        return resolve_gtr_asset("gtr_s_coco").path
    except AutoVisionError:
        pytest.skip("local GTR weights not present")


@pytest.fixture(scope="module")
def gtr_s_checkpoint():
    return _gtr_s_checkpoint()


class TestGtrAssets:
    def test_registry_shape(self):
        """All GTR task groups are registered (doc 31 §11): det + obj365
        pretrain + seg/pose/depth/obb/semseg, one MIT sha256-pinned entry
        per published checkpoint."""
        from autovision.registry.gtr_assets import gtr_assets_for_task

        assert set(gtr_assets_for_task("det")) == {
            "gtr_s_coco", "gtr_m_coco", "gtr_l_coco", "gtr_x_coco",
        }
        assert set(gtr_assets_for_task("det-obj365")) == {
            "gtr_s_obj365", "gtr_m_obj365", "gtr_l_obj365", "gtr_x_obj365",
        }
        assert set(gtr_assets_for_task("seg")) == {
            "gtrseg_s_coco", "gtrseg_m_coco", "gtrseg_l_coco", "gtrseg_x_coco",
        }
        assert set(gtr_assets_for_task("pose")) == {
            "gtrpose_s_coco", "gtrpose_m_coco", "gtrpose_l_coco", "gtrpose_x_coco",
        }
        assert set(gtr_assets_for_task("depth")) == {
            "gtrdepth_s", "gtrdepth_m", "gtrdepth_l", "gtrdepth_x",
        }
        assert set(gtr_assets_for_task("obb")) == {"gtrobb_s_dota", "gtrobb_x_dota"}
        assert set(gtr_assets_for_task("semseg")) == {
            "gtrsemseg_s_cityscapes", "gtrsemseg_m_cityscapes",
            "gtrsemseg_l_cityscapes", "gtrsemseg_x_cityscapes",
        }
        assert len(GTR_ASSETS) == 26
        for asset in GTR_ASSETS.values():
            assert asset.license == "MIT"
            assert asset.format == "gtr_ckpt_v1"
            assert len(asset.sha256) == 64
            assert asset.path.exists(), asset.name
        for asset in gtr_assets_for_task("det").values():
            assert asset.coco_ap is not None and asset.coco_ap > 50
        # Per-task benchmark口径 comes from the README tables.
        assert gtr_assets_for_task("semseg")["gtrsemseg_s_cityscapes"].metric_value == 81.5
        assert gtr_assets_for_task("seg")["gtrseg_s_coco"].metric_name == "coco_mask_ap"
        assert gtr_assets_for_task("pose")["gtrpose_s_coco"].num_keypoints == 17
        assert gtr_assets_for_task("obb")["gtrobb_s_dota"].metric_name == "dota_ap50"
        assert gtr_assets_for_task("depth")["gtrdepth_s"].metric_name == "nyu_absrel"
        # The obj365 pretrain stage publishes no benchmark row.
        assert gtr_assets_for_task("det-obj365")["gtr_s_obj365"].metric_value is None

    def test_unknown_asset_raises(self):
        with pytest.raises(AutoVisionError, match="Unknown GTR asset"):
            resolve_gtr_asset("gtr_zzz", verify=False)

    def test_sha256_pin_matches_real_file(self, gtr_s_checkpoint):
        verify_gtr_asset(resolve_gtr_asset("gtr_s_coco"))  # must not raise

    def test_missing_file_refused(self, tmp_path):
        from autovision.registry.gtr_assets import GTRAsset

        ghost = GTRAsset(
            name="ghost", path=tmp_path / "none.pth", sha256="0" * 64
        )
        with pytest.raises(AutoVisionError, match="not found"):
            verify_gtr_asset(ghost)


class TestGtrTeacherBundle:
    def test_ema_semantics_and_inventory(self, gtr_s_checkpoint):
        from autovision.tasks.detection.detr.gtr_teacher import load_gtr_teacher

        bundle = load_gtr_teacher(
            gtr_s_checkpoint,
            asset=resolve_gtr_asset("gtr_s_coco"),
        )
        assert bundle.use_ema is True
        assert bundle.license == "MIT"
        assert bundle.inventory["decoder_hidden"] == 192
        assert bundle.inventory["class_embed_rows"] == 81
        assert bundle.last_epoch == 29
        # EMA weights must differ from the raw model tail.
        raw = torch.load(gtr_s_checkpoint, map_location="cpu", weights_only=True)
        probe = "decoder.decoder.layers.3.self_attn.in_proj_weight"
        assert not torch.equal(bundle.state[probe], raw["model"][probe])

    def test_model_source_fallback(self, gtr_s_checkpoint):
        from autovision.tasks.detection.detr.gtr_teacher import load_gtr_teacher

        bundle = load_gtr_teacher(gtr_s_checkpoint, use_ema=False)
        assert bundle.use_ema is False
        assert bundle.inventory["weight_source"] == "model"

    def test_backbone_extraction_strips_prefix(self, gtr_s_checkpoint):
        from autovision.tasks.detection.detr.gtr_teacher import (
            extract_gtr_backbone,
            load_gtr_teacher,
        )

        bundle = load_gtr_teacher(gtr_s_checkpoint)
        backbone = extract_gtr_backbone(bundle)
        assert backbone
        assert all(not k.startswith("backbone.") for k in backbone)
        assert any(k.startswith("patch_embed") for k in backbone)

    def test_forward_contract_refuses(self, gtr_s_checkpoint):
        from autovision.tasks.detection.detr.gtr_teacher import (
            GTRBackendUnavailableError,
            load_gtr_teacher,
            teacher_forward,
        )

        bundle = load_gtr_teacher(gtr_s_checkpoint)
        with pytest.raises(GTRBackendUnavailableError, match="G2"):
            teacher_forward(bundle, images=[])
        assert bundle.decoder_decoder_state  # state-dict handoff works


class TestGtrDecoderLineage:
    """Gold-check (2026-10-16): GTR freezes the query positional embedding
    across decoder layers (GTR-main ``decoder.py``:543-544: computed once
    from the initial references; their deploy ``pos_qk`` fusion relies on
    it being constant). D-FINE recomputes it per layer (upstream
    ``zoo/dfine/dfine_decoder.py``:406). Evaluating a GTR checkpoint under
    the per-layer variant cost ~8.5 COCO AP (45.07 vs 53.60)."""

    @staticmethod
    def _decoders(gtr_compat: bool):
        from autovision.tasks.detection.shared.dfine_official_decoder import (
            DFINEOfficialDecoder,
        )

        decoder = DFINEOfficialDecoder(
            num_classes=2,
            hidden_dim=32,
            num_queries=10,
            feat_channels=[32, 32, 32],
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            num_points=[3, 6, 3],
            activation="silu" if gtr_compat else "relu",
            gtr_compat=gtr_compat,
            num_groups=3,
            eval_spatial_size=(64, 64),
            reg_max=8,
        )
        return decoder, decoder.decoder

    @staticmethod
    def _query_pos_per_layer(decoder, inner):
        seen: list[object] = []
        handles = [
            layer.register_forward_pre_hook(lambda module, args: seen.append(args[5]))
            for layer in inner.layers
        ]
        try:
            shapes = torch.tensor([[8, 8], [4, 4], [2, 2]])
            memory = torch.randn(1, 8 * 8 + 4 * 4 + 2 * 2, inner.hidden_dim)
            with torch.no_grad():
                inner(
                    torch.randn(1, 10, inner.hidden_dim),
                    torch.randn(1, 10, 4),
                    memory,
                    shapes,
                    decoder.dec_bbox_head,
                    decoder.dec_score_head,
                    decoder.query_pos_head,
                    decoder.pre_bbox_head,
                    decoder.integral,
                    decoder.up,
                    decoder.reg_scale,
                )
        finally:
            for handle in handles:
                handle.remove()
        return seen

    def test_gtr_lineage_freezes_query_pos(self):
        decoder, inner = self._decoders(gtr_compat=True)
        assert inner.frozen_query_pos is True
        seen = self._query_pos_per_layer(decoder, inner)
        assert len(seen) == len(inner.layers)
        # One shared embedding object across every layer.
        assert len({id(query_pos) for query_pos in seen}) == 1

    def test_dfine_lineage_recomputes_query_pos(self):
        decoder, inner = self._decoders(gtr_compat=False)
        assert inner.frozen_query_pos is False
        seen = self._query_pos_per_layer(decoder, inner)
        assert len(seen) == len(inner.layers)
        # Distinct per-layer embeddings (references update between layers).
        assert len({id(query_pos) for query_pos in seen}) > 1


class TestTeacherModelCard:
    def test_teacher_card_generates(self, tmp_path):
        from autovision.registry.model_card import ModelCard

        asset = resolve_gtr_asset("gtr_s_coco", verify=False)
        card = ModelCard(
            model="teacher/gtr-s",
            name="GTR-S COCO detection teacher (Intellindust GTR)",
            train_dataset="COCO 2017 train2017 (upstream 30ep recipe)",
            schedule="upstream frozen checkpoint (EMA weights)",
            seed=0,
            license=asset.license,
            weights_sha256={asset.path.name: asset.sha256},
            provenance={"source": asset.source, "format": asset.format},
            known_limitations="Teacher-only until doc 31 G2 wires the GLA "
            "inference backend.",
        )
        path = card.save(tmp_path / "teacher.pt")
        assert path.exists()
