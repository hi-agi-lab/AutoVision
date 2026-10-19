# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR semantic-segmentation family (roadmap design doc 31 §11 S1).

Gold parity (2026-10-16, RTX 4090D): official ``GTRSemSeg`` (their
engine, their semseg-S config) vs this family, same official EMA
weights, same inputs — logits **bit-exact (maxdiff 0.0)** on random 512,
a real COCO image at 512 and at 1024; argmax agreement 1.0. The in-repo
tests keep the CPU-loadable half (key/shape/coverage) plus a tiny
CUDA-gated training closure.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tests.test_tasks_classification_segmentation import _write_segmentation_tree

from autovision.registry.gtr_assets import (
    GTR_SEMSEG_VARIANTS,
    resolve_gtr_asset,
)

_fla = True
try:
    from autovision.tasks.detection.shared.gtr import gtr_available as _gtr_available

    _fla = _gtr_available()
except Exception:  # noqa: BLE001 - fla missing entirely counts as unavailable
    _fla = False

requires_gla = pytest.mark.skipif(
    not (_fla and torch.cuda.is_available()),
    reason="GTR GLA forward needs flash-linear-attention + CUDA",
)


def _semseg_checkpoint() -> Path:
    try:
        return resolve_gtr_asset("gtrsemseg_s_cityscapes").path
    except Exception:  # noqa: BLE001 - absent local weights keep fresh checkouts green
        pytest.skip("local GTR semseg weights not present")


@pytest.fixture(scope="module")
def semseg_checkpoint() -> Path:
    return _semseg_checkpoint()


class TestGtrSemSegFamily:
    def test_variant_table_matches_official_widths(self):
        # configs/semseg/cityscapes_finetune/gtrsemseg_{s,m,l,x}.yml
        # (embed_dim, num_heads, ffn_ratio, encoder/head width): L/X keep
        # a 256-wide encoder projection under their 384-wide ViT.
        assert GTR_SEMSEG_VARIANTS == {
            "s": (192, 3, 4.0, 192),
            "m": (256, 4, 4.0, 256),
            "l": (384, 6, 4.0, 256),
            "x": (384, 6, 6.0, 256),
        }

    def test_official_checkpoint_keys_match(self, semseg_checkpoint):
        """Every ``encoder.*``/``head.*`` tensor the official checkpoint
        carries must exist in our family net with identical shape (the
        S1 load-correctness contract; the backbone half goes through
        ``extract_gtr_backbone``)."""
        from autovision.tasks.detection.detr.gtr_teacher import (
            extract_gtr_backbone,
            load_gtr_teacher,
        )
        from autovision.tasks.semantic_segmentation.gtr import (
            GTRSemanticSegmentation,
        )

        net = GTRSemanticSegmentation(
            num_classes=19, variant="s", image_size=(64, 64), load_checkpoint=False
        )
        target = net.state_dict()
        bundle = load_gtr_teacher(semseg_checkpoint)
        enc_head = {
            key: tensor
            for key, tensor in bundle.state.items()
            if key.startswith(("encoder.", "head."))
        }
        assert len(enc_head) == 40
        for key, tensor in enc_head.items():
            assert key in target, key
            assert tensor.shape == target[key].shape, key
        backbone_state = extract_gtr_backbone(bundle)
        assert len(backbone_state) > 100
        # Strict load proves the extracted layout matches the wrapper.
        incompatible = net.backbone.backbone_model.load_state_dict(
            backbone_state, strict=True
        )
        assert not incompatible.missing_keys

    def test_official_load_reports_full_coverage(self, semseg_checkpoint):
        from autovision.tasks.semantic_segmentation.gtr import (
            GTRSemanticSegmentation,
        )

        net = GTRSemanticSegmentation(
            num_classes=19,
            variant="s",
            image_size=(64, 64),
            checkpoint="gtrsemseg_s_cityscapes",  # registered asset (sha256 verified)
        )
        assert net.load_report is not None
        assert net.load_report["encoder_head_coverage"] == 1.0
        assert net.load_report["skipped_shape"] == 0
        assert net.load_report["unexpected"] == 0
        assert net.load_report["weights_source"] == "ema.module"

    @pytest.mark.parametrize("variant", ["s", "m", "l", "x"])
    def test_all_variants_load_official_checkpoints(self, variant):
        """Per-variant load contract (m/l/x gold-checked 2026-10-16):
        L/X keep a 256-wide encoder/head under the 384-wide ViT — getting
        that wrong is shape-visible at the coverage gate, so this test
        locks every published width into CI."""
        from autovision.tasks.semantic_segmentation.gtr import (
            GTRSemanticSegmentation,
        )

        try:
            resolve_gtr_asset(f"gtrsemseg_{variant}_cityscapes")
        except Exception:  # noqa: BLE001 - absent weights keep fresh checkouts green
            pytest.skip("local GTR semseg weights not present")
        net = GTRSemanticSegmentation(
            num_classes=19, variant=variant, image_size=(64, 64),
            checkpoint=f"gtrsemseg_{variant}_cityscapes",
        )
        assert net.load_report["encoder_head_coverage"] == 1.0
        assert net.load_report["skipped_shape"] == 0
        assert net.load_report["unexpected"] == 0
        embed_dim = {"s": 192, "m": 256, "l": 384, "x": 384}[variant]
        hidden = 192 if variant == "s" else 256
        assert net.backbone.embed_dim == embed_dim
        assert net.encoder.hidden_dim == hidden
        assert net.head.conv.in_channels == hidden

    def test_wrong_variant_checkpoint_refused(self, semseg_checkpoint):
        """An S checkpoint into an L-width net must fail with a curated
        message (first-block shape mismatch), not torch's raw spew."""
        from autovision.tasks.semantic_segmentation.gtr import (
            GTRSemanticSegmentation,
        )

        with pytest.raises(ValueError, match="does not match variant"):
            GTRSemanticSegmentation(
                num_classes=19, variant="l", image_size=(64, 64),
                checkpoint="gtrsemseg_s_cityscapes",
            )

    def test_dispatch_rejects_unknown_variant(self, tmp_path):
        from autovision import train_semantic_segmentation

        images = tmp_path / "images"
        masks = tmp_path / "masks"
        _write_segmentation_tree(images, masks)
        with pytest.raises(Exception, match="variant"):
            train_semantic_segmentation(
                out=tmp_path / "run",
                data={"train": {"images": images, "masks": masks}},
                model="gtr/zzz",
                num_classes=3,
                steps=2,
                batch_size=2,
                num_workers=0,
                model_args={"image_size": (32, 32), "load_checkpoint": False},
            )

    @requires_gla
    def test_training_closure(self, tmp_path: Path):
        """End-to-end closure on synthetic pairs: gtr/s dispatch, forward,
        CE loss, mIoU val — a handful of steps is enough to prove the
        family is trainable through the shared engine."""
        from autovision import train_semantic_segmentation

        images = tmp_path / "images"
        masks = tmp_path / "masks"
        _write_segmentation_tree(images, masks)
        result = train_semantic_segmentation(
            out=tmp_path / "run",
            data={"train": {"images": images, "masks": masks}},
            model="gtr/s",
            num_classes=3,
            steps=4,
            batch_size=2,
            num_workers=0,
            model_args={
                "image_size": (64, 64),
                "load_checkpoint": False,
                "lr_warmup_steps": 2,
            },
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.exported_model_path is not None
        assert result.exported_model_path.exists()
