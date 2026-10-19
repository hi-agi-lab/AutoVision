# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""G2: GTR GLA-backbone family (roadmap design doc 31).

Forward tests need BOTH the ``gtr`` extra (flash-linear-attention) and a
CUDA device (Triton GLA kernels) — anything less skips, so CPU-only
checkouts stay green while the family remains fully exercised on the
dev box.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from autovision.data.coco8 import ensure_coco8
from autovision.tasks.detection.detr.variants import resolve_detr_variant

_TINY_TRANSFORM_ARGS = {
    "scale_jitter": None,
    "photometric_distort_prob": 0.5,
    "random_zoom_out_prob": None,
    "random_iou_crop_prob": None,
}

_fla = importlib.util.find_spec("fla") is not None
_cuda = torch.cuda.is_available()
requires_gla = pytest.mark.skipif(
    not (_fla and _cuda),
    reason="GTR GLA forward needs flash-linear-attention + CUDA",
)

def _gtr_s_checkpoint() -> Path:
    # Single source of truth for the user-held location: the registry's
    # resolvable root (AUTOVISION_GTR_ROOT / known locations).
    from autovision.registry.gtr_assets import GTR_ASSETS

    path = GTR_ASSETS["gtr_s_coco"].path
    if not path.exists():
        pytest.skip("local GTR weights not present")
    return path


class TestGtrVariantRegistry:
    def test_gtrvit_s_spec(self):
        _, spec = resolve_detr_variant("detr/gtrvit-s")
        assert spec.family == "gtr"
        assert spec.backbone_package == "gtr"
        assert spec.hidden_dim == 192
        assert spec.interaction_indexes == (3, 7, 11)
        assert spec.patch_size == 16
        assert spec.decoder_default == "dfine_official"

class TestGtrFamilyForward:
    @requires_gla
    def test_backbone_pyramid_shapes(self):
        from autovision.tasks.detection.detr.backbone_wrappers import GTRViTBackbone

        backbone = GTRViTBackbone(embed_dim=192, num_heads=3).cuda().eval()
        assert backbone.in_channels == [192, 192, 192]
        with torch.no_grad():
            feats = backbone(torch.randn(1, 3, 64, 64, device="cuda"))
        assert [tuple(f.shape) for f in feats] == [
            (1, 192, 4, 4),
            (1, 192, 4, 4),
            (1, 192, 4, 4),
        ]

    @requires_gla
    def test_full_network_forward_and_predict(self):
        from autovision.tasks.detection.detr.task_model import DETRObjectDetection

        net = DETRObjectDetection(
            num_classes=2,
            model_name="gtrvit-s",
            image_size=(64, 64),
            load_weights=False,
            decoder="dfine_official",
            num_queries=50,
            num_top_queries=50,
            num_denoising=10,
        ).cuda().eval()
        with torch.no_grad():
            out = net.forward_backend(
                net.preprocess_batch(torch.randn(1, 3, 64, 64, device="cuda"))
            )
        logits, boxes = out if isinstance(out, tuple) else (out.logits, out.boxes)
        assert logits.shape[:2] == (1, 50)
        assert boxes.shape == (1, 50, 4)

    @requires_gla
    def test_extracted_backbone_weights_load(self, tmp_path):
        """GTR->GTR backbone transfer: extract_gtr_backbone keys load
        verbatim into the family backbone (``backbone_model`` contract)."""
        from autovision.tasks.detection.detr.backbone_wrappers import GTRViTBackbone
        from autovision.tasks.detection.detr.gtr_teacher import (
            extract_gtr_backbone,
            load_gtr_teacher,
        )

        bundle = load_gtr_teacher(_gtr_s_checkpoint())
        state = extract_gtr_backbone(bundle)
        torch.save(state, tmp_path / "gtr_backbone.pt")

        backbone = GTRViTBackbone(embed_dim=192, num_heads=3).cuda().eval()
        incompatible = backbone.backbone_model.load_state_dict(
            torch.load(tmp_path / "gtr_backbone.pt", weights_only=True),
            strict=False,
        )
        assert not incompatible.unexpected_keys

    @requires_gla
    def test_train_closure_coco8(self, tmp_path):
        """detr/gtrvit-s 加载闭环: coco8 3-step train on CUDA."""
        from autovision import train_object_detection

        result = train_object_detection(
            out=tmp_path / "run_gtrvit_s",
            data=ensure_coco8(tmp_path / "coco8"),
            model="detr/gtrvit-s",
            steps=3,
            batch_size=2,
            num_workers=0,
            model_args={
                "image_size": (64, 64),
                "num_queries": 50,
                "num_top_queries": 50,
                "num_denoising": 10,
            },
            transform_args=_TINY_TRANSFORM_ARGS,
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.steps_run == 3
        assert result.exported_model_path and result.exported_model_path.exists()
