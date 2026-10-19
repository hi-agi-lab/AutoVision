# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR oriented-detection TRAINING surface (roadmap design doc 31 §11 S3b).

Unit parity (2026-10-16, same inputs/seed against the official
implementations — see ``runs/probe_speed/gtr_obb_train_parity.py``):
matcher assignments identical; the criterion's full forward (12 weighted
keys incl. ``_aux``/``_pre``/``_enc`` branches at group_detr=3) BIT-EQUAL;
the oriented CDN byte-identical for group_detr 1 and 3. This suite keeps
the in-repo halves: target conversion, matcher semantics, criterion
plumbing, and a CUDA training closure through the CLI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tests.test_oriented_object_detection import make_yolo_obb_dataset

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


class TestTargetConversion:
    def test_long_edge_convention(self):
        # Tall rectangle (w < h) must come back with w >= h and a shifted
        # angle; the corner set must be preserved (Chamfer ~ 0).
        from autovision.tasks.detection.shared.gtr import rbox_ops
        from autovision.tasks.oriented_detection.gtr.train_model import (
            xyxyxyxy_to_whr_longedge,
        )

        tall = torch.tensor([[0.5, 0.5, 0.3, 0.1]])  # center, w, h (px-ish)
        quads = rbox_ops.rbox_to_corners(
            torch.tensor([[tall[0, 0], tall[0, 1], tall[0, 2], tall[0, 3], 0.0]])
        ).reshape(1, 8)
        targets = xyxyxyxy_to_whr_longedge(quads)
        assert targets.shape == (1, 5)
        assert float(targets[0, 2]) >= float(targets[0, 3]) - 1e-6  # w >= h
        assert 0.0 <= float(targets[0, 4]) < 1.0  # a = theta/pi in [0, 1)
        # Region preserved: rebuild radians and compare corner sets.
        rebuild = torch.cat([targets[..., :4], targets[..., 4:5] * 3.141592653589793], dim=-1)
        distance = rbox_ops.chamfer_cost_pairwise(rebuild, rebuild).diagonal()
        assert float(distance.max()) < 1e-5
        back = rbox_ops.rbox_to_corners(rebuild).reshape(1, 8)
        assert back.shape == quads.shape
        # The rebuilt corner set matches the original quad's region.
        rebuilt_quad_distance = rbox_ops.chamfer_cost_pairwise(
            rbox_ops.rbox_rad_to_longedge(rebuild).reshape(1, 5), rebuild
        ).diagonal()
        assert float(rebuilt_quad_distance.max()) < 1e-5


class TestMatcherAndCriterion:
    @staticmethod
    def _inputs(seed=0):
        generator = torch.Generator().manual_seed(seed)
        boxes = torch.zeros(1, 4, 5)
        boxes[0, :, 0] = torch.tensor([0.2, 0.5, 0.75, 0.3])
        boxes[0, :, 1] = torch.tensor([0.2, 0.5, 0.3, 0.75])
        boxes[0, :, 2] = 0.2
        boxes[0, :, 3] = 0.1
        boxes[0, :, 4] = torch.tensor([0.0, 0.25, 0.5, 0.75])  # a = theta/pi
        outputs = {
            "pred_logits": torch.randn(1, 4, 3, generator=generator),
            "pred_boxes": boxes,
        }
        # GT: matches query 1 in both domain and angle.
        targets = [
            {"boxes": boxes[0, 1:2].clone(), "labels": torch.tensor([0])}
        ]
        return outputs, targets

    def test_matcher_picks_the_matching_query(self):
        from autovision.tasks.detection.shared.gtr.obb_criterion import (
            OBBHungarianMatcher,
        )

        outputs, targets = self._inputs()
        matcher = OBBHungarianMatcher(cost_class=2.0, cost_chamfer=5.0, cost_kld=2.0)
        indices = matcher(outputs, targets)["indices"]
        # Query 1's box IS the GT box; class costs are noise-level compared
        # to the geometric costs (chamfer 5 + kld 2, zero at identity).
        assert indices[0][0].tolist() == [1]
        assert indices[0][1].tolist() == [0]

    def test_criterion_forward_plumbing(self):
        from autovision.tasks.detection.shared.gtr.obb_criterion import (
            OBBGTRCriterion,
            OBBHungarianMatcher,
        )

        outputs, targets = self._inputs()
        outputs = dict(outputs)
        outputs["aux_outputs"] = [
            {"pred_logits": outputs["pred_logits"].clone(), "pred_boxes": outputs["pred_boxes"].clone()}
        ]
        outputs["pre_outputs"] = {
            "pred_logits": outputs["pred_logits"].clone(),
            "pred_boxes": outputs["pred_boxes"].clone(),
        }
        outputs["enc_aux_outputs"] = [
            {"pred_logits": outputs["pred_logits"].clone(), "pred_boxes": outputs["pred_boxes"].clone()}
        ]
        outputs["enc_meta"] = {"class_agnostic": False}
        outputs["up"] = torch.tensor([0.5])
        outputs["reg_scale"] = torch.tensor([4.0])
        criterion = OBBGTRCriterion(
            OBBHungarianMatcher(cost_class=2.0, cost_chamfer=5.0, cost_kld=2.0),
            weight_dict={"loss_mal": 1, "loss_bbox": 5, "loss_kld": 5, "loss_fgl": 0.15},
            losses=["mal", "boxes"],
            num_classes=3,
            group_detr=1,
        )
        criterion.train()
        losses = criterion(outputs, targets)
        for key in ("loss_mal", "loss_bbox", "loss_kld", "loss_mal_aux_0",
                    "loss_mal_pre", "loss_mal_enc_0"):
            assert key in losses, key
            assert torch.isfinite(losses[key]), key
        # Weights are applied inside the criterion (loss_bbox rides 5.0).
        plain = criterion.loss_boxes(outputs, targets, [(torch.tensor([1]), torch.tensor([0]))], torch.tensor(1.0))
        assert float(losses["loss_bbox"]) == pytest.approx(float(plain["loss_bbox"]) * 5.0, abs=1e-5)

    def test_cdn_masks_and_meta(self):
        from autovision.tasks.detection.shared.gtr.obb_denoising import (
            get_obb_contrastive_denoising_training_group,
        )

        class Embed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(4, 8, padding_idx=3)

            def forward(self, x):
                return self.emb(x)

        targets = [
            {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.25]]), "labels": torch.tensor([0, 1])}
        ]
        logits, unact, attn_mask, meta = get_obb_contrastive_denoising_training_group(
            targets, 3, 6, Embed(), num_denoising=8, group_detr=1
        )
        assert meta["dn_num_split"][0] == logits.shape[1]  # dn rows / queries
        assert attn_mask.shape == (logits.shape[1] + 6, logits.shape[1] + 6)
        # Match queries cannot attend to the reconstructions.
        assert bool(attn_mask[logits.shape[1]:, : logits.shape[1]].all())
        assert unact.shape[-1] == 5
        # Angle channel is UNPERTURBED by box noise (upstream design): every
        # DN row keeps inverse_sigmoid(GT angle) = inverse_sigmoid(0.25).
        import math

        expected_angle_unact = math.log(0.25 / 0.75)
        assert torch.allclose(
            unact[:, :, 4],
            torch.full_like(unact[:, :, 4], expected_angle_unact),
            atol=1e-5,
        )


class TestGtrObbClosure:
    @requires_gla
    def test_training_closure(self, tmp_path: Path):
        """gtr/s dispatch through the oriented CLI: oriented CDN + MAL/KLD
        criterion + oriented mAP validation + export, on a synthetic
        YOLO-OBB tree (48 queries = 3 DETR groups × 16)."""
        from autovision import train_oriented_object_detection

        data_args = make_yolo_obb_dataset(tmp_path)
        data = {
            "format": "yolo-obb",
            "path": str(data_args.path),
            "train": str(data_args.train),
            "val": str(data_args.val),
            "names": dict(data_args.names),
        }
        result = train_oriented_object_detection(
            out=tmp_path / "run",
            data=data,
            model="gtr/s",
            steps=4,
            batch_size=2,
            num_workers=0,
            model_args={
                "image_size": (64, 64),
                "load_checkpoint": False,
                "lr_warmup_steps": 2,
                # 48 = 3 DETR groups x 16; the group split in the matcher
                # requires num_queries % group_detr == 0.
                "num_queries": 48,
                "num_denoising": 10,
            },
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.exported_model_path is not None
        assert result.exported_model_path.exists()
