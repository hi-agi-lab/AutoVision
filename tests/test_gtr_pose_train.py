# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR pose TRAINING surface (roadmap design doc 31 §11 S5b).

Unit parity (2026-10-16, same inputs/seed vs the official code — see
``runs/probe_speed/gtr_pose_train_parity.py``): matcher assignments
identical; VFL / keypoint-L1 / OKS loss values BIT-EQUAL; the pose CDN
byte-identical (label embedding, pose un-activation, attn mask, meta).
This suite keeps the in-repo halves: OKS semantics, matcher, CDN
invariants, and a CUDA training closure with the mean-OKS validation.
"""

from __future__ import annotations

import pytest
import torch

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

K = 17


def _targets(batch_size=2, persons=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(batch_size):
        keypoints = torch.cat(
            [
                torch.rand(persons, 2 * K, generator=gen) * 0.5 + 0.25,
                (torch.rand(persons, K, generator=gen) > 0.2).float(),
            ],
            dim=1,
        )
        boxes = torch.cat(
            [
                torch.rand(persons, 2, generator=gen) * 0.5 + 0.25,
                torch.rand(persons, 2, generator=gen) * 0.3 + 0.1,
            ],
            dim=1,
        )
        out.append(
            {
                "keypoints": keypoints,
                "area": torch.rand(persons, generator=gen) * 0.2 + 0.05,
                "boxes": boxes,
                "labels": torch.zeros(persons, dtype=torch.int64),
            }
        )
    return out


class TestOksAndMatcher:
    def test_oks_perfect_and_degraded(self):
        from autovision.tasks.detection.shared.gtr.pose_criterion import (
            _sigmas_for,
            oks_overlaps,
        )

        gt = torch.rand(2, 2 * K)
        vis = torch.ones(2, K)
        area = torch.tensor([0.2, 0.3])
        perfect = oks_overlaps(gt, gt, vis, area, _sigmas_for(K))
        assert torch.allclose(perfect, torch.ones(2), atol=1e-5)
        shifted = oks_overlaps(gt + 0.05, gt, vis, area, _sigmas_for(K))
        assert bool((shifted < perfect).all())
        assert bool((shifted > 0).all())
        # Invisible keypoints contribute nothing: zeroing visibility makes
        # the OKS of a fully-wrong prediction collapse accordingly.
        vis_zero = torch.zeros(2, K)
        degenerate = oks_overlaps(gt + 0.5, gt, vis_zero, area, _sigmas_for(K))
        assert torch.allclose(degenerate, torch.zeros(2), atol=1e-5)

    def test_matcher_prefers_exact_keypoints(self):
        from autovision.tasks.detection.shared.gtr.pose_criterion import (
            GTRPoseHungarianMatcher,
        )

        targets = _targets(batch_size=1, persons=1)
        exact = targets[0]["keypoints"][:, : 2 * K].clone()
        outputs = {
            "pred_logits": torch.zeros(1, 3, 1),
            "pred_keypoints": torch.cat(
                [exact + 0.4, exact + 0.3, exact], dim=0
            ).unsqueeze(0),
        }
        matcher = GTRPoseHungarianMatcher(cost_class=2.0, cost_keypoints=10.0, cost_oks=4.0)
        indices = matcher(outputs, targets)
        assert indices[0][0].tolist() == [2]  # the query that IS the GT


class TestPoseCdn:
    def test_cdn_shapes_masks_and_non_viz(self):
        from autovision.tasks.detection.shared.gtr.pose_dn import prepare_for_cdn

        targets = _targets(batch_size=2, persons=3)
        torch.manual_seed(0)
        label_enc = torch.nn.Embedding(81, 8)
        pose_enc = torch.nn.Embedding(K, 8)
        labels, poses, mask, meta = prepare_for_cdn(
            (targets, 20, 0.5), True, 6, 80, K, 8, label_enc, pose_enc, (64, 64), torch.device("cpu")
        )
        pad_size = meta["pad_size"]
        assert labels.shape == (2, pad_size, K + 1, 8)  # concat-ready with tgt_pose
        assert poses.shape == (2, pad_size, K + 1, 2)
        assert mask.shape == (pad_size + 6, pad_size + 6)
        # Match queries cannot see the reconstructions.
        assert bool(mask[pad_size:, :pad_size].all())
        # Keys with zero visibility keep their center-anchored pose (the
        # noise radius is zeroed for non-viz keypoints upstream) — the
        # center slot is always "visible" and gets noise.
        assert torch.isfinite(poses).all()


class TestGtrPoseClosure:
    @requires_gla
    def test_training_closure_with_oks_validation(self):
        """3 manual steps through GTRPoseEstimationTrain on a synthetic
        batch (trained on GPU): CDN + VFL/keypoint losses + mean-OKS val."""
        from autovision.tasks.pose_estimation.gtr import (
            GTRPoseEstimationTrain,
            GTRPoseEstimationTrainArgs,
        )

        args = GTRPoseEstimationTrainArgs(
            num_classes=2,
            variant="s",
            image_size=(64, 64),
            num_queries=60,
            num_decoder_layers=3,
            dim_feedforward=512,
            checkpoint=None,
            load_checkpoint=False,
            lr_warmup_steps=2,
        )
        train_model = GTRPoseEstimationTrain(args).cuda()
        optimizer, scheduler = train_model.get_optimizer(total_steps=3, global_batch_size=2)

        batch = {
            "image": torch.randn(2, 3, 64, 64).cuda(),
            "keypoints": [t["keypoints"] for t in _targets()],
            "bboxes": [t["boxes"] for t in _targets()],
            "classes": [t["labels"] for t in _targets()],
            "area": [t["area"] for t in _targets()],
        }
        before = next(train_model.model.parameters()).detach().clone()
        losses = []
        train_model.train()
        for step in range(3):
            result = train_model.training_step(batch, step)
            assert torch.isfinite(result.loss), result.log_dict
            losses.append(float(result.loss.detach()))
            result.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        after = next(train_model.model.parameters()).detach().clone()
        assert not torch.equal(before, after)  # parameters actually moved

        train_model.eval()
        val = train_model.validation_step(batch, step=3)
        assert torch.isfinite(val.loss)
        metrics = train_model.compute_metrics("val")
        oks = metrics["val_metric/oks"]
        assert 0.0 <= oks <= 1.0
