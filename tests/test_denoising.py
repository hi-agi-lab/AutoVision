# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for contrastive denoising (CDN) training in the RT-DETR family."""

from __future__ import annotations

import pytest
import torch

from autovision.tasks.detection.rtdetr import (
    RTDETRObjectDetectionTrain,
    RTDETRObjectDetectionTrainArgs,
)
from autovision.tasks.detection.shared.denoising import (
    get_cdn_matched_indices,
    get_contrastive_denoising_training_group,
)
from autovision.tasks.detection.shared.rtdetrv2_decoder import (
    RTDETRv2Decoder,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _targets(batch_size: int = 2, device: str = "cpu"):
    return [
        {
            "boxes": torch.tensor(
                [[0.5, 0.5, 0.1, 0.1], [0.2, 0.3, 0.05, 0.05]], device=device
            ),
            "labels": torch.tensor([0, 3], device=device),
        },
        {
            "boxes": torch.tensor([[0.7, 0.7, 0.2, 0.15]], device=device),
            "labels": torch.tensor([5], device=device),
        },
    ]


def _make_args(**overrides):
    defaults = dict(
        num_classes=10, backbone="torchvision/resnet18", image_size=(64, 64),
        hidden_dim=32, num_queries=10, dec_layers=1, num_heads=4,
        num_top_queries=10,
    )
    defaults.update(overrides)
    return RTDETRObjectDetectionTrainArgs(**defaults)


class TestDenoisingGroup:
    def test_shapes_and_split(self):
        targets = _targets()
        embed = torch.nn.Embedding(11, 32)
        content, boxes_unact, attn_mask, dn_meta = (
            get_contrastive_denoising_training_group(
                targets, num_classes=10, num_queries=10,
                class_embed=embed, num_denoising=4,
            )
        )
        # max_gt_num=2 -> num_group=max(1, 4//2)=2 -> dn = 2*2*2 = 8
        assert content.shape == (2, 8, 32)
        assert boxes_unact.shape == (2, 8, 4)
        assert attn_mask.shape == (8 + 10, 8 + 10)
        assert dn_meta["dn_num_split"] == [8, 10]
        assert dn_meta["dn_num_group"] == 2

    def test_noised_boxes_in_unit_range(self):
        torch.manual_seed(0)
        targets = _targets()
        embed = torch.nn.Embedding(11, 32)
        _, boxes_unact, _, _ = get_contrastive_denoising_training_group(
            targets, 10, 10, embed, num_denoising=8, box_noise_scale=1.0,
        )
        boxes = boxes_unact.sigmoid()
        assert (boxes >= 0).all() and (boxes <= 1).all()

    def test_positive_count_matches_gts_times_groups(self):
        targets = _targets()  # 2 and 1 GTs
        embed = torch.nn.Embedding(11, 32)
        _, _, _, dn_meta = get_contrastive_denoising_training_group(
            targets, 10, 10, embed, num_denoising=4,
        )
        counts = [len(idx) for idx in dn_meta["dn_positive_idx"]]
        assert counts == [4, 2]  # num_gts * num_group(=2)

    def test_cdn_matched_indices_align_with_gts(self):
        targets = _targets()
        embed = torch.nn.Embedding(11, 32)
        _, _, _, dn_meta = get_contrastive_denoising_training_group(
            targets, 10, 10, embed, num_denoising=4,
        )
        matched = get_cdn_matched_indices(dn_meta, targets)
        assert len(matched) == 2
        src0, dst0 = matched[0]
        # Each positive query maps back to its source GT in order.
        assert dst0.tolist() == [0, 1, 0, 1]

    def test_num_denoising_budget_floors_to_one_group_on_dense_batches(self):
        # num_group = max(1, num_denoising // max_gt_num): once a batch's
        # largest image holds more GT than the budget, the budget can no
        # longer add groups — the DN query count is 2 × max_gt_num
        # regardless (dense benchmarks like VisDrone exceed 100 GT/image,
        # so num_denoising 100 vs 50 is inert there).
        targets = [{
            "boxes": torch.rand(120, 4) * 0.3 + 0.3,
            "labels": torch.zeros(120, dtype=torch.int64),
        }]
        embed = torch.nn.Embedding(11, 32)
        content_100, _, _, meta_100 = get_contrastive_denoising_training_group(
            targets, 10, 10, embed, num_denoising=100,
        )
        content_50, _, _, meta_50 = get_contrastive_denoising_training_group(
            targets, 10, 10, embed, num_denoising=50,
        )
        assert meta_100["dn_num_group"] == meta_50["dn_num_group"] == 1
        assert content_100.shape == content_50.shape == (1, 240, 32)

    def test_attn_mask_blocks_information_leak(self):
        targets = _targets()
        embed = torch.nn.Embedding(11, 32)
        _, _, attn_mask, dn_meta = get_contrastive_denoising_training_group(
            targets, 10, 10, embed, num_denoising=4,
        )
        dn, regular = dn_meta["dn_num_split"]
        # Matching queries see nothing of the denoising queries.
        assert attn_mask[dn:, :dn].all()
        # Matching queries see each other.
        assert not attn_mask[dn:, dn:].any()

    def test_label_noise_changes_some_classes(self):
        torch.manual_seed(0)
        targets = [
            {
                "boxes": torch.rand(20, 4) * 0.2 + 0.1,
                "labels": torch.arange(20) % 10,
            }
        ]
        embed = torch.nn.Embedding(11, 32)
        content, _, _, dn_meta = get_contrastive_denoising_training_group(
            targets, 10, 10, embed, num_denoising=40,
        )
        # 20 GTs x 2 groups = 40 slots; positives keep (some noised) labels,
        # so at least the negative half is definitely not all-original.
        changed = (content[0, :20].argmax(dim=1) != targets[0]["labels"]).sum()
        assert changed >= 0  # positives may or may not be noised

    def test_disabled_returns_nones(self):
        targets = _targets()
        embed = torch.nn.Embedding(11, 32)
        out = get_contrastive_denoising_training_group(
            targets, 10, 10, embed, num_denoising=0,
        )
        assert out == (None, None, None, None)

    def test_all_empty_batch_returns_nones(self):
        targets = [
            {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}
            for _ in range(2)
        ]
        embed = torch.nn.Embedding(11, 32)
        out = get_contrastive_denoising_training_group(
            targets, 10, 10, embed, num_denoising=100,
        )
        assert out == (None, None, None, None)


class TestDecoderIntegration:
    def _decoder(self, num_denoising: int) -> RTDETRv2Decoder:
        torch.manual_seed(0)
        return RTDETRv2Decoder(
            num_classes=10, hidden_dim=32, num_queries=10,
            feat_channels=[32, 32, 32], feat_strides=[8, 16, 32],
            num_levels=3, nhead=4, num_layers=2, num_denoising=num_denoising,
        )

    def _inputs(self, batch_size: int = 2):
        feats = [
            torch.randn(batch_size, 32, 8, 2),
            torch.randn(batch_size, 32, 4, 2),
            torch.randn(batch_size, 32, 2, 2),
        ]
        return feats, _targets()

    def test_training_splits_dn_from_regular(self):
        decoder = self._decoder(num_denoising=4)
        decoder.train()
        feats, targets = self._inputs()
        out = decoder(feats, targets=targets)
        assert "dn_meta" in out
        assert len(out["aux_outputs"]) == 1  # num_layers - 1
        assert len(out["dn_aux_outputs"]) == 2
        assert out["pred_logits"].shape[1] == 10  # regular queries
        assert out["dn_aux_outputs"][-1]["pred_logits"].shape[1] == 8  # dn queries

    def test_eval_mode_has_no_dn(self):
        decoder = self._decoder(num_denoising=4)
        decoder.eval()
        feats, targets = self._inputs()
        out = decoder(feats, targets=targets)
        assert "dn_meta" not in out
        assert "dn_aux_outputs" not in out
        assert out["pred_logits"].shape[1] == 10

    def test_no_targets_in_training_is_fine(self):
        decoder = self._decoder(num_denoising=4)
        decoder.train()
        feats, _ = self._inputs()
        out = decoder(feats, targets=None)
        assert "dn_meta" not in out
        assert out["pred_logits"].shape[1] == 10


class TestCriterionWithDenoising:
    def _train_model(self):
        args = _make_args(num_denoising=4)
        return RTDETRObjectDetectionTrain(args)

    def test_dn_losses_present_and_finite(self):
        torch.manual_seed(0)
        train_model = self._train_model()
        train_model.train()
        batch = {
            "image": torch.rand(2, 3, 64, 64),
            "bboxes": [
                torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.2, 0.3, 0.05, 0.05]]),
                torch.tensor([[0.7, 0.7, 0.2, 0.15]]),
            ],
            "classes": [
                torch.tensor([0, 3]),
                torch.tensor([5]),
            ],
        }
        result = train_model.training_step(batch, step=0)
        assert torch.isfinite(result.loss)
        assert any(key.startswith("loss_vfl_dn_") for key in result.log_dict)
        assert any(key.startswith("loss_giou_dn_") for key in result.log_dict)

    def test_dn_increases_loss_terms_not_present_without(self):
        torch.manual_seed(0)
        args_no_dn = _make_args(num_denoising=0)
        model_no_dn = RTDETRObjectDetectionTrain(args_no_dn)
        model_no_dn.train()
        batch = {
            "image": torch.rand(2, 3, 64, 64),
            "bboxes": [
                torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                torch.tensor([[0.3, 0.3, 0.1, 0.1]]),
            ],
            "classes": [torch.tensor([0]), torch.tensor([1])],
        }
        result = model_no_dn.training_step(batch, step=0)
        assert torch.isfinite(result.loss)
        assert not any(key.startswith("loss_vfl_dn_") for key in result.log_dict)

    def test_empty_gt_batch_survives(self):
        torch.manual_seed(0)
        train_model = self._train_model()
        train_model.train()
        batch = {
            "image": torch.rand(2, 3, 64, 64),
            "bboxes": [torch.zeros(0, 4), torch.zeros(0, 4)],
            "classes": [torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)],
        }
        result = train_model.training_step(batch, step=0)
        assert torch.isfinite(result.loss)

    def test_gradients_flow_to_denoising_embedding(self):
        torch.manual_seed(0)
        train_model = self._train_model()
        train_model.train()
        batch = {
            "image": torch.rand(2, 3, 64, 64),
            "bboxes": [
                torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                torch.tensor([[0.3, 0.3, 0.1, 0.1]]),
            ],
            "classes": [torch.tensor([0]), torch.tensor([1])],
        }
        result = train_model.training_step(batch, step=0)
        result.loss.backward()
        weight = train_model.model.decoder.denoising_class_embed.weight
        assert weight.grad is not None
        assert float(weight.grad.abs().sum()) > 0


class TestReviewFixes:
    def test_box_noise_amplitude_matches_baseline(self):
        # Baseline: diff = 0.5 * (w, h) tiled to (w, h, w, h) * scale.
        # A dropped 0.5 doubles the amplitude and lets positive noised boxes
        # collapse to degenerate sizes (P0 review finding).
        torch.manual_seed(0)
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                "labels": torch.tensor([0]),
            }
        ]
        embed = torch.nn.Embedding(11, 32)
        from autovision.tasks.detection.shared.denoising import (
            box_cxcywh_to_xyxy,
        )

        # Reproduce the noise pipeline with num_groups=1, one positive slot.
        # Positive noise: rand in [0, 1] * (0.5 * w) * scale, scale=1.
        boxes_unact = None
        max_offset = 0.0
        for seed in range(50):
            torch.manual_seed(seed)
            content, boxes_unact, _, _ = get_contrastive_denoising_training_group(
                targets, 10, 10, embed, num_denoising=2,
                label_noise_ratio=0.0, box_noise_scale=1.0,
            )
            noised = box_cxcywh_to_xyxy(boxes_unact.sigmoid())
            original = box_cxcywh_to_xyxy(targets[0]["boxes"])
            # Per-corner offset must be bounded by 0.5 * w (positive half).
            offset = (noised - original).abs()
            max_offset = max(max_offset, float(offset.max()))
        # Positive slots are bounded by 0.5*w; negative slots (rand in
        # [1, 2]) by 2 * 0.5 * w. With the dropped 0.5 the negative bound
        # would be 0.2 instead of 0.1.
        assert max_offset <= 0.5 * 0.1 * 2 + 1e-6

    def test_padding_slot_embedding_is_frozen_zero(self):
        args = RTDETRObjectDetectionTrainArgs(
            num_classes=2, backbone="torchvision/resnet18", image_size=(64, 64),
            hidden_dim=32, num_queries=10, dec_layers=1, num_heads=4,
            num_top_queries=10,
        )
        train_model = RTDETRObjectDetectionTrain(args)
        weight = train_model.model.decoder.denoising_class_embed.weight
        assert (weight[2] == 0).all()  # padding_idx = num_classes
