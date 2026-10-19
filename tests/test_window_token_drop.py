# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for in-backbone background token dropping (design doc 17).

Core invariants (mirroring the reference rf-detr-ssl-v2
``test_window_token_drop.py``, adapted to the HF global-attention ViT):

- keeping EVERY window reproduces the dense forward bit-exactly on every
  tap (keep-all ≡ dense);
- a partial keep zero-fills exactly the dropped windows' feature regions
  while the kept region stays finite;
- a batch of per-image masks equals the per-image forwards concatenated;
- the default contract (``token_drop=None``) and the val/export paths run
  dense, untouched.

Selection semantics reuse the box-aware eraser: box-overlapping windows
are never dropped, the background subset is sampled to the exact
``keep_ratio`` budget.
"""

import pytest
import torch
import transformers

from autovision.augment.detection.box_aware_eraser import (
    protected_cell_mask,
    sample_erase_mask,
)
from autovision.encoders.dinov2.dinov2_wrapper import TransformersDinov2Wrapper
from autovision.encoders.dinov3.dinov3_wrapper import TransformersDinov3ViTWrapper
from autovision.tasks.detection.detr import (
    DETRObjectDetectionTrainArgs,
    DETRTokenDropArgs,
)
from autovision.tasks.detection.detr.task_model import (
    DETRObjectDetection,
    _dinov3_supported,
)
from autovision.tasks.detection.detr.train_model import (
    DETRObjectDetectionTrain,
    _build_window_keep_masks,
)

TAPS = (0, 1)


def make_wrapper() -> TransformersDinov2Wrapper:
    """Tiny offline DINOv2: 56px canvas, patch 14 → 4x4 patches, 28px
    windows → a 2x2 window grid (2x2 patches per window)."""
    config = transformers.Dinov2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=56,
        patch_size=14,
        num_register_tokens=0,
    )
    return TransformersDinov2Wrapper(transformers.Dinov2Model(config)).eval()


def make_task_model(token_drop: dict | None = None) -> DETRObjectDetection:
    """vittest task model on a 112px canvas: 8x8 patches, 28px windows →
    a 4x4 window grid."""
    model = DETRObjectDetection(
        num_classes=4,
        model_name="dinov2-vittest-notpretrained",
        image_size=(112, 112),
        num_queries=10,
        num_top_queries=10,
        token_drop=token_drop,
    )
    return model.eval()


def make_dinov3_wrapper() -> TransformersDinov3ViTWrapper:
    """Tiny offline DINOv3 ViT: same geometry as ``make_wrapper`` (56px
    canvas, patch 14 → 4x4 patches, 2x2 windows) plus 2 register tokens so
    the CLS+registers prefix formula is exercised. Eval mode: the default
    rope augmentation (``pos_embed_rescale``) only runs in train mode, and
    the bitwise invariants below require the dense path's rope draws
    (design doc 20 §4, difference V1)."""
    config = transformers.DINOv3ViTConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_register_tokens=2,
        image_size=56,
        patch_size=14,
        intermediate_size=64,
    )
    return TransformersDinov3ViTWrapper(
        transformers.DINOv3ViTModel(config)
    ).eval()


class TestWrapperMaskedForward:
    def test_keep_all_matches_dense_bit_exact(self):
        torch.manual_seed(0)
        wrapper = make_wrapper()
        x = torch.randn(2, 3, 56, 56)
        dense = wrapper.stage_feature_maps(x, TAPS)
        all_keep = torch.ones(2, 2, dtype=torch.bool)
        masked = wrapper.stage_feature_maps_masked(
            x, TAPS, [all_keep, all_keep]
        )
        for tap_dense, tap_masked in zip(dense, masked):
            assert torch.equal(tap_dense["features"], tap_masked["features"])
            assert torch.equal(tap_dense["cls_token"], tap_masked["cls_token"])

    def test_partial_keep_zero_fills_dropped_windows(self):
        torch.manual_seed(0)
        wrapper = make_wrapper()
        x = torch.randn(1, 3, 56, 56)
        keep = torch.zeros(2, 2, dtype=torch.bool)
        keep[0, 0] = True  # top-left window only: patches rows/cols 0-1
        (features,) = [
            tap["features"]
            for tap in wrapper.stage_feature_maps_masked(
                x, TAPS[:1], [keep]
            )
        ]
        assert tuple(features.shape) == (1, 32, 4, 4)
        assert features[:, :, :2, 2:].abs().sum() == 0, "right half not zero"
        assert features[:, :, 2:, :].abs().sum() == 0, "bottom half not zero"
        kept_region = features[0, :, :2, :2]
        assert kept_region.abs().sum() > 0, "kept window carries no features"
        assert torch.isfinite(kept_region).all()

    def test_batch_masks_equal_per_image_concat(self):
        torch.manual_seed(0)
        wrapper = make_wrapper()
        x = torch.randn(2, 3, 56, 56)
        keep_a = torch.tensor([[True, True], [False, True]])
        keep_b = torch.tensor([[True, False], [True, True]])
        batched = wrapper.stage_feature_maps_masked(
            x, TAPS, [keep_a, keep_b]
        )
        single_a = wrapper.stage_feature_maps_masked(x[:1], TAPS, [keep_a])
        single_b = wrapper.stage_feature_maps_masked(x[1:2], TAPS, [keep_b])
        for tap_batched, tap_a, tap_b in zip(batched, single_a, single_b):
            assert torch.equal(tap_batched["features"][0], tap_a["features"][0])
            assert torch.equal(tap_batched["features"][1], tap_b["features"][0])

    def test_none_masks_match_dense(self):
        torch.manual_seed(0)
        wrapper = make_wrapper()
        x = torch.randn(2, 3, 56, 56)
        dense = wrapper.stage_feature_maps(x, TAPS)
        masked = wrapper.stage_feature_maps_masked(x, TAPS, [None, None])
        for tap_dense, tap_masked in zip(dense, masked):
            assert torch.equal(tap_dense["features"], tap_masked["features"])

    def test_mask_grid_mismatch_raises(self):
        wrapper = make_wrapper()
        x = torch.randn(1, 3, 56, 56)
        with pytest.raises(ValueError, match="evenly tile"):
            wrapper.stage_feature_maps_masked(
                x, TAPS, [torch.ones(3, 3, dtype=torch.bool)]
            )

    def test_mask_count_mismatch_raises(self):
        wrapper = make_wrapper()
        x = torch.randn(2, 3, 56, 56)
        with pytest.raises(ValueError, match="keep masks"):
            wrapper.stage_feature_maps_masked(
                x, TAPS, [torch.ones(2, 2, dtype=torch.bool)]
            )

    def test_gradients_flow_to_kept_tokens(self):
        torch.manual_seed(0)
        wrapper = make_wrapper()
        wrapper.train()
        x = torch.randn(1, 3, 56, 56)
        keep = torch.zeros(2, 2, dtype=torch.bool)
        keep[0, 0] = True
        (features,) = [
            tap["features"]
            for tap in wrapper.stage_feature_maps_masked(
                x, TAPS[:1], [keep]
            )
        ]
        features.sum().backward()
        weight = wrapper.hf_model.embeddings.patch_embeddings.projection.weight
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0


class TestWrapperMaskedForwardDinov3:
    """The same wrapper invariants on the DINOv3 layout (design doc 20):

    the masked orchestration differs (rope gathered with the same kept
    indices as the tokens), so every invariant is re-proven here rather
    than assumed from the DINOv2 suite — keep-all ≡ dense in particular
    fails on ANY rope misalignment, because apply_rotary_pos_emb consumes
    the gathered table positionally.
    """

    def test_keep_all_matches_dense_bit_exact(self):
        torch.manual_seed(0)
        wrapper = make_dinov3_wrapper()
        x = torch.randn(2, 3, 56, 56)
        dense = wrapper.stage_feature_maps(x, TAPS)
        all_keep = torch.ones(2, 2, dtype=torch.bool)
        masked = wrapper.stage_feature_maps_masked(
            x, TAPS, [all_keep, all_keep]
        )
        for tap_dense, tap_masked in zip(dense, masked):
            assert torch.equal(tap_dense["features"], tap_masked["features"])
            assert torch.equal(tap_dense["cls_token"], tap_masked["cls_token"])

    def test_partial_keep_zero_fills_dropped_windows(self):
        torch.manual_seed(0)
        wrapper = make_dinov3_wrapper()
        x = torch.randn(1, 3, 56, 56)
        keep = torch.zeros(2, 2, dtype=torch.bool)
        keep[0, 0] = True  # top-left window only: patches rows/cols 0-1
        (features,) = [
            tap["features"]
            for tap in wrapper.stage_feature_maps_masked(
                x, TAPS[:1], [keep]
            )
        ]
        assert tuple(features.shape) == (1, 32, 4, 4)
        assert features[:, :, :2, 2:].abs().sum() == 0, "right half not zero"
        assert features[:, :, 2:, :].abs().sum() == 0, "bottom half not zero"
        kept_region = features[0, :, :2, :2]
        assert kept_region.abs().sum() > 0, "kept window carries no features"
        assert torch.isfinite(kept_region).all()

    def test_batch_masks_equal_per_image_concat(self):
        torch.manual_seed(0)
        wrapper = make_dinov3_wrapper()
        x = torch.randn(2, 3, 56, 56)
        keep_a = torch.tensor([[True, True], [False, True]])
        keep_b = torch.tensor([[True, False], [True, True]])
        batched = wrapper.stage_feature_maps_masked(
            x, TAPS, [keep_a, keep_b]
        )
        single_a = wrapper.stage_feature_maps_masked(x[:1], TAPS, [keep_a])
        single_b = wrapper.stage_feature_maps_masked(x[1:2], TAPS, [keep_b])
        for tap_batched, tap_a, tap_b in zip(batched, single_a, single_b):
            assert torch.equal(tap_batched["features"][0], tap_a["features"][0])
            assert torch.equal(tap_batched["features"][1], tap_b["features"][0])

    def test_none_masks_match_dense(self):
        torch.manual_seed(0)
        wrapper = make_dinov3_wrapper()
        x = torch.randn(2, 3, 56, 56)
        dense = wrapper.stage_feature_maps(x, TAPS)
        masked = wrapper.stage_feature_maps_masked(x, TAPS, [None, None])
        for tap_dense, tap_masked in zip(dense, masked):
            assert torch.equal(tap_dense["features"], tap_masked["features"])

    def test_mask_grid_mismatch_raises(self):
        wrapper = make_dinov3_wrapper()
        x = torch.randn(1, 3, 56, 56)
        with pytest.raises(ValueError, match="evenly tile"):
            wrapper.stage_feature_maps_masked(
                x, TAPS, [torch.ones(3, 3, dtype=torch.bool)]
            )

    def test_mask_count_mismatch_raises(self):
        wrapper = make_dinov3_wrapper()
        x = torch.randn(2, 3, 56, 56)
        with pytest.raises(ValueError, match="keep masks"):
            wrapper.stage_feature_maps_masked(
                x, TAPS, [torch.ones(2, 2, dtype=torch.bool)]
            )

    def test_gradients_flow_to_kept_tokens(self):
        torch.manual_seed(0)
        wrapper = make_dinov3_wrapper()
        wrapper.train()
        x = torch.randn(1, 3, 56, 56)
        keep = torch.zeros(2, 2, dtype=torch.bool)
        keep[0, 0] = True
        (features,) = [
            tap["features"]
            for tap in wrapper.stage_feature_maps_masked(
                x, TAPS[:1], [keep]
            )
        ]
        # A plain sum is degenerate: the DINOv3 tap carries the baseline's
        # final LayerNorm, whose fresh-init all-ones weight makes the
        # per-token channel sum identically zero — the gradient vanishes
        # for a reason unrelated to token dropping. Weight the tap with a
        # fixed random probe instead (same intent, honest check).
        probe = torch.randn_like(features)
        (features * probe).sum().backward()
        # DINOv3 embeddings expose the patch stem directly (a Conv2d).
        weight = wrapper.hf_model.embeddings.patch_embeddings.weight
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0

    def test_supports_masked_forward_layouts(self):
        # DINOv2 and DINOv3 ViT wrappers declare support; the ConvNeXt
        # wrapper is not part of the inheritance tree — it has no masked
        # entry at all, and the task-model family gate keeps it from ever
        # reaching one (design doc 20 §3.2).
        from autovision.encoders.dinov3.dinov3_wrapper import (
            TransformersDinov3ConvNextWrapper,
        )

        assert make_wrapper().supports_masked_forward()
        assert make_dinov3_wrapper().supports_masked_forward()
        convnext_config = transformers.DINOv3ConvNextConfig(
            hidden_sizes=[16, 16, 16, 16],
            depths=[1, 1, 1, 1],
            image_size=64,
            patch_size=4,
        )
        convnext = TransformersDinov3ConvNextWrapper(
            transformers.DINOv3ConvNextModel(convnext_config)
        )
        assert not isinstance(convnext, TransformersDinov2Wrapper)
        assert not hasattr(convnext, "stage_feature_maps_masked")


class TestMaskConstruction:
    @staticmethod
    def _batch(boxes_per_image: list[torch.Tensor], size: int = 112):
        x = torch.randn(len(boxes_per_image), 3, size, size)
        targets = [
            {"boxes": boxes, "labels": torch.zeros(boxes.shape[0], dtype=torch.long)}
            for boxes in boxes_per_image
        ]
        return x, targets

    @staticmethod
    def _build(x, targets, args):
        """Adapter for the CPU-side signature: boxes pre-device + canvas."""
        return _build_window_keep_masks(
            [t["boxes"] for t in targets],
            int(x.shape[-2]),
            int(x.shape[-1]),
            args,
        )

    def test_box_windows_never_dropped(self):
        torch.manual_seed(0)
        # One box pinned to the top-left 28px window of the 112px canvas.
        boxes = torch.tensor([[0.125, 0.125, 0.02, 0.02]])
        x, targets = self._batch([boxes])
        args = DETRTokenDropArgs(window_size=28, keep_ratio=0.4, prob=1.0)
        for _ in range(20):  # resample the background subset repeatedly
            masks, _ = self._build(x, targets, args)
            mask = masks[0]
            assert mask is not None
            assert mask[0, 0], "box window must survive every draw"

    def test_exact_kept_window_count(self):
        torch.manual_seed(0)
        boxes = torch.tensor([[0.125, 0.125, 0.02, 0.02]])
        x, targets = self._batch([boxes])
        args = DETRTokenDropArgs(window_size=28, keep_ratio=0.4)
        masks, _ = self._build(x, targets, args)
        protected = protected_cell_mask(boxes, 112, 112, cell=28, padding=0.15)
        background = int((~protected).sum())
        expected_kept = 16 - int(round(0.6 * background))
        assert int(masks[0].sum()) == expected_kept

    def test_empty_boxes_run_dense(self):
        x, targets = self._batch([torch.zeros(0, 4)])
        masks, ratio = self._build(x, targets, DETRTokenDropArgs(window_size=28))
        assert masks == [None]
        assert ratio == 1.0

    def test_undivisible_canvas_runs_dense(self):
        boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
        x, targets = self._batch([boxes], size=100)  # 100 % 28 != 0
        masks, _ = self._build(x, targets, DETRTokenDropArgs(window_size=28))
        assert masks == [None]

    def test_prob_zero_is_identity(self):
        torch.manual_seed(0)
        boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
        x, targets = self._batch([boxes, boxes])
        masks, _ = self._build(x, targets, DETRTokenDropArgs(window_size=28, prob=0.0))
        assert masks == [None, None]

    def test_fully_protected_sample_runs_dense(self):
        torch.manual_seed(0)
        # A 2x2-window box on the 4x4 grid with keep_ratio=0: background
        # keeps nothing... its complement — the draw drops every background
        # window, so keep.all() is False. Instead protect EVERY window:
        # four boxes pinning all four quadrants.
        boxes = torch.tensor(
            [[0.25, 0.25, 0.49, 0.49], [0.75, 0.25, 0.49, 0.49],
             [0.25, 0.75, 0.49, 0.49], [0.75, 0.75, 0.49, 0.49]]
        )
        x, targets = self._batch([boxes])
        masks, ratio = self._build(
            x, targets, DETRTokenDropArgs(window_size=28, keep_ratio=0.4)
        )
        assert masks == [None], "an all-keep draw must take the dense path"
        assert ratio == 1.0

    def test_seed_reproducibility(self):
        boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
        x, targets = self._batch([boxes])
        args = DETRTokenDropArgs(window_size=28)
        torch.manual_seed(7)
        first, _ = self._build(x, targets, args)
        torch.manual_seed(7)
        second, _ = self._build(x, targets, args)
        assert torch.equal(first[0], second[0])


class TestConfigContract:
    def test_off_by_default(self):
        args = DETRObjectDetectionTrainArgs()
        assert args.token_drop is None

    def test_task_model_default_has_no_token_drop(self):
        model = make_task_model()
        assert model._token_drop_args is None

    def test_args_dict_coerced_and_snapshotted(self):
        model = make_task_model(
            token_drop={"window_size": 28, "keep_ratio": 0.4, "step_stop": None}
        )
        assert isinstance(model._token_drop_args, DETRTokenDropArgs)
        assert model._token_drop_args.window_size == 28
        # The export-safe snapshot is a plain dict (weights_only loader).
        assert isinstance(model._init_args["token_drop"], dict)

    def test_window_size_must_align_patch_grid(self):
        with pytest.raises(ValueError, match="multiple of the backbone patch"):
            make_task_model(token_drop={"window_size": 32})  # patch 14

    def test_convnext_family_rejected(self):
        with pytest.raises(ValueError, match="ViT-family"):
            DETRObjectDetection(
                num_classes=2,
                model_name="dinov3-convnext-tiny",
                token_drop={"window_size": 32},
            )

    @pytest.mark.skipif(
        not _dinov3_supported(), reason="transformers lacks DINOv3"
    )
    def test_dinov3_vit_masked_forward_accepted(self):
        # Since design doc 20 the DINOv3 ViT wrapper implements the masked
        # forward itself (embeddings -> rope -> encoder with rope gathered
        # alongside the tokens), so the structural gate accepts it.
        model = DETRObjectDetection(
            num_classes=2,
            model_name="dinov3-vits16-notpretrained",
            image_size=(128, 128),
            num_queries=10,
            num_top_queries=10,
            token_drop={"window_size": 32},
        )
        assert model._token_drop_args is not None


class TestTaskModelIntegration:
    def test_all_keep_masks_match_dense_bit_exact(self):
        torch.manual_seed(0)
        model = make_task_model(token_drop={"window_size": 28, "step_stop": None})
        x = torch.randn(2, 3, 112, 112)
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "labels": torch.tensor([0]),
                "window_keep_mask": torch.ones(4, 4, dtype=torch.bool),
            },
            {
                "boxes": torch.tensor([[0.1, 0.1, 0.1, 0.1]]),
                "labels": torch.tensor([1]),
                "window_keep_mask": torch.ones(4, 4, dtype=torch.bool),
            },
        ]
        with torch.no_grad():
            dense = model._forward_train(x)
            masked = model._forward_train(x, targets=targets)
        assert torch.equal(dense["pred_logits"], masked["pred_logits"])
        assert torch.equal(dense["pred_boxes"], masked["pred_boxes"])

    def test_val_and_export_paths_run_dense(self):
        model = make_task_model(token_drop={"window_size": 28, "step_stop": None})
        x = torch.randn(1, 3, 112, 112)
        assert model._extract_window_keep_masks(x, None) is None
        targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
        assert model._extract_window_keep_masks(x, targets) is None

    def test_partial_mask_padding_semantics(self):
        model = make_task_model(token_drop={"window_size": 28, "step_stop": None})
        x = torch.randn(2, 3, 112, 112)
        keep = torch.zeros(4, 4, dtype=torch.bool)
        keep[0, 0] = True
        masks = model._extract_window_keep_masks(
            x, [{"boxes": torch.zeros(1, 4), "window_keep_mask": keep}, {}]
        )
        assert masks is not None
        # Image 0 keeps its mask; image 1 (no mask) keeps everything.
        assert torch.equal(masks[0], keep)
        assert bool(masks[1].all())
        assert masks[0].device == x.device

    def test_non_vit_backbone_guard(self):
        model = make_task_model()
        # Simulate a mask leaking through to a non-ViT backbone: the hard
        # guard must fire even though the config gate already prevents it.
        model.backbone = torch.nn.Module()  # not a ViTDetectionBackbone
        x = torch.randn(1, 3, 112, 112)
        keep = torch.ones(4, 4, dtype=torch.bool)
        with pytest.raises(ValueError, match="ViTDetectionBackbone"):
            model._extract_window_keep_masks(
                x, [{"boxes": torch.zeros(1, 4), "window_keep_mask": keep}]
            )


class TestTrainingStepIntegration:
    def make_train_model(self, token_drop: dict | None) -> DETRObjectDetectionTrain:
        args = DETRObjectDetectionTrainArgs(
            num_classes=4,
            backbone="dinov2-vittest-notpretrained",
            image_size=(112, 112),
            num_queries=10,
            num_top_queries=10,
            token_drop=token_drop,
        )
        return DETRObjectDetectionTrain(args)

    @staticmethod
    def make_batch():
        torch.manual_seed(0)
        return {
            "image": torch.randn(2, 3, 112, 112),
            "bboxes": [
                torch.tensor([[0.125, 0.125, 0.02, 0.02]]),
                torch.tensor([[0.5, 0.5, 0.02, 0.02]]),
            ],
            "classes": [torch.tensor([0]), torch.tensor([1])],
        }

    def test_training_step_attaches_masks_and_logs_ratio(self):
        train_model = self.make_train_model(
            {"window_size": 28, "keep_ratio": 0.4, "step_stop": None}
        )
        train_model.train()
        result = train_model.training_step(self.make_batch(), step=0)
        assert "token_keep_ratio" in result.log_dict
        assert 0.0 < result.log_dict["token_keep_ratio"] < 1.0

    def test_step_stop_closes_the_window(self):
        train_model = self.make_train_model(
            {"window_size": 28, "keep_ratio": 0.4, "step_stop": 5}
        )
        train_model.train()
        result = train_model.training_step(self.make_batch(), step=7)
        assert "token_keep_ratio" not in result.log_dict

    def test_off_by_default_has_no_diagnostics(self):
        train_model = self.make_train_model(None)
        train_model.train()
        result = train_model.training_step(self.make_batch(), step=0)
        assert "token_keep_ratio" not in result.log_dict

    def test_get_optimizer_resolves_auto_step_stop(self):
        train_model = self.make_train_model(
            {"window_size": 28, "keep_ratio": 0.4, "step_stop": "auto"}
        )
        train_model.get_optimizer(
            total_steps=100,
            global_batch_size=2,
            train_num_batches=50,
            gradient_accumulation_steps=1,
        )
        resolved = train_model.model_args.token_drop.step_stop
        assert isinstance(resolved, int)
        assert 0 < resolved <= 100


class TestSelectionReuse:
    def test_build_mask_matches_eraser_semantics(self):
        """``~sample_erase_mask(protected, 1 - r)`` IS the keep mask: the
        protected set survives verbatim and the background budget matches
        the eraser's exact rounding."""
        torch.manual_seed(3)
        boxes = torch.tensor([[0.3, 0.6, 0.1, 0.15], [0.7, 0.2, 0.2, 0.1]])
        protected = protected_cell_mask(boxes, 112, 112, cell=28, padding=0.15)
        keep = ~sample_erase_mask(protected, 1.0 - 0.4)
        assert bool(keep[protected].all()), "protected windows must survive"
        background = int((~protected).sum())
        assert int((~keep).sum()) == int(round(0.6 * background))
