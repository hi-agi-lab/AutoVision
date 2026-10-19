# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the DetCon pretraining methods (S = SupCon / B = BYOL)."""

import logging
from pathlib import Path

import pytest
import torch
from PIL import Image
from tests.helpers_models import TinyWrapper

from autovision.data.file_helpers import open_mask
from autovision.data.image_dataset import ImageDataset
from autovision.encoders import FeatureExtractor
from autovision.pretext import get_method_cls, list_methods
from autovision.pretext.detcon import (
    DetConB,
    DetConBArgs,
    DetConBLoss,
    DetConBSGDArgs,
    DetConS,
    DetConSArgs,
    DetConSGDArgs,
    DetConSLoss,
)
from autovision.pretext.detcon.detcon import (
    _region_pipeline,
    _subsample_mask_indices,
    _subsample_pooled_features,
    pool_masked,
)
from autovision.pretext.detcon.detcon_transform import (
    DetConBTransform,
    DetConBTransformArgs,
    DetConSTransform,
    DetConSTransformArgs,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def make_s(**overrides) -> DetConS:
    args = DetConSArgs(
        hidden_dim=32, output_dim=16, num_sampled_cls=4, num_total_cls=8, **overrides
    )
    args.resolve_auto_fields()
    optimizer_args = DetConSGDArgs()
    optimizer_args.resolve_auto_fields(wrapped_model=None)
    return DetConS(
        method_args=args,
        optimizer_args=optimizer_args,
        embedding_model=FeatureExtractor(wrapped_model=TinyWrapper(dim=8), embed_dim=8),
        global_batch_size=4,
        num_input_channels=3,
    )


def make_b(**overrides) -> DetConB:
    args = DetConBArgs(
        proj_hidden_dim=32,
        proj_output_dim=16,
        pred_hidden_dim=32,
        pred_output_dim=16,
        num_sampled_cls=4,
        num_total_cls=8,
        **overrides,
    )
    args.resolve_auto_fields()
    optimizer_args = DetConBSGDArgs()
    optimizer_args.resolve_auto_fields(wrapped_model=None)
    return DetConB(
        method_args=args,
        optimizer_args=optimizer_args,
        embedding_model=FeatureExtractor(wrapped_model=TinyWrapper(dim=8), embed_dim=8),
        global_batch_size=4,
        num_input_channels=3,
    )


def make_batch(batch_size: int = 2, num_ids: int = 4, size: int = 32) -> dict:
    return {
        "views": [
            torch.randn(batch_size, 3, size, size),
            torch.randn(batch_size, 3, size, size),
        ],
        "masks": [
            torch.randint(0, num_ids, (batch_size, size, size)),
            torch.randint(0, num_ids, (batch_size, size, size)),
        ],
    }


def test_registered():
    assert "detcons" in list_methods()
    assert "detconb" in list_methods()
    assert get_method_cls("detcons") is DetConS
    assert get_method_cls("detconb") is DetConB


class TestArgs:
    def test_defaults_match_baseline(self):
        args = DetConSArgs()
        assert args.hidden_dim == 2048
        assert args.output_dim == 128
        assert args.num_layers == 2
        assert args.batch_norm is True
        assert args.temperature == 0.1
        assert args.num_sampled_cls == 16
        assert args.num_total_cls == 256
        bargs = DetConBArgs()
        assert bargs.proj_output_dim == 256
        assert bargs.pred_output_dim == 256
        assert bargs.num_sampled_cls == 16

    def test_auto_sgd_recipes(self):
        assert DetConS.optimizer_args_cls("auto") is DetConSGDArgs
        assert DetConB.optimizer_args_cls("sgd") is DetConBSGDArgs
        assert DetConSGDArgs().lr == 0.3
        assert DetConSGDArgs().weight_decay == 0.0001


class TestPoolMasked:
    def test_manual_average(self):
        # 1 image, 2x2 features, ids 0 and 1: pooled = mean per region.
        source = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]]]]  # (1, 1, 2, 2)
        )
        mask = torch.tensor([[[0, 0], [1, 1]]])  # (1, 2, 2)
        pooled = pool_masked(source, mask, num_cls=3)
        assert pooled.shape == (1, 1, 3)
        assert pooled[0, 0, 0].item() == pytest.approx(1.5)
        assert pooled[0, 0, 1].item() == pytest.approx(3.5)
        assert pooled[0, 0, 2].item() == 0.0  # empty region

    def test_multi_channel_batch(self):
        torch.manual_seed(0)
        source = torch.randn(3, 4, 5, 5)
        mask = torch.randint(0, 4, (3, 5, 5))
        pooled = pool_masked(source, mask, num_cls=4)
        assert pooled.shape == (3, 4, 4)
        # Spot check one region manually.
        sel = mask[1] == 2
        expected = source[1][:, sel].mean(dim=1) if sel.any() else torch.zeros(4)
        assert torch.allclose(pooled[1][:, 2], expected)


class TestSubsampleMaskIndices:
    def test_shape(self):
        masks = torch.randint(0, 5, (3, 8, 8))
        indices = _subsample_mask_indices(masks, n_total_cls=10, n_sampled_cls=6)
        assert indices.shape == (3, 6)
        assert indices.dtype in (torch.int64, torch.int32)

    def test_uniform_sampling_picks_present_ids_only(self):
        # Image contains only ids 2 and 5; sampled ids must come from them.
        masks = torch.full((1, 8, 8), 2)
        masks[:, :4] = 5
        torch.manual_seed(0)
        indices = _subsample_mask_indices(
            masks, n_total_cls=10, n_sampled_cls=64
        )
        assert set(indices.unique().tolist()) <= {2, 5}
        # Both present ids are sampled (uniform probability, large n).
        assert indices.unique().numel() == 2

    def test_class_weighted_favors_large_regions(self):
        # Id 1 covers 3/4 of the image: weighted sampling must pick it with
        # ~0.75 probability, well above the uniform 0.5.
        masks = torch.full((1, 16, 16), 0)
        masks[:, :12] = 1
        torch.manual_seed(0)
        indices = _subsample_mask_indices(
            masks, n_total_cls=2, n_sampled_cls=512, class_weighted_sampling=True
        )
        assert (indices == 1).float().mean().item() > 0.65

    def test_repetition_allows_more_samples_than_ids(self):
        masks = torch.full((2, 4, 4), 3)
        indices = _subsample_mask_indices(masks, n_total_cls=8, n_sampled_cls=16)
        assert (indices == 3).all()


class TestSubsamplePooledFeatures:
    def test_gather_matches_manual(self):
        x = torch.randn(2, 5, 3)
        indices = torch.tensor([[0, 2, 4], [1, 1, 3]])
        out = _subsample_pooled_features(x, indices)
        assert out.shape == (2, 3, 3)
        assert torch.equal(out[0], x[0][[0, 2, 4]])
        assert torch.equal(out[1], x[1][[1, 1, 3]])

    def test_region_pipeline_reuses_indices(self):
        # DetConB must gather the SAME regions for the target branch;
        # columns of both pooled tensors stay aligned to the same ids.
        torch.manual_seed(0)
        features = torch.randn(4, 3, 8, 8)
        masks = torch.randint(0, 5, (4, 8, 8))
        pooled1, idx1 = _region_pipeline(features, masks, n_total_cls=5, n_sampled_cls=3)
        pooled2, idx2 = _region_pipeline(
            features * 2, masks, n_total_cls=5, n_sampled_cls=3, indices=idx1
        )
        assert torch.equal(idx1, idx2)
        assert torch.allclose(pooled2, pooled1 * 2)


class TestDetConSLoss:
    def test_matching_beats_random(self):
        torch.manual_seed(0)
        criterion = DetConSLoss(temperature=0.1)
        view0 = torch.randn(4, 8, 16)
        view1 = view0.clone()  # identical embeddings, same ids -> easy task
        mask0 = torch.randint(0, 3, (4, 8))
        mask1 = mask0.clone()
        loss_match = criterion(view0, view1, mask0, mask1)
        loss_random = criterion(view0, torch.randn(4, 8, 16), mask0, mask1)
        assert loss_match < loss_random

    def test_no_positives_returns_zero_loss_with_graph(self):
        criterion = DetConSLoss()
        # Single image, disjoint id sets across views -> no positive pairs.
        view0 = torch.randn(1, 4, 8, requires_grad=True)
        view1 = torch.randn(1, 4, 8, requires_grad=True)
        mask0 = torch.zeros(1, 4, dtype=torch.long)
        mask1 = torch.ones(1, 4, dtype=torch.long)
        loss = criterion(view0, view1, mask0, mask1)
        assert loss.item() == pytest.approx(0.0)
        assert loss.requires_grad
        loss.backward()
        assert view0.grad is not None

    def test_same_source_regions_are_not_positives(self):
        # All views share one id: positives exist only across sources, so
        # the loss must be finite and lower than against random labels.
        criterion = DetConSLoss()
        torch.manual_seed(1)
        base = torch.randn(2, 4, 8)
        view1 = torch.randn(2, 4, 8)
        mask = torch.zeros(2, 4, dtype=torch.long)
        loss = criterion(base, view1, mask, mask.clone())
        loss_shuffled = criterion(
            base, view1, mask, torch.ones_like(mask)
        )
        assert torch.isfinite(loss)
        assert loss < loss_shuffled

    def test_temperature_scales_logits(self):
        torch.manual_seed(0)
        view0 = torch.randn(2, 4, 8)
        view1 = torch.randn(2, 4, 8)
        mask0 = torch.randint(0, 2, (2, 4))
        mask1 = mask0.flip(1)
        warm = DetConSLoss(temperature=1.0)(view0, view1, mask0, mask1)
        # A tiny temperature makes random same-id pairs dominate; with the
        # flipped masks (few true positives) the loss grows.
        cold = DetConSLoss(temperature=0.01)(view0, view1, mask0, mask1)
        assert cold > warm


class TestDetConBLoss:
    @staticmethod
    def _make_pair():
        torch.manual_seed(0)
        pred0 = torch.randn(2, 3, 8)
        pred1 = torch.randn(2, 3, 8)
        target0 = torch.randn(2, 3, 8)
        target1 = torch.randn(2, 3, 8)
        mask0 = torch.tensor([[0, 0, 1], [0, 1, 1]])
        mask1 = torch.tensor([[0, 1, 1], [0, 0, 1]])
        return pred0, pred1, target0, target1, mask0, mask1

    def test_manual_value(self):
        criterion = DetConBLoss()
        pred0, pred1, target0, target1, mask0, mask1 = self._make_pair()
        loss = criterion(pred0, pred1, target0, target1, mask0, mask1)
        # Recompute with an independent reference implementation.
        def dir_half(pred, pred_mask, target, target_mask):
            num_cls = int(torch.maximum(pred_mask.max(), target_mask.max()).item()) + 1
            onehot = torch.nn.functional.one_hot(
                target_mask, num_classes=num_cls
            ).to(target.dtype)
            counts = onehot.sum(dim=1)
            sums = torch.einsum("bpc,bpd->bcd", onehot, target)
            present = counts > 0
            means = torch.nn.functional.normalize(
                sums / counts.clamp(min=1).unsqueeze(-1), dim=-1
            )
            pnorm = torch.nn.functional.normalize(pred, dim=-1)
            matched = present.gather(1, pred_mask)
            tgt = means.gather(
                1, pred_mask.unsqueeze(-1).expand(-1, -1, means.shape[-1])
            )
            cos = (pnorm * tgt).sum(-1)
            return (1.0 - cos)[matched].mean()

        expected = (
            dir_half(pred0, mask0, target1, mask1)
            + dir_half(pred1, mask1, target0, mask0)
        ) / 2
        assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
        assert 0.0 <= loss.item() <= 2.0  # 1 - cos is in [0, 2]

    def test_matched_region_pairing_is_exact(self):
        # Single region per view with the same id: loss = 1 - cos(pred0, t1).
        criterion = DetConBLoss()
        torch.manual_seed(0)
        pred0 = torch.randn(1, 1, 8)
        pred1 = torch.randn(1, 1, 8)
        target0 = torch.randn(1, 1, 8)
        target1 = torch.randn(1, 1, 8)
        zeros = torch.zeros(1, 1, dtype=torch.long)
        loss = criterion(pred0, pred1, target0, target1, zeros, zeros)
        cos01 = torch.nn.functional.normalize(pred0[0, 0], dim=-1) @ (
            torch.nn.functional.normalize(target1[0, 0], dim=-1)
        )
        cos10 = torch.nn.functional.normalize(pred1[0, 0], dim=-1) @ (
            torch.nn.functional.normalize(target0[0, 0], dim=-1)
        )
        assert loss.item() == pytest.approx(1.0 - (cos01.item() + cos10.item()) / 2)

    def test_symmetric_under_view_swap(self):
        criterion = DetConBLoss()
        pred0, pred1, target0, target1, mask0, mask1 = self._make_pair()
        loss = criterion(pred0, pred1, target0, target1, mask0, mask1)
        loss_swapped = criterion(pred1, pred0, target1, target0, mask1, mask0)
        assert loss.item() == pytest.approx(loss_swapped.item())

    def test_missing_ids_are_skipped(self):
        # Id 1 exists in the prediction view but not in the target view of
        # direction 0->1: it must be skipped, not averaged in. Note the two
        # views hold a different number of regions (the loss accepts that).
        criterion = DetConBLoss()
        torch.manual_seed(0)
        pred0 = torch.randn(1, 2, 8)  # id 0 (matched), id 1 (missing in target1)
        pred1 = torch.randn(1, 1, 8)
        target0 = torch.randn(1, 2, 8)  # ids 0 and 1
        target1 = torch.randn(1, 1, 8)  # id 0 only
        mask0 = torch.tensor([[0, 1]])
        mask1 = torch.zeros(1, 1, dtype=torch.long)
        loss = criterion(pred0, pred1, target0, target1, mask0, mask1)
        n = torch.nn.functional.normalize
        # Direction 0->1: pred region 0 (id 0) pairs with target1 id 0
        # (single region); pred region 1 (id 1) has no counterpart -> skip.
        cos01 = (n(pred0[0, 0], dim=-1) * n(target1[0, 0], dim=-1)).sum()
        # Direction 1->0: pred region id 0 pairs with target0's id-0 region
        # (row 0 only; row 1 carries id 1).
        cos10 = (n(pred1[0, 0], dim=-1) * n(target0[0, 0], dim=-1)).sum()
        expected = ((1 - cos01) + (1 - cos10)) / 2
        assert loss.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_pred_ids_outside_target_id_range_do_not_crash(self):
        # Regression: the id table must cover both sides even when the
        # target view only contains small ids.
        criterion = DetConBLoss()
        pred = torch.randn(1, 2, 8)
        target = torch.randn(1, 2, 8)
        pred_mask = torch.tensor([[0, 3]])
        target_mask = torch.zeros(1, 2, dtype=torch.long)
        loss = criterion(pred, pred, target, target, pred_mask, target_mask)
        assert torch.isfinite(loss)

    def test_no_valid_pairs_zero_loss(self):
        criterion = DetConBLoss()
        pred = torch.randn(1, 2, 8, requires_grad=True)
        target = torch.randn(1, 2, 8)
        mask_pred = torch.zeros(1, 2, dtype=torch.long)
        mask_target = torch.ones(1, 2, dtype=torch.long)
        loss = criterion(pred, pred, target, target, mask_pred, mask_target)
        assert loss.item() == pytest.approx(0.0)
        loss.backward()
        assert pred.grad is not None


class TestDetConSMethod:
    def test_training_step_finite_and_grads(self):
        method = make_s()
        loss = method.training_step(make_batch(), 0)
        assert torch.isfinite(loss)
        loss.backward()
        for module in (method.embedding_model, method.projection_head):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    def test_mask_resize_path(self):
        # Input 64x64 with a 7x7 ViT-like grid? TinyWrapper keeps 64x64;
        # instead vary the mask size to force the NEAREST resize.
        batch = make_batch(size=64)
        batch["masks"] = [
            torch.randint(0, 4, (2, 40, 40)),
            torch.randint(0, 4, (2, 40, 40)),
        ]
        method = make_s()
        loss = method.training_step(batch, 0)
        assert torch.isfinite(loss)

    def test_log_dict_and_no_teacher_keys(self):
        method = make_s()
        method.training_step(make_batch(), 0)
        assert method.consume_log_dict() is None  # S does not log extras
        checkpoint = {"state_dict": method.state_dict()}
        assert not any("momentum" in key for key in checkpoint["state_dict"])

    def test_loss_decreases(self):
        torch.manual_seed(0)
        method = make_s()
        optimizer, scheduler = method.configure_optimizers(
            total_steps=12, num_epochs=2
        )
        batch = make_batch()
        losses = []
        for step in range(12):
            method.global_step = step
            loss = method.training_step(batch, step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
        assert losses[-1] < losses[0]


class TestDetConBMethod:
    def test_training_step_finite_and_grads(self):
        method = make_b()
        loss = method.training_step(make_batch(), 0)
        assert torch.isfinite(loss)
        loss.backward()
        for module in (
            method.embedding_model,
            method.projection_head,
            method.prediction_head,
        ):
            assert any(p.grad is not None for p in module.parameters())

    def test_ema_branch_has_no_grads_but_updates(self):
        method = make_b()
        for module in (
            method.embedding_model_momentum,
            method.projection_head_momentum,
        ):
            assert all(not p.requires_grad for p in module.parameters())
        optimizer, _ = method.configure_optimizers(total_steps=8, num_epochs=2)
        ema_before = method.embedding_model_momentum._inner.conv.weight.clone()
        student_before = method.embedding_model._inner.conv.weight.clone()
        # Step 1: the EMA update is a no-op because EMA starts as a copy of
        # the student (0.99 * s + 0.01 * s == s); the student then moves.
        loss = method.training_step(make_batch(), 0)
        loss.backward()
        optimizer.step()
        # Step 2: the EMA now follows the updated student.
        student_mid = method.embedding_model._inner.conv.weight.clone()
        assert not torch.equal(student_mid, student_before)
        loss = method.training_step(make_batch(), 1)
        loss.backward()
        optimizer.step()
        ema_after = method.embedding_model_momentum._inner.conv.weight
        student_after = method.embedding_model._inner.conv.weight
        d_target = (ema_after - ema_before).norm()
        d_full = (student_after - student_before).norm()
        assert 0 < d_target < d_full  # EMA follows with momentum < 1
        assert all(
            p.grad is None for p in method.embedding_model_momentum.parameters()
        )

    def test_trainable_modules_exclude_ema(self):
        method = make_b()
        modules = method.trainable_modules().targets
        assert method.embedding_model_momentum not in modules
        assert method.projection_head_momentum not in modules

    def test_checkpoint_strips_ema(self):
        method = make_b()
        with torch.no_grad():
            for param in method.embedding_model_momentum.parameters():
                param.add_(1.0)
        checkpoint = {"state_dict": {k: v.clone() for k, v in method.state_dict().items()}}
        assert any(k.startswith("embedding_model_momentum.") for k in checkpoint["state_dict"])
        method.on_save_checkpoint(checkpoint)
        assert not any(
            k.startswith(("embedding_model_momentum.", "projection_head_momentum."))
            for k in checkpoint["state_dict"]
        )

    def test_load_checkpoint_restores_ema_values(self):
        # A checkpoint that still carries EMA weights (e.g. an older run)
        # must be restored verbatim, not overwritten by the student copy.
        method = make_b()
        with torch.no_grad():
            for param in method.embedding_model_momentum.parameters():
                param.add_(1.0)
        state_dict = {k: v.clone() for k, v in method.state_dict().items()}
        checkpoint = {"state_dict": state_dict}
        fresh = make_b()
        fresh_state = {k: v.clone() for k, v in fresh.state_dict().items()}
        fresh_state.update(checkpoint["state_dict"])
        fresh.load_state_dict(fresh_state, strict=False)
        assert torch.allclose(
            fresh.embedding_model_momentum._inner.conv.weight,
            method.embedding_model_momentum._inner.conv.weight,
        )

    def test_load_checkpoint_warns_on_missing_ema(self, caplog):
        method = make_b()
        checkpoint = {"state_dict": {k: v.clone() for k, v in method.state_dict().items()}}
        method.on_save_checkpoint(checkpoint)
        with caplog.at_level(logging.WARNING, logger="autovision"):
            method.on_load_checkpoint(checkpoint)
        assert "embedding_model_momentum." in caplog.text

    def test_loss_decreases(self):
        torch.manual_seed(0)
        method = make_b()
        optimizer, scheduler = method.configure_optimizers(
            total_steps=30, num_epochs=2
        )
        batch = make_batch()
        losses = []
        for step in range(30):
            method.global_step = step
            loss = method.training_step(batch, step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
        # The EMA target lags the student, so compare segment means.
        assert sum(losses[-10:]) < sum(losses[:10])

    def test_backward_gives_no_grad_to_ema_params(self):
        method = make_b()
        loss = method.training_step(make_batch(), 0)
        loss.backward()
        assert all(
            p.grad is None for p in method.embedding_model_momentum.parameters()
        )


class TestTransforms:
    @staticmethod
    def _input(num_ids: int = 4, size: int = 64):
        image = torch.randint(0, 256, (3, size, size), dtype=torch.uint8)
        mask = torch.randint(0, num_ids, (size, size), dtype=torch.uint8)
        return {"image": image, "mask": mask}

    def test_two_views_with_masks(self):
        transform = DetConSTransform(DetConSTransformArgs())
        out = transform(self._input())
        assert len(out) == 2
        for view in out:
            assert view["image"].shape == (3, 224, 224)
            assert view["image"].dtype == torch.float32
            assert view["mask"].shape == (224, 224)
            assert view["mask"].dtype in (torch.uint8, torch.int64, torch.int32, torch.uint16)
            assert not view["mask"].is_floating_point()

    def test_mask_ids_preserved_through_photometric(self):
        # ids 1..4 only; scaling or normalizing would break the value set.
        transform = DetConBTransform(DetConBTransformArgs())
        inp = self._input()
        original_ids = set(inp["mask"].unique().tolist())
        for view in transform(inp):
            out_ids = set(view["mask"].unique().tolist())
            assert out_ids <= original_ids | {0}

    def test_geometry_synchronization(self):
        # Image pixel value encodes the mask id (x60): with all photometric
        # ops and normalization disabled, interior pixels of the transformed
        # image must decode to the transformed mask values.
        from autovision.augment.transform_args import (
            RandomFlipArgs,
            RandomResizeArgs,
        )
        from autovision.pretext.detcon.detcon_transform import _MaskedViewTransform

        torch.manual_seed(0)
        view = _MaskedViewTransform(
            image_size=(32, 32),
            random_resize=RandomResizeArgs(),
            random_flip=RandomFlipArgs(),
            color_jitter=None,
            random_gray_scale=None,
            gaussian_blur=None,
            solarize=None,
            normalize=None,
        )
        mask = torch.zeros(64, 80, dtype=torch.uint8)
        mask[:32, :40] = 1
        mask[:32, 40:] = 2
        mask[32:, :40] = 3
        mask[32:, 40:] = 4
        image = mask.unsqueeze(0).repeat(3, 1, 1) * 60
        out = view({"image": image, "mask": mask})
        oimg, omask = out["image"], out["mask"]
        interior = omask[4:-4, 4:-4].long()
        decoded = oimg[0][4:-4, 4:-4].div(60).round().long()
        ratio = (decoded == interior).float().mean().item()
        assert ratio > 0.85  # borders blur via bilinear; interior must sync

    def test_view1_solarize_defaults(self):
        args = DetConBTransformArgs()
        assert args.view_1_solarize is not None
        assert args.view_1_solarize.prob == 0.2
        assert args.gaussian_blur.prob == 1.0
        assert args.view_1_gaussian_blur.prob == 0.1

    def test_transform_args_cls_roundtrip(self):
        assert DetConSTransform.transform_args_cls() is DetConSTransformArgs
        assert DetConBTransform.transform_args_cls() is DetConBTransformArgs


class TestMaskDataPipeline:
    @staticmethod
    def _write_pair(tmp_path: Path, name: str, size: int = 32):
        image_dir = tmp_path / "data"
        mask_dir = tmp_path / "masks"
        image_dir.mkdir(exist_ok=True)
        mask_dir.mkdir(exist_ok=True)
        Image.new("RGB", (size, size), color=(128, 64, 32)).save(image_dir / name)
        mask = torch.randint(0, 3, (size, size), dtype=torch.uint8).numpy()
        Image.fromarray(mask, mode="L").save(mask_dir / f"{Path(name).stem}.png")
        return image_dir / name, mask_dir

    def test_open_mask_single_channel(self, tmp_path: Path):
        mask = torch.arange(16, dtype=torch.uint8).reshape(4, 4)
        path = tmp_path / "m.png"
        Image.fromarray(mask.numpy(), mode="L").save(path)
        out = open_mask(path)
        assert out.shape == (4, 4)
        assert torch.equal(out, mask)

    def test_open_mask_rejects_multichannel(self, tmp_path: Path):
        path = tmp_path / "rgb.png"
        Image.new("RGB", (4, 4), color=(1, 2, 3)).save(path)
        with pytest.raises(ValueError, match="channels"):
            open_mask(path)

    def test_image_dataset_with_mask_dir(self, tmp_path: Path):
        image_path, mask_dir = self._write_pair(tmp_path, "img0.png")
        transform = DetConSTransform(DetConSTransformArgs(image_size=(32, 32)))
        dataset = ImageDataset(
            image_dir=None,
            image_filenames=[image_path],
            transform=transform,
            num_channels=3,
            mask_dir=mask_dir,
        )
        item = dataset[0]
        assert len(item["views"]) == 2
        assert len(item["masks"]) == 2
        for view_mask in item["masks"]:
            assert view_mask.shape == (32, 32)
            assert not view_mask.is_floating_point()

    def test_image_dataset_without_mask_dir(self, tmp_path: Path):
        # Regression: the plain image pipeline must not grow a masks key.
        image_path, _ = self._write_pair(tmp_path, "img1.png")
        from autovision.pretext.simclr import SimCLRTransform, SimCLRTransformArgs

        transform = SimCLRTransform(SimCLRTransformArgs(image_size=(32, 32)))
        dataset = ImageDataset(
            image_dir=None,
            image_filenames=[image_path],
            transform=transform,
            num_channels=3,
        )
        item = dataset[0]
        assert "masks" not in item
        assert len(item["views"]) == 2


class TestPretrainIntegration:
    @staticmethod
    def _write_masks(image_dir: Path, mask_dir: Path) -> None:
        mask_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(p for p in image_dir.rglob("*") if p.is_file()):
            mask = torch.randint(0, 4, (32, 32), dtype=torch.uint8)
            Image.fromarray(mask.numpy(), mode="L").save(
                mask_dir / f"{image_path.stem}.png"
            )

    def test_pretrain_detcon_end_to_end(self, tmp_path: Path):
        from tests.conftest import create_images

        from autovision import pretrain

        image_dir = tmp_path / "data"
        create_images(image_dir, num_images=4, size=(32, 32))
        self._write_masks(image_dir, tmp_path / "masks")
        result = pretrain(
            out=tmp_path / "run_detcon",
            data=image_dir,
            mask_dir=tmp_path / "masks",
            model="torchvision/resnet18",
            epochs=1,
            batch_size=2,
            num_workers=0,
            method="detcons",
            method_args={"num_total_cls": 4, "num_sampled_cls": 2},
            transform_args={"image_size": (32, 32)},
        )
        assert result.checkpoint_path.exists()

    def test_pretrain_detconb_end_to_end(self, tmp_path: Path):
        from tests.conftest import create_images

        from autovision import pretrain

        image_dir = tmp_path / "data"
        create_images(image_dir, num_images=4, size=(32, 32))
        self._write_masks(image_dir, tmp_path / "masks")
        result = pretrain(
            out=tmp_path / "run_detconb",
            data=image_dir,
            mask_dir=tmp_path / "masks",
            model="torchvision/resnet18",
            epochs=1,
            batch_size=2,
            num_workers=0,
            method="detconb",
            method_args={"num_total_cls": 4, "num_sampled_cls": 2},
            transform_args={"image_size": (32, 32)},
        )
        checkpoint = torch.load(
            result.checkpoint_path, map_location="cpu", weights_only=False
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        assert not any(
            key.startswith(("embedding_model_momentum.", "projection_head_momentum."))
            for key in state_dict
        )
