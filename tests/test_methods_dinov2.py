# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the DINOv2 pretraining method (DINO + iBOT + KoLeo)."""

import pytest
import torch
from torch import nn

from autovision.encoders import FeatureExtractor
from autovision.pretext import get_method_cls, list_methods
from autovision.pretext.dinov2 import DINOv2, DINOv2AdamWArgs, DINOv2Args
from autovision.pretext.dinov2.dinov2 import (
    _dino_pair_loss,
    _koleo_loss,
    _MaskGenerator,
)
from autovision.scaling import ScalingInfo

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class TinyMaskedViTWrapper(nn.Module):
    """ViT-ish wrapper with mask-token support and block-indexed names.

    Parameters are named ``embeddings.patch_embeddings.*`` and
    ``encoder.layer.<i>.*`` so the layer-wise LR grouping can be exercised.
    """

    def __init__(self, dim: int = 8, grid: int = 4, num_layers: int = 2):
        super().__init__()
        self.dim = dim
        self.grid = grid
        self.embeddings = nn.Module()
        self.embeddings.patch_embeddings = nn.Conv2d(
            3, dim, kernel_size=32 // grid, stride=32 // grid
        )
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU()) for _ in range(num_layers)]
        )
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.config = type("Config", (), {"num_hidden_layers": num_layers})()

    def feature_maps(self, x, bool_masked_pos=None):
        hidden = self.embeddings.patch_embeddings(x)
        if bool_masked_pos is not None:
            tokens = hidden.flatten(2).transpose(1, 2)
            replacement = self.mask_token.expand_as(tokens)
            tokens = torch.where(
                bool_masked_pos.unsqueeze(-1), replacement, tokens
            )
            hidden = tokens.transpose(1, 2).unflatten(
                2, (self.grid, self.grid)
            )
        for block in self.encoder.layer:
            hidden = hidden + block(hidden)
        return {"features": hidden, "cls_token": hidden.mean(dim=(2, 3))}

    def pool_features(self, x):
        pooled = torch.nn.functional.adaptive_avg_pool2d(x["features"], 1)
        return {"pooled_features": pooled}

    def encoder_dim(self) -> int:
        return self.dim

    def patch_size(self) -> int:
        return 32 // self.grid

    def raw_module(self):
        return self.embeddings.patch_embeddings

    def arch_traits(self):
        return {"model_type": "transformer", "norm_type": "layernorm"}


def make_method(**args_overrides) -> DINOv2:
    args = DINOv2Args(
        output_dim=16,
        hidden_dim=32,
        bottleneck_dim=8,
        teacher_temp=0.07,
        warmup_teacher_temp=0.04,
        **args_overrides,
    )
    args.resolve_auto_fields(scaling=ScalingInfo(dataset_size=64, epochs=10))
    optimizer_args = DINOv2AdamWArgs()
    optimizer_args.resolve_auto_fields(wrapped_model=None)
    student = FeatureExtractor(
        wrapped_model=TinyMaskedViTWrapper(grid=4),
        embed_dim=8,
    )
    return DINOv2(
        method_args=args,
        optimizer_args=optimizer_args,
        embedding_model=student,
        global_batch_size=8,
        num_input_channels=3,
    )


def make_views(batch_size: int = 4) -> dict:
    return {
        "views": [torch.randn(batch_size, 3, 32, 32) for _ in range(4)]
    }


def test_registered():
    assert "dinov2" in list_methods()
    assert get_method_cls("dinov2") is DINOv2


class TestArgs:
    def test_defaults(self):
        args = DINOv2Args()
        assert args.auto_steps == 125_000
        assert args.ibot_loss_weight == 1.0
        assert args.koleo_loss_weight == 0.1
        assert args.mask_ratio_min == 0.1
        assert args.mask_ratio_max == 0.5
        assert args.weight_decay_start == 0.04
        assert args.weight_decay_end == 0.4
        assert args.layerwise_decay == 0.9
        assert args.patch_embed_lr == 0.2

    def test_bad_mask_ratio_raises(self):
        with pytest.raises(ValueError, match="Mask ratios"):
            DINOv2Args(mask_ratio_min=0.5, mask_ratio_max=0.1).resolve_auto_fields()

    def test_bad_layerwise_decay_raises(self):
        with pytest.raises(ValueError, match="layerwise_decay"):
            DINOv2Args(layerwise_decay=1.5).resolve_auto_fields()


class TestMaskGenerator:
    def test_shape_and_ratio_bounds(self):
        generator = _MaskGenerator(0.1, 0.5)
        num_patches = 16
        mask = generator(8, num_patches, torch.device("cpu"))
        assert mask.shape == (8, num_patches)
        assert mask.dtype == torch.bool
        # num_masked = floor(ratio * N) with ratio ~ U(0.1, 0.5).
        counts = mask.sum(dim=1)
        assert (counts >= int(0.1 * num_patches)).all()
        assert (counts <= int(0.5 * num_patches)).all()

    def test_zero_ratio_produces_no_mask(self):
        generator = _MaskGenerator(0.0, 0.0)
        mask = generator(4, 16, torch.device("cpu"))
        assert not mask.any()


class TestKoleo:
    def test_identical_features_have_high_loss(self):
        z = torch.ones(4, 8)
        spread = torch.eye(4, 8)
        assert _koleo_loss(z) > _koleo_loss(spread)

    def test_finite(self):
        assert torch.isfinite(_koleo_loss(torch.randn(6, 8)))


class TestDinoPairLoss:
    def test_identical_pair_skipped(self):
        teacher = [torch.randn(4, 16), torch.randn(4, 16)]
        student = [torch.randn(4, 16), torch.randn(4, 16)]
        center = torch.zeros(1, 16)
        loss, targets, raw = _dino_pair_loss(
            teacher, student, center, 0.07, 0.1, skip_identical=True
        )
        assert len(targets) == 2
        assert torch.isfinite(loss)
        # Raw outputs are the L2-normalized pre-softmax teacher features
        # (the quantity the center must track), never the softmaxed ones.
        for view, out in zip(teacher, raw):
            assert torch.allclose(
                out, torch.nn.functional.normalize(view, dim=-1)
            )
            assert out.max() <= 1.0 + 1e-6  # softmax outputs would sum to 1

    def test_all_pairs_counted_when_not_skipped(self):
        teacher = [torch.randn(4, 16), torch.randn(4, 16)]
        student = [torch.randn(4, 16), torch.randn(4, 16), torch.randn(4, 16)]
        loss, _, raw = _dino_pair_loss(
            teacher, student, torch.zeros(1, 16), 0.07, 0.1, skip_identical=False
        )
        assert len(raw) == 2
        assert torch.isfinite(loss)


class TestConstruction:
    def test_teacher_is_ema_copy_sharing_patch_grid(self):
        method = make_method()
        assert method.teacher_embedding_model._inner.patch_size() == 8
        assert method.embedding_model._inner.patch_size() == 8

    def test_centers_initialized_to_zero(self):
        method = make_method()
        assert torch.all(method.dino_center == 0)
        assert torch.all(method.ibot_center == 0)


class TestTrainingStep:
    def test_loss_decreases(self):
        torch.manual_seed(0)
        method = make_method()
        optimizer, scheduler = method.configure_optimizers(
            total_steps=10, num_epochs=2
        )
        losses = []
        for step in range(10):
            method.global_step = step
            loss = method.training_step(make_views(), step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
        assert losses[-1] < losses[0]

    def test_log_dict_keys(self):
        method = make_method()
        method.training_step(make_views(), 0)
        log_dict = method.consume_log_dict()
        for key in ("train_loss/dino_global", "train_loss/dino_local",
                    "train_loss/ibot", "train_loss/koleo"):
            assert key in log_dict

    def test_centers_update(self):
        method = make_method()
        before = method.dino_center.clone()
        method.training_step(make_views(), 0)
        assert not torch.allclose(method.dino_center, before)
        assert not torch.allclose(method.ibot_center, torch.zeros_like(method.ibot_center))

    def test_gradients_flow_to_student_only(self):
        method = make_method()
        loss = method.training_step(make_views(), 0)
        loss.backward()
        assert any(
            p.grad is not None
            for p in method.embedding_model.parameters()
        )
        assert not any(
            p.grad is not None
            for p in method.teacher_embedding_model.parameters()
        )

    def test_mask_token_actually_reaches_student(self):
        # Masking must change the student patch features (mask token path).
        method = make_method()
        x = torch.randn(2, 3, 32, 32)
        mask = torch.zeros(2, 16, dtype=torch.bool)
        mask[:, 0] = True
        with_mask = method.embedding_model(x, pool=False, bool_masked_pos=mask)
        without_mask = method.embedding_model(x, pool=False)
        assert not torch.allclose(with_mask, without_mask)

    def test_train_keeps_teachers_in_eval(self):
        method = make_method()
        method.train()
        assert not method.teacher_embedding_model.training
        assert not method.teacher_projection_head.training


class TestLayerwiseParamGroups:
    def test_group_factors_and_lrs(self):
        method = make_method()
        groups, factors = method._layerwise_param_groups()
        names = {group["name"] for group in groups}
        # num_layers=2: block 0 → 0.9, block 1 → 1.0, patch embed → 0.2.
        assert "backbone_wd_0.2" in names
        assert "backbone_wd_0.9" in names
        assert factors["backbone_wd_0.2"] == 0.2
        assert factors["backbone_wd_0.9"] == 0.9

    def test_optimizer_lrs_scale_by_factor(self):
        method = make_method()
        optimizer, _ = method.configure_optimizers(total_steps=10, num_epochs=2)
        lr_by_name = {
            group.get("name"): group["lr"] for group in optimizer.param_groups
        }
        base = lr_by_name["params"]
        assert lr_by_name["backbone_wd_0.2"] == pytest.approx(base * 0.2)
        assert lr_by_name["backbone_wd_0.9"] == pytest.approx(base * 0.9)
        assert lr_by_name["params"] == base

    def test_no_weight_decay_groups_excluded_from_wd_schedule(self):
        method = make_method()
        optimizer, _ = method.configure_optimizers(total_steps=10, num_epochs=2)
        method.on_before_optimizer_step(optimizer)
        scheduled = 0
        for group in optimizer.param_groups:
            name = group.get("name") or ""
            if "no_wd" in name:
                # Both backbone no-decay groups and the projection head's
                # params_no_weight_decay group must stay at 0.
                assert group["weight_decay"] == 0.0
            else:
                scheduled += 1
        assert scheduled > 0
        assert any("params_no_weight_decay" in (g.get("name") or "") for g in optimizer.param_groups)

    def test_center_tracks_pre_softmax_teacher_outputs(self):
        method = make_method()
        views = make_views()
        method.training_step(views, 0)
        # Teacher weights are EMA-updated before the teacher forward and
        # untouched afterwards (no optimizer step here), so re-running the
        # teacher forward reproduces the exact outputs the center saw.
        with torch.no_grad():
            raw = method._forward_teacher(torch.cat(views["views"][:2]))[0]
        n0, n1 = (
            torch.nn.functional.normalize(view, dim=-1) for view in raw.chunk(2)
        )
        expected = torch.cat([n0, n1]).mean(dim=0, keepdim=True) * (1 - 0.9)
        assert torch.allclose(method.dino_center, expected, atol=1e-6)

    def test_ibot_center_tracks_pre_softmax_tokens(self):
        method = make_method()
        out_dim = method.method_args.output_dim
        teacher = torch.zeros(2, 4, out_dim)
        teacher[:, 0, :] = 3.0  # non-uniform raw tokens
        student = torch.zeros(2, 4, out_dim)
        mask = torch.tensor(
            [[True, False, True, False], [False, True, False, True]]
        )
        method._ibot_loss([teacher], [student], [mask], teacher_temp=0.07)
        normalized = torch.nn.functional.normalize(teacher, dim=-1)
        selected = mask.reshape(-1)
        expected = (
            normalized.reshape(-1, out_dim)[selected]
            .mean(dim=0, keepdim=True)
            * (1 - 0.9)
        )
        assert torch.allclose(method.ibot_center, expected, atol=1e-6)


class TestCheckpoint:
    def test_teachers_stripped_and_restored(self):
        method = make_method()
        checkpoint = {"state_dict": method.state_dict()}
        method.on_save_checkpoint(checkpoint)
        assert not any(
            key.startswith("teacher_") for key in checkpoint["state_dict"]
        )
        fresh_state = make_method().state_dict()
        fresh_state.update(checkpoint["state_dict"])
        method.on_load_checkpoint(fresh_state)
        assert any(
            key.startswith("teacher_embedding_model.") for key in fresh_state
        )
        assert any(
            key.startswith("teacher_projection_head.") for key in fresh_state
        )
