# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the DINOv3 pretraining method (DINOv2 + PaKA post-train)."""

import random

import pytest
import torch
from tests.test_methods_dinov2 import TinyMaskedViTWrapper

from autovision.augment.transform_args import (
    NormalizeArgs,
)
from autovision.encoders import FeatureExtractor
from autovision.pretext import get_method_cls, list_methods
from autovision.pretext.dinov3 import DINOv3, DINOv3Args, DINOv3Transform
from autovision.pretext.dinov3.constrained_crop import (
    _GeometryViewTransform,
    render_clean_global,
    sample_high_overlap_box,
)
from autovision.pretext.dinov3.paika import _linear_cka_loss, _roi_resample_to_grid
from autovision.scaling import ScalingInfo

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ── fixtures ────────────────────────────────────────────────────────────


def make_method(**args_overrides) -> DINOv3:
    from autovision.pretext.dinov2 import DINOv2AdamWArgs

    args = DINOv3Args(
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
        wrapped_model=TinyMaskedViTWrapper(grid=4), embed_dim=8
    )
    return DINOv3(
        method_args=args,
        optimizer_args=optimizer_args,
        embedding_model=student,
        global_batch_size=4,
        num_input_channels=3,
    )


def make_batch(batch_size: int = 2, num_locals: int = 2, paka_locals: int = 4) -> dict:
    """Views in the DINOv3 layout + valid non-overlapping geometries."""
    total = 2 + num_locals + 2 + paka_locals
    views = [torch.randn(batch_size, 3, 32, 32) for _ in range(total)]
    # Globals cover the full image; paka locals sit inside (containment 1.0)
    # so every cross-view pair overlaps.
    global_geom = [0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 0.0, 0.0]
    local_box = [10.0, 10.0, 40.0, 40.0, 100.0, 100.0, 0.0, 0.0]
    geometries = []
    for view_index in range(total):
        row = global_geom if view_index < 2 or 2 + num_locals <= view_index < 2 + num_locals + 2 else local_box
        geometries.append(torch.tensor([row] * batch_size))
    return {"views": views, "geometries": geometries}


# ── args ────────────────────────────────────────────────────────────────


class TestArgs:
    def test_defaults(self):
        args = DINOv3Args()
        assert args.paka_weight == 1.0
        assert args.paka_start_step == 0
        assert args.paka_num_local == 8
        assert args.paka_max_tokens == 512
        assert args.auto_steps == 125_000

    def test_bad_paka_num_local_raises(self):
        with pytest.raises(ValueError, match="paka_num_local"):
            DINOv3Args(paka_num_local=-1).resolve_auto_fields()

    def test_bad_paka_weight_raises(self):
        with pytest.raises(ValueError, match="paka_weight"):
            DINOv3Args(paka_weight=0.0).resolve_auto_fields()

    def test_registered(self):
        assert "dinov3" in list_methods()
        assert get_method_cls("dinov3") is DINOv3


# ── constrained crop ────────────────────────────────────────────────────


class TestConstrainedCrop:
    def test_sample_box_inside_parent(self):
        rng = random.Random(0)
        parent = (10, 20, 110, 220)  # 100x200
        for _ in range(50):
            box = sample_high_overlap_box(parent, 0.9, rng)
            assert parent[0] <= box[0] < box[2] <= parent[2]
            assert parent[1] <= box[1] < box[3] <= parent[3]
            area = (box[2] - box[0]) * (box[3] - box[1])
            parent_area = 100 * 200
            assert 0.04 < area / parent_area <= 0.41

    def test_sample_box_degenerate_parent(self):
        assert sample_high_overlap_box((5, 5, 6, 6), 0.9, random.Random(0)) == (5, 5, 6, 6)

    def test_geometry_view_transform_records_geometry(self):
        transform = _GeometryViewTransform(
            image_size=(32, 32),
            random_resize=(0.5, 1.0),
            random_flip_h=0.5,
            random_flip_v=0.0,
            color_jitter=None,
            random_gray_scale=None,
            gaussian_blur=None,
            solarize=None,
            normalize=NormalizeArgs(),
        )
        image = torch.randint(0, 255, (3, 64, 80), dtype=torch.uint8)
        out = transform(image)
        assert out["image"].shape == (3, 32, 32)
        assert out["image"].dtype == torch.float32
        geometry = out["geometry"]
        assert geometry.shape == (8,)
        assert 0 <= geometry[0] < geometry[2] <= 80
        assert 0 <= geometry[1] < geometry[3] <= 64
        assert geometry[4] == 80 and geometry[5] == 64
        assert geometry[6] in (0.0, 1.0) and geometry[7] == 0.0

    def test_clean_global_reuses_geometry_and_skips_photometric(self):
        normalize = NormalizeArgs()
        image = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        geometry = torch.tensor([8.0, 8.0, 56.0, 56.0, 64.0, 64.0, 1.0, 0.0])
        out = render_clean_global(image, geometry, (32, 32), normalize)
        assert out["image"].shape == (3, 32, 32)
        assert torch.equal(out["geometry"], geometry)
        # Deterministic: no photometric randomness for the clean render.
        again = render_clean_global(image, geometry, (32, 32), normalize)
        assert torch.allclose(out["image"], again["image"])


# ── transform layout ────────────────────────────────────────────────────


class TestTransform:
    def make_transform(self, **overrides) -> DINOv3Transform:
        from autovision.pretext.dinov3 import DINOv3TransformArgs

        args = DINOv3TransformArgs(
            image_size=(32, 32),
            local_view_size=(16, 16),
            num_local_views=2,
            **{"paka_num_local": 4, **overrides},
        )
        return DINOv3Transform(args)

    def test_view_layout_and_geometry_keys(self):
        transform = self.make_transform()
        image = torch.randint(0, 255, (3, 96, 96), dtype=torch.uint8)
        views = transform({"image": image})
        assert len(views) == 2 + 2 + 2 + 4
        assert all("geometry" in view for view in views)
        # Clean globals render at the global size, paka locals at local size.
        assert views[4]["image"].shape == (3, 32, 32)
        assert views[5]["image"].shape == (3, 32, 32)
        assert views[6]["image"].shape == (3, 16, 16)
        # Clean globals reuse the global geometry verbatim.
        assert torch.equal(views[4]["geometry"], views[0]["geometry"])
        assert torch.equal(views[5]["geometry"], views[1]["geometry"])

    def test_paka_locals_inside_parent(self):
        transform = self.make_transform()
        image = torch.randint(0, 255, (3, 96, 96), dtype=torch.uint8)
        views = transform({"image": image})
        for index in range(6, 10):
            geometry = views[index]["geometry"]
            parent = views[(index - 6) % 2]["geometry"]
            assert geometry[0] >= parent[0] - 1e-4
            assert geometry[1] >= parent[1] - 1e-4
            assert geometry[2] <= parent[2] + 1e-4
            assert geometry[3] <= parent[3] + 1e-4

    def test_zero_paka_locals(self):
        transform = self.make_transform(paka_num_local=0)
        image = torch.randint(0, 255, (3, 96, 96), dtype=torch.uint8)
        views = transform({"image": image})
        assert len(views) == 2 + 2 + 2

    def test_paka_sampling_seeded_by_torch_rng(self):
        # Paka local sampling must be reproducible under the global seeding
        # conventions (torch manual seed); a fresh random.Random() would
        # draw from OS entropy and make seeded runs non-deterministic.
        import random as random_module

        transform = self.make_transform()
        image = torch.randint(0, 255, (3, 96, 96), dtype=torch.uint8)
        torch.manual_seed(0)
        random_module.seed(0)
        views_a = transform({"image": image})
        torch.manual_seed(0)
        random_module.seed(0)
        views_b = transform({"image": image})
        assert len(views_a) == len(views_b)
        for a, b in zip(views_a, views_b):
            assert torch.allclose(a["image"], b["image"])
            assert torch.equal(a["geometry"], b["geometry"])


# ── PaKA math ───────────────────────────────────────────────────────────


class TestPaika:
    def test_identical_features_zero_loss(self):
        features = torch.randn(2, 6, 8)
        valid = torch.ones(2, 6, dtype=torch.bool)
        loss = _linear_cka_loss(features, features.clone(), valid)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_identical_lower_loss_than_independent(self):
        torch.manual_seed(0)
        student = torch.randn(2, 6, 8)
        teacher = torch.randn(2, 6, 8)
        valid = torch.ones(2, 6, dtype=torch.bool)
        assert float(_linear_cka_loss(student, student.clone(), valid)) == pytest.approx(
            0.0, abs=1e-6
        )
        independent = float(_linear_cka_loss(student, teacher, valid))
        assert independent > 0.3

    def test_invalid_pairs_skipped_with_zero_grad_path(self):
        student = torch.randn(1, 6, 8, requires_grad=True)
        teacher = torch.randn(1, 6, 8)
        valid = torch.zeros(1, 6, dtype=torch.bool)
        loss = _linear_cka_loss(student, teacher, valid)
        assert float(loss) == 0.0
        loss.backward()
        assert student.grad is not None
        assert float(student.grad.abs().sum()) == 0.0

    def test_max_tokens_subsample(self):
        torch.manual_seed(0)
        student = torch.randn(1, 100, 8)
        teacher = torch.randn(1, 100, 8)
        valid = torch.ones(1, 100, dtype=torch.bool)
        loss = _linear_cka_loss(student, teacher, valid, max_tokens=16)
        assert torch.isfinite(loss)

    def test_roi_resample_identity_box(self):
        feat = torch.randn(2, 4, 5, 5)
        boxes = torch.tensor([[0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 5.0, 5.0]])
        out = _roi_resample_to_grid(feat, boxes, out_h=5, out_w=5)
        assert out.shape == (2, 4, 5, 5)
        assert torch.allclose(out, feat, atol=1e-5)

    def test_roi_resample_shape(self):
        feat = torch.randn(2, 4, 5, 5)
        boxes = torch.tensor([[1.0, 1.0, 4.0, 4.0]] * 2)
        out = _roi_resample_to_grid(feat, boxes, out_h=3, out_w=7)
        assert out.shape == (2, 4, 3, 7)


# ── method ──────────────────────────────────────────────────────────────


class TestMethod:
    def test_training_step_has_paka_term(self):
        method = make_method(paka_num_local=4)
        method.total_steps = 10
        result = method.training_step_impl(make_batch(), 0)
        assert torch.isfinite(result.loss)
        assert "train_loss/paka" in result.log_dict

    def test_gradients_reach_paka_head_only_via_student(self):
        method = make_method(paka_num_local=4)
        method.total_steps = 10
        loss = method.training_step(make_batch(), 0)
        loss.backward()
        assert any(p.grad is not None for p in method.student_paka_head.parameters())
        assert not any(
            p.grad is not None for p in method.teacher_paka_head.parameters()
        )

    def test_paka_start_step_defers_the_loss(self):
        method = make_method(paka_num_local=4, paka_start_step=5)
        method.total_steps = 10
        method.global_step = 0
        result = method.training_step_impl(make_batch(), 0)
        assert "train_loss/paka" not in result.log_dict

    def test_too_few_views_raise(self):
        method = make_method(paka_num_local=4)
        method.total_steps = 10
        batch = make_batch()
        batch["views"] = batch["views"][:3]
        batch["geometries"] = batch["geometries"][:3]
        with pytest.raises(ValueError, match="at least 2 global views"):
            method.training_step_impl(batch, 0)

    def test_missing_geometries_skip_paka(self):
        method = make_method(paka_num_local=4)
        method.total_steps = 10
        batch = make_batch()
        batch.pop("geometries")
        result = method.training_step_impl(batch, 0)
        assert "train_loss/paka" not in result.log_dict

    def test_train_keeps_paka_teacher_eval(self):
        method = make_method()
        method.train()
        assert not method.teacher_paka_head.training

    def test_trainable_modules_include_paka_head(self):
        method = make_method()
        modules = method.trainable_modules().targets
        assert method.student_paka_head in modules

    def test_optimizer_covers_paka_head(self):
        method = make_method()
        optimizer, _ = method.configure_optimizers(total_steps=10, num_epochs=2)
        grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
        for parameter in method.student_paka_head.parameters():
            assert id(parameter) in grouped

    def test_checkpoint_strips_all_teachers(self):
        method = make_method(paka_num_local=4)
        checkpoint = {"state_dict": method.state_dict()}
        method.on_save_checkpoint(checkpoint)
        assert not any(
            key.startswith("teacher_") for key in checkpoint["state_dict"]
        )
        fresh = make_method(paka_num_local=4).state_dict()
        fresh.update(checkpoint["state_dict"])
        method.on_load_checkpoint(fresh)
        assert any(
            key.startswith("teacher_paka_head.") for key in fresh
        )

    def test_loss_decreases(self):
        # paka_num_local=0 exercises the pure DINOv2 path; the PaKA term has
        # its own structural tests (CKA == 0 for identical features, etc.)
        # and its descent is too noisy to assert on a tiny model.
        torch.manual_seed(1)
        method = make_method(paka_num_local=0)
        method.total_steps = 10
        optimizer, scheduler = method.configure_optimizers(
            total_steps=10, num_epochs=2
        )
        losses = []
        for step in range(10):
            method.global_step = step
            loss = method.training_step(make_batch(paka_locals=0), step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        assert all(torch.isfinite(torch.tensor(value)) for value in losses)
        assert losses[-1] < losses[0]
