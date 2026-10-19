# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
import json
import math
from pathlib import Path

import pytest
import torch
from tests.helpers_models import TinyWrapper
from torch import nn

from autovision.encoders import FeatureExtractor
from autovision.pretext import PretextRecipe, StepOutput, get_method_cls, list_methods
from autovision.pretext.distillationv3 import (
    DistillationV3,
    DistillationV3AdamWArgs,
    DistillationV3Args,
    DistillationV3LARSArgs,
    DistillationV3Transform,
    DistillationV3TransformArgs,
)
from autovision.scaling import ScalingInfo

# ── registry ────────────────────────────────────────────────────────────


def test_list_methods():
    names = list_methods()
    assert "distillation" in names
    assert "distillationv3" in names


def test_get_method_cls_alias():
    assert get_method_cls("distillation") is DistillationV3
    assert get_method_cls("distillationv3") is DistillationV3
    with pytest.raises(ValueError, match="Unknown method"):
        get_method_cls("nonexistent")


# ── args ────────────────────────────────────────────────────────────────


class TestDistillationV3Args:
    def test_defaults(self):
        args = DistillationV3Args()
        assert args.queue_size == "auto"
        assert args.temperature_global == 0.07
        assert args.temperature_local == 0.07
        assert args.loss_local_weight == 1.0
        assert args.lr_scale_method == "sqrt"
        assert args.reference_batch_size == 1536
        assert args.auto_epochs == 100

    def test_queue_auto_bucketing(self):
        args = DistillationV3Args()
        args.resolve_auto_fields(scaling=ScalingInfo(dataset_size=800, epochs=100))
        assert args.queue_size == 256  # bucket <1000
        args = DistillationV3Args()
        args.resolve_auto_fields(scaling=ScalingInfo(dataset_size=50_000, epochs=100))
        assert args.queue_size == 8192

    def test_queue_larger_than_dataset_raises(self):
        args = DistillationV3Args(queue_size=100)
        with pytest.raises(ValueError, match="cannot be larger"):
            args.resolve_auto_fields(scaling=ScalingInfo(dataset_size=50, epochs=10))

    def test_explicit_queue_ok(self):
        args = DistillationV3Args(queue_size=16)
        args.resolve_auto_fields(scaling=ScalingInfo(dataset_size=50, epochs=10))
        assert args.queue_size == 16

    def test_optimizer_recipes(self):
        assert DistillationV3AdamWArgs().lr == 0.0005
        assert DistillationV3AdamWArgs().weight_decay == "auto"
        lars = DistillationV3LARSArgs()
        assert lars.lr == 1.8
        assert lars.momentum == 0.9
        assert lars.weight_decay == 1e-6
        assert lars.trust_coefficient == 0.001


class TestDistillationV3Transform:
    def test_recipe_defaults(self):
        args = DistillationV3TransformArgs()
        assert args.image_size == (224, 224)
        assert args.random_resize.min_scale == 0.14
        assert args.gaussian_blur.prob == 1.0
        assert args.gaussian_blur.sigmas == (0.0, 0.1)
        assert args.color_jitter.prob == 0.8
        assert args.solarize is None

    def test_single_view(self):
        transform = DistillationV3Transform(DistillationV3TransformArgs())
        out = transform({"image": torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)})
        assert len(out) == 1
        assert out[0]["image"].shape == (3, 224, 224)

    def test_transform_cls_binding(self):
        assert DistillationV3.transform_cls() is DistillationV3Transform
        assert DistillationV3.method_args_cls() is DistillationV3Args


# ── method construction & step ──────────────────────────────────────────


def make_method(
    dataset_size: int = 100,
    student_dim: int = 8,
    embed_dim: int = 16,
    teacher_dim: int = 8,
    queue_size: int | str = 16,
) -> DistillationV3:
    args = DistillationV3Args(queue_size=queue_size, teacher="unused")
    args.resolve_auto_fields(scaling=ScalingInfo(dataset_size=dataset_size, epochs=100))
    # Follow the real flow: method-specific optimizer args, then resolve auto.
    optimizer_args = DistillationV3.optimizer_args_cls("auto")()
    optimizer_args.resolve_auto_fields(
        wrapped_model=TinyWrapper(dim=teacher_dim, model_type="convolutional")
    )
    student = FeatureExtractor(
        wrapped_model=TinyWrapper(dim=student_dim), embed_dim=embed_dim
    )
    teacher_wrapper = TinyWrapper(dim=teacher_dim)
    args.teacher = teacher_wrapper  # instance teacher: no downloads
    method = DistillationV3(
        method_args=args,
        optimizer_args=optimizer_args,
        embedding_model=student,
        global_batch_size=4,
        num_input_channels=3,
    )
    return method


def make_batch(batch_size: int = 2, size: int = 16):
    torch.manual_seed(0)
    view = torch.randn(batch_size, 3, size, size)
    return {"filename": [f"img_{i}" for i in range(batch_size)], "views": [view]}


class TestDistillationV3Method:
    def test_structure(self):
        method = make_method()
        assert isinstance(method, PretextRecipe)
        assert method.teacher_embedding_dim == 8
        assert method.teacher_queue.shape == (16, 8)
        assert method.student_projection_head_global.in_features == 16
        assert method.student_projection_head_global.out_features == 8
        # teacher frozen and eval
        assert all(not p.requires_grad for p in method.teacher_embedding_model.parameters())
        assert not method.teacher_embedding_model.training
        # heads trunc-normal initialized with std 0.02
        std = method.student_projection_head_global.weight.std().item()
        assert 0.001 < std < 0.1

    def test_trainable_modules_exclude_teacher(self):
        method = make_method()
        modules = method.trainable_modules()
        all_named = dict(method.named_parameters())
        trainable_ids = {
            id(p) for m in modules.targets for p in m.parameters()
        }
        for name, param in all_named.items():
            if name.startswith("teacher_embedding_model."):
                assert id(param) not in trainable_ids

    def test_training_step_shapes_and_finite(self):
        method = make_method()
        batch = make_batch()
        result = method.training_step_impl(batch, batch_idx=0)
        assert isinstance(result, StepOutput)
        assert result.loss.ndim == 0
        assert torch.isfinite(result.loss)
        assert set(result.log_dict) == {
            "train_loss/global_loss",
            "train_loss/local_loss",
        }
        # base training_step stores the log dict
        loss = method.training_step(batch, batch_idx=0)
        assert loss.ndim == 0
        log_dict = method.consume_log_dict()
        assert log_dict is not None
        assert method.consume_log_dict() is None  # cleared

    def test_queue_update_semantics(self):
        method = make_method(queue_size=8)
        batch = make_batch(batch_size=2)
        with torch.no_grad():
            teacher_global, _, _ = method._forward_teacher(batch["views"][0])
            method._update_queue(teacher_global)
        # Current batch occupies the first slots (normalized).
        for i in range(2):
            assert torch.allclose(
                method.teacher_queue[i], teacher_global[i], atol=1e-6
            )
        # Older slots are still zero (initial queue).
        assert torch.count_nonzero(method.teacher_queue[2:]) == 0

    def test_queue_oversized_batch(self):
        method = make_method(queue_size=2)
        batch = make_batch(batch_size=4, size=8)
        method.training_step(batch, batch_idx=0)
        assert method.teacher_queue.shape[0] == 2

    def test_queue_exact_batch_size(self):
        # B == queue_size hits the replace branch (B >= queue_size).
        method = make_method(queue_size=4)
        batch = make_batch(batch_size=4, size=8)
        method.training_step(batch, batch_idx=0)
        assert method.teacher_queue.shape[0] == 4

    def test_student_single_forward_per_step(self):
        # The pooled and patch representations must come from one backbone
        # evaluation: two forwards would double-update BN running stats and
        # sample dropout twice.
        method = make_method()
        wrapper = method.student_embedding_model._inner
        original = wrapper.feature_maps
        calls = 0

        def counting(x, **kwargs):
            nonlocal calls
            calls += 1
            return original(x, **kwargs)

        wrapper.feature_maps = counting
        try:
            method.training_step(make_batch(), batch_idx=0)
        finally:
            wrapper.feature_maps = original
        assert calls == 1

    def test_loss_is_zero_when_student_matches_teacher(self):
        # Golden direction check: KLDivLoss(input=student_log, target=teacher)
        # reaches 0 exactly when the student distribution equals the teacher's.
        torch.manual_seed(0)
        method = make_method(
            student_dim=8, embed_dim=8, teacher_dim=8, queue_size=16
        )
        method.student_embedding_model._inner.load_state_dict(
            method.teacher_embedding_model.state_dict()
        )
        for head in (
            method.student_projection_head_global,
            method.student_projection_head_local,
        ):
            nn.init.eye_(head.weight)
            nn.init.zeros_(head.bias)
        result = method.training_step_impl(make_batch(batch_size=4), batch_idx=0)
        assert result.loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gradients_flow_to_student_only(self):
        method = make_method()
        loss = method.training_step(make_batch(), batch_idx=0)
        loss.backward()
        grads = {
            name: p.grad
            for name, p in method.named_parameters()
            if p.grad is not None
        }
        assert any(
            name.startswith("embedding_model._inner")
            or name.startswith("embedding_model._embed")
            for name in grads
        )
        assert any("projection_head" in name for name in grads)
        assert not any(name.startswith("teacher_embedding_model") for name in grads)

    def test_checkpoint_strips_teacher(self):
        method = make_method()
        checkpoint = {"state_dict": method.state_dict()}
        teacher_keys = [
            k for k in checkpoint["state_dict"] if k.startswith("teacher_embedding_model.")
        ]
        assert teacher_keys, "teacher must be part of the live state dict"
        method.on_save_checkpoint(checkpoint)
        assert not [
            k for k in checkpoint["state_dict"] if k.startswith("teacher_embedding_model.")
        ]
        assert "student_projection_head_global.weight" in checkpoint["state_dict"]

    def test_checkpoint_roundtrip_preserves_student(self, tmp_path):
        method = make_method()
        checkpoint = {"state_dict": method.state_dict()}
        method.on_save_checkpoint(checkpoint)
        method.on_load_checkpoint(checkpoint)  # restore teacher keys
        torch.save(checkpoint, tmp_path / "ckpt.pt")
        restored = torch.load(tmp_path / "ckpt.pt", weights_only=True)
        fresh = make_method()
        fresh.load_state_dict(restored["state_dict"], strict=True)

    def test_lr_scaling_sqrt(self):
        method = make_method()
        method.global_batch_size = 1536  # == reference -> scale 1.0
        optimizer, scheduler = method.configure_optimizers(total_steps=100, num_epochs=10)
        assert isinstance(optimizer, torch.optim.AdamW)
        # The LRScheduler constructor steps once (warmup starts at lr 0);
        # the unscaled base lr is the ground truth for the scaling formula.
        assert scheduler.base_lrs[0] == pytest.approx(0.0005 * math.sqrt(1.0))

    def test_lr_scaling_sqrt_half(self):
        method = make_method()
        method.global_batch_size = 384  # 1536/4 -> scale 0.5
        optimizer, scheduler = method.configure_optimizers(total_steps=100, num_epochs=10)
        assert scheduler.base_lrs[0] == pytest.approx(0.0005 * math.sqrt(0.25))

    def test_warmup_and_scheduler(self):
        method = make_method()
        optimizer, scheduler = method.configure_optimizers(total_steps=100, num_epochs=10)
        # 10 epochs of 100 total steps -> warmup = 100 * 10/100 = 10 steps
        assert scheduler.warmup_epochs == 10
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)  # step 0

    def test_gradient_clipping_value(self):
        method = make_method()
        assert method.gradient_clip_val == 1.0
        optimizer, _ = method.configure_optimizers(total_steps=100, num_epochs=10)
        loss = method.training_step(make_batch(), batch_idx=0)
        loss.backward()
        norm = method.clip_gradients(optimizer)
        assert norm is not None and norm >= 0

    def test_weight_decay_auto_resolved_by_architecture(self):
        method = make_method()
        # TinyWrapper declares convolutional/batchnorm -> wd 1e-6
        assert method.optimizer_args.weight_decay == 1e-6


# ── golden test: loss trajectory snapshot (design doc §13 L4) ───────────

GOLDEN_PATH = Path(__file__).parent / "golden" / "distillationv3_losses.json"
NUM_GOLDEN_STEPS = 20


def run_golden_trajectory() -> list[float]:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    method = make_method(dataset_size=1000, queue_size=64)
    optimizer, scheduler = method.configure_optimizers(total_steps=50, num_epochs=5)
    losses = []
    batch = make_batch(batch_size=2, size=16)
    for step in range(NUM_GOLDEN_STEPS):
        optimizer.zero_grad()
        loss = method.training_step(batch, batch_idx=step)
        loss.backward()
        method.clip_gradients(optimizer)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
    torch.use_deterministic_algorithms(False)
    return losses


class TestGoldenTrajectory:
    """Fixed-seed loss sequence snapshot: catches unintended recipe changes.

    Delete the golden file to re-record after an *intentional* change.
    """

    def test_loss_trajectory_matches_snapshot(self):
        losses = run_golden_trajectory()
        if not GOLDEN_PATH.exists():
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_text(json.dumps(losses, indent=2))
            pytest.skip("golden snapshot recorded; re-run to compare")
        golden = json.loads(GOLDEN_PATH.read_text())
        assert len(golden) == len(losses)
        for step, (expected, actual) in enumerate(zip(golden, losses)):
            assert actual == pytest.approx(expected, rel=0.005), (
                f"Loss trajectory diverged at step {step}: "
                f"expected {expected}, got {actual}. If this change is "
                f"intentional, delete {GOLDEN_PATH} to re-record."
            )
