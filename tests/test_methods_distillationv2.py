# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the DistillationV2 pretraining method."""

import pytest
import torch
from tests.helpers_models import TinyMultiscaleWrapper

from autovision.encoders import FeatureExtractor
from autovision.pretext import get_method_cls, list_methods
from autovision.pretext.distillationv2 import (
    DistillationV2,
    DistillationV2Args,
    DistillationV2LARSArgs,
    DistillationV2TransformArgs,
)
from autovision.scaling import ScalingInfo

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def make_method(
    student_dim: int = 8,
    teacher_dim: int = 8,
    student_grid: int = 4,
    teacher_grid: int = 4,
    **args_overrides,
) -> DistillationV2:
    args = DistillationV2Args(teacher="unused", **args_overrides)
    args.resolve_auto_fields(scaling=ScalingInfo(dataset_size=64, epochs=10))
    optimizer_args = DistillationV2.optimizer_args_cls("auto")()
    optimizer_args.resolve_auto_fields(
        wrapped_model=TinyMultiscaleWrapper(dim=teacher_dim)
    )
    student = FeatureExtractor(
        wrapped_model=TinyMultiscaleWrapper(dim=student_dim, grid=student_grid),
        embed_dim=student_dim,
    )
    teacher_wrapper = TinyMultiscaleWrapper(dim=teacher_dim, grid=teacher_grid)
    args.teacher = teacher_wrapper  # instance teacher: no downloads
    return DistillationV2(
        method_args=args,
        optimizer_args=optimizer_args,
        embedding_model=student,
        global_batch_size=8,
        num_input_channels=3,
    )


def test_registered():
    assert "distillationv2" in list_methods()
    assert get_method_cls("distillationv2") is DistillationV2


class TestArgs:
    def test_defaults(self):
        args = DistillationV2Args()
        assert args.teacher == "dinov2/vits14"
        assert args.teacher_layers == "auto"
        assert args.reference_batch_size == 1536
        assert args.lr_scale_method == "sqrt"

    def test_lars_recipe(self):
        recipe = DistillationV2LARSArgs()
        assert recipe.lr == 9.0
        assert recipe.momentum == 0.9
        assert recipe.weight_decay == 1e-6

    def test_auto_optimizer_is_lars(self):
        assert DistillationV2.optimizer_args_cls("auto") is DistillationV2LARSArgs

    def test_transform_recipe(self):
        args = DistillationV2TransformArgs()
        assert args.image_size == (224, 224)
        assert args.solarize is None

    def test_bad_layer_pair_raises(self):
        args = DistillationV2Args(teacher_layers=(3, 1))
        with pytest.raises(ValueError, match="ordered"):
            args.resolve_auto_fields()

    def test_negative_layers_raise(self):
        args = DistillationV2Args(teacher_layers=(-1, 1))
        with pytest.raises(ValueError, match="0-based"):
            args.resolve_auto_fields()

    def test_bad_layer_length_raises(self):
        # A 3-tuple fails pydantic validation at construction time.
        with pytest.raises(ValueError):
            DistillationV2Args(teacher_layers=(1, 2, 3))

    def test_layers_beyond_depth_raise(self):
        # TinyMultiscaleWrapper has num_hidden_layers=2; (0, 5) must fail at
        # construction with a clear message instead of an IndexError later.
        with pytest.raises(ValueError, match="exceed"):
            make_method(teacher_layers=(0, 5))


class TestConstruction:
    def test_dim_mismatch_raises(self):
        with pytest.raises(ValueError, match="must equal"):
            make_method(student_dim=8, teacher_dim=16)

    def test_auto_layers_resolve_to_last_two(self):
        method = make_method()
        assert method.teacher_layers == (0, 1)  # num_hidden_layers=2
        assert method.student_layers == (0, 1)

    def test_explicit_layers_apply_to_both(self):
        method = make_method(teacher_layers=(0, 0))
        assert method.teacher_layers == (0, 0)
        assert method.student_layers == (0, 0)

    def test_trainable_modules_student_only(self):
        method = make_method()
        modules = list(method.trainable_modules().targets)
        assert method.embedding_model in modules
        assert method.teacher_embedding_model not in modules

    def test_concat_patches_grid_mismatch_raises(self):
        outputs = [
            {"features": torch.zeros(2, 4, 4, 4)},
            {"features": torch.zeros(2, 4, 2, 2)},
        ]
        with pytest.raises(ValueError, match="patch grid"):
            DistillationV2._concat_patches(outputs)


class TestTrainingStep:
    def test_loss_decreases(self):
        torch.manual_seed(0)
        method = make_method()
        optimizer, scheduler = method.configure_optimizers(
            total_steps=8, num_epochs=2
        )
        losses = []
        for step in range(8):
            method.global_step = step
            loss = method.training_step(
                {"views": [torch.randn(4, 3, 32, 32)]}, step
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
        assert losses[-1] < losses[0]

    def test_log_dict_has_mse(self):
        method = make_method()
        method.training_step({"views": [torch.randn(2, 3, 32, 32)]}, 0)
        assert "train_loss/mse" in method.consume_log_dict()

    def test_grid_misalignment_is_aligned(self):
        # Student grid 8 vs teacher grid 4: bilinear alignment must run and
        # the loss stays finite.
        method = make_method(student_grid=8, teacher_grid=4)
        loss = method.training_step({"views": [torch.randn(2, 3, 32, 32)]}, 0)
        assert torch.isfinite(loss)

    def test_gradients_flow_to_student_only(self):
        method = make_method()
        loss = method.training_step({"views": [torch.randn(2, 3, 32, 32)]}, 0)
        loss.backward()
        student_grads = [
            p.grad for p in method.embedding_model.parameters() if p.grad is not None
        ]
        assert student_grads
        teacher_grads = [
            p.grad
            for p in method.teacher_embedding_model.parameters()
            if p.grad is not None
        ]
        assert not teacher_grads

    def test_train_keeps_teacher_in_eval(self):
        method = make_method()
        method.train()
        assert not method.teacher_embedding_model.training
        assert method.embedding_model.training


class TestCheckpoint:
    def test_teacher_stripped_on_save_and_restored_on_load(self):
        method = make_method()
        checkpoint = {"state_dict": method.state_dict()}
        method.on_save_checkpoint(checkpoint)
        assert not any(
            key.startswith("teacher_embedding_model.") for key in checkpoint["state_dict"]
        )
        # A fresh method re-inserts teacher weights so load stays strict.
        restored = make_method()
        fresh_state = restored.state_dict()
        fresh_state.update(checkpoint["state_dict"])
        restored.on_load_checkpoint(fresh_state)
        assert "teacher_embedding_model.proj.weight" in fresh_state

    def test_checkpoint_roundtrip_strict_load(self):
        # Saved-then-loaded checkpoint must satisfy strict=True against a
        # fresh method (teacher keys re-inserted on load).
        method = make_method()
        checkpoint = {"state_dict": method.state_dict()}
        method.on_save_checkpoint(checkpoint)
        method.on_load_checkpoint(checkpoint)
        fresh = make_method()
        fresh.load_state_dict(checkpoint["state_dict"], strict=True)
