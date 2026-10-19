# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the DINO pretraining method."""

import pytest
import torch
from tests.helpers_models import TinyWrapper
from torch import nn

from autovision.encoders import FeatureExtractor
from autovision.pretext import get_method_cls, list_methods
from autovision.pretext.dino import (
    DINO,
    DINOAdamWArgs,
    DINOArgs,
    DINOLoss,
    DINOProjectionHead,
    DINOSGDArgs,
    DINOTransform,
    DINOTransformArgs,
)
from autovision.scaling import ScalingInfo

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

SMALL_DATASET_ARGS = dict(
    output_dim=32,
    teacher_temp=0.04,
    warmup_teacher_temp=0.02,
    momentum_start=0.99,
    weight_decay_start=0.04,
    weight_decay_end=0.4,
    warmup_teacher_temp_steps=4,
    student_freeze_last_layer_steps=2,
)


def make_dino(embed_dim: int = 8, **args_overrides) -> DINO:
    overrides = dict(SMALL_DATASET_ARGS)
    overrides.update(args_overrides)
    method_args = DINOArgs(**overrides)
    method_args.resolve_auto_fields(
        scaling=ScalingInfo(dataset_size=64, epochs=10),
        optimizer_args=DINOSGDArgs(),
    )
    optimizer_args = DINO.optimizer_args_cls("auto")()
    optimizer_args.resolve_auto_fields(wrapped_model=None)
    embedding_model = FeatureExtractor(
        wrapped_model=TinyWrapper(dim=embed_dim), embed_dim=None
    )
    return DINO(
        method_args=method_args,
        optimizer_args=optimizer_args,
        embedding_model=embedding_model,
        global_batch_size=8,
        num_input_channels=3,
    )


def _set_total_steps(method: DINO, total_steps: int = 10) -> DINO:
    # The engine assigns this before the loop (design: engine owns schedules).
    method.total_steps = total_steps
    return method


def make_batch(batch_size: int = 2, size: int = 32, num_local: int = 0) -> dict:
    views = [torch.randn(batch_size, 3, size, size) for _ in range(2)]
    views += [torch.randn(batch_size, 3, size // 2, size // 2) for _ in range(num_local)]
    return {"views": views}


def test_registered():
    assert "dino" in list_methods()
    assert get_method_cls("dino") is DINO


class TestDINOArgs:
    def test_auto_resolution_small_dataset(self):
        args = DINOArgs()
        args.resolve_auto_fields(
            scaling=ScalingInfo(dataset_size=10_000, epochs=100),
            optimizer_args=DINOSGDArgs(),
        )
        assert args.output_dim == 1024  # < 20k bucket
        assert args.teacher_temp == 0.02
        assert args.momentum_start == 0.99
        assert args.warmup_teacher_temp_steps == 37500
        assert args.student_freeze_last_layer_steps == 1250
        assert args.weight_decay_start == 0.0001  # from DINOSGDArgs

    def test_auto_resolution_large_dataset(self):
        args = DINOArgs()
        args.resolve_auto_fields(
            scaling=ScalingInfo(dataset_size=1_000_000, epochs=100),
            optimizer_args=DINOAdamWArgs(),
        )
        assert args.output_dim == 65536
        assert args.teacher_temp == 0.07
        assert args.momentum_start == 0.996
        assert args.weight_decay_start == 0.04

    def test_auto_requires_dataset_size(self):
        args = DINOArgs()
        with pytest.raises(ValueError, match="dataset size"):
            args.resolve_auto_fields(
                scaling=None, optimizer_args=DINOSGDArgs()
            )

    def test_non_numeric_weight_decay_rejected(self):
        args = DINOArgs()
        with pytest.raises(ValueError, match="weight decay"):
            args.resolve_auto_fields(
                scaling=ScalingInfo(dataset_size=100, epochs=10),
                optimizer_args=None,
            )


class TestDINOProjectionHead:
    def test_output_shape(self):
        head = DINOProjectionHead(input_dim=8, hidden_dim=32, bottleneck_dim=16, output_dim=64)
        assert head(torch.randn(4, 8)).shape == (4, 64)

    def test_norm_last_layer_frozen(self):
        head = DINOProjectionHead(
            input_dim=8, hidden_dim=32, bottleneck_dim=16, output_dim=64,
            norm_last_layer=True,
        )
        assert head.last_layer.weight_g.requires_grad is False

    def test_norm_last_layer_trainable(self):
        head = DINOProjectionHead(
            input_dim=8, hidden_dim=32, bottleneck_dim=16, output_dim=64,
            norm_last_layer=False,
        )
        assert head.last_layer.weight_g.requires_grad is True


class TestDINOLoss:
    def test_minimal_with_matching_distributions(self):
        loss_fn = DINOLoss(output_dim=16, teacher_temp=0.07, warmup_teacher_temp=0.04)
        torch.manual_seed(0)
        teacher = [torch.randn(4, 16), torch.randn(4, 16)]
        student = [torch.randn(4, 16) for _ in range(8)]
        loss = loss_fn(teacher, student, teacher_temp=0.07)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_center_moves_towards_teacher(self):
        loss_fn = DINOLoss(output_dim=8, teacher_temp=0.07, warmup_teacher_temp=0.04)
        torch.manual_seed(0)
        teacher = [torch.full((4, 8), 5.0)]
        student = [torch.randn(4, 8)]
        before = loss_fn.center.clone()
        loss_fn(teacher, student, teacher_temp=0.07)
        assert not torch.equal(before, loss_fn.center)
        assert (loss_fn.center > before).all()

    def test_center_tracks_pre_softmax_outputs(self):
        # The center must EMA-track the raw (pre-softmax) teacher outputs
        # it is subtracted from, matching the baseline update_center call.
        loss_fn = DINOLoss(output_dim=8, teacher_temp=0.07, warmup_teacher_temp=0.04)
        torch.manual_seed(0)
        teacher = [torch.randn(4, 8), torch.randn(4, 8)]
        student = [torch.randn(4, 8)]
        loss_fn(teacher, student, teacher_temp=0.07)
        normalized = [
            torch.nn.functional.normalize(view, dim=-1) for view in teacher
        ]
        expected = torch.cat(normalized).mean(dim=0, keepdim=True) * (1 - 0.9)
        assert torch.allclose(loss_fn.center, expected, atol=1e-6)

    def test_identical_global_pair_skipped(self):
        loss_fn = DINOLoss(output_dim=8, teacher_temp=0.07, warmup_teacher_temp=0.04)
        teacher = [torch.zeros(2, 8), torch.ones(2, 8)]
        student = [torch.zeros(2, 8), torch.zeros(2, 8)]
        # With the identical pair skipped the loss is finite, not NaN from
        # a zero-vs-zero cross entropy.
        assert torch.isfinite(loss_fn(teacher, student, teacher_temp=0.07))


class TestDINOTransform:
    def test_view_count_and_sizes(self):
        args = DINOTransformArgs()
        transform = DINOTransform(args)
        views = transform({"image": torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)})
        assert len(views) == 2 + args.num_local_views
        assert views[0]["image"].shape == (3, 224, 224)
        assert views[-1]["image"].shape == (3, 96, 96)

    def test_local_views_disabled(self):
        args = DINOTransformArgs(num_local_views=0)
        transform = DINOTransform(args)
        views = transform({"image": torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)})
        assert len(views) == 2


class TestDINOMethod:
    def test_training_step_runs_and_logs_schedules(self):
        method = make_dino()
        result = method.training_step_impl(make_batch(), batch_idx=0)
        assert torch.isfinite(result.loss)
        assert "schedule/momentum" in result.log_dict
        assert "schedule/teacher_temp" in result.log_dict

    def test_teacher_tracks_student(self):
        method = _set_total_steps(make_dino())
        # Teacher starts as an exact copy of the student; perturb the student
        # so the EMA update has something to move towards.
        with torch.no_grad():
            for param in method.embedding_model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        teacher_param = next(method.teacher_embedding_model.parameters())
        before = teacher_param.detach().clone()
        method.training_step_impl(make_batch(), batch_idx=0)
        assert not torch.equal(before, teacher_param)
        # Momentum 0.99 keeps the teacher mostly at its old value.
        moved = float((teacher_param - before).abs().mean())
        gap = float(
            (
                next(method.embedding_model.parameters()).detach() - before
            ).abs().mean()
        )
        assert 0 < moved < gap

    def test_teacher_stays_in_eval(self):
        method = make_dino()
        method.train()
        assert method.embedding_model.training  # student trains
        assert not method.teacher_embedding_model.training
        assert not method.teacher_projection_head.training

    def test_loss_decreases_over_steps(self):
        torch.manual_seed(0)
        method = make_dino()
        optimizer, scheduler = method.configure_optimizers(
            total_steps=10, num_epochs=1
        )
        losses = []
        for step in range(10):
            method.global_step = step
            result = method.training_step_impl(make_batch(num_local=2), step)
            optimizer.zero_grad()
            result.loss.backward()
            method.clip_gradients(optimizer)
            method.on_before_optimizer_step(optimizer)
            optimizer.step()
            scheduler.step()
            losses.append(float(result.loss.detach()))
        assert losses[-1] < losses[0]

    def test_last_layer_frozen_then_released(self):
        method = make_dino()
        optimizer, _ = method.configure_optimizers(total_steps=10, num_epochs=1)
        last_group = next(
            group for group in optimizer.param_groups
            if group.get("name") == "params_last_layer"
        )
        method.global_step = 0
        method.on_before_optimizer_step(optimizer)
        assert last_group["lr"] == 0.0
        assert last_group["weight_decay"] == 0.0
        # After the freeze window the hook stops touching lr (the scheduler
        # owns it) and applies the cosine weight decay instead.
        method.global_step = method.method_args.student_freeze_last_layer_steps
        last_group["lr"] = 0.05  # pretend the scheduler set its scheduled lr
        method.on_before_optimizer_step(optimizer)
        assert last_group["lr"] == 0.05
        assert last_group["weight_decay"] > 0.0

    def test_checkpoint_strips_teacher(self, tmp_path):
        method = make_dino()
        checkpoint = {"state_dict": method.state_dict()}
        method.on_save_checkpoint(checkpoint)
        assert not any(
            key.startswith("teacher_") for key in checkpoint["state_dict"]
        )
        # Roundtrip: teacher keys are re-inserted so strict loading works.
        method.on_load_checkpoint(checkpoint)
        method.load_state_dict(checkpoint["state_dict"])

    def test_transform_views_match_chunking(self):
        method = make_dino()
        batch = make_batch(num_local=2)
        views = batch["views"]
        assert len(views) == 4  # 2 global + 2 local
        student_out = torch.cat(
            [method._forward_student(torch.cat(views[:2])), method._forward_student(torch.cat(views[2:]))]
        )
        assert student_out.chunk(len(views))[0].shape == (2, method.method_args.output_dim)


class TestReviewFixes:
    def test_teacher_parameters_frozen(self):
        # Under DDP every registered parameter must either receive a
        # gradient or have requires_grad=False (P0 review finding).
        method = make_dino()
        for parameter in method.teacher_embedding_model.parameters():
            assert parameter.requires_grad is False
        for parameter in method.teacher_projection_head.parameters():
            assert parameter.requires_grad is False

    def test_momentum_updates_buffers(self):
        # A BN teacher must follow the student's running stats (P1 finding).
        from autovision.pretext.dino.dino import _update_momentum

        student = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        teacher = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        for teacher_param, student_param in zip(
            teacher.parameters(), student.parameters()
        ):
            teacher_param.detach().copy_(student_param.detach())
        with torch.no_grad():
            for batch_index in range(5):
                student(torch.randn(2, 3, 8, 8))
        student_bn = student[1]
        teacher_bn = teacher[1]
        assert not torch.allclose(
            teacher_bn.running_mean, student_bn.running_mean
        )
        expected = 0.5 * teacher_bn.running_mean + 0.5 * student_bn.running_mean
        _update_momentum(student, teacher, momentum=0.5)
        assert torch.allclose(teacher_bn.running_mean, expected)
        # Integer buffers (num_batches_tracked) are copied, not averaged:
        # the teacher mirrors the student's counter instead of mixing EMA
        # averages with an arbitrary count.
        assert teacher_bn.num_batches_tracked.item() == student_bn.num_batches_tracked.item()
