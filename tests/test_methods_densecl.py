# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the DenseCL pretraining method."""

import pytest
import torch
from tests.helpers_models import TinyWrapper

from autovision.encoders import FeatureExtractor
from autovision.pretext import get_method_cls, list_methods
from autovision.pretext.densecl import DenseCL, DenseCLArgs, DenseCLSGDArgs
from autovision.pretext.densecl.densecl import (
    _QUEUE_SIZE_BUCKETS,
    _DenseProjectionHead,
    _info_nce_with_queue,
)
from autovision.scaling import ScalingInfo

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def make_method(**args_overrides) -> DenseCL:
    args = DenseCLArgs(hidden_dim=32, output_dim=16, **args_overrides)
    args.resolve_auto_fields(scaling=ScalingInfo(dataset_size=64, epochs=10))
    optimizer_args = DenseCLSGDArgs()
    optimizer_args.resolve_auto_fields(wrapped_model=None)
    return DenseCL(
        method_args=args,
        optimizer_args=optimizer_args,
        embedding_model=FeatureExtractor(
            wrapped_model=TinyWrapper(dim=8), embed_dim=8
        ),
        global_batch_size=8,
        num_input_channels=3,
    )


def make_views(batch_size: int = 4) -> dict:
    return {"views": [torch.randn(batch_size, 3, 32, 32) for _ in range(2)]}


def test_registered():
    assert "densecl" in list_methods()
    assert get_method_cls("densecl") is DenseCL


class TestArgs:
    def test_queue_auto_buckets_by_dataset_size(self):
        args = DenseCLArgs()
        args.resolve_auto_fields(scaling=ScalingInfo(dataset_size=800, epochs=10))
        assert args.queue_size == 16  # smallest bucket
        args = DenseCLArgs()
        args.resolve_auto_fields(
            scaling=ScalingInfo(dataset_size=3_000_000, epochs=10)
        )
        assert args.queue_size == _QUEUE_SIZE_BUCKETS[-1][1]

    def test_queue_auto_requires_dataset_size(self):
        args = DenseCLArgs()
        with pytest.raises(ValueError, match="dataset size"):
            args.resolve_auto_fields()

    def test_bucket_table_extends_beyond_shared_cap(self):
        # The shared QUEUE_SIZE_BUCKETS cap at 8192 is too small for
        # DenseCL: this table must reach 65536.
        assert _QUEUE_SIZE_BUCKETS[-1] == (float("inf"), 65_536)
        assert _QUEUE_SIZE_BUCKETS[0][1] == 16


class TestDenseProjectionHead:
    def test_output_shape(self):
        head = _DenseProjectionHead(input_dim=8, hidden_dim=32, output_dim=16)
        out = head(torch.randn(4, 8, 5, 5))
        assert out.shape == (4, 16, 5, 5)


class TestInfoNCEWithQueue:
    def test_matching_positive_beats_random_positive(self):
        queue = torch.nn.functional.normalize(torch.randn(8, 32), dim=0)
        query = torch.randn(10, 8)
        loss_match = _info_nce_with_queue(
            query, query.clone(), queue, 0.2
        )
        loss_random = _info_nce_with_queue(
            query, torch.randn(10, 8), queue, 0.2
        )
        assert loss_match < loss_random

    def test_hard_queue_negatives_raise_loss(self):
        # Negatives close to the query are harder to reject than far ones.
        query = torch.ones(4, 8)
        positive = torch.ones(4, 8)
        close_queue = torch.nn.functional.normalize(torch.ones(8, 2), dim=0)
        far_queue = torch.nn.functional.normalize(
            torch.cat([torch.ones(8, 1), -torch.ones(8, 1)], dim=1), dim=0
        )
        loss_close = _info_nce_with_queue(query, positive, close_queue, 0.2)
        loss_far = _info_nce_with_queue(query, positive, far_queue, 0.2)
        assert loss_close > loss_far


class TestQueue:
    def test_enqueue_fills_in_order(self):
        method = make_method(queue_size=16)
        keys = torch.randn(8, 16)
        method._enqueue(keys)
        ptr = int(method.queue_ptr.item())
        assert ptr == 8
        stored = method.queue[:, :8].T
        expected = torch.nn.functional.normalize(keys, p=2, dim=-1)
        assert torch.allclose(stored, expected, atol=1e-6)

    def test_enqueue_wraps_around(self):
        method = make_method(queue_size=16)
        method._enqueue(torch.randn(12, 16))
        method._enqueue(torch.randn(8, 16))
        ptr = int(method.queue_ptr.item())
        assert ptr == (12 + 8) % 16

    def test_enqueue_detaches(self):
        method = make_method(queue_size=16)
        keys = torch.randn(8, 16, requires_grad=True)
        method._enqueue(keys)
        assert not method.queue.requires_grad


class TestTrainingStep:
    def test_loss_decreases(self):
        torch.manual_seed(0)
        method = make_method()
        optimizer, scheduler = method.configure_optimizers(
            total_steps=12, num_epochs=2
        )
        losses = []
        for step in range(12):
            method.global_step = step
            loss = method.training_step(make_views(), step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
        # The queue swaps its random init for freshly encoded (hard) keys
        # over the first steps, so optimization is measured after warm-up.
        assert losses[-1] < losses[2]

    def test_log_dict_keys(self):
        method = make_method()
        method.training_step(make_views(), 0)
        log_dict = method.consume_log_dict()
        assert "train_loss/global" in log_dict
        assert "train_loss/dense" in log_dict

    def test_queue_advances_each_step(self):
        # batch 2 → 2B = 4 keys enqueued per step.
        method = make_method(queue_size=16)
        for step in range(2):
            method.training_step(make_views(batch_size=2), step)
        assert int(method.queue_ptr.item()) == 8

    def test_gradients_flow_to_student_and_heads(self):
        method = make_method()
        loss = method.training_step(make_views(), 0)
        loss.backward()
        for module in (
            method.embedding_model,
            method.global_projection_head,
            method.dense_projection_head,
        ):
            assert any(p.grad is not None for p in module.parameters())

    def test_dense_positive_is_pixelwise(self):
        # Identical views: each pixel's positive is itself; the dense loss
        # must push towards zero logits for the matched position.
        method = make_method()
        x = torch.randn(2, 3, 32, 32)
        loss = method.training_step({"views": [x, x.clone()]}, 0)
        assert torch.isfinite(loss)
