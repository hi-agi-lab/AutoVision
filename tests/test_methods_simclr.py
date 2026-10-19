# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the SimCLR pretraining method."""

import pytest
import torch
from tests.helpers_models import TinyWrapper

from autovision.encoders import FeatureExtractor
from autovision.optim import AdamWArgs
from autovision.pretext import get_method_cls, list_methods
from autovision.pretext.simclr import (
    NTXentLoss,
    SimCLR,
    SimCLRArgs,
    SimCLRSGDArgs,
    SimCLRTransform,
    SimCLRTransformArgs,
)
from autovision.scaling import ScalingInfo

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def make_simclr(embed_dim: int = 8, **args_overrides) -> SimCLR:
    method_args = SimCLRArgs(**args_overrides)
    method_args.resolve_auto_fields(scaling=ScalingInfo(dataset_size=64, epochs=10))
    optimizer_args = SimCLR.optimizer_args_cls("auto")()
    optimizer_args.resolve_auto_fields(wrapped_model=None)
    embedding_model = FeatureExtractor(
        wrapped_model=TinyWrapper(dim=embed_dim), embed_dim=None
    )
    return SimCLR(
        method_args=method_args,
        optimizer_args=optimizer_args,
        embedding_model=embedding_model,
        global_batch_size=8,
        num_input_channels=3,
    )


def test_registered():
    assert "simclr" in list_methods()
    assert get_method_cls("simclr") is SimCLR


class TestSimCLRArgs:
    def test_defaults(self):
        args = SimCLRArgs()
        assert args.hidden_dim == 2048
        assert args.output_dim == 128
        assert args.num_layers == 2
        assert args.batch_norm is True
        assert args.temperature == 0.1

    def test_sgd_recipe(self):
        assert SimCLRSGDArgs().lr == 0.3

    def test_transform_recipe(self):
        args = SimCLRTransformArgs()
        assert args.image_size == (224, 224)
        assert args.color_jitter.strength == 1.0
        assert args.gaussian_blur.prob == 0.5
        assert args.solarize is None


class TestNTXentLoss:
    def test_identical_views_give_low_loss_than_random(self):
        loss_fn = NTXentLoss(temperature=0.1)
        torch.manual_seed(0)
        z = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        identical = loss_fn(z, z)
        shuffled = z[torch.randperm(8)]
        random = loss_fn(z, shuffled)
        assert identical < random

    def test_loss_is_nonnegative_finite(self):
        loss_fn = NTXentLoss(temperature=0.5)
        z0, z1 = torch.randn(4, 8), torch.randn(4, 8)
        assert torch.isfinite(loss_fn(z0, z1))
        assert loss_fn(z0, z1) >= 0

    def test_gradient_flows(self):
        loss_fn = NTXentLoss()
        z0 = torch.randn(4, 8, requires_grad=True)
        z1 = torch.randn(4, 8, requires_grad=True)
        loss_fn(z0, z1).backward()
        assert z0.grad is not None


class TestSimCLRMethod:
    def test_training_step_reduces_loss(self):
        torch.manual_seed(0)
        method = make_simclr()
        optimizer, scheduler = method.configure_optimizers(
            total_steps=8, num_epochs=2
        )
        losses = []
        for step in range(8):
            method.global_step = step
            views = [torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)]
            loss = method.training_step({"views": views}, step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        assert losses[-1] < losses[0]

    def test_trainable_modules_exclude_head_bn_from_no_wd_auto(self):
        method = make_simclr()
        trainable = method.trainable_modules()
        modules = list(trainable.targets)
        assert method.embedding_model in modules
        assert method.projection_head in modules

    def test_projection_head_output_dim(self):
        method = make_simclr(embed_dim=8, output_dim=16)
        out = method.projection_head(torch.randn(2, 8))
        assert out.shape == (2, 16)

    def test_transform_returns_two_views(self):
        transform = SimCLRTransform(SimCLRTransformArgs(image_size=(32, 32)))
        views = transform({"image": torch.randint(0, 255, (3, 40, 40), dtype=torch.uint8)})
        assert len(views) == 2
        assert views[0]["image"].shape == (3, 32, 32)

    def test_adamw_fallback(self):
        cls = SimCLR.optimizer_args_cls("adamw")
        assert cls is not SimCLRSGDArgs  # generic AdamWArgs family
        assert issubclass(cls, AdamWArgs)
