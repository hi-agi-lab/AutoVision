# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
import math

import pytest
import torch
from torch import nn

from autovision.optim import (
    AdamWArgs,
    CosineWarmupScheduler,
    LARSArgs,
    OptimizationTargets,
    SGDArgs,
    get_optimizer,
    get_optimizer_args_cls,
    get_weight_decay_parameters,
)
from autovision.optim.lars import LARS


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)  # weight decays, bias does not (ndim<=1)
        self.bn = nn.BatchNorm2d(8)  # all params no-wd
        self.norm = nn.LayerNorm(8)  # all params no-wd
        self.embedding = nn.Embedding(10, 8)  # no-wd
        self.linear = nn.Linear(8, 4)
        self.register_parameter(
            "cls_token", nn.Parameter(torch.zeros(1, 8))
        )  # substring rule
        self.register_parameter(
            "logit_scale", nn.Parameter(torch.tensor(1.0))
        )  # whole-word rule
        self.register_parameter(
            "pos_embed", nn.Parameter(torch.zeros(1, 4, 8))
        )  # substring rule


class TestWeightDecayParameters:
    def test_split(self):
        model = DummyModel()
        wd, no_wd = get_weight_decay_parameters([model])
        no_wd_ids = {id(p) for p in no_wd}
        wd_ids = {id(p) for p in wd}
        # All parameters accounted for exactly once.
        all_params = {id(p) for p in model.parameters()}
        assert no_wd_ids | wd_ids == all_params
        assert not no_wd_ids & wd_ids
        # Decay: conv weight + linear weight.
        assert id(model.conv.weight) in wd_ids
        assert id(model.linear.weight) in wd_ids
        # No decay: biases, norm layers, embeddings, tokens, scales.
        for param in (
            model.conv.bias,
            model.linear.bias,
            *model.bn.parameters(),
            *model.norm.parameters(),
            model.embedding.weight,
            model.cls_token,
            model.logit_scale,
            model.pos_embed,
        ):
            assert id(param) in no_wd_ids

    def test_class_name_norm_rule(self):
        class CustomNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.gamma = nn.Parameter(torch.ones(4))

        module = CustomNorm()
        _, no_wd = get_weight_decay_parameters([module])
        assert len(no_wd) == 1

    def test_shared_parameters_deduplicated(self):
        linear = nn.Linear(4, 4)
        wrapper = nn.Sequential(linear, linear)
        wd, no_wd = get_weight_decay_parameters([wrapper])
        assert len(wd) + len(no_wd) == 2  # weight + bias, each once

    def test_extra_no_decay_keys(self):
        model = nn.Sequential(nn.Linear(4, 4))
        extra = ["0.weight"]
        wd, no_wd = get_weight_decay_parameters([model], extra_no_decay_keys=extra)
        assert len(no_wd) == 1  # the weight is excluded via extra key
        assert len(wd) == 1

    def test_whole_word_matching_does_not_catch_substrings(self):
        class Holder(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(8, 4))  # contains "gamma"? no
                self.tau_scale = nn.Parameter(torch.randn(8, 4))

        wd, no_wd = get_weight_decay_parameters([Holder()])
        # "tau_scale" contains "tau" as prefix but not as whole word component;
        # "tau" must NOT match, so both weights decay.
        assert len(wd) == 2
        assert len(no_wd) == 0


class TestGetOptimizer:
    @pytest.mark.parametrize(
        "args_cls", [AdamWArgs, SGDArgs, LARSArgs], ids=lambda c: c.__name__
    )
    def test_lr_scale_applied(self, args_cls):
        model = DummyModel()
        optimizer = get_optimizer(
            optim_args=args_cls(),
            trainable_modules=OptimizationTargets(targets=[model]),
            lr_scale=4.0,
        )
        base_lr = args_cls().lr
        for group in optimizer.param_groups:
            assert group["lr"] == pytest.approx(base_lr * 4.0)

    def test_no_weight_decay_group_created(self):
        model = DummyModel()
        optimizer = get_optimizer(
            optim_args=AdamWArgs(),
            trainable_modules=OptimizationTargets(targets=[model]),
            lr_scale=1.0,
        )
        names = [group["name"] for group in optimizer.param_groups]
        assert names == ["params", "params_no_weight_decay"]
        no_wd_group = optimizer.param_groups[1]
        assert no_wd_group["weight_decay"] == 0.0

    def test_no_empty_no_wd_group(self):
        model = nn.Sequential(nn.Linear(4, 4))  # has biases -> no wd group exists
        optimizer = get_optimizer(
            optim_args=AdamWArgs(),
            trainable_modules=OptimizationTargets(targets=[model]),
            lr_scale=1.0,
        )
        assert len(optimizer.param_groups) == 2

    def test_decay_free_override(self):
        model = DummyModel()
        optimizer = get_optimizer(
            optim_args=AdamWArgs(),
            trainable_modules=OptimizationTargets(
                targets=[model], decay_free=[model.linear]
            ),
            lr_scale=1.0,
        )
        no_wd_group = optimizer.param_groups[1]
        no_wd_ids = {id(p) for p in no_wd_group["params"]}
        assert id(model.linear.weight) in no_wd_ids
        # And not present in the decay group.
        wd_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
        assert id(model.linear.weight) not in wd_ids


class TestOptimizerArgs:
    def test_registry_lookup(self):
        assert get_optimizer_args_cls("adamw") is AdamWArgs
        assert get_optimizer_args_cls("sgd") is SGDArgs
        assert get_optimizer_args_cls("lars") is LARSArgs
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            get_optimizer_args_cls("sgdm")

    def test_types(self):
        assert AdamWArgs().kind() == "adamw"
        assert SGDArgs().kind() == "sgd"
        assert LARSArgs().kind() == "lars"

    def test_weight_decay_auto_transformer(self):
        class TransformerModel(nn.Module):
            def arch_traits(self):
                return {"model_type": "transformer", "norm_type": "layernorm"}

        args = AdamWArgs(weight_decay="auto")
        args.resolve_auto_fields(wrapped_model=TransformerModel())
        assert args.weight_decay == 0.04

    def test_weight_decay_auto_conv_batchnorm(self):
        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 3)
                self.bn = nn.BatchNorm2d(4)

            def arch_traits(self):
                return {"model_type": "convolutional", "norm_type": "batchnorm"}

        args = AdamWArgs(weight_decay="auto")
        args.resolve_auto_fields(wrapped_model=ConvModel())
        assert args.weight_decay == 1e-6

    def test_weight_decay_auto_unknown_architecture_raises(self):
        class WeirdModel(nn.Module):
            def arch_traits(self):
                return {"model_type": "convolutional", "norm_type": "rmsnorm"}

        args = AdamWArgs(weight_decay="auto")
        with pytest.raises(ValueError, match="Cannot resolve weight_decay"):
            args.resolve_auto_fields(wrapped_model=WeirdModel())

    def test_weight_decay_auto_fallback_heuristic(self):
        conv = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
        args = AdamWArgs(weight_decay="auto")
        args.resolve_auto_fields(wrapped_model=conv)
        assert args.weight_decay == 1e-6

    def test_optimizer_kwargs_forwarded(self):
        model = DummyModel()
        args = AdamWArgs(lr=0.01, betas=(0.8, 0.99), weight_decay=0.05)
        optimizer = args.get_optimizer(
            params=[{"params": model.parameters()}], lr_scale=2.0
        )
        assert isinstance(optimizer, torch.optim.AdamW)
        group = optimizer.param_groups[0]
        assert group["lr"] == pytest.approx(0.02)
        assert group["betas"] == (0.8, 0.99)
        assert group["weight_decay"] == 0.05


class TestLARS:
    def _make(self):
        torch.manual_seed(0)
        model = nn.Linear(4, 4)
        return model

    def test_step_reduces_loss(self):
        model = self._make()
        optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
        x = torch.randn(16, 4)
        y = torch.randn(16, 4)
        first_loss = None
        for _ in range(10):
            optimizer.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            if first_loss is None:
                first_loss = loss.item()
            loss.backward()
            optimizer.step()
        assert loss.item() < first_loss

    def test_local_lr_formula(self):
        """With zero weight decay, update = trust_coef * |w| / |g| * g."""
        model = self._make()
        lr, trust, eps = 0.5, 0.01, 1e-8
        optimizer = LARS(model.parameters(), lr=lr, trust_coefficient=trust, eps=eps)
        model.weight.grad = torch.ones_like(model.weight)
        model.bias.grad = torch.zeros_like(model.bias)
        w_before = model.weight.detach().clone()
        grad_norm = model.weight.grad.norm()
        optimizer.step()
        p_norm = w_before.norm()
        expected = trust * p_norm / (grad_norm + eps)
        actual_delta = (model.weight - w_before).abs().max().item()
        assert actual_delta == pytest.approx(lr * expected, rel=1e-5)

    def test_nesterov_requires_momentum(self):
        with pytest.raises(ValueError, match="Nesterov"):
            LARS([torch.nn.Parameter(torch.zeros(1))], lr=0.1, nesterov=True)


class TestCosineWarmupScheduler:
    def _make(self, **kwargs):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = CosineWarmupScheduler(optimizer, **kwargs)
        return optimizer, scheduler

    def test_warmup_from_zero(self):
        optimizer, scheduler = self._make(warmup_epochs=10, max_epochs=100)
        assert optimizer.param_groups[0]["lr"] == 0.0  # step 0
        for _ in range(9):
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.09)  # 9/10

    def test_peak_at_end_of_warmup(self):
        optimizer, scheduler = self._make(warmup_epochs=10, max_epochs=100)
        for _ in range(10):
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    def test_cosine_decay_reaches_end_value(self):
        optimizer, scheduler = self._make(
            warmup_epochs=10, max_epochs=110, end_value=0.0
        )
        for _ in range(110):
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)

    def test_midpoint_halfway(self):
        optimizer, scheduler = self._make(warmup_epochs=0, max_epochs=100, end_value=0.0)
        for _ in range(50):
            scheduler.step()
        expected = 0.1 * 0.5 * (1 + math.cos(math.pi * 0.5))
        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-6)

    def test_nonzero_end_value(self):
        optimizer, scheduler = self._make(
            warmup_epochs=0, max_epochs=100, end_value=0.1
        )
        for _ in range(100):
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 0.1)

    def test_validation(self):
        model = nn.Linear(2, 2)
        with pytest.raises(ValueError, match="max_epochs"):
            CosineWarmupScheduler(torch.optim.SGD(model.parameters(), lr=0.1), 5, 0)
        with pytest.raises(ValueError, match="warmup_epochs"):
            CosineWarmupScheduler(
                torch.optim.SGD(model.parameters(), lr=0.1), -1, 10
            )
        with pytest.raises(ValueError, match="end_value"):
            CosineWarmupScheduler(
                torch.optim.SGD(model.parameters(), lr=0.1), 0, 10, end_value=1.5
            )
