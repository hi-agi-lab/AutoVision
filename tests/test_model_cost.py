# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Model cost measurement: parameters, MACs/FLOPs, benchmark integration.

Convention under test: PyTorch's ``FlopCounterMode`` counts a multiply-add
as two FLOPs, so ``macs == flops / 2`` — the number published detection
tables quote as GFLOPs.
"""

from __future__ import annotations

import pytest
import torch

from autovision.export import ModelCost, count_parameters, measure_model_cost
from autovision.tasks import RTDETRObjectDetection


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return RTDETRObjectDetection(
        num_classes=2, backbone="torchvision/resnet18",
        image_size=(64, 64), hidden_dim=32, num_queries=10,
        dec_layers=1, num_heads=4, num_top_queries=10,
    )


class TestCountingConvention:
    def test_linear_macs_and_flops(self):
        model = torch.nn.Linear(4, 8)
        cost = measure_model_cost(model, example_input=torch.zeros(1, 4))
        assert cost.parameters == 4 * 8 + 8  # weight + bias
        assert cost.trainable_parameters == cost.parameters
        assert cost.macs == pytest.approx(1 * 4 * 8)
        assert cost.flops == pytest.approx(2 * 1 * 4 * 8)
        assert cost.input_shape == (1, 4)

    def test_conv2d_macs(self):
        model = torch.nn.Conv2d(3, 16, 3, padding=1)
        cost = measure_model_cost(model, example_input=torch.zeros(1, 3, 32, 32))
        expected = 16 * 3 * 3 * 3 * 32 * 32  # out_c * in_c * k^2 * H * W
        assert cost.macs == pytest.approx(expected)
        assert cost.gmacs == pytest.approx(expected / 1e9)

    def test_count_parameters(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU())
        total, trainable = count_parameters(model)
        assert total == trainable == 4 * 8 + 8


class TestTaskNetworkCost:
    def test_cost_from_input_spec(self, tiny_model):
        cost = measure_model_cost(tiny_model, batch_size=2)
        assert cost.input_shape == (2, 3, 64, 64)
        assert cost.parameters > 0
        assert cost.macs > 0
        assert isinstance(cost, ModelCost)

    def test_deterministic_and_serializable(self, tiny_model):
        first = measure_model_cost(tiny_model, batch_size=1)
        second = measure_model_cost(tiny_model, batch_size=1)
        assert first.macs == second.macs
        payload = first.to_dict()
        assert payload["parameters"] == first.parameters
        assert payload["input_shape"] == [1, 3, 64, 64]
        assert payload["gmacs"] == pytest.approx(first.macs / 1e9)

    def test_eval_mode_restored(self, tiny_model):
        tiny_model.train()
        try:
            measure_model_cost(tiny_model, batch_size=1)
            assert tiny_model.training
        finally:
            tiny_model.eval()


class TestBenchmarkIntegration:
    def test_benchmark_reports_cost(self, tiny_model, tmp_path):
        from autovision.export import benchmark
        from autovision.export.benchmark import TorchBackendArgs

        # Snapshot convention: counts taken before the backend deploys.
        expected_parameters, _ = count_parameters(tiny_model)
        results = benchmark(
            tiny_model,
            tmp_path / "bench",
            batch_size=1,
            runs=2,
            warmup=1,
            backends={"torch": TorchBackendArgs()},
        )
        extra = results["torch"].extra
        assert extra["parameters"] == expected_parameters
        assert extra["macs"] > 0 and extra["flops"] == pytest.approx(2 * extra["macs"])
        markdown = results["torch"].to_markdown()
        assert "parameters (M)" in markdown and "MACs (G)" in markdown
