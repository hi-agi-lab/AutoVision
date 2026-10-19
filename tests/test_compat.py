# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for third-party compatibility shims (src/autovision/_compat.py)."""

import copy

import pytest

torch = pytest.importorskip("torch")
tv_feature_extraction = pytest.importorskip(
    "torchvision.models.feature_extraction",
    reason="torchvision not installed",
)

from autovision._compat import install_graphmodule_deepcopy_fix  # noqa: E402


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.block2 = torch.nn.ReLU()

    def forward(self, x):
        return self.block2(self.block1(x))


def _make_graph_module():
    return tv_feature_extraction.create_feature_extractor(
        _Tiny(), {"block1": "feat"}
    )


class TestGraphModuleDeepcopyFix:
    def test_patch_is_idempotent(self):
        # autovision/__init__ already installed the shim on package import.
        # torchvision < 0.21 exposes only DualGraphModule here; older versions
        # also had a plain GraphModule — cover both when present.
        flag = "_autovision_graphmodule_deepcopy_fix"
        assert getattr(tv_feature_extraction.DualGraphModule, flag, False)
        if hasattr(tv_feature_extraction, "GraphModule"):
            assert getattr(tv_feature_extraction.GraphModule, flag, False)
        install_graphmodule_deepcopy_fix()  # second call must not corrupt
        assert callable(tv_feature_extraction.DualGraphModule.__deepcopy__)

    def test_deepcopy_keeps_train_and_eval_graphs(self):
        model = _make_graph_module()
        assert hasattr(model, "train_graph") and hasattr(model, "eval_graph")
        replica = copy.deepcopy(model)
        # Regression: torch.fx.GraphModule.__deepcopy__ dropped the
        # torchvision subclass attributes, breaking later .train()/.eval().
        assert hasattr(replica, "train_graph") and hasattr(replica, "eval_graph")

    def test_deepcopied_model_mode_switching(self):
        model = _make_graph_module()
        replica = copy.deepcopy(model)
        replica.eval()
        assert replica.graph is replica.eval_graph
        replica.train()
        assert replica.graph is replica.train_graph

    def test_deepcopied_model_matches_outputs(self):
        model = _make_graph_module().eval()
        replica = copy.deepcopy(model)
        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            # create_feature_extractor models return node-name dicts.
            original, copied = model(x), replica(x)
        assert isinstance(original, dict)
        for key, value in original.items():
            assert torch.allclose(value, copied[key])

    def test_deepcopy_parameters_are_independent(self):
        model = _make_graph_module()
        replica = copy.deepcopy(model)
        original_weight = next(model.parameters())
        replica_weight = next(replica.parameters())
        assert replica_weight.data_ptr() != original_weight.data_ptr()
        with torch.no_grad():
            replica_weight.fill_(123.0)
        assert not torch.all(original_weight == 123.0)
