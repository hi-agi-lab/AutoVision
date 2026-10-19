# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Checkpoint RNG state must round-trip through the weights_only loader."""

import random

import numpy as np
import torch

from autovision.engine.engine import _capture_rng_state, _restore_rng_state


def test_rng_state_roundtrip_through_weights_only(tmp_path):
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    state = _capture_rng_state()
    expected = (torch.rand(4), np.random.rand(4), random.random())

    path = tmp_path / "rng.pt"
    torch.save({"rng": state}, path)
    loaded = torch.load(path, weights_only=True)["rng"]

    _restore_rng_state(loaded)
    actual = (torch.rand(4), np.random.rand(4), random.random())
    assert torch.allclose(expected[0], actual[0])
    np.testing.assert_allclose(expected[1], actual[1])
    assert expected[2] == actual[2]
