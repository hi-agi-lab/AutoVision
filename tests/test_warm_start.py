# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the AVWarmStart v1 format (design doc 28 §4.3/§6.2)."""

import pytest
import torch
from torch import nn

from autovision.registry.assets import LicenseRef
from autovision.registry.warm_start import (
    AV_WARM_START_FORMAT,
    WarmStartError,
    load_warm_start,
    save_warm_start,
)

LICENSE = LicenseRef(id="Apache-2.0", origin="https://github.com/x@abc")


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)


def test_round_trip(tmp_path):
    model = Tiny()
    out = save_warm_start(
        tmp_path / "ws" / "warm.pt",
        model=model.state_dict(),
        spec={"family": "detr", "classes": 80},
        license=LICENSE,
    )
    payload = load_warm_start(out)
    assert payload.license == LICENSE
    assert payload.spec["family"] == "detr"
    for key, value in model.state_dict().items():
        assert torch.equal(payload.model[key], value)


def test_missing_license_refused(tmp_path):
    with pytest.raises(WarmStartError, match="LicenseRef"):
        save_warm_start(
            tmp_path / "w.pt", model={"a": torch.zeros(1)}, spec={}, license=None
        )


def test_empty_state_dict_refused(tmp_path):
    with pytest.raises(WarmStartError, match="non-empty"):
        save_warm_start(tmp_path / "w.pt", model={}, spec={}, license=LICENSE)


def test_bad_format_tag_rejected(tmp_path):
    path = tmp_path / "w.pt"
    torch.save({"format": "something_else", "model": {}}, path)
    with pytest.raises(WarmStartError, match="AVWarmStart v1"):
        load_warm_start(path)


def test_missing_license_field_rejected(tmp_path):
    path = tmp_path / "w.pt"
    torch.save(
        {
            "format": AV_WARM_START_FORMAT,
            "model": {"a": torch.zeros(1)},
            "spec": {},
        },
        path,
    )
    with pytest.raises(WarmStartError, match="license"):
        load_warm_start(path)


def test_non_tensor_state_dict_rejected(tmp_path):
    path = tmp_path / "w.pt"
    torch.save(
        {
            "format": AV_WARM_START_FORMAT,
            "model": {"a": [1, 2, 3]},
            "spec": {},
            "license": {"id": "x", "origin": "y"},
        },
        path,
    )
    with pytest.raises(WarmStartError, match="state dict"):
        load_warm_start(path)
