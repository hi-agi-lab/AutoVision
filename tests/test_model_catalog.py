# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Task-level model catalog: curated rows must not drift from the registries.

The catalog lives in the registry layer (no task imports), so this test is
the cross-layer guard: every RF-DETR variant, GTR detection asset and DETR
spectrum tier is represented with the right status/license/metrics.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from autovision.cli._app import app
from autovision.registry.catalog import (
    MODEL_CATALOG,
    describe_model,
    format_model_table,
    get_model_entry,
    list_task_models,
)
from autovision.registry.gtr_assets import GTR_ASSETS
from autovision.tasks.detection.detr.variants import SPECTRUM
from autovision.tasks.detection.rfdetr.config import RFDETR_MODEL_REGISTRY


def _keys(**filters) -> set[str]:
    return {entry.key for entry in list_task_models(**filters)}


def test_rfdetr_rows_cover_every_variant():
    expected = {
        f"rfdetr/{name}" for name in RFDETR_MODEL_REGISTRY if name != "rf-detr-base"
    }
    assert _keys(pretrained_only=True) & expected == expected
    for key in expected:
        entry = get_model_entry(key)
        assert entry.source == "upstream"
        assert entry.pretrained == "full"
        assert entry.license == "Apache-2.0"
        assert entry.input_size == (
            RFDETR_MODEL_REGISTRY[key.split("/", 1)[1]]["resolution"],
        ) * 2
        assert "onnx" in entry.exportable


def test_gtr_rows_match_registered_assets_and_metrics():
    rows = [entry for entry in MODEL_CATALOG if entry.family == "gtr"]
    assert {entry.key for entry in rows} == {
        "detr/gtrvit-s",
        "detr/gtrvit-m",
        "detr/gtrvit-l",
        "detr/gtrvit-x",
    }
    for entry in rows:
        asset = GTR_ASSETS[entry.weights_asset]
        assert asset.coco_ap == entry.coco_ap
        assert asset.license == entry.license
        assert entry.exportable == ()
        assert entry.latency_ms is not None


def test_detr_spectrum_status_matches_the_machine_readable_table():
    for tier, spec in SPECTRUM.items():
        entry = get_model_entry(f"detr/{tier}")
        assert entry.status == spec.status
        if spec.variant_key:
            assert entry.source == "autovision"
            assert entry.pretrained == "none"


def test_available_architecture_without_weights_is_not_pretrained():
    entry = get_model_entry("rtdetr/resnet18")
    assert entry.status == "available"
    assert entry.pretrained == "none"
    assert "rtdetr/resnet18" not in _keys(pretrained_only=True)


def test_planned_rows_are_hidden_by_default():
    assert "detr/m" not in _keys()
    assert "detr/m" in _keys(include_planned=True)


def test_filters_and_unknown_key():
    assert _keys(task="detection", pretrained_only=True) >= {
        "rfdetr/rf-detr-nano",
        "detr/gtrvit-s",
    }
    assert _keys(task="pose_estimation") == set()
    with pytest.raises(KeyError, match="Unknown task model"):
        get_model_entry("not/a-model")


def test_table_and_description_are_renderable():
    table = format_model_table(list_task_models(pretrained_only=True))
    assert "rfdetr/rf-detr-small" in table
    assert "Apache-2.0" in table
    assert "detr/gtrvit-s" in table
    description = describe_model("detr/gtrvit-s")
    assert "license:    MIT" in description
    assert "exportable: none" in description


def test_cli_default_stays_encoder_listing():
    result = CliRunner().invoke(app, ["list-models"])
    assert result.exit_code == 0, result.output
    assert "torchvision/resnet18" in result.output


def test_cli_catalog_views_and_model_info():
    runner = CliRunner()
    result = runner.invoke(
        app, ["list-models", "--task", "detection", "--pretrained", "--format", "table"]
    )
    assert result.exit_code == 0, result.output
    assert "rfdetr/rf-detr-small" in result.output
    assert "detr/gtrvit-s" in result.output

    result = runner.invoke(app, ["model-info", "rfdetr/rf-detr-small"])
    assert result.exit_code == 0, result.output
    assert "Apache-2.0" in result.output
    assert "weights:    full" in result.output

    result = runner.invoke(app, ["model-info", "not/a-model"])
    assert result.exit_code != 0
