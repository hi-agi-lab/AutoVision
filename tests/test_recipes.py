# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Detection recipes: overlay semantics and one real one-step run."""

from __future__ import annotations

import pytest

from autovision import ensure_coco8, train_object_detection
from autovision.cli.recipes import (
    BUILTIN_RECIPES,
    RECIPE_NAMES,
    apply_recipe,
    load_recipe,
)
from autovision.errors import AutoVisionError


def test_builtin_names_are_stable():
    assert RECIPE_NAMES == ("coco-finetune", "finetune", "low-data")
    for name in RECIPE_NAMES:
        assert set(BUILTIN_RECIPES[name]) <= {
            "model_args", "transform_args", "trainer_args"
        }


def test_builtin_load_returns_a_copy():
    first = load_recipe("finetune")
    first["model_args"]["lr"] = 9.0
    assert load_recipe("finetune")["model_args"]["lr"] == 1.0e-4


def test_explicit_values_override_recipe_values():
    model_args, transform_args, trainer_args = apply_recipe(
        "finetune",
        model_args={"lr": 0.01, "backbone_lr_factor": 0.5},
        trainer_args={"log_every_n_steps": 7},
    )
    assert model_args["lr"] == 0.01  # explicit wins
    assert model_args["weight_decay"] == 1.0e-4  # recipe fills
    assert model_args["backbone_lr_factor"] == 0.5
    assert trainer_args["log_every_n_steps"] == 7
    assert trainer_args["val_every_num_steps"] == 500
    assert transform_args == {}


def test_yaml_recipe_and_unknown_keys(tmp_path):
    good = tmp_path / "recipe.yaml"
    good.write_text(
        "model_args:\n  lr: 0.0002\ntrainer_args:\n  log_every_n_steps: 3\n"
    )
    model_args, _, trainer_args = apply_recipe(good)
    assert model_args["lr"] == 2.0e-4
    assert trainer_args["log_every_n_steps"] == 3

    bad = tmp_path / "bad.yaml"
    bad.write_text("model_args:\n  lr: 0.1\nepochs: 3\n")
    with pytest.raises(AutoVisionError, match="unknown top-level"):
        load_recipe(bad)

    nested_bad = tmp_path / "nested.yaml"
    nested_bad.write_text("model_args: 3\n")
    with pytest.raises(AutoVisionError, match="must be a mapping"):
        load_recipe(nested_bad)


def test_unknown_recipe_and_empty_yaml(tmp_path):
    with pytest.raises(AutoVisionError, match="Unknown recipe"):
        load_recipe("does-not-exist")
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(AutoVisionError, match="empty"):
        load_recipe(empty)


def test_recipe_runs_through_the_training_entry(tmp_path):
    """The only check that proves the wiring: one real CPU step."""
    result = train_object_detection(
        out=tmp_path / "run",
        data=ensure_coco8(),
        model="rtdetr/resnet18",
        steps=1,
        batch_size=1,
        num_workers=0,
        devices=1,
        precision="32-true",
        recipe="finetune",
    )
    assert result.exported_model_path.exists()
