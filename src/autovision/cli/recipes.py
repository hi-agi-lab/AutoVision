# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Training recipes (roadmap 90d, phase 1).

A recipe is a small YAML/dict overlay for the three nested argument groups
(``model_args`` / ``transform_args`` / ``trainer_args``). It is deliberately
not a second configuration system: values land in the existing Pydantic
args, so unknown or out-of-range keys fail exactly where they always do.

Precedence: explicit call-site values win over recipe values; recipe values
win over the family defaults. Recipe files are plain mappings — no template
language, no environment expansion.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from autovision.errors import AutoVisionError

__all__ = [
    "BUILTIN_RECIPES",
    "RECIPE_NAMES",
    "apply_recipe",
    "load_recipe",
]

#: Generic detection defaults only. Family-specific knobs (GTR warm-start,
#: RF-DETR EMA, DETR denoising) stay out: a recipe must be safe for every
#: detection family, otherwise a user gets a confusing extra-field error.
BUILTIN_RECIPES: dict[str, dict[str, Any]] = {
    "finetune": {
        "model_args": {"lr": 1.0e-4, "weight_decay": 1.0e-4},
        "trainer_args": {"val_every_num_steps": 500},
    },
    "coco-finetune": {
        "model_args": {"lr": 1.0e-4, "weight_decay": 1.0e-4},
        "trainer_args": {"val_every_num_steps": 1000},
    },
    "low-data": {
        "model_args": {"lr": 5.0e-5, "weight_decay": 2.0e-4},
        "trainer_args": {"val_every_num_steps": 100},
    },
}

RECIPE_NAMES: tuple[str, ...] = tuple(sorted(BUILTIN_RECIPES))


def _coerce_mapping(payload: Any, source: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise AutoVisionError(f"Recipe '{source}' must be a mapping.")
    allowed = {"model_args", "transform_args", "trainer_args"}
    unknown = set(payload) - allowed
    if unknown:
        raise AutoVisionError(
            f"Recipe '{source}' has unknown top-level key(s) {sorted(unknown)}; "
            f"expected a subset of {sorted(allowed)}."
        )
    result: dict[str, Any] = {}
    for key in allowed:
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise AutoVisionError(
                f"Recipe '{source}' field '{key}' must be a mapping."
            )
        result[key] = dict(value)
    return result


def load_recipe(name_or_path: str | Path) -> dict[str, Any]:
    """Load a built-in recipe name or a YAML file.

    Built-ins are returned as a deep copy so callers can mutate the result
    without poisoning later calls.
    """
    key = str(name_or_path)
    if key in BUILTIN_RECIPES:
        return deepcopy(BUILTIN_RECIPES[key])
    path = Path(key)
    if path.is_file():
        try:
            payload = yaml.safe_load(path.read_text())
        except yaml.YAMLError as error:
            raise AutoVisionError(f"Recipe '{path}' is not valid YAML: {error}") from error
        if payload is None:
            raise AutoVisionError(f"Recipe '{path}' is empty.")
        return _coerce_mapping(payload, str(path))
    raise AutoVisionError(
        f"Unknown recipe '{key}'. Built-ins: {', '.join(RECIPE_NAMES)}; "
        "or pass a path to a YAML file."
    )


def _merge(base: Mapping[str, Any] | None, overlay: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(base or {})
    merged.update(dict(overlay or {}))
    return merged


def apply_recipe(
    recipe: str | Path,
    *,
    model_args: Mapping[str, Any] | None = None,
    transform_args: Mapping[str, Any] | None = None,
    trainer_args: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Merge one recipe under explicit call-site argument groups.

    Returns ``(model_args, transform_args, trainer_args)``. Explicit values
    always win; the recipe only fills missing keys. The inputs are not
    mutated.
    """
    loaded = load_recipe(recipe)
    return (
        _merge(loaded.get("model_args"), model_args),
        _merge(loaded.get("transform_args"), transform_args),
        _merge(loaded.get("trainer_args"), trainer_args),
    )
