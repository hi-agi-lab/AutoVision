# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Release gate for the public detection spectrum (design doc 29 §3.4).

A ``released`` tier is a public promise, not a roadmap label: it must have
downloadable weights, a valid model card (license + sha256), a full-COCO
AP reading, and at least one reproducible latency record. This module
keeps that promise data-driven: Phase 1 adds a
:class:`ReleasedSpectrumWeight` when n/s weights are published, and the
gate fails if the tier is marked ``released`` without one.

``spec-frozen`` is the pre-release state used by n/s today: the components
and recipe are committed, but no pretrained weights or COCO reading are
published. ``planned`` tiers (m/l/x) have no registry key at all.

The machine-readable tier table is passed in by the caller (the test and
the release tooling use ``variants.SPECTRUM``); this registry module does
not import the task layer, preserving the one-way layering contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autovision.registry.model_card import ModelCard, ModelCardError

__all__ = [
    "RELEASED_SPECTRUM_WEIGHTS",
    "ReleasedSpectrumWeight",
    "validate_spectrum_release",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ReleasedSpectrumWeight:
    """One published spectrum tier: asset identity plus its mandatory card."""

    tier: str
    asset: str  # registry asset name / HF repo id
    weights_file: str  # file name covered by card.weights_sha256
    card: ModelCard


#: Empty until Phase 1 publishes n/s. Adding an entry alone is not enough:
#: the corresponding ``SPECTRUM`` tier must also be status="released".
RELEASED_SPECTRUM_WEIGHTS: dict[str, ReleasedSpectrumWeight] = {}


def _script_exists(record_script: str) -> bool:
    path = Path(record_script)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.is_file()


def validate_spectrum_release(
    tiers: Mapping[str, Any],
    assets: Mapping[str, ReleasedSpectrumWeight] | None = None,
) -> list[str]:
    """Return every release-gate violation (empty list means publishable).

    The gate is deliberately strict about the fields that customers use to
    choose a model: what license, which bytes, what AP, and how fast.
    """

    assets = RELEASED_SPECTRUM_WEIGHTS if assets is None else assets
    problems: list[str] = []

    for asset_tier in assets:
        if asset_tier not in tiers:
            problems.append(
                f"published asset keyed '{asset_tier}' has no spectrum tier"
            )

    for tier, entry in tiers.items():
        released = assets.get(tier)
        if entry.status == "released":
            if not entry.variant_key:
                problems.append(f"released tier '{tier}' has no model key")
            if released is None:
                problems.append(
                    f"released tier '{tier}' has no published weight asset"
                )
                continue
            if released.tier != tier:
                problems.append(
                    f"published asset keyed '{tier}' declares tier '{released.tier}'"
                )
            if not released.asset.strip():
                problems.append(f"released tier '{tier}' has an empty asset name")
            if released.card.model != f"detr/{tier}":
                problems.append(
                    f"released tier '{tier}': model card names "
                    f"'{released.card.model}', expected 'detr/{tier}'"
                )
            try:
                released.card.validate()
            except ModelCardError as error:
                problems.append(f"released tier '{tier}': {error}")
            if released.card.seed is None:
                problems.append(
                    f"released tier '{tier}': training seed is required for "
                    "our own published weights"
                )
            if released.weights_file not in released.card.weights_sha256:
                problems.append(
                    f"released tier '{tier}': weights file "
                    f"'{released.weights_file}' is not pinned in the model card"
                )
            if released.card.metrics.get("coco_map", 0.0) <= 0.0:
                problems.append(
                    f"released tier '{tier}': no full-COCO AP reading in the model card"
                )
            if not released.card.latency:
                problems.append(
                    f"released tier '{tier}': no latency record in the model card"
                )
            for record in released.card.latency:
                if not _script_exists(record.script):
                    problems.append(
                        f"released tier '{tier}': latency script "
                        f"'{record.script}' does not exist"
                    )
        else:
            if released is not None:
                problems.append(
                    f"tier '{tier}' is status='{entry.status}' but has a "
                    "published weight asset"
                )
            if entry.status == "spec-frozen" and not entry.variant_key:
                problems.append(f"spec-frozen tier '{tier}' has no model key")

    return problems
