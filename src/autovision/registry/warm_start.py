# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""AVWarmStart v1: the framework's own warm-start checkpoint format.

Design doc 28, sections 4.3 (format) and 6.2 (self-built COCO warm-start
pipeline). A warm-start checkpoint is a plain, self-describing payload:

    {
        "format": "av_warm_start_v1",
        "model": state_dict,      # target-network parameter names
        "spec": {                 # free-form provenance metadata
            "family": "detr", "backbone": "...", "classes": 80,
            "created": "2026-10-14", ...
        },
        "license": {"id": ..., "origin": ...},   # REQUIRED (rule R7)
    }

Unlike the legacy reference payloads there is no ``ema_model.model.``
stripping, no embedded class path, and no missing license field: assets
produced by this framework always carry first-class license metadata and
target-network keys directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from autovision.registry.assets import LicenseRef
from autovision.registry.downloadable import DownloadableModelError

__all__ = [
    "AV_WARM_START_FORMAT",
    "WarmStartError",
    "WarmStartPayload",
    "load_warm_start",
    "save_warm_start",
]

logger = logging.getLogger("autovision")

AV_WARM_START_FORMAT = "av_warm_start_v1"


class WarmStartError(DownloadableModelError):
    """Raised when a warm-start checkpoint is malformed or unlicensed."""


@dataclass(frozen=True)
class WarmStartPayload:
    """Validated in-memory view of an AVWarmStart v1 file."""

    model: dict[str, torch.Tensor]
    spec: dict[str, Any]
    license: LicenseRef


def save_warm_start(
    path: str | Path,
    model: dict[str, torch.Tensor],
    spec: dict[str, Any],
    license: LicenseRef,
) -> Path:
    """Write an AVWarmStart v1 checkpoint.

    ``license`` is mandatory — an unlicensed warm-start asset must not be
    produced in the first place (design doc 28, R7).
    """
    if not isinstance(license, LicenseRef):
        raise WarmStartError(
            "save_warm_start requires a LicenseRef (id + origin); refusing "
            "to write an unlicensed warm-start asset."
        )
    if not model:
        raise WarmStartError("save_warm_start requires a non-empty state dict.")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": AV_WARM_START_FORMAT,
        "model": model,
        "spec": dict(spec),
        "license": {"id": license.id, "origin": license.origin},
    }
    torch.save(payload, out_path)
    logger.info(
        "Wrote AVWarmStart v1 checkpoint '%s' (%d tensors, license %s).",
        out_path, len(model), license.id,
    )
    return out_path


def load_warm_start(path: str | Path) -> WarmStartPayload:
    """Read and validate an AVWarmStart v1 checkpoint."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or raw.get("format") != AV_WARM_START_FORMAT:
        raise WarmStartError(
            f"'{path}' is not an AVWarmStart v1 checkpoint (format tag "
            f"missing or unknown)."
        )
    model = raw.get("model")
    if not isinstance(model, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in model.items()
    ):
        raise WarmStartError(f"'{path}' has no valid tensor state dict under 'model'.")
    license_field = raw.get("license")
    if (
        not isinstance(license_field, dict)
        or not license_field.get("id")
        or not license_field.get("origin")
    ):
        raise WarmStartError(f"'{path}' is missing its license metadata (R7).")
    spec = raw.get("spec")
    if not isinstance(spec, dict):
        raise WarmStartError(f"'{path}' has no 'spec' metadata dict.")
    return WarmStartPayload(
        model=model,
        spec=spec,
        license=LicenseRef(id=str(license_field["id"]), origin=str(license_field["origin"])),
    )
