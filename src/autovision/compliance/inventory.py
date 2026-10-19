# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Framework / model / dataset inventory for the compliance report."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from autovision._version import __version__
from autovision.errors import AutoVisionError
from autovision.registry.model_card import load_model_card

__all__ = ["dataset_inventory", "framework_inventory", "model_inventory"]


def framework_inventory() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    commit: str | None = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - source snapshots may not be a git checkout
        commit = None
    return {
        "name": "autovision-framework",
        "version": __version__,
        "commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_inventory(
    model: str | Path | None,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    if model is None:
        return None, []
    path = Path(model)
    if not path.is_file():
        raise AutoVisionError(f"Model file not found: '{path}'.")
    entry: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }
    issues: list[dict[str, str]] = []
    card_path = Path(str(path) + ".modelcard.json")
    if not card_path.is_file():
        issues.append(
            {
                "severity": "warning",
                "kind": "missing_model_card",
                "detail": f"No model card next to '{path.name}'.",
            }
        )
    else:
        try:
            card = load_model_card(path)
            entry["card"] = card.to_dict()
            entry["license"] = card.license
            entry["model"] = card.model
            declared = card.weights_sha256.get(path.name)
            if declared and declared != entry["sha256"]:
                issues.append(
                    {
                        "severity": "error",
                        "kind": "model_sha256_mismatch",
                        "detail": (
                            f"Card sha256 for '{path.name}' does not match "
                            "the file on disk."
                        ),
                    }
                )
            if not declared:
                issues.append(
                    {
                        "severity": "warning",
                        "kind": "model_card_missing_file",
                        "detail": (
                            f"Card does not list '{path.name}' in "
                            "weights_sha256."
                        ),
                    }
                )
        except Exception as error:  # noqa: BLE001 - card content is the finding
            issues.append(
                {
                    "severity": "error",
                    "kind": "invalid_model_card",
                    "detail": str(error),
                }
            )
    return entry, issues


def dataset_inventory(
    snapshot: str | Path | None,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    if snapshot is None:
        return None, [
            {
                "severity": "info",
                "kind": "no_dataset_snapshot",
                "detail": "No data-snapshot provided; dataset provenance is incomplete.",
            }
        ]
    path = Path(snapshot)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AutoVisionError(
            f"Cannot read dataset snapshot '{path}': {error}"
        ) from error
    if not isinstance(payload, dict):
        raise AutoVisionError("Dataset snapshot must be a JSON object.")
    issues: list[dict[str, str]] = []
    try:
        from autovision.data.versioning import verify_dataset_snapshot

        result = verify_dataset_snapshot(payload)
    except Exception as error:  # noqa: BLE001 - evidence is the finding
        issues.append(
            {
                "severity": "warning",
                "kind": "dataset_snapshot_unverifiable",
                "detail": (
                    "Dataset snapshot could not be re-derived (data moved, "
                    f"incomplete, or legacy format): {error}"
                ),
            }
        )
    else:
        if not result.get("ok", False):
            issues.append(
                {
                    "severity": "error",
                    "kind": "dataset_snapshot_drift",
                    "detail": (
                        "Dataset content does not match the snapshot: "
                        f"{result.get('differences', [])[:5]}"
                    ),
                }
            )
    return {"path": str(path), "snapshot": payload}, issues
