# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Deployment gates for the private model registry (roadmap 90d, phase 4).

A gate returns a list of human-readable failure reasons; an empty list is a
pass. Gates read the immutable registry entry (file + sidecars + metadata),
so a promotion cannot pass on a stale source path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from autovision.registry.model_card import load_model_card
from autovision.registry.model_registry import RegistryEntry, compute_file_sha256

__all__ = [
    "BenchmarkGate",
    "DataSnapshotGate",
    "DeploymentGate",
    "LicenseGate",
    "ModelCardGate",
    "ParityGate",
    "Sha256Gate",
    "default_gates",
]

_DEFAULT_ALLOWLIST = frozenset(
    {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MPL-2.0"}
)


class DeploymentGate(Protocol):
    name: str

    def check(self, entry: RegistryEntry) -> list[str]: ...


class Sha256Gate:
    name = "sha256"

    def check(self, entry: RegistryEntry) -> list[str]:
        if not entry.path.is_file():
            return [f"artifact missing: {entry.path}"]
        actual = compute_file_sha256(entry.path)
        return [] if actual == entry.sha256 else [f"sha256 mismatch ({actual})"]


class ModelCardGate:
    name = "model_card"

    def check(self, entry: RegistryEntry) -> list[str]:
        card_path = Path(str(entry.path) + ".modelcard.json")
        if not card_path.is_file():
            return [f"missing model card: {card_path.name}"]
        try:
            card = load_model_card(entry.path)
            card.validate()
        except Exception as error:  # noqa: BLE001 - card content is the finding
            return [f"invalid model card: {error}"]
        declared = card.weights_sha256.get(entry.path.name)
        if declared and declared != entry.sha256:
            return ["card sha256 does not match registered artifact"]
        if not declared:
            return [f"card does not list '{entry.path.name}' in weights_sha256"]
        return []


class LicenseGate:
    name = "license"

    def __init__(self, allowlist: set[str] | frozenset[str] | None = None) -> None:
        self.allowlist = frozenset(allowlist or _DEFAULT_ALLOWLIST)

    def check(self, entry: RegistryEntry) -> list[str]:
        try:
            card = load_model_card(entry.path)
        except Exception as error:  # noqa: BLE001
            return [f"cannot read model card: {error}"]
        if not card.license:
            return ["model card has no license"]
        if card.license not in self.allowlist:
            return [f"license '{card.license}' is not in the deployment allowlist"]
        return []


class BenchmarkGate:
    name = "benchmark"

    def check(self, entry: RegistryEntry) -> list[str]:
        try:
            card = load_model_card(entry.path)
        except Exception as error:  # noqa: BLE001
            return [f"cannot read model card: {error}"]
        problems: list[str] = []
        if float(card.metrics.get("coco_map", 0.0)) <= 0.0:
            problems.append("model card has no positive coco_map")
        if not card.latency:
            problems.append("model card has no latency record")
        return problems


def _metadata_path(entry: RegistryEntry, key: str) -> Path | None:
    value = entry.metadata.get(key)
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else entry.path.parent / path


class ParityGate:
    name = "parity"

    def __init__(self, *, require_report: bool = True) -> None:
        # A self-asserted metadata flag is not evidence: require the report
        # file (and, when recorded, its sha256) by default.
        self.require_report = require_report

    def check(self, entry: RegistryEntry) -> list[str]:
        if not self.require_report and entry.metadata.get("parity_passed") is True:
            return []
        path = _metadata_path(entry, "parity_report")
        if path is None:
            candidate = Path(str(entry.path) + ".parity.json")
            path = candidate if candidate.is_file() else None
        if path is None:
            return ["no parity_report file (or artifact.parity.json sidecar)"]
        if not path.is_file():
            return [f"parity report missing: {path}"]
        expected_sha = entry.metadata.get("parity_report_sha256")
        if expected_sha and compute_file_sha256(path) != expected_sha:
            return ["parity report sha256 does not match registry metadata"]
        try:
            payload = json.loads(path.read_text())
        except Exception as error:  # noqa: BLE001
            return [f"invalid parity report: {error}"]
        return [] if payload.get("passed") is True else ["parity report did not pass"]


class DataSnapshotGate:
    name = "data_snapshot"

    def check(self, entry: RegistryEntry) -> list[str]:
        path = _metadata_path(entry, "data_snapshot")
        if path is None:
            return ["no data_snapshot in registry metadata"]
        return [] if path.is_file() else [f"data snapshot missing: {path}"]


def default_gates() -> list[DeploymentGate]:
    """The standard production promotion set (no dataset-specific gate)."""
    return [Sha256Gate(), ModelCardGate(), LicenseGate(), BenchmarkGate(), ParityGate()]
