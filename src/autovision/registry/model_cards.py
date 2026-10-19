# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Model cards for the RF-DETR family (Apache-2.0, HF-verified weights).

These are *official* cards: metrics are the values published by Roboflow
(COCO val2017, official T4 / TensorRT 10.4 / FP16 protocol), not AutoVision
reproductions. Our own full-chain AP and latency readings are still pending
(see ``docs/benchmarks.md`` §3.1); the card's ``known_limitations`` says so
explicitly.

The weights are not redistributed by this repository: the pinned identity
(revision + sha256) lives in ``autovision.encoders.weight_hashes``. The
manifest is passed in by the caller (``rfdetr_model_cards(WEIGHT_MANIFEST)``)
so this registry module keeps the one-way layering contract (no upward
import into encoders). Base is retained for compatibility only — upstream
deprecated it in favour of Medium and no longer publishes its current
official metrics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from autovision.registry.model_card import ModelCard

__all__ = ["rfdetr_model_cards"]

_RF_DETR_SPECS: dict[str, dict[str, Any]] = {
    "rfdetr/rf-detr-nano": {
        "repo_id": "Roboflow/rf-detr-nano",
        "name": "RF-DETR Nano",
        "metrics": {
            "coco_map": 48.4,
            "coco_map50": 67.6,
            "params_m": 30.5,
            "input_px": 384.0,
        },
        "note": (
            "官方发布权重；我方全链 AP / 延迟复测待补。官方延迟为 "
            "NVIDIA T4 / TensorRT 10.4 / FP16 口径（见 docs/benchmarks.md §3.1）。"
        ),
    },
    "rfdetr/rf-detr-small": {
        "repo_id": "Roboflow/rf-detr-small",
        "name": "RF-DETR Small",
        "metrics": {
            "coco_map": 53.0,
            "coco_map50": 72.1,
            "params_m": 32.1,
            "input_px": 512.0,
        },
        "note": (
            "官方发布权重；我方全链 AP / 延迟复测待补。官方延迟为 "
            "NVIDIA T4 / TensorRT 10.4 / FP16 口径（见 docs/benchmarks.md §3.1）。"
        ),
    },
    "rfdetr/rf-detr-medium": {
        "repo_id": "Roboflow/rf-detr-medium",
        "name": "RF-DETR Medium",
        "metrics": {
            "coco_map": 54.7,
            "coco_map50": 73.6,
            "params_m": 33.7,
            "input_px": 576.0,
        },
        "note": (
            "官方发布权重；我方全链 AP / 延迟复测待补。官方延迟为 "
            "NVIDIA T4 / TensorRT 10.4 / FP16 口径（见 docs/benchmarks.md §3.1）。"
        ),
    },
    "rfdetr/rf-detr-large": {
        "repo_id": "Roboflow/rf-detr-large",
        "name": "RF-DETR Large",
        "metrics": {
            "coco_map": 56.5,
            "coco_map50": 75.1,
            "params_m": 33.9,
            "input_px": 704.0,
        },
        "note": (
            "官方发布权重；我方全链 AP / 延迟复测待补。官方延迟为 "
            "NVIDIA T4 / TensorRT 10.4 / FP16 口径（见 docs/benchmarks.md §3.1）。"
        ),
    },
    "rfdetr/rf-detr-base": {
        "repo_id": "Roboflow/rf-detr-base",
        "name": "RF-DETR Base (upstream deprecated)",
        "metrics": {},
        "note": (
            "上游已弃用 Base（由 Medium 取代），当前官方表不再发布指标；"
            "仓库保留该 HF 资产用于兼容，不建议用于新项目。"
        ),
    },
}


def _weights_file(sha256: dict[str, str]) -> str:
    """Main weight file for a pinned HF artifact."""
    if "model.safetensors" in sha256:
        return "model.safetensors"
    return sorted(sha256)[0]


def rfdetr_model_cards(manifest: Mapping[str, Any]) -> dict[str, ModelCard]:
    """Return one validated card per registered RF-DETR checkpoint.

    ``manifest`` is the pinned HF manifest (``WEIGHT_MANIFEST``); it is a
    parameter rather than an import so the registry layer does not depend on
    the encoders layer.
    """

    cards: dict[str, ModelCard] = {}
    for model_key, spec in _RF_DETR_SPECS.items():
        repo_id = str(spec["repo_id"])
        artifact = manifest[repo_id]
        weights_file = _weights_file(artifact.sha256)
        card = ModelCard(
            model=model_key,
            name=str(spec["name"]),
            train_dataset=(
                "COCO 2017（官方发布权重；训练集细节以 Roboflow model card 为准）"
            ),
            schedule="官方发布权重（未在 AutoVision 复现训练）",
            seed=None,
            metrics=dict(spec["metrics"]),
            license="Apache-2.0",
            weights_sha256={weights_file: artifact.sha256[weights_file]},
            provenance={
                "repo_id": repo_id,
                "revision": artifact.revision,
                "origin": "Roboflow official release",
            },
            known_limitations=str(spec["note"]),
            version=artifact.revision[:8],
        )
        card.validate()
        cards[model_key] = card
    return cards
