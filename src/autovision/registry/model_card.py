# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Model cards for published spectrum weights (roadmap design doc 29 §3.4).

A model card is the mandatory metadata sidecar of every weight-zoo
entry. Two fields are hard requirements (D4): ``license`` must be a
non-empty SPDX identifier and every listed weight file must carry a
64-hex sha256 — a card that cannot answer "what license and which exact
bytes" is not publishable. Serialization is plain JSON next to the
weights (``<weights>.modelcard.json``).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from autovision.errors import AutoVisionError

__all__ = ["ModelCard", "ModelCardError", "load_model_card"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ModelCardError(AutoVisionError):
    pass


@dataclass
class LatencyRecord:
    backend: str  # e.g. "tensorrt", "onnxruntime", "torch"
    precision: str  # "fp32" | "fp16" | "int8"
    hardware: str  # logged device, e.g. "NVIDIA T4"
    batch_size: int
    p50_ms: float
    script: str  # reproduction script path


@dataclass
class ModelCard:
    """Publishable metadata for one weight entry (doc 29 §3.4 fields)."""

    model: str  # registry key, e.g. "detr/s"
    name: str  # human-readable, e.g. "DETR-S (HGNetV2-B1 + D-FINE-S decoder)"
    train_dataset: str  # e.g. "COCO 2017 train2017"
    schedule: str  # e.g. "12ep distill (RF-DETR-n teacher)" / "6ep finetune"
    #: None for third-party official weights whose training seed is not
    #: published; our own released weights must record the seed.
    seed: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)  # {"coco_map": ...}
    latency: list[LatencyRecord] = field(default_factory=list)
    license: str = ""  # SPDX id of the WEIGHTS — required
    weights_sha256: dict[str, str] = field(default_factory=dict)  # file -> digest
    provenance: dict[str, str] = field(default_factory=dict)  # assembly lineage
    known_limitations: str = ""
    version: str = ""

    def validate(self) -> None:
        """Raise :class:`ModelCardError` unless the card is publishable."""
        problems: list[str] = []
        if not self.model or not self.name:
            problems.append("model/name must be non-empty")
        if not self.license.strip():
            problems.append("license is required (D4: no unlicensed weights)")
        for path, digest in self.weights_sha256.items():
            if not _SHA256_RE.match(digest):
                problems.append(f"weights_sha256['{path}'] is not a 64-hex sha256")
        for record in self.latency:
            if record.precision not in {"fp32", "fp16", "int8"}:
                problems.append(f"latency[{record.backend}] precision "
                                f"'{record.precision}' is not fp32/fp16/int8")
        if problems:
            raise ModelCardError(
                "Invalid model card: " + "; ".join(problems)
            )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["latency"] = [asdict(record) for record in self.latency]
        return payload

    def save(self, weights_path: Path | str) -> Path:
        """Validate and write ``<weights>.modelcard.json``; returns path."""
        self.validate()
        path = Path(str(weights_path) + ".modelcard.json")
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def load_model_card(weights_path: Path | str) -> ModelCard:
    """Read back a card written by :meth:`ModelCard.save`."""
    path = Path(str(weights_path) + ".modelcard.json")
    if not path.exists():
        raise ModelCardError(f"No model card next to weights: '{path}'")
    payload = json.loads(path.read_text())
    latency = [LatencyRecord(**record) for record in payload.pop("latency", [])]
    return ModelCard(latency=latency, **payload)
