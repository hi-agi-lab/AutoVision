# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Hard-example manifests and weighted sampling (roadmap 90d, phase 3).

``analyze-errors`` writes ``hard_examples.json`` next to its report. The
manifest is deliberately a separate file (not a mutated COCO JSON): the
same annotation set can be trained with and without hard-example
weighting, and the weighting is reproducible from the manifest + seed.

v1 boundary: single-device weighted sampling. Multi-GPU DDP uses a
DistributedSampler that would override these weights; the training entry
rejects the combination instead of silently ignoring the manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import WeightedRandomSampler

from autovision.errors import AutoVisionError

__all__ = [
    "HardExample",
    "HardExampleManifest",
    "build_weighted_sampler",
]


@dataclass(frozen=True)
class HardExample:
    image: str
    split: str = "train"
    tp: int = 0
    fp: int = 0
    fn: int = 0
    fp_reasons: dict[str, int] = field(default_factory=dict)
    max_fn_area: float = 0.0
    classes_fn: tuple[int, ...] = ()

    @property
    def error(self) -> int:
        return max(0, int(self.fp)) + max(0, int(self.fn))

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "split": self.split,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "fp_reasons": dict(self.fp_reasons),
            "max_fn_area": self.max_fn_area,
            "classes_fn": list(self.classes_fn),
        }


@dataclass
class HardExampleManifest:
    images: tuple[HardExample, ...]
    iou_threshold: float = 0.5
    score_threshold: float = 0.0
    source_run: str | None = None
    schema_version: int = 1
    _base_dir: Path | None = field(default=None, repr=False, compare=False)

    @classmethod
    def load(cls, path: str | Path) -> "HardExampleManifest":
        source = Path(path)
        try:
            payload = json.loads(source.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise AutoVisionError(
                f"Cannot read hard-example manifest '{source}': {error}"
            ) from error
        if not isinstance(payload, Mapping):
            raise AutoVisionError("Hard-example manifest must be a JSON object.")
        version = int(payload.get("schema_version", 0))
        if version != 1:
            raise AutoVisionError(
                f"Unsupported hard-example schema_version {version}; expected 1."
            )
        raw_images = payload.get("images")
        if not isinstance(raw_images, list):
            raise AutoVisionError("Hard-example manifest needs an 'images' list.")
        images: list[HardExample] = []
        for index, row in enumerate(raw_images):
            if not isinstance(row, Mapping) or "image" not in row:
                raise AutoVisionError(
                    f"Hard-example entry {index} needs an 'image' path."
                )
            images.append(
                HardExample(
                    image=str(row["image"]),
                    split=str(row.get("split", "train")),
                    tp=int(row.get("tp", 0)),
                    fp=int(row.get("fp", 0)),
                    fn=int(row.get("fn", 0)),
                    fp_reasons={
                        str(key): int(value)
                        for key, value in dict(row.get("fp_reasons", {})).items()
                    },
                    max_fn_area=float(row.get("max_fn_area", 0.0)),
                    classes_fn=tuple(int(v) for v in row.get("classes_fn", ())),
                )
            )
        return cls(
            images=tuple(images),
            iou_threshold=float(payload.get("iou_threshold", 0.5)),
            score_threshold=float(payload.get("score_threshold", 0.0)),
            source_run=payload.get("source_run"),
            schema_version=version,
            _base_dir=source.parent,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_run": self.source_run,
            "iou_threshold": self.iou_threshold,
            "score_threshold": self.score_threshold,
            "images": [entry.to_dict() for entry in self.images],
        }

    def _resolve(self, image: str) -> str:
        path = Path(image)
        if not path.is_absolute() and self._base_dir is not None:
            path = self._base_dir / path
        return str(path.resolve())

    def weights_for(
        self,
        paths: Sequence[str | Path],
        *,
        alpha: float = 1.0,
        max_repeat: float = 5.0,
        split: str = "train",
    ) -> torch.Tensor:
        """Per-image sampling weights (1.0 baseline, higher for errors).

        Error signal is ``fp + fn`` normalized by the manifest maximum; the
        weight is ``min(max_repeat, 1 + alpha * normalized_error)``. Images
        absent from the manifest keep weight 1.0, so a partial audit is
        safe.
        """
        if alpha < 0:
            raise AutoVisionError(f"alpha must be >= 0, got {alpha}.")
        if max_repeat < 1:
            raise AutoVisionError(f"max_repeat must be >= 1, got {max_repeat}.")
        entries = {
            self._resolve(entry.image): entry
            for entry in self.images
            if split is None or entry.split == split
        }
        errors = [entry.error for entry in entries.values()]
        maximum = max(errors, default=0)
        weights: list[float] = []
        for path in paths:
            entry = entries.get(str(Path(path).resolve()))
            if entry is None or maximum <= 0:
                weights.append(1.0)
            else:
                weights.append(
                    min(max_repeat, 1.0 + alpha * entry.error / maximum)
                )
        return torch.tensor(weights, dtype=torch.double)


def build_weighted_sampler(
    manifest: HardExampleManifest,
    paths: Sequence[str | Path],
    *,
    alpha: float = 1.0,
    max_repeat: float = 5.0,
    seed: int = 0,
) -> WeightedRandomSampler:
    """Build a deterministic replacement sampler for the training split."""
    if not paths:
        raise AutoVisionError("Hard-example sampling needs at least one image path.")
    weights = manifest.weights_for(paths, alpha=alpha, max_repeat=max_repeat)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
