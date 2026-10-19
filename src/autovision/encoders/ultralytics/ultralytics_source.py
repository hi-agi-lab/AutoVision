# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Ultralytics package: YOLOv5/v6/v8/11/12/26 and RT-DETR backbones.

**License isolation (AGPL-3.0)**: ``ultralytics`` is AGPL-licensed and must
never become a core dependency. It is imported lazily inside every method,
declared only as the ``autovision[ultralytics]`` extra, and this package
degrades gracefully (empty model list) when the dependency is absent.

Model names mirror the baseline: pre-trained ``"<family><scale>.pt"`` names
synthesized from ``ultralytics.utils.downloads.GITHUB_ASSETS_STEMS`` plus
architecture ``"<family><scale>.yaml"`` configs, gated per family by the
installed ultralytics version and validated through ``yaml_model_load``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from pathlib import Path
from typing import Any

import torch

from autovision.encoders.package import CatalogModelSource
from autovision.encoders.ultralytics.ultralytics import UltralyticsModelWrapper

__all__ = ["ULTRALYTICS_SOURCE", "UltralyticsSource"]

# Detection-only families exposed by this package, with the minimum
# ultralytics version that ships each family.
_FAMILIES: dict[str, tuple[int, int, int]] = {
    "yolov5": (8, 0, 0),
    "yolov6": (8, 0, 0),
    "yolov8": (8, 0, 0),
    "yolo11": (8, 3, 0),
    "yolo12": (8, 3, 78),
    "yolo26": (8, 4, 0),
    "rtdetr": (8, 0, 140),
}

# Architecture yaml scales per family (rtdetr ships l/x only).
_YAML_SCALES: dict[str, list[str]] = {
    "yolov8": ["n", "s", "m", "l", "x"],
    "yolo11": ["n", "s", "m", "l", "x"],
    "yolo12": ["n", "s", "m", "l", "x"],
    "yolo26": ["n", "s", "m", "l", "x"],
    "rtdetr": ["l", "x"],
}

_MIN_PACKAGE_VERSION = (8, 1, 0)


def _get_ultralytics() -> Any:
    try:
        return importlib.import_module("ultralytics")
    except ImportError as error:
        raise ImportError(
            "The 'ultralytics' package is required for ultralytics models. "
            "Install it with: pip install 'autovision[ultralytics]'. Note "
            "that ultralytics is AGPL-3.0 licensed."
        ) from error


def _ultralytics_version() -> tuple[int, ...]:
    try:
        raw = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        try:
            raw = _get_ultralytics().__version__
        except (ImportError, AttributeError):
            return (0, 0, 0)
    numbers = re.findall(r"\d+", raw)
    return tuple(int(n) for n in numbers[:3]) if len(numbers) >= 3 else (0, 0, 0)


def _family_of(name: str) -> str | None:
    for family in _FAMILIES:
        if name.startswith(family):
            return family
    return None


class UltralyticsSource(CatalogModelSource):
    name = "ultralytics"

    @classmethod
    def catalog_names(cls) -> list[str]:
        try:
            ultralytics = _get_ultralytics()
        except ImportError:
            return []
        version = _ultralytics_version()
        if version < _MIN_PACKAGE_VERSION:
            return []
        names: set[str] = set()
        # Pre-trained .pt checkpoints shipped on GitHub assets.
        stems = getattr(ultralytics.utils.downloads, "GITHUB_ASSETS_STEMS", set())
        for stem in stems:
            family = _family_of(stem)
            if family is not None and version >= _FAMILIES[family]:
                names.add(f"{cls.name}/{stem}.pt")
        # Architecture yamls per family scale (validated through the loader).
        from ultralytics.nn.tasks import yaml_model_load

        for family, scales in _YAML_SCALES.items():
            if version < _FAMILIES[family]:
                continue
            extra = (
                ["-p2", "-p6"] if family == "yolo26" else []
            )
            for scale in scales:
                candidates = [f"{family}{scale}{suffix}.yaml" for suffix in ["", *extra]]
                for candidate in candidates:
                    try:
                        yaml_model_load(candidate)
                    except Exception:
                        continue
                    names.add(f"{cls.name}/{candidate}")
        return sorted(names)

    @classmethod
    def recognizes(cls, model: Any) -> bool:
        return isinstance(model, UltralyticsModelWrapper)

    @classmethod
    def build(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> Any:
        if num_input_channels != 3:
            raise ValueError(
                f"Ultralytics models only support 3 input channels, got "
                f"{num_input_channels}."
            )
        ultralytics = _get_ultralytics()
        name = model_name
        if not load_weights and name.endswith(".pt"):
            # Untrained: build from the architecture yaml instead of the
            # pre-trained checkpoint.
            name = name[: -len(".pt")] + ".yaml"
        args = dict(model_args or {})
        if "detr" in name.lower():
            return ultralytics.RTDETR(model=name, **args)
        return ultralytics.YOLO(model=name, **args)

    @classmethod
    def as_encoder(cls, model: Any) -> UltralyticsModelWrapper:
        return UltralyticsModelWrapper(model)

    @classmethod
    def export_weights(
        cls, model: Any, out: Path, log_example: bool = True
    ) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        yolo = (
            model.raw_module()
            if isinstance(model, UltralyticsModelWrapper)
            else model
        )
        if _ultralytics_version() >= (8, 3, 0) and hasattr(yolo, "save"):
            yolo.save(str(out))
        else:
            torch.save({"model": yolo.model}, out)
        if log_example:
            print(
                f"Exported ultralytics model to '{out}'. Load with "
                f"ultralytics.YOLO('{out}')."
            )


ULTRALYTICS_SOURCE = UltralyticsSource()

from autovision.encoders.package_helpers import register_source  # noqa: E402

register_source(ULTRALYTICS_SOURCE)
