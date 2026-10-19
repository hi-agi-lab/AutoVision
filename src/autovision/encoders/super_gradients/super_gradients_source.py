# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""super_gradients package (optional extra ``[super-gradients]``).

Only the detection-oriented ``CustomizableDetector`` family is adapted for
now — SuperGradients' segmentation modules land together with the M4+
segmentation task support. Like every package, SuperGradients itself is an
optional dependency: without it the model list is empty and building a model
raises a helpful error.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from autovision.encoders.model_wrapper import ArchTraits
from autovision.encoders.package import CatalogModelSource
from autovision.errors import UnknownModelError

__all__ = ["SUPER_GRADIENTS_SOURCE", "SuperGradientsSource"]

logger = logging.getLogger("autovision")


class CustomizableDetectorModelWrapper(nn.Module):
    """Wrapper for SuperGradients ``CustomizableDetector`` models.

    The raw model is stored in a one-element list (``self._model = [model]``)
    so the wrapper does not re-register the model's parameters as its own —
    mirrors the baseline trick for non-standard module trees.
    """

    _supported_model_classes: tuple[type[nn.Module], ...] = ()

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = [model]
        self._backbone = model.backbone
        self._pool = nn.AdaptiveAvgPool2d((1, 1))

    @classmethod
    def recognizes_cls(cls, model_cls: type) -> bool:
        if not cls._supported_model_classes:
            return False
        try:
            return issubclass(model_cls, cls._supported_model_classes)
        except TypeError:  # pragma: no cover - non-class argument
            return False

    def raw_module(self) -> nn.Module:
        return self._model[0]

    def encoder_dim(self) -> int:
        out_channels: list[int] | int = self._backbone.out_channels  # type: ignore[attr-defined]
        return out_channels[-1] if isinstance(out_channels, list) else out_channels

    def feature_maps(self, x: Tensor) -> dict[str, Tensor]:
        features = self._backbone(x)
        if isinstance(features, Tensor):
            features = [features]
        # Some backbones return ``None`` placeholders for unused levels.
        feats = [f for f in features if f is not None][-1]
        return {"features": feats}

    def pool_features(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        return {"pooled_features": self._pool(x["features"])}

    def arch_traits(self) -> ArchTraits:
        return {"model_type": "convolutional", "norm_type": "batchnorm"}


def _load_customizable_detector_cls() -> tuple[type, ...]:
    """The CustomizableDetector class when super_gradients is installed."""
    try:
        from super_gradients.training.models import CustomizableDetector
    except ImportError:
        return ()
    return (CustomizableDetector,)


CustomizableDetectorModelWrapper._supported_model_classes = (  # type: ignore[assignment]
    _load_customizable_detector_cls()
)


class SuperGradientsSource(CatalogModelSource):
    name = "super_gradients"

    _FEATURE_EXTRACTORS: list[type[CustomizableDetectorModelWrapper]] = [
        CustomizableDetectorModelWrapper,
    ]

    @classmethod
    def catalog_names(cls) -> list[str]:
        try:
            from super_gradients.training import models
        except ImportError:
            return []
        names = {
            f"{cls.name}/{model_name}"
            for model_name, model_cls in models.ARCHITECTURES.items()
            if cls.recognizes_cls(model_cls)
        }
        return sorted(names)

    @classmethod
    def recognizes(cls, model: Any) -> bool:
        if isinstance(model, CustomizableDetectorModelWrapper):
            return True
        raw = model.raw_module() if hasattr(model, "raw_module") else model
        return cls.recognizes_cls(type(raw))

    @classmethod
    def recognizes_cls(cls, model_cls: type) -> bool:
        return any(fe.recognizes_cls(model_cls) for fe in cls._FEATURE_EXTRACTORS)

    @classmethod
    def build(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> nn.Module:
        try:
            from super_gradients.training import models
        except ImportError as error:
            raise ValueError(
                f"Cannot create model 'super_gradients/{model_name}' because "
                f"super-gradients is not installed. Install it with: "
                f"pip install 'autovision[super-gradients]'."
            ) from error
        if num_input_channels != 3:
            raise ValueError(
                f"SuperGradients models only support 3 input channels, got "
                f"{num_input_channels}."
            )
        args: dict[str, Any] = {"num_classes": 10}
        if model_args:
            args.update(model_args)
        if not load_weights:
            args["checkpoint_path"] = None
            args["pretrained_weights"] = None
        return models.get(model_name=model_name, **args)

    @classmethod
    def as_encoder(cls, model: nn.Module) -> CustomizableDetectorModelWrapper:
        for feature_extractor in cls._FEATURE_EXTRACTORS:
            if feature_extractor.recognizes_cls(type(model)):
                return feature_extractor(model)
        raise UnknownModelError(
            f"Unknown super_gradients model of type '{type(model).__name__}'. "
            f"Supported families: customizable detectors."
        )

    @classmethod
    def export_weights(cls, model: Any, out: Path, log_example: bool = True) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        raw = model.raw_module() if hasattr(model, "raw_module") else model
        if not cls.recognizes(raw):
            raise ValueError(
                f"The super_gradients package only exports SuperGradients "
                f"models, got '{type(raw).__name__}'."
            )
        torch.save(raw.state_dict(), out)
        if log_example:
            print(
                f"Exported super_gradients model to '{out}'. Load with: "
                f"super_gradients.training.models.get(model_name=..., "
                f"checkpoint_path='{out}')."
            )


SUPER_GRADIENTS_SOURCE = SuperGradientsSource()

from autovision.encoders.package_helpers import register_source  # noqa: E402

register_source(SUPER_GRADIENTS_SOURCE)
