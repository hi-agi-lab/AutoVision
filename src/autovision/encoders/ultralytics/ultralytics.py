# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
r"""Ultralytics (YOLO/RT-DETR) backbone wrapper.

The ``YOLO``/``RTDETR`` objects are *not* ``nn.Module``\ s, so this wrapper
builds on :class:`NonModuleModelWrapper`. The backbone end is located by
exact-type adjacency patterns over the inner layer sequence — one pattern
per YOLO family.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from autovision.encoders._non_module_wrapper import NonModuleModelWrapper
from autovision.encoders.model_wrapper import (
    ArchTraits,
    FeatureMapsOut,
)

__all__ = ["UltralyticsModelWrapper"]


# (previous_type, boundary_type, end_at) — the backbone ends at the boundary
# module: the *second* element for v5/v6/v8 (SPPF closes the backbone) and
# the *first* element for v11/v12 (C2PSA/A2C2f close it; Upsample/Classify
# start the neck/head). Type names are matched exactly (not isinstance) like
# the baseline.
_BACKBONE_END_PATTERNS: list[tuple[str, str, str]] = [
    ("C3", "SPPF", "second"),  # YOLOv5
    ("Sequential", "SPPF", "second"),  # YOLOv6
    ("C2f", "SPPF", "second"),  # YOLOv8
    ("C2f", "Classify", "second"),  # YOLOv8 classification variants
    ("HGBlock", "Conv", "second"),  # RT-DETR
    ("C2PSA", "Upsample", "first"),  # YOLO11
    ("C2PSA", "Classify", "first"),
    ("A2C2f", "Upsample", "first"),  # YOLO12
    ("A2C2f", "Classify", "first"),
]


def _find_backbone_end(layers: nn.Sequential) -> int:
    type_names = [type(layer).__name__ for layer in layers]
    for prev_type, boundary_type, end_at in _BACKBONE_END_PATTERNS:
        for index in range(len(type_names) - 1):
            if (
                type_names[index] == prev_type
                and type_names[index + 1] == boundary_type
            ):
                return index + 1 if end_at == "second" else index
    raise ValueError(
        "Cannot locate the backbone end: no known backbone boundary pattern "
        f"found in layer types {type_names}. Unsupported model?"
    )


def _module_out_channels(module: nn.Module) -> int:
    """Output channels of a backbone-end module.

    Resolution order: explicit ``c2`` attribute, ``cv2`` projection conv,
    the module's own conv, a linear head, then a last-conv scan.
    """
    c2 = getattr(module, "c2", None)
    if isinstance(c2, int):
        return c2
    cv2 = getattr(module, "cv2", None)
    if isinstance(cv2, nn.Module):
        out = getattr(cv2, "out_channels", None)
        if isinstance(out, int):
            return out
    conv = getattr(module, "conv", None)
    if isinstance(conv, nn.Conv2d):
        return int(conv.out_channels)
    linear = getattr(module, "linear", None)
    if isinstance(linear, nn.Linear):
        return int(linear.out_features)
    # Fallback: the last Conv2d/Linear in registration order.
    last: nn.Module | None = None
    for sub_module in module.modules():
        if isinstance(sub_module, (nn.Conv2d, nn.Linear)):
            last = sub_module
    if last is None:
        raise ValueError(
            f"Cannot determine output channels of backbone end module "
            f"'{type(module).__name__}'."
        )
    if isinstance(last, nn.Conv2d):
        return int(last.out_channels)
    return int(last.out_features)


class UltralyticsModelWrapper(NonModuleModelWrapper):
    """Adapts an ultralytics ``YOLO``/``RTDETR`` object to ``EncoderProtocol``."""

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self._enable_gradients(model)
        inner = model.model
        inner.train()
        backbone_end = _find_backbone_end(inner.model)
        self._backbone_end = backbone_end
        # Plain list, NOT a registered submodule: the backbone layers are the
        # same objects as inner.model[0:end+1]; registering them would
        # duplicate every parameter in the state dict under a second name.
        self._backbone_layers: list[nn.Module] = list(inner.model)[: backbone_end + 1]
        self._feature_dim = _module_out_channels(inner.model[backbone_end])

    # ── EncoderProtocol protocol ───────────────────────────────────────────
    @property
    def inner_model(self) -> nn.Module:
        return self._model[0].model

    def feature_maps(self, x: Tensor) -> FeatureMapsOut:
        features = x
        for layer in self._backbone_layers:
            features = layer(features)
        return {"features": features}

    def encoder_dim(self) -> int:
        return self._feature_dim

    def arch_traits(self) -> ArchTraits:
        return {"model_type": "convolutional", "norm_type": "batchnorm"}

    # ── gradients ───────────────────────────────────────────────────────
    @staticmethod
    def _enable_gradients(model: Any) -> None:
        """Re-enable gradients that checkpoint loading disables.

        Honors the model's ``freeze`` spec: an int freezes the first N
        layers, a list freezes the given layer indices.
        """
        inner = model.model
        inner.requires_grad_(True)
        freeze = None
        args = getattr(inner, "args", None)
        if isinstance(args, dict):
            freeze = args.get("freeze")
        if freeze is None:
            freeze = getattr(inner, "yaml", {}).get("freeze", None)
        if not freeze:
            return
        layers = list(inner.model)
        frozen_indices = (
            range(int(freeze))
            if isinstance(freeze, int)
            else [int(index) for index in freeze]
        )
        for index in frozen_indices:
            if 0 <= index < len(layers):
                layers[index].requires_grad_(False)
