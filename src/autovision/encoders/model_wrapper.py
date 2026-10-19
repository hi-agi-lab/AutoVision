# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Backbone adapter contract for the pretraining path.

Any backbone enters the framework through :class:`EncoderProtocol` — a
runtime-checkable structural type, so third-party ``nn.Module`` subclasses
need not inherit anything. Two optional capabilities extend it for
intermediate-layer methods: :class:`MultiScaleViTEncoder` and
:class:`MultiScaleCNNEncoder` (per-stage feature taps). ``ArchTraits``
feeds the ``"auto"`` decisions such as per-architecture weight decay.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict

from torch import Tensor, nn
from typing_extensions import (
    NotRequired,
    Required,
    get_protocol_members,
    runtime_checkable,
)

__all__ = [
    "ArchTraits",
    "ArchTraitsSource",
    "EncoderProtocol",
    "FeatureMapsOut",
    "MultiScaleCNNEncoder",
    "MultiScaleViTEncoder",
    "PatchSize",
    "PoolOut",
    "missing_encoder_members",
]


class FeatureMapsOut(TypedDict, total=False):
    features: Required[Tensor]  # (B, C, H_out, W_out)
    cls_token: NotRequired[Tensor]  # (B, C)


class PoolOut(TypedDict, total=False):
    pooled_features: Required[Tensor]  # (B, C, 1, 1)


class ArchTraits(TypedDict):
    model_type: Literal["convolutional", "transformer", "hybrid"]
    norm_type: Literal["batchnorm", "layernorm"]


@runtime_checkable
class EncoderProtocol(Protocol):
    """Everything the pretraining path needs from a backbone.

    Declared flat on purpose: one structural type, one ``isinstance``.
    The trailing members are the subset of the ``torch.nn.Module`` surface
    that the engine and the checkpoint code rely on.
    """

    def feature_maps(self, x: Tensor, *args: Any, **kwargs: Any) -> FeatureMapsOut: ...

    def pool_features(self, x: FeatureMapsOut, *args: Any, **kwargs: Any) -> PoolOut: ...

    def encoder_dim(self) -> int: ...

    def raw_module(self) -> Any: ...

    def parameters(self, *args: Any, **kwargs: Any) -> Any: ...

    def modules(self) -> Any: ...

    def eval(self) -> Any: ...

    def state_dict(self, *args: Any, **kwargs: Any) -> Any: ...

    def load_state_dict(
        self, state_dict: Any, strict: bool = True, assign: bool = False
    ) -> Any: ...

    def requires_grad_(self, requires_grad: bool = True) -> Any: ...


@runtime_checkable
class ArchTraitsSource(Protocol):
    """Optional capability: per-architecture traits for ``"auto"`` rules."""

    def arch_traits(self) -> ArchTraits: ...


@runtime_checkable
class PatchSize(Protocol):
    """Optional capability: ViT patch extent, used for multi-scale math."""

    def patch_size(self) -> int: ...


@runtime_checkable
class MultiScaleViTEncoder(EncoderProtocol, Protocol):
    """ViT backbones with intermediate-layer taps (per-block indices)."""

    def stage_feature_maps(
        self, x: Tensor, layer_indices: Any
    ) -> list[FeatureMapsOut]: ...

    def stage_dims(self) -> list[int]: ...

    def patch_size(self) -> int: ...


@runtime_checkable
class MultiScaleCNNEncoder(EncoderProtocol, Protocol):
    """Convolutional backbones with per-stage taps (stage indices)."""

    def stage_feature_maps(
        self, x: Tensor, layer_indices: Any
    ) -> list[FeatureMapsOut]: ...

    def stage_dims(self) -> list[int]: ...

    def stage_strides(self) -> list[int]: ...


def missing_encoder_members(
    candidate: Any, exclude_module_attrs: bool = False
) -> list[str]:
    """Protocol members ``candidate`` lacks — used to build actionable
    ``UnknownModelError`` messages.

    With ``exclude_module_attrs``, members that plain ``nn.Module`` already
    provides are not reported (they cannot be missing on an ``nn.Module``).
    """
    inherited = set(dir(nn.Module))
    gaps = []
    for member in get_protocol_members(EncoderProtocol):
        if exclude_module_attrs and member in inherited:
            continue
        if not hasattr(candidate, member):
            gaps.append(member)
    return sorted(gaps)
