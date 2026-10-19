# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""PretextRecipe registry and lookup helpers."""

from __future__ import annotations

from typing import Any

import torch

from autovision.config.registry import Registry
from autovision.encoders._model_helpers import load_state_dict_compat
from autovision.pretext.densecl import DenseCL
from autovision.pretext.detcon import DetConB, DetConS
from autovision.pretext.dino import DINO
from autovision.pretext.dinov2 import DINOv2
from autovision.pretext.dinov3 import DINOv3
from autovision.pretext.distillationv2 import DistillationV2
from autovision.pretext.distillationv3 import DistillationV3
from autovision.pretext.recipe import PretextRecipe, prepare_teacher
from autovision.pretext.simclr import SimCLR

__all__ = ["get_method_cls", "build_teacher", "list_methods"]

_METHOD_REGISTRY: Registry[type[PretextRecipe]] = Registry("method")

_METHOD_REGISTRY.register("distillationv3", "distillation")(DistillationV3)
_METHOD_REGISTRY.register("distillationv2")(DistillationV2)
_METHOD_REGISTRY.register("simclr")(SimCLR)
_METHOD_REGISTRY.register("dino")(DINO)
_METHOD_REGISTRY.register("dinov2")(DINOv2)
_METHOD_REGISTRY.register("dinov3")(DINOv3)
_METHOD_REGISTRY.register("densecl")(DenseCL)
_METHOD_REGISTRY.register("detcons")(DetConS)
_METHOD_REGISTRY.register("detconb")(DetConB)


def build_teacher(method_args: Any, num_input_channels: int = 3) -> Any:
    """Instantiate, freeze, and eval the teacher model.

    Shared by the distillation methods. Reads ``teacher``, ``teacher_args``,
    ``teacher_weights``, and ``teacher_pretrained`` off the method args (any
    args class with these fields works). By default
    (``teacher_pretrained=True``) the teacher is requested with pretrained
    weights (``weights="DEFAULT"`` for packages that support it). Set
    ``teacher_pretrained=False`` for a randomly initialized teacher —
    distilling from a random teacher produces meaningless embeddings.
    """
    from autovision.encoders.package_helpers import get_wrapped_model

    teacher = method_args.teacher
    if isinstance(teacher, str):
        model_args = dict(method_args.teacher_args or {})
        if method_args.teacher_pretrained and method_args.teacher_weights is None:
            model_args.setdefault("weights", "DEFAULT")
        teacher = get_wrapped_model(
            teacher,
            num_input_channels=num_input_channels,
            model_args=model_args,
            load_weights=method_args.teacher_weights is None,
        )
    if method_args.teacher_weights is not None:
        state_dict = torch.load(
            method_args.teacher_weights, map_location="cpu", weights_only=True
        )
        load_state_dict_compat(teacher.raw_module(), state_dict)
    # eval + no grads + zeroed dropout/drop-path when the wrapper supports
    # it.
    prepare_teacher(teacher)
    return teacher


def list_methods() -> list[str]:
    """All registered method names (hidden ones excluded)."""
    return _METHOD_REGISTRY.list_names()


def get_method_cls(method: str) -> type[PretextRecipe]:
    try:
        return _METHOD_REGISTRY.get(method)
    except KeyError as error:
        raise ValueError(str(error)) from None
