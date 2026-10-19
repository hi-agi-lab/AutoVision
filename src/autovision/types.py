# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Public type aliases and batch schemas shared across subsystems."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, List, TypedDict

import torch
from pydantic import BeforeValidator, Field
from typing_extensions import NotRequired

__all__ = [
    "Batch",
    "DatasetItem",
    "ImageSizeTuple",
    "PathLike",
]

PathLike = str | Path


def _try_convert_to_tuple(value: object) -> object:
    """Coerce lists (as produced by CLI/YAML inputs) into tuples.

    Pydantic strict mode rejects lists for tuple fields, so list inputs are
    converted before validation.
    """
    if isinstance(value, list):
        return tuple(value)
    return value


ImageSizeTuple = Annotated[
    tuple[int, int],
    BeforeValidator(_try_convert_to_tuple),
    Field(strict=False),
]


class DatasetItem(TypedDict):
    """One sample produced by the pretraining dataset before collation."""

    filename: str
    views: List[torch.Tensor]  # one (C, H, W) tensor per view
    masks: NotRequired[List[torch.Tensor]]  # one (H, W) region-id tensor per view
    geometries: NotRequired[List[torch.Tensor]]


class Batch(DatasetItem):
    """Collated pretraining batch (default torch collate).

    ``views`` holds one ``(B, C, H, W)`` tensor per view; ``filename`` is a
    list with one entry per sample.
    """
