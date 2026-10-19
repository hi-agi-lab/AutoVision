# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pretraining method layer."""

from autovision.pretext import teachers  # noqa: F401  (registers methods)
from autovision.pretext.method_args import PretextArgs
from autovision.pretext.recipe import PretextRecipe, StepOutput
from autovision.pretext.teachers import get_method_cls, list_methods

__all__ = ["PretextRecipe", "PretextArgs", "StepOutput", "get_method_cls", "list_methods"]
