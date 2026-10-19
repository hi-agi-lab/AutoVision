# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""AutoVision: enterprise visual model training framework.

Public API:

- :func:`autovision.pretrain` — self-supervised pretraining / distillation.
- :func:`autovision.detection` / :func:`autovision.load_model` — object
  detection model handles (``train / val / predict / export``).
- :func:`autovision.ensure_coco8` — bundled synthetic smoke dataset.
- :func:`autovision.list_models` — all models exposed by registered packages.
- :func:`autovision.list_methods` — all registered pretraining methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from autovision import errors
from autovision._compat import install_graphmodule_deepcopy_fix
from autovision._version import __version__
from autovision.cli import (
    PretrainConfig,
    PretrainResult,
    TaskTrainResult,
    pretrain,
    train_depth_estimation,
    train_image_classification,
    train_instance_segmentation,
    train_object_detection,
    train_oriented_object_detection,
    train_panoptic_segmentation,
    train_pose_estimation,
    train_semantic_segmentation,
)
from autovision.cli.extract_video_frames import extract_video_frames
from autovision.cli.model_handle import (
    DetectionModel,
    detection,
    evaluate_object_detection,
    load_model,
)
from autovision.data.coco8 import ensure_coco8
from autovision.encoders import list_models
from autovision.errors import AutoVisionError
from autovision.export import benchmark, measure_model_cost
from autovision.features import embed
from autovision.plugins import (
    PluginInfo,
    discover_plugins,
    list_plugins,
    register_plugin,
)
from autovision.pretext import list_methods
from autovision.registry.catalog import (
    describe_model,
    get_model_entry,
    list_task_models,
)
from autovision.runtime.checkpoint import CHECKPOINT_AUTOVISION_KEY
from autovision.tasks.task_model_helpers import load_task_model

# Third-party defect shims must be installed before any model construction.
install_graphmodule_deepcopy_fix()

__all__ = [
    "AutoVisionError",
    "DetectionModel",
    "PluginInfo",
    "PretrainConfig",
    "PretrainResult",
    "__version__",
    "detection",
    "describe_model",
    "discover_plugins",
    "errors",
    "get_model_entry",
    "list_methods",
    "list_models",
    "list_plugins",
    "list_task_models",
    "register_plugin",
    "embed",
    "benchmark",
    "measure_model_cost",
    "ensure_coco8",
    "evaluate_object_detection",
    "extract_video_frames",
    "load_embedding_model",
    "load_model",
    "load_task_model",
    "pretrain",
    "train_image_classification",
    "train_instance_segmentation",
    "train_object_detection",
    "train_oriented_object_detection",
    "train_panoptic_segmentation",
    "train_semantic_segmentation",
    "train_depth_estimation",
    "train_pose_estimation",
]


def load_embedding_model(path: str | Path, device: str | None = None) -> Any:
    """Load an exported embedding model (``exported_models/*.pt``).

    Rebuilds the ``FeatureExtractor`` from the exported metadata. The backbone
    must be resolvable via ``model_name`` recorded in the export — the
    export stores the state dict and normalization args; the backbone name
    is part of the run config, so this loader accepts it explicitly.
    """
    from autovision.encoders import FeatureExtractor, get_wrapped_model
    from autovision.encoders._model_helpers import load_state_dict_compat

    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or "kind" not in raw:
        raise ValueError(f"File '{path}' is not an exported AutoVision model.")
    model_name = raw.get("model_name")
    if model_name is None:
        raise ValueError(
            f"Export '{path}' does not record 'model_name'; pass the model "
            f"specification to rebuild the backbone."
        )
    wrapper = get_wrapped_model(model_name)
    embedding_model = FeatureExtractor(
        wrapped_model=wrapper, embed_dim=raw.get("embed_dim")
    )
    load_state_dict_compat(embedding_model, raw["state_dict"])
    if device is not None:
        embedding_model.to(device)
    embedding_model.eval()
    return embedding_model
