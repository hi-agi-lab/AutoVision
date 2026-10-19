# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""load_model: rebuild a TaskNetwork from an exported model file."""

from __future__ import annotations

import importlib
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import torch

from autovision.errors import CheckpointError
from autovision.runtime.accelerator import npu_available
from autovision.runtime.env import Env  # noqa: F401  (cache dir contract)
from autovision.tasks.task_model import TaskNetwork
from autovision.types import PathLike

__all__ = ["load_task_model"]

logger = logging.getLogger("autovision")

# Class-path migration table: old exports keep loading after refactors.
_CLASS_PATH_MIGRATIONS: dict[str, str] = {}

# Module-path prefix migrations: exports/checkpoints embed the class path
# of the code that trained them; when a task package moves, old payloads
# keep loading by rewriting the module prefix (class names unchanged).
# Design doc 30 (task-first layout, 2026-10-15): flat packages regrouped
# under per-task directories; detection families drop the redundant
# ``_object_detection`` suffix.
_MODULE_PATH_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("autovision.tasks.dinov2_eomt_semantic_segmentation.",
     "autovision.tasks.semantic_segmentation.dinov2_eomt."),
    ("autovision.tasks.dinov3_eomt_semantic_segmentation.",
     "autovision.tasks.semantic_segmentation.dinov3_eomt."),
    ("autovision.tasks.linear_semantic_segmentation.",
     "autovision.tasks.semantic_segmentation.linear."),
    ("autovision.tasks.dinov2_eomt_instance_segmentation.",
     "autovision.tasks.instance_segmentation.dinov2_eomt."),
    ("autovision.tasks.dinov3_eomt_instance_segmentation.",
     "autovision.tasks.instance_segmentation.dinov3_eomt."),
    ("autovision.tasks.dinov2_eomt_panoptic_segmentation.",
     "autovision.tasks.panoptic_segmentation.dinov2_eomt."),
    ("autovision.tasks.dinov3_eomt_panoptic_segmentation.",
     "autovision.tasks.panoptic_segmentation.dinov3_eomt."),
    ("autovision.tasks.image_classification.",
     "autovision.tasks.classification."),
    ("autovision.tasks.detection.rtdetr_oriented_object_detection.",
     "autovision.tasks.oriented_detection.rtdetr."),
    ("autovision.tasks.detection.rtdetr_object_detection.",
     "autovision.tasks.detection.rtdetr."),
    ("autovision.tasks.detection.avdetr.",
     "autovision.tasks.detection.detr."),
    ("autovision.tasks.eomt.",
     "autovision.tasks.shared.eomt."),
    ("autovision.optimizer_type.",
     "autovision.optim.optimizer_type."),
    ("autovision.tasks.detection.dfine_object_detection.",
     "autovision.tasks.detection.dfine."),
    ("autovision.tasks.detection.rfdetr_object_detection.",
     "autovision.tasks.detection.rfdetr."),
    ("autovision.tasks.detection.picodet_object_detection.",
     "autovision.tasks.detection.picodet."),
)


# Class-name generations: the AV-DETR -> DETR family rename (2026-10-15)
# also renamed the public classes; the module-prefix rule above fixes the
# module, these exact entries fix the class segment. (Old exports embed
# the full old class path.)
for _module, _names in {
    "task_model": ["DETRObjectDetection"],
    "train_model": ["DETRObjectDetectionTrain"],
    "config": [
        "DETRObjectDetectionTrainArgs", "DETRDensityArgs", "DETRSSLArgs",
        "DETRTokenDropArgs", "DETRDistillArgs", "DETROpenVocabArgs",
    ],
    "variants": ["DETRVariantSpec"],
}.items():
    for _name in _names:
        _old = f"autovision.tasks.detection.avdetr.{_module}.AV{_name}"
        _new = f"autovision.tasks.detection.detr.{_module}.{_name}"
        _CLASS_PATH_MIGRATIONS.setdefault(_old, _new)
del _module, _names, _name, _old, _new


#: Family-key generations: pre-rename exports embedded the old family key
#: in ``model_init_args``. The public aliases completed their deprecation
#: window (``resolve_detr_variant`` no longer accepts them), so artifacts
#: are rewritten here instead (design doc 28 §4.3).
_FAMILY_KEY_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("avdetr/", "detr/"),
    ("ltdetr/", "detr/"),
)


def _migrate_family_key(model_name: str) -> str:
    """Rewrite a payload's family key onto the current ``detr/`` prefix."""
    for old_prefix, new_prefix in _FAMILY_KEY_MIGRATIONS:
        if model_name.startswith(old_prefix):
            return new_prefix + model_name[len(old_prefix) :]
    return model_name


def _migrate_class_path(class_path: str) -> str:
    """Rewrite a checkpoint's class path onto the current tree."""
    class_path = _CLASS_MIGRATIONS.get(class_path, class_path)
    for old_prefix, new_prefix in _MODULE_PATH_MIGRATIONS:
        if class_path.startswith(old_prefix):
            return new_prefix + class_path[len(old_prefix):]
    return class_path

# Allowlist for dynamically imported model classes: exported files carry
# ``model_class_path`` and are untrusted input, so only concrete TaskNetwork
# classes inside this package may ever be reconstructed (a tampered file
# must not be able to instantiate arbitrary callables).
_TASK_MODEL_PACKAGE_PREFIX = "autovision.tasks."

# Exports legitimately embed autovision class references (the task-model
# class itself rides along in ``model_init_args``), which the default
# weights_only unpickler refuses. Blocked globals are extracted from the
# loader error and only admitted after verifying they really resolve to
# ``autovision.*`` objects — anything else (os.system, builtins, ...) stays
# rejected, so the unpickler never constructs foreign objects.
_BLOCKED_GLOBAL_RE = re.compile(r"GLOBAL ([\w.]+) was not an allowed global")
_MAX_ALLOWLIST_UPGRADES = 8


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if npu_available():
        return torch.device("npu")
    return torch.device("cpu")


def _verify_autovision_global(qualname: str) -> Any | None:
    """Return the object for ``qualname`` if it is an autovision class.

    Classes only: the weights_only unpickler *calls* any user-allowed
    callable it meets via pickle REDUCE (attacker-controlled arguments),
    while classes go through NEWOBJ and never run ``__init__``. Admitting
    autovision functions would hand a crafted file a gadget set of every
    module-level helper in the package.
    """
    if not qualname.startswith("autovision."):
        return None
    qualname = _migrate_class_path(qualname)
    module_path, _, attr = qualname.rpartition(".")
    if not module_path.startswith("autovision."):
        return None
    try:
        module = importlib.import_module(module_path)
        obj = getattr(module, attr)
    except (ImportError, AttributeError):
        return None
    if not str(getattr(obj, "__module__", "")).startswith("autovision."):
        return None
    if not isinstance(obj, type):
        return None
    return obj


def _safe_load_export(path: Path) -> Any:
    """weights_only load that self-upgrades its allowlist with verified
    autovision-internal globals the export legitimately embedded."""
    allowed: list[Any] = []
    for attempt in range(_MAX_ALLOWLIST_UPGRADES + 1):
        try:
            with torch.serialization.safe_globals(allowed):
                return torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as error:
            if attempt == _MAX_ALLOWLIST_UPGRADES:
                break
            match = _BLOCKED_GLOBAL_RE.search(str(error))
            if match is None:
                raise
            obj = _verify_autovision_global(match.group(1))
            if obj is None or any(obj is seen for seen in allowed):
                raise
            allowed.append(obj)
    raise CheckpointError(
        f"Exported model '{path}' references too many non-default "
        f"globals; refusing to extend the safe-loader allowlist further."
    )


def load_task_model(
    model: PathLike, device: str | torch.device | None = None
) -> Any:
    """Load an exported task model (``exported_models/*.pt``).

    The exported file carries ``model_class_path`` + ``model_init_args`` +
    ``train_model`` (EMA-prefixed state dict); the class is imported
    dynamically and the state dict loaded through the model's loader
    (EMA prefix stripped by the TaskNetwork).

    Security contract: exported files are untrusted input. They are
    deserialized with ``torch.load(weights_only=True)`` (no pickle object
    execution; autovision-internal class references are admitted only
    after verification) and the embedded class path may only resolve to a
    concrete :class:`TaskNetwork` subclass inside
    ``autovision.tasks``.
    """
    device = _resolve_device(device)
    path = Path(model)
    if not path.exists():
        raise CheckpointError(f"Exported model does not exist: '{path}'")
    try:
        checkpoint = _safe_load_export(path)
    except CheckpointError:
        raise
    except Exception as error:
        raise CheckpointError(
            f"Failed to load exported model '{path}': {error}. Exported "
            f"models are only ever read with the pickle-free safe loader "
            f"(weights_only=True); if this file came from another "
            f"framework or an unsafe writer, re-export it from a trusted "
            f"training run instead of disabling the loader guard."
        ) from error
    if not isinstance(checkpoint, dict) or "model_class_path" not in checkpoint:
        raise CheckpointError(
            f"File '{path}' is not an exported AutoVision task model."
        )
    class_path: str = _migrate_class_path(checkpoint["model_class_path"])
    if not isinstance(class_path, str) or not class_path.startswith(
        _TASK_MODEL_PACKAGE_PREFIX
    ):
        raise CheckpointError(
            f"Refusing to rebuild '{class_path!r}' from exported model "
            f"'{path}': only autovision task-model classes "
            f"('{_TASK_MODEL_PACKAGE_PREFIX}.*') may be imported."
        )
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
    except (ImportError, AttributeError) as error:
        raise CheckpointError(
            f"Cannot import '{class_path}' from the installed autovision "
            f"version: {error}"
        ) from error
    if not (
        isinstance(model_class, type)
        and issubclass(model_class, TaskNetwork)
        and model_class is not TaskNetwork
    ):
        raise CheckpointError(
            f"Refusing to instantiate '{class_path}' from exported model "
            f"'{path}': not a concrete autovision TaskNetwork subclass."
        )

    init_args = dict(checkpoint.get("model_init_args", {}))
    init_args.pop("load_weights", None)
    if isinstance(init_args.get("model_name"), str):
        init_args["model_name"] = _migrate_family_key(init_args["model_name"])
    model = model_class(**init_args)
    normalize_args = checkpoint.get("normalize_args")
    if normalize_args:
        # The exported checkpoint records the training-time mean/std; the
        # constructor only installs ImageNet defaults. Restoring them is
        # required for the exported model to reproduce training predictions
        # (and the ONNX sidecar's normalize contract).
        from autovision.augment.transform_args import NormalizeArgs

        try:
            model.normalize = NormalizeArgs.from_dict(normalize_args)
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointError(
                f"Exported model '{path}' has invalid normalize_args: {error}"
            ) from error
    state_dict = checkpoint["train_model"]
    if hasattr(model, "load_train_state_dict"):
        model.load_train_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


_CLASS_MIGRATIONS = _CLASS_PATH_MIGRATIONS
