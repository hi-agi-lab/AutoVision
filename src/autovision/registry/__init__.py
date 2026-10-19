# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Model asset registry (filesystem + MLflow backends) + downloadable cache."""

# Registers the Apache warm-start component assets on import (design doc 28
# section 6.3). Keep this import before the names below re-export the
# manifest so ``from autovision.registry import ASSET_MANIFEST`` observes it.
from autovision.registry import apache_assets  # noqa: F401
from autovision.registry.catalog import (
    MODEL_CATALOG,
    ModelEntry,
    describe_model,
    format_model_table,
    get_model_entry,
    list_task_models,
)
from autovision.registry.deployment_gates import (
    BenchmarkGate,
    DataSnapshotGate,
    DeploymentGate,
    LicenseGate,
    ModelCardGate,
    ParityGate,
    Sha256Gate,
    default_gates,
)
from autovision.registry.downloadable import (
    ASSET_MANIFEST,
    DownloadableModel,
    DownloadableModelError,
    register_downloadable,
    resolve_downloadable,
)
from autovision.registry.mlflow_registry import MLflowModelRegistry
from autovision.registry.model_registry import (
    MODEL_STAGES,
    ModelRegistry,
    RegistryEntry,
    RegistryError,
)

__all__ = [
    "ASSET_MANIFEST",
    "BenchmarkGate",
    "DataSnapshotGate",
    "DeploymentGate",
    "DownloadableModel",
    "DownloadableModelError",
    "LicenseGate",
    "MLflowModelRegistry",
    "MODEL_CATALOG",
    "MODEL_STAGES",
    "ModelCardGate",
    "ModelEntry",
    "ModelRegistry",
    "ParityGate",
    "RegistryEntry",
    "RegistryError",
    "Sha256Gate",
    "default_gates",
    "describe_model",
    "format_model_table",
    "get_model_entry",
    "list_task_models",
    "register_downloadable",
    "resolve_downloadable",
]
