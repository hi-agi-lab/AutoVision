# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Experiment loggers.

:func:`get_loggers` builds a logger set from a spec dict. JSONL is always
available (framework-owned); TensorBoard/MLflow/W&B are lazy extras that
degrade gracefully when their packages are missing.

All loggers share the minimal interface::

    log_metrics(metrics: dict, step: int) -> None
    log_hyperparams(params: dict) -> None
    finalize() -> None
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from autovision.observability.loggers.jsonl import JSONLLogger
from autovision.runtime.distributed import is_global_rank_zero

__all__ = [
    "JSONLLogger",
    "MLflowLogger",
    "TensorBoardLogger",
    "WandbLogger",
    "get_loggers",
]

logger = logging.getLogger("autovision")


class TensorBoardLogger:
    """Thin wrapper over ``torch.utils.tensorboard.SummaryWriter``."""

    def __init__(self, save_dir: Path | str, name: str = "tensorboard") -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise ImportError(
                "TensorBoard logging requires the 'tensorboard' package: "
                "pip install 'autovision[tensorboard]'"
            ) from error
        self.log_dir = Path(save_dir) / name
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        for key, value in metrics.items():
            self._writer.add_scalar(key, float(value), global_step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        text = "\n".join(f"{key}: {value}" for key, value in params.items())
        self._writer.add_text("hyperparameters", text.replace("\n", "  \n"))

    def finalize(self) -> None:
        self._writer.flush()
        self._writer.close()


class MLflowLogger:
    """Lazy MLflow run logger (extra: ``autovision[mlflow]``)."""

    def __init__(
        self, save_dir: Path | str, tracking_uri: str | None = None,
        experiment_name: str = "autovision", run_name: str | None = None,
    ) -> None:
        import mlflow

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        self.artifacts_dir = Path(save_dir)

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        self._mlflow.log_metrics(
            {key: float(value) for key, value in metrics.items()}, step=step
        )

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_artifact(self, path: Path | str) -> None:
        self._mlflow.log_artifact(str(path))

    def finalize(self) -> None:
        self._mlflow.end_run()


class WandbLogger:
    """Lazy Weights & Biases logger (extra: ``autovision[wandb]``)."""

    def __init__(
        self, save_dir: Path | str, project: str | None = None,
        name: str | None = None, offline: bool = False,
    ) -> None:
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=project or "autovision",
            name=name,
            dir=str(save_dir),
            mode="offline" if offline else None,
        )

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        self._wandb.config.update(params)

    def finalize(self) -> None:
        self._wandb.finish()


def get_loggers(
    spec: dict[str, dict[str, Any] | None] | None, out_dir: Path | str
) -> list[Any]:
    """Build loggers from a spec dict.

    Defaults: JSONL always; TensorBoard when its package is importable;
    MLflow/W&B only when explicitly enabled (non-None spec entry). Missing
    optional packages log a warning instead of raising (except when the
    entry was explicitly requested).
    """
    out_dir = Path(out_dir)
    if not is_global_rank_zero():
        # Constructing MLflow/W&B/TensorBoard loggers has side effects
        # (start_run / init / SummaryWriter); DDP must not create one run
        # per rank. Non-zero ranks train without experiment tracking.
        return []
    spec = dict(spec or {})
    jsonl_spec = spec.pop("jsonl", None)
    loggers: list[Any] = [JSONLLogger(out_dir, **(jsonl_spec or {}))]

    tensorboard_spec = spec.pop("tensorboard", None)
    if tensorboard_spec is not False:  # False explicitly disables
        try:
            loggers.append(
                TensorBoardLogger(out_dir, **(tensorboard_spec or {}))
            )
        except ImportError:
            if tensorboard_spec is not None:
                raise  # explicitly requested but missing
            logger.info("tensorboard not installed; TensorBoard logging skipped.")

    for name, args, cls in (
        ("mlflow", spec.pop("mlflow", None), MLflowLogger),
        ("wandb", spec.pop("wandb", None), WandbLogger),
    ):
        if args is None:
            continue
        loggers.append(cls(out_dir, **(args or {})))

    if spec:
        unknown = ", ".join(sorted(spec))
        raise ValueError(f"Unknown loggers: {unknown}.")
    return loggers


def log_metrics_to_loggers(loggers: list[Any], metrics: dict[str, Any], step: int) -> None:
    if not is_global_rank_zero():
        return
    for lg in loggers:
        lg.log_metrics(metrics, step)


def log_hyperparams_to_loggers(loggers: list[Any], params: dict[str, Any]) -> None:
    if not is_global_rank_zero():
        return
    for lg in loggers:
        lg.log_hyperparams(params)


def finalize_loggers(loggers: list[Any]) -> None:
    for lg in loggers:
        finalize = getattr(lg, "finalize", None)
        if callable(finalize):
            try:
                finalize()
            except Exception:  # noqa: BLE001 - one backend must not block others
                logger.warning(
                    "%s.finalize failed; continuing shutdown.",
                    type(lg).__name__, exc_info=True,
                )
