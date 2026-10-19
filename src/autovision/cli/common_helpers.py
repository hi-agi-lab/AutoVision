# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Shared command helpers: out dir, batch size, num workers."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import torch

from autovision.runtime.distributed import is_global_rank_zero
from autovision.runtime.env import Env
from autovision.runtime.manifest import write_run_manifest
from autovision.types import PathLike

__all__ = [
    "get_global_batch_size",
    "get_num_workers",
    "get_out_dir",
    "resolve_gradient_checkpointing",
    "resolve_num_devices",
    "resolve_parallel",
    "validate_trainer_args",
]

logger = logging.getLogger("autovision")


def resolve_num_devices(devices: int | str) -> int:
    """Process count for the batch-size math, from a ``devices`` spec.

    ``"auto"`` follows ``WORLD_SIZE`` (torchrun); an int (or digit string)
    is the process count; any other device spec (``"cpu"``, ``"cuda:1"``,
    ``"npu:0"``) is a SINGLE device — the spec itself is resolved by
    ``engine.resolve_device``, so it contributes one process here.
    """
    if devices == "auto":
        return int(os.getenv("WORLD_SIZE", "1"))
    if isinstance(devices, int):
        return devices
    text = str(devices)
    return int(text) if text.isdigit() else 1


def resolve_gradient_checkpointing(trainer_args: dict | None) -> bool:
    """Read the opt-in ``gradient_checkpointing`` trainer flag."""
    value = (trainer_args or {}).get("gradient_checkpointing", False)
    if not isinstance(value, bool):
        raise ValueError(
            "trainer_args.gradient_checkpointing must be a bool, got "
            f"{type(value).__name__}."
        )
    return value


def resolve_parallel(trainer_args: dict | None) -> str:
    """Read the ``parallel`` strategy (``ddp`` | ``fsdp``)."""
    value = str((trainer_args or {}).get("parallel", "ddp")).lower()
    if value not in ("ddp", "fsdp"):
        raise ValueError(
            "trainer_args.parallel must be 'ddp' or 'fsdp' (DeepSpeed/ZeRO/"
            f"elastic need an external launcher); got '{value}'."
        )
    return value


def validate_trainer_args(trainer_args: dict | None, *, allowed: set[str]) -> None:
    """Reject unknown ``trainer_args`` keys instead of silently ignoring them.

    The trainer entries read a small allowlist of keys; anything else was
    previously dropped without a word (e.g. ``ema_warmup_steps`` is a
    ``model_args`` field — passing it as a trainer arg silently left the
    2000-step default in place).
    """
    if not trainer_args:
        return
    unknown = sorted(set(trainer_args) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown trainer_args key(s) {unknown} — accepted: {sorted(allowed)}. "
            "Model-side settings (lr, ema_warmup_steps, criterion flags, ...) "
            "belong in model_args."
        )


def get_out_dir(
    out: PathLike, resume_interrupted: bool = False, overwrite: bool = False
) -> Path:
    """Resolve and create the output directory.

    A non-empty existing directory requires ``resume_interrupted`` or
    ``overwrite`` — protects previous runs from silent clobbering.
    ``resume_interrupted=True`` keeps the previous contents (resume may pick
    up ``last.ckpt``); ``overwrite=True`` replaces them, per the contract in
    the error message — stale artifacts (e.g. an old ``metrics.jsonl``)
    would otherwise silently mix into the new run.
    """
    out_dir = Path(out)
    if os.environ.get("WORLD_SIZE", "1") != "1" and not is_global_rank_zero():
        # Multi-process run (torchrun): rank 0 owns the directory lifecycle
        # (it performs the conflict check above/below and may rmtree on
        # overwrite). Later ranks may legitimately arrive after rank 0 has
        # already populated the shared output directory, so they must not
        # fail the non-empty check nor delete it.
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    if (
        out_dir.exists()
        and any(out_dir.iterdir())
        and not (resume_interrupted or overwrite)
    ):
        raise ValueError(
            f"Output directory '{out_dir}' exists and is not empty. Use "
            f"overwrite=True to replace it, or resume_interrupted=True to "
            f"continue a previous run."
        )
    if overwrite and not resume_interrupted and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_global_rank_zero():
        try:
            write_run_manifest(out_dir, resume=resume_interrupted)
        except Exception as error:  # noqa: BLE001 - audit artifact is best-effort
            logger.warning("Could not write run manifest to '%s': %s", out_dir, error)
    return out_dir


def get_global_batch_size(
    batch_size: int, dataset_size: int, num_devices: int
) -> int:
    """Resolve the global batch size (per-device = global / devices)."""
    if batch_size > dataset_size:
        logger.warning(
            "Global batch size (%d) is larger than the dataset size (%d); "
            "reducing it to the dataset size.",
            batch_size, dataset_size,
        )
        batch_size = dataset_size
    if batch_size % num_devices != 0:
        raise ValueError(
            f"Global batch size ({batch_size}) must be divisible by the "
            f"number of devices ({num_devices})."
        )
    return batch_size


def _is_slurm() -> bool:
    return "SLURM_JOB_ID" in os.environ


def get_num_workers(num_workers: int | str, num_devices_per_node: int) -> int:
    """Resolve dataloader workers per dataloader (one per device)."""
    if num_workers != "auto":
        return int(num_workers)
    if _is_slurm() and (cpus := os.getenv("SLURM_CPUS_PER_TASK")):
        num_workers = int(cpus) - 1
    else:
        cpus = os.cpu_count() or Env.DEFAULT_NUM_WORKERS_AUTO.value
        num_workers = max(1, cpus // num_devices_per_node) - 1
    num_workers = min(num_workers, Env.MAX_NUM_WORKERS_AUTO.value)
    return max(0, num_workers)


def resolve_data_paths(data: PathLike | Sequence[PathLike]) -> list[Path]:
    """Normalize ``data`` (dir or file list) to a list of paths."""
    if isinstance(data, (str, Path)):
        return [Path(data)]
    return [Path(item) for item in data]


def resolve_gradient_accumulation(
    gradient_accumulation_steps: int | str,
    nominal_batch_size: int,
    global_batch_size: int,
) -> int:
    """Resolve ``"auto"`` accumulation from a nominal batch size.

    ``"auto"`` → ``max(1, round(nominal / global_batch))`` (the
    ultralytics ``nbs`` convention): the effective batch lands near the
    nominal one regardless of the per-step batch, so recipes can hold a
    fixed nominal batch while memory decides the micro-batch. Explicit
    integers pass through unchanged. LR scaling is deliberately NOT
    touched here — it keeps referencing the per-step batch
    (``build_detr_optimizer``), matching the pre-existing semantics.
    """
    if isinstance(gradient_accumulation_steps, str):
        if gradient_accumulation_steps != "auto":
            raise ValueError(
                "gradient_accumulation_steps must be a positive int or "
                f"\"auto\", got {gradient_accumulation_steps!r}."
            )
        return max(1, round(nominal_batch_size / max(1, global_batch_size)))
    resolved = int(gradient_accumulation_steps)
    if resolved < 1:
        raise ValueError(
            f"gradient_accumulation_steps must be >= 1, got {resolved}."
        )
    return resolved


def build_task_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    train: bool,
    persistent_workers: bool | None = None,
    sampler=None,
):
    """Train/val task dataloaders (global batch already split per device).

    ``sampler`` (hard-example weighted sampling) replaces ``shuffle``: pass
    one only for the training split.

    ``persistent_workers`` defaults to ``num_workers > 0``; the CLIs whose
    dataset carries a step-scheduled transform (mosaic/mixup windows) pass
    ``False``: persistent workers survive iterator resets, so the engine's
    reinitialization handshake could never re-pickle the transform with the
    current step, and the windows would silently never open. The gate is
    deliberately fail-closed (always non-persistent for those loaders, at
    the cost of one worker respawn per epoch) rather than keyed off the
    configured window names — a renamed/re-added window arg must not be
    able to reintroduce the silent no-op. NOTE: a respawn also resets the
    transforms' per-worker donor caches, so mosaic/mixup donors never span
    epochs (a minor augmentation-diversity change, documented here).
    """
    import math

    from torch.utils.data import DataLoader

    from autovision.runtime.env import Env

    # Do not create more worker processes than the loader has batches to
    # produce: tiny datasets would otherwise pay a persistent worker pool
    # (and, on CUDA, hold GPU context in every idle worker) for a single
    # batch. Fewer than two batches run in-process.
    #
    # Measured alternatives that did NOT land here (2026-10-15, batch-level
    # isolation on this box's 12-thread CPU): torch's default
    # ``prefetch_factor=2`` beat 4 (23.1 vs 26.4 ms/batch — data is ~3% of
    # a detection step, so extra in-flight batches only add contention),
    # and capping worker intra-op threads at 1 (the ultralytics
    # OMP_NUM_THREADS=1 recipe) was slower still (28.2 ms/batch).
    num_batches = math.ceil(len(dataset) / batch_size) if batch_size else 0
    if num_workers > 0:
        num_workers = 0 if num_batches <= 1 else min(num_workers, num_batches)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        drop_last=train,
        num_workers=num_workers,
        timeout=Env.DATALOADER_TIMEOUT_SEC.value if num_workers > 0 else 0.0,
        pin_memory=torch.cuda.is_available(),
        # See the signature note: step-scheduled transforms pass
        # persistent_workers=False so every iterator reset re-pickles the
        # transform with the current step.
        persistent_workers=(
            num_workers > 0
            if persistent_workers is None
            else bool(persistent_workers) and num_workers > 0
        ),
    )
