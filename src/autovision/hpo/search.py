# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Search drivers and reports (roadmap B2).

``run_search`` drives a user-supplied trial function over a
:class:`~autovision.hpo.space.SearchSpace` with one of three strategies:

- ``random``: ``trials`` seeded samples (the gate: one run → one report);
- ``grid``: the full cartesian product of discrete axes;
- ``asha``: asynchronous successive halving — every configuration starts at
  ``min_budget`` steps, the top ``1/eta`` fraction is promoted to ``eta``
  times the budget, and so on to ``max_budget`` (Li et al., "A System for
  Massively Parallel Hyperparameter Tuning"; native implementation, no Ray
  dependency — a Ray Tune ``ASHAScheduler`` can drive the same ``trial_fn``
  if you prefer that stack).

Artifacts land in ``out_dir``: ``trials.jsonl`` (every trial: params,
budget, metrics, status, per-trial run dir), ``report.md`` (space summary,
trial table sorted by the objective, the best configuration) and
``best.json``. A trial is REUSED only when its run dir holds both the
metrics file and a provenance sidecar (``<metrics name>.meta.json``) whose
sha256 fingerprint over ``params + steps`` matches the current trial, so a
crashed or interrupted search resumes where it stopped while a changed
space/budget (or a legacy cache without a sidecar) re-runs instead of
silently reporting stale metrics.

Trial function contract::

    def trial_fn(params: Mapping[str, Any], steps: int, run_dir: Path) -> Mapping[str, float]

``params`` carries the RAW sampled point (dotted keys); use
:func:`~autovision.hpo.space.split_overrides` to turn it into training
keyword arguments. :func:`task_trial` does exactly that for the supported
training entries (the same override surface as the CLI, so a mistyped key
fails loudly through the Pydantic validators).

Trial metrics include ``steps_run``, which is the number of steps executed
IN THIS INVOCATION (a resumed rung runs only the increment towards its
total ``steps``), matching the engine's ``TaskTrainResult`` semantics.

Reserved keys: a trial owns ``out`` and ``steps`` (the search sets both),
and ``task_trial`` may add ``resume_interrupted`` — a space axis with any
of those names is rejected at run time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from autovision.errors import AutoVisionError
from autovision.hpo.space import SearchSpace, split_overrides

logger = logging.getLogger("autovision")

__all__ = [
    "HpoResult",
    "TrialRecord",
    "run_search",
    "task_trial",
]

TrialFn = Callable[[Mapping[str, Any], int, Path], Mapping[str, float]]


@dataclass
class TrialRecord:
    """One search trial: the sampled point, its budget and outcome."""

    index: int
    params: dict[str, Any]
    steps: int
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending | ok | failed | reused
    error: str = ""
    run_dir: str = ""
    rung: int = 0
    seconds: float = 0.0
    #: True when the rung produced no validation read and the objective was
    #: carried forward from an earlier rung (the best checkpoint is unchanged).
    carried: bool = False

    def objective(self, key: str) -> float | None:
        value = self.metrics.get(key)
        return float(value) if value is not None else None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HpoResult:
    """The search's final state (also fully reflected in the artifacts)."""

    out_dir: Path
    objective: str
    maximize: bool
    strategy: str
    trials: list[TrialRecord]
    report_md: Path
    trials_jsonl: Path
    best_json: Path
    space: str

    @property
    def best(self) -> TrialRecord | None:
        ranked = [
            trial
            for trial in self.trials
            if trial.status in ("ok", "reused") and trial.objective(self.objective) is not None
        ]
        if not ranked:
            return None
        return max(
            ranked,
            key=lambda trial: (
                trial.objective(self.objective)
                if self.maximize
                else -trial.objective(self.objective)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        best = self.best
        return {
            "out_dir": str(self.out_dir),
            "objective": self.objective,
            "maximize": self.maximize,
            "strategy": self.strategy,
            "space": self.space,
            "trials": [trial.to_json() for trial in self.trials],
            "best": best.to_json() if best else None,
            "report_md": str(self.report_md),
        }


def _write_report(
    out_dir: Path,
    space: SearchSpace,
    records: list[TrialRecord],
    *,
    objective: str,
    maximize: bool,
    strategy: str,
) -> Path:
    ranked = [trial for trial in records if trial.status in ("ok", "reused")]
    with_objective = [trial for trial in ranked if trial.objective(objective) is not None]
    ordered = sorted(
        with_objective,
        key=lambda trial: trial.objective(objective),  # type: ignore[arg-type]
        reverse=maximize,
    )
    lines: list[str] = ["# 超参搜索结果\n"]
    lines.append(
        f"- 策略 {strategy}；目标 `{objective}`（{'最大化' if maximize else '最小化'}）；"
        f"试验 {len(records)} 次（成功 {len(ranked)}，失败了 "
        f"{sum(1 for trial in records if trial.status == 'failed')}）\n"
    )
    lines.append("## 搜索空间\n")
    lines.append("| 轴 | 分布 |")
    lines.append("|---|---|")
    for axis in space.axes:
        lines.append(f"| `{axis.key}` | `{axis.describe()}` |")
    lines.append("")
    lines.append("## 最优配置\n")
    if ordered:
        best = ordered[0]
        lines.append(
            f"- 试验 #{best.index}（budget {best.steps}，`{objective}` = "
            f"{best.objective(objective):.6g}）\n"
        )
        lines.append("```json")
        lines.append(json.dumps(best.params, indent=2, sort_keys=True))
        lines.append("```\n")
        lines.append(f"- 运行目录：`{best.run_dir}`\n")
    else:
        lines.append("_（没有成功且有目标读数的试验）_\n")
    lines.append("## 试验明细（按目标排序，失败试验列在末尾）\n")
    lines.append("| # | 状态 | budget | rung | 目标 | 用时 s | 参数 |")
    lines.append("|---|---|---|---|---|---|---|")
    carried_marker = "†"
    any_carried = any(trial.carried for trial in records)
    for trial in ordered:
        params = ", ".join(f"{key}={value!r}" for key, value in sorted(trial.params.items()))
        marker = carried_marker if trial.carried else ""
        lines.append(
            f"| {trial.index} | {trial.status} | {trial.steps} | {trial.rung} "
            f"| {trial.objective(objective):.6g}{marker} | {trial.seconds:.1f} | `{params}` |"
        )
    for trial in records:
        if trial.status == "failed":
            params = ", ".join(f"{key}={value!r}" for key, value in sorted(trial.params.items()))
            detail = trial.error.splitlines()[-1] if trial.error else "unknown error"
            lines.append(
                f"| {trial.index} | failed | {trial.steps} | {trial.rung} | — | "
                f"{trial.seconds:.1f} | `{params}` — {detail} |"
            )
    lines.append("")
    if any_carried:
        lines.append(
            "> † 该 rung 的验证未见改进（引擎只在改进时记录 best），目标取该配置上一\n"
            "> 个 rung 的读数（best checkpoint 未变，正是 ASHA 需要的输入）。\n"
        )
    lines.append(
        "> `trials.jsonl` 为逐试验机器可读记录（含完整指标）；重跑同一次搜索会复用已完成的\n"
        "> 试验目录（仅当 `metrics.json` 旁的 `<name>.meta.json` 指纹与当前参数/budget 一致\n"
        "> 时才跳过；参数或 budget 变化会重跑），失败试验可删目录后重跑。\n"
    )
    report = out_dir / "report.md"
    report.write_text("\n".join(lines))
    return report


_METADATA_SCHEMA_VERSION = 1
_METADATA_SUFFIX = ".meta.json"


def _metadata_path(metrics_path: Path) -> Path:
    """The provenance sidecar that sits next to ``metrics_path``."""
    return metrics_path.with_name(metrics_path.name + _METADATA_SUFFIX)


def _trial_fingerprint(params: Mapping[str, Any], steps: int) -> str:
    """sha256 over canonical JSON of the trial's params and budget.

    ``sort_keys`` + compact separators make the digest independent of dict
    insertion order. ``default=repr`` keeps data-driven choice values that
    JSON cannot encode (arbitrary YAML scalars) fingerprintable instead of
    crashing the search.
    """
    canonical = json.dumps(
        {"params": dict(params), "steps": int(steps)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=repr,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _metadata_payload(record: TrialRecord) -> dict[str, Any]:
    """The sidecar body: what the metrics file is valid for."""
    return {
        "schema_version": _METADATA_SCHEMA_VERSION,
        "params": dict(record.params),
        "steps": int(record.steps),
        "fingerprint": _trial_fingerprint(record.params, record.steps),
        "trial_index": int(record.index),
        "rung": int(record.rung),
        "run_dir": record.run_dir,
    }


def _write_metadata_atomic(meta_path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` through a sibling temp file + ``os.replace``.

    A crash mid-write leaves at most a dot-prefixed temp file, never a
    partial sidecar that could pass as valid.
    """
    tmp_path = meta_path.with_name(
        f".{meta_path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    try:
        tmp_path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, default=repr),
            encoding="utf-8",
        )
        os.replace(tmp_path, meta_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _reusable_metrics(metrics_path: Path, record: TrialRecord) -> dict[str, float] | None:
    """Cached metrics for ``record``, or None when provenance does not match.

    Reuse requires BOTH the metrics file and a parseable sidecar whose
    fingerprint equals this trial's params + budget. A missing sidecar means
    a legacy directory (predating provenance tracking) or a trial that
    failed after writing metrics; neither is trusted.
    """
    meta_path = _metadata_path(metrics_path)
    if not meta_path.exists():
        logger.warning(
            "Not reusing '%s': provenance sidecar '%s' is missing (legacy or "
            "unfinished trial); re-running trial #%d.",
            metrics_path, meta_path.name, record.index,
        )
        return None
    try:
        metadata = json.loads(meta_path.read_text())
        if not isinstance(metadata, dict):
            raise TypeError(
                f"metadata payload is {type(metadata).__name__}, not a mapping"
            )
        schema = metadata.get("schema_version")
        if schema != _METADATA_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported metadata schema {schema!r} "
                f"(expected {_METADATA_SCHEMA_VERSION})"
            )
        expected = _trial_fingerprint(record.params, record.steps)
        if metadata.get("fingerprint") != expected:
            raise ValueError(
                "fingerprint mismatch: recorded params/budget differ from the "
                "current trial"
            )
        payload = json.loads(metrics_path.read_text())
        if not isinstance(payload, dict):
            raise TypeError(
                f"metrics payload is {type(payload).__name__}, not a mapping"
            )
        return {key: float(value) for key, value in payload.items()}
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "Not reusing '%s' (%s: %s); re-running trial #%d.",
            metrics_path, type(exc).__name__, exc, record.index,
        )
        return None


def run_trial(
    trial_fn: TrialFn,
    record: TrialRecord,
    out_dir: Path,
    *,
    force: bool = False,
    metrics_name: str = "metrics.json",
) -> TrialRecord:
    """Run (or reuse) one trial; a failure is recorded, never raised.

    ``metrics_name`` keys the reuse check (ASHA uses per-rung files so a
    promoted rung is not mistaken for a finished one). Reuse additionally
    requires a matching provenance sidecar (``<metrics_name>.meta.json``)
    carrying the params/budget fingerprint; without it the trial runs again
    and overwrites the metrics on success.
    """
    run_dir = out_dir / f"trial_{record.index:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / metrics_name
    meta_path = _metadata_path(metrics_path)
    record.run_dir = str(run_dir)
    if metrics_path.exists() and not force:
        cached = _reusable_metrics(metrics_path, record)
        if cached is not None:
            record.metrics = cached
            record.status = "reused"
            record.seconds = 0.0
            return record
    # Invalidate provenance BEFORE the re-run: should the trial fail below,
    # no sidecar may survive that would let a later run reuse the stale
    # metrics file. The metrics themselves are kept (not deleted) and are
    # overwritten only after a successful run.
    meta_path.unlink(missing_ok=True)
    started = time.monotonic()
    try:
        metrics = dict(trial_fn(dict(record.params), record.steps, run_dir))
        record.metrics = {key: float(value) for key, value in metrics.items()}
        record.status = "ok"
        metrics_path.write_text(json.dumps(record.metrics, indent=2))
        _write_metadata_atomic(meta_path, _metadata_payload(record))
    except Exception as exc:  # noqa: BLE001 - a failed trial must not kill the search
        record.status = "failed"
        record.error = f"{type(exc).__name__}: {exc}"
        meta_path.unlink(missing_ok=True)
    finally:
        record.seconds = time.monotonic() - started
    return record


def _finalize(
    out_dir: Path,
    space: SearchSpace,
    records: list[TrialRecord],
    *,
    objective: str,
    maximize: bool,
    strategy: str,
    trials_jsonl: Path,
) -> HpoResult:
    report = _write_report(
        out_dir, space, records, objective=objective, maximize=maximize, strategy=strategy
    )
    result = HpoResult(
        out_dir=out_dir,
        objective=objective,
        maximize=maximize,
        strategy=strategy,
        trials=records,
        report_md=report,
        trials_jsonl=trials_jsonl,
        best_json=out_dir / "best.json",
        space=repr(space),
    )
    best = result.best
    result.best_json.write_text(json.dumps(best.to_json() if best else {}, indent=2))
    return result


def run_search(
    trial_fn: TrialFn,
    space: SearchSpace,
    out_dir: "str | Path",
    *,
    strategy: str = "random",
    trials: int = 10,
    budget: int = 1000,
    objective: str = "val_metric/map",
    maximize: bool = True,
    seed: int = 0,
    asha_min_budget: int | None = None,
    asha_max_budget: int | None = None,
    asha_eta: int = 3,
    force: bool = False,
) -> HpoResult:
    """Run one search; returns the result and writes the artifacts.

    ``budget`` is the number of training steps per trial. For
    ``strategy="asha"`` the initial budget is ``asha_min_budget`` (default
    ``budget``) and promoted rungs multiply it by ``asha_eta`` up to
    ``asha_max_budget`` (default ``budget``); classic ASHA is therefore
    ``budget=<min>`` plus ``asha_max_budget=<max>``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if trials < 1:
        raise AutoVisionError("trials must be >= 1.")
    if strategy == "asha":
        from autovision.hpo.asha import run_asha

        return run_asha(
            trial_fn,
            space,
            out_dir,
            trials=trials,
            objective=objective,
            maximize=maximize,
            seed=seed,
            min_budget=int(asha_min_budget or budget),
            max_budget=int(asha_max_budget or budget),
            eta=asha_eta,
            force=force,
        )
    if strategy == "random":
        records = [
            TrialRecord(index=index, params=params, steps=int(budget))
            for index, params in enumerate(space.sample_many(trials, seed=seed))
        ]
    elif strategy == "grid":
        points = space.grid()
        if trials < len(points):
            raise AutoVisionError(
                f"strategy='grid' enumerates {len(points)} points; raise trials "
                f"(>= {len(points)}) or restrict the space."
            )
        records = [
            TrialRecord(index=index, params=params, steps=int(budget))
            for index, params in enumerate(points[:trials])
        ]
    else:
        raise AutoVisionError(
            f"Unknown strategy '{strategy}'; supported: random, grid, asha."
        )

    jsonl = out_dir / "trials.jsonl"
    with jsonl.open("w") as handle:
        for record in records:
            run_trial(trial_fn, record, out_dir, force=force)
            handle.write(json.dumps(record.to_json()) + "\n")
            handle.flush()
    return _finalize(
        out_dir,
        space,
        records,
        objective=objective,
        maximize=maximize,
        strategy=strategy,
        trials_jsonl=jsonl,
    )


#: Keys the engine's per-step JSONL adds around the metric values.
_METRICS_JSONL_BOOKKEEPING = ("ts", "step", "hyperparams")


def _last_validation_metrics(metrics_jsonl: Path) -> dict[str, float]:
    """The last validation read from an engine ``metrics.jsonl`` (or {})."""
    if not metrics_jsonl.is_file():
        return {}
    last: dict[str, float] = {}
    try:
        for line in metrics_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            values = {
                key: float(value)
                for key, value in payload.items()
                if key not in _METRICS_JSONL_BOOKKEEPING
                and isinstance(value, (int, float))
            }
            if values:
                last = values
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "Could not recover metrics from '%s' (%s: %s).",
            metrics_jsonl, type(exc).__name__, exc,
        )
        return {}
    return last


_RESERVED_KEYS = ("out", "steps", "checkpoint", "resume_interrupted")


def _task_entries() -> dict[str, Any]:
    """The training entries a trial may drive (imported lazily, statically).

    The concrete ``cli.train_task`` modules are imported HERE so the
    layering scanner sees the upward edge (hpo -> cli) that
    ``ALLOWED_UPWARD`` records, instead of an invisible ``import autovision``
    reaching the whole public API.
    """
    from autovision.cli.train_task.depth_estimation import train_depth_estimation
    from autovision.cli.train_task.image_classification import (
        train_image_classification,
    )
    from autovision.cli.train_task.instance_segmentation import (
        train_instance_segmentation,
    )
    from autovision.cli.train_task.object_detection import train_object_detection
    from autovision.cli.train_task.oriented_object_detection import (
        train_oriented_object_detection,
    )
    from autovision.cli.train_task.panoptic_segmentation import (
        train_panoptic_segmentation,
    )
    from autovision.cli.train_task.pose_estimation import train_pose_estimation
    from autovision.cli.train_task.semantic_segmentation import (
        train_semantic_segmentation,
    )

    return {
        "object_detection": train_object_detection,
        "oriented_object_detection": train_oriented_object_detection,
        "image_classification": train_image_classification,
        "instance_segmentation": train_instance_segmentation,
        "panoptic_segmentation": train_panoptic_segmentation,
        "semantic_segmentation": train_semantic_segmentation,
        "pose_estimation": train_pose_estimation,
        "depth_estimation": train_depth_estimation,
    }


def resolve_task_entry(task: str) -> Any:
    """The training entry for ``task``; fails loud with the supported names."""
    entries = _task_entries()
    entry = entries.get(task)
    if entry is None:
        raise AutoVisionError(
            f"Unknown HPO task '{task}'. Supported: {sorted(entries)}."
        )
    return entry


def task_trial(task: str, base: Mapping[str, Any] | None = None) -> TrialFn:
    """A trial function training ``task`` with the sampled overrides.

    ``task`` names a supported training entry (see
    :func:`resolve_task_entry`); ``base`` carries the fixed kwargs
    (``data``, ``model``, ``devices``, ``precision``, ...). Dotted
    ``model_args.*`` / ``transform_args.*`` / ``trainer_args.*`` keys merge
    into the corresponding dicts; bare keys override top-level kwargs. The
    trial returns ``best_metrics`` — pick an objective the family actually
    publishes (the engine's watch keys: detection/OBB ``val_metric/map``,
    pose ``val_metric/kp_map``, instance seg ``val_metric/mask_map``,
    semantic seg ``val_metric/miou``, panoptic ``val_metric/pq``,
    classification ``val_metric/accuracy_top1``, depth
    ``val_metric/delta1``).

    Reserved kwargs (``out``, ``steps``, ``checkpoint``,
    ``resume_interrupted``) are owned by the search: a sampled axis with one
    of those names is a hard error.
    """
    fixed = dict(base or {})
    entry = resolve_task_entry(task)  # fail at construction, not trial 1

    def trial_fn(params: Mapping[str, Any], steps: int, run_dir: Path) -> Mapping[str, float]:
        overrides = split_overrides(params)
        reserved = sorted(set(overrides) & set(_RESERVED_KEYS))
        if reserved:
            raise AutoVisionError(
                f"Search axis {reserved} collides with a reserved trial kwarg "
                f"({list(_RESERVED_KEYS)}); rename the axis."
            )
        kwargs = dict(fixed)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(kwargs.get(key), Mapping):
                kwargs[key] = {**dict(kwargs[key]), **value}
            else:
                kwargs[key] = value
        run_root = run_dir / "run"
        kwargs["out"] = run_root
        # A promoted ASHA rung (or a re-entered crashed trial) continues the
        # SAME run dir: ``resume_interrupted=True`` lets the engine pick up
        # the run's last checkpoint and train on to ``steps`` (the TOTAL
        # target) — a genuine longer read instead of a retrain. Do NOT pass
        # ``checkpoint=`` as well: the engine rejects the pair
        # ("'checkpoint' and 'resume_interrupted' cannot both be set"), and a
        # dir that crashed before its first checkpoint resumes cleanly (the
        # engine reports "No last.ckpt found" and starts fresh) instead of
        # tripping the non-empty-dir guard.
        if run_root.exists() and any(run_root.iterdir()):
            kwargs.setdefault("resume_interrupted", True)
        result = entry(steps=steps, **kwargs)
        metrics = dict(result.best_metrics)
        if not metrics:
            # The engine only fills ``best_metrics`` when THIS invocation
            # improved the best checkpoint; a resumed rung that did not
            # improve (or a re-entered trial) reports an empty dict. The
            # engine's own per-step log carries the validation read of this
            # run — recover it so the rung still has its objective at the
            # current budget.
            metrics = _last_validation_metrics(run_root / "metrics.jsonl")
        metrics["steps_run"] = float(result.steps_run)
        return metrics

    return trial_fn
