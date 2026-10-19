# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Asynchronous successive halving (ASHA) driver (roadmap B2).

Native implementation of the ASHA rule (Li et al., 2020 —
"rung" = a budget level): every configuration starts at ``min_budget``;
after each rung the best ``1/eta`` fraction is promoted to the next rung,
whose budget is ``eta`` times larger, until ``max_budget``. Promoted
configurations CONTINUE training in place — the trial runs in the same run
directory and ``resume_interrupted=True`` lets the engine pick up where the
previous rung stopped (passing ``checkpoint=`` alongside is rejected by the
engine, so the hpo driver never sets it) — a rung's read is a genuine
longer-training read, not a retrain from scratch.

Per-rung metrics land in ``metrics.rung<r>.json`` inside the trial's run
dir (``metrics.json`` mirrors the latest rung). Every metrics file has a
``<name>.meta.json`` provenance sidecar whose fingerprint over params +
budget is the reuse key: an interrupted search re-enters at the rung it
lost, while a changed space, budget or eta re-runs the stale rung instead
of reusing it.

Ray Tune users: the same ``trial_fn`` can be handed to
``ray.tune`` with an ``ASHAScheduler`` if that stack is preferred — this
module exists so the algorithm works without the dependency.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from autovision.errors import AutoVisionError
from autovision.hpo.search import (
    HpoResult,
    TrialFn,
    TrialRecord,
    _finalize,
    _metadata_path,
    _metadata_payload,
    _write_metadata_atomic,
    run_trial,
)

__all__ = ["run_asha"]


def _ranked(records: list[TrialRecord], objective: str, maximize: bool) -> list[TrialRecord]:
    usable = [
        record
        for record in records
        if record.status in ("ok", "reused") and record.objective(objective) is not None
    ]
    return sorted(
        usable,
        key=lambda record: record.objective(objective),  # type: ignore[arg-type]
        reverse=maximize,
    )


def run_asha(
    trial_fn: TrialFn,
    space: Any,
    out_dir: Path,
    *,
    trials: int,
    objective: str,
    maximize: bool,
    seed: int,
    min_budget: int,
    max_budget: int,
    eta: int,
    force: bool = False,
) -> HpoResult:
    """Run successive halving from ``min_budget`` to ``max_budget``."""
    if eta < 2:
        raise AutoVisionError("asha_eta must be >= 2.")
    if min_budget < 1 or max_budget < min_budget:
        raise AutoVisionError(
            f"ASHA budgets must satisfy 1 <= min_budget <= max_budget, got "
            f"[{min_budget}, {max_budget}]."
        )
    jsonl = out_dir / "trials.jsonl"
    all_records: list[TrialRecord] = []
    # Configuration index is stable across rungs; the run dir is shared.
    configs = space.sample_many(trials, seed=seed)
    # Budget ladder: min, min*eta, ... with the LAST rung pinned to
    # max_budget (the promoted survivors train all the way to the budget).
    ladder = [int(min_budget)]
    while ladder[-1] * int(eta) < int(max_budget):
        ladder.append(ladder[-1] * int(eta))
    if ladder[-1] != int(max_budget):
        ladder.append(int(max_budget))

    surviving = list(range(len(configs)))
    carried: dict[int, float] = {}  # config index -> last measured objective
    with jsonl.open("w") as handle:
        for rung, budget in enumerate(ladder):
            if not surviving:
                break
            rung_records: list[TrialRecord] = []
            for index in surviving:
                record = TrialRecord(
                    index=index, params=configs[index], steps=budget, rung=rung
                )
                run_trial(
                    trial_fn,
                    record,
                    out_dir,
                    force=force,
                    metrics_name=f"metrics.rung{rung}.json",
                )
                if record.status in ("ok", "reused"):
                    # Only a completed rung mirrors into metrics.json (a
                    # failed rung's empty dict would poison later reuse).
                    # The mirror carries the same provenance sidecar as the
                    # rung file so a later run cannot mistake it for a match.
                    metrics_path = Path(record.run_dir) / "metrics.json"
                    metrics_path.write_text(json.dumps(record.metrics, indent=2))
                    _write_metadata_atomic(
                        _metadata_path(metrics_path), _metadata_payload(record)
                    )
                all_records.append(record)
                rung_records.append(record)
                handle.write(json.dumps(record.to_json()) + "\n")
                handle.flush()
            # A rung whose validation did not improve reports no objective
            # (the engine keeps the previous best); carrying the config's
            # last measured objective forward is the correct ASHA input —
            # the best checkpoint's metric is unchanged — and it is flagged
            # in the record/report instead of silently dropping the config.
            for record in rung_records:
                value = record.objective(objective)
                if value is None:
                    if record.index in carried:
                        record.metrics[objective] = carried[record.index]
                        record.carried = True
                else:
                    carried[record.index] = value
            if budget >= int(max_budget):
                break
            ranked = _ranked(rung_records, objective, maximize)
            if not ranked:
                break  # every trial in this rung failed; the report records why
            keep = max(1, math.ceil(len(ranked) / int(eta)))
            surviving = [record.index for record in ranked[:keep]]
    return _finalize(
        out_dir,
        space,
        all_records,
        objective=objective,
        maximize=maximize,
        strategy="asha",
        trials_jsonl=jsonl,
    )
