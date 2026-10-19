# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Hyperparameter search (roadmap B2): random / grid / ASHA with reports.

Entry points: :class:`SearchSpace` (+ :func:`load_space`) for the space
spec, :func:`run_search` for the driver (artifacts: ``trials.jsonl``,
``report.md``, ``best.json``) and :func:`task_trial` to run any
``autovision.train_*`` entry through the sampled overrides.
"""

from autovision.hpo.search import (
    HpoResult,
    TrialRecord,
    resolve_task_entry,
    run_search,
    task_trial,
)
from autovision.hpo.space import SearchSpace, load_space, split_overrides

__all__ = [
    "HpoResult",
    "SearchSpace",
    "TrialRecord",
    "load_space",
    "resolve_task_entry",
    "run_search",
    "split_overrides",
    "task_trial",
]
