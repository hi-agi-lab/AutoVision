# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Reference prediction-drift monitor (PSI) for the serving app.

This is a *reference* monitor, not a full model-performance monitor: it watches
the served prediction distribution (class frequencies and score histogram)
against a baseline built from a reference run. It cannot detect accuracy drift
without labels — pair it with an offline labelled evaluation for that.

PSI (population stability index) is computed per bin:

    PSI = Σ (actual_pct - expected_pct) · ln(actual_pct / expected_pct)

with epsilon smoothing so unseen bins do not divide by zero. Conventional
thresholds: < 0.1 stable, 0.1–0.2 monitor, > 0.2 drift; the threshold is a
constructor argument because it is model/business specific.
"""

from __future__ import annotations

import json
import math
import threading
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from autovision.errors import AutoVisionError

__all__ = [
    "DriftBaseline",
    "DriftMonitor",
    "build_drift_baseline",
    "load_drift_baseline",
]

_SCHEMA_VERSION = 1
_DEFAULT_BINS = 10
_EPS = 1e-6


@dataclass(frozen=True)
class DriftBaseline:
    """Reference distribution for label counts and score bins."""

    sample_count: int
    label_counts: dict[int, int]
    score_histogram: list[int]
    bins: int = _DEFAULT_BINS
    schema_version: int = _SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bins": self.bins,
            "sample_count": self.sample_count,
            "label_counts": {
                str(label): count for label, count in sorted(self.label_counts.items())
            },
            "score_histogram": list(self.score_histogram),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DriftBaseline":
        if not isinstance(payload, dict):
            raise AutoVisionError("Drift baseline must be a JSON object.")
        version = int(payload.get("schema_version", 0))
        if version != _SCHEMA_VERSION:
            raise AutoVisionError(
                f"Unsupported drift baseline schema_version {version}; "
                f"expected {_SCHEMA_VERSION}."
            )
        bins = int(payload.get("bins", _DEFAULT_BINS))
        histogram = [int(value) for value in payload.get("score_histogram", [])]
        if bins < 1 or len(histogram) != bins:
            raise AutoVisionError(
                "Drift baseline score_histogram must have exactly 'bins' entries."
            )
        sample_count = int(payload.get("sample_count", 0))
        if sample_count < 1:
            raise AutoVisionError("Drift baseline sample_count must be >= 1.")
        label_counts = {
            int(label): int(count)
            for label, count in dict(payload.get("label_counts", {})).items()
        }
        if not label_counts or any(count < 0 for count in label_counts.values()):
            raise AutoVisionError("Drift baseline has no positive label_counts.")
        if sum(label_counts.values()) <= 0:
            raise AutoVisionError("Drift baseline has no positive label_counts.")
        if any(value < 0 for value in histogram):
            raise AutoVisionError(
                "Drift baseline score_histogram counts must be non-negative."
            )
        return cls(
            sample_count=sample_count,
            label_counts=label_counts,
            score_histogram=histogram,
            bins=bins,
            schema_version=version,
        )


def _record_stats(
    record: dict[str, Any], bins: int
) -> tuple[dict[int, int], list[int], int] | None:
    labels = record.get("labels")
    scores = record.get("scores")
    if labels is None and scores is None:
        return None
    label_counts: Counter[int] = Counter()
    if labels is not None:
        for label in labels:
            try:
                label_counts[int(label)] += 1
            except (TypeError, ValueError):
                continue
    histogram = [0] * bins
    score_total = 0
    if scores is not None:
        for score in scores:
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            index = min(bins - 1, max(0, int(value * bins)))
            histogram[index] += 1
            score_total += 1
    if not label_counts and score_total == 0:
        return None
    return dict(label_counts), histogram, score_total


def build_drift_baseline(
    records: Iterable[dict[str, Any]], *, bins: int = _DEFAULT_BINS
) -> DriftBaseline:
    """Aggregate reference prediction records into a baseline."""
    if bins < 1:
        raise AutoVisionError(f"bins must be >= 1, got {bins}.")
    label_counts: Counter[int] = Counter()
    histogram = [0] * bins
    sample_count = 0
    for record in records:
        stats = _record_stats(record, bins)
        if stats is None:
            continue
        labels, scores, _ = stats
        label_counts.update(labels)
        histogram = [a + b for a, b in zip(histogram, scores)]
        sample_count += 1
    if sample_count == 0:
        raise AutoVisionError("Cannot build a drift baseline from zero predictions.")
    if not label_counts:
        raise AutoVisionError("Reference predictions contain no labels.")
    return DriftBaseline(
        sample_count=sample_count,
        label_counts=dict(label_counts),
        score_histogram=histogram,
        bins=bins,
    )


def load_drift_baseline(path: str | Path) -> DriftBaseline:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as error:
        raise AutoVisionError(
            f"Cannot read drift baseline '{target}': {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise AutoVisionError(
            f"Invalid drift baseline JSON '{target}': {error}"
        ) from error
    return DriftBaseline.from_dict(payload)


def _psi(expected: dict[Any, float], actual: dict[Any, float]) -> float:
    total = 0.0
    for key in sorted(set(expected) | set(actual), key=str):
        exp = max(float(expected.get(key, 0.0)), _EPS)
        act = max(float(actual.get(key, 0.0)), _EPS)
        total += (act - exp) * math.log(act / exp)
    return float(total)


class DriftMonitor:
    """Rolling label/score distribution monitor against a baseline."""

    def __init__(
        self,
        baseline: DriftBaseline | dict[str, Any],
        *,
        window: int = 1000,
        threshold: float = 0.2,
        min_samples: int = 30,
    ) -> None:
        if isinstance(baseline, dict):
            baseline = DriftBaseline.from_dict(baseline)
        if window < 1:
            raise AutoVisionError(f"window must be >= 1, got {window}.")
        if threshold < 0:
            raise AutoVisionError(f"threshold must be >= 0, got {threshold}.")
        if min_samples < 1:
            raise AutoVisionError(f"min_samples must be >= 1, got {min_samples}.")
        self.baseline = baseline
        self.window = window
        self.threshold = threshold
        self.min_samples = min_samples
        self._records: deque[tuple[dict[int, int], list[int], int]] = deque(
            maxlen=window
        )
        self._lock = threading.Lock()

    def record(self, record: dict[str, Any]) -> None:
        stats = _record_stats(record, self.baseline.bins)
        if stats is None:
            return
        with self._lock:
            self._records.append(stats)

    def record_many(self, records: Iterable[dict[str, Any]]) -> None:
        for record in records:
            self.record(record)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
        sample_count = len(records)
        payload: dict[str, Any] = {
            "enabled": True,
            "status": "insufficient_data",
            "sample_count": sample_count,
            "min_samples": self.min_samples,
            "window": self.window,
            "threshold": self.threshold,
            "psi_label": None,
            "psi_score": None,
        }
        if sample_count < self.min_samples:
            return payload

        labels: Counter[int] = Counter()
        histogram = [0] * self.baseline.bins
        for label_counts, score_histogram, _ in records:
            labels.update(label_counts)
            histogram = [a + b for a, b in zip(histogram, score_histogram)]

        expected_labels = {
            label: count / max(1, sum(self.baseline.label_counts.values()))
            for label, count in self.baseline.label_counts.items()
        }
        label_total = sum(labels.values())
        actual_labels = (
            {label: count / label_total for label, count in labels.items()}
            if label_total else {}
        )
        score_total = sum(histogram)
        baseline_score_total = sum(self.baseline.score_histogram)
        expected_scores = {
            index: count / max(1, baseline_score_total)
            for index, count in enumerate(self.baseline.score_histogram)
        }
        actual_scores = (
            {index: count / score_total for index, count in enumerate(histogram)}
            if score_total else {}
        )
        psi_label = _psi(expected_labels, actual_labels) if label_total else 0.0
        psi_score = (
            _psi(expected_scores, actual_scores)
            if score_total and baseline_score_total else 0.0
        )
        payload.update({
            "status": "drift" if max(psi_label, psi_score) > self.threshold else "ok",
            "psi_label": psi_label,
            "psi_score": psi_score,
            "label_counts": dict(sorted(labels.items())),
            "score_histogram": histogram,
        })
        return payload
