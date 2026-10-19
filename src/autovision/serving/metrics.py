# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Minimal dependency-free Prometheus-text metrics (F2 serving monitoring)."""

from __future__ import annotations

import threading
from collections import defaultdict

__all__ = ["MetricsRegistry"]


class MetricsRegistry:
    """Counters, gauges and summaries rendered in Prometheus exposition text.

    Deliberately dependency-free (no ``prometheus_client``): the exposition
    format is a few lines of text, and the framework's extras policy keeps
    the serving stack minimal. Thread-safe (one lock; the served workload is
    the model forward, not the counters).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._summaries: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, float]] = {}
        self._help: dict[str, tuple[str, str]] = {}  # name -> (type, help)

    # ── write ───────────────────────────────────────────────────────────
    def _key(self, name: str, labels: dict[str, str] | None) -> tuple:
        return name, tuple(sorted((labels or {}).items()))

    def _declare(self, name: str, kind: str, help_text: str) -> None:
        existing = self._help.get(name)
        if existing is None:
            self._help[name] = (kind, help_text)
        elif not existing[1] and help_text:
            # First call site may pass no help text; a later one upgrades it so
            # the HELP line never depends on call order.
            self._help[name] = (existing[0], help_text)

    def inc(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0,
            help_text: str = "") -> None:
        with self._lock:
            self._declare(name, "counter", help_text)
            self._counters[self._key(name, labels)] += value

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None,
                  help_text: str = "") -> None:
        with self._lock:
            self._declare(name, "gauge", help_text)
            self._gauges[self._key(name, labels)] = float(value)

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None,
                help_text: str = "") -> None:
        """Record one observation of a summary.

        The exposition carries the standard ``_count``/``_sum`` pair; the
        highest observation rides in its own ``<name>_max`` gauge family
        (``_max`` is not part of the Prometheus summary contract).
        """
        with self._lock:
            self._declare(name, "summary", help_text)
            entry = self._summaries.setdefault(
                self._key(name, labels), {"count": 0.0, "sum": 0.0}
            )
            entry["count"] += 1
            entry["sum"] += float(value)
            max_name = f"{name}_max"
            self._declare(max_name, "gauge", f"Highest observed value of {name}.")
            max_key = self._key(max_name, labels)
            self._gauges[max_key] = max(
                self._gauges.get(max_key, float("-inf")), float(value)
            )

    # ── read ────────────────────────────────────────────────────────────
    def render(self) -> str:
        lines: list[str] = []

        def labels_text(labels: tuple[tuple[str, str], ...]) -> str:
            if not labels:
                return ""
            inner = ",".join(f'{key}="{value}"' for key, value in labels)
            return "{" + inner + "}"

        with self._lock:
            for name in sorted(self._help):
                kind, help_text = self._help[name]
                if help_text:
                    lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {kind}")
                if kind == "counter":
                    for (key_name, labels), value in sorted(self._counters.items()):
                        if key_name == name:
                            lines.append(f"{name}{labels_text(labels)} {value:g}")
                elif kind == "gauge":
                    for (key_name, labels), value in sorted(self._gauges.items()):
                        if key_name == name:
                            lines.append(f"{name}{labels_text(labels)} {value:g}")
                else:
                    for (key_name, labels), entry in sorted(self._summaries.items()):
                        if key_name == name:
                            suffix = labels_text(labels)
                            lines.append(f"{name}_count{suffix} {entry['count']:g}")
                            lines.append(f"{name}_sum{suffix} {entry['sum']:g}")
        return "\n".join(lines) + "\n"
