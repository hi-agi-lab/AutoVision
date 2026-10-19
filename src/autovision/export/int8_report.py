# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""int8 vs fp32 accuracy report (roadmap 90d, phase 2).

Tensor parity answers "did the graph stay close?"; this module answers the
release question: **how much AP did int8 lose?** The script
``scripts/measure_int8_accuracy.py`` produces the metrics; this module
formats the verdict and writes ``report.json`` / ``report.md``.

Thresholds are absolute AP points (COCO convention, 0-100). A model card's
int8 claim should only be marked verified when ``passed`` is true.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from autovision.errors import AutoVisionError

__all__ = ["Int8AccuracyReport", "summarize_int8_accuracy"]

_METRIC_ORDER = (
    "map", "map_50", "map_75", "map_small", "map_medium", "map_large",
    "mar", "precision", "recall",
)


@dataclass
class Int8AccuracyReport:
    fp32: dict[str, float]
    int8: dict[str, float]
    deltas: dict[str, float]
    max_ap_drop: float
    passed: bool
    model: str | None = None
    backend: str | None = None
    calibration: str | None = None
    images: int | None = None
    hardware: dict[str, Any] = field(default_factory=dict)
    report_json: Path | None = None
    report_md: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model": self.model,
            "backend": self.backend,
            "calibration": self.calibration,
            "images": self.images,
            "max_ap_drop": self.max_ap_drop,
            "passed": self.passed,
            "fp32": dict(self.fp32),
            "int8": dict(self.int8),
            "deltas": dict(self.deltas),
            "hardware": dict(self.hardware),
        }

    def summary_text(self) -> str:
        fp32_map = self.fp32.get("map", float("nan"))
        int8_map = self.int8.get("map", float("nan"))
        return (
            f"map fp32={fp32_map:.4f} int8={int8_map:.4f} "
            f"delta={self.deltas.get('map', float('nan')):+.4f} "
            f"max_drop={self.max_ap_drop:.4f} "
            f"verdict={'PASS' if self.passed else 'FAIL'}"
        )

    def save(self, out_dir: str | Path) -> "Int8AccuracyReport":
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.report_json = directory / "report.json"
        self.report_md = directory / "report.md"
        self.report_json.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True)
        )
        self.report_md.write_text(self._render_markdown())
        return self

    def _render_markdown(self) -> str:
        lines = ["# int8 vs fp32 精度报告\n"]
        lines.append(
            f"- 模型：`{self.model or '-'}`；后端：`{self.backend or '-'}`；"
            f"校准：`{self.calibration or '-'}`；图像数：{self.images or '-'}\n"
        )
        lines.append(
            f"- 判定：**{'PASS' if self.passed else 'FAIL'}**"
            f"（AP 下降 ≤ {self.max_ap_drop}；实际 "
            f"{self.fp32.get('map', 0.0) - self.int8.get('map', 0.0):+.4f}）\n"
        )
        lines.append("| metric | fp32 | int8 | delta |")
        lines.append("|---|---|---|---|")
        for key in _METRIC_ORDER:
            if key not in self.fp32 and key not in self.int8:
                continue
            lines.append(
                f"| {key} | {self.fp32.get(key, 0.0):.4f} | "
                f"{self.int8.get(key, 0.0):.4f} | "
                f"{self.deltas.get(key, 0.0):+.4f} |"
            )
        if self.hardware:
            lines.append("\n## 环境\n")
            for key, value in sorted(self.hardware.items()):
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)


def summarize_int8_accuracy(
    fp32: Mapping[str, float],
    int8: Mapping[str, float],
    *,
    max_ap_drop: float = 1.0,
    model: str | None = None,
    backend: str | None = None,
    calibration: str | None = None,
    images: int | None = None,
    hardware: Mapping[str, Any] | None = None,
) -> Int8AccuracyReport:
    """Build the verdict from two metric dicts (same evaluation protocol)."""
    if max_ap_drop < 0:
        raise AutoVisionError(f"max_ap_drop must be >= 0, got {max_ap_drop}.")
    if "map" not in fp32 or "map" not in int8:
        raise AutoVisionError("Both metric dicts need a 'map' entry.")
    common = [key for key in _METRIC_ORDER if key in fp32 and key in int8]
    deltas = {key: float(int8[key]) - float(fp32[key]) for key in common}
    drop = float(fp32["map"]) - float(int8["map"])
    return Int8AccuracyReport(
        fp32={key: float(value) for key, value in fp32.items()},
        int8={key: float(value) for key, value in int8.items()},
        deltas=deltas,
        max_ap_drop=float(max_ap_drop),
        passed=drop <= float(max_ap_drop),
        model=model,
        backend=backend,
        calibration=calibration,
        images=images,
        hardware=dict(hardware or {}),
    )
