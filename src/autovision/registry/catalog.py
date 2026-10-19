# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Task-level model catalog (roadmap 90d, phase 1).

The encoder catalog answers "which backbone can I build?"; this module
answers the product question "which *detector* can I load and deploy?".
Entries are curated, not discovered: a model appears here only when its
weights, license and export surface are known. The registry layer may not
import the task layer (layering contract), so the table is static and
``tests/test_model_catalog.py`` cross-checks every row against the task
registries (RF-DETR variants, GTR assets, DETR spectrum).

``status`` values:

- ``released``: downloadable weights exist (upstream or AutoVision);
- ``available``: architecture can be built, but no task-level weights;
- ``spec-frozen``: product commitment, weights/AP pending;
- ``planned``: declared target without a registry key yet.

``verified`` is deliberately stricter than ``released``: it means the
AutoVision release gate has a model card, a full-val metric, a reproducible
latency record and an end-to-end parity report. Upstream releases start
unverified until that evidence exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "MODEL_CATALOG",
    "ModelEntry",
    "get_model_entry",
    "list_task_models",
    "format_model_table",
    "describe_model",
]

ModelStatus = Literal["released", "available", "spec-frozen", "planned"]
ModelSource = Literal["autovision", "upstream"]
PretrainedCoverage = Literal["full", "backbone", "none"]


@dataclass(frozen=True)
class ModelEntry:
    """One curated task-level model (license + weights + deployment surface)."""

    key: str
    task: str
    family: str
    status: ModelStatus
    source: ModelSource
    pretrained: PretrainedCoverage
    license: str
    weights_asset: str | None = None
    input_size: tuple[int, int] | None = None
    params_m: float | None = None
    gflops: float | None = None
    coco_ap: float | None = None
    latency_ms: float | None = None
    exportable: tuple[str, ...] = ()
    verified: bool = False
    notes: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def has_full_weights(self) -> bool:
        return self.pretrained == "full"

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "key": self.key,
            "task": self.task,
            "family": self.family,
            "status": self.status,
            "source": self.source,
            "pretrained": self.pretrained,
            "license": self.license,
            "weights_asset": self.weights_asset,
            "input_size": list(self.input_size) if self.input_size else None,
            "params_m": self.params_m,
            "gflops": self.gflops,
            "coco_ap": self.coco_ap,
            "latency_ms": self.latency_ms,
            "exportable": list(self.exportable),
            "verified": self.verified,
            "notes": self.notes,
            **self.extra,
        }
        return payload


_DETR_EXPORTABLE = ("onnx", "tensorrt", "openvino")

#: Curated table. Metrics are the upstream published values (see
#: docs/benchmarks.md); AutoVision readings, when they exist, live there too.
#: Keeping them in one static table is what lets the registry layer stay
#: independent of the task layer; the consistency test is the guard.
MODEL_CATALOG: tuple[ModelEntry, ...] = (
    # ── RF-DETR: Apache-2.0, deployable flagship ───────────────────────
    ModelEntry(
        key="rfdetr/rf-detr-nano",
        task="detection",
        family="rfdetr",
        status="released",
        source="upstream",
        pretrained="full",
        license="Apache-2.0",
        weights_asset="Roboflow/rf-detr-nano",
        input_size=(384, 384),
        params_m=30.5,
        coco_ap=48.4,
        latency_ms=2.3,
        exportable=_DETR_EXPORTABLE,
        notes="Upstream Roboflow weights (T4/TensorRT latency); pretrained handle.",
    ),
    ModelEntry(
        key="rfdetr/rf-detr-small",
        task="detection",
        family="rfdetr",
        status="released",
        source="upstream",
        pretrained="full",
        license="Apache-2.0",
        weights_asset="Roboflow/rf-detr-small",
        input_size=(512, 512),
        params_m=32.1,
        coco_ap=53.0,
        latency_ms=3.5,
        exportable=_DETR_EXPORTABLE,
        notes="Upstream Roboflow weights (T4/TensorRT latency); pretrained handle.",
    ),
    ModelEntry(
        key="rfdetr/rf-detr-medium",
        task="detection",
        family="rfdetr",
        status="released",
        source="upstream",
        pretrained="full",
        license="Apache-2.0",
        weights_asset="Roboflow/rf-detr-medium",
        input_size=(576, 576),
        params_m=33.7,
        coco_ap=54.7,
        latency_ms=4.4,
        exportable=_DETR_EXPORTABLE,
        notes="Upstream Roboflow weights (T4/TensorRT latency); pretrained handle.",
    ),
    ModelEntry(
        key="rfdetr/rf-detr-large",
        task="detection",
        family="rfdetr",
        status="released",
        source="upstream",
        pretrained="full",
        license="Apache-2.0",
        weights_asset="Roboflow/rf-detr-large",
        input_size=(704, 704),
        params_m=33.9,
        coco_ap=56.5,
        latency_ms=6.8,
        exportable=_DETR_EXPORTABLE,
        notes="Upstream Roboflow weights (T4/TensorRT latency); pretrained handle.",
    ),
    # ── GTR: MIT, accuracy flagship (GLA kernel -> no export) ───────────
    ModelEntry(
        key="detr/gtrvit-s",
        task="detection",
        family="gtr",
        status="released",
        source="upstream",
        pretrained="full",
        license="MIT",
        weights_asset="gtr_s_coco",
        input_size=(640, 640),
        params_m=12.1,
        gflops=33.8,
        coco_ap=53.6,
        latency_ms=1.225,
        exportable=(),
        notes="Official GTR EMA weights; GLA custom kernel, not ONNX/TRT exportable.",
    ),
    ModelEntry(
        key="detr/gtrvit-m",
        task="detection",
        family="gtr",
        status="released",
        source="upstream",
        pretrained="full",
        license="MIT",
        weights_asset="gtr_m_coco",
        input_size=(640, 640),
        params_m=22.7,
        gflops=62.6,
        coco_ap=57.3,
        latency_ms=1.462,
        exportable=(),
        notes="Official GTR EMA weights; GLA custom kernel, not ONNX/TRT exportable.",
    ),
    ModelEntry(
        key="detr/gtrvit-l",
        task="detection",
        family="gtr",
        status="released",
        source="upstream",
        pretrained="full",
        license="MIT",
        weights_asset="gtr_l_coco",
        input_size=(640, 640),
        params_m=37.2,
        gflops=106.0,
        coco_ap=58.9,
        latency_ms=1.908,
        exportable=(),
        notes="Official GTR EMA weights; GLA custom kernel, not ONNX/TRT exportable.",
    ),
    ModelEntry(
        key="detr/gtrvit-x",
        task="detection",
        family="gtr",
        status="released",
        source="upstream",
        pretrained="full",
        license="MIT",
        weights_asset="gtr_x_coco",
        input_size=(640, 640),
        params_m=46.5,
        gflops=130.2,
        coco_ap=59.4,
        latency_ms=2.115,
        exportable=(),
        notes="Official GTR EMA weights; GLA custom kernel, not ONNX/TRT exportable.",
    ),
    # ── AutoVision DETR spectrum ───────────────────────────────────────
    ModelEntry(
        key="detr/n",
        task="detection",
        family="detr",
        status="spec-frozen",
        source="autovision",
        pretrained="none",
        license="Apache-2.0",
        weights_asset=None,
        input_size=(640, 640),
        params_m=17.79,
        gflops=59.1,
        exportable=_DETR_EXPORTABLE,
        notes="Spec-frozen; no published weights/AP. Current size is ~M-class.",
    ),
    ModelEntry(
        key="detr/s",
        task="detection",
        family="detr",
        status="spec-frozen",
        source="autovision",
        pretrained="none",
        license="Apache-2.0",
        weights_asset=None,
        input_size=(640, 640),
        params_m=18.14,
        gflops=61.8,
        exportable=_DETR_EXPORTABLE,
        notes="Spec-frozen; no published weights/AP.",
    ),
    ModelEntry(
        key="detr/m",
        task="detection",
        family="detr",
        status="planned",
        source="autovision",
        pretrained="none",
        license="Apache-2.0",
        notes="Planned (timm ConvNeXt line); no registry key yet.",
    ),
    ModelEntry(
        key="detr/l",
        task="detection",
        family="detr",
        status="planned",
        source="autovision",
        pretrained="none",
        license="Apache-2.0",
        notes="Planned (timm ConvNeXt line); no registry key yet.",
    ),
    ModelEntry(
        key="detr/x",
        task="detection",
        family="detr",
        status="planned",
        source="autovision",
        pretrained="none",
        license="Apache-2.0",
        notes="Planned (timm ConvNeXt line); no registry key yet.",
    ),
    # ── In-tree RT-DETR: architecture available, no task weights ───────
    ModelEntry(
        key="rtdetr/resnet18",
        task="detection",
        family="rtdetr",
        status="available",
        source="autovision",
        pretrained="none",
        license="Apache-2.0",
        input_size=(640, 640),
        params_m=26.46,
        gflops=87.1,
        exportable=_DETR_EXPORTABLE,
        notes="Train from scratch / custom data; no COCO task checkpoint registered.",
    ),
)


def _matches(
    entry: ModelEntry,
    *,
    task: str | None,
    pretrained_only: bool,
    include_planned: bool,
) -> bool:
    if task is not None and entry.task != task:
        return False
    if pretrained_only and not entry.has_full_weights:
        return False
    if not include_planned and entry.status == "planned":
        return False
    return True


def list_task_models(
    *,
    task: str | None = None,
    pretrained_only: bool = False,
    include_planned: bool = False,
) -> list[ModelEntry]:
    """Curated task models, optionally filtered.

    ``pretrained_only=True`` returns entries with full task weights;
    planned rows are hidden unless ``include_planned=True``.
    """
    return [
        entry
        for entry in MODEL_CATALOG
        if _matches(
            entry,
            task=task,
            pretrained_only=pretrained_only,
            include_planned=include_planned,
        )
    ]


def get_model_entry(key: str) -> ModelEntry:
    """Look up one catalog row by model key (``rfdetr/rf-detr-small``)."""
    for entry in MODEL_CATALOG:
        if entry.key == key:
            return entry
    known = ", ".join(entry.key for entry in MODEL_CATALOG)
    raise KeyError(f"Unknown task model '{key}'. Known: {known}.")


def _cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, tuple):
        return ",".join(str(part) for part in value) or "-"
    return str(value)


def format_model_table(entries: list[ModelEntry]) -> str:
    """Human-readable table for the CLI / docs (no third-party dependency)."""
    headers = (
        "key", "status", "weights", "license", "AP", "params(M)",
        "exportable",
    )
    rows = [
        (
            entry.key,
            entry.status,
            entry.pretrained,
            entry.license,
            _cell(entry.coco_ap),
            _cell(entry.params_m),
            _cell(entry.exportable),
        )
        for entry in entries
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(
            header.ljust(widths[index]) for index, header in enumerate(headers)
        )
    ]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append(
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        )
    return "\n".join(lines)


def describe_model(key: str) -> str:
    """Detailed multi-line description (``model-info`` CLI)."""
    entry = get_model_entry(key)
    exports = ", ".join(entry.exportable) or "none (not exportable)"
    lines = [
        f"model:      {entry.key}",
        f"task:       {entry.task}",
        f"family:     {entry.family}",
        f"status:     {entry.status}",
        f"source:     {entry.source}",
        f"weights:    {entry.pretrained}"
        + (f" ({entry.weights_asset})" if entry.weights_asset else ""),
        f"license:    {entry.license}",
        f"input:      {entry.input_size or '-'}",
        f"params:     {_cell(entry.params_m)} M",
        f"GFLOPs:     {_cell(entry.gflops)}",
        f"COCO AP:    {_cell(entry.coco_ap)}",
        f"latency:    {_cell(entry.latency_ms)} ms (source protocol)",
        f"exportable: {exports}",
        f"verified:   {_cell(entry.verified)}",
    ]
    if entry.notes:
        lines.append(f"notes:      {entry.notes}")
    return "\n".join(lines)
