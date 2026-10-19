# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Model export: an exported ``.pt`` payload -> ONNX / TensorRT.

Thin CLI-side wrapper over :func:`load_task_model` + the network's export
mixes; the target format is dispatched on the ``out`` suffix and the
export kwargs (``precision=...``, batch profile, ...) pass through to the
network's own validators unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autovision.errors import AutoVisionError
from autovision.tasks.task_model_helpers import load_task_model
from autovision.types import PathLike

__all__ = ["export_model"]


def export_model(
    model: PathLike, out: PathLike, *, device: str = "cpu", **kwargs: Any
) -> Path:
    """Export the ``.pt`` artifact at ``model`` to the ``out`` suffix format.

    ``.onnx`` runs the standard pipeline (FP16 blacklist + numerical
    parity gate by default); ``.engine``/``.trt`` build a TensorRT engine
    from the staged ONNX; ``.xml`` converts to an OpenVINO IR (parity-gated
    against torch). Any task family with ONNX export support works — the
    artifact is rebuilt via :func:`load_task_model`.
    """
    out_path = Path(out)
    suffix = out_path.suffix.lower()
    if suffix not in {".onnx", ".engine", ".trt", ".xml"}:
        raise AutoVisionError(
            f"Unsupported export target '{out_path.name}' — use .onnx, "
            ".engine, .trt, or .xml (OpenVINO IR)."
        )
    network = load_task_model(model, device=device)
    if suffix == ".onnx":
        return Path(network.export_onnx(str(out_path), **kwargs))
    if suffix == ".xml":
        return Path(network.export_openvino(str(out_path), **kwargs))
    return Path(network.export_tensorrt(str(out_path), **kwargs))
