# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""OpenVINO backend: ONNX -> IR conversion + runtime parity gate (D3).

OpenVINO is the CPU-first deployment target (Apache-2.0, pip-installable,
large installed base). The IR is produced from the framework's OWN ONNX
export, so the ONNX-side gates carry over (fp32/fp16 with the op
blacklist); this module adds the OpenVINO-specific steps: convert the ONNX
graph to IR (:func:`openvino.convert_model`), compile it on the requested
device, and — before the artifact is accepted — verify torch vs OpenVINO
outputs on the model's example input through the same
``verify_onnx_export_outputs`` hook every other exporter uses.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Any

from autovision.errors import AutoVisionError

__all__ = [
    "OpenVINOEngineRunner",
    "export_onnx_to_openvino",
]

logger = logging.getLogger("autovision")


def _import_openvino():
    try:
        import openvino
    except ImportError as error:  # pragma: no cover - exercised via the gate path
        raise AutoVisionError(
            "The 'openvino' package is required for the OpenVINO backend. "
            "Install it with: pip install 'autovision[openvino]'."
        ) from error
    return openvino


def _resolve_ir_path(out: "str | os.PathLike[str]") -> pathlib.Path:
    """``out`` is the ``.xml`` path; anything else is a directory target
    (``<dir>/model.xml``) — including extensionless paths, but NOT
    ``*.onnx``-like file names, which would silently become a directory."""
    target = pathlib.Path(os.fspath(out))
    if target.suffix.lower() != ".xml":
        if target.suffix and not target.is_dir():
            raise ValueError(
                f"OpenVINO IR target '{target.name}' is neither a .xml file "
                "nor a directory; pass '<dir>' or '<dir>/model.xml'."
            )
        target = target / "model.xml"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _verify_openvino_parity(openvino, model: Any, xml_path: pathlib.Path, device: str) -> None:
    """Torch vs the COMPILED OpenVINO model on the example input, through the
    model's own export-parity hook — an artifact that fails is not accepted."""
    import copy
    import dataclasses

    compiled = openvino.Core().compile_model(str(xml_path), device)
    reference = copy.deepcopy(model).cpu().float().eval()
    if hasattr(reference, "deploy"):
        reference.deploy()
    example_inputs = model.model_input_spec.example_inputs(batch_size=2)
    input_name = next(iter(example_inputs))
    example = example_inputs[input_name].cpu().float()
    torch_outputs = reference.forward_backend(example)
    if not isinstance(torch_outputs, (list, tuple)) and dataclasses.is_dataclass(torch_outputs):
        torch_outputs = [
            getattr(torch_outputs, f.name) for f in dataclasses.fields(torch_outputs)
        ]
    result = compiled(example.numpy())
    openvino_outputs = [result[port] for port in compiled.outputs]
    model.verify_onnx_export_outputs(torch_outputs, openvino_outputs)
    logger.info("OpenVINO parity gate passed on device '%s'.", device)


def export_onnx_to_openvino(
    *,
    out: "str | os.PathLike[str]",
    export_onnx_fn: Any,
    model: Any,
    precision: str = "fp32",
    device: str = "CPU",
    onnx_args: dict[str, Any] | None = None,
    verify: bool = True,
    keep_onnx: bool = False,
) -> pathlib.Path:
    """Convert ``model``'s ONNX export to an OpenVINO IR at ``out``.

    ``out`` is the ``.xml`` path or a directory (then ``<dir>/model.xml``;
    the ``.bin`` weights land beside it). The intermediate ONNX is removed
    unless ``keep_onnx``. Metadata rides in a ``<stem>.json`` sidecar (IR
    cannot carry ONNX metadata); with ``verify`` the torch reference and
    the COMPILED OpenVINO model are compared on the example input via
    ``model.verify_onnx_export_outputs`` — an OpenVINO artifact that fails
    the parity gate is never accepted.
    """
    openvino = _import_openvino()
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"precision must be 'fp32' or 'fp16', got '{precision}'")

    xml_path = _resolve_ir_path(out)
    onnx_path = xml_path.with_suffix(".onnx")

    def _drop_intermediate() -> None:
        for candidate in (onnx_path, onnx_path.with_suffix(".onnx.data")):
            candidate.unlink(missing_ok=True)

    export_kwargs: dict[str, Any] = dict(onnx_args or {})
    export_kwargs.setdefault("precision", precision)
    try:
        export_onnx_fn(onnx_path, **export_kwargs)
        ov_model = openvino.convert_model(str(onnx_path))
        openvino.save_model(ov_model, str(xml_path))
        logger.info("Saved OpenVINO IR to '%s'.", xml_path)
        if verify:
            _verify_openvino_parity(openvino, model, xml_path, device)
    finally:
        # The intermediate ONNX (and dynamo's external-data sidecar) never
        # outlives the call unless the caller asked for it — including the
        # gate-failure path (the IR stays, mirroring the ONNX exporter
        # leaving its artifact after a failed gate).
        if not keep_onnx:
            _drop_intermediate()

    metadata = {
        **model.onnx_export_metadata(),
        "backend": "openvino",
        "precision": precision,
        "device": device,
    }
    xml_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return xml_path


class OpenVINOEngineRunner:
    """Compiled OpenVINO model with the framework's input/output ordering.

    ``run(images)`` takes the same CPU tensors as the other backends and
    returns framework-ordered outputs, so parity comparisons and timing
    harnesses stay backend-agnostic.
    """

    def __init__(self, ir_path: "str | os.PathLike[str]", device: str = "CPU") -> None:
        openvino = _import_openvino()
        self.ir_path = pathlib.Path(os.fspath(ir_path))
        if not self.ir_path.exists():
            raise FileNotFoundError(f"OpenVINO IR not found: '{self.ir_path}'.")
        self.device = device
        self.core = openvino.Core()
        self.compiled = self.core.compile_model(str(self.ir_path), device)

    def run(self, images: Any) -> list[Any]:
        import torch

        result = self.compiled(images.detach().cpu().numpy())
        return [torch.as_tensor(result[port]) for port in self.compiled.outputs]
