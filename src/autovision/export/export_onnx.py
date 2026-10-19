# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""ONNX export pipeline (Path A: torch.export -> dynamo ONNX -> gates).

Pipeline: eval/fp32 -> deploy hooks -> ``torch.export`` (strict, batch-dim
dynamic) -> ``torch.onnx.export(dynamo=True)`` -> optional FP16 conversion
with a per-model fp32 op blacklist -> onnxslim -> value-info repair ->
metadata -> **verification gate** (onnx.checker + onnxruntime parity with
atol 5e-3 / rtol 1e-1 on dim-1 sums).

int8 PTQ (``export_onnx_int8``) reuses this pipeline for the FP32 graph and
post-processes it with ``onnxruntime.quantization.quantize_static`` (QDQ,
calibrated) — see ``autovision.export.quantization``.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
from torch import nn

from autovision._license import LICENSE_INFO
from autovision._version import __version__
from autovision.errors import AutoVisionError
from autovision.runtime.fs_utils import sweep_stale_temp_files

__all__ = ["ONNXExportMixin", "ONNXExportPrecisionPolicy"]

# Default int8 parity-gate tolerances: int8 activation quantization is
# expected to shift logits; the gate only catches structural breakage.
INT8_PARITY_ATOL = 2.0
INT8_PARITY_RTOL = 0.5

logger = logging.getLogger("autovision")


def _spec_dynamic_axes(spec: Any, input_name: str) -> dict[int, Any]:
    """Explicit axis markers from ``ModelInputSpec.input_dynamic_shapes``.

    Returns ``{axis_index: marker}`` for the input's declared axes in rank
    order (None entries are static). The legacy exporter only uses the axis
    indices; the dynamo path converts markers via :func:`_dim_from_marker`.
    """
    declared = getattr(spec, "input_dynamic_shapes", None) or {}
    axes = declared.get(input_name)
    if not axes:
        return {}
    return {
        index: marker for index, marker in enumerate(axes) if marker is not None
    }


def _dim_from_marker(marker: Any, axis: int) -> Any:
    """``torch.export.Dim`` for one dynamic-axis marker.

    Accepted forms: a ready ``Dim`` (used as-is), an axis NAME (str →
    ``Dim(name, min=1)``), or a ``(min, max)`` tuple. NOTE: the name form
    gets a ``min=1`` bound, which a model with internal size guards (e.g.
    feature pyramids asserting ``size // divisor >= 1``) will REJECT with
    ``Constraints violated``; use the tuple form (or the
    ``dynamic_spatial`` flag, whose bounds derive from the example) there.
    """
    from torch.export import Dim

    if isinstance(marker, Dim):
        return marker
    if isinstance(marker, str):
        return Dim(marker, min=1)
    if isinstance(marker, (tuple, list)) and len(marker) == 2:
        low, high = marker
        return Dim(f"dynamic_axis{axis}", min=int(low), max=int(high))
    raise ValueError(
        f"input_dynamic_shapes marker {marker!r} on axis {axis} is not a "
        "Dim / axis name / (min, max) tuple."
    )


def _drop_onnx_artifacts(*paths: Path) -> None:
    """Remove ONNX files with their external-data sidecars (dynamo writes
    ``<name>.onnx.data`` for large graphs — 100MB-class litter otherwise)."""
    for path in paths:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".data").unlink(missing_ok=True)


def _move_onnx_artifacts(source: Path, target: Path) -> None:
    """Move an ONNX graph and its external-data sidecar atomically."""
    os.replace(source, target)
    source_data = source.with_suffix(source.suffix + ".data")
    if source_data.exists():
        os.replace(source_data, target.with_suffix(target.suffix + ".data"))


def _dynamic_export_error(exc: BaseException) -> str:
    """Actionable message for a failed dynamic-spatial dynamo export.

    ``torch.export``/``torch.onnx.export(dynamo=True)`` raise deep dynamo
    internals; the two failure classes users hit are a missing/misbehaving
    dynamo toolchain and a model that branches in PYTHON on the input size.
    """
    hint = ""
    if "Constraints violated" in str(exc):
        # A trace that generated an input-size guard cannot hold across a
        # whole dynamic range: such backbones branch or guard in PYTHON on
        # the input size — the transformers ViT embeddings switch between
        # "native size" and "interpolate" on the patch count, and the HF
        # ConvNeXt stage grid guards its divisor relations.
        hint = (
            "Cause: this backbone branches/guards in PYTHON on the INPUT "
            "SIZE (e.g. the transformers ViT patch-count branch or HF "
            "stage-grid guards), which strict export cannot make symbolic — "
            "dynamic spatial export is not available for it. "
        )
    return (
        hint
        + "Dynamic spatial axes require the dynamo export path "
        "(torch.export + torch.onnx.export(dynamo=True)); it failed with "
        f"{type(exc).__name__}: {exc}. Export with dynamic_spatial=False "
        "(static H/W), use a backbone whose graph traces symbolically "
        "(e.g. rtdetr/resnet18, the HGNetV2 tiers), or verify the "
        "'onnxscript' package is installed (part of the 'export' extra)."
    )


class ONNXExportPrecisionPolicy:
    """Per-model ops that must stay FP32 in an FP16 export."""

    def __init__(self, fp32_onnx_op_types: tuple[str, ...] = ()) -> None:
        self.fp32_onnx_op_types = tuple(fp32_onnx_op_types)


class ONNXExportMixin(nn.Module):
    """Mixin for :class:`TaskNetwork` subclasses that can export ONNX."""

    onnx_export_precision_policy = ONNXExportPrecisionPolicy()

    #: Non-``None`` blocks export with this message. Submodules declare it
    #: when their forward contains ops the pipeline cannot lower (e.g. the
    #: GTR GLA custom CUDA kernels) — the guard scans the whole module tree
    #: so any composition containing such a submodule is covered.
    export_blocker: str | None = None

    @property
    def model_input_spec(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def verify_onnx_export_outputs(
        self, torch_outputs: Any, onnx_outputs: list
    ) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _export_blocker(self) -> str | None:
        """First non-``None`` ``export_blocker`` in the module tree."""
        if self.export_blocker is not None:
            return self.export_blocker
        for module in self.modules():
            blocker = getattr(module, "export_blocker", None)
            if blocker is not None:
                return blocker
        return None

    def onnx_export_metadata(self) -> dict[str, str]:
        metadata = {
            "autovision_version": __version__,
            "license_info": LICENSE_INFO,
        }
        model_name = self.init_args.get("model_name") or self.init_args.get("backbone")
        if model_name:
            metadata["model_name"] = str(model_name)
        normalize = getattr(self, "normalize", None)
        if normalize is not None:
            metadata["image_normalize"] = json.dumps(normalize.to_dict(), sort_keys=True)
        if hasattr(self, "num_classes"):
            metadata["num_classes"] = str(self.num_classes)
        return metadata

    @torch.no_grad()
    def export_onnx(
        self,
        out: Path | str,
        *,
        precision: str = "fp32",
        batch_size: int = 1,
        dynamic_batch_size: bool = True,
        dynamic_spatial: bool = False,
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        write_spec: bool = True,
        format_args: dict[str, Any] | None = None,
    ) -> Path:
        """Export to a temp graph, verify, then atomically replace ``out``.

        A failed export never leaves a partial graph at the target path (the
        previous good artifact survives), and the postprocess sidecar is only
        written after the graph has passed its gates.
        """
        final_out = Path(out)
        final_out.parent.mkdir(parents=True, exist_ok=True)
        # Hygiene: sweep temp graphs orphaned by a killed exporter. The
        # pattern ends in ".tmp<suffix>" so a kept FP32 reference
        # ("{stem}.fp32.onnx") can never match.
        sweep_stale_temp_files(
            final_out.parent,
            f"{glob.escape(final_out.stem)}"
            f".*.tmp{glob.escape(final_out.suffix or '.onnx')}",
        )
        # No leading dot: ONNX external-data locations starting with '..'
        # are rejected by onnx.load as path traversal.
        tmp_out = final_out.with_name(
            f"{final_out.stem}.{uuid.uuid4().hex}.tmp{final_out.suffix or '.onnx'}"
        )
        try:
            self._export_onnx_to_path(
                tmp_out,
                precision=precision,
                batch_size=batch_size,
                dynamic_batch_size=dynamic_batch_size,
                dynamic_spatial=dynamic_spatial,
                opset_version=opset_version,
                simplify=simplify,
                verify=verify,
                write_spec=False,
                format_args=format_args,
            )
            _move_onnx_artifacts(tmp_out, final_out)
        except Exception:
            _drop_onnx_artifacts(tmp_out)
            raise
        if write_spec:
            _maybe_write_postprocess_spec(self, final_out)
        return final_out

    @torch.no_grad()
    def _export_onnx_to_path(
        self,
        out: Path | str,
        *,
        precision: str = "fp32",
        batch_size: int = 1,
        dynamic_batch_size: bool = True,
        dynamic_spatial: bool = False,
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        write_spec: bool = True,
        format_args: dict[str, Any] | None = None,
    ) -> Path:
        """Export the model to ONNX with verification gates.

        ``write_spec=False`` is used by the TensorRT/OpenVINO wrappers for
        their intermediate graph: only the final artifact gets a
        ``.postprocess.json`` sidecar.

        ``dynamic_spatial=True`` marks the H/W axes dynamic (roadmap D2):
        the graph then accepts any input size the network itself supports
        (e.g. multiples of the variant's patch/stride divisor — the caller
        resizes). ``model_input_spec.input_dynamic_shapes`` markers are
        honored on top: a non-``None`` entry (a ``torch.export.Dim``, an
        axis name, or a ``(min, max)`` tuple) marks that axis dynamic. The
        parity gate then verifies a SECOND spatial size (torch vs ONNX
        Runtime) so a graph that merely traces cannot pass.
        """

        blocker = self._export_blocker()
        if blocker is not None:
            raise AutoVisionError(blocker)

        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision must be 'fp32' or 'fp16', got '{precision}'")
        if precision == "fp16" and not simplify:
            raise ValueError("fp16 export requires simplify=True.")

        # 1. trace in fp32 (fp16 conversion happens on the graph afterwards).
        module: nn.Module = self
        module.eval()
        module.to(torch.float32)
        if hasattr(self, "deploy"):
            self.deploy()

        # 2. torch.export (strict) with batch-dim dynamic shapes.
        from torch.export import Dim

        spec = self.model_input_spec
        batch_dim = Dim("batch_size", min=1, max=128) if dynamic_batch_size else batch_size
        # Trace on the model's own device: a CUDA-resident model (the normal
        # case right after training) would otherwise hit a device mismatch
        # against a CPU example input inside the strict export.
        parameter = next(module.parameters(), None)
        example_device = (
            parameter.device if parameter is not None else torch.device("cpu")
        )
        example_name, example_tensor = next(iter(spec.example_inputs(
            batch_size=2 if dynamic_batch_size else batch_size, device=example_device
        ).items()))
        # Warm one eager forward BEFORE tracing: some backbones run one-time
        # setup on their FIRST eager call that dynamo cannot trace — the
        # transformers ViT family installs its output-capturing hooks under a
        # ``threading.Lock``, and dynamo cannot trace ``with <lock>``, so the
        # strict trace fails outright for every HF-backed variant. After the
        # warm call the hooks are already installed and the trace takes the
        # fast path. (Numerically transparent: the hooks only record.)
        module(example_tensor)
        # The TaskNetwork forward contract is ``forward(self, x: Tensor)``.
        dynamic_shapes = {"x": self._onnx_dynamic_shapes(
            spec, example_name, batch_dim, dynamic_batch_size, dynamic_spatial,
            example_tensor=example_tensor,
        )}
        dynamic_axes_requested = dynamic_spatial or any(
            axis != 0 for axis in _spec_dynamic_axes(spec, example_name)
        )
        program = None
        trace_error: Exception | None = None
        try:
            program = torch.export.export(
                module, args=(example_tensor,), dynamic_shapes=dynamic_shapes,
                strict=True,
            )
        except Exception as exc:  # noqa: BLE001 - routed below
            trace_error = exc
        if program is None and dynamic_axes_requested:
            # The legacy TorchScript exporter bakes the traced spatial size
            # into the graph: it would produce a "dynamic" graph whose sizes
            # silently break elsewhere, so dynamic EXPORT requires the dynamo
            # path outright (no legacy fallback).
            assert trace_error is not None
            raise AutoVisionError(_dynamic_export_error(trace_error)) from trace_error
        if program is None:
            # A strict-trace failure with STATIC axes is not fatal: the legacy
            # TorchScript exporter handles some graphs dynamo cannot (the
            # documented supported fallback across the torch>=2.4 range).
            logger.warning(
                "torch.export strict trace failed (%s: %s); retrying with "
                "the legacy TorchScript exporter.",
                type(trace_error).__name__, trace_error,
            )
            self._legacy_onnx_export(
                module, example_tensor, out, spec, example_name, format_args,
                opset_version, dynamic_batch_size,
            )
        else:
            # 3. dynamo ONNX export (graph inputs get the spec names).
            export_kwargs: dict[str, Any] = {"opset_version": opset_version}
            export_kwargs.update(format_args or {})
            try:
                torch.onnx.export(
                    program,
                    args=(),
                    f=str(out),
                    input_names=list(spec.input_specs),
                    output_names=self._onnx_output_names(example_tensor),
                    dynamo=True,
                    dynamic_shapes=dynamic_shapes,
                    **export_kwargs,
                )
            except Exception as exc:
                if dynamic_axes_requested:
                    raise AutoVisionError(_dynamic_export_error(exc)) from exc
                # torch 2.4/2.5 dynamo exporter gaps (e.g. unregistered
                # ``_native_batch_norm_legit_no_training`` decompositions,
                # tuple attrs in fp16 paths) break models that the legacy
                # TorchScript exporter handles fine; the legacy path is the
                # supported fallback across the declared torch>=2.4 range.
                logger.warning(
                    "dynamo ONNX export failed (%s: %s); retrying with the "
                    "legacy TorchScript exporter.", type(exc).__name__, exc,
                )
                self._legacy_onnx_export(
                    module, example_tensor, out, spec, example_name, format_args,
                    opset_version, dynamic_batch_size,
                )

        # 4. FP16 conversion with the op blacklist.
        if precision == "fp16":
            self._apply_fp16(out)

        # 5. onnxslim (constant folding skipped to avoid model inflation).
        if simplify:
            import onnxslim

            onnxslim.slim(str(out), output=str(out), skip_optimizations=["constant_folding"])

        # 6. repair value_info (dynamo exporter can leave wrong rank hints).
        _repair_value_info(out)

        # 7. metadata.
        _write_onnx_metadata(out, self.onnx_export_metadata())

        # 8. verification gate.
        if verify:
            self._verify_onnx(
                out, example_name, example_tensor, precision,
                dynamic_spatial=dynamic_spatial,
            )
        logger.info("Exported ONNX model to '%s' (%s).", out, precision)
        if write_spec:
            _maybe_write_postprocess_spec(self, out)
        return out

    # ── internals ───────────────────────────────────────────────────────
    def _legacy_onnx_export(
        self,
        module: nn.Module,
        example_tensor: torch.Tensor,
        out: Path,
        spec: Any,
        example_name: str,
        format_args: dict[str, Any] | None,
        opset_version: int | None,
        dynamic_batch_size: bool,
    ) -> None:
        """Legacy TorchScript export (the documented dynamo fallback).

        Only reached with STATIC spatial axes (the caller rejects it when
        dynamic axes were requested): TorchScript traces the example's sizes
        into the graph, so a "dynamic" legacy graph would be a lie.
        """
        export_kwargs: dict[str, Any] = {"opset_version": opset_version}
        export_kwargs.update(format_args or {})
        legacy_kwargs = {
            key: value
            for key, value in export_kwargs.items()
            if key not in ("dynamo", "dynamic_shapes", "dynamic_axes")
        }
        input_names = list(spec.input_specs)
        axes: dict[int, str] = {}
        # Mirror ``_onnx_dynamic_shapes``: the batch axis is governed by the
        # FLAG only (a marker on axis 0 is warned about and dropped there), and
        # spatial axes never reach this path (the caller rejects legacy export
        # when any dynamic axis was requested — the TorchScript graph would
        # bake the traced size in).
        if dynamic_batch_size:
            axes[0] = "batch_size"
        for axis, marker in _spec_dynamic_axes(spec, example_name).items():
            if axis == 0:
                continue
            axes[axis] = marker if isinstance(marker, str) else f"dynamic_axis{axis}"
        dynamic_axes = {input_names[0]: axes} if (axes and input_names) else None
        torch.onnx.export(
            module,
            (example_tensor,),
            f=str(out),
            input_names=input_names,
            output_names=self._onnx_output_names(example_tensor),
            dynamo=False,
            dynamic_axes=dynamic_axes,
            **legacy_kwargs,
        )

    def _onnx_dynamic_shapes(
        self,
        spec: Any,
        example_name: str,
        batch_dim: Any,
        dynamic_batch_size: bool,
        dynamic_spatial: bool,
        *,
        example_tensor: torch.Tensor | None = None,
    ) -> tuple[Any, ...]:
        """Per-axis dynamic markers for the image input.

        Batch follows ``dynamic_batch_size``; H/W are dynamic when
        ``dynamic_spatial`` is set; ``spec.input_dynamic_shapes`` entries
        mark additional/other axes explicitly. The H/W bounds are derived
        from the EXAMPLE size (``min = min(64, example)``): a fixed lower
        bound can undercut a model's own internal guard (e.g. feature
        pyramids assert ``size // divisor >= 1``), and torch.export then
        refuses to export because the Dim range admits violating values.
        """
        from torch.export import Dim

        rank = 4  # the TaskNetwork image contract is NCHW
        axes: list[Any] = [None] * rank
        axes[0] = batch_dim if dynamic_batch_size else None
        if dynamic_spatial:
            examples = (
                tuple(int(v) for v in example_tensor.shape[2:4])
                if example_tensor is not None else (640, 640)
            )
            axes[2] = Dim("height", min=min(64, examples[0]), max=8192)
            axes[3] = Dim("width", min=min(64, examples[1]), max=8192)
        for axis, marker in _spec_dynamic_axes(spec, example_name).items():
            if axis >= rank:
                raise ValueError(
                    f"input_dynamic_shapes for '{example_name}' marks axis "
                    f"{axis}, but the image input has rank {rank}."
                )
            if axis == 0:
                if not dynamic_batch_size:
                    logger.warning(
                        "input_dynamic_shapes declares a dynamic batch axis, "
                        "but dynamic_batch_size=False; the batch axis stays "
                        "STATIC."
                    )
                continue  # batch is governed by dynamic_batch_size
            axes[axis] = _dim_from_marker(marker, axis)
        return tuple(axes)

    def _onnx_output_names(self, example_tensor: torch.Tensor) -> list[str]:
        import dataclasses

        sample = self.forward_backend(example_tensor)
        if dataclasses.is_dataclass(sample):
            return [f.name for f in dataclasses.fields(sample)]
        if isinstance(sample, dict):
            return list(sample)
        return [f"output_{i}" for i in range(len(sample) if isinstance(sample, (list, tuple)) else 1)]

    def _apply_fp16(self, out: Path) -> None:
        import onnx
        from onnxruntime.transformers import float16 as ort_float16

        model = onnx.load(str(out))
        op_block_list = list(ort_float16.DEFAULT_OP_BLOCK_LIST) + list(
            self.onnx_export_precision_policy.fp32_onnx_op_types
        )
        model = ort_float16.convert_float_to_float16(model, op_block_list=op_block_list)
        _remove_duplicate_casts(model)
        _fix_topological_order(model)
        onnx.save(model, str(out))

    def _verify_onnx(
        self, out: Path, example_name: str, example_tensor: torch.Tensor,
        precision: str,
        dynamic_spatial: bool = False,
    ) -> None:
        import copy

        import onnx
        import onnxruntime as ort

        onnx.checker.check_model(str(out), full_check=True)
        if precision == "fp16" and "CUDAExecutionProvider" not in ort.get_available_providers():
            logger.warning("FP16 parity check skipped: no CUDA execution provider.")
            return

        reference = copy.deepcopy(self).cpu().float().eval()
        if hasattr(reference, "deploy"):
            reference.deploy()
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])

        def _compare(tensor: torch.Tensor) -> None:
            torch_outputs = reference.forward_backend(tensor.float().cpu())
            if not isinstance(torch_outputs, (list, tuple)):
                import dataclasses
                if dataclasses.is_dataclass(torch_outputs):
                    torch_outputs = [getattr(torch_outputs, f.name) for f in dataclasses.fields(torch_outputs)]
            feeds = {}
            for session_input in session.get_inputs():
                feed = tensor.float().cpu()
                if session_input.type == "tensor(float16)":
                    feed = feed.half()
                feeds[session_input.name] = feed.numpy()
            onnx_outputs = session.run(None, feeds)
            self.verify_onnx_export_outputs(torch_outputs, onnx_outputs)

        # The comparison input is SEEDED RANDOM, not the spec's all-zero
        # example: on a zero image a DETR's query selection has tie-broken
        # grid coordinates (many equal scores), so element-wise near-ties
        # between torch and the backend flip top-k boxes and the gate turns
        # flaky on a perfectly valid graph (largest observed: box sums
        # differing by ~0.5 on a 640px two-class model). Tracing still uses
        # the spec example.
        generator = torch.Generator().manual_seed(0)
        compare_tensor = torch.rand(
            example_tensor.shape, generator=generator, dtype=example_tensor.dtype
        )
        _compare(compare_tensor)
        if dynamic_spatial:
            # A graph that merely traced at one size can still be wrong at
            # another; the gate runs a SECOND spatial size (torch vs ORT at
            # the same size — the "does the dynamic graph stay accurate"
            # question, independent of the variant's resize conventions).
            # The probe scales DOWN for large examples (the common
            # production need) and UP for small ones (a downscale would
            # undercut feature-pyramid floors).
            _, _, height, width = example_tensor.shape
            factor = (2, 3) if min(height, width) >= 128 else (3, 2)
            probe_height = max(1, height * factor[0] // factor[1])
            probe_width = max(1, width * factor[0] // factor[1])
            probe = torch.rand(
                (example_tensor.shape[0], example_tensor.shape[1],
                 probe_height, probe_width),
                dtype=example_tensor.dtype,
            )
            logger.info(
                "Dynamic-spatial parity probe at %dx%d (example %dx%d).",
                probe_width, probe_height, width, height,
            )
            _compare(probe)

    def export_onnx_int8(
        self,
        out: Path | str,
        *,
        calibration_data: Iterable[dict[str, torch.Tensor]] | None = None,
        calibration_reader: Any | None = None,
        quantization_config: Any | None = None,
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        parity_atol: float = INT8_PARITY_ATOL,
        parity_rtol: float = INT8_PARITY_RTOL,
        keep_fp32_reference: bool = False,
    ) -> Path:
        """Export a QDQ int8 ONNX model via static PTQ (§12.2 extension).

        The FP32 ONNX is produced by the standard dynamo pipeline
        (:meth:`export_onnx`), then post-processed with
        ``onnxruntime.quantization.quantize_static`` using the calibration
        samples. Provide either ``calibration_data`` (an iterable of
        ``{input_name: Tensor}`` dicts) or a prebuilt ORT
        ``calibration_reader`` — the parity gate replays the first
        calibration sample against the FP32 graph.
        """
        from autovision.export.quantization import (
            Int8CalibrationReader,
            quantize_onnx_int8,
        )

        if calibration_data is None and calibration_reader is None:
            raise ValueError(
                "int8 PTQ requires calibration data: pass calibration_data "
                "(an image path/dir or an iterable of {input_name: Tensor}) "
                "or calibration_reader."
            )
        if calibration_data is not None:
            from autovision.export.quantization import resolve_calibration_samples

            calibration_data = resolve_calibration_samples(self, calibration_data)
        final_out = Path(out)
        final_out.parent.mkdir(parents=True, exist_ok=True)
        # Hygiene: only the int8 chain's own temp names (".int8.tmp…") —
        # a kept "{stem}.fp32.onnx" reference must never match.
        stem = glob.escape(final_out.stem)
        sweep_stale_temp_files(final_out.parent, f"{stem}.*.int8.tmp.onnx")
        sweep_stale_temp_files(final_out.parent, f"{stem}.*.int8.tmp.fp32.onnx")
        tmp_out = final_out.with_name(
            f"{final_out.stem}.{uuid.uuid4().hex}.int8.tmp.onnx"
        )

        # 1. FP32 reference graph (passes the standard FP32 gate itself).
        fp32_path = tmp_out.parent / f"{tmp_out.stem}.fp32.onnx"
        try:
            self.export_onnx(
                fp32_path,
                precision="fp32",
                opset_version=opset_version,
                simplify=simplify,
                verify=verify,
                write_spec=False,
            )

            # 2. Static quantization (QDQ) with calibration.
            reader = calibration_reader or Int8CalibrationReader(
                calibration_data, expected_input_names=list(self.model_input_spec.input_specs)
            )
            quantize_onnx_int8(fp32_path, tmp_out, reader, quantization_config)
            # The reader's sample count is a property of the QUANTIZATION
            # pass; capture it BEFORE the parity gate replays a sample
            # (its rewind+get_next would inflate the count otherwise).
            num_calibration_samples = getattr(reader, "num_samples", None)

            # 3. int8 parity gate: int8-ONNX vs FP32-ONNX on dim-1 sums.
            if verify:
                self._verify_int8_parity(fp32_path, tmp_out, reader, parity_atol, parity_rtol)
        except Exception:
            # A failed int8 chain must not leave a partial graph or delete a
            # pre-existing good export at the target path.
            _drop_onnx_artifacts(tmp_out, fp32_path)
            raise

        # 4. Quantization metadata (written before the atomic publish).
        from autovision.export.quantization import (
            Int8QuantizationConfig,
            _int8_metadata_payload,
        )

        config = quantization_config or Int8QuantizationConfig()
        _write_onnx_metadata(
            tmp_out,
            {
                "int8_quantization": _int8_metadata_payload(config, num_calibration_samples),
                "int8_parity_atol": str(parity_atol),
                "int8_parity_rtol": str(parity_rtol),
            },
        )
        _move_onnx_artifacts(tmp_out, final_out)

        # 5. FP32 reference bookkeeping (with the dynamo .data sidecar).
        if keep_fp32_reference:
            kept_fp32 = final_out.parent / f"{final_out.stem}.fp32.onnx"
            _move_onnx_artifacts(fp32_path, kept_fp32)
            logger.info("Kept FP32 reference graph at '%s'.", kept_fp32)
        else:
            _drop_onnx_artifacts(fp32_path)
        logger.info("Exported int8 ONNX model to '%s'.", final_out)
        _maybe_write_postprocess_spec(self, final_out)
        return final_out

    def _verify_int8_parity(
        self,
        fp32_path: Path,
        int8_path: Path,
        reader: Any,
        parity_atol: float,
        parity_rtol: float,
    ) -> None:
        import onnx
        import onnxruntime as ort

        onnx.checker.check_model(str(int8_path), full_check=True)
        # The reader was consumed by quantize_static; rewind it (supported by
        # Int8CalibrationReader) to replay the first calibration sample.
        rewind = getattr(reader, "rewind", None)
        if rewind is not None:
            rewind()
        sample = reader.get_next()
        if sample is None:
            raise ValueError(
                "int8 parity gate has no calibration sample to replay; the "
                "calibration reader must support replaying samples (use "
                "Int8CalibrationReader) or provide at least two passes."
            )
        sessions = {}
        for label, path in (("fp32", fp32_path), ("int8", int8_path)):
            sessions[label] = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
        fp32_outputs = sessions["fp32"].run(None, sample)
        int8_outputs = sessions["int8"].run(None, sample)
        for index, (fp32_out, int8_out) in enumerate(zip(fp32_outputs, int8_outputs)):
            if fp32_out.shape != int8_out.shape:
                raise AssertionError(
                    f"int8 parity gate: output {index} shape mismatch "
                    f"{int8_out.shape} vs FP32 {fp32_out.shape}."
                )
            fp32_sum = torch.as_tensor(fp32_out).float().sum(dim=1)
            int8_sum = torch.as_tensor(int8_out).float().sum(dim=1)
            torch.testing.assert_close(
                int8_sum, fp32_sum,
                atol=parity_atol, rtol=parity_rtol, equal_nan=True,
                check_device=False, check_dtype=False,
            )

    def export_tensorrt(
        self,
        out: Path | str,
        *,
        precision: Literal["auto", "fp32", "fp16", "int8"] = "fp32",
        calibration_data: Iterable[dict[str, torch.Tensor]] | None = None,
        calibration_reader: Any | None = None,
        quantization_config: Any | None = None,
        keep_int8_onnx: bool = False,
        min_batch_size: int = 1,
        opt_batch_size: int = 1,
        max_batch_size: int = 1,
        fp32_attention_scores: bool = False,
        strongly_typed: bool = False,
        onnx_args: dict[str, Any] | None = None,
        verbose: bool = False,
        debug: bool = False,
        verify: bool = True,
    ) -> Path:
        """Build a TensorRT engine from this model (design §12.3).

        Requires the ``tensorrt`` package (not a framework dependency) and a
        CUDA-capable machine for the engine build. The intermediate ONNX is
        produced by :meth:`export_onnx` — see
        :func:`autovision.export.tensorrt_helpers.export_onnx_to_tensorrt`
        for the engine-build semantics.

        ``precision="int8"`` (roadmap D4) quantizes first — the QDQ ONNX
        from :meth:`export_onnx_int8` (its int8-vs-fp32 parity gate runs
        here) is staged and built into a strongly-typed engine, so the
        graph's Q/DQ nodes carry the precision contract; ``calibration_data``
        / ``calibration_reader`` are required. ``calibration_data`` accepts
        an image path/directory as well as sample dicts.
        ``keep_int8_onnx=True`` keeps the SANITIZED QDQ graph (the
        TensorRT-side int32 fold mutates it in place).

        ``verify=True`` runs a seeded eager-vs-engine parity check after the
        build (final tensors, precision-appropriate tolerances); a mismatch
        deletes the engine and raises. Set ``verify=False`` only for
        throwaway debug builds.
        """
        from autovision.export.tensorrt_helpers import export_onnx_to_tensorrt

        def _verify_engine(engine_path: Path) -> None:
            """Eager-vs-engine parity gate (fail closed: delete the engine)."""
            if not verify:
                return
            from autovision.export.tensorrt_helpers import (
                TensorRTEngineRunner,
                verify_engine_outputs,
            )

            spec = self.model_input_spec
            parameter = next(self.parameters(), None)
            example_device = (
                parameter.device if parameter is not None else torch.device("cpu")
            )
            sample = next(
                iter(
                    spec.example_inputs(
                        batch_size=min_batch_size, device=example_device
                    ).values()
                )
            )
            generator = torch.Generator().manual_seed(0)
            compare = torch.rand(
                sample.shape, generator=generator, dtype=sample.dtype
            ).to(sample.device)
            resolved_precision = precision
            if resolved_precision == "auto":
                # The engine may be fp16 (explicit request or an fp16 model
                # at build time); use the fp16 band for auto builds.
                resolved_precision = "fp16"
            atol, rtol = {
                "fp32": (5e-3, 1e-2),
                "fp16": (5e-2, 5e-2),
                "int8": (2.0, 0.5),
            }[resolved_precision]
            with torch.no_grad():
                expected = self.forward_backend(compare)
            runner_device = (
                str(compare.device)
                if compare.device.type == "cuda"
                else "cuda"
            )
            runner = TensorRTEngineRunner(engine_path, device=runner_device)
            actual = runner.run(compare.cpu())
            try:
                # Detection outputs can permute tied queries between
                # backends; compare the FINAL boxes/scores (label + IoU
                # matching) instead of raw query order. This is the
                # deployment question and avoids false failures when two
                # anchors tie under fp16.
                used_e2e = False
                image_size = getattr(self, "image_size", None)
                postprocess = getattr(self, "postprocess", None)
                if (
                    resolved_precision in ("fp32", "fp16")
                    and image_size is not None
                    and callable(postprocess)
                ):
                    size = int(image_size[0] if isinstance(image_size, (tuple, list)) else image_size)
                    metadata = [
                        {"orig_w": size, "orig_h": size}
                        for _ in range(int(compare.shape[0]))
                    ]
                    try:
                        reference_records = postprocess(
                            expected, metadata, threshold=0.0
                        )
                        engine_records = postprocess(
                            actual, metadata, threshold=0.0
                        )
                    except (AutoVisionError, TypeError, AttributeError, KeyError):
                        pass
                    else:
                        if (
                            reference_records
                            and isinstance(reference_records[0], dict)
                            and "bboxes" in reference_records[0]
                        ):
                            from autovision.export.e2e_parity import (
                                compare_detection_records,
                            )

                            report = compare_detection_records(
                                reference_records,
                                engine_records,
                                box_tolerance_px={
                                    "fp32": 1.0,
                                    "fp16": 1.0,
                                    "int8": 5.0,
                                }[resolved_precision],
                                score_tolerance={
                                    "fp32": 1e-2,
                                    "fp16": 5e-2,
                                    "int8": 0.5,
                                }[resolved_precision],
                            )
                            if not report.passed:
                                raise RuntimeError(
                                    "TensorRT end-to-end parity failed: "
                                    f"{report.to_dict()}"
                                )
                            used_e2e = True
                if not used_e2e:
                    if resolved_precision == "int8":
                        # int8 accepts a looser tensor-level contract; the
                        # full-AP regression gate lives in measure_int8_accuracy.
                        verify_engine_outputs(
                            engine_path,
                            expected,
                            compare.cpu(),
                            atol=atol,
                            rtol=rtol,
                        )
                    else:
                        verifier = getattr(self, "verify_onnx_export_outputs", None)
                        if callable(verifier):
                            verifier(expected, actual)
                        else:
                            verify_engine_outputs(
                                engine_path,
                                expected,
                                compare.cpu(),
                                atol=atol,
                                rtol=rtol,
                            )
            except Exception:
                Path(engine_path).unlink(missing_ok=True)
                raise

        # The intermediate ONNX is not the delivered artifact: only the
        # engine gets a postprocess sidecar.
        effective_onnx_args = {**(onnx_args or {}), "write_spec": False}

        if precision == "int8":
            if calibration_data is None and calibration_reader is None:
                raise ValueError(
                    "precision='int8' requires calibration: pass "
                    "calibration_data (iterable of {input_name: Tensor}) or "
                    "calibration_reader."
                )
            int8_onnx = Path(out).with_suffix(".int8.onnx")

            def _drop(path: Path) -> None:
                for candidate in (path, path.with_suffix(path.suffix + ".data")):
                    candidate.unlink(missing_ok=True)

            try:
                self.export_onnx_int8(
                    int8_onnx,
                    calibration_data=calibration_data,
                    calibration_reader=calibration_reader,
                    quantization_config=quantization_config,
                    verify=True,
                )
            except Exception:
                # The QDQ chain stages an fp32 reference (and the dynamo
                # exporter external-data sidecar) next to the target; failed
                # gates must not leave 100MB-class litter behind.
                _drop(int8_onnx.with_suffix(".fp32.onnx"))
                _drop(int8_onnx)
                raise
            try:
                engine = export_onnx_to_tensorrt(
                    export_onnx_fn=self.export_onnx,
                    out=out,
                    precision="int8",
                    model_dtype=next(self.parameters()).dtype,
                    onnx_args=effective_onnx_args,
                    min_batch_size=min_batch_size,
                    opt_batch_size=opt_batch_size,
                    max_batch_size=max_batch_size,
                    fp32_attention_scores=fp32_attention_scores,
                    strongly_typed=True,
                    verbose=verbose,
                    debug=debug,
                    pre_staged_onnx=int8_onnx,
                )
            finally:
                _drop(int8_onnx.with_suffix(".fp32.onnx"))
                if not keep_int8_onnx:
                    _drop(int8_onnx)
            _verify_engine(engine)
            _maybe_write_postprocess_spec(self, engine)
            if not keep_int8_onnx:
                # export_onnx_int8 wrote the staged ONNX sidecar; the engine
                # carries its own, so do not leave a dangling contract.
                Path(str(int8_onnx) + ".postprocess.json").unlink(missing_ok=True)
            return engine

        engine = export_onnx_to_tensorrt(
            export_onnx_fn=self.export_onnx,
            out=out,
            precision=precision,
            model_dtype=next(self.parameters()).dtype,
            onnx_args=effective_onnx_args,
            min_batch_size=min_batch_size,
            opt_batch_size=opt_batch_size,
            max_batch_size=max_batch_size,
            fp32_attention_scores=fp32_attention_scores,
            strongly_typed=strongly_typed,
            verbose=verbose,
            debug=debug,
        )
        _verify_engine(engine)
        _maybe_write_postprocess_spec(self, engine)
        return engine

    def export_openvino(
        self,
        out: "Path | str",
        *,
        precision: Literal["fp32", "fp16"] = "fp32",
        device: str = "CPU",
        onnx_args: dict[str, Any] | None = None,
        verify: bool = True,
        keep_onnx: bool = False,
    ) -> Path:
        """Export an OpenVINO IR (roadmap D3).

        ``out`` is the ``.xml`` path or a directory (``<dir>/model.xml``).
        The IR converts from this model's own ONNX export (its fp32/fp16
        gates run first), and with ``verify`` the COMPILED OpenVINO model is
        compared against torch on the example input via
        ``verify_onnx_export_outputs`` before the artifact is accepted.
        Requires the ``openvino`` package (``pip install
        'autovision[openvino]'``).
        """
        from autovision.export.openvino_helpers import export_onnx_to_openvino

        xml_path = export_onnx_to_openvino(
            out=out,
            export_onnx_fn=self.export_onnx,
            model=self,
            precision=precision,
            device=device,
            onnx_args={**(onnx_args or {}), "write_spec": False},
            verify=verify,
            keep_onnx=keep_onnx,
        )
        _maybe_write_postprocess_spec(self, xml_path)
        return xml_path


def _maybe_write_postprocess_spec(network: Any, artifact: Path) -> None:
    """Write the decode contract only for task families that have one.

    Segmentation / pose / depth / classification exports must keep working;
    they simply have no DETR top-k contract to serialize. The explicit
    ``postprocess_spec_family`` marker distinguishes the DETR detection task
    models from other heads that also carry ``image_size``/``num_classes``
    or ``num_top_queries`` (OBB / instance segmentation / PicoDet).
    """
    family = getattr(network, "postprocess_spec_family", None)
    if family is None:
        logger.debug("No postprocess spec for %s.", type(network).__name__)
        return
    from autovision.export.postprocess import write_postprocess_spec

    try:
        write_postprocess_spec(network, artifact, family=family)
    except NotImplementedError:
        logger.debug("No postprocess spec for %s.", type(network).__name__)


def _repair_value_info(out: Path) -> None:
    import onnx

    model = onnx.load(str(out))
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    onnx.save(model, str(out))


def _write_onnx_metadata(out: Path, metadata: dict[str, str]) -> None:
    import onnx

    model = onnx.load(str(out))
    existing = {prop.key: prop.value for prop in model.metadata_props}
    existing.update(metadata)
    del model.metadata_props[:]
    for key, value in existing.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(out))


def _remove_duplicate_casts(model) -> None:
    """Restore SSA form after FP16 conversion: the ORT float16 converter can
    emit several Cast nodes sharing one output name; keep the first."""
    import copy

    graph = model.graph
    seen_outputs: set[str] = set()
    survivors = []
    for node in graph.node:
        if node.output and node.output[0] in seen_outputs:
            continue  # duplicate-name node; its output name stays served
        seen_outputs.update(node.output)
        survivors.append(copy.deepcopy(node))
    if len(survivors) != len(graph.node):
        graph.ClearField("node")
        for node in survivors:
            graph.node.append(node)


def _fix_topological_order(model) -> None:
    """Kahn topological sort of graph nodes (FP16 cast insertion can leave
    nodes out of order, which onnx.checker rejects)."""
    import collections
    import copy

    graph = model.graph
    nodes = list(graph.node)
    produced_by: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for output in node.output:
            produced_by[output] = index
    dependents: dict[int, list[int]] = collections.defaultdict(list)
    indegree: dict[int, int] = {}
    for index, node in enumerate(nodes):
        deps = {
            produced_by[tensor]
            for tensor in node.input
            if tensor in produced_by and produced_by[tensor] != index
        }
        indegree[index] = len(deps)
        for dep in deps:
            dependents[dep].append(index)

    ready = collections.deque(sorted(i for i, count in indegree.items() if count == 0))
    order: list[int] = []
    while ready:
        index = ready.popleft()
        order.append(index)
        for dependent in sorted(dependents.get(index, ())):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    # Cycles (rare) keep their original relative order at the end.
    order.extend(sorted(set(range(len(nodes))) - set(order)))

    # Deep-copy BEFORE clearing: protobuf list views are invalidated by
    # ClearField.
    ordered_nodes = [copy.deepcopy(nodes[index]) for index in order]
    graph.ClearField("node")
    for node in ordered_nodes:
        graph.node.append(node)
