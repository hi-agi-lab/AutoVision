# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""TensorRT engine construction for the ONNX export pipeline.

The public entry point, ``export_onnx_to_tensorrt``, runs a small staged
pipeline: after argument validation it (a) materializes the intermediate
ONNX file by delegating to the caller-supplied exporter, (b) parses that
file into a TensorRT network, (c) applies the precision policy, and (d)
serializes the engine next to the ONNX file. ``TensorRTEngineRunner`` is
the execution side — it loads a serialized engine and runs batches; the
reference inference runtime and the benchmark harness both drive engines
through it.

Engine-build contract (design doc §12.3):

- Networks are created EXPLICIT_BATCH, plus STRONGLY_TYPED when requested.
  A strongly-typed network derives every precision from the ONNX graph, so
  per-layer overrides are unavailable — hence ``strongly_typed`` and
  ``fp32_attention_scores`` are rejected together up front.
- The ``images`` input must exist and must have static spatial extents;
  only the batch axis is made dynamic, via the optimization profile
  (min/opt/max).
- The workspace is capped at 1 GiB and TF32 is cleared so mixed-precision
  paths keep stable numerics.
- ``precision="fp16"`` (or ``"auto"`` with a float16 model) sets the FP16
  builder flag together with OBEY_PRECISION_CONSTRAINTS, so per-layer FP32
  overrides are honored rather than merely preferred.
- ``fp32_attention_scores`` pins Softmax layers — and the MatMul layers
  feeding them — to FP32, avoiding the known FP16 overflow in attention
  scores.

TensorRT is an optional dependency: with the package missing, a
``ModuleNotFoundError`` carrying install guidance is raised. Every argument
check runs before that import so misconfiguration is reportable anywhere.

Annotation convention: the public signature quotes its annotations, which
Python therefore never evaluates at runtime; quoting keeps that explicit and
the annotation text free-form.
"""

import os
import pathlib
import typing as ty
from importlib import import_module
from logging import getLogger

import torch

__all__ = ["TensorRTEngineRunner", "export_onnx_to_tensorrt"]

LOG = getLogger("autovision")

#: Shown when the TensorRT package itself is missing.
_INSTALL_GUIDANCE = (
    "TensorRT is required, but is not installed. Install TensorRT for your "
    "system following NVIDIA's guide: "
    "https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html "
    "(on CUDA 12.x systems, `pip install tensorrt-cu12` often works)."
)

#: Suffix of the intermediate ONNX file staged next to the engine.
_ONNX_SUFFIX = ".onnx"

#: Builder workspace memory-pool ceiling: 1 GiB.
_WORKSPACE_CEILING = 1 << 30

#: Name of the mandatory image tensor produced by ``export_onnx``.
_IMAGE_INPUT_NAME = "images"

#: Builder flags cleared whenever the installed TensorRT defines them.
#: TF32 can silently change numerics in mixed-precision paths.
_FLAGS_DROPPED_IF_PRESENT = ("TF32",)

#: Precision-constraint flags, most authoritative first.
_PRECISION_CONSTRAINT_LADDER = (
    "OBEY_PRECISION_CONSTRAINTS",
    "PREFER_PRECISION_CONSTRAINTS",
)

#: Layer-name fragments the attention path pins to FP32.
_ATTENTION_LAYER_MARKS = ("Softmax", "MatMul")


def _announce(message, *fields):
    """Emit a progress line for the engine-build pipeline."""
    LOG.info(message, *fields)


def _reject_inverted_batch_bounds(lower: int, middle: int, upper: int):
    """Require ``lower <= middle <= upper`` for the batch profile."""
    if lower <= middle <= upper:
        return
    raise _batch_order_error(lower, middle, upper)


def _batch_order_error(lower, middle, upper):
    return ValueError(
        "Batch sizes must satisfy: min_batch_size <= opt_batch_size <= "
        f"max_batch_size (got {lower}/{middle}/{upper})."
    )


def _reject_conflicting_precision_modes(strongly_typed: bool, pin_attention: bool):
    """Strong typing and attention-score pinning cannot coexist."""
    if strongly_typed and pin_attention:
        raise _mutually_exclusive_error()


def _mutually_exclusive_error():
    return ValueError(
        "strongly_typed and fp32_attention_scores are mutually exclusive: "
        "a strongly-typed network derives all precisions from the ONNX "
        "graph and does not allow the per-layer overrides used by "
        "fp32_attention_scores."
    )


def _missing_tensorrt_error():
    return ModuleNotFoundError(_INSTALL_GUIDANCE)


def _import_tensorrt():
    """Load the ``tensorrt`` module, or raise with install guidance."""
    try:
        return import_module("tensorrt")
    except ModuleNotFoundError as failure:  # pragma: no cover - trt absent in CI
        raise _missing_tensorrt_error() from failure


def export_onnx_to_tensorrt(
    *,
    out: "str | pathlib.Path",
    export_onnx_fn: "ty.Callable[..., ty.Any]",
    debug: "bool" = False,
    precision: "ty.Literal['auto', 'fp32', 'fp16', 'int8']" = "fp32",
    pre_staged_onnx: "pathlib.Path | None" = None,
    strongly_typed: "bool" = False,
    min_batch_size: "int" = 1,
    opt_batch_size: "int" = 1,
    max_batch_size: "int" = 1,
    onnx_args: "dict[str, ty.Any] | None" = None,
    fp32_attention_scores: "bool" = False,
    verbose: "bool" = False,
    model_dtype: "torch.dtype" = torch.float32,
) -> pathlib.Path:
    """Build a TensorRT engine at ``out`` from the model's ONNX export.

    ``export_onnx_fn`` (typically ``model.export_onnx``) materializes the
    intermediate ONNX file; the entries of ``onnx_args`` are forwarded to
    it as keywords. Unless ``onnx_args["out"]`` points elsewhere, that file
    is written next to the engine (same stem, ``.onnx`` suffix).

    ``precision="int8"`` takes a **pre-quantized QDQ ONNX** (roadmap D4):
    pass ``pre_staged_onnx`` with the graph produced by
    ``export_onnx_int8`` (its own int8-vs-fp32 parity gate has already
    run) and this builds a strongly-typed engine whose precisions come
    from the graph's QuantizeLinear/DequantizeLinear nodes — the
    TensorRT 11 route; older builds also get the ``INT8`` builder flag.
    Dynamic spatial extents stay rejected (TRT needs static H/W).
    """
    _reject_inverted_batch_bounds(min_batch_size, opt_batch_size, max_batch_size)
    _reject_conflicting_precision_modes(strongly_typed, fp32_attention_scores)
    tensorrt = _import_tensorrt()

    # TensorRT 11 removed the precision builder flags (FP16 and the
    # precision-constraint ladder): there the graph dtype is the only
    # precision control, so an fp16 request stages the intermediate ONNX in
    # fp16 and derives the engine from a strongly-typed network. The
    # exporter's fp32 op blacklist (Softmax, attention MatMuls) keeps the
    # overflow-sensitive ops in fp32 inside that fp16 graph.
    if precision == "int8":
        if pre_staged_onnx is None:
            raise ValueError(
                "precision='int8' needs a pre-quantized QDQ ONNX: pass "
                "pre_staged_onnx (produced by export_onnx_int8, whose "
                "int8-vs-fp32 parity gate runs there)."
            )
        # Quantization-aware graph: the Q/DQ nodes ARE the precision
        # contract, so the network is strongly typed and no calibration
        # cache is involved (TRT consumes explicit QDQ).
        strongly_typed = True
    graph_drives_precision = _prefers_half_precision(
        precision, model_dtype
    ) and not hasattr(tensorrt.BuilderFlag, "FP16")
    if graph_drives_precision:
        strongly_typed = True
        onnx_args = {**(onnx_args or {}), "precision": "fp16"}
        _announce(
            "This TensorRT build has no precision builder flags: staging an "
            "fp16 ONNX and building a strongly-typed engine."
        )

    engine_path = pathlib.Path(os.fspath(out))
    severity_name = "VERBOSE" if verbose else "INFO"
    log_sink = tensorrt.Logger(getattr(tensorrt.Logger, severity_name))
    engine_builder = tensorrt.Builder(log_sink)
    trt_network = engine_builder.create_network(
        _network_creation_flags(tensorrt, strongly_typed=strongly_typed)
    )
    onnx_parser = tensorrt.OnnxParser(trt_network, log_sink)

    staged_onnx = (
        pathlib.Path(pre_staged_onnx)
        if pre_staged_onnx is not None
        else _stage_intermediate_onnx(
            export_onnx_fn, destination=out, user_options=onnx_args
        )
    )
    if precision == "int8":
        _sanitize_qdq_for_tensorrt(staged_onnx)
    if debug:  # pragma: no cover - requires onnx + trt debugging workflow
        _expose_intermediate_tensors(staged_onnx)

    _ingest_onnx(onnx_parser, staged_onnx)

    if fp32_attention_scores:
        if graph_drives_precision:
            _announce(
                "fp32 attention scores are carried by the fp16 ONNX op "
                "blacklist in this build (strongly-typed engine)."
            )
        else:
            _pin_attention_precision(tensorrt, trt_network)

    _require_static_spatial_input(trt_network)
    build_config = _prepare_build_config(tensorrt, engine_builder)
    if precision == "int8":
        int8_flag = _first_available_flag(
            tensorrt.BuilderFlag, ("INT8",)
        )
        if int8_flag is not None:
            # TensorRT <= 10 still gates INT8 behind the builder flag; TRT 11
            # removed the precision flags (the QDQ graph is the contract).
            build_config.set_flag(int8_flag)
            _announce("INT8 builder flag set (legacy TensorRT precision flag).")
    _apply_precision_policy(
        tensorrt,
        engine_builder,
        build_config,
        precision=precision,
        model_dtype=model_dtype,
        strongly_typed=strongly_typed,
    )
    if debug:  # pragma: no cover - requires trt debugging workflow
        _switch_on_engine_introspection(tensorrt, build_config)

    _register_batch_profiles(
        trt_network,
        engine_builder,
        build_config,
        min_batch_size=min_batch_size,
        opt_batch_size=opt_batch_size,
        max_batch_size=max_batch_size,
    )

    _announce("Building TensorRT engine (this can take several minutes)...")
    engine_blob = None
    build_error: Exception | None = None
    try:
        engine_blob = engine_builder.build_serialized_network(trt_network, build_config)
    except Exception as exc:  # the build API returns None OR raises by version
        build_error = exc
    if engine_blob is None:
        # ``build_serialized_network`` returns None on failure (it does not
        # raise), so the remediation must ride this path too — the earlier
        # try/except-only form never fired for the real failure mode.
        hint = ""
        if precision == "int8":
            hint = (
                " A common int8 cause is a quantized node without a TensorRT "
                "kernel — the log names TRT's FUSED node (e.g. a stem "
                "Conv+Relu+MaxPool fusion), which need not match any ONNX "
                "node name. The supported recipe is keeping the first layers "
                "in fp32: pass a quantization_config whose nodes_to_exclude "
                "lists the fp32 graph's leading Conv node names."
            )
            # Auto-suggest the concrete recipe. The fp32 staging graph is
            # usually already deleted by export_onnx_int8's cleanup
            # (keep_fp32_reference defaults off), so the QDQ graph in
            # ``pre_staged_onnx`` is the primary source — quantize_static
            # preserves Conv node names there.
            from autovision.export.quantization import suggest_first_layer_nodes

            candidates = suggest_first_layer_nodes(
                pre_staged_onnx.with_suffix(".fp32.onnx")
            ) or suggest_first_layer_nodes(pre_staged_onnx)
            if candidates:
                hint += (
                    " In this build the leading Conv nodes are "
                    + repr(candidates)
                    + " — try quantization_config="
                    + repr({"nodes_to_exclude": candidates})
                    + " (TensorRT fusion names may still differ; iterate on "
                    "the build log if needed)."
                )
        raise RuntimeError(
            "TensorRT engine building failed."
            + (f" ({type(build_error).__name__}: {build_error})" if build_error else "")
            + hint
        ) from build_error
    _require_engine_blob(engine_blob)

    os.makedirs(engine_path.parent, exist_ok=True)
    engine_path.write_bytes(engine_blob)
    _announce("TensorRT engine saved to '%s'.", engine_path)
    return engine_path


def _prefers_half_precision(precision, model_dtype) -> bool:
    """FP16 is requested explicitly, or automatic for a float16 model."""
    if precision == "fp16":
        return True
    return precision == "auto" and model_dtype is torch.float16


def _first_available_flag(builder_flags, candidates):
    """First builder flag the installed TensorRT defines, if any."""
    for candidate in candidates:
        flag = getattr(builder_flags, candidate, None)
        if flag is not None:
            return flag
    return None


def _enable_half_precision(tensorrt, engine_builder, build_config):
    """Set the FP16 flag when the platform accelerates it.

    ``IBuilder.platform_has_fast_fp16`` was removed in TensorRT 11 (the
    platform query left the builder API); a missing attribute now means
    "unknown", so the flag is set and the builder decides.
    """
    platform_has_fast_fp16 = getattr(engine_builder, "platform_has_fast_fp16", None)
    if platform_has_fast_fp16 is False:
        LOG.warning("FP16 not supported on this platform; using FP32.")
        return
    builder_flags = tensorrt.BuilderFlag
    fp16_flag = getattr(builder_flags, "FP16", None)
    if fp16_flag is None:
        LOG.warning("This TensorRT build defines no FP16 builder flag; using FP32.")
        return
    build_config.set_flag(fp16_flag)
    constraint_flag = _first_available_flag(
        builder_flags, _PRECISION_CONSTRAINT_LADDER
    )
    if constraint_flag is not None:
        # Ensure layer.precision overrides are respected, not merely preferred.
        build_config.set_flag(constraint_flag)
    _announce("FP16 optimization enabled.")


def _apply_precision_policy(
    tensorrt,
    engine_builder,
    build_config,
    *,
    precision,
    model_dtype,
    strongly_typed,
):
    """Resolve the precision knob into builder flags."""
    if strongly_typed:
        # Precisions come from the ONNX graph's dtypes and Cast nodes; the
        # FP16/precision-constraint flags do not apply to such networks.
        _announce("Strongly-typed network: precision taken from the ONNX graph.")
        return
    if not _prefers_half_precision(precision, model_dtype):
        return
    _enable_half_precision(tensorrt, engine_builder, build_config)


def _input_tensors(trt_network):
    """All input tensors of the network, in declaration order."""
    return map(trt_network.get_input, range(trt_network.num_inputs))


def _require_static_spatial_input(trt_network):
    """Locate the ``images`` input; reject dynamic spatial extents."""
    image_inputs = [
        candidate
        for candidate in _input_tensors(trt_network)
        if candidate.name == _IMAGE_INPUT_NAME
    ]
    if not image_inputs:
        raise _missing_images_input_error()
    detected = image_inputs[0].shape
    batch_extent, channel_count, image_height, image_width = map(int, detected)
    if -1 in (image_height, image_width):
        raise _dynamic_spatial_error()
    _announce(
        "Detected input shape: (N, %d, %d, %d)",
        channel_count, image_height, image_width,
    )


def _batch_bounds(min_batch_size, opt_batch_size, max_batch_size, static_tail):
    """min/opt/max extents with the batch axis swapped in."""
    profile_sides = (
        ("min", min_batch_size),
        ("opt", opt_batch_size),
        ("max", max_batch_size),
    )
    return {side: (size, *static_tail) for side, size in profile_sides}


def _dynamic_shape_plans(trt_network):
    """Yield ``(input name, static non-batch tail)`` per profiled input.

    The ``images`` input always gets a profile; every other input only when
    its full shape is dynamic, in which case all non-batch extents must be
    static.
    """
    for entry in _input_tensors(trt_network):
        extents = [int(extent) for extent in entry.shape]
        batch_extent = extents.pop(0)
        tail = tuple(extents)
        if entry.name == _IMAGE_INPUT_NAME:
            yield entry.name, tail
        elif tail.count(-1):
            raise _dynamic_tail_error(entry.name)
        elif batch_extent == -1:
            yield entry.name, tail


def _register_batch_profiles(
    trt_network,
    engine_builder,
    build_config,
    *,
    min_batch_size,
    opt_batch_size,
    max_batch_size,
):
    """Give every eligible input a batch-only dynamic shape profile."""
    plans = dict(_dynamic_shape_plans(trt_network))
    shape_profile = engine_builder.create_optimization_profile()
    for input_name, static_tail in plans.items():
        shape_profile.set_shape(
            input_name,
            **_batch_bounds(
                min_batch_size, opt_batch_size, max_batch_size, static_tail
            ),
        )
    build_config.add_optimization_profile(shape_profile)


def _network_creation_flags(tensorrt, *, strongly_typed: bool) -> int:
    """EXPLICIT_BATCH, plus STRONGLY_TYPED for strongly-typed networks.

    ``EXPLICIT_BATCH`` was folded into the default network mode and removed
    from the enum in TensorRT 11; it is requested only when the installed
    build still defines it.
    """
    definition_flags = tensorrt.NetworkDefinitionCreationFlag
    requested = []
    if hasattr(definition_flags, "EXPLICIT_BATCH"):
        requested.append(definition_flags.EXPLICIT_BATCH)
    if strongly_typed:
        requested.append(definition_flags.STRONGLY_TYPED)
    return sum(1 << int(bit) for bit in requested)


def _stage_intermediate_onnx(export_onnx_fn, *, destination, user_options):
    """Produce the intermediate ONNX file and return its path.

    ``user_options`` are forwarded to ``export_onnx_fn``; a ``"verify"``
    entry defaults to False, and an ``"out"`` entry overrides the default
    location (engine stem with an ``.onnx`` suffix).
    """
    options = dict(user_options or {})
    staged = pathlib.Path(
        options.pop("out", pathlib.Path(destination).with_suffix(_ONNX_SUFFIX))
    )
    if "verify" not in options:
        options["verify"] = False
    os.makedirs(staged.parent, exist_ok=True)
    export_onnx_fn(out=staged, **options)
    return staged


def _sanitize_qdq_for_tensorrt(staged_onnx: pathlib.Path) -> int:
    """Fold int32 ``DequantizeLinear`` constants for the TensorRT parser.

    onnxruntime's QDQ quantization emits ``DequantizeLinear`` nodes whose
    input is an int32 CONSTANT (quantized biases/weights). TensorRT 11
    accepts only FP8/FP4/Int4/Int8/UInt8 inputs there and refuses to parse
    such a graph. Folding the constant back to fp32 (``(q - zp) * scale``,
    broadcast along the node's ``axis``) keeps the bias fp32 in the engine —
    the standard recipe — and drops the offending nodes. Returns the number
    of folded nodes.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(staged_onnx))
    initializers = {init.name: init for init in model.graph.initializer}

    folded = 0
    for node in list(model.graph.node):
        if node.op_type != "DequantizeLinear" or not node.input:
            continue
        source = initializers.get(node.input[0])
        if source is None or source.data_type != onnx.TensorProto.INT32:
            continue
        scale_init = initializers.get(node.input[1]) if len(node.input) > 1 else None
        if scale_init is None:
            continue  # non-constant scale: cannot fold, leave it (parser will complain loudly)
        values = numpy_helper.to_array(source).astype(np.float32)
        scale = numpy_helper.to_array(scale_init).astype(np.float32)
        zero_point = np.zeros(1, dtype=np.float32)
        if len(node.input) > 2 and node.input[2]:
            zp_init = initializers.get(node.input[2])
            if zp_init is not None:
                zero_point = numpy_helper.to_array(zp_init).astype(np.float32)
        axis = next(
            (int(attr.i) for attr in node.attribute if attr.name == "axis"), 1
        )
        if scale.ndim == 1 and values.ndim > 1 and scale.size == values.shape[axis]:
            shape = [1] * values.ndim
            shape[axis] = -1
            scale = scale.reshape(shape)
            zero_point = zero_point.reshape(shape)
        folded_values = (values - zero_point) * scale
        model.graph.initializer.append(
            numpy_helper.from_array(folded_values.astype(np.float32), name=node.output[0])
        )
        model.graph.node.remove(node)
        for input_name in node.input[1:]:
            _drop_initializer_if_unused(model, initializers, input_name)
        for input_name in (node.input[0],):
            # The quantized constant itself is now unused.
            _drop_initializer_if_unused(model, initializers, input_name)
        folded += 1
    if folded:
        onnx.save(model, str(staged_onnx))
        _announce(
            "Folded %d int32 DequantizeLinear constant(s) to fp32 for the "
            "TensorRT Q/DQ parser.",
            folded,
        )
    return folded


def _drop_initializer_if_unused(model, initializers: dict, name: str) -> None:
    """Remove ``name``'s initializer when no node consumes it any more."""
    if not name or name not in initializers:
        return
    if any(name in node.input for node in model.graph.node):
        return
    if any(name in output.name for output in model.graph.output):
        return
    for index, init in enumerate(model.graph.initializer):
        if init.name == name:
            del model.graph.initializer[index]
            initializers.pop(name, None)
            return


def _ingest_onnx(onnx_parser, staged_onnx):
    """Parse the staged ONNX file into the TensorRT network."""
    _announce("Loading ONNX file from %s", staged_onnx)
    if onnx_parser.parse(staged_onnx.read_bytes()):
        return
    _report_parser_errors(onnx_parser)
    raise _unparseable_onnx_error(staged_onnx)


def _report_parser_errors(onnx_parser):
    """Log every parse error collected by the ONNX parser."""
    for complaint in (
        onnx_parser.get_error(position)
        for position in range(onnx_parser.num_errors)
    ):
        LOG.error("%s", complaint)


def _prepare_build_config(tensorrt, engine_builder):
    """Create the builder config: TF32 off, 1 GiB workspace."""
    build_config = engine_builder.create_builder_config()
    for flag_name in _FLAGS_DROPPED_IF_PRESENT:
        flag = getattr(tensorrt.BuilderFlag, flag_name, None)
        if flag is not None:
            build_config.clear_flag(flag)
    memory_pools = tensorrt.MemoryPoolType
    build_config.set_memory_pool_limit(memory_pools.WORKSPACE, _WORKSPACE_CEILING)
    return build_config


def _switch_on_engine_introspection(tensorrt, build_config):
    """Make debug engines introspectable (per-layer value dumps)."""
    verbosity_levels = tensorrt.ProfilingVerbosity
    build_config.set_flag(tensorrt.BuilderFlag.DEBUG)
    build_config.profiling_verbosity = verbosity_levels.DETAILED


def _require_engine_blob(engine_blob):
    """Guard: TensorRT signals a failed engine build by returning None."""
    if engine_blob is not None:
        return
    raise _aborted_build_error()


def _missing_images_input_error():
    return RuntimeError(
        "Could not find the 'images' input in the ONNX network; the "
        "TensorRT path expects the standard export input spec."
    )


def _unparseable_onnx_error(staged_onnx):
    return RuntimeError(f"Failed to parse ONNX file '{staged_onnx}' with TensorRT.")


def _aborted_build_error():
    return RuntimeError("TensorRT engine building failed.")


def _dynamic_tail_error(input_name):
    return ValueError(
        f"Dynamic non-batch dimensions are not supported for input '{input_name}'."
    )


def _dynamic_spatial_error():
    return ValueError(
        "Dynamic image height and width are not supported by the "
        "TensorRT export; export the ONNX model with a static image size."
    )


def _attention_mark(layer_name):
    """Classify a layer by the attention-relevant mark in its name."""
    for mark in _ATTENTION_LAYER_MARKS:
        if mark in layer_name:
            return mark
    return None


def _pin_softmax_layer(trt_layer, float_precision, boundary_tensors):
    """Pin a Softmax layer, plus any boundary outputs it has, to FP32."""
    trt_layer.precision = float_precision
    for produced in _each_output(trt_layer):
        if produced in boundary_tensors:
            produced.dtype = float_precision


def _pin_score_matmul(trt_layer, float_precision, boundary_tensors, softmax_feeds) -> bool:
    """Pin a MatMul feeding a Softmax input; report whether it was pinned."""
    for produced in _each_output(trt_layer):
        if produced is None or produced.name not in softmax_feeds:
            continue
        trt_layer.precision = float_precision
        if produced in boundary_tensors:
            produced.dtype = float_precision
        return True
    return False


def _softmax_feeds(trt_network):
    """Names of the tensors consumed by Softmax layers (input 0)."""
    operands = []
    for trt_layer in _each_layer(trt_network):
        if "Softmax" in trt_layer.name:
            operands.append(trt_layer.get_input(0))
    return {operand.name for operand in operands if operand is not None}


def _each_layer(trt_network):
    """Iterate the network's layers in definition order."""
    return map(trt_network.get_layer, range(trt_network.num_layers))


def _each_output(trt_layer):
    """Iterate a layer's output tensors."""
    return map(trt_layer.get_output, range(trt_layer.num_outputs))


def _boundary_tensors(trt_network) -> set:
    """Every network-level input and output tensor."""
    accessors = (
        ("get_input", trt_network.num_inputs),
        ("get_output", trt_network.num_outputs),
    )
    return {
        getattr(trt_network, accessor)(slot)
        for accessor, arity in accessors
        for slot in range(arity)
    }


def _pin_attention_precision(tensorrt, trt_network):
    """Force FP32 on Softmax layers and on the MatMuls feeding them.

    Attention scores are the known FP16 overflow hotspot; pinning their
    producers keeps ``fp16`` builds numerically safe.
    """
    float_precision = getattr(tensorrt.DataType, "FLOAT")
    boundary_tensors = _boundary_tensors(trt_network)
    softmax_feeds = _softmax_feeds(trt_network)
    pinned_counts = dict.fromkeys(_ATTENTION_LAYER_MARKS, 0)
    for trt_layer in _each_layer(trt_network):
        layer_mark = _attention_mark(trt_layer.name)
        if not layer_mark:
            continue
        if layer_mark == "MatMul":
            was_pinned = _pin_score_matmul(
                trt_layer, float_precision, boundary_tensors, softmax_feeds
            )
        else:
            _pin_softmax_layer(trt_layer, float_precision, boundary_tensors)
            was_pinned = True
        if was_pinned:
            pinned_counts[layer_mark] += 1
    _announce(
        "Forced FP32 on %d Softmax layers and %d attention-score MatMul layers.",
        pinned_counts["Softmax"],
        pinned_counts["MatMul"],
    )


def _expose_intermediate_tensors(staged_onnx):
    """Mark every intermediate tensor as a graph output (debug aid).

    Lets tools such as polygraphy dump per-layer values from a debug
    engine; ONNX shape inference supplies each tensor's type.
    """
    import onnx

    describe_tensor = onnx.helper.make_tensor_value_info
    staged_location = os.fspath(staged_onnx)
    raw_model = onnx.load(staged_location)
    shape_tools = onnx.shape_inference
    enriched = shape_tools.infer_shapes(raw_model)
    proto = enriched.graph
    known_types = {
        info.name: info
        for info in (*proto.value_info, *proto.input, *proto.output)
    }
    exposed_names = {info.name for info in proto.output}
    pending = []
    for producer in proto.node:
        for emitted in producer.output:
            fresh = emitted not in exposed_names and emitted in known_types
            if emitted and fresh:
                described = known_types[emitted]
                pending.append(
                    describe_tensor(emitted, described.type.tensor_type)
                )
                exposed_names.add(emitted)
    proto.output.extend(pending)
    onnx.save(enriched, os.fspath(staged_onnx))


# ── engine execution ─────────────────────────────────────────────────────


class TensorRTEngineRunner:
    """Load a serialized TensorRT engine and execute batches on CUDA.

    Shapes are set per call (the engine's batch profile covers min/opt/max);
    outputs are allocated per call and the asynchronous execution is
    synchronized on a torch CUDA stream before the buffers are returned.
    Inputs are cast to the engine's declared input dtype (builder-flag FP16
    engines keep FP32 IO bindings, strongly-typed ones follow the graph).
    """

    def __init__(self, engine_path, *, device="cuda"):
        import numpy

        self._trt = _import_tensorrt()
        self.engine_path = pathlib.Path(engine_path)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                f"TensorRTEngineRunner requires a CUDA device, got '{device}'."
            )
        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"TensorRT engine does not exist: '{self.engine_path}'."
            )

        trt_logger = self._trt.Logger(self._trt.Logger.WARNING)
        with open(self.engine_path, "rb") as handle:
            runtime = self._trt.Runtime(trt_logger)
            engine = runtime.deserialize_cuda_engine(handle.read())
        if engine is None:
            raise ValueError(
                f"Could not deserialize the TensorRT engine at "
                f"'{self.engine_path}' (version mismatch?)."
            )
        self.engine = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            raise ValueError("Could not create a TensorRT execution context.")

        self.output_names: list[str] = []
        self.input_name = ""
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(name) == self._trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)
        if not self.input_name:
            raise ValueError("The TensorRT engine exposes no input tensor.")
        nptype = self._trt.nptype(engine.get_tensor_dtype(self.input_name))
        self.input_dtype = torch.from_numpy(numpy.zeros(1, dtype=nptype)).dtype
        self.stream = torch.cuda.Stream(device=self.device)

    def run(self, images: torch.Tensor) -> list[torch.Tensor]:
        import numpy

        with torch.cuda.device(self.device):
            images = images.contiguous().to(
                device=self.device, dtype=self.input_dtype
            )
            if self.context.set_input_shape(
                self.input_name, list(images.shape)
            ) is False:
                raise RuntimeError(
                    "TensorRT rejected the input shape "
                    f"{list(images.shape)} for '{self.input_name}' (is the "
                    "batch outside the engine's optimization profile?)."
                )
            if self.context.set_tensor_address(
                self.input_name, images.data_ptr()
            ) is False:
                raise RuntimeError(
                    f"TensorRT rejected the input address for "
                    f"'{self.input_name}'."
                )
            outputs: list[torch.Tensor] = []
            for name in self.output_names:
                shape = list(self.context.get_tensor_shape(name))
                if any(dimension < 0 for dimension in shape):
                    raise RuntimeError(
                        f"TensorRT output '{name}' still has dynamic shape "
                        f"{shape} after setting the input; the engine profile "
                        "does not resolve it."
                    )
                dtype = self._trt.nptype(self.engine.get_tensor_dtype(name))
                torch_dtype = torch.from_numpy(numpy.zeros(1, dtype=dtype)).dtype
                buffer = torch.empty(shape, dtype=torch_dtype, device=self.device)
                if self.context.set_tensor_address(
                    name, buffer.data_ptr()
                ) is False:
                    raise RuntimeError(
                        f"TensorRT rejected the output address for '{name}'."
                    )
                outputs.append(buffer)
            if self.context.execute_async_v3(self.stream.cuda_stream) is False:
                raise RuntimeError(
                    "TensorRT execution failed (execute_async_v3 returned False)."
                )
            self.stream.synchronize()
        return outputs


def _flatten_tensors(value: ty.Any) -> list[torch.Tensor]:
    """Flatten nested dict/list/tuple/dataclass outputs into tensor order."""
    import dataclasses

    if isinstance(value, torch.Tensor):
        return [value]
    if dataclasses.is_dataclass(value):
        return _flatten_tensors(
            [getattr(value, field.name) for field in dataclasses.fields(value)]
        )
    if isinstance(value, dict):
        flattened: list[torch.Tensor] = []
        # Preserve insertion order: it mirrors the forward's return order,
        # which is the order the exporter assigns to graph outputs.
        for item in value.values():
            flattened.extend(_flatten_tensors(item))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_tensors(item))
        return flattened
    return []


def verify_engine_outputs(
    engine_path: str | pathlib.Path,
    expected: ty.Any,
    inputs: torch.Tensor,
    *,
    device: str = "cuda",
    atol: float = 5e-3,
    rtol: float = 1e-2,
) -> None:
    """Fail-closed parity check: serialized engine vs eager reference.

    ``expected`` is the eager model output (tensor or nested
    dict/list/tuple). Raising here is intentional: an engine that builds but
    computes different values must never be published.
    """
    runner = TensorRTEngineRunner(engine_path, device=device)
    with torch.no_grad():
        actual = runner.run(inputs)
    expected_tensors = _flatten_tensors(expected)
    if not expected_tensors:
        raise RuntimeError(
            "TensorRT parity check received no reference tensors; pass the "
            "eager model's tensor output."
        )
    if len(actual) != len(expected_tensors):
        raise RuntimeError(
            "TensorRT parity check output-count mismatch: engine produced "
            f"{len(actual)}, reference produced {len(expected_tensors)}."
        )
    for index, (reference, produced) in enumerate(
        zip(expected_tensors, actual, strict=True)
    ):
        reference = reference.detach().to(produced.device, produced.dtype)
        if tuple(reference.shape) != tuple(produced.shape):
            raise RuntimeError(
                f"TensorRT parity check output {index} shape mismatch: "
                f"engine {tuple(produced.shape)} vs reference "
                f"{tuple(reference.shape)}."
            )
        try:
            torch.testing.assert_close(
                produced, reference, atol=atol, rtol=rtol
            )
        except AssertionError as error:
            raise RuntimeError(
                f"TensorRT parity check failed for output {index}: {error}"
            ) from error
