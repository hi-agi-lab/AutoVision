# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""int8 post-training quantization (PTQ) for exported ONNX models.

Design (§12.2 extension): the FP32 ONNX produced by the existing dynamo
pipeline is post-processed with ``onnxruntime.quantization.quantize_static``
into a QDQ int8 graph. Calibration samples are provided as an iterable of
``{input_name: torch.Tensor}`` dicts aligned with the task model's
``ModelInputSpec``; a convenience factory builds them from an image folder
through the model's own ``preprocess_image``/``preprocess_batch``.

Parity gate: int8-ONNX vs FP32-ONNX outputs are compared on dim-1 sums
(loose tolerances — int8 activation quantization is expected to shift
logits; the gate only catches structural quantization breakage).

onnxruntime is an optional dependency: this module has no top-level ORT
import (``Int8CalibrationReader`` satisfies ORT's structural
``CalibrationDataReader`` ABC via its ``get_next`` subclasshook).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch

__all__ = [
    "Int8CalibrationReader",
    "Int8QuantizationConfig",
    "image_folder_calibration_reader",
    "quantize_onnx_int8",
    "suggest_first_layer_nodes",
]

#: Default ops to quantize. LayerNorm/Softmax stay FP32 (numerically hot,
#: mirroring the FP16 ``fp32_onnx_op_types`` policy).
DEFAULT_OP_TYPES_TO_QUANTIZE = ("Conv", "Gemm", "MatMul")

_CALIBRATE_METHODS = ("MinMax", "Entropy", "Percentile", "Distribution")
_QUANT_TYPES = ("QInt8", "QUInt8")


class Int8CalibrationReader:
    """Iterable adapter: ``{name: torch.Tensor}`` dicts -> ORT calibration.

    Tensors are lazily converted to numpy. Input-name alignment is checked
    on the first sample (against ``expected_input_names`` when provided).
    Satisfies ``onnxruntime.quantization.CalibrationDataReader`` structurally
    (its ABC subclasshook only requires a callable ``get_next``).
    """

    def __init__(
        self,
        samples: Iterable[dict[str, torch.Tensor]],
        *,
        expected_input_names: Iterable[str] | None = None,
        factory: Callable[[], Iterator[dict[str, torch.Tensor]]] | None = None,
    ) -> None:
        self._samples = samples
        self._factory = factory
        self._iterator: Iterator[dict[str, torch.Tensor]] | None = None
        self._expected_names = (
            sorted(expected_input_names) if expected_input_names is not None else None
        )
        self.num_samples = 0

    def _check_alignment(self, sample: dict[str, torch.Tensor]) -> None:
        if self._expected_names is not None and sorted(sample) != self._expected_names:
            raise ValueError(
                f"Calibration sample input names {sorted(sample)} do not match "
                f"the model inputs {self._expected_names}."
            )

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iterator is None:
            self._iterator = iter(self._samples)
        for sample in self._iterator:
            if not isinstance(sample, dict):
                raise TypeError(
                    f"Calibration samples must be dicts of name -> Tensor, got {type(sample)}."
                )
            self._check_alignment(sample)
            self.num_samples += 1
            return {name: tensor.detach().cpu().numpy() for name, tensor in sample.items()}
        return None

    def rewind(self) -> None:
        """Restart iteration from the beginning (the int8 parity gate replays
        the first sample after ``quantize_static`` consumed the reader).

        One-shot iterators (e.g. the image-folder generator) are rebuilt via
        ``factory`` — without it, rewind on an exhausted generator is a no-op
        and the parity gate will report missing samples.
        """
        self._iterator = None
        if self._factory is not None and hasattr(self._samples, "__next__"):
            self._samples = self._factory()


@dataclass
class Int8QuantizationConfig:
    """Static-quantization settings for ORT ``quantize_static``.

    ``activation_type``/``weight_type`` are ``"QInt8"``/``"QUInt8"`` and
    ``calibrate_method`` is ``"MinMax"``/``"Entropy"``/``"Percentile"``/
    ``"Distribution"`` (strings so the config is constructible without
    onnxruntime installed).
    """

    per_channel: bool = True
    activation_type: str = "QInt8"
    weight_type: str = "QInt8"
    calibrate_method: str = "MinMax"
    op_types_to_quantize: tuple[str, ...] = DEFAULT_OP_TYPES_TO_QUANTIZE
    nodes_to_exclude: tuple[str, ...] = ()
    extra_options: dict[str, Any] = field(default_factory=dict)
    #: Symmetric activation quantization (zero_point = 0). ON by default:
    #: TensorRT's Q/DQ parser rejects non-zero zero points outright
    #: (``buildZeroPointTensor``), so an asymmetric QDQ graph is an
    #: ORT-only artifact; symmetric activations are consumed by both
    #: runtimes (and are the standard TRT int8 recipe). Set to False for
    #: an ORT-only asymmetric graph.
    activation_symmetric: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_channel": self.per_channel,
            "activation_type": self.activation_type,
            "weight_type": self.weight_type,
            "calibrate_method": self.calibrate_method,
            "op_types_to_quantize": list(self.op_types_to_quantize),
            "nodes_to_exclude": list(self.nodes_to_exclude),
            "activation_symmetric": self.activation_symmetric,
            "extra_options": {k: str(v) for k, v in self.extra_options.items()},
        }

    def _validate(self) -> None:
        if self.calibrate_method not in _CALIBRATE_METHODS:
            raise ValueError(
                f"calibrate_method must be one of {list(_CALIBRATE_METHODS)}, "
                f"got '{self.calibrate_method}'"
            )
        for name in ("activation_type", "weight_type"):
            value = getattr(self, name)
            if value not in _QUANT_TYPES:
                raise ValueError(
                    f"{name} must be one of {list(_QUANT_TYPES)}, got '{value}'"
                )


def _resolve_ort_quantization():
    """Import onnxruntime.quantization lazily with install guidance."""
    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as error:
        raise ImportError(
            "onnxruntime is required for int8 PTQ but is not installed. "
            "Install it with `pip install onnxruntime` (CPU) or "
            "`pip install onnxruntime-gpu`."
        ) from error
    quant_types = {"QInt8": QuantType.QInt8, "QUInt8": QuantType.QUInt8}
    calibrate_methods = {
        "MinMax": CalibrationMethod.MinMax,
        "Entropy": CalibrationMethod.Entropy,
        "Percentile": CalibrationMethod.Percentile,
        "Distribution": CalibrationMethod.Distribution,
    }
    return quantize_static, QuantFormat.QDQ, quant_types, calibrate_methods


def resolve_calibration_samples(model: Any, calibration_data: Any) -> Any:
    """Calibration samples for the int8 chains.

    ``None`` passes through (no int8); a path (image file/dir) is expanded
    through the model's own ``preprocess_image`` + ``preprocess_batch`` (the
    exact tensors the exported graph consumes, matching
    ``image_folder_calibration_reader``); anything iterable is materialized.
    """
    if calibration_data is None:
        return None
    from collections.abc import Iterable
    from pathlib import Path

    from autovision.data.file_helpers import list_image_filenames_from_dir

    if isinstance(calibration_data, (str, Path)):
        root = Path(calibration_data)
        paths = (
            list(list_image_filenames_from_dir(root)) if root.is_dir() else [root]
        )
        if not paths:
            raise ValueError(f"No calibration images found at '{root}'.")
        input_name = next(iter(model.model_input_spec.input_specs))
        return [
            {
                input_name: model.preprocess_batch(
                    model.preprocess_image(str(path))[0].unsqueeze(0)
                )
            }
            for path in paths
        ]
    if isinstance(calibration_data, Iterable):
        return list(calibration_data)
    raise TypeError(
        "calibration_data must be a path or an iterable of {input_name: Tensor}."
    )


def quantize_onnx_int8(
    model_input: Path | str,
    model_output: Path | str,
    calibration_data_reader: Any,
    config: Int8QuantizationConfig | None = None,
) -> Path:
    """Quantize an FP32 ONNX model to QDQ int8 (static PTQ with calibration)."""
    import onnx

    config = config or Int8QuantizationConfig()
    config._validate()
    quantize_static, qdq_format, quant_types, calibrate_methods = (
        _resolve_ort_quantization()
    )
    model_output = Path(model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=str(model_input),
        model_output=str(model_output),
        calibration_data_reader=calibration_data_reader,
        quant_format=qdq_format,
        op_types_to_quantize=list(config.op_types_to_quantize),
        per_channel=config.per_channel,
        activation_type=quant_types[config.activation_type],
        weight_type=quant_types[config.weight_type],
        nodes_to_exclude=list(config.nodes_to_exclude),
        calibrate_method=calibrate_methods[config.calibrate_method],
        extra_options={
            "ActivationSymmetric": bool(config.activation_symmetric),
            **dict(config.extra_options),
        },
    )
    onnx.checker.check_model(str(model_output), full_check=True)
    return model_output


def suggest_first_layer_nodes(onnx_path: Path | str, limit: int = 2) -> list[str]:
    """Leading ``Conv`` node names of an ONNX graph (first-layer-fp32 recipe).

    ``quantize_static``'s ``nodes_to_exclude`` matches the FP32 graph's NODE
    names. Some stems (e.g. a ``Conv+Relu+MaxPool`` fusion) have no TensorRT
    int8 tactic, and the supported recipe is keeping the first layers in
    fp32::

        from autovision.export.quantization import (
            Int8QuantizationConfig, suggest_first_layer_nodes,
        )
        nodes = suggest_first_layer_nodes("model.int8.fp32.onnx")
        config = Int8QuantizationConfig(nodes_to_exclude=tuple(nodes))

    Enumeration order is the graph's topological order, so the names are the
    stem layers. Best-effort by design: a missing file or an unreadable graph
    returns ``[]`` — callers use the list for the supported first-layer fp32
    recipe and for build-failure hints, never as a gate input.
    """
    path = Path(onnx_path)
    if not path.is_file():
        return []
    try:
        import onnx

        graph = onnx.load(str(path), load_external_data=False).graph
    except Exception:  # noqa: BLE001 - hint-only helper, never raises
        return []
    return [
        node.name
        for node in graph.node
        if node.op_type == "Conv" and node.name
    ][: max(0, int(limit))]


def image_folder_calibration_reader(
    model: Any,
    image_dir: Path | str,
    *,
    batch_size: int = 1,
    max_samples: int | None = None,
) -> Int8CalibrationReader:
    """Build calibration samples from an image folder via the model's own
    preprocessing pipeline (``preprocess_image`` -> ``preprocess_batch``),
    so calibration sees exactly the distribution the deployed model gets.
    """
    from autovision.data.file_helpers import list_image_filenames_from_dir

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    paths = list(list_image_filenames_from_dir(Path(image_dir)))
    if not paths:
        raise ValueError(f"No images found in '{image_dir}'.")
    if max_samples is not None:
        paths = paths[:max_samples]

    input_names = list(model.model_input_spec.input_specs)

    def _generate() -> Iterator[dict[str, torch.Tensor]]:
        batch: list[torch.Tensor] = []
        for path in paths:
            tensor, _ = model.preprocess_image(path)
            batch.append(tensor)
            while len(batch) >= batch_size:
                yield _make_sample(model, input_names, batch[:batch_size])
                batch = batch[batch_size:]
        if batch:
            yield _make_sample(model, input_names, batch)

    return Int8CalibrationReader(
        _generate(),
        expected_input_names=input_names,
        factory=_generate,
    )


def _make_sample(
    model: Any,
    input_names: list[str],
    batch: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    if len(input_names) != 1:
        raise ValueError(
            "image_folder_calibration_reader only supports single-input task "
            f"models; model declares inputs {input_names}."
        )
    processed = model.preprocess_batch(torch.stack(batch))
    if not isinstance(processed, torch.Tensor):
        raise TypeError(
            f"preprocess_batch must return a Tensor, got {type(processed)}."
        )
    return {input_names[0]: processed}


def _int8_metadata_payload(
    config: Int8QuantizationConfig, num_calibration_samples: int | None
) -> str:
    payload = config.to_dict()
    payload["num_calibration_samples"] = num_calibration_samples
    payload["calibration"] = "onnxruntime.quantization.quantize_static (QDQ)"
    return json.dumps(payload, sort_keys=True)
