# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Versioned detection post-processing (roadmap 90d, phase 2).

The exported graph stops at ``(logits, boxes)``; decode/NMS and the
original-frame mapping live here as a small, serializable contract. A
device team gets ``<artifact>.postprocess.json`` next to the model and can
reproduce the final boxes without importing PyTorch or AutoVision.

The reference decoder is NumPy-only. It is deliberately separate from the
task models' ``postprocess`` so parity is a real comparison, not the same
Python function dressed up twice. ``schema_version`` is frozen at 1: adding
fields is compatible; changing semantics requires a new version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from autovision.errors import AutoVisionError

__all__ = [
    "NMSConfig",
    "PostprocessSpec",
    "PreprocessSpec",
    "build_postprocess_spec",
    "decode_detections_numpy",
    "load_postprocess_spec",
    "write_postprocess_spec",
]

SCHEMA_VERSION = 1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class NMSConfig:
    """Greedy NMS contract (unused by the DETR family, kept for others)."""

    iou_threshold: float = 0.5
    max_detections: int = 300
    class_aware: bool = True


@dataclass(frozen=True)
class PreprocessSpec:
    """What happened before the forward pass, so device code can match it."""

    resize: str = "resize"  # "resize" | "letterbox"
    mean: tuple[float, ...] = IMAGENET_MEAN
    std: tuple[float, ...] = IMAGENET_STD
    pad_value: float = 0.0


@dataclass(frozen=True)
class PostprocessSpec:
    """One model's decode contract (JSON round-trippable)."""

    schema_version: int
    family: str
    input_size: tuple[int, int]
    num_classes: int
    num_queries: int
    top_k: int
    score_threshold: float = 0.0
    box_format: str = "cxcywh_normalized"
    nms: NMSConfig | None = None
    preprocess: PreprocessSpec | None = None
    classes: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = ("logits", "boxes")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "family": self.family,
            "input_size": list(self.input_size),
            "num_classes": self.num_classes,
            "num_queries": self.num_queries,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "box_format": self.box_format,
            "nms": None,
            "preprocess": None,
            "classes": list(self.classes),
            "outputs": list(self.outputs),
        }
        if self.nms is not None:
            payload["nms"] = {
                "iou_threshold": self.nms.iou_threshold,
                "max_detections": self.nms.max_detections,
                "class_aware": self.nms.class_aware,
            }
        if self.preprocess is not None:
            payload["preprocess"] = {
                "resize": self.preprocess.resize,
                "mean": list(self.preprocess.mean),
                "std": list(self.preprocess.std),
                "pad_value": self.preprocess.pad_value,
            }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PostprocessSpec":
        version = int(payload.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise AutoVisionError(
                f"Unsupported postprocess schema_version {version}; expected "
                f"{SCHEMA_VERSION}."
            )
        nms_payload = payload.get("nms")
        nms = NMSConfig(**nms_payload) if isinstance(nms_payload, Mapping) else None
        prep_payload = payload.get("preprocess")
        preprocess = (
            PreprocessSpec(
                resize=str(prep_payload.get("resize", "resize")),
                mean=tuple(float(v) for v in prep_payload.get("mean", IMAGENET_MEAN)),
                std=tuple(float(v) for v in prep_payload.get("std", IMAGENET_STD)),
                pad_value=float(prep_payload.get("pad_value", 0.0)),
            )
            if isinstance(prep_payload, Mapping)
            else None
        )
        input_size = payload.get("input_size")
        if not isinstance(input_size, (list, tuple)) or len(input_size) != 2:
            raise AutoVisionError("Postprocess spec needs input_size=[h, w].")
        return cls(
            schema_version=version,
            family=str(payload["family"]),
            input_size=(int(input_size[0]), int(input_size[1])),
            num_classes=int(payload.get("num_classes", 0)),
            num_queries=int(payload.get("num_queries", 0)),
            top_k=int(payload.get("top_k", 0)),
            score_threshold=float(payload.get("score_threshold", 0.0)),
            box_format=str(payload.get("box_format", "cxcywh_normalized")),
            nms=nms,
            preprocess=preprocess,
            classes=tuple(str(name) for name in payload.get("classes", ())),
            outputs=tuple(str(name) for name in payload.get("outputs", ("logits", "boxes"))),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path


def load_postprocess_spec(path: str | Path) -> PostprocessSpec:
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AutoVisionError(f"Cannot read postprocess spec '{path}': {error}") from error
    if not isinstance(payload, Mapping):
        raise AutoVisionError(f"Postprocess spec '{path}' must be a JSON object.")
    return PostprocessSpec.from_dict(payload)


def _image_size(network: Any) -> tuple[int, int]:
    size = getattr(network, "image_size", None)
    if size is None:
        spec = getattr(network, "model_input_spec", None)
        try:
            shape = spec.input_specs["images"].shape
            size = tuple(shape[-2:])
        except Exception as error:  # noqa: BLE001 - fall through to the error
            raise AutoVisionError(
                "Cannot infer the model input size for the postprocess spec."
            ) from error
    if len(size) != 2:
        raise AutoVisionError(f"Model image_size must be (h, w), got {size!r}.")
    return int(size[0]), int(size[1])


def _normalize(network: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    normalize = getattr(network, "normalize", None)
    if normalize is None:
        return IMAGENET_MEAN, IMAGENET_STD
    return tuple(float(v) for v in normalize.mean), tuple(float(v) for v in normalize.std)


def _class_names(network: Any) -> tuple[str, ...]:
    names = getattr(network, "class_names", None)
    if isinstance(names, Mapping):
        return tuple(str(names[key]) for key in sorted(names))
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return tuple(str(name) for name in names)
    return ()


def build_postprocess_spec(
    network: Any,
    *,
    family: str = "detr",
    score_threshold: float = 0.0,
) -> PostprocessSpec:
    """Build the spec for a loaded task network.

    DETR-family decoders (RT-DETR / DETR / RF-DETR) share sigmoid + top-k.
    Any other family is rejected explicitly: a wrong decode contract is
    worse than no contract.
    """
    if family != "detr":
        if family != "picodet":
            raise NotImplementedError(
                f"Postprocess spec for family '{family}' is not implemented; "
                "add a decoder + tests before emitting a contract."
            )
    image_size = _image_size(network)
    num_classes = int(getattr(network, "num_classes", 0))
    num_queries = int(getattr(network, "num_queries", 0))
    top_k = int(getattr(network, "num_top_queries", num_queries or 300))
    if num_queries == 0:
        # RT-DETR / PicoDet keep the query count on their decoder/head; the
        # decode width is the contract that matters.
        num_queries = top_k
    mean, std = _normalize(network)
    resize = str(getattr(network, "preprocess_mode", "resize"))
    nms = None
    if family == "picodet":
        nms = NMSConfig(
            iou_threshold=float(getattr(network, "nms_iou_threshold", 0.6)),
            max_detections=top_k,
            class_aware=True,
        )
    return PostprocessSpec(
        schema_version=SCHEMA_VERSION,
        family=family,
        input_size=image_size,
        num_classes=num_classes,
        num_queries=num_queries,
        top_k=top_k,
        score_threshold=float(score_threshold),
        nms=nms,
        preprocess=PreprocessSpec(resize=resize, mean=mean, std=std),
        classes=_class_names(network),
    )


def write_postprocess_spec(
    network: Any,
    artifact: str | Path,
    *,
    score_threshold: float = 0.0,
    family: str | None = None,
) -> Path:
    """Write ``<artifact>.postprocess.json`` next to an export.

    ``family=None`` infers the contract from the task model's
    ``postprocess_spec_family`` marker (DETR / PicoDet).
    """
    resolved = family or getattr(network, "postprocess_spec_family", "detr")
    spec = build_postprocess_spec(
        network, family=resolved, score_threshold=score_threshold
    )
    return spec.save(Path(str(artifact) + ".postprocess.json"))


def _unpack(raw_outputs: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(raw_outputs, (tuple, list)) and len(raw_outputs) == 2:
        logits, boxes = raw_outputs
    elif isinstance(raw_outputs, Mapping):
        logits, boxes = raw_outputs["logits"], raw_outputs["boxes"]
    else:
        logits, boxes = raw_outputs.logits, raw_outputs.boxes
    logits_array = np.asarray(logits, dtype=np.float32)
    boxes_array = np.asarray(boxes, dtype=np.float32)
    if logits_array.ndim == 2:
        logits_array = logits_array[None, ...]
    if boxes_array.ndim == 2:
        boxes_array = boxes_array[None, ...]
    if logits_array.ndim != 3 or boxes_array.ndim != 3:
        raise AutoVisionError(
            "Expected logits (B, Q, C) and boxes (B, Q, 4); got "
            f"{logits_array.shape} and {boxes_array.shape}."
        )
    if logits_array.shape[:2] != boxes_array.shape[:2] or boxes_array.shape[-1] != 4:
        raise AutoVisionError(
            "logits/boxes query dimensions disagree: "
            f"{logits_array.shape} vs {boxes_array.shape}."
        )
    return logits_array, boxes_array


def _iou_vector(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(
        0.0, others[:, 3] - others[:, 1]
    )
    union = area + areas - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _class_offset_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    offset: float,
    iou_threshold: float,
    max_detections: int | None,
) -> np.ndarray:
    """Greedy NMS matching torchvision.ops.nms (+ class-offset semantics)."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    shifted = boxes + labels.astype(np.float32)[:, None] * float(offset)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if max_detections is not None and len(keep) >= max_detections:
            break
        if order.size == 1:
            break
        rest = order[1:]
        ious = _iou_vector(shifted[current], shifted[rest])
        order = rest[ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def decode_detections_numpy(
    raw_outputs: Any,
    metadata: Sequence[Mapping[str, Any]],
    spec: PostprocessSpec,
    *,
    threshold: float | None = None,
) -> list[dict[str, np.ndarray]]:
    """Reference decode: sigmoid + top-k + original-frame boxes.

    ``detr`` mirrors ``decode_detection_outputs``; ``picodet`` adds the
    class-offset NMS used by ``decode_pico_outputs`` (same offset, same
    descending-score suppression). Both use a strict ``>`` threshold.
    """
    if spec.family not in {"detr", "picodet"}:
        raise NotImplementedError(
            f"The NumPy reference decoder does not implement family "
            f"'{spec.family}'."
        )
    if spec.family == "detr" and spec.nms is not None:
        raise AutoVisionError("DETR postprocess specs must not carry NMS.")
    if spec.family == "picodet" and spec.nms is None:
        raise AutoVisionError("PicoDet postprocess specs require an NMS config.")
    if spec.box_format != "cxcywh_normalized":
        raise AutoVisionError(
            f"Unsupported box_format '{spec.box_format}'; expected "
            "'cxcywh_normalized'."
        )
    logits, boxes = _unpack(raw_outputs)
    if logits.shape[0] != len(metadata):
        raise AutoVisionError(
            f"Batch size {logits.shape[0]} does not match metadata "
            f"length {len(metadata)}."
        )
    cutoff = spec.score_threshold if threshold is None else float(threshold)
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    num_classes = scores.shape[2]
    flat = scores.reshape(scores.shape[0], -1)
    top_k = min(spec.top_k, flat.shape[1])
    # Stable descending sort: torch.topk returns sorted values, and stable
    # ordering keeps ties deterministic across runs.
    order = np.argsort(-flat, axis=1, kind="stable")[:, :top_k]
    top_scores = np.take_along_axis(flat, order, axis=1)
    classes = order % num_classes
    query_indices = order // num_classes

    results: list[dict[str, np.ndarray]] = []
    for batch_index, meta in enumerate(metadata):
        keep = top_scores[batch_index] > cutoff
        selected = boxes[batch_index, query_indices[batch_index]][keep]
        labels = classes[batch_index][keep].astype(np.int64)
        confidences = top_scores[batch_index][keep].astype(np.float32)
        width = float(meta["orig_w"])
        height = float(meta["orig_h"])
        cx, cy, bw, bh = selected.T
        xyxy = np.stack(
            [
                (cx - bw / 2.0) * width,
                (cy - bh / 2.0) * height,
                (cx + bw / 2.0) * width,
                (cy + bh / 2.0) * height,
            ],
            axis=-1,
        ).astype(np.float32)
        if spec.family == "picodet":
            nms = spec.nms
            offset = (
                float(max(width, height)) + 1.0
                if nms.class_aware
                else 0.0
            )
            keep_indices = _class_offset_nms(
                xyxy,
                confidences,
                labels,
                offset=offset,
                iou_threshold=nms.iou_threshold,
                max_detections=nms.max_detections,
            )
            xyxy = xyxy[keep_indices]
            labels = labels[keep_indices]
            confidences = confidences[keep_indices]
        results.append(
            {
                "labels": labels,
                "bboxes": xyxy,
                "scores": confidences,
            }
        )
    return results
