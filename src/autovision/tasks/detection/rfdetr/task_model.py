# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RF-DETR object detection task model (inference/export facing).

Per framework policy (see ``autovision.encoders.rfdetr``), the detector is
HuggingFace ``RfDetrForObjectDetection`` (transformers >= 5.9, gated at
runtime). The HF class carries the official training criterion (IA-BCE +
L1 + GIoU behind a Hungarian matcher) inside ``forward(labels=...)`` — the
task model forwards it instead of re-implementing it. Inference keeps the
AutoVision three-stage contract: ``forward`` returns raw
``(logits, boxes)`` and ``postprocess`` decodes sigmoid top-k to original
pixels via the shared DETR decode.
"""

from __future__ import annotations

import pathlib
from typing import Any

import torch
from torch import nn

from autovision.augment.transform_args import NormalizeArgs
from autovision.encoders.weight_verification import load_pretrained_verified
from autovision.export import ONNXExportMixin
from autovision.geometry.box_ops import (
    elementwise_siou,
)
from autovision.tasks.detection.rfdetr.config import (
    RFDETRSSLArgs,
    resolve_rfdetr_variant,
)
from autovision.tasks.detection.shared.decode import (
    decode_detection_outputs,
)
from autovision.tasks.detection.shared.ssl_contrastive import (
    SSLProjectionHead,
)
from autovision.tasks.task_model import TaskNetwork
from autovision.tasks.task_model_io import BaseModelOutput

__all__ = ["RFDETRObjectDetection", "RFDETRObjectDetectionOutput"]

# Reuse the SSL package's variant table (HF repo ids + decoder depths +
# backbone layouts — same checkpoints, verified supply chain).
from autovision.encoders.rfdetr.rfdetr_source import (  # noqa: E402
    _MODEL_INFO as _SSL_MODEL_INFO,
)
from autovision.encoders.rfdetr.rfdetr_source import (  # noqa: E402
    _support_error as _ssl_support_error,
)

#: Native pretrained-config geometry for hermetic builds, transcribed from
#: each variant's cached HF ``config.json`` (fields where the rf-detr
#: checkpoints differ from the transformers RfDetrConfig defaults AND that
#: shape parameters). Hermetic builds exist for exported-model rebuilds, so
#: a variant without recorded geometry is rejected instead of approximated.
_PRETRAINED_CONFIG_OVERRIDES: dict[str, dict[str, Any]] = {
    "rf-detr-nano": dict(
        decoder_layers=2,
        decoder_n_points=2,
        decoder_cross_attention_heads=16,
        backbone=dict(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            image_size=384,
        ),
    ),
    "rf-detr-small": dict(
        decoder_layers=3,
        decoder_n_points=2,
        decoder_cross_attention_heads=16,
        backbone=dict(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            image_size=512,
        ),
    ),
    "rf-detr-medium": dict(
        decoder_layers=4,
        decoder_n_points=2,
        decoder_cross_attention_heads=16,
        backbone=dict(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            image_size=576,
        ),
    ),
    "rf-detr-base": dict(
        decoder_layers=3,
        decoder_n_points=2,
        decoder_cross_attention_heads=16,
        backbone=dict(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            image_size=518,
        ),
    ),
    "rf-detr-large": dict(
        decoder_layers=4,
        decoder_n_points=2,
        decoder_cross_attention_heads=16,
        backbone=dict(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            image_size=704,
        ),
    ),
}


def _od_classes() -> dict[str, Any] | None:
    """Live detection of the RF-DETR detection classes in transformers."""
    try:
        import transformers
    except ImportError:
        return None
    config = getattr(transformers, "RfDetrConfig", None)
    od_model = getattr(transformers, "RfDetrForObjectDetection", None)
    backbone_config = getattr(transformers, "RfDetrDinov2Config", None)
    if None in (config, od_model, backbone_config):
        return None
    return {
        "config": config,
        "od_model": od_model,
        "backbone_config": backbone_config,
    }


class RFDETRObjectDetectionOutput(BaseModelOutput):
    """Exported graph outputs: class logits + normalized cxcywh boxes."""

    def __init__(self, logits: torch.Tensor, boxes: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.boxes = boxes


def _reinitialize_class_head(model: nn.Module, num_labels: int) -> None:
    """Resize the classification heads to ``num_labels`` (rf-detr semantics).

    Mirrors the official ``reinitialize_detection_head``: growing tiles the
    existing rows, shrinking truncates — the pretrained head is never
    randomly reset. Covers the decoder head and the per-group two-stage
    encoder heads.
    """

    def _retile(head: nn.Linear) -> nn.Linear:
        old_n = head.weight.shape[0]
        if old_n == num_labels:
            return head
        new = nn.Linear(head.in_features, num_labels, bias=head.bias is not None)
        with torch.no_grad():
            index = torch.arange(num_labels, device=head.weight.device) % old_n
            new.weight.copy_(head.weight.data[index])
            if head.bias is not None:
                new.bias.copy_(head.bias.data[index])
        return new

    model.class_embed = _retile(model.class_embed)
    enc_heads = getattr(model.model, "enc_out_class_embed", None)
    if enc_heads is not None:
        for index in range(len(enc_heads)):
            enc_heads[index] = _retile(enc_heads[index])


def _make_siou_box_loss_class() -> type:
    """Build the ``LwDetrImageLoss`` subclass with the SIoU box emission.

    Defined in a factory because the base class lives in an optional
    transformers module (the class body needs it at definition time).
    Overrides ONLY ``loss_boxes`` — L1, labels, cardinality and the
    Hungarian matcher (still GIoU-costed, design doc 15 semantics: matcher
    cost untouched) come from the parent. The emission is renamed
    ``loss_siou`` so the logs never dress SIoU up as GIoU.
    """
    from transformers.image_transforms import center_to_corners_format
    from transformers.loss.loss_lw_detr import LwDetrImageLoss

    class _SIoUBoxLoss(LwDetrImageLoss):
        def __init__(self, *args: Any, siou_theta: float = 4.0, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.siou_theta = siou_theta

        def loss_boxes(self, outputs, targets, indices, num_boxes):
            idx = self._get_source_permutation_idx(indices)
            source_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat(
                [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )
            loss_bbox = nn.functional.l1_loss(
                source_boxes, target_boxes, reduction="none"
            )
            return {
                "loss_bbox": loss_bbox.sum() / num_boxes,
                "loss_siou": elementwise_siou(
                    center_to_corners_format(source_boxes),
                    center_to_corners_format(target_boxes),
                    theta=self.siou_theta,
                ).sum()
                / num_boxes,
            }

    return _SIoUBoxLoss


#: transformers renamed the DINOv2 attention submodule between the 5.9
#: layout (this repo's rfdetr training overlay env: ``attention.attention.
#: query`` + ``attention.output.dense``) and 5.19 (``q_proj``/``k_proj``/
#: ``v_proj``/``o_proj``). Checkpoints trained under one layout do not load
#: into a model built with the other — HF's own ``from_pretrained`` carries
#: a conversion mapping, but raw ``load_state_dict`` (the export/checkpoint
#: path) does not.
_DINOV2_ATTENTION_RENAMES = (
    (".attention.attention.query.", ".attention.q_proj."),
    (".attention.attention.key.", ".attention.k_proj."),
    (".attention.attention.value.", ".attention.v_proj."),
    (".attention.output.dense.", ".attention.o_proj."),
)


def _remap_legacy_dinov2_attention_keys(
    state_dict: dict[str, Any], hf_model: nn.Module
) -> dict[str, Any]:
    """Rename 5.9-layout attention keys when the consumer is 5.19-layout.

    Fires only when the model expects the new names AND the checkpoint
    carries the old ones (bulk, model-level detection — attention weights
    never mix layouts within one checkpoint); otherwise the dict passes
    through untouched, so 5.19-trained checkpoints and already-matching
    loads are bit-identical no-ops.
    """
    expected = hf_model.state_dict().keys()
    if not any(".attention.q_proj." in key for key in expected):
        return state_dict
    if not any(".attention.attention.query." in key for key in state_dict):
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        for old, new in _DINOV2_ATTENTION_RENAMES:
            if old in key:
                key = key.replace(old, new)
                break
        remapped[key] = value
    return remapped


def _make_rfdetr_siou_detection_loss(siou_theta: float):
    """Bind ``siou_theta`` and return the SIoU variant of the HF detection loss.

    Mirrors ``LwDetrForObjectDetectionLoss`` (matcher + criterion + aux/enc
    weighting) one-for-one; only the box IoU term differs (design doc 15's
    ablation swap, applied across every branch — main, aux and encoder).
    Attached as the HF model's ``loss_function`` when
    ``box_regression_iou="siou"``.
    """

    def loss_fn(
        logits,
        labels,
        device,
        pred_boxes,
        config,
        outputs_class=None,
        outputs_coord=None,
        enc_outputs_class=None,
        enc_outputs_coord=None,
        **kwargs: Any,
    ):
        del kwargs  # same explicit signature as the HF call site
        from transformers.loss.loss_lw_detr import LwDetrHungarianMatcher

        matcher = LwDetrHungarianMatcher(
            class_cost=config.class_cost,
            bbox_cost=config.bbox_cost,
            giou_cost=config.giou_cost,
        )
        criterion = _make_siou_box_loss_class()(
            matcher=matcher,
            num_classes=config.num_labels,
            focal_alpha=config.focal_alpha,
            losses=["labels", "boxes", "cardinality"],
            group_detr=config.group_detr,
            siou_theta=siou_theta,
        )
        criterion.to(device)
        outputs_loss = {
            "logits": logits,
            "pred_boxes": pred_boxes,
            "enc_outputs": {
                "logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            },
        }
        auxiliary_outputs = None
        if config.auxiliary_loss and outputs_class is not None:
            auxiliary_outputs = [
                {"logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
            outputs_loss["auxiliary_outputs"] = auxiliary_outputs
        loss_dict = criterion(outputs_loss, labels)
        weight_dict = {
            "loss_ce": config.class_loss_coefficient,
            "loss_bbox": config.bbox_loss_coefficient,
            # doc 15 E0: SIoU inherits the GIoU loss weight (scale ratio ~0.95).
            "loss_siou": config.giou_loss_coefficient,
        }
        if config.auxiliary_loss:
            aux_weight_dict: dict = {}
            for i in range(config.decoder_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict.items()}
                )
            aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        loss = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict
        )
        return loss, loss_dict, auxiliary_outputs

    return loss_fn


class RFDETRObjectDetection(ONNXExportMixin, TaskNetwork):
    """RF-DETR detector (HF transformers) in the AutoVision task contract."""

    #: Export-side postprocess contract marker (see export/postprocess.py).
    postprocess_spec_family = "detr"

    def __init__(
        self,
        num_classes: int,
        model: str = "rf-detr-nano",
        image_size: tuple[int, int] = (384, 384),
        num_queries: int = 300,
        num_select: int = 300,
        load_weights: bool = False,
        box_regression_iou: str = "giou",
        siou_theta: float = 4.0,
        ssl: RFDETRSSLArgs | dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            init_args=locals(), ignore_args={"load_weights", "kwargs", "self"}
        )
        resolve_rfdetr_variant(model)
        if box_regression_iou not in ("giou", "siou"):
            raise ValueError(
                f"box_regression_iou must be 'giou' or 'siou', got "
                f"'{box_regression_iou}'."
            )
        self.box_regression_iou = box_regression_iou
        self.siou_theta = siou_theta
        # Per-object contrastive SSL branch (design doc 14, rfdetr port —
        # doc 18). ``None`` or mode="off" keeps the baseline contract: no
        # projection head is built (mode="off" + view="two_crop" is the
        # batch-matched baseline — the doubled data pipeline runs, the loss
        # does not; the command layer reads the train args' ssl field).
        if isinstance(ssl, dict):
            ssl = RFDETRSSLArgs(**ssl)
        self._ssl_args: RFDETRSSLArgs | None = ssl
        self.ssl_head: SSLProjectionHead | None = None
        # RAW projector taps accompany every active-head training forward —
        # the SIGReg leg consumes them when weighted (sigreg_coef>0), and
        # the unweighted statistic doubles as the always-on calibration
        # readout for choosing λ_sr (doc 18 §2).
        if model.endswith("-notpretrained"):
            raise ValueError(
                f"Unknown rfdetr detection model '{model}'. The '-notpretrained' "
                "suffix is a pretrain-package concept; set load_weights=False "
                "here instead."
            )
        self.num_classes = num_classes
        self.model_name = model
        self.image_size = tuple(image_size)
        self.num_top_queries = num_select
        self.normalize = NormalizeArgs()

        classes = _od_classes()
        if classes is None:
            raise _ssl_support_error(f"rfdetr/{model}")
        if load_weights:
            # Load the verified checkpoint with its native class count, then
            # tile/truncate the head — growing randomly-initialized heads is
            # exactly what the official loader avoids.
            hf_model = load_pretrained_verified(classes["od_model"], _repo_id(model))
        else:
            # Hermetic build — NO weight download. Exported-model rebuilds
            # land here (load_task_model drops load_weights), so the config
            # MUST reproduce the pretrained geometry exactly: the library
            # defaults differ from every rf-detr checkpoint in fields that
            # shape parameters (decoder_n_points, decoder_cross_attention_
            # heads, decoder_layers), and a wrong value fails the rebuild
            # with size mismatches. The overrides below are transcribed
            # from the cached HF config.json per variant; variants without
            # recorded geometry are rejected rather than silently
            # approximated.
            overrides = _PRETRAINED_CONFIG_OVERRIDES.get(model)
            if overrides is None:
                raise ValueError(
                    f"No recorded pretrained-config geometry for "
                    f"'{model}'; rfdetr hermetic builds (load_weights="
                    f"False) require it. Record the variant's HF config "
                    f"fields in _PRETRAINED_CONFIG_OVERRIDES."
                )
            info = _SSL_MODEL_INFO[model]
            backbone_overrides = overrides["backbone"]
            backbone_kwargs = dict(info["backbone"])
            backbone_kwargs["image_size"] = backbone_overrides["image_size"]
            backbone_config = classes["backbone_config"](
                hidden_size=backbone_overrides["hidden_size"],
                num_hidden_layers=backbone_overrides["num_hidden_layers"],
                num_attention_heads=backbone_overrides["num_attention_heads"],
                use_mask_token=True,
                **backbone_kwargs,
            )
            # The two-stage top-k selects num_queries proposals out of the
            # patch pool of the requested canvas ((canvas / patch)^2), so a
            # small hermetic canvas floors the query count below the 300
            # default (224px -> 196 proposals). Rebuilds at native-or-larger
            # canvases keep 300 and stay shape-exact with the training run.
            patch = int(backbone_kwargs["patch_size"])
            proposals = (self.image_size[0] // patch) * (
                self.image_size[1] // patch
            )
            config = classes["config"](
                backbone_config=backbone_config,
                num_labels=num_classes,
                num_queries=min(num_queries, proposals),
                decoder_layers=overrides["decoder_layers"],
                decoder_n_points=overrides["decoder_n_points"],
                decoder_cross_attention_heads=overrides[
                    "decoder_cross_attention_heads"
                ],
            )
            hf_model = classes["od_model"](config)
        _reinitialize_class_head(hf_model, num_classes)
        if box_regression_iou == "siou":
            # Design doc 15's ablation swap, RF-DETR edition: the HF model
            # resolves its criterion through the ``loss_function`` property
            # (LOSS_MAPPING -> LwDetrForObjectDetectionLoss); the setter
            # binds our SIoU mirror without touching HF internals. The
            # exported graph and inference path never call it.
            hf_model.loss_function = _make_rfdetr_siou_detection_loss(siou_theta)
        self._model = hf_model
        if self._ssl_args is not None and self._ssl_args.mode != "off":
            # Train-only projection head over the decoder's final-layer query
            # features (d_model); inference/export never touch it. mode="off"
            # (the batch-matched baseline) builds no head (doc 14 §3.3).
            self.ssl_head = SSLProjectionHead(
                hidden_dim=int(hf_model.config.d_model),
                projection_dim=self._ssl_args.projector_dim,
                kind=self._ssl_args.projector,
                dropout=self._ssl_args.projector_dropout,
            )

    # ── three-stage contract ────────────────────────────────────────────
    def preprocess_image(self, image: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        from torchvision.transforms.v2 import functional as TF

        from autovision.data.file_helpers import open_image

        if isinstance(image, (str, pathlib.Path)):
            tensor = open_image(image).float() / 255.0
        elif isinstance(image, torch.Tensor):
            tensor = image.float()
            if tensor.ndim == 3 and tensor.max() > 1.5:
                tensor = tensor / 255.0
        else:
            tensor = TF.pil_to_tensor(image).float() / 255.0
        _, height, width = tensor.shape
        resized = TF.resize(
            tensor,
            [self.image_size[0], self.image_size[1]],
            antialias=True,
        )
        return resized, {"orig_w": width, "orig_h": height}

    def preprocess_batch(self, batch: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.normalize.mean, device=batch.device).view(-1, 1, 1)
        std = torch.tensor(self.normalize.std, device=batch.device).view(-1, 1, 1)
        return (batch - mean) / std

    def _labels(
        self, targets: list[dict[str, torch.Tensor]] | None
    ) -> list[dict] | None:
        if targets is None:
            return None
        return [
            {"class_labels": target["labels"], "boxes": target["boxes"]}
            for target in targets
        ]

    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a training checkpoint, remapping legacy attention layouts.

        Checkpoints trained under the transformers 5.9 overlay env carry
        the pre-rename DINOv2 attention keys; under 5.19 the tensors are
        identical but renamed (see ``_DINOV2_ATTENTION_RENAMES``). The
        remap is a no-op whenever the layouts already match.
        """
        super().load_train_state_dict(
            _remap_legacy_dinov2_attention_keys(state_dict, self._model)
        )

    def _forward_train(
        self, x: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None
    ) -> dict:
        outputs = self._model(pixel_values=x, labels=self._labels(targets))
        out = {
            "pred_logits": outputs["logits"],
            "pred_boxes": outputs["pred_boxes"],
            "loss": outputs.get("loss"),
            "loss_dict": outputs.get("loss_dict") or {},
        }
        if self.ssl_head is not None and self.training:
            # Two stochastic draws through the projection head over the
            # final-layer query features (train only — the val path never
            # consumes them). The RAW pre-normalization outputs always
            # accompany the draws: the SIGReg leg consumes them when
            # weighted (sigreg_coef>0), and the unweighted statistic is the
            # always-on λ_sr calibration readout (doc 18 §2).
            features = outputs["last_hidden_state"]
            out["ssl_z1"], out["ssl_s1"] = self.ssl_head.forward_with_raw(features)
            out["ssl_z2"], out["ssl_s2"] = self.ssl_head.forward_with_raw(features)
        return out

    def forward_backend(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, boxes); the tuple form keeps the export graph
        free of custom dataclasses (dynamo limitation)."""
        out = self._forward_train(x)
        return out["pred_logits"], out["pred_boxes"]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_backend(x)

    def _as_output(self, raw) -> RFDETRObjectDetectionOutput:
        if isinstance(raw, RFDETRObjectDetectionOutput):
            return raw
        logits, boxes = raw
        return RFDETRObjectDetectionOutput(logits=logits, boxes=boxes)

    def postprocess(
        self,
        raw_outputs: Any,
        metadata: list[dict[str, Any]],
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Decode raw outputs to per-image boxes in original-frame pixels
        (sigmoid top-k; official ``PostProcess`` semantics)."""
        return decode_detection_outputs(
            raw_outputs,
            metadata,
            num_top_queries=self.num_top_queries,
            threshold=threshold,
        )

    # ── export ──────────────────────────────────────────────────────────
    @property
    def model_input_spec(self):
        from autovision.tasks.task_model_io import ModelInputSpec, TensorSpec

        return ModelInputSpec(
            input_specs={
                "images": TensorSpec(
                    shape=(3, *self.image_size), dtype=torch.float32, is_batched=True
                )
            }
        )

    def verify_onnx_export_outputs(
        self, torch_outputs: tuple, onnx_outputs: list
    ) -> None:
        torch.testing.assert_close(
            torch_outputs[0].sum(dim=1),
            torch.as_tensor(onnx_outputs[0]).float().sum(dim=1),
            atol=5e-3,
            rtol=1e-1,
            equal_nan=True,
            check_device=False,
            check_dtype=False,
        )
        torch.testing.assert_close(
            torch_outputs[1].sum(dim=1),
            torch.as_tensor(onnx_outputs[1]).float().sum(dim=1),
            atol=5e-3,
            rtol=1e-1,
            equal_nan=True,
            check_device=False,
            check_dtype=False,
        )


def _repo_id(model: str) -> str:
    """``rf-detr-nano[-notpretrained]`` → verified HF repo id."""
    info = _SSL_MODEL_INFO[model.removesuffix("-notpretrained")]
    return str(info["hf"])
