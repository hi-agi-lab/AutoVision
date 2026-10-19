# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Oriented (rotated-box) RT-DETRv2 decoder + OBB output decoding.

The oriented variant reuses the RT-DETRv2 box pipeline verbatim (encoder
proposals, contrastive denoising, iterative box refinement) and adds one
MLP angle head per decoder layer. Angles have no iterative refinement:
every layer predicts its own angle from the current decoder state, bounded
to ``[-π/4, π/4]`` via ``tanh`` (the long-edge canonical convention shared
with :mod:`autovision.geometry.obb`).
Angle noise is NOT injected into the denoising group (box noise only) —
the dn angle predictions are supervised against the clean GT angles
through the same dn matching indices.
"""

from __future__ import annotations

import math
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from autovision.geometry.obb import (
    obb_nms,
    xywhr_to_xyxyxyxy,
)
from autovision.tasks.detection.shared.denoising import (
    get_contrastive_denoising_training_group,
)
from autovision.tasks.detection.shared.rtdetrv2_decoder import (
    MLP,
    RTDETRv2Decoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    inverse_sigmoid,
)

__all__ = [
    "OrientedRTDETRDecoder",
    "OrientedTransformerDecoder",
    "decode_oriented_detection_outputs",
]


class OrientedTransformerDecoder(TransformerDecoder):
    """Baseline decoder loop + per-layer angle predictions.

    Identical to the parent loop except each layer's angle head runs inside
    the same fp32 island as the box refinement (the AMP overflow hotspot)
    and is appended alongside the per-layer boxes for deep supervision.
    """

    def forward(
        self,
        target: Tensor,
        ref_points_unact: Tensor,
        memory: Tensor,
        memory_spatial_shapes,
        bbox_head: nn.ModuleList,
        score_head: nn.ModuleList,
        angle_head: nn.ModuleList,
        query_pos_head: nn.Module,
        attn_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ):
        dec_out_bboxes: list[Tensor] = []
        dec_out_angles: list[Tensor] = []
        dec_out_logits: list[Tensor] = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        output = target
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                attn_mask,
                memory_mask,
                query_pos_embed,
            )

            # fp32 island: box refinement and the angle head share the
            # known AMP overflow hotspot.
            with torch.autocast(device_type=output.device.type, enabled=False):
                output_fp32 = output.float()
                ref_points_detach_fp32 = ref_points_detach.float()
                bbox_delta_fp32 = bbox_head[i](output_fp32)
                inter_ref_bbox_fp32 = F.sigmoid(
                    bbox_delta_fp32 + inverse_sigmoid(ref_points_detach_fp32)
                )
                angle_fp32 = torch.tanh(angle_head[i](output_fp32)) * (math.pi / 4)

            ref_points = inter_ref_bbox_fp32.to(dtype=output.dtype)

            if self.training:
                dec_out_logits.append(score_head[i](output))
                dec_out_angles.append(angle_fp32)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox_fp32)
                else:
                    with torch.autocast(device_type=output.device.type, enabled=False):
                        dec_out_bboxes.append(
                            F.sigmoid(
                                bbox_delta_fp32 + inverse_sigmoid(inter_ref_bbox_fp32)
                            )
                        )
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_angles.append(angle_fp32)
                dec_out_bboxes.append(ref_points)
                break

            ref_points_detach = ref_points.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_angles),
            torch.stack(dec_out_logits),
        )


class OrientedRTDETRDecoder(RTDETRv2Decoder):
    """RT-DETRv2 decoder + per-layer angle heads (OBB variant).

    Output dict adds ``pred_angles`` (B, Q, 1) radians in ``[-π/4, π/4]``;
    ``aux_outputs`` / ``dn_aux_outputs`` entries carry ``pred_angles`` too
    so deep supervision covers the angle heads.
    """

    def __init__(
        self,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        feat_channels: List[int] | None = None,
        feat_strides: List[int] | None = None,
        num_levels: int = 3,
        num_points: int = 4,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learn_query_content: bool = False,
        eval_idx: int = -1,
        eps: float = 1e-2,
        cross_attn_method: str = "default",
        query_select_method: str = "default",
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_points=num_points,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learn_query_content=learn_query_content,
            eval_idx=eval_idx,
            eps=eps,
            cross_attn_method=cross_attn_method,
            query_select_method=query_select_method,
        )
        # Rebuild the inner decoder loop with the oriented (angle-predicting)
        # variant; construction parameters mirror the parent.
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_levels, num_points, cross_attn_method=cross_attn_method,
        )
        self.decoder = OrientedTransformerDecoder(
            hidden_dim, decoder_layer, num_layers, eval_idx
        )

        # Per-layer angle heads; zero-init the last layer of each MLP so the
        # initial angle prediction is exactly 0 (tanh(0) * pi/4).
        self.dec_angle_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 1, 3) for _ in range(num_layers)]
        )
        for head in self.dec_angle_head:
            init.constant_(head.layers[-1].weight, 0)
            init.constant_(head.layers[-1].bias, 0)

    def forward(self, feats: List[torch.Tensor], targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                )
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                None, None, None, None,
            )

        init_ref_contents, init_ref_points_unact, _, _ = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_bbox_unact
        )

        out_bboxes, out_angles, out_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.dec_angle_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes, dn_meta["dn_num_split"], dim=2
            )
            dn_out_angles, out_angles = torch.split(
                out_angles, dn_meta["dn_num_split"], dim=2
            )
            dn_out_logits, out_logits = torch.split(
                out_logits, dn_meta["dn_num_split"], dim=2
            )

        out: dict = {
            "pred_logits": out_logits[-1],
            "pred_boxes": out_bboxes[-1],
            "pred_angles": out_angles[-1],
        }

        if self.training:
            out["aux_outputs"] = self._set_aux_loss(
                out_logits[:-1], out_bboxes[:-1], out_angles[:-1]
            )
            if dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(
                    dn_out_logits, dn_out_bboxes, dn_out_angles
                )
                out["dn_meta"] = dn_meta

        return out

    @staticmethod
    def _set_aux_loss(outputs_class, outputs_coord, outputs_angle):
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_angles": c}
            for a, b, c in zip(outputs_class, outputs_coord, outputs_angle)
        ]


def decode_oriented_detection_outputs(
    raw_outputs: Any,
    metadata: list[dict[str, Any]],
    num_top_queries: int = 300,
    threshold: float = 0.0,
    iou_threshold: float = 0.65,
) -> list[dict[str, Any]]:
    """Top-k decode raw oriented DETR outputs to per-image OBBs in pixels.

    ``raw_outputs``: object with ``.logits``/``.boxes``/``.angles`` (or a
    ``(logits, boxes, angles)`` tuple) — logits (B, Q, C) raw scores, boxes
    (B, Q, 4) normalized cxcywh, angles (B, Q, 1) radians. ``metadata``:
    per-image ``{"orig_w", "orig_h"}``. Per-class rotated NMS suppresses
    duplicate detections of the same class (the one-to-one assignment does
    not survive the class-agnostic top-k flattening).

    Returns per-image ``{"labels", "polygons", "scores"}`` with polygons as
    (N, 4, 2) absolute-pixel corner arrays.
    """
    if isinstance(raw_outputs, tuple):
        logits, boxes, angles = raw_outputs
    else:
        logits, boxes, angles = (
            raw_outputs.logits,
            raw_outputs.boxes,
            raw_outputs.angles,
        )
    scores = logits.sigmoid()
    flat_scores = scores.flatten(1)
    num_top = min(num_top_queries, flat_scores.shape[1])
    scores_flat, indices = flat_scores.topk(num_top, dim=1)
    classes = indices % scores.shape[2]
    query_indices = indices // scores.shape[2]
    results = []
    for batch_index, meta in enumerate(metadata):
        keep = scores_flat[batch_index] > threshold
        labels = classes[batch_index][keep]
        conf = scores_flat[batch_index][keep]
        box_norm = boxes[batch_index, query_indices[batch_index]][keep]  # (K, 4)
        angle = angles[batch_index, query_indices[batch_index]][keep]  # (K, 1)
        w, h = float(meta["orig_w"]), float(meta["orig_h"])
        xywhr_norm = torch.cat([box_norm, angle], dim=-1)  # (K, 5)
        xywhr_px = xywhr_norm.clone()
        xywhr_px[:, 0] *= w
        xywhr_px[:, 1] *= h
        xywhr_px[:, 2] *= w
        xywhr_px[:, 3] *= h

        # Per-class greedy rotated NMS.
        final_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
        for c in labels.unique():
            cls_mask = labels == c
            cls_keep = obb_nms(xywhr_px[cls_mask], conf[cls_mask], iou_threshold)
            mask_indices = cls_mask.nonzero(as_tuple=True)[0]
            final_mask[mask_indices[cls_keep]] = True

        polygons = xywhr_to_xyxyxyxy(xywhr_px[final_mask])
        results.append(
            {
                "labels": labels[final_mask],
                "polygons": polygons,
                "scores": conf[final_mask],
            }
        )
    return results
