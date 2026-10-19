# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""D-FINE decoder: the RT-DETRv2 skeleton with distributional refinement.

Shares the RT-DETRv2 pipeline verbatim (encoder proposals, contrastive
denoising, self/cross attention layers, fp32 islands around box heads) and
swaps the box-delta head for D-FINE's distributional head: each layer
predicts per-side distance distributions over ``reg_max + 1`` non-uniform
bins (the weighting function ``W``, curvature ``reg_scale``) and
refines the previous layer's box (the fine-grained distribution refinement
loop). See :mod:`dfine_utils` for the distance parameterization.

Training-only output extensions (consumed by ``DFINECriterionV2``):

- ``pred_corners_list`` / ``ref_boxes_list``: per-layer raw corner logits
  and the reference box each layer refined from (matched-query segment);
- ``dn_pred_corners_list`` / ``dn_ref_boxes_list``: same for the
  contrastive-denoising segment (denoising queries are prepended, so the
  matched segment starts at ``dn_num_split[0]``);
- ``query_features``: the final layer's hidden states after the
  denoising split — the exact input of the last ``dec_score_head``
  (the classification head). Emitted only in training; the per-object
  SSL contrastive head (design doc 14) consumes it.
- ``ssl_candidate_features`` (Route B, ``ssl_candidate_features=True``):
  the top-k candidate features gathered from the encoder memory before
  detach — the label-free SSL-only pool. With the flag set, both this
  and ``query_features`` are emitted in EVAL mode as well, so the
  validation loss is the held-out contrastive loss (design doc 14
  §4.10). Inert for every baseline contract.

The export/inference contract is unchanged: ``pred_logits`` + normalized
cxcywh ``pred_boxes`` (final layer), feeding the shared decode path.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.init as init
from torch import nn

from autovision.tasks.detection.shared.denoising import (
    get_contrastive_denoising_training_group,
    inverse_sigmoid,
)
from autovision.tasks.detection.shared.dfine_utils import (
    DEFAULT_REG_SCALE,
    LQE,
    WEIGHTING_UP,
    distance2bbox,
    integrate_corners,
    weighting_function,
)
from autovision.tasks.detection.shared.rtdetrv2_decoder import (
    MLP,
    RTDETRv2Decoder,
    _score_head_logits,
)

__all__ = ["DFINEv2Decoder"]


class DFINEv2Decoder(RTDETRv2Decoder):
    """RT-DETRv2 decoder + D-FINE distributional bbox heads (FDR loop)."""

    def __init__(
        self,
        *args,
        reg_max: int = 32,
        reg_scale: float = DEFAULT_REG_SCALE,
        ssl_candidate_features: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reg_max = reg_max
        self.reg_scale = reg_scale
        # Route B emission switch (design doc 14 §4.10): when set, the
        # decoder exposes the top-k candidate features (the SSL-only pool)
        # and the final-layer hidden states in EVAL mode too — the
        # contrastive loss is the whole objective, so the validation loss
        # is the held-out SSL loss. Inert for every baseline contract.
        self._emit_query_features = ssl_candidate_features
        self._emit_candidate_features = ssl_candidate_features
        # The integration weights ARE the weighting function W: bins are
        # non-uniform in ``w / reg_scale`` units (official D-FINE v2
        # geometry), shared by the decode integral and the FGL targets.
        self.register_buffer(
            "proj",
            weighting_function(reg_max, WEIGHTING_UP, reg_scale),
            persistent=False,
        )
        # Rebuild the per-layer box heads as distributional heads: the base
        # class initialized them for 4-dim box deltas. Zero-initialized last
        # layers emit uniform distributions → distance 0 → the initial box
        # IS the reference (refinement starts from the encoder proposals).
        self.dec_bbox_head = nn.ModuleList(
            MLP(self.hidden_dim, self.hidden_dim, 4 * (reg_max + 1), 3)
            for _ in range(self.num_layers)
        )
        for head in self.dec_bbox_head:
            init.constant_(head.layers[-1].weight, 0)
            init.constant_(head.layers[-1].bias, 0)
        # Layer Quality Estimator (official D-FINE module): per-layer
        # additive calibration of the class logits from the layer's own
        # corner distributions. Zero-init at construction (a no-op that
        # learns); eval only exercises the eval_idx layer's estimator
        # (the loop breaks there).
        self.lqe_layers = nn.ModuleList(
            LQE(4, 64, 2, reg_max) for _ in range(self.num_layers)
        )
        # Layer-0 box regressor (official pre_bbox_head): an RT-DETRv2-style
        # delta head producing the INITIAL anchor box from the encoder
        # proposals. Its (detached) output is the fixed decode anchor for
        # EVERY layer's corner distribution — the shared coordinate frame
        # that makes FGL targets cacheable and DDF geometry consistent.
        self.pre_bbox_head = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        # Zero-init (baseline dfine_decoder:794-795): at construction the
        # anchor IS the reference proposal — a randomly-initialized anchor
        # poisons every cached FGL/DDF target computed against it.
        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

    @property
    def num_bins(self) -> int:
        return self.reg_max + 1

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
                    max_gt_cap=self.max_gt_cap,
                )
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                None, None, None, None,
            )

        (
            init_ref_contents,
            init_ref_points_unact,
            enc_topk_logits,
            enc_topk_bboxes,
        ) = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_bbox_unact
        )

        (
            matched_bboxes,
            matched_logits,
            corners_list,
            refs_list,
            dn_out_bboxes,
            dn_out_logits,
            dn_corners_list,
            dn_refs_list,
            matched_hidden,
            pre_bboxes,
            pre_scores,
            dn_pre_bboxes,
            dn_pre_scores,
        ) = self._refine_layers(
            init_ref_contents, init_ref_points_unact, memory, spatial_shapes,
            attn_mask, dn_meta,
            text_embeds=self._ov_forward_text_embeds(),
        )

        out: dict = {
            "pred_logits": matched_logits[-1],
            "pred_boxes": matched_bboxes[-1],
        }

        if self.training or self._emit_query_features:
            out["query_features"] = matched_hidden
            if self._emit_candidate_features:
                # Set during _get_decoder_input above (same forward call).
                out["ssl_candidate_features"] = self._candidate_features

        if self.training:
            out["aux_outputs"] = self._set_aux_loss(
                matched_logits[:-1], matched_bboxes[:-1]
            )
            out["pre_outputs"] = {
                "pred_logits": pre_scores,
                "pred_boxes": pre_bboxes,
            }
            out["enc_aux_outputs"] = self._set_aux_loss(
                [enc_topk_logits], [enc_topk_bboxes]
            )
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}
            out["pred_corners_list"] = corners_list
            out["ref_boxes_list"] = refs_list
            if dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(
                    dn_out_logits, dn_out_bboxes
                )
                out["dn_pre_outputs"] = {
                    "pred_logits": dn_pre_scores,
                    "pred_boxes": dn_pre_bboxes,
                }
                out["dn_meta"] = dn_meta
                out["dn_pred_corners_list"] = dn_corners_list
                out["dn_ref_boxes_list"] = dn_refs_list

        return out

    def _refine_layers(
        self,
        init_ref_contents: torch.Tensor,
        init_ref_points_unact: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes,
        attn_mask: torch.Tensor | None,
        dn_meta: dict | None,
        text_embeds: torch.Tensor | None = None,
    ):
        """FDR refinement loop + matched/denoising split (shared contract).

        Consumes the prepared query contents/boxes (baseline top-k or the
        density-guided dynamic selection in the Dome subclass) and runs the
        per-layer refinement; returns the stacked per-layer outputs split
        into the matched and denoising segments, plus the final layer's
        hidden states (matched segment) for the SSL contrastive head:

        ``(matched_bboxes, matched_logits, corners_list, refs_list,
        dn_out_bboxes, dn_out_logits, dn_corners_list, dn_refs_list,
        matched_hidden)``
        with the ``dn_*`` entries ``None`` / ``[]`` when there is no
        denoising segment.
        """
        # D-FINE FDR loop (official semantics): layer 0 predicts an initial
        # anchor box with a delta head; every layer predicts a CORNER
        # RESIDUAL accumulated across layers (undetached, so deep losses
        # backprop into shallow heads) over the previous layer's detached
        # hidden state, and decodes against the FIXED anchor — a shared
        # coordinate frame for all layers (cacheable FGL targets, consistent
        # DDF geometry). The per-layer decoded box only feeds the attention
        # reference of the next layer.
        ref_boxes = init_ref_points_unact.sigmoid().float()
        output = init_ref_contents
        out_bboxes: list[torch.Tensor] = []
        out_logits: list[torch.Tensor] = []
        corners_list: list[torch.Tensor] = []
        refs_list: list[torch.Tensor] = []
        pred_corners_undetach: torch.Tensor | None = None
        output_detach: torch.Tensor | None = None
        ref_points_initial: torch.Tensor | None = None
        pre_bboxes: torch.Tensor | None = None
        pre_scores: torch.Tensor | None = None

        for layer_index, layer in enumerate(self.decoder.layers):
            ref_detached = ref_boxes.detach()
            output = layer(
                output,
                ref_detached.unsqueeze(2),
                memory,
                spatial_shapes,
                attn_mask,
                None,
                self.query_pos_head(ref_detached).clamp(min=-10.0, max=10.0),
                text_kv=text_embeds,
            )

            # fp32 island: the distribution head + integration is the AMP
            # overflow hotspot (softmax over reg_max+1 bins × 4 sides). The
            # per-layer score head + LQE live here too (baseline parity):
            # their logits feed VFL/BCE directly, and bf16 rounding there
            # noise-floors the classification targets.
            with torch.autocast(device_type=output.device.type, enabled=False):
                output_fp32 = output.float()
                if layer_index == 0:
                    pre_bboxes = torch.sigmoid(
                        self.pre_bbox_head(output_fp32)
                        + inverse_sigmoid(ref_boxes.float())
                    )
                    ref_points_initial = pre_bboxes.detach()
                    pre_scores = _score_head_logits(
                        self.dec_score_head[layer_index], output_fp32, text_embeds
                    )
                head_input = output_fp32
                if output_detach is not None:
                    head_input = head_input + output_detach.float()
                corners = self.dec_bbox_head[layer_index](head_input)
                if pred_corners_undetach is not None:
                    corners = corners + pred_corners_undetach
                bbox = distance2bbox(
                    ref_points_initial,
                    integrate_corners(corners, self.proj.to(corners.dtype)),
                    self.reg_scale,
                )
                # Layer 0 reuses the pre-output head's scores (baseline
                # score_head[0] serves both roles).
                base_scores = (
                    pre_scores
                    if layer_index == 0
                    else _score_head_logits(
                        self.dec_score_head[layer_index], output_fp32, text_embeds
                    )
                )
                layer_scores = self.lqe_layers[layer_index](base_scores, corners)

            out_bboxes.append(bbox)
            out_logits.append(layer_scores)
            corners_list.append(corners)
            refs_list.append(ref_points_initial)

            if not self.training and layer_index == self.decoder.eval_idx:
                break
            # Next layer's attention reference: the detached decoded box.
            pred_corners_undetach = corners
            ref_boxes = bbox.detach()
            output_detach = output.detach()

        out_bboxes = torch.stack(out_bboxes)
        out_logits = torch.stack(out_logits)

        if self.training and dn_meta is not None:
            dn_out_bboxes, matched_bboxes = torch.split(
                out_bboxes, dn_meta["dn_num_split"], dim=2
            )
            dn_out_logits, matched_logits = torch.split(
                out_logits, dn_meta["dn_num_split"], dim=2
            )
            dn_num = dn_meta["dn_num_split"][0]
            dn_corners_list = [c[:, :dn_num] for c in corners_list]
            dn_refs_list = [r[:, :dn_num] for r in refs_list]
            corners_list = [c[:, dn_num:] for c in corners_list]
            refs_list = [r[:, dn_num:] for r in refs_list]
            # The pre anchor/scores cover the padded (denoising + matched)
            # query set — split into the supervised dn_pre head (baseline
            # supervises it like the matched pre head) and the matched
            # pre head.
            dn_pre_bboxes, pre_bboxes = torch.split(
                pre_bboxes, dn_meta["dn_num_split"], dim=1
            )
            dn_pre_scores, pre_scores = torch.split(
                pre_scores, dn_meta["dn_num_split"], dim=1
            )
            # The hidden states live under the same prepended-denoising
            # layout as the per-layer outputs: only the matched segment is
            # exposed (the denoising segment's hidden states have no
            # consumer — its queries never enter the contrastive pool).
            matched_hidden = output[:, dn_num:]
        else:
            matched_bboxes, matched_logits = out_bboxes, out_logits
            dn_out_bboxes = dn_out_logits = None
            dn_corners_list = dn_refs_list = []
            dn_pre_bboxes = dn_pre_scores = None
            matched_hidden = output

        return (
            matched_bboxes,
            matched_logits,
            corners_list,
            refs_list,
            dn_out_bboxes,
            dn_out_logits,
            dn_corners_list,
            dn_refs_list,
            matched_hidden,
            pre_bboxes,
            pre_scores,
            dn_pre_bboxes,
            dn_pre_scores,
        )
