# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Density-guided dynamic-query decoder (Dome-DETR PAQI port, doc 13 §4.3).

Subclasses :class:`DFINEv2Decoder` and replaces the fixed top-``num_queries``
proposal selection with the baseline Dome ``_get_decoder_input`` semantics
(``src/zoo/dome/dome_decoder.py:767-943``):

1. score every grid cell with the encoder heads and take the top
   ``max_num_select`` candidates;
2. the first ``min_num_select`` (highest-scoring) pass through unconditionally;
3. the rest are gated by the MWAS window mask — candidates whose anchor
   center falls into an unselected (low-density) window are dropped;
4. survivors get one per-class NMS pass whose per-candidate IoU threshold
   is modulated by the window-pooled density at the candidate center
   (``base + scale * density``): dense clusters keep more overlapping
   candidates, sparse regions are pruned aggressively;
5. per-image query counts end up data-dependent (``min ≤ n_b ≤ max``);
   the batch is zero-padded to the maximum and described by
   ``batch_queries_num`` / ``query_valid_mask``.

Deviations from the baseline (design doc 13 §5):

- D4: pooled-density lookups use consistent (y, x) indexing (the baseline
  transposes x/y at ``dome_decoder.py:847-850``, invisible on square
  inputs);
- D5: padded queries get their logits filled with -1e4 at the source, so
  neither the Hungarian matcher nor the decode path can pick them (the
  baseline only masks the classification loss).

The FDR refinement loop is inherited verbatim via
:meth:`DFINEv2Decoder._refine_layers`; this class only changes query
preparation, the denoising-group call (variable query count), and the
output contract (three extra keys).

Export: the density-gated selection is data-dependent (nonzero, per-class
loops) and cannot trace through ``torch.export`` — :meth:`convert_to_deploy`
flips the decoder back to the fixed top-``min_num_select`` baseline path
(design doc 13 §4.8).
"""

from __future__ import annotations

import torch

from autovision.tasks.detection.shared.denoising import (
    box_cxcywh_to_xyxy,
    get_contrastive_denoising_training_group,
)
from autovision.tasks.detection.shared.dfine_decoder import (
    DFINEv2Decoder,
)
from autovision.tasks.detection.shared.dynamic_nms import (
    per_class_dynamic_nms,
)

__all__ = ["DomeDFINEv2Decoder"]

#: Fill value for padded-query logits (D5): sigmoid(-1e4) == 0 exactly in
#: fp32, so padded queries contribute ~0 to VFL, are prohibitive in the
#: matcher's focal cost, and vanish under the decode threshold.
_PADDED_LOGIT = -1e4


class DomeDFINEv2Decoder(DFINEv2Decoder):
    """D-FINE decoder with density-guided dynamic query selection."""

    def __init__(
        self,
        *args,
        min_num_select: int = 300,
        max_num_select: int = 500,
        nms_iou_base: float = 0.4,
        nms_iou_density_scale: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0 < min_num_select <= max_num_select:
            raise ValueError(
                "Dome decoder needs 0 < min_num_select <= max_num_select, "
                f"got {min_num_select}/{max_num_select}."
            )
        self.min_num_select = min_num_select
        self.max_num_select = max_num_select
        self.nms_iou_base = nms_iou_base
        self.nms_iou_density_scale = nms_iou_density_scale
        self.num_heads = self.decoder.layers[0].self_attn.num_heads
        self._static_export = False

    def convert_to_deploy(self) -> None:
        """Static-export mode: fixed top-``min_num_select`` proposals."""
        self._static_export = True

    # ── query selection ─────────────────────────────────────────────────
    def _get_decoder_input_dynamic(
        self,
        memory: torch.Tensor,
        spatial_shapes,
        defe_info: dict,
    ):
        """Density-gated query selection (baseline Dome flow, D4/D5 fixes).

        Returns ``(content, ref_boxes_unact, batch_queries_num,
        query_valid_mask)`` with the batch padded to the per-batch maximum.
        """
        anchors, valid_mask = self._generate_anchors(
            spatial_shapes, device=memory.device, dtype=memory.dtype
        )
        if anchors.shape[0] == 1 and memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        memory = valid_mask.to(memory.dtype) * memory

        output_memory: torch.Tensor = self.enc_output(memory)
        # Open-vocab v2 (design doc 27 §3.5): the same reduced-view dispatch
        # the baseline path uses — under language-guided selection the
        # ranking signal becomes text alignment, the code is untouched.
        enc_logits = self._enc_class_logits(output_memory)
        enc_coords_unact = self.enc_bbox_head(output_memory) + anchors

        # Candidate pool: top-max by max class logit (baseline "default"
        # query selection); fewer cells may exist on tiny inputs.
        topk = min(self.max_num_select, enc_logits.shape[1])
        enc_scores, enc_class_ids = enc_logits.max(-1)
        topk_ind = enc_scores.topk(topk, dim=-1).indices

        def _gather(t: torch.Tensor) -> torch.Tensor:
            return t.gather(1, topk_ind.unsqueeze(-1).expand(-1, -1, t.shape[-1]))

        mem_k = _gather(output_memory)
        # Undetached full-logit view for the encoder-proposal supervision
        # head: without it the enc_output/enc_score/enc_bbox heads receive
        # no gradient at all (every decoder input below is detached) and
        # proposal quality freezes at random init — the baseline dome
        # criterion supervises these via ``enc_aux_outputs``.
        logits_k = _gather(enc_logits)
        # The NMS only reads per-candidate max score + class id — gather
        # those 1-D views instead of the full (B, K, C) logit matrix.
        scores_k = enc_scores.gather(1, topk_ind)
        class_k = enc_class_ids.gather(1, topk_ind)
        anchors_k = _gather(anchors)
        coords_k = _gather(enc_coords_unact)

        # Window gating: candidates beyond the first min_num_select must
        # fall into a density-selected window (consistent (y, x) — D4).
        window_mask = defe_info["window_mask"]  # (B, n_h, n_w) bool
        pooled = defe_info["density_pooled"]  # (B, 1, n_h, n_w) fp32
        n_h, n_w = window_mask.shape[1], window_mask.shape[2]
        anchor_x = anchors_k[..., 0].sigmoid()
        anchor_y = anchors_k[..., 1].sigmoid()
        window_y = (anchor_y * n_h).long().clamp(0, n_h - 1)
        window_x = (anchor_x * n_w).long().clamp(0, n_w - 1)
        batch_index = torch.arange(mem_k.shape[0], device=mem_k.device)
        gate = window_mask[batch_index[:, None], window_y, window_x]  # (B, K)
        min_num = min(self.min_num_select, topk)
        gate[:, :min_num] = True

        pooled_h, pooled_w = pooled.shape[-2:]
        # Density-modulated per-candidate IoU thresholds for the whole
        # batch (same per-candidate math the per-image loop used — the
        # lookups are elementwise, so computing them for all K candidates
        # and gathering later is value-identical).
        proposals_k = coords_k.sigmoid()
        density_y = (proposals_k[..., 1] * pooled_h).long().clamp(0, pooled_h - 1)
        density_x = (proposals_k[..., 0] * pooled_w).long().clamp(0, pooled_w - 1)
        density_k = pooled[:, 0][batch_index[:, None], density_y, density_x].detach()
        thresholds_k = self.nms_iou_base + self.nms_iou_density_scale * density_k
        nms_boxes_k = box_cxcywh_to_xyxy(proposals_k).detach()

        # Single host transfer of the selection inputs (gate + NMS feeds);
        # every per-image gate/NMS decision below runs on the host. The
        # previous per-image boolean indexing (five tensors × B images)
        # forced a data-dependent-shape pipeline sync per tensor —
        # hundreds per training step.
        gate_cpu = gate.detach().cpu()
        boxes_cpu = nms_boxes_k.cpu()
        scores_cpu = scores_k.detach().cpu()
        class_cpu = class_k.detach().cpu()
        thresholds_cpu = thresholds_k.cpu()

        kept_memory: list[torch.Tensor] = []
        kept_coords: list[torch.Tensor] = []
        kept_logits: list[torch.Tensor] = []
        queries_num: list[int] = []
        for b in range(mem_k.shape[0]):
            gate_b = gate_cpu[b]
            rest_positions = (
                torch.nonzero(gate_b[min_num:], as_tuple=True)[0] + min_num
            )
            if rest_positions.numel() > 0:
                # NMS candidates: the forced-first block plus the
                # window-gated rest, in original position order (the old
                # per-image candidate list). The NMS may suppress
                # first-block candidates as suppressors; they are
                # force-kept afterwards (baseline semantics).
                cand = torch.cat([torch.arange(min_num), rest_positions])
                keep = per_class_dynamic_nms(
                    boxes_cpu[b][cand],
                    scores_cpu[b][cand],
                    class_cpu[b][cand],
                    thresholds_cpu[b][cand],
                )
                keep = keep[keep >= min_num]
                # NMS indices address the candidate list — map them back to
                # original K-space before gathering (the first block is
                # identity under this mapping, so only survivors move).
                keep_b = torch.cat(
                    [torch.arange(min_num), cand[keep]]
                )
            else:
                keep_b = torch.arange(min_num)
            queries_num.append(int(keep_b.numel()))
            keep_gpu = keep_b.to(mem_k.device)
            kept_memory.append(mem_k[b][keep_gpu])
            kept_coords.append(coords_k[b][keep_gpu])
            kept_logits.append(logits_k[b][keep_gpu])

        padded_queries = max(queries_num)
        content = output_memory.new_zeros(
            (mem_k.shape[0], padded_queries, mem_k.shape[-1])
        )
        ref_boxes_unact = coords_k.new_zeros((mem_k.shape[0], padded_queries, 4))
        # Padded proposal rows are buried at the source: the matcher's
        # focal cost keeps them far from every GT, and the supervision
        # losses only touch matched rows.
        enc_topk_logits = logits_k.new_full(
            (mem_k.shape[0], padded_queries, logits_k.shape[-1]), _PADDED_LOGIT
        )
        enc_topk_bboxes = coords_k.new_zeros((mem_k.shape[0], padded_queries, 4))
        query_valid_mask = torch.zeros(
            (mem_k.shape[0], padded_queries),
            dtype=torch.bool,
            device=mem_k.device,
        )
        for b in range(mem_k.shape[0]):
            n_b = queries_num[b]
            content[b, :n_b] = kept_memory[b]
            ref_boxes_unact[b, :n_b] = kept_coords[b]
            enc_topk_logits[b, :n_b] = kept_logits[b]
            enc_topk_bboxes[b, :n_b] = kept_coords[b].sigmoid()
            query_valid_mask[b, :n_b] = True

        return (
            content.detach(),
            ref_boxes_unact.detach(),
            queries_num,
            query_valid_mask,
            enc_topk_logits,
            enc_topk_bboxes,
        )

    def _padding_attention_mask(
        self,
        batch_queries_num: list[int],
        padded_queries: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Self-attention mask isolating padded queries (no denoising group)."""
        if all(n == padded_queries for n in batch_queries_num):
            return None
        mask = torch.zeros(
            (len(batch_queries_num) * self.num_heads, padded_queries, padded_queries),
            dtype=torch.bool,
            device=device,
        )
        for b, n_b in enumerate(batch_queries_num):
            if n_b >= padded_queries:
                continue
            head_start = b * self.num_heads
            head_end = (b + 1) * self.num_heads
            mask[head_start:head_end, :n_b, n_b:] = True
            mask[head_start:head_end, n_b:, :n_b] = True
        return mask

    # ── forward ─────────────────────────────────────────────────────────
    def forward(self, feats, targets=None, defe_info: dict | None = None):
        if self._static_export or defe_info is None:
            # Export mode / non-density path: verbatim baseline contract
            # (fixed top-num_queries proposals, no density involvement).
            return super().forward(feats, targets=targets)

        memory, spatial_shapes = self._get_encoder_input(feats)

        (
            init_ref_contents,
            init_ref_points_unact,
            batch_queries_num,
            query_valid_mask,
            enc_topk_logits,
            enc_topk_bboxes,
        ) = self._get_decoder_input_dynamic(memory, spatial_shapes, defe_info)
        padded_queries = max(batch_queries_num)

        denoising_logits = denoising_bbox_unact = None
        attn_mask = dn_meta = None
        if self.training and self.num_denoising > 0 and targets is not None:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    padded_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                    batch_queries_num=batch_queries_num,
                    num_heads=self.num_heads,
                    max_gt_cap=self.max_gt_cap,
                )
            )
            if denoising_bbox_unact is not None:
                init_ref_points_unact = torch.cat(
                    [denoising_bbox_unact, init_ref_points_unact], dim=1
                )
                init_ref_contents = torch.cat(
                    [denoising_logits, init_ref_contents], dim=1
                )
        if attn_mask is None:
            # No denoising group (zero-GT batch / eval): still isolate the
            # padded queries from the valid ones.
            attn_mask = self._padding_attention_mask(
                batch_queries_num, padded_queries, memory.device
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
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            attn_mask,
            dn_meta,
            text_embeds=self._ov_forward_text_embeds(),
        )

        # D5: padded queries must never produce detections or be matched —
        # bury their class logits at the source (every decoder layer).
        if not all(n == padded_queries for n in batch_queries_num):
            invalid = (~query_valid_mask).unsqueeze(0).unsqueeze(-1)
            matched_logits = matched_logits.masked_fill(invalid, _PADDED_LOGIT)

        out: dict = {
            "pred_logits": matched_logits[-1],
            "pred_boxes": matched_bboxes[-1],
        }
        if self.training:
            out["aux_outputs"] = self._set_aux_loss(
                matched_logits[:-1], matched_bboxes[:-1]
            )
            out["pred_corners_list"] = corners_list
            out["ref_boxes_list"] = refs_list
            out["pre_outputs"] = {
                "pred_logits": pre_scores,
                "pred_boxes": pre_bboxes,
            }
            # Encoder-proposal supervision: the dynamic selection detaches
            # everything it hands the decoder, so the enc heads are trained
            # exclusively through this branch (baseline dome criterion).
            out["enc_aux_outputs"] = self._set_aux_loss(
                [enc_topk_logits], [enc_topk_bboxes]
            )
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}
            # Final-layer hidden states over the padded query set — the
            # per-object SSL head consumes them (design doc 14); padded
            # rows are filtered by the criterion (matcher + query mask).
            out["query_features"] = matched_hidden
            if dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out["dn_pre_outputs"] = {
                    "pred_logits": dn_pre_scores,
                    "pred_boxes": dn_pre_bboxes,
                }
                out["dn_meta"] = dn_meta
                out["dn_pred_corners_list"] = dn_corners_list
                out["dn_ref_boxes_list"] = dn_refs_list

        out["batch_queries_num"] = batch_queries_num
        out["query_valid_mask"] = query_valid_mask
        # Stamp the selection bounds so the criterion's count loss reads a
        # single source of truth (design doc 13 §4.1).
        defe_info.setdefault("min_num_select", self.min_num_select)
        defe_info.setdefault("max_num_select", self.max_num_select)
        out["defe_info"] = defe_info
        return out
