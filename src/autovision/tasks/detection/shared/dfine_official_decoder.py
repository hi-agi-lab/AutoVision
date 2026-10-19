# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Official D-FINE decoder (the ``DFINETransformer`` gateway design).

Based on the official D-FINE implementation (D-FINE: Redefine Regression
Task of DETRs as Fine-grained Distribution Refinement, Apache-2.0,
``Peterande/D-FINE`` @ ``956d170``, ``src/zoo/dfine/dfine_decoder.py``),
modified by AutoVision per Apache-2.0 section 4b; provenance ledger entry (P1-2). This is the decoder architecture the
published baseline D-FINE COCO checkpoints were trained with — its
decoder layer fuses self-attention
and cross-attention outputs through a learned :class:`Gate` instead of the
RT-DETRv2 residual+norm, and it has NO per-layer ``value_proj`` /
``output_proj`` (the encoder memory feeds deformable attention directly).

The fork's own :class:`~autovision...dfine_decoder.DFINEv2Decoder` is a
different (RT-DETRv2-based) D-FINE port and is NOT weight-compatible with
those checkpoints; use this decoder (``decoder="dfine_official"``) when
warm-starting from them.

Divergences from the vendored original, all load-neutral:

- The training output dict is reorganized into the fork's criterion
  contract (``pred_corners_list`` / ``ref_boxes_list`` /
  ``dn_pred_corners_list`` / ``dn_ref_boxes_list`` as flat per-layer lists
  alongside plain ``aux_outputs`` dicts) instead of corners nested inside
  ``aux_outputs`` entries. Keys, shapes, and orderings are otherwise
  identical.
- Denoising routes through the fork's
  :func:`~...denoising.get_contrastive_denoising_training_group` (adds the
  optional ``max_gt_cap`` guard; the original hardcodes
  ``box_noise_scale=1.0`` at the call site, this port passes the configured
  value — identical at the default).
- The upstream ``num_classes``-adaptation load hooks are not registered;
  class-count-dependent tensors are handled explicitly by the fork's
  baseline-checkpoint loader instead.
- The upstream ``convert_to_deploy`` switch is exposed as
  :meth:`_OfficialMSDeformableAttention.enable_gather_core` (same core-func
  rebuild, export-oriented naming and loop entry point on the decoder).
- The decoder returns the plain upstream tuple; no extra per-layer query
  states are returned.
"""

from __future__ import annotations

import copy
import functools
import math
from collections import OrderedDict
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from autovision.tasks.detection.shared.denoising import (
    get_contrastive_denoising_training_group,
)
from autovision.tasks.detection.shared.dfine_utils import (
    distance2bbox,
    weighting_function,
)
from autovision.tasks.detection.shared.gtr.segmentation_head import (
    SegmentationHead,
)
from autovision.tasks.detection.shared.open_vocab import (
    ContrastiveClassHead,
    TextCrossAttention,
)
from autovision.tasks.detection.shared.rtdetrv2_decoder import (
    MLP,
    _score_head_logits,
    bias_init_with_prob,
    deformable_attention_core_func_v2,
    get_activation,
    inverse_sigmoid,
)

__all__ = ["DFINEOfficialDecoder"]


class _OfficialMSDeformableAttention(nn.Module):
    """Upstream D-FINE MSDeformableAttention (no value/output proj)."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int | List[int] = 4,
        method: str = "default",
        offset_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels
            num_points_list = list(num_points)
        else:
            num_points_list = [num_points for _ in range(num_levels)]
        self.num_points_list = num_points_list

        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        # Persistent by default (upstream parity): the published checkpoints
        # carry this buffer under cross_attn.num_points_scale.
        self.register_buffer(
            "num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32)
        )

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self.ms_deformable_attn_core = functools.partial(
            deformable_attention_core_func_v2, method=self.method
        )

        self._reset_parameters()

        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self) -> None:
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile(
            [1, sum(self.num_points_list), 1]
        )
        scaling = torch.concat(
            [torch.arange(1, n + 1) for n in self.num_points_list]
        ).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def enable_gather_core(self) -> None:
        """Rebuild the deformable core on the export-safe gather path
        (the export-time counterpart of the upstream ``convert_to_deploy``
        switch)."""
        if self.method == "default":
            self.method = "bilinear_gather"
            self.ms_deformable_attn_core = functools.partial(
                deformable_attention_core_func_v2, method=self.method
            )

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: Any,
    ) -> torch.Tensor:
        bs, Len_q = query.shape[:2]

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list), 2
        )

        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list)
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2
            )
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = (
                sampling_offsets
                * num_points_scale
                * reference_points[:, :, None, :, 2:]
                * self.offset_scale
            )
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but got "
                f"{reference_points.shape[-1]} instead."
            )

        return self.ms_deformable_attn_core(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points_list,
        )


class _OfficialDecoderLayer(nn.Module):
    """Self-attn → cross-attn → Gate fusion → FFN (official D-FINE layer)."""

    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        n_levels: int = 4,
        n_points: int | List[int] = 4,
        cross_attn_method: str = "default",
        layer_scale: float | None = None,
        text_kv_dim: int | None = None,
        text_attn_heads: int = 8,
    ) -> None:
        super().__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = _OfficialMSDeformableAttention(
            d_model, n_head, n_levels, n_points, method=cross_attn_method
        )
        self.dropout2 = nn.Dropout(dropout)

        self.gateway = _Gate(d_model)

        # Open-vocab text cross-attention (design doc 27 §3.6): the gated
        # residual runs after the gateway fusion, before the FFN. Zero-init
        # gate — a bitwise identity until training moves it.
        self.text_attn: TextCrossAttention | None = None
        if text_kv_dim is not None:
            self.text_attn = TextCrossAttention(d_model, text_kv_dim, text_attn_heads)
        self.dropout_txt = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(
        self,
        target: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: Any,
        attn_mask: torch.Tensor | None = None,
        query_pos_embed: torch.Tensor | None = None,
        text_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes,
        )

        target = self.gateway(target, self.dropout2(target2))

        if self.text_attn is not None and text_kv is not None:
            target = target + self.dropout_txt(self.text_attn(target, text_kv))

        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


class _Gate(nn.Module):
    """Learned per-channel fusion of the self- and cross-attention streams."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class _Integral(nn.Module):
    """``sum{Pr(n) * W(n)}`` over the ``reg_max + 1`` distribution bins."""

    def __init__(self, reg_max: int = 32) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x: torch.Tensor, project: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class _LQE(nn.Module):
    """Location Quality Estimator: additive logit calibration from corners."""

    def __init__(
        self, k: int, hidden_dim: int, num_layers: int, reg_max: int, act: str = "relu"
    ) -> None:
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores: torch.Tensor, pred_corners: torch.Tensor) -> torch.Tensor:
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class _OfficialFDRDecoder(nn.Module):
    """TransformerDecoder implementing Fine-grained Distribution Refinement."""

    def __init__(
        self,
        hidden_dim: int,
        decoder_layer: nn.Module,
        decoder_layer_wide: nn.Module,
        num_layers: int,
        num_head: int,
        reg_max: int,
        reg_scale: nn.Parameter,
        up: nn.Parameter,
        eval_idx: int = -1,
        layer_scale: float = 2,
        activation: str = "relu",
        frozen_query_pos: bool = False,
        segmentation_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.frozen_query_pos = frozen_query_pos
        self.segmentation_head = segmentation_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        # References to the owner's parameters: registered under BOTH
        # ``decoder.up`` and ``decoder.decoder.up`` (published-checkpoint
        # layout).
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [
                copy.deepcopy(decoder_layer_wide)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )
        self.lqe_layers = nn.ModuleList(
            [copy.deepcopy(_LQE(4, 64, 2, reg_max, act=activation)) for _ in range(num_layers)]
        )

    def value_op(
        self,
        memory: torch.Tensor,
        value_proj: nn.Module | None,
        value_scale: int | None,
        memory_mask: torch.Tensor | None,
        memory_spatial_shapes: Any,
    ) -> torch.Tensor:
        value = value_proj(memory) if value_proj is not None else memory
        value = (
            F.interpolate(memory, size=value_scale)
            if value_scale is not None
            else value
        )
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        return value.reshape(value.shape[0], value.shape[1], self.num_head, -1)

    def convert_to_deploy(self) -> None:
        self.project = weighting_function(
            self.reg_max,
            float(self.up.item()),
            float(self.reg_scale.item()),
        )
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList(
            [nn.Identity()] * self.eval_idx + [self.lqe_layers[self.eval_idx]]
        )

    def forward(
        self,
        target: torch.Tensor,
        ref_points_unact: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: Any,
        bbox_head: nn.ModuleList,
        score_head: nn.ModuleList,
        query_pos_head: nn.Module,
        pre_bbox_head: nn.Module,
        integral: _Integral,
        up: torch.Tensor,
        reg_scale: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        dn_meta: dict | None = None,
        text_embeds: torch.Tensor | None = None,
        spatial_features: torch.Tensor | None = None,
    ) -> tuple:
        output = target
        output_detach: torch.Tensor | int = 0
        pred_corners_undetach: torch.Tensor | int = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes: list[torch.Tensor] = []
        dec_out_logits: list[torch.Tensor] = []
        dec_out_pred_corners: list[torch.Tensor] = []
        dec_out_refs: list[torch.Tensor] = []
        #: Query hidden states per producing layer (mask head input; only
        #: collected when a segmentation head is attached).
        dec_out_hidden: list[torch.Tensor] = []
        if not hasattr(self, "project"):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)

        if self.frozen_query_pos:
            # GTR lineage: the query positional embedding is computed ONCE
            # from the initial references and reused for every layer
            # (GTR-main engine/gtr/det/decoder.py:543-544). Their deploy-time
            # ``pos_qk`` fusion depends on this being constant across layers.
            # D-FINE instead recomputes it per layer (upstream
            # zoo/dfine/dfine_decoder.py:406) — lineage-gated, not a
            # stylistic choice: GTR-S checkpoints were trained with the
            # frozen embedding (measured: recomputing per layer costs ~8.5
            # AP on COCO val).
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            if not self.frozen_query_pos:
                query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(
                    query_pos_embed, scale_factor=self.layer_scale
                )
                value = self.value_op(
                    memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes
                )
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(
                output,
                ref_points_input,
                value,
                spatial_shapes,
                attn_mask,
                query_pos_embed,
                text_kv=text_embeds,
            )

            # Bbox refinement in fp32 (fp16 overflow hotspot: MLP output +
            # inverse_sigmoid before sigmoid, esp. with drifted EMA weights).
            with torch.amp.autocast(device_type=output.device.type, enabled=False):
                output_fp32 = output.float()

                if i == 0:
                    ref_points_detach_fp32 = ref_points_detach.float()
                    pre_bboxes_fp32 = F.sigmoid(
                        pre_bbox_head(output_fp32)
                        + inverse_sigmoid(ref_points_detach_fp32)
                    )
                    ref_points_initial_fp32 = pre_bboxes_fp32.detach()

                    pre_scores = _score_head_logits(score_head[0], output, text_embeds)

                bbox_head_input_fp32 = output_fp32
                if isinstance(output_detach, torch.Tensor):
                    output_detach_fp32 = output_detach.float()
                    bbox_head_input_fp32 = bbox_head_input_fp32 + output_detach_fp32
                pred_corners_fp32 = bbox_head[i](bbox_head_input_fp32)
                if isinstance(pred_corners_undetach, torch.Tensor):
                    pred_corners_undetach_fp32 = pred_corners_undetach.float()
                    pred_corners_fp32 = pred_corners_fp32 + pred_corners_undetach_fp32

                inter_ref_bbox_fp32 = distance2bbox(
                    ref_points_initial_fp32,
                    integral(pred_corners_fp32, project),
                    reg_scale,
                )

            if self.training or i == self.eval_idx:
                with torch.amp.autocast(device_type=output.device.type, enabled=False):
                    output_fp32 = output.float()
                    scores_fp32 = _score_head_logits(score_head[i], output_fp32, text_embeds)
                    scores_fp32 = self.lqe_layers[i](scores_fp32, pred_corners_fp32)

                if self.training:
                    dec_out_logits.append(scores_fp32)
                    dec_out_bboxes.append(inter_ref_bbox_fp32)
                    dec_out_pred_corners.append(pred_corners_fp32)
                    dec_out_refs.append(ref_points_initial_fp32)
                else:
                    dec_out_logits.append(scores_fp32.to(output.dtype))
                    dec_out_bboxes.append(inter_ref_bbox_fp32.to(output.dtype))
                    dec_out_pred_corners.append(pred_corners_fp32.to(output.dtype))
                    dec_out_refs.append(ref_points_initial_fp32.to(output.dtype))
                if self.segmentation_head is not None:
                    dec_out_hidden.append(output)
                if not self.training:
                    break

            pred_corners_undetach = pred_corners_fp32
            ref_points_detach = inter_ref_bbox_fp32.detach().to(dtype=output.dtype)
            output_detach = output.detach()

        # Segmentation: per-layer mask logits from the collected query
        # features and the stride-8 spatial features (official placement —
        # the head runs once over the loop's hidden states).
        masks = None
        pre_segs = None
        if self.segmentation_head is not None:
            if spatial_features is None:
                raise ValueError(
                    "the segmentation head requires spatial_features "
                    "(encoder level 0); the decoder was called without them."
                )
            mask_logits = self.segmentation_head(
                spatial_features=spatial_features,
                query_features=dec_out_hidden,
            )
            masks = torch.stack(mask_logits)
            pre_segs = mask_logits[0]

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes_fp32,
            pre_scores,
            masks,
            pre_segs,
        )


class DFINEOfficialDecoder(nn.Module):
    """Official D-FINE decoder, weight-compatible with the published
    published baseline D-FINE COCO checkpoints (the
    family). Emits the fork's training contract (see module docstring)."""

    def __init__(
        self,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        feat_channels: List[int] | None = None,
        feat_strides: List[int] | None = None,
        num_levels: int = 3,
        num_points: int | List[int] = 4,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        gtr_compat: bool = False,
        num_groups: int = 3,
        mask_downsample_ratio: int | None = None,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        max_gt_cap: int | None = None,
        learn_query_content: bool = False,
        eval_spatial_size: tuple[int, int] | None = None,
        eval_idx: int = -1,
        eps: float = 1e-2,
        cross_attn_method: str = "default",
        query_select_method: str = "default",
        reg_max: int = 32,
        reg_scale: float = 4.0,
        layer_scale: float = 1,
        text_embed_dim: int | None = None,
        language_guided_selection: bool = False,
        decoder_text_attention: bool = False,
        text_attn_heads: int = 8,
        ov_logit_scale_init: float | None = None,
        ov_logit_scale_max: float | None = None,
        ov_class_bias: bool = True,
    ) -> None:
        super().__init__()
        feat_channels = feat_channels or [512, 1024, 2048]
        feat_strides = list(feat_strides or [8, 16, 32])
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = True
        self.reg_max = reg_max

        assert query_select_method in ("default", "one2many", "agnostic")
        assert cross_attn_method in ("default", "discrete")
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # GTR-lineage compatibility mode (gtr_s/m/l/x official checkpoints,
        # 2026-10-16 gold-check): GTR's GTRTransformer is a modified D-FINE
        # decoder whose selection stage differs on four axes that are
        # invisible to shape-based loading — see the construction sites
        # below for each:
        #   1. query_pos_head is a 3-layer hidden-width MLP with the decoder
        #      activation (D-FINE: MLP(4, 2H, H, 2), no act);
        #   2. selection heads are per-group copies and EVAL uses group 0 —
        #      under group_detr>1 training the shared heads never receive
        #      gradients (their own code comment), so loading/using the
        #      shared pair silently selects queries with template weights;
        #   3. memory feeds the selection heads directly — D-FINE's
        #      ``enc_output`` (proj + norm) does not exist in GTR;
        #   4. the decoder MLPs (selection/box heads, LQE) run with the
        #      decoder activation (silu for GTR; relu is the D-FINE default).
        # The second per-layer norm is also named ``norm2`` in GTR vs
        # ``norm3`` here — pure renaming, handled by the GTR assembly
        # loader's key remap.
        self.gtr_compat = bool(gtr_compat)
        self.num_groups = int(num_groups)
        if self.gtr_compat and (
            text_embed_dim is not None
            or language_guided_selection
            or decoder_text_attention
        ):
            raise ValueError(
                "gtr_compat=True is the closed-set GTR lineage decoder; "
                "the open-vocabulary branch requires the D-FINE layout."
            )
        # Instance segmentation head (GTR seg lineage): per-query mask
        # logits from the decoder's per-layer hidden states × the stride-8
        # spatial features, at eval_spatial_size / mask_downsample_ratio.
        # Owned by the INNER decoder only (official layout: the parameter
        # keys are ``decoder.decoder.segmentation_head.*``; registering it
        # here too would double every weight in state_dict).
        segmentation_head: nn.Module | None = None
        if mask_downsample_ratio is not None:
            if not self.gtr_compat:
                raise NotImplementedError(
                    "mask_downsample_ratio is the GTR-seg lineage feature "
                    "(gtr_compat=True); upstream D-FINE does not ship an "
                    "instance-segmentation decoder."
                )
            if eval_spatial_size is None:
                raise ValueError(
                    "the segmentation head bakes eval_spatial_size into its "
                    "output resolution; pass it (e.g. (640, 640))."
                )
            segmentation_head = SegmentationHead(
                hidden_dim,
                num_layers,
                downsample_ratio=int(mask_downsample_ratio),
                image_size=(int(eval_spatial_size[0]), int(eval_spatial_size[1])),
            )
        # Open-vocabulary branch (design doc 27) — same contract as the
        # fork decoders (rtdetrv2_decoder.py): contrastive score heads over
        # the persistent ``ov_text_embeds`` buffer, reduced (B, Q, C) view.
        self._ov_text_embed_dim = text_embed_dim
        if language_guided_selection and query_select_method == "agnostic":
            raise ValueError(
                "language_guided_selection requires a class-scored encoder "
                "head; query_select_method='agnostic' replaces it with a "
                "single objectness logit."
            )
        self._ov_lgs = bool(language_guided_selection) and text_embed_dim is not None
        self._ov_text_attn = (
            bool(decoder_text_attention) and text_embed_dim is not None
        )
        self._ov_logit_scale_init = ov_logit_scale_init
        self._ov_logit_scale_max = ov_logit_scale_max
        self._ov_class_bias = ov_class_bias
        self._ov_embeds_loaded = False
        if text_embed_dim is not None:
            self.register_buffer(
                "ov_text_embeds", torch.zeros(num_classes, text_embed_dim)
            )

        self._build_input_proj_layer(feat_channels)

        # Non-trainable geometry parameters: registered under BOTH
        # ``up``/``reg_scale`` and ``decoder.up``/``decoder.reg_scale``
        # (TransformerDecoder holds the same objects) — the published
        # checkpoint layout.
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = _OfficialDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            text_kv_dim=text_embed_dim if self._ov_text_attn else None,
            text_attn_heads=text_attn_heads,
        )
        decoder_layer_wide = _OfficialDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
            text_kv_dim=text_embed_dim if self._ov_text_attn else None,
            text_attn_heads=text_attn_heads,
        )
        self.decoder = _OfficialFDRDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
            activation,
            frozen_query_pos=gtr_compat,
            segmentation_head=segmentation_head,
        )

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.max_gt_cap = max_gt_cap
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )
            init.normal_(self.denoising_class_embed.weight[:-1])

        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        if self.gtr_compat:
            # GTR: 3-layer hidden-width MLP with the decoder activation.
            self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, act=activation)
        else:
            self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)
            self.enc_output = nn.Sequential(
                OrderedDict(
                    [
                        ("proj", nn.Linear(hidden_dim, hidden_dim)),
                        ("norm", nn.LayerNorm(hidden_dim)),
                    ]
                )
            )

        if query_select_method == "agnostic":
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        elif self._ov_lgs:
            # v2 only (design doc 27 §3.5 / D5); plain v1 keeps the
            # closed-set encoder head.
            self.enc_score_head = self._make_score_head(hidden_dim)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=activation)
        if self.gtr_compat:
            # Per-group selection heads; eval consumes group 0 (the GTR
            # training layout — shared heads stay as inert templates).
            self.enc_score_head_group = nn.ModuleList(
                [copy.deepcopy(self.enc_score_head) for _ in range(self.num_groups)]
            )
            self.enc_bbox_head_group = nn.ModuleList(
                [copy.deepcopy(self.enc_bbox_head) for _ in range(self.num_groups)]
            )

        resolved_eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.eval_idx = resolved_eval_idx
        # Wide-tail score heads: layers up to eval_idx run at hidden_dim,
        # the scaled tail at scaled_dim (official layer_scale layout).
        self.dec_score_head = nn.ModuleList(
            [
                self._make_score_head(
                    hidden_dim if i <= resolved_eval_idx else scaled_dim
                )
                for i in range(num_layers)
            ]
        )
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=activation)
        self.dec_bbox_head = nn.ModuleList(
            [
                MLP(
                    hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3, act=activation
                )
                for _ in range(resolved_eval_idx + 1)
            ]
            + [
                MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3, act=activation)
                for _ in range(num_layers - resolved_eval_idx - 1)
            ]
        )
        self.integral = _Integral(self.reg_max)

        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors, persistent=False)
            self.register_buffer("valid_mask", valid_mask, persistent=False)

        self._reset_parameters(feat_channels)

    def _make_score_head(self, in_dim: int) -> nn.Module:
        """Baseline per-class Linear, or the open-vocab contrastive head
        (mirrors the fork decoders' dispatch, rtdetrv2_decoder.py)."""
        if self._ov_text_embed_dim is not None:
            return ContrastiveClassHead(
                in_dim,
                self.num_classes,
                self._ov_text_embed_dim,
                logit_scale_init=(
                    self._ov_logit_scale_init
                    if self._ov_logit_scale_init is not None
                    else 2.659
                ),
                logit_scale_max=(
                    self._ov_logit_scale_max
                    if self._ov_logit_scale_max is not None
                    else 4.605
                ),
                use_class_bias=self._ov_class_bias,
            )
        return nn.Linear(in_dim, self.num_classes)

    def load_ov_text_embeds(self, embeds: torch.Tensor) -> None:
        """Fill the persistent ``ov_text_embeds`` buffer (task-model hook)."""
        if self._ov_text_embed_dim is None:
            raise ValueError(
                "load_ov_text_embeds requires the open-vocab branch "
                "(construct the decoder with text_embed_dim)."
            )
        expected = (self.num_classes, self._ov_text_embed_dim)
        if tuple(embeds.shape) != expected:
            raise ValueError(
                f"ov_text_embeds must have shape {expected}, got "
                f"{tuple(embeds.shape)}."
            )
        self.ov_text_embeds.copy_(F.normalize(embeds.float(), dim=-1))
        self._ov_embeds_loaded = True

    def _enc_class_logits(
        self, output_memory: torch.Tensor, head: nn.Module | None = None
    ) -> torch.Tensor:
        """Encoder-head logits for query selection (doc 27 §3.5).

        ``head`` overrides the shared score head (the GTR lineage selects
        with ``enc_score_head_group[0]``; the open-vocab path always uses
        the shared head).
        """
        if self._ov_lgs:
            return self.enc_score_head(output_memory, self.ov_text_embeds)
        return (head or self.enc_score_head)(output_memory)

    def _ov_forward_text_embeds(self) -> torch.Tensor | None:
        if self._ov_text_embed_dim is None:
            return None
        if not self._ov_embeds_loaded:
            raise ValueError(
                "the open-vocab decoder was built but its text embeddings "
                "were never loaded — call load_ov_text_embeds() with the "
                "encoded vocabulary before the first forward."
            )
        return self.ov_text_embeds

    def convert_to_deploy(self) -> None:
        self.decoder.convert_to_deploy()
        self.dec_score_head = nn.ModuleList(
            [nn.Identity()] * self.eval_idx + [self.dec_score_head[self.eval_idx]]
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity()
                for i in range(len(self.dec_bbox_head))
            ]
        )

    def enable_gather_core(self) -> None:
        """Switch every decoder layer's cross-attention to the export-safe
        gather core (see :meth:`_OfficialMSDeformableAttention.enable_gather_core`)."""
        for layer in self.decoder.layers:
            layer.cross_attn.enable_gather_core()

    def _reset_parameters(self, feat_channels: List[int]) -> None:
        bias = bias_init_with_prob(0.01)
        if isinstance(self.enc_score_head, ContrastiveClassHead):
            self.enc_score_head.reset_calibration(bias)
        else:
            init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            if isinstance(cls_, ContrastiveClassHead):
                cls_.reset_calibration(bias)
            else:
                init.constant_(cls_.bias, bias)
            if hasattr(reg_, "layers"):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        if not self.gtr_compat:
            init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        for layer in self.query_pos_head.layers[:-1]:
            init.xavier_uniform_(layer.weight)
        init.xavier_uniform_(self.query_pos_head.layers[-1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels: List[int]) -> None:
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        in_channels, self.hidden_dim, 1, bias=False
                                    ),
                                ),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )

        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        in_channels,
                                        self.hidden_dim,
                                        3,
                                        2,
                                        padding=1,
                                        bias=False,
                                    ),
                                ),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(
        self, feats: List[torch.Tensor]
    ) -> tuple[torch.Tensor, list[Any]]:
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))
        feat_flatten = []
        spatial_shapes = []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
        return torch.concat(feat_flatten, 1), spatial_shapes

    def _generate_anchors(
        self,
        spatial_shapes: Any = None,
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h), torch.arange(w), indexing="ij"
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor(
                [w, h], dtype=dtype
            )
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(
            -1, keepdim=True
        )
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes: Any,
        denoising_logits: torch.Tensor | None = None,
        denoising_bbox_unact: torch.Tensor | None = None,
    ) -> tuple:
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(
                spatial_shapes, device=memory.device
            )
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        memory = valid_mask.to(memory.dtype) * memory

        if self.gtr_compat:
            # GTR selection: memory feeds the heads directly (no enc_output
            # projection) and eval uses the group-0 heads.
            output_memory = memory
            score_head = self.enc_score_head_group[0]
            bbox_head = self.enc_bbox_head_group[0]
        else:
            output_memory = self.enc_output(memory)
            score_head = self.enc_score_head
            bbox_head = self.enc_bbox_head
        enc_outputs_logits = self._enc_class_logits(output_memory, head=score_head)

        enc_topk_bboxes_list: list[torch.Tensor] = []
        enc_topk_logits_list: list[torch.Tensor] = []
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            output_memory, enc_outputs_logits, anchors, self.num_queries
        )

        enc_topk_bbox_unact = bbox_head(enc_topk_memory) + enc_topk_anchors

        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat(
                [denoising_bbox_unact, enc_topk_bbox_unact], dim=1
            )
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(
        self,
        memory: torch.Tensor,
        outputs_logits: torch.Tensor,
        outputs_anchors_unact: torch.Tensor,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        elif self.query_select_method == "agnostic":
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
        else:
            raise ValueError(f"Unsupported query selection: {self.query_select_method}")

        topk_anchors = outputs_anchors_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]),
        )

        topk_logits = outputs_logits.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])
        )

        topk_memory = memory.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
        )

        return topk_memory, topk_logits, topk_anchors

    @torch.jit.unused
    def _set_aux_loss(
        self, outputs_class: Any, outputs_coord: Any, outputs_mask: Any = None
    ) -> list[dict[str, torch.Tensor]]:
        if outputs_mask is None:
            return [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class, outputs_coord)
            ]
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_masks": m}
            for a, b, m in zip(outputs_class, outputs_coord, outputs_mask)
        ]

    def forward(
        self, feats: List[torch.Tensor], targets: Any = None
    ) -> dict[str, Any]:
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
                None,
                None,
                None,
                None,
            )

        (
            init_ref_contents,
            init_ref_points_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
        ) = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_bbox_unact
        )

        (
            out_bboxes,
            out_logits,
            out_corners,
            out_refs,
            pre_bboxes,
            pre_logits,
            out_masks,
            pre_segs,
        ) = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
            text_embeds=self._ov_forward_text_embeds(),
            spatial_features=(
                feats[0] if self.decoder.segmentation_head is not None else None
            ),
        )

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(
                pre_logits, dn_meta["dn_num_split"], dim=1
            )
            dn_pre_bboxes, pre_bboxes = torch.split(
                pre_bboxes, dn_meta["dn_num_split"], dim=1
            )
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes, dn_meta["dn_num_split"], dim=2
            )
            dn_out_logits, out_logits = torch.split(
                out_logits, dn_meta["dn_num_split"], dim=2
            )
            dn_out_corners, out_corners = torch.split(
                out_corners, dn_meta["dn_num_split"], dim=2
            )
            dn_out_refs, out_refs = torch.split(
                out_refs, dn_meta["dn_num_split"], dim=2
            )
            if out_masks is not None:
                dn_out_masks, out_masks = torch.split(
                    out_masks, dn_meta["dn_num_split"], dim=2
                )
                dn_pre_segs, pre_segs = torch.split(
                    pre_segs, dn_meta["dn_num_split"], dim=1
                )
            else:
                dn_out_masks = None

        if self.training:
            out: dict[str, Any] = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "aux_outputs": self._set_aux_loss(
                    out_logits[:-1],
                    out_bboxes[:-1],
                    out_masks[:-1] if out_masks is not None else None,
                ),
                "pre_outputs": {"pred_logits": pre_logits, "pred_boxes": pre_bboxes},
                "enc_aux_outputs": self._set_aux_loss(
                    enc_topk_logits_list, enc_topk_bboxes_list
                ),
                "enc_meta": {
                    "class_agnostic": self.query_select_method == "agnostic"
                },
                "pred_corners_list": list(out_corners),
                "ref_boxes_list": list(out_refs),
            }
            if out_masks is not None:
                out["pred_masks"] = out_masks[-1]
                out["pre_outputs"]["pred_masks"] = pre_segs
            if dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_masks if dn_out_masks is not None else None,
                )
                out["dn_pre_outputs"] = {
                    "pred_logits": dn_pre_logits,
                    "pred_boxes": dn_pre_bboxes,
                }
                if out_masks is not None:
                    out["dn_pre_outputs"]["pred_masks"] = dn_pre_segs
                out["dn_meta"] = dn_meta
                out["dn_pred_corners_list"] = list(dn_out_corners)
                out["dn_ref_boxes_list"] = list(dn_out_refs)
        else:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
            }
            if out_masks is not None:
                out["pred_masks"] = out_masks[-1]

        return out
