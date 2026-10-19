# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Full RT-DETRv2 decoder: encoder proposal selection + denoising + refinement.

Source lineage: RT-DETRv2 (``lyuwenyu/RT-DETR`` @ ``df2a96c``, Apache-2.0), re-expressed
by AutoVision; provenance ledger entry (P1-4).

Faithful port of the baseline ``rtdetrv2_decoder.py`` (originally by
lyuwenyu, RT-DETR; Apache-2.0). This replaces an earlier simplified decoder
that used random uniform anchors and no encoder proposals — on dense
small-object data (VisDrone) that variant could not localize at all,
because the decoder had to shrink 0.9-wide random anchor boxes down to
0.02-wide GT boxes through gradient descent alone.

Key mechanisms (all baseline):
- **Grid anchor priors**: every feature cell contributes an anchor at its
  center with a level-scaled prior size (0.05 × 2^level); the encoder heads
  predict a class score and a box delta per cell, and the top-``num_queries``
  cells (by max class logit) become decoder queries. Queries therefore
  start *near objects*.
- **Content from the encoder**: query content is the selected cell's
  projected feature (detached), matching RT-DETR's design.
- **Box-driven positional embedding** (``query_pos_head``): the query
  positional embedding is an MLP over the current reference box, updated
  every layer.
- **Contrastive denoising** (see ``denoising.py``): noised GT queries are
  prepended behind a block attention mask and split off after decoding.
- **fp32 islands** around the box-refinement heads (AMP overflow hotspot).
"""

from __future__ import annotations

import copy
import functools
import math
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from autovision.runtime.env import Env
from autovision.tasks.detection.shared.denoising import (
    get_contrastive_denoising_training_group,
)
from autovision.tasks.detection.shared.open_vocab import (
    ContrastiveClassHead,
    TextCrossAttention,
)

__all__ = ["RTDETRv2Decoder"]


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    """FC/conv bias that yields the desired prior probability at init."""
    return float(-math.log((1 - prior_prob) / prior_prob))


def get_activation(act: str | None, inpace: bool = True) -> nn.Module:
    """Resolve an activation by name (baseline behaviour)."""
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act

    act = act.lower()
    if act in ("silu", "swish"):
        module: nn.Module = nn.SiLU()
    elif act == "relu":
        module = nn.ReLU()
    elif act == "leaky_relu":
        module = nn.LeakyReLU()
    elif act == "gelu":
        module = nn.GELU()
    elif act == "hardsigmoid":
        module = nn.Hardsigmoid()
    else:
        raise RuntimeError(f"Unsupported activation: {act}")

    if hasattr(module, "inplace"):
        module.inplace = inpace
    return module


def bilinear_grid_sample(
    im: Tensor, grid: Tensor, align_corners: bool = False
) -> Tensor:
    """Gather-based bilinear ``grid_sample`` (zero padding) that exports to
    a TensorRT-compatible ONNX graph (no GridSample op)."""
    n, c, h, w = im.shape
    _, gh, gw, _ = grid.shape

    x = grid[..., 0]
    y = grid[..., 1]
    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.reshape(n, -1)
    y = y.reshape(n, -1)
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    im = F.pad(im, [1, 1, 1, 1])
    padded_h = h + 2
    padded_w = w + 2
    x0 = (x0 + 1).long().clamp(0, padded_w - 1)
    x1 = (x1 + 1).long().clamp(0, padded_w - 1)
    y0 = (y0 + 1).long().clamp(0, padded_h - 1)
    y1 = (y1 + 1).long().clamp(0, padded_h - 1)

    im = im.reshape(n, c, padded_h * padded_w)

    def _gather(xx: Tensor, yy: Tensor) -> Tensor:
        idx = (xx + yy * padded_w).unsqueeze(1).expand(-1, c, -1)
        return torch.gather(im, 2, idx)

    ia = _gather(x0, y0)
    ib = _gather(x0, y1)
    ic = _gather(x1, y0)
    id_ = _gather(x1, y1)

    return (ia * wa + ib * wb + ic * wc + id_ * wd).reshape(n, c, gh, gw)


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method: str = "default",
) -> torch.Tensor:
    """Multi-scale deformable sampling + weighting (baseline implementation)."""
    bs, _, n_head, c = value.shape
    _, Len_q, _, _, _ = sampling_locations.shape

    if method == "cuda":
        # Fused MSDA op (opt-in via AUTOVISION_MSDA_CROSS_ATTN=1): same
        # bilinear math as the "default" grid_sample path in one kernel.
        # Our value layout (bs, Len, heads, C) is already the op's.
        from autovision.tasks.detection.shared.msda import (
            ms_deform_attn_cuda,
        )

        shapes_tensor = torch.as_tensor(
            value_spatial_shapes, dtype=torch.long, device=value.device
        )
        level_start_index = torch.cat(
            [
                shapes_tensor.new_zeros(1),
                shapes_tensor.prod(-1).cumsum(0)[:-1],
            ]
        )
        num_levels = shapes_tensor.shape[0]
        points_per_level = sampling_locations.shape[3] // num_levels
        # fp32 in/out: the op's fp32 kernel matches the grid_sample path's
        # internal-fp32 accumulation bit-for-bit; a native-bf16 kernel
        # rounds differently mid-interpolation. The cast is bandwidth-
        # trivial next to the fused sampling.
        return ms_deform_attn_cuda(
            value.float().contiguous(),
            shapes_tensor,
            level_start_index,
            sampling_locations.float().view(
                *sampling_locations.shape[:3], num_levels, points_per_level, 2
            ),
            attention_weights.float(),
        ).to(value.dtype)

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)

    if method in ("default", "bilinear_gather"):
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations
    else:
        raise ValueError(f"Unknown deformable attention method: {method}")

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list: list[torch.Tensor] = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]

        if method == "default":
            sampling_value_l = F.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        elif method == "bilinear_gather":
            sampling_value_l = bilinear_grid_sample(
                value_l, sampling_grid_l, align_corners=False
            )
        else:  # discrete
            sampling_coord = (
                sampling_grid_l * torch.tensor([[w, h]], device=value.device) + 0.5
            ).to(torch.int64)
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(
                bs * n_head, Len_q * num_points_list[level], 2
            )
            s_idx = (
                torch.arange(sampling_coord.shape[0], device=value.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l: torch.Tensor = value_l[
                s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]
            ]
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                bs * n_head, c, Len_q, num_points_list[level]
            )
        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        bs * n_head, 1, Len_q, sum(num_points_list)
    )
    weighted = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted.sum(-1).reshape(bs, n_head * c, Len_q)
    return output.permute(0, 2, 1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int, act: str = "relu") -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = get_activation(act)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    """Multi-scale deformable attention (baseline implementation)."""

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
        self.register_buffer(
            "num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32)
        )

        self.total_points = num_heads * sum(num_points_list)
        if method == "default" and Env.MSDA_CROSS_ATTN.value:
            method = "cuda"
        self.method = method
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

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
        grid_init = grid_init * scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def convert_to_deploy(self) -> None:
        """Switch the grid_sample sampler to the gather-based bilinear
        equivalent that exports to a TensorRT-compatible ONNX graph."""
        if self.method in ("default", "cuda"):
            self.method = "bilinear_gather"
            self.ms_deformable_attn_core = functools.partial(
                deformable_attention_core_func_v2, method=self.method
            )

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)

        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list), 2
        )

        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list)
        )
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list)
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(
                value_spatial_shapes, device=query.device
            )
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

        output = self.ms_deformable_attn_core(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points_list,
        )
        return self.output_proj(output)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        n_levels: int = 4,
        n_points: int = 4,
        cross_attn_method: str = "default",
        text_kv_dim: int | None = None,
        text_attn_heads: int = 8,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points, method=cross_attn_method
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Open-vocab text cross-attention (design doc 27 §3.6): a zero-init
        # gated residual sublayer after the image cross-attention. Built
        # only when the decoder enables it — the baseline module tree is
        # unchanged.
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

    def with_pos_embed(self, tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: Tensor) -> Tensor:
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(
        self,
        target: Tensor,
        reference_points: Tensor,
        memory: Tensor,
        memory_spatial_shapes,
        attn_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        query_pos_embed: Tensor | None = None,
        text_kv: Tensor | None = None,
    ) -> Tensor:
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            memory,
            memory_spatial_shapes,
            memory_mask,
        )
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        if self.text_attn is not None and text_kv is not None:
            # Zero-init gate: an exact identity at init (the warm-start
            # contract, doc 27 §3.6); no extra norm before the FFN so the
            # first forward stays bitwise-equal to the no-branch decoder.
            target = target + self.dropout_txt(self.text_attn(target, text_kv))

        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        return self.norm3(target)


def _score_head_logits(
    head: nn.Module, features: Tensor, text_embeds: Tensor | None
) -> Tensor:
    """Dispatch a score head: plain Linear (and post-deploy ``Identity``)
    heads take the features only; open-vocab :class:`ContrastiveClassHead`
    heads additionally consume the shared ``text_embeds`` buffer (design
    doc 27). Dispatching on the head TYPE (not on ``text_embeds``) keeps
    the deployed official decoder — whose non-eval layers are replaced by
    ``nn.Identity`` — exportable with the branch on."""
    if isinstance(head, ContrastiveClassHead):
        return head(features, text_embeds)
    return head(features)


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim: int, decoder_layer: nn.Module,
                 num_layers: int, eval_idx: int = -1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        target: Tensor,
        ref_points_unact: Tensor,
        memory: Tensor,
        memory_spatial_shapes,
        bbox_head: nn.ModuleList,
        score_head: nn.ModuleList,
        query_pos_head: nn.Module,
        attn_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        text_embeds: Tensor | None = None,
    ):
        dec_out_bboxes: list[Tensor] = []
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
                text_kv=text_embeds,
            )

            # fp32 island: bbox refinement is the known AMP overflow hotspot.
            with torch.autocast(device_type=output.device.type, enabled=False):
                output_fp32 = output.float()
                ref_points_detach_fp32 = ref_points_detach.float()
                bbox_delta_fp32 = bbox_head[i](output_fp32)
                inter_ref_bbox_fp32 = F.sigmoid(
                    bbox_delta_fp32 + inverse_sigmoid(ref_points_detach_fp32)
                )

            ref_points = inter_ref_bbox_fp32.to(dtype=output.dtype)

            if self.training:
                dec_out_logits.append(_score_head_logits(score_head[i], output, text_embeds))
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
                dec_out_logits.append(_score_head_logits(score_head[i], output, text_embeds))
                dec_out_bboxes.append(ref_points)
                break

            ref_points_detach = ref_points.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class RTDETRv2Decoder(nn.Module):
    """Full RT-DETRv2 decoder: proposals, denoising, iterative refinement.

    Consumes the HybridEncoder's multi-scale maps and returns a dict with
    ``pred_logits`` / ``pred_boxes`` (final layer) plus, during training,
    ``aux_outputs`` (deep supervision) and ``dn_aux_outputs`` / ``dn_meta``
    for the contrastive denoising loss.
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
        max_gt_cap: int | None = None,
        learn_query_content: bool = False,
        eval_idx: int = -1,
        eps: float = 1e-2,
        cross_attn_method: str = "default",
        query_select_method: str = "default",
        text_embed_dim: int | None = None,
        language_guided_selection: bool = False,
        decoder_text_attention: bool = False,
        text_attn_heads: int = 8,
        ov_logit_scale_init: float | None = None,
        ov_logit_scale_max: float | None = None,
        ov_class_bias: bool = True,
    ) -> None:
        super().__init__()
        feat_channels = feat_channels or [256, 256, 256]
        feat_strides = list(feat_strides or [8, 16, 32])
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.feat_strides = feat_strides

        assert query_select_method in ("default", "one2many", "agnostic")
        assert cross_attn_method in ("default", "discrete", "cuda")
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # Open-vocabulary branch (design doc 27). ``text_embed_dim is None``
        # keeps every head a plain Linear and the module tree baseline-
        # identical; when set, the per-layer decoder heads become
        # ContrastiveClassHead instances emitting the (B, Q, num_classes)
        # reduced view, backed by the persistent ``ov_text_embeds`` buffer
        # (filled by the task model once the vocabulary is encoded).
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
        if text_embed_dim is not None:
            self.register_buffer(
                "ov_text_embeds", torch.zeros(num_classes, text_embed_dim)
            )
        self._ov_embeds_loaded = False

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_levels, num_points, cross_attn_method=cross_attn_method,
            text_kv_dim=text_embed_dim if self._ov_text_attn else None,
            text_attn_heads=text_attn_heads,
        )
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx)

        # contrastive denoising
        self.num_denoising = num_denoising
        self.max_gt_cap = max_gt_cap
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
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
            # v2 only (design doc 27 §3.5 / D5): language-guided selection
            # swaps the ranking signal to text alignment. Plain v1 keeps
            # the closed-set encoder head — during training the selection
            # competes inside the training vocabulary anyway.
            self.enc_score_head = self._make_score_head(hidden_dim)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        # per-layer decoder heads
        self.dec_score_head = nn.ModuleList(
            [self._make_score_head(hidden_dim) for _ in range(num_layers)]
        )
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)]
        )

        self._reset_parameters()

    def _make_score_head(self, in_dim: int) -> nn.Module:
        """One classification head: the baseline per-class Linear, or the
        text-contrastive head when the open-vocab branch is enabled (the
        logit-scale/bias hyperparameters ride the decoder ctor kwargs the
        task model fills from DETROpenVocabArgs)."""
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

    def load_ov_text_embeds(self, embeds: Tensor) -> None:
        """Fill the persistent ``ov_text_embeds`` buffer (task-model hook).

        ``embeds`` (num_classes, text_dim), any dtype — normalized and
        stored fp32. The buffer ships with the export; the frozen text
        tower does not.
        """
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

    def _enc_class_logits(self, output_memory: Tensor) -> Tensor:
        """Encoder-head logits for query selection: the baseline closed-set
        Linear, or the text-contrastive head under language-guided
        selection (v2) — the reduced (B, HW, num_classes) view keeps
        ``_select_topk``'s ``max(-1)`` semantics either way (doc 27 §3.5)."""
        if self._ov_lgs:
            return self.enc_score_head(output_memory, self.ov_text_embeds)
        return self.enc_score_head(output_memory)

    def _ov_forward_text_embeds(self) -> Tensor | None:
        """The shared text embeddings for the decoder forward, or ``None``
        on the baseline path. Raises on an unfilled buffer: with zeros the
        contrastive heads emit class-agnostic bias-only logits — silently
        wrong, so fail loud instead."""
        if self._ov_text_embed_dim is None:
            return None
        if not self._ov_embeds_loaded:
            raise ValueError(
                "the open-vocab decoder was built but its text embeddings "
                "were never loaded — call load_ov_text_embeds() with the "
                "encoded vocabulary before the first forward."
            )
        return self.ov_text_embeds

    def _reset_parameters(self) -> None:
        bias = bias_init_with_prob(0.01)
        if isinstance(self.enc_score_head, ContrastiveClassHead):
            self.enc_score_head.reset_calibration(bias)
        else:
            init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for _cls, _reg in zip(self.dec_score_head, self.dec_bbox_head):
            if isinstance(_cls, ContrastiveClassHead):
                _cls.reset_calibration(bias)
            else:
                init.constant_(_cls.bias, bias)
            init.constant_(_reg.layers[-1].weight, 0)
            init.constant_(_reg.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for proj in self.input_proj:
            init.xavier_uniform_(proj[0].weight)

    def _build_input_proj_layer(self, feat_channels: List[int]) -> None:
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                            ("norm", nn.BatchNorm2d(self.hidden_dim)),
                        ]
                    )
                )
            )
        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                            ("norm", nn.BatchNorm2d(self.hidden_dim)),
                        ]
                    )
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        feat_flatten: list[torch.Tensor] = []
        spatial_shapes: list[list[int]] = []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(
        self,
        spatial_shapes=None,
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        if spatial_shapes is None:
            raise ValueError("spatial_shapes is required (no cached eval size).")

        anchors: list[torch.Tensor] = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h), torch.arange(w), indexing="ij"
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
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

    def _select_topk(
        self,
        memory: Tensor,
        outputs_logits: Tensor,
        outputs_coords_unact: Tensor,
        topk: int,
    ):
        # Tiny inputs may produce fewer proposal cells than requested queries.
        topk = min(topk, outputs_logits.shape[1])
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        else:  # agnostic
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_memory = memory.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
        )
        topk_logits = outputs_logits.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])
        )
        topk_coords = outputs_coords_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1]),
        )
        return topk_memory, topk_logits, topk_coords

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes,
        denoising_logits: torch.Tensor | None = None,
        denoising_bbox_unact: torch.Tensor | None = None,
    ):
        anchors, valid_mask = self._generate_anchors(
            spatial_shapes, device=memory.device, dtype=memory.dtype
        )
        memory = valid_mask.to(memory.dtype) * memory

        output_memory: torch.Tensor = self.enc_output(memory)
        enc_outputs_logits: torch.Tensor = self._enc_class_logits(output_memory)
        enc_outputs_coord_unact: torch.Tensor = (
            self.enc_bbox_head(output_memory) + anchors
        )

        enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = self._select_topk(
            output_memory, enc_outputs_logits, enc_outputs_coord_unact, self.num_queries
        )
        # Pre-detach snapshot for the encoder-proposal supervision head: the
        # enc_score/enc_bbox heads otherwise receive no gradient at all
        # (the topk gather is non-differentiable and the query content is
        # detached), freezing proposal quality at its random init.
        enc_topk_bboxes = enc_topk_bbox_unact.sigmoid() if self.training else None
        enc_topk_logits_supervised = enc_topk_logits if self.training else None

        if getattr(self, "_emit_candidate_features", False):
            # Route B (design doc 14 §4.10): the pre-detach top-k candidate
            # features are the contrastive pool's instances — gradients flow
            # back through enc_output into the encoder memory. Inert unless
            # the decoder was built with ssl_candidate_features=True.
            self._candidate_features = enc_topk_memory

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

        return content, enc_topk_bbox_unact, enc_topk_logits_supervised, enc_topk_bboxes

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

        out_bboxes, out_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            text_embeds=self._ov_forward_text_embeds(),
        )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes, dn_meta["dn_num_split"], dim=2
            )
            dn_out_logits, out_logits = torch.split(
                out_logits, dn_meta["dn_num_split"], dim=2
            )

        out: dict = {
            "pred_logits": out_logits[-1],
            "pred_boxes": out_bboxes[-1],
        }

        if self.training:
            out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out["enc_aux_outputs"] = self._set_aux_loss(
                [enc_topk_logits], [enc_topk_bboxes]
            )
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}
            if dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out["dn_meta"] = dn_meta

        return out

    @staticmethod
    def _set_aux_loss(outputs_class, outputs_coord):
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class, outputs_coord)
        ]
