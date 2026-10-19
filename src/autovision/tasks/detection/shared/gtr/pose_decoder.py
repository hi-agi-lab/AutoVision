# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR pose decoder (``GTRPoseTransformer``, DETRPose lineage).

Re-port from Intellindust-AI-Lab GTR (``engine/gtr/pose/decoder.py`` +
``engine/gtr/pose/ms_deform_attn.py`` + the MLP/activation helpers of
``engine/gtr/pose/utils.py``, MIT License, Copyright (c) 2026 The GTR
Authors; those files are "Modified from DETRPose"
(https://github.com/SebastianJanampa/DETRPose), itself from Conditional
DETR / DETR), adapted by AutoVision per MIT/Apache-2.0 terms; provenance ledger entry (GTR multi-task S5, roadmap design doc 31
§11).

This lineage is deliberately SELF-CONTAINED (own relu-only MLP, own
``Gate``/``Integral``/``LQE``, own ``MSDeformAttn`` with the per-level
tuple-value grid-sample core) because it diverges from the shared
D-FINE core in several state-dict-invisible ways — notably
``inverse_sigmoid(eps=1e-3)`` (vs 1e-5 elsewhere) and the hard-coded
256-wide LQE. Fidelity beats reuse here; module/parameter names are kept
verbatim so the official ``decoder.*`` keys load unchanged.

Training-side pieces (CDN, keypoint loss/matcher) are roadmap §11 **S5b**:
the training branch raises with that pointer instead of half-wiring.
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "GTRPoseTransformer",
    "MSDeformAttn",
    "distance2pose",
]


# ── DETRPose helpers (verbatim semantics) ───────────────────────────────
class _PoseMLP(nn.Module):
    """Very simple multi-layer perceptron (relu between layers)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _inverse_sigmoid(x, eps=1e-3):
    """DETRPose's inverse sigmoid — note eps=1e-3 (not the 1e-5 elsewhere)."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def distance2pose(points, distance, reg_scale):
    """Decode per-point edge distances into refined point coordinates."""
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] + distance[..., 0] / reg_scale
    y1 = points[..., 1] + distance[..., 1] / reg_scale
    return torch.stack([x1, y1], -1)


class _Gate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = float(-math.log((1 - 0.5) / 0.5))
        nn.init.constant_(self.gate.bias, bias)
        nn.init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = F.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class _Integral(nn.Module):
    """``sum{Pr(n) * W(n)}`` over the ``reg_max + 1`` distribution bins."""

    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class _LQE(nn.Module):
    """Pose location-quality estimator: samples the encoder's finest level at
    the predicted keypoints and adds a learned logit correction."""

    def __init__(self, topk, hidden_dim, num_layers, num_body_points):
        super().__init__()
        self.k = topk
        self.hidden_dim = hidden_dim
        self.reg_conf = _PoseMLP(num_body_points * (topk + 1), hidden_dim, 1, num_layers)
        nn.init.constant_(self.reg_conf.layers[-1].weight.data, 0)
        nn.init.constant_(self.reg_conf.layers[-1].bias.data, 0)
        self.num_body_points = num_body_points

    def forward(self, scores, pred_poses, feat):
        batch, length = pred_poses.shape[:2]
        pred_poses = pred_poses.reshape(batch, length, self.num_body_points, 2)
        sampling_values = F.grid_sample(
            feat,
            2 * pred_poses - 1,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        prob_topk = sampling_values.topk(self.k, dim=-1)[0]
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(batch, length, -1))
        return scores + quality_score


def ms_deform_attn_core_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Deformable-DETR core over a per-level value tuple (grid_sample path)."""
    _, d_, _ = value[0].shape
    n_, lq_, m_, l_, p_, _ = sampling_locations.shape
    sampling_grids = 2 * sampling_locations - 1
    sampling_grids = sampling_grids.transpose(1, 2).flatten(0, 1)
    sampling_value_list = []
    for lid_, (h_, w_) in enumerate(value_spatial_shapes):
        value_l_ = value[lid_].unflatten(2, (h_, w_))
        sampling_grid_l_ = sampling_grids[:, :, lid_]
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(n_ * m_, 1, lq_, l_ * p_)
    output = (
        (torch.concat(sampling_value_list, dim=-1) * attention_weights).sum(-1).view(n_, m_ * d_, lq_)
    )
    return output.transpose(1, 2)


class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention (Deformable-DETR layout)."""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, use_4D_normalizer=False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}"
            )
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.use_4D_normalizer = use_4D_normalizer
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        ).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i % 4 + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        if self.n_points % 4 != 0:
            nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)

    def forward(self, query, reference_points, value, input_spatial_shapes):
        n_, len_q, _ = query.shape
        sampling_offsets = self.sampling_offsets(query).view(
            n_, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            n_, len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            n_, len_q, self.n_heads, self.n_levels, self.n_points
        )
        reference_points = torch.transpose(reference_points, 2, 3).flatten(1, 2)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(input_spatial_shapes, device=query.device)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.n_levels, 1, 2
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            if self.use_4D_normalizer:
                offset_normalizer = torch.stack(
                    [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
                )
                sampling_locations = (
                    reference_points[:, :, None, :, None, :2]
                    + sampling_offsets
                    / offset_normalizer[None, None, None, :, None, :]
                    * reference_points[:, :, None, :, None, 2:]
                    * 0.5
                )
            else:
                sampling_locations = (
                    reference_points[:, :, None, :, None, :2]
                    + sampling_offsets
                    / self.n_points
                    * reference_points[:, :, None, :, None, 2:]
                    * 0.5
                )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but get "
                f"{reference_points.shape[-1]} instead."
            )
        return ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()
        # within-instance self-attention
        self.within_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.within_dropout = nn.Dropout(dropout)
        self.within_norm = nn.LayerNorm(d_model)
        # across-instance self-attention
        self.across_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.across_dropout = nn.Dropout(dropout)
        self.across_norm = nn.LayerNorm(d_model)
        # deformable cross-attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.gateway = _Gate(d_model)
        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos, training):
        # Out-of-place equivalent of DETRPose's in-place ``tensor[..., -np:] +=
        # pos``; callers re-assign the returned tensor.
        if pos is not None:
            np_ = pos.shape[2]
            tensor = torch.cat(
                [tensor[:, :, :-np_], tensor[:, :, -np_:] + pos], dim=2
            )
        return tensor

    def forward_FFN(self, tgt):
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return self.norm2(tgt.clamp(min=-65504, max=65504))

    def forward(
        self,
        tgt_pose: Optional[Tensor],
        tgt_pose_query_pos: Optional[Tensor],
        tgt_pose_reference_points: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None,
        memory_spatial_shapes: Optional[Tensor] = None,
    ):
        bs, nq, num_kpt, d_model = tgt_pose.shape
        # within-instance self-attention
        tgt_pose = self.with_pos_embed(tgt_pose, tgt_pose_query_pos, self.training)
        q = k = tgt_pose.flatten(0, 1)
        tgt2 = self.within_attn(q, k, tgt_pose.flatten(0, 1))[0].reshape(
            bs, nq, num_kpt, d_model
        )
        tgt_pose = tgt_pose + self.within_dropout(tgt2)
        tgt_pose = self.within_norm(tgt_pose)
        # across-instance self-attention
        tgt_pose = tgt_pose.transpose(1, 2).flatten(0, 1)  # bs*kpts, nq, d
        q_pose = k_pose = tgt_pose
        tgt2_pose = self.across_attn(q_pose, k_pose, tgt_pose, attn_mask=attn_mask)[
            0
        ].reshape(bs * num_kpt, nq, d_model)
        tgt_pose = tgt_pose + self.across_dropout(tgt2_pose)
        tgt_pose = (
            self.across_norm(tgt_pose)
            .reshape(bs, num_kpt, nq, d_model)
            .transpose(1, 2)
        )
        # deformable cross-attention
        tgt_pose = self.with_pos_embed(tgt_pose, tgt_pose_query_pos, self.training)
        tgt2_pose = self.cross_attn(
            tgt_pose.flatten(1, 2),
            tgt_pose_reference_points,
            memory,
            memory_spatial_shapes,
        ).reshape(bs, nq, num_kpt, d_model)
        tgt_pose = self.gateway(tgt_pose, self.dropout1(tgt2_pose))
        return self.forward_FFN(tgt_pose)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        return_intermediate=False,
        hidden_dim=256,
        num_body_points=17,
    ):
        super().__init__()
        if num_layers > 0:
            self.layers = nn.ModuleList(
                [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
            )
        else:
            self.layers = nn.ModuleList([])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_body_points = num_body_points
        self.return_intermediate = return_intermediate
        self.class_embed = None
        self.pose_embed = None
        self.half_pose_ref_point_head = _PoseMLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.eval_idx = num_layers - 1
        dim_t = torch.arange(hidden_dim // 2, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / (hidden_dim // 2))
        self.register_buffer("dim_t", dim_t)
        self.scale = 2 * math.pi

    def sine_embedding(self, pos_tensor):
        x_embed = pos_tensor[..., 0:1] * self.scale
        y_embed = pos_tensor[..., 1:2] * self.scale
        pos_x = x_embed / self.dim_t
        pos_y = y_embed / self.dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4
        ).flatten(3)
        if pos_tensor.size(-1) == 2:
            return torch.cat((pos_y, pos_x), dim=3)
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))

    def forward(
        self,
        tgt,
        memory,
        refpoints_sigmoid,
        pre_pose_head,
        pose_head,
        class_head,
        lqe_head,
        feat_lqe,
        integral,
        up,
        reg_scale,
        reg_max,
        project,
        attn_mask=None,
        spatial_shapes: Optional[Tensor] = None,
        eval_aux=False,
    ):
        output = tgt
        refpoint_pose = refpoints_sigmoid
        output_pose_detach: torch.Tensor | int = 0
        pred_corners_undetach: torch.Tensor | int = 0
        dec_out_poses: list[torch.Tensor] = []
        dec_out_logits: list[torch.Tensor] = []
        dec_out_refs: list[torch.Tensor] = []
        dec_out_pred_corners: list[torch.Tensor] = []

        for layer_id, layer in enumerate(self.layers):
            refpoint_pose_input = refpoint_pose[:, :, None]
            refpoint_only_pose = refpoint_pose[:, :, 1:]
            pose_query_sine_embed = self.sine_embedding(refpoint_only_pose)
            pose_query_pos = self.half_pose_ref_point_head(pose_query_sine_embed)
            output = layer(
                tgt_pose=output,
                tgt_pose_query_pos=pose_query_pos,
                tgt_pose_reference_points=refpoint_pose_input,
                attn_mask=attn_mask,
                memory=memory,
                memory_spatial_shapes=spatial_shapes,
            )
            output_pose = output[:, :, 1:]
            output_instance = output[:, :, 0]
            if layer_id == 0:
                pre_poses = F.sigmoid(
                    pre_pose_head(output_pose) + _inverse_sigmoid(refpoint_only_pose)
                )
                pre_scores = class_head[0](output_instance)
                ref_pose_initial = pre_poses.detach()

            pred_corners = (
                pose_head[layer_id](output_pose + output_pose_detach)
                + pred_corners_undetach
            )
            refpoint_pose_without_center = distance2pose(
                ref_pose_initial, integral(pred_corners, project), reg_scale
            )
            refpoint_center_pose = torch.mean(
                refpoint_pose_without_center, dim=2, keepdim=True
            )
            refpoint_pose = torch.cat(
                [refpoint_center_pose, refpoint_pose_without_center], dim=2
            )
            if self.training or eval_aux or layer_id == self.eval_idx:
                score = class_head[layer_id](output_instance)
                logit = lqe_head[layer_id](
                    score, refpoint_pose_without_center, feat_lqe
                )
                dec_out_logits.append(logit)
                dec_out_poses.append(refpoint_pose_without_center)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_pose_initial)
                if (not self.training) and (not eval_aux):
                    break
            pred_corners_undetach = pred_corners
            if self.training:
                refpoint_pose = refpoint_pose.detach()
                output_pose_detach = output_pose.detach()
            else:
                output_pose_detach = output_pose

        return (
            torch.stack(dec_out_poses),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_poses,
            pre_scores,
        )


class GTRPoseTransformer(nn.Module):
    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        hidden_dim=256,
        nhead=8,
        num_queries=300,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        num_feature_levels=1,
        enc_n_points=4,
        dec_n_points=4,
        learnable_tgt_init=False,
        two_stage_type="no",
        num_classes=2,
        aux_loss=True,
        dec_pred_class_embed_share=False,
        dec_pred_pose_embed_share=False,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        cls_no_bias=False,
        num_body_points=17,
        feat_strides=None,
        eval_spatial_size=None,
        reg_max=32,
        reg_scale=4.0,
    ):
        super().__init__()
        self.eval_aux = getattr(self, "eval_aux", False)
        self.num_feature_levels = num_feature_levels
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.aux_loss = aux_loss

        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            return_intermediate=return_intermediate_dec,
            hidden_dim=hidden_dim,
            num_body_points=num_body_points,
        )
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries

        self.num_body_points = num_body_points
        self.keypoint_embedding = nn.Embedding(num_body_points, self.hidden_dim)
        self.instance_embedding = nn.Embedding(1, self.hidden_dim)
        self.learnable_tgt_init = learnable_tgt_init
        if learnable_tgt_init:
            self.tgt_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        else:
            self.tgt_embed = None
        # NOTE: upstream hard-codes 80 label classes for the CDN embedding
        # regardless of the task's num_classes (kept verbatim: it is a
        # checkpoint-shape-visible tensor).
        self.label_enc = nn.Embedding(80 + 1, hidden_dim)
        self.pose_enc = nn.Embedding(num_body_points, hidden_dim)
        self._reset_parameters()

        self.two_stage_type = two_stage_type
        if two_stage_type in ["standard"]:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)
        self.enc_out_class_embed = None
        self.enc_pose_embed = None

        _class_embed = nn.Linear(hidden_dim, num_classes, bias=(not cls_no_bias))
        if not cls_no_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        _pre_point_embed = _PoseMLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_pre_point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_pre_point_embed.layers[-1].bias.data, 0)
        _point_embed = _PoseMLP(hidden_dim, hidden_dim, 2 * (reg_max + 1), 3)
        nn.init.constant_(_point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_point_embed.layers[-1].bias.data, 0)
        # Upstream hard-codes the LQE's hidden width to 256 (not hidden_dim).
        _lqe_embed = _LQE(4, 256, 2, num_body_points)
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for _ in range(num_decoder_layers)]
            lqe_embed_layerlist = [_lqe_embed for _ in range(num_decoder_layers)]
        else:
            class_embed_layerlist = [
                copy.deepcopy(_class_embed) for _ in range(num_decoder_layers)
            ]
            lqe_embed_layerlist = [
                copy.deepcopy(_lqe_embed) for _ in range(num_decoder_layers)
            ]
        if dec_pred_pose_embed_share:
            pose_embed_layerlist = [_point_embed for _ in range(num_decoder_layers)]
        else:
            pose_embed_layerlist = [
                copy.deepcopy(_point_embed) for _ in range(num_decoder_layers)
            ]
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.pose_embed = nn.ModuleList(pose_embed_layerlist)
        self.lqe_embed = nn.ModuleList(lqe_embed_layerlist)
        self.pre_pose_embed = _pre_point_embed
        self.integral = _Integral(reg_max)
        self.up = nn.Parameter(torch.tensor([1 / 2]), requires_grad=False)
        self.reg_max = reg_max
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        self.deploy = False

        _keypoint_embed = _PoseMLP(hidden_dim, 2 * hidden_dim, 2 * num_body_points, 4)
        nn.init.constant_(_keypoint_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_keypoint_embed.layers[-1].bias.data, 0)
        if two_stage_bbox_embed_share:
            self.enc_pose_embed = _keypoint_embed
        else:
            self.enc_pose_embed = copy.deepcopy(_keypoint_embed)
        if two_stage_class_embed_share:
            self.enc_out_class_embed = _class_embed
        else:
            self.enc_out_class_embed = copy.deepcopy(_class_embed)

        self.feat_strides = feat_strides
        self.eval_spatial_size = eval_spatial_size
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        feat_flatten = []
        spatial_shapes = []
        split_sizes = []
        for feat in feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
            split_sizes.append(h * w)
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes, split_sizes

    def _generate_anchors(self, spatial_shapes=None, device="cpu"):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])
        anchors = []
        for lvl, (h_, w_) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, h_ - 1, h_, dtype=torch.float32, device=device),
                torch.linspace(0, w_ - 1, w_, dtype=torch.float32, device=device),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], -1)
            grid = (grid.unsqueeze(0).expand(1, -1, -1, -1) + 0.5) / torch.tensor(
                [w_, h_], dtype=torch.float32, device=device
            )
            anchors.append(grid.view(1, -1, 2))
        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors > 0.01) & (anchors < 0.99)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        # NOTE the inverted convention: the registered ``valid_mask`` holds
        # the PADDING mask (True = invalid), matching upstream's use of
        # ``memory.masked_fill(self.valid_mask, 0)``.
        return anchors, ~valid_mask

    def convert_to_deploy(self):
        self.register_buffer(
            "project",
            weighting_function(self.reg_max, self.up, self.reg_scale),
            persistent=False,
        )
        self.lqe_embed = nn.ModuleList(
            [nn.Identity()] * (self.dec_layers - 1) + [self.lqe_embed[self.dec_layers - 1]]
        )
        self.deploy = True

    def forward(self, feats, targets=None, samples=None):
        memory, spatial_shapes, split_sizes = self._get_encoder_input(feats)
        if self.training:
            output_proposals, valid_mask = self._generate_anchors(
                spatial_shapes, memory.device
            )
            output_memory = memory.masked_fill(valid_mask, float(0))
            output_proposals = output_proposals.repeat(memory.size(0), 1, 1)
        else:
            output_proposals = self.anchors.repeat(memory.size(0), 1, 1)
            output_memory = memory.masked_fill(self.valid_mask, float(0))

        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        topk = self.num_queries
        enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
        topk_idx = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
        topk_memory = output_memory.gather(
            dim=1, index=topk_idx.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )
        topk_anchors = output_proposals.gather(
            dim=1, index=topk_idx.unsqueeze(-1).repeat(1, 1, 2)
        )
        bs, nq = topk_memory.shape[:2]
        delta_unsig_keypoint = self.enc_pose_embed(topk_memory).reshape(
            bs, nq, self.num_body_points, 2
        )
        enc_outputs_pose_coord = F.sigmoid(
            delta_unsig_keypoint + topk_anchors.unsqueeze(-2)
        )
        enc_outputs_center_coord = torch.mean(
            enc_outputs_pose_coord, dim=2, keepdim=True
        )
        enc_outputs_pose_coord = torch.cat(
            [enc_outputs_center_coord, enc_outputs_pose_coord], dim=2
        )
        refpoint_pose_sigmoid = enc_outputs_pose_coord.detach()

        if self.learnable_tgt_init:
            tgt = (
                self.tgt_embed.weight.unsqueeze(0)
                .repeat([memory.shape[0], 1, 1])
                .unsqueeze(-2)
            )
        else:
            tgt = topk_memory.detach().unsqueeze(-2)
        tgt_pose = (
            self.keypoint_embedding.weight[None, None]
            .repeat(1, topk, 1, 1)
            .expand(bs, -1, -1, -1)
            + tgt
        )
        tgt_global = (
            self.instance_embedding.weight[None, None]
            .repeat(1, topk, 1, 1)
            .expand(bs, -1, -1, -1)
        )
        tgt_pose = torch.cat([tgt_global, tgt_pose], dim=2)

        interm_class = (
            torch.gather(
                enc_outputs_class_unselected,
                1,
                topk_idx.unsqueeze(-1).repeat(
                    1, 1, enc_outputs_class_unselected.shape[-1]
                ),
            )
            if self.training
            else None
        )

        # Denoising (training): DETRPose-style CDN group prepended to the queries.
        if self.training and targets is not None:
            from autovision.tasks.detection.shared.gtr.pose_dn import prepare_for_cdn

            input_query_label, input_query_pose, attn_mask, dn_meta = prepare_for_cdn(
                dn_args=(targets, 20, 0.5),
                training=True,
                num_queries=self.num_queries,
                hidden_dim=self.hidden_dim,
                num_classes=80,
                label_enc=self.label_enc,
                pose_enc=self.pose_enc,
                num_keypoints=self.num_body_points,
                img_dim=samples.shape[-2:],
                device=feats[0].device,
            )
            tgt_pose = torch.cat([input_query_label, tgt_pose], dim=1)
            refpoint_pose_sigmoid = torch.cat(
                [input_query_pose.sigmoid(), refpoint_pose_sigmoid], dim=1
            )
        else:
            input_query_label = input_query_pose = attn_mask = dn_meta = None

        # MSDeformAttn value: per-level [bs*n_heads, d/head, hw] tuple.
        value = memory.unflatten(2, (self.nhead, -1))
        value = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_sizes, dim=-1)

        if not hasattr(self, "project"):
            project = weighting_function(self.reg_max, self.up, self.reg_scale)
        else:
            project = self.project

        (
            out_poses,
            out_logits,
            _out_corners,
            _out_references,
            _out_pre_poses,
            _out_pre_scores,
        ) = self.decoder(
            tgt=tgt_pose,
            memory=value,
            refpoints_sigmoid=refpoint_pose_sigmoid,
            spatial_shapes=spatial_shapes,
            attn_mask=attn_mask,
            pre_pose_head=self.pre_pose_embed,
            pose_head=self.pose_embed,
            class_head=self.class_embed,
            lqe_head=self.lqe_embed,
            feat_lqe=feats[0],
            up=self.up,
            reg_max=self.reg_max,
            reg_scale=self.reg_scale,
            integral=self.integral,
            project=project,
            eval_aux=self.eval_aux,
        )
        if not self.deploy:
            # Upstream flattens the pose dim BEFORE the DN split, so both the
            # matched segment and the reconstruction segment are (…, 2K).
            out_poses = out_poses.flatten(-2)
        if self.training and dn_meta is not None:
            out_pre_poses = _out_pre_poses.flatten(-2)
            dn_out_poses, out_poses = torch.split(
                out_poses, [dn_meta["pad_size"], self.num_queries], dim=2
            )
            dn_out_logits, out_logits = torch.split(
                out_logits, [dn_meta["pad_size"], self.num_queries], dim=2
            )
            dn_out_corners, out_corners = torch.split(
                _out_corners, [dn_meta["pad_size"], self.num_queries], dim=2
            )
            dn_out_refs, out_refs = torch.split(
                _out_references, [dn_meta["pad_size"], self.num_queries], dim=2
            )
            dn_out_pre_poses, out_pre_poses = torch.split(
                out_pre_poses, [dn_meta["pad_size"], self.num_queries], dim=1
            )
            dn_out_pre_scores, out_pre_scores = torch.split(
                _out_pre_scores, [dn_meta["pad_size"], self.num_queries], dim=1
            )

        out = {"pred_logits": out_logits[-1], "pred_keypoints": out_poses[-1]}
        if self.training:
            # Upstream picks the SPLIT tensors everywhere once DN ran: the
            # aux/pre outputs carry only the matched query segment, while
            # the dn_* outputs carry the reconstruction segment.
            out.update(
                {
                    "pred_corners": out_corners[-1] if dn_meta is not None else _out_corners[-1],
                    "ref_points": out_refs[-1] if dn_meta is not None else _out_references[-1],
                    "up": self.up,
                    "reg_scale": self.reg_scale,
                    "reg_max": self.reg_max,
                }
            )
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_poses[:-1],
                out_corners[:-1] if dn_meta is not None else _out_corners[:-1],
                out_refs[:-1] if dn_meta is not None else _out_references[:-1],
            )
            out["aux_interm_outputs"] = [
                {
                    "pred_logits": interm_class,
                    "pred_keypoints": enc_outputs_pose_coord[:, :, 1:].flatten(-2),
                }
            ]
            out["aux_pre_outputs"] = {
                "pred_logits": out_pre_scores if dn_meta is not None else _out_pre_scores,
                "pred_keypoints": out_pre_poses,
            }
            if dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss2(
                    dn_out_logits, dn_out_poses, dn_out_corners, dn_out_refs
                )
                out["dn_aux_pre_outputs"] = {
                    "pred_logits": dn_out_pre_scores,
                    "pred_keypoints": dn_out_pre_poses,
                }
                out["dn_meta"] = dn_meta
        return out

    @torch.jit.unused
    def _set_aux_loss2(
        self, outputs_class, outputs_keypoints, outputs_corners, outputs_ref
    ):
        return [
            {
                "pred_logits": a,
                "pred_keypoints": b,
                "pred_corners": c,
                "ref_points": d,
            }
            for a, b, c, d in zip(
                outputs_class, outputs_keypoints, outputs_corners, outputs_ref
            )
        ]


def weighting_function(reg_max, up, reg_scale, deploy=False):
    """The POSE lineage's own weighting function.

    Measured 2026-10-16: differs from the shared D-FINE one by up to
    9.5e-07 (different geometric construction) — the pose decoder must use
    THIS one for weight-correct outputs (port verbatim from the upstream
    pose decoder, which carries its own copy).
    """
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.tensor([values], dtype=up.dtype, device=up.device)
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
    values = (
        [-upper_bound2]
        + left_values
        + [torch.zeros_like(up[0][None])]
        + right_values
        + [upper_bound2]
    )
    return torch.cat(values, 0)
