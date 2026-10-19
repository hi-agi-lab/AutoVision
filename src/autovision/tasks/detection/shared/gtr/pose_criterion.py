# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR pose training pieces: OKS loss + Hungarian matcher + criterion.

Re-port from Intellindust-AI-Lab GTR (``engine/gtr/pose/keypoint_loss.py``
+ ``matcher.py`` + ``criterion.py``, MIT License, Copyright (c) 2026 The
GTR Authors; those files are "Modified from DETRPose", itself from
DINO/DN-DETR and DETR), adapted by AutoVision per MIT/Apache-2.0 terms;
ledger: (GTR multi-task S5b, roadmap design
doc 31 §11).

Official pose recipe (``gtrpose_base.yml``): losses ``['vfl',
'keypoints']`` with weights ``{loss_vfl: 2.0, loss_keypoints: 10.0,
loss_oks: 4.0}`` (``loss_oks`` is produced by the ``keypoints`` group);
matcher costs class 2 / keypoints 10 / OKS 4. Predicted keypoints are
``(N, 34)`` = 17 × (x, y) at the normalized model frame; targets carry
``keypoints`` as ``(n, 51)`` = 17 × (x, y, v) plus per-instance ``area``
and ``boxes`` (the pose centers).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GTRPoseCriterion",
    "GTRPoseHungarianMatcher",
    "OKSLoss",
    "oks_overlaps",
    "sigmoid_focal_loss",
]

_SIGMAS_17 = np.array(
    [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89],
    dtype=np.float32,
) / 10.0
_SIGMAS_14 = np.array(
    [.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79],
    dtype=np.float32,
) / 10.0


def _sigmas_for(num_keypoints: int) -> np.ndarray:
    if num_keypoints == 17:
        return _SIGMAS_17
    if num_keypoints == 14:
        return _SIGMAS_14
    raise ValueError(f"Unsupported keypoints number {num_keypoints}")


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """Upstream utils' focal loss (RetinaNet form)."""
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


def oks_overlaps(kpt_preds, kpt_gts, kpt_valids, kpt_areas, sigmas):
    sigmas = kpt_preds.new_tensor(sigmas)
    variances = (sigmas * 2) ** 2
    assert kpt_preds.size(0) == kpt_gts.size(0)
    kpt_preds = kpt_preds.reshape(-1, kpt_preds.size(-1) // 2, 2)
    kpt_gts = kpt_gts.reshape(-1, kpt_gts.size(-1) // 2, 2)
    squared_distance = (kpt_preds[:, :, 0] - kpt_gts[:, :, 0]) ** 2 + (
        kpt_preds[:, :, 1] - kpt_gts[:, :, 1]
    ) ** 2
    squared_distance0 = squared_distance / (kpt_areas[:, None] * variances[None, :] * 2)
    squared_distance1 = torch.exp(-squared_distance0)
    squared_distance1 = squared_distance1 * kpt_valids
    return squared_distance1.sum(dim=1) / (kpt_valids.sum(dim=1) + 1e-6)


def oks_loss(pred, target, valid=None, area=None, linear=False, sigmas=None, eps=1e-6):
    oks = oks_overlaps(pred, target, valid, area, sigmas).clamp(min=eps)
    return oks if linear else -oks.log()


class OKSLoss(nn.Module):
    def __init__(self, linear=False, num_keypoints=17, eps=1e-6, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.sigmas = _sigmas_for(num_keypoints)

    def forward(self, pred, target, valid, area, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, "none", "mean", "sum")
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        return self.loss_weight * oks_loss(
            pred, target, valid=valid, area=area, linear=self.linear, sigmas=self.sigmas, eps=self.eps
        )


class GTRPoseHungarianMatcher(nn.Module):
    """Pose matching: focal class cost + keypoint L1 (valid-masked) + 1-OKS."""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_keypoints: float = 10.0,
        cost_oks: float = 4.0,
        focal_alpha: float = 0.25,
        num_body_points: int = 17,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_keypoints = cost_keypoints
        self.cost_oks = cost_oks
        self.focal_alpha = focal_alpha
        self.num_body_points = num_body_points
        self.sigmas = _sigmas_for(num_body_points)

    @torch.no_grad()
    def forward(self, outputs, targets):
        from scipy.optimize import linear_sum_assignment

        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_keypoints = outputs["pred_keypoints"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_keypoints = torch.cat([v["keypoints"] for v in targets])
        tgt_area = torch.cat([v["area"] for v in targets])

        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        Z_pred = out_keypoints[:, 0 : (self.num_body_points * 2)]
        Z_gt = tgt_keypoints[:, 0 : (self.num_body_points * 2)]
        V_gt = tgt_keypoints[:, (self.num_body_points * 2) :]
        if Z_pred.sum() > 0:
            sigmas = Z_pred.new_tensor(self.sigmas)
            variances = (sigmas * 2) ** 2
            kpt_preds = Z_pred.reshape(-1, Z_pred.size(-1) // 2, 2)
            kpt_gts = Z_gt.reshape(-1, Z_gt.size(-1) // 2, 2)
            squared_distance = (kpt_preds[:, None, :, 0] - kpt_gts[None, :, :, 0]) ** 2 + (
                kpt_preds[:, None, :, 1] - kpt_gts[None, :, :, 1]
            ) ** 2
            squared_distance0 = squared_distance / (tgt_area[:, None] * variances[None, :] * 2)
            squared_distance1 = torch.exp(-squared_distance0)
            squared_distance1 = squared_distance1 * V_gt
            oks = (squared_distance1.sum(dim=-1) / (V_gt.sum(dim=-1) + 1e-6)).clamp(min=1e-6)
            cost_oks = 1 - oks
            cost_keypoints = torch.abs(Z_pred[:, None, :] - Z_gt[None])
            cost_keypoints = cost_keypoints * V_gt.repeat_interleave(2, dim=1)[None]
            cost_keypoints = cost_keypoints.sum(-1)
        else:
            cost_keypoints = torch.zeros(out_keypoints.shape[0], tgt_keypoints.shape[0])
            cost_oks = torch.zeros_like(cost_keypoints)
        cost = (
            self.cost_class * cost_class
            + self.cost_keypoints * cost_keypoints
            + self.cost_oks * cost_oks
        )
        cost = cost.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class GTRPoseCriterion(nn.Module):
    """Pose criterion: VFL/MAL class losses + L1 & OKS keypoint losses."""

    def __init__(
        self,
        num_classes: int,
        matcher: GTRPoseHungarianMatcher,
        weight_dict: dict[str, float] | None = None,
        losses: list[str] | None = None,
        num_body_points: int = 17,
        focal_alpha: float = 0.25,
        mal_alpha: float | None = None,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = dict(
            weight_dict or {"loss_vfl": 2.0, "loss_keypoints": 10.0, "loss_oks": 4.0}
        )
        self.losses = list(losses or ["vfl", "keypoints"])
        self.focal_alpha = focal_alpha
        self.mal_alpha = mal_alpha
        self.gamma = gamma
        self.vis = 0.1
        self.abs = 1
        self.num_body_points = num_body_points
        self.oks = OKSLoss(
            linear=True, num_keypoints=num_body_points, eps=1e-6, reduction="mean", loss_weight=1.0
        )

    # ── losses ──────────────────────────────────────────────────────────
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][j] for t, (_, j) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
            * src_logits.shape[1]
        )
        return {"loss_ce": loss_ce}

    def loss_vfl(self, outputs, targets, indices, num_boxes, log=True):
        idx = self._get_src_permutation_idx(indices)
        assert "pred_keypoints" in outputs
        src_keypoints = outputs["pred_keypoints"][idx]
        Z_pred = src_keypoints[:, 0 : (self.num_body_points * 2)]
        targets_keypoints = torch.cat(
            [t["keypoints"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        targets_area = torch.cat(
            [t["area"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        Z_gt = targets_keypoints[:, 0 : (self.num_body_points * 2)]
        V_gt = targets_keypoints[:, (self.num_body_points * 2) :]
        oks = self.oks(
            Z_pred, Z_gt, V_gt, targets_area, weight=None, avg_factor=None, reduction_override=None
        ).detach()
        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = oks.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target
        pred_score = F.sigmoid(src_logits).detach()
        weight = self.focal_alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_mal(self, outputs, targets, indices, num_boxes, log=True):
        idx = self._get_src_permutation_idx(indices)
        assert "pred_keypoints" in outputs
        src_keypoints = outputs["pred_keypoints"][idx]
        Z_pred = src_keypoints[:, 0 : (self.num_body_points * 2)]
        targets_keypoints = torch.cat(
            [t["keypoints"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        targets_area = torch.cat(
            [t["area"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        Z_gt = targets_keypoints[:, 0 : (self.num_body_points * 2)]
        V_gt = targets_keypoints[:, (self.num_body_points * 2) :]
        oks = self.oks(
            Z_pred, Z_gt, V_gt, targets_area, weight=None, avg_factor=None, reduction_override=None
        ).detach()
        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = oks.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target
        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha is not None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_mal": loss}

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """FGL/DDF self-distillation branch (off in the shipped recipe)."""
        losses = {}
        if "pred_corners" in outputs and "teacher_corners" in outputs:
            idx = self._get_src_permutation_idx(indices)
            reg_max = outputs["reg_max"]
            src_keypoints = outputs["pred_keypoints"][idx]
            Z_pred = src_keypoints[:, 0 : (self.num_body_points * 2)]
            targets_keypoints = torch.cat(
                [t["keypoints"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )
            targets_area = torch.cat(
                [t["area"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )
            Z_gt = targets_keypoints[:, 0 : (self.num_body_points * 2)]
            V_gt = targets_keypoints[:, (self.num_body_points * 2) :]
            oks = self.oks(
                Z_pred, Z_gt, V_gt, targets_area, weight=None, avg_factor=None, reduction_override=None
            )
            pred_corners = outputs["pred_corners"].reshape(-1, (reg_max + 1))
            target_corners = outputs["teacher_corners"].reshape(-1, (reg_max + 1))
            if torch.equal(pred_corners, target_corners):
                losses["loss_dfl"] = pred_corners.sum() * 0
            else:
                weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]
                mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                mask[idx] = True
                mask = mask.unsqueeze(-1).repeat(1, 1, (self.num_body_points * 2)).reshape(-1)
                weight_targets_local[idx] = oks.reshape_as(weight_targets_local[idx]).to(
                    weight_targets_local.dtype
                )
                weight_targets_local = (
                    weight_targets_local.unsqueeze(-1)
                    .repeat(1, 1, (self.num_body_points * 2))
                    .reshape(-1)
                    .detach()
                )
                loss_match_local = (
                    weight_targets_local
                    * (T**2)
                    * (
                        nn.KLDivLoss(reduction="none")(
                            F.log_softmax(pred_corners / T, dim=1),
                            F.softmax(target_corners.detach() / T, dim=1),
                        )
                    ).sum(-1)
                )
                if "is_dn" not in outputs:
                    batch_scale = 8 / outputs["pred_keypoints"].shape[0]
                    self.num_pos, self.num_neg = (
                        (mask.sum() * batch_scale) ** 0.5,
                        ((~mask).sum() * batch_scale) ** 0.5,
                    )
                loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                losses["loss_dfl"] = (
                    loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg
                ) / (self.num_pos + self.num_neg)
        return losses

    def loss_keypoints(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_keypoints = outputs["pred_keypoints"][idx]
        if len(src_keypoints) == 0:
            device = outputs["pred_logits"].device
            zero = torch.as_tensor(0.0, device=device) + src_keypoints.sum() * 0
            return {"loss_keypoints": zero, "loss_oks": zero}
        Z_pred = src_keypoints[:, 0 : (self.num_body_points * 2)]
        targets_keypoints = torch.cat(
            [t["keypoints"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        targets_area = torch.cat(
            [t["area"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        Z_gt = targets_keypoints[:, 0 : (self.num_body_points * 2)]
        V_gt = targets_keypoints[:, (self.num_body_points * 2) :]
        oks_loss = 1 - self.oks(
            Z_pred, Z_gt, V_gt, targets_area, weight=None, avg_factor=None, reduction_override=None
        )
        pose_loss = F.l1_loss(Z_pred, Z_gt, reduction="none")
        pose_loss = pose_loss * V_gt.repeat_interleave(2, dim=1)
        return {
            "loss_keypoints": pose_loss.sum() / num_boxes,
            "loss_oks": oks_loss.sum() / num_boxes,
        }

    @torch.no_grad()
    def loss_matching_cost(self, outputs, targets, indices, num_boxes):
        cost_mean_dict = indices[1]
        return {"set_{}".format(k): v for k, v in cost_mean_dict.items()}

    # ── plumbing ────────────────────────────────────────────────────────
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Union matching set across all decoder layers (upstream verbatim)."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]
        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "keypoints": self.loss_keypoints,
            "matching": self.loss_matching_cost,
            "vfl": self.loss_vfl,
            "mal": self.loss_mal,
            "local": self.loss_local,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
        device = next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)
        self._clear_cache()

        if "aux_outputs" in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            for aux_outputs in outputs["aux_outputs"] + [outputs["aux_pre_outputs"]]:
                indices_aux = self.matcher(aux_outputs, targets)
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for aux_outputs in outputs["aux_interm_outputs"]:
                indices_enc = self.matcher(aux_outputs, targets)
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)
            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor(
                [num_boxes_go], dtype=torch.float, device=device
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(num_boxes_go)
                num_boxes_go = torch.clamp(
                    num_boxes_go / torch.distributed.get_world_size(), min=1
                ).item()
        else:
            raise AssertionError("'aux_outputs' expected in criterion outputs")

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(
                num_boxes / torch.distributed.get_world_size(), min=1
            ).item()

        losses: dict[str, torch.Tensor] = {}
        for loss in self.losses:
            indices_in = indices_go if loss in ["keypoints", "local"] else indices
            num_boxes_in = num_boxes_go if loss in ["keypoints", "local"] else num_boxes
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in)
            l_dict = {
                k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
            }
            losses.update(l_dict)

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_outputs["up"], aux_outputs["reg_scale"], aux_outputs["reg_max"] = (
                    outputs["up"],
                    outputs["reg_scale"],
                    outputs["reg_max"],
                )
                for loss in self.losses:
                    indices_in = indices_go if loss in ["keypoints", "local"] else cached_indices[i]
                    num_boxes_in = num_boxes_go if loss in ["keypoints", "local"] else num_boxes
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "aux_pre_outputs" in outputs:
            aux_outputs = outputs["aux_pre_outputs"]
            for loss in self.losses:
                indices_in = indices_go if loss in ["keypoints", "local"] else cached_indices[-1]
                num_boxes_in = num_boxes_go if loss in ["keypoints", "local"] else num_boxes
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in)
                l_dict = {
                    k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                }
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        if "aux_interm_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_interm_outputs"]):
                for loss in self.losses:
                    indices_in = indices_go if loss == "keypoints" else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if loss == "keypoints" else num_boxes
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "dn_aux_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            dn_meta = outputs["dn_meta"]
            _single_pad, scalar = self.prep_for_dn(dn_meta)
            single_pad = dn_meta["pad_size"] // dn_meta["num_dn_group"]
            dn_pos_idx = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.arange(len(targets[i]["labels"]), device=device).long()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (
                        torch.tensor(range(scalar), device=device) * single_pad
                    ).long().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([], device=device).long()
                dn_pos_idx.append((output_idx, tgt_idx))

            for i, aux_outputs in enumerate(outputs["dn_aux_outputs"]):
                aux_outputs["is_dn"] = True
                aux_outputs["up"], aux_outputs["reg_scale"], aux_outputs["reg_max"] = (
                    outputs["up"],
                    outputs["reg_scale"],
                    outputs["reg_max"],
                )
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, dn_pos_idx, num_boxes * scalar
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if "dn_aux_pre_outputs" in outputs:
                aux_outputs = outputs["dn_aux_pre_outputs"]
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, dn_pos_idx, num_boxes * scalar
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return {k: v for k, v in sorted(losses.items(), key=lambda item: item[0])}

    def prep_for_dn(self, dn_meta):
        num_dn_groups, pad_size = dn_meta["num_dn_group"], dn_meta["pad_size"]
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups
        return single_pad, num_dn_groups
