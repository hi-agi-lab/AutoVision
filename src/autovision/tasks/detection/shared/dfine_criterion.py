# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""D-FINE criterion: RT-DETRv2 losses + FGL (0.15) + DDF (1.5) [+ SSL] [+ SIoU].

Source lineage: D-FINE ``dfine_criterion.py`` (Apache-2.0, ``Peterande/D-FINE`` @ ``956d170``)
over the RT-DETRv2 loss set (``lyuwenyu/RT-DETR`` @ ``df2a96c``), restructured by AutoVision;
provenance ledger entry (P1-3).

Runs ONE Hungarian matching pass per decoder layer (the D-FINE losses share
the classification/box assignments) and extends the RT-DETRv2 loss set
(VFL 1.0 + L1 5.0 + GIoU 2.0) with:

- ``loss_fgl``: fine-grained localization (see :func:`dfine_utils.fg_loss`);
- ``loss_ddf``: decoupled distillation focal — the final layer's side
  distributions teach every intermediate layer (aux and denoising), see
  :func:`dfine_utils.ddf_loss`;
- ``loss_ssl``: the per-object contrastive objective (design doc 14),
  computed ONCE at the final layer with the final layer's Hungarian match.
  Like the DeFE pair it carries no ``_aux_i`` / ``_dn_i`` suffixes; the
  weight-aggregation loop copies every ``weight_dict`` entry to the aux/dn
  suffixes, but the ``if key in losses`` filter drops the never-produced
  suffixed SSL keys.

``box_regression_iou="siou"`` (design doc 15) swaps the box loss's GIoU
term for SIoU (``loss_siou``) across every branch — main, ``_pre``,
``_dn_pre``, and the copied ``_aux_i`` / ``_dn_i`` / ``_enc_i`` suffixes.

Auxiliary layers are deep-supervised with factor 1.0 through the ``*_aux_i``
/ ``*_dn_i`` suffixes like the baseline — VFL on each layer's own Hungarian
match; L1/GIoU (or SIoU), FGL and DDF on the union assignment. The dn branch
is normalized by ``num_boxes × dn_num_group``.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from autovision.tasks.detection.shared.defe import (
    defe_count_loss,
    defe_density_loss,
)
from autovision.tasks.detection.shared.denoising import (
    box_cxcywh_to_xyxy,
    get_cdn_matched_indices,
)
from autovision.tasks.detection.shared.dfine_utils import (
    DEFAULT_REG_SCALE,
    bbox2distance,
    ddf_loss,
    fg_loss,
    fg_loss_from_targets,
    flatten_match_indices,
    matched_pair_iou,
    matched_pair_iou_flat,
)
from autovision.tasks.detection.shared.distillation import (
    DistillParams,
    cross_model_distill_losses,
)
from autovision.tasks.detection.shared.matcher import (
    forward_match_many,
    indices_to_device,
)
from autovision.tasks.detection.shared.rtdetrv2_criterion import (
    loss_boxes,
    loss_labels_vfl,
)
from autovision.tasks.detection.shared.ssl_contrastive import (
    SSLParams,
    epps_pulley_sigreg,
    ssl_contrastive_loss,
)

__all__ = ["DFINECriterionV2"]


# ── mask losses (GTR seg lineage, design doc 31 §11 S2) ─────────────────
# Point-sampled BCE + Dice on matched masks, re-ported from GTR's
# engine/gtr/det/criterion.py::GTRCriterion.loss_masks (MIT; the point
# sampler is PointRend's, via GTR's segmentation_head) — provenance ledger entry
# provenance ledger entry.
def mask_point_losses(
    pred_masks: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    num_boxes: torch.Tensor,
    sampling_ratio: int,
) -> dict[str, torch.Tensor]:
    """``{'loss_mask_ce', 'loss_mask_dice'}`` over the matched pairs.

    ``pred_masks``: (B, Q, h, w) mask logits of one layer; target masks
    are sampled at the SAME point grid at their own resolution (their
    pipeline keeps target masks at dataset resolution). Point count:
    ``h * w // sampling_ratio`` (official; the ``max(h, ...)`` guard only
    matters for degenerate tiny grids).
    """
    from autovision.tasks.detection.shared.gtr.segmentation_head import (
        get_uncertain_point_coords_with_randomness,
        point_sample,
    )

    device = pred_masks.device
    flat_q, flat_gt, batch_idx = flatten_match_indices(
        indices, pred_masks.shape[1], device
    )
    src_masks = pred_masks.reshape(-1, *pred_masks.shape[-2:])[flat_q]
    if src_masks.numel() == 0:
        zeros = pred_masks.sum() * 0.0
        return {"loss_mask_ce": zeros, "loss_mask_dice": zeros}

    # Per-pair slices are 2-D (Ht, Wt) — stack, not cat (a 2-D cat along
    # dim 0 would flatten the rows). Official indexes with the (tensor)
    # match indices and cats the kept leading dim; same result.
    target_masks = torch.stack(
        [
            targets[b].get("masks")[g].to(device=device)
            for b, g in zip(batch_idx.tolist(), flat_gt.tolist())
        ],
        dim=0,
    )
    src_masks = src_masks.unsqueeze(1)
    target_masks = target_masks.unsqueeze(1).float()

    height, width = src_masks.shape[-2], src_masks.shape[-1]
    num_points = max(height, height * width // sampling_ratio)

    with torch.no_grad():
        point_coords = get_uncertain_point_coords_with_randomness(
            src_masks,
            lambda logits: -(torch.abs(logits)),
            num_points,
            3,
            0.75,
        )
        point_labels = point_sample(
            target_masks, point_coords, align_corners=False, mode="nearest"
        ).squeeze(1)

    point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)

    denominator = num_boxes.clamp(min=1) if torch.is_tensor(num_boxes) else max(num_boxes, 1)
    return {
        "loss_mask_ce": _mask_sigmoid_ce(point_logits, point_labels, denominator),
        "loss_mask_dice": _mask_dice(point_logits, point_labels, denominator),
    }


def _mask_dice(inputs: torch.Tensor, targets: torch.Tensor, num_masks) -> torch.Tensor:
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def _mask_sigmoid_ce(inputs: torch.Tensor, targets: torch.Tensor, num_masks) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks



class DFINECriterionV2(nn.Module):
    """RT-DETRv2 criterion + distributional refinement losses (FGL/DDF).

    With ``defe_info`` (density-guided dynamic queries, design doc 13
    §4.5) two auxiliary DeFE losses join the set: the density-map loss
    and the object-count regression loss. With ``ssl_params`` the
    per-object contrastive ``loss_ssl`` joins it (design doc 14 §4.4).
    Both ride the same ``weight_dict`` mechanics as every other term —
    computed once, no ``_aux_i`` / ``_dn_i`` suffixes.

    Args:
        matcher: module able to compute the matching between proposals and targets
        weight_dict: dict containing as key the names of the losses and as values their relative weight.
        dn_fgl_loss_weight: FGL weight on the denoising branches (``_dn_*``).
            ``None`` follows the main ``loss_fgl`` weight (baseline parity).
            Dense tiny-object batches (e.g. VisDrone) push CDN to a single
            group of 2 × max-GT noised queries whose FGL residuals saturate
            the reg_max bins — the dn FGL term then dominates the total loss
            without converging; lower it there.
        ssl_params: per-object contrastive SSL contract (design doc 14); ``None``
            disables the objective entirely (no ``loss_ssl`` key, no head wiring).
        box_regression_iou: IoU term of the supervised box loss (design doc 15).
            ``"giou"`` (default) keeps the upstream ``loss_giou`` emission;
            ``"siou"`` replaces it with ``loss_siou`` (Scylla-IoU,
            arXiv:2205.12740) across every box-loss branch (main, ``_pre``,
            ``_dn_pre``, ``_aux_i``, ``_dn_i``, ``_enc_i``) — mutually exclusive
            emissions. The matcher's GIoU cost is intentionally untouched.
        siou_loss_coef: weight of ``loss_siou`` in siou mode; ``None`` (default)
            inherits the ``loss_giou`` weight so the swapped-in term keeps the
            anchor weight (E0 probe: loss-scale ratio 0.953 on VisDrone).
        siou_theta: shape-cost exponent of the SIoU loss (paper default 4).
    """

    def __init__(
        self,
        matcher: nn.Module,
        weight_dict: dict[str, float] | None = None,
        reg_max: int = 32,
        reg_scale: float = DEFAULT_REG_SCALE,
        fgl_loss_weight: float = 0.15,
        dn_fgl_loss_weight: float | None = None,
        ddf_loss_weight: float = 1.5,
        defe_density_map_weight: float = 1.0,
        defe_count_loss_weight: float = 1.0,
        defe_underestimation_penalty: float = 0.3,
        defe_count_underestimation_weight: float = 2.0,
        ssl_params: SSLParams | None = None,
        box_regression_iou: Literal["giou", "siou"] = "giou",
        siou_loss_coef: float | None = None,
        siou_theta: float = 4.0,
        flat_losses: bool = False,
        distill_params: DistillParams | None = None,
        mask_losses: bool = False,
        mask_sampling_ratio: int = 16,
        mask_ce_weight: float = 5.0,
        mask_dice_weight: float = 5.0,
    ) -> None:
        super().__init__()
        if box_regression_iou not in ("giou", "siou"):
            raise ValueError(
                f"Unknown box_regression_iou {box_regression_iou!r}; "
                "expected 'giou' or 'siou'."
            )
        self.matcher = matcher
        self.weight_dict = dict(
            weight_dict
            or {
                "loss_vfl": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_fgl": fgl_loss_weight,
                "loss_ddf": ddf_loss_weight,
                "defe_density_loss": defe_density_map_weight,
                "defe_count_loss": defe_count_loss_weight,
            }
        )
        self.reg_max = reg_max
        self.reg_scale = reg_scale
        self.fgl_loss_weight = fgl_loss_weight
        self.dn_fgl_loss_weight = dn_fgl_loss_weight
        self.ddf_loss_weight = ddf_loss_weight
        self.defe_density_map_weight = defe_density_map_weight
        self.defe_count_loss_weight = defe_count_loss_weight
        self.defe_underestimation_penalty = defe_underestimation_penalty
        self.defe_count_underestimation_weight = defe_count_underestimation_weight
        self.ssl_params = ssl_params
        if ssl_params is not None:
            self.weight_dict["loss_ssl"] = ssl_params.loss_coef
        if ssl_params is not None and ssl_params.sigreg_coef > 0:
            # SIGReg third leg (design doc 16, LeVJEPA R1-A1): own weight so
            # the raw statistic is scaled exactly once, independent of
            # loss_ssl.
            self.weight_dict["loss_sigreg"] = ssl_params.sigreg_coef
        # IoU term of the box loss (design doc 15): siou mode swaps the
        # weight_dict entry — the inheritance base is the PRE-pop loss_giou
        # weight (default dict: 2.0; an explicitly supplied weight_dict is
        # the same source) — then retires loss_giou so the dict matches the
        # mutually-exclusive emission set.
        self.box_regression_iou = box_regression_iou
        self.siou_theta = siou_theta
        self._iou_loss_key = (
            "loss_siou" if box_regression_iou == "siou" else "loss_giou"
        )
        if box_regression_iou == "siou":
            inherited = self.weight_dict.get("loss_giou", 2.0)
            self.weight_dict.pop("loss_giou", None)
            self.weight_dict["loss_siou"] = (
                inherited if siou_loss_coef is None else siou_loss_coef
            )
        # Written by the training wrapper each step; drives the
        # every_n_steps gate of the SSL objective (state lives outside the
        # forward call, mirroring the RF-DETR-SSL contract).
        self.ssl_step: int = 0
        # Vectorized match-index paths for FGL/DDF (design doc 26 W2).
        # Off by default: the flattened sums reorder float ops, so
        # in-flight ablation runs stay bitwise-comparable.
        self.flat_losses = flat_losses
        # Mask branch (GTR seg lineage): the point count follows the
        # matcher's ``mask_point_sample_ratio`` when present (official
        # fallback order), else the explicit ratio.
        self.mask_losses = bool(mask_losses)
        self.mask_sampling_ratio = int(
            getattr(matcher, "mask_point_sample_ratio", None) or mask_sampling_ratio
        )
        if self.mask_losses:
            self.weight_dict.setdefault("loss_mask_ce", mask_ce_weight)
            self.weight_dict.setdefault("loss_mask_dice", mask_dice_weight)
        # Cross-model distillation contract (design doc 23). None
        # disables the branch entirely — no weight_dict entries, no
        # distill handling in forward.
        self.distill_params = distill_params
        if distill_params is not None:
            if distill_params.cls_coef > 0:
                self.weight_dict["loss_distill_cls"] = distill_params.cls_coef
            if distill_params.box_coef > 0:
                self.weight_dict["loss_distill_box"] = distill_params.box_coef
            if distill_params.ddf_coef > 0:
                self.weight_dict["loss_distill_ddf"] = distill_params.ddf_coef
        # Diagnostics from the most recent distillation forward (design
        # doc 23 §3.4): covered/union union-match pair counts.
        self.last_distill_stats: dict[str, int] | None = None
        # DDF pos/neg normalization counts cached from the aux pass for the
        # dn pass within the same forward (official semantics). Kept as
        # on-device scalars — float() here would sync the GPU every step.
        self._ddf_num_pos: torch.Tensor | None = None
        self._ddf_num_neg: torch.Tensor | None = None
        # Diagnostics from the most recent SSL evaluation (design doc 14
        # D10): matched-pool size and paired-pool size on this rank.
        self.last_ssl_stats: dict[str, float] | None = None

    def _loss_boxes(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Module-level box loss bound to this criterion's IoU term (doc 15)."""
        return loss_boxes(
            outputs,
            targets,
            indices,
            num_boxes,
            iou_term=self.box_regression_iou,
            siou_theta=self.siou_theta,
            flat_indexing=self.flat_losses,
        )

    def forward(
        self,
        outputs_list: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
        world_size: int = 1,
        dn_outputs: list[dict[str, torch.Tensor]] | None = None,
        dn_meta: dict | None = None,
        dn_pre_outputs: dict[str, torch.Tensor] | None = None,
        corners_list: list[torch.Tensor] | None = None,
        ref_boxes_list: list[torch.Tensor] | None = None,
        dn_corners_list: list[torch.Tensor] | None = None,
        dn_ref_boxes_list: list[torch.Tensor] | None = None,
        query_mask: torch.Tensor | None = None,
        pre_outputs: dict[str, torch.Tensor] | None = None,
        enc_aux_outputs: list[dict[str, torch.Tensor]] | None = None,
        enc_meta: dict | None = None,
        defe_info: dict | None = None,
        ssl_z1: torch.Tensor | None = None,
        ssl_z2: torch.Tensor | None = None,
        ssl_s1: torch.Tensor | None = None,
        ssl_s2: torch.Tensor | None = None,
        distill_targets: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        # Route B (design doc 14 §4.10): label-free SSL-only pretraining.
        # The candidate pool needs no Hungarian match and no targets, so
        # the detection losses (and their matcher pass) are skipped
        # entirely; the contrastive objective IS the loss — in train AND
        # eval, so the validation loss is the held-out SSL loss (the
        # stage-1 model-selection signal). The stride gate does not apply:
        # there is no competing detection loss to interleave with.
        if (
            self.ssl_params is not None
            and self.ssl_params.instance_source == "encoder_candidates"
        ):
            if not (
                isinstance(ssl_z1, torch.Tensor) and isinstance(ssl_z2, torch.Tensor)
            ):
                raise ValueError(
                    "ssl instance_source='encoder_candidates' requires the model "
                    "outputs to carry ssl_z1/ssl_z2 candidate features; build the "
                    "task model with the same ssl config (design doc 14 §4.10)."
                )
            ssl_stats: dict[str, float] = {}
            ssl_only_losses: dict[str, torch.Tensor] = {
                "loss_ssl": ssl_contrastive_loss(
                    ssl_z1,
                    ssl_z2,
                    outputs_list[-1],
                    targets,
                    [],
                    self.ssl_params,
                    stats_sink=ssl_stats,
                )
            }
            self.last_ssl_stats = ssl_stats
            ssl_only_losses["loss"] = (
                ssl_only_losses["loss_ssl"] * self.weight_dict["loss_ssl"]
            )
            # Keep the loss graph-attached for backward; logging reductions
            # belong to the train model, not the criterion.
            return ssl_only_losses

        num_boxes = sum(int(t["boxes"].shape[0]) for t in targets)
        num_boxes = torch.tensor(
            float(num_boxes), device=outputs_list[-1]["pred_logits"].device
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            num_boxes /= torch.distributed.get_world_size()

        losses: dict[str, torch.Tensor] = {}
        num_entries = len(outputs_list)
        device = outputs_list[-1]["pred_logits"].device
        # One consolidated matcher pass for every output group of this step
        # (decoder layers + pre-output + encoder proposals): the merged
        # problem batch (5 groups x B at the S3 spectrum configs) turns the
        # per-image scipy loop into a single batched GPU solve when the
        # optional ``lsap`` extra is present, and cuts the host transfers
        # to one otherwise — see matcher.forward_match_many for the
        # measurements and the sentinel/padding contract.
        match_groups: list[
            tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]
        ] = [(outputs, targets) for outputs in outputs_list]
        pre_group_index: int | None = None
        if pre_outputs is not None:
            pre_group_index = len(match_groups)
            match_groups.append((pre_outputs, targets))
        enc_first_index: int | None = None
        if enc_aux_outputs:
            class_agnostic = bool(enc_meta.get("class_agnostic", False))
            enc_targets_list = (
                [
                    {"boxes": t["boxes"], "labels": torch.zeros_like(t["labels"])}
                    for t in targets
                ]
                if class_agnostic
                else targets
            )
            enc_first_index = len(match_groups)
            match_groups.extend(
                (enc_outputs, enc_targets_list) for enc_outputs in enc_aux_outputs
            )
        solved = forward_match_many(self.matcher, match_groups)
        layer_indices: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            indices_to_device(solved[output_index], device)
            for output_index in range(num_entries)
        ]
        for output_index, outputs in enumerate(outputs_list):
            suffix = "" if output_index == num_entries - 1 else f"_aux_{output_index}"
            # VFL keeps each layer's own Hungarian assignment (official).
            losses[f"loss_vfl{suffix}"] = loss_labels_vfl(
                outputs,
                targets,
                layer_indices[output_index],
                num_boxes,
                query_mask=query_mask,
                flat_indexing=self.flat_losses,
            )
            # Mask losses ride the same per-layer matching (official).
            if self.mask_losses and outputs.get("pred_masks") is not None:
                for name, value in mask_point_losses(
                    outputs["pred_masks"],
                    targets,
                    layer_indices[output_index],
                    num_boxes,
                    self.mask_sampling_ratio,
                ).items():
                    losses[f"{name}{suffix}"] = value

        # Union matching across all decoded layers + the pre-output head
        # + the encoder proposals (official ``_get_go_indices``): box losses
        # (final AND every aux layer, baseline aux loop) and the FGL targets
        # ride the union; FGL targets are cached once against the fixed
        # initial anchor and reused by every layer.
        aux_indices = layer_indices[:-1]
        pre_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if pre_group_index is not None:
            pre_indices = indices_to_device(solved[pre_group_index], device)
        enc_indices_list: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        if enc_first_index is not None:
            enc_indices_list = [
                indices_to_device(solved[enc_first_index + enc_index], device)
                for enc_index in range(len(enc_aux_outputs or []))
            ]
        indices_go = self._get_go_indices(
            layer_indices[-1],
            aux_indices + ([pre_indices] if pre_indices else []) + enc_indices_list,
        )
        num_boxes_go = torch.tensor(
            float(sum(int(src.numel()) for src, _ in indices_go)),
            device=num_boxes.device,
        )
        # Baseline normalizes the union size across ranks like num_boxes —
        # a per-rank denominator makes the box/FGL/DDF losses rank-dependent.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go /= torch.distributed.get_world_size()
        num_boxes_go = num_boxes_go.clamp(min=1)

        if pre_outputs is not None:
            losses["loss_vfl_pre"] = loss_labels_vfl(
                pre_outputs, targets, pre_indices, num_boxes, query_mask=query_mask,
                flat_indexing=self.flat_losses,
            )
            # Box losses ride the union assignment (baseline: the pre head
            # is one more entry in the go-indices union for "boxes").
            for name, value in self._loss_boxes(
                pre_outputs, targets, indices_go, num_boxes_go
            ).items():
                losses[f"{name}_pre"] = value
            if self.mask_losses and pre_outputs.get("pred_masks") is not None:
                for name, value in mask_point_losses(
                    pre_outputs["pred_masks"], targets, pre_indices, num_boxes,
                    self.mask_sampling_ratio,
                ).items():
                    losses[f"{name}_pre"] = value

        final_outputs = outputs_list[-1]
        for name, value in self._loss_boxes(
            final_outputs, targets, indices_go, num_boxes_go
        ).items():
            losses[name] = value

        # Per-aux-layer L1+GIoU deep supervision (baseline aux loop: every
        # intermediate layer re-runs the full loss set, and "boxes" rides
        # the union assignment with num_boxes_go normalization — official
        # ``indices_in = indices_go if loss in ["boxes", "local"]``). Without
        # this, intermediate boxes are supervised only indirectly through
        # FGL/DDF. The ``{name}_aux_{i}`` weight expansion below already
        # covers the new keys.
        for output_index in range(num_entries - 1):
            for name, value in self._loss_boxes(
                outputs_list[output_index], targets, indices_go, num_boxes_go
            ).items():
                losses[f"{name}_aux_{output_index}"] = value

        # Cached FGL targets against the fixed initial anchor over the union
        # assignment — identical for every layer (official caching semantics).
        anchor = ref_boxes_list[-1] if ref_boxes_list is not None else None
        if corners_list is not None and anchor is not None and self.flat_losses:
            # Vectorized path (design doc 26 W2): one bbox2distance over the
            # flattened union assignment, then one IoU + one CE per layer.
            # Same math and normalization; sums are reordered.
            flat_q, flat_gt, _ = flatten_match_indices(
                indices_go, anchor.shape[1], anchor.device
            )
            denominator = num_boxes_go.clamp(min=1)
            if flat_q.numel() == 0:
                for output_index in range(num_entries):
                    suffix = (
                        ""
                        if output_index == num_entries - 1
                        else f"_aux_{output_index}"
                    )
                    losses[f"loss_fgl{suffix}"] = corners_list[output_index].sum() * 0.0
            else:
                gt_cxcywh = torch.cat(
                    [
                        targets[batch_index]["boxes"][dst.to(anchor.device)]
                        for batch_index, (_, dst) in enumerate(indices_go)
                        if dst.numel()
                    ]
                )
                lab, w_r, w_l = bbox2distance(
                    anchor.reshape(-1, 4)[flat_q].detach(),
                    box_cxcywh_to_xyxy(gt_cxcywh),
                    self.reg_max,
                    self.reg_scale,
                )
                for output_index in range(num_entries):
                    suffix = (
                        ""
                        if output_index == num_entries - 1
                        else f"_aux_{output_index}"
                    )
                    layer_logits = corners_list[output_index].reshape(
                        -1, 4 * (self.reg_max + 1)
                    )[flat_q].reshape(-1, 4, self.reg_max + 1)
                    layer_boxes = outputs_list[output_index]["pred_boxes"]
                    quality = matched_pair_iou_flat(
                        layer_boxes.reshape(-1, 4)[flat_q].detach(), gt_cxcywh
                    )
                    losses[f"loss_fgl{suffix}"] = fg_loss_from_targets(
                        layer_logits,
                        lab,
                        w_r,
                        w_l,
                        quality,
                        avg_factor=denominator,
                    )
        else:
            fgl_targets: dict | None = None
            if corners_list is not None and anchor is not None:
                fgl_targets = {}
                for batch_index, (src, dst) in enumerate(indices_go):
                    if src.numel() == 0:
                        continue
                    gt_xyxy = box_cxcywh_to_xyxy(
                        targets[batch_index]["boxes"][dst].to(anchor.device)
                    )
                    lab, w_r, w_l = bbox2distance(
                        anchor[batch_index, src].detach(),
                        gt_xyxy,
                        self.reg_max,
                        self.reg_scale,
                    )
                    fgl_targets[batch_index] = (src, dst, lab, w_r, w_l)

            if fgl_targets is not None:
                for output_index in range(num_entries):
                    suffix = (
                        "" if output_index == num_entries - 1 else f"_aux_{output_index}"
                    )
                    layer_corners = corners_list[output_index]
                    layer_boxes = outputs_list[output_index]["pred_boxes"]
                    layer_loss = layer_corners.sum() * 0.0
                    for batch_index, (src, dst, lab, w_r, w_l) in fgl_targets.items():
                        quality = matched_pair_iou(
                            layer_boxes[batch_index, src].detach(),
                            targets[batch_index]["boxes"][dst].to(layer_boxes.device),
                        )
                        layer_loss = layer_loss + fg_loss_from_targets(
                            layer_corners[batch_index, src].reshape(
                                src.numel(), 4, self.reg_max + 1
                            ),
                            lab,
                            w_r,
                            w_l,
                            quality,
                            avg_factor=1.0,
                        )
                    losses[f"loss_fgl{suffix}"] = layer_loss / num_boxes_go.clamp(min=1)

        dn_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        dn_num_boxes: torch.Tensor | None = None
        if dn_outputs and dn_meta is not None:
            dn_indices = get_cdn_matched_indices(dn_meta, targets)
            dn_num_boxes = torch.clamp(num_boxes * dn_meta["dn_num_group"], min=1)
            # The denoising segment of the pre-output head (baseline
            # ``dn_pre_outputs``) gets the same VFL + box pass as the
            # matched pre head, over the CDN assignment.
            if dn_pre_outputs is not None:
                for name, value in self._loss_boxes(
                    dn_pre_outputs, targets, dn_indices, dn_num_boxes
                ).items():
                    losses[f"{name}_dn_pre"] = value
                losses["loss_vfl_dn_pre"] = loss_labels_vfl(
                    dn_pre_outputs, targets, dn_indices, dn_num_boxes,
                flat_indexing=self.flat_losses,
            )
                if self.mask_losses and dn_pre_outputs.get("pred_masks") is not None:
                    for name, value in mask_point_losses(
                        dn_pre_outputs["pred_masks"], targets, dn_indices, dn_num_boxes,
                        self.mask_sampling_ratio,
                    ).items():
                        losses[f"{name}_dn_pre"] = value
            for dn_index, dn_layer in enumerate(dn_outputs):
                suffix = f"_dn_{dn_index}"
                losses[f"loss_vfl{suffix}"] = loss_labels_vfl(
                    dn_layer, targets, dn_indices, dn_num_boxes,
                flat_indexing=self.flat_losses,
            )
                if self.mask_losses and dn_layer.get("pred_masks") is not None:
                    for name, value in mask_point_losses(
                        dn_layer["pred_masks"], targets, dn_indices, dn_num_boxes,
                        self.mask_sampling_ratio,
                    ).items():
                        losses[f"{name}{suffix}"] = value
                for name, value in self._loss_boxes(
                    dn_layer, targets, dn_indices, dn_num_boxes
                ).items():
                    losses[f"{name}{suffix}"] = value
                if dn_corners_list is not None and dn_ref_boxes_list is not None:
                    losses[f"loss_fgl{suffix}"] = fg_loss(
                        dn_corners_list[dn_index],
                        dn_ref_boxes_list[dn_index],
                        dn_layer["pred_boxes"],
                        targets,
                        dn_indices,
                        dn_num_boxes,
                        self.reg_max,
                        self.reg_scale,
                        flat_indexing=self.flat_losses,
                    )

        # DDF: the final layer's distributions teach every intermediate
        # layer over ALL queries (official full form — matched weight IoU,
        # unmatched weight teacher max prob; the num_pos/num_neg counts are
        # computed on this pass and reused by the dn branch).
        if (
            corners_list is not None
            and len(corners_list) >= 2
            and self.ddf_loss_weight > 0
        ):
            for output_index in range(num_entries - 1):
                loss_ddf, num_pos, num_neg = ddf_loss(
                    corners_list[output_index],
                    corners_list[-1],
                    outputs_list[output_index]["pred_logits"],
                    outputs_list[-1]["pred_logits"],
                    outputs_list[output_index]["pred_boxes"],
                    outputs_list[-1]["pred_boxes"],
                    indices_go,
                    self.reg_max,
                    targets=targets,
                    flat_indexing=self.flat_losses,
                )
                self._ddf_num_pos, self._ddf_num_neg = num_pos, num_neg
                losses[f"loss_ddf_aux_{output_index}"] = loss_ddf
            if dn_corners_list and dn_outputs and dn_indices is not None:
                for dn_index in range(len(dn_corners_list) - 1):
                    loss_ddf, _, _ = ddf_loss(
                        dn_corners_list[dn_index],
                        dn_corners_list[-1],
                        dn_outputs[dn_index]["pred_logits"],
                        dn_outputs[-1]["pred_logits"],
                        dn_outputs[dn_index]["pred_boxes"],
                        dn_outputs[-1]["pred_boxes"],
                        dn_indices,
                        self.reg_max,
                        targets=targets,
                        counts=(self._ddf_num_pos, self._ddf_num_neg),
                        flat_indexing=self.flat_losses,
                    )
                    losses[f"loss_ddf_dn_{dn_index}"] = loss_ddf

        # Cross-model distillation (design doc 23 §4.3): the three soft-target
        # legs over the student's union match. Train-only — the eval path
        # never carries distill_targets, and self.training gates it a second
        # time so a stray call cannot enter the branch. The teacher-side
        # join (which GTs the teacher matched) is precomputed by the
        # wrapper; this is where it meets indices_go.
        if self.distill_params is not None and self.training:
            if distill_targets is None:
                raise ValueError(
                    "distill_params is set but the training step carried no "
                    "distill_targets; build the train wrapper with the same "
                    "distill config so the teacher runs each step (design "
                    "doc 23 §4.4)."
                )
            if distill_targets.get("gated_out"):
                # start_step / every_n_steps miss: no teacher pass ran this
                # step; don't leak the previous hit's counts into the log.
                self.last_distill_stats = None
            else:
                stats_sink: dict[str, int] = {}
                for name, value in cross_model_distill_losses(
                    outputs_list[-1],
                    corners_list[-1] if corners_list is not None else None,
                    targets,
                    indices_go,
                    num_boxes_go,
                    distill_targets,
                    self.distill_params,
                    stats_sink,
                ).items():
                    losses[name] = value
                self.last_distill_stats = stats_sink
        elif self.distill_params is not None:
            # Eval mode: never leak the previous training step's counts.
            self.last_distill_stats = None

        # DeFE auxiliary losses (density-guided dynamic queries). The GT
        # heatmap is only present in training; min/max_num_select ride
        # defe_info so the decoder config stays the single source of truth.
        # The loss weights live exclusively in weight_dict (applied once
        # in the aggregation below) — no inline multiplication.
        if defe_info is not None and "gt_heatmap" in defe_info:
            losses["defe_density_loss"] = defe_density_loss(
                defe_info["density_map"],
                defe_info["gt_heatmap"],
                underestimation_penalty=self.defe_underestimation_penalty,
            )
            losses["defe_count_loss"] = defe_count_loss(
                defe_info["reg_value"],
                [int(t["boxes"].shape[0]) for t in targets],
                min_num_select=defe_info["min_num_select"],
                max_num_select=defe_info["max_num_select"],
                underestimation_weight=self.defe_count_underestimation_weight,
            )

        # Per-object contrastive SSL objective (design doc 14). Final-layer
        # only — the aux/dn branches below repeat every entry of the loss
        # set with suffixed keys, and their outputs carry no SSL features.
        # Eval mode never enters the contrastive path (no gather
        # collectives on the validation branch). A training-mode call with
        # ssl_params but missing features is a head/criterion configuration
        # mismatch — fail loud instead of silently training without SSL.
        if self.ssl_params is not None and self.training:
            if not (
                isinstance(ssl_z1, torch.Tensor) and isinstance(ssl_z2, torch.Tensor)
            ):
                raise ValueError(
                    "ssl_params is set but the model outputs carry no "
                    "ssl_z1/ssl_z2 features; build the task model with the "
                    "same ssl config so the SSLProjectionHead is attached "
                    "(design doc 14 §4.4)."
                )
            stride = self.ssl_params.every_n_steps
            if stride <= 1 or self.ssl_step % stride == 0:
                ssl_stats: dict[str, float] = {}
                losses["loss_ssl"] = ssl_contrastive_loss(
                    ssl_z1,
                    ssl_z2,
                    outputs_list[-1],
                    targets,
                    layer_indices[-1],
                    self.ssl_params,
                    query_mask=query_mask,
                    stats_sink=ssl_stats,
                )
                if self.ssl_params.sigreg_coef > 0:
                    if not (
                        isinstance(ssl_s1, torch.Tensor)
                        and isinstance(ssl_s2, torch.Tensor)
                    ):
                        raise ValueError(
                            "ssl sigreg_coef>0 requires the model outputs to "
                            "carry ssl_s1/ssl_s2 raw projector taps; build the "
                            "task model with the same ssl sigreg config "
                            "(design doc 16 R1-A1)."
                        )
                    # Per-view halves of the query pool, batch-flattened to
                    # (B*Q, D). The reference pools the group-0 match per
                    # view; here the regularizer covers the full query
                    # embedding distribution (keeps the unmatched-query
                    # half of the cloud Gaussian too). Each half is a
                    # per-sample mean, so no pairing and no gather. Padding
                    # queries of dynamic-query decoders are EXCLUDED: their
                    # hidden states are decoder computations on zero content
                    # queries, and folding them into the distribution target
                    # would pollute the statistic (the matched-pool
                    # contrastive already filters the same rows via
                    # query_mask).
                    s1_pool = ssl_s1.reshape(-1, ssl_s1.shape[-1])
                    s2_pool = ssl_s2.reshape(-1, ssl_s2.shape[-1])
                    if query_mask is not None:
                        mask_flat = query_mask.reshape(-1)
                        s1_pool = s1_pool[mask_flat]
                        s2_pool = s2_pool[mask_flat]
                    losses["loss_sigreg"] = 0.5 * (
                        epps_pulley_sigreg(
                            s1_pool,
                            num_slices=self.ssl_params.sigreg_slices,
                            num_knots=self.ssl_params.sigreg_knots,
                        )
                        + epps_pulley_sigreg(
                            s2_pool,
                            num_slices=self.ssl_params.sigreg_slices,
                            num_knots=self.ssl_params.sigreg_knots,
                        )
                    )
                self.last_ssl_stats = ssl_stats
            else:
                # Stride miss: don't let the previous hit's diagnostics
                # leak into this step's log.
                self.last_ssl_stats = None

        # Zero-fill any NaN term (degenerate empty-GT / zero-area-box paths)
        # so one poisoned branch cannot NaN the total: nan_to_num's backward
        # routes zero gradient to the NaN input, so the term drops out of
        # the step instead of corrupting it. The engine's non-finite guard
        # still catches a NaN TOTAL — the failure mode this must not mask.
        losses = {
            key: torch.nan_to_num(value, nan=0.0) for key, value in losses.items()
        }
        # Encoder-proposal supervision (baseline enc_aux chain — see the
        # rtdetrv2 criterion for rationale): the enc top-k selection is
        # non-differentiable and the decoder inputs are detached, so
        # without this branch the proposal heads never receive a gradient.
        # Full VFL pass on its own match; box losses ride the union
        # assignment (baseline ``use_uni_set``).
        if enc_aux_outputs:
            class_agnostic = bool(enc_meta.get("class_agnostic", False))
            for enc_index, enc_outputs in enumerate(enc_aux_outputs):
                enc_targets = (
                    [
                        {"boxes": t["boxes"], "labels": torch.zeros_like(t["labels"])}
                        for t in targets
                    ]
                    if class_agnostic
                    else targets
                )
                suffix = f"_enc_{enc_index}"
                losses[f"loss_vfl{suffix}"] = loss_labels_vfl(
                    enc_outputs,
                    enc_targets,
                    enc_indices_list[enc_index],
                    num_boxes,
                flat_indexing=self.flat_losses,
            )
                for name, value in self._loss_boxes(
                    enc_outputs, enc_targets, indices_go, num_boxes_go
                ).items():
                    losses[f"{name}{suffix}"] = value

        weights = dict(self.weight_dict)
        iou_key = self._iou_loss_key
        if pre_outputs is not None:
            weights["loss_vfl_pre"] = self.weight_dict["loss_vfl"]
            weights["loss_bbox_pre"] = self.weight_dict["loss_bbox"]
            weights[f"{iou_key}_pre"] = self.weight_dict[iou_key]
        if dn_pre_outputs is not None:
            weights["loss_vfl_dn_pre"] = self.weight_dict["loss_vfl"]
            weights["loss_bbox_dn_pre"] = self.weight_dict["loss_bbox"]
            weights[f"{iou_key}_dn_pre"] = self.weight_dict[iou_key]
        if self.mask_losses:
            for mask_suffix in ("_pre", "_dn_pre"):
                weights[f"loss_mask_ce{mask_suffix}"] = self.weight_dict["loss_mask_ce"]
                weights[f"loss_mask_dice{mask_suffix}"] = self.weight_dict["loss_mask_dice"]
        for enc_index in range(len(enc_aux_outputs or [])):
            for key, value in self.weight_dict.items():
                weights[f"{key}_enc_{enc_index}"] = value
        for output_index in range(num_entries - 1):
            for key, value in self.weight_dict.items():
                weights[f"{key}_aux_{output_index}"] = value
        for dn_index in range(len(dn_outputs or [])):
            for key, value in self.weight_dict.items():
                weights[f"{key}_dn_{dn_index}"] = value
            if self.dn_fgl_loss_weight is not None:
                weights[f"loss_fgl_dn_{dn_index}"] = self.dn_fgl_loss_weight

        total = sum(
            losses[key] * weight for key, weight in weights.items() if key in losses
        )
        losses["loss"] = total
        # Keep the loss graph-attached: DDP averages gradients across ranks,
        # so the training loss must not be all-reduced/detached here.
        # ``reduce_dict`` is for logging only and would break backward.
        return losses

    def _get_go_indices(
        self,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        indices_aux_list: list[list[tuple[torch.Tensor, torch.Tensor]]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Union of the per-layer Hungarian assignments (official form).

        Accumulates every layer's (query, gt) pairs, keeps pairs seen in
        multiple layers first, and de-duplicates per query so each query
        appears once with its most-frequent GT.

        Vectorized form of the official sequential loop: unique rows sorted
        by count (descending, STABLE so equal counts keep the lexicographic
        order of ``torch.unique``), then the first occurrence of each query
        row — its highest-count assignment (earliest-lexicographic on
        ties). Rows are emitted in first-appearance order, matching the
        official dict-insertion order. No ``.item()`` round-trips: the
        per-row Python loop cost grew with VisDrone's union size (hundreds
        of rows per image, twice per training step).
        """
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]
        for ind in [
            torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices
        ]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            # Stable count-descending sort: equal counts keep torch.unique's
            # lexicographic order. A query row's pairs are NOT contiguous in
            # this order, so group by row first (stable keeps the
            # count-desc order within a row) — each group's first entry is
            # that row's highest-count assignment (earliest-lexicographic
            # on ties), exactly what the old sequential dict loop kept.
            unique_sorted = unique[torch.argsort(counts, descending=True, stable=True)]
            row_order = torch.argsort(unique_sorted[:, 0], stable=True)
            grouped_rows = unique_sorted[row_order, 0]
            group_mask = torch.ones(
                grouped_rows.shape[0], dtype=torch.bool, device=grouped_rows.device
            )
            group_mask[1:] = grouped_rows[1:] != grouped_rows[:-1]
            positions = row_order[group_mask.nonzero(as_tuple=True)[0]]
            # Ascending positions reproduce the old loop's visit (and
            # dict-insertion) order.
            positions = positions.sort().values
            results.append(
                (
                    unique_sorted[positions, 0].long(),
                    unique_sorted[positions, 1].long(),
                )
            )
        return results
