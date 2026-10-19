# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DETR object detection training wrapper.

Mirrors the RT-DETR training wrapper (criterion, EMA, flat-cosine,
metrics) with baseline DETR LR grouping:

- ViT variants: only the pretrained ViT trains at the backbone LR — the
  STA prior and the 1x1 projection layers are randomly initialized and
  merged into the detector group.
- ConvNeXt variants: the whole backbone trains at the backbone LR.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import replace
from typing import Any

import torch

from autovision.augment.detection.box_aware_eraser import (
    protected_cell_mask,
    sample_erase_mask,
)
from autovision.errors import CheckpointError
from autovision.metrics.map_evaluator import COCOMAPEvaluator
from autovision.runtime.env import Env
from autovision.tasks.detection.detr.config import (
    DETRDistillArgs,
    DETRObjectDetectionTrainArgs,
    DETRTokenDropArgs,
    resolve_detr_variant,
)
from autovision.tasks.detection.detr.optimizer import (
    build_detr_optimizer,
)
from autovision.tasks.detection.detr.task_model import (
    DETRObjectDetection,
)
from autovision.tasks.detection.shared.dfine_criterion import (
    DFINECriterionV2,
)
from autovision.tasks.detection.shared.matcher import (
    HungarianMatcher,
)
from autovision.tasks.detection.shared.rtdetrv2_criterion import (
    RTDETRCriterionV2,
)
from autovision.tasks.shared.ema import ModelEMA
from autovision.tasks.train_model import TaskStepResult, TrainModel

logger = logging.getLogger("autovision")


def _build_window_keep_masks(
    boxes_list: list[torch.Tensor],
    height: int,
    width: int,
    args: DETRTokenDropArgs,
) -> tuple[list[torch.Tensor | None], float]:
    """Per-sample window keep masks for in-backbone token dropping (doc 17).

    Selection reuses the box-aware eraser's protection semantics (doc 16)
    at ``window_size`` granularity: windows overlapping a margin-dilated GT
    box are always kept, a uniform random ``keep_ratio`` subset of the
    background windows survives. Runs in ``training_step`` on the FINAL
    canvas (after batch scale jitter), so the grid semantics have a single
    definition point and transform/collate stay untouched — the reference
    implementation's dataset-wrapper masks silently missed the model grid
    by a whole factor (doc 17 §2.3).

    The boxes are the CPU-side collate outputs read BEFORE the device
    transfer: the host-side per-sample math on on-device tensors would
    cost 2-4 GPU→CPU syncs per sample per step. Returned masks are CPU
    tensors; the model's mask extraction moves them to the batch device.

    A sample runs dense (``None`` mask, keeping the batched path) when its
    coin fails, when it has no boxes, when the canvas is not divisible by
    ``window_size``, or when the draw happens to keep every window.
    """
    masks: list[torch.Tensor | None] = []
    keep_fractions: list[float] = []
    for boxes in boxes_list:
        if args.prob < 1.0 and torch.rand(()).item() >= args.prob:
            masks.append(None)
            continue
        if boxes is None or boxes.numel() == 0:
            masks.append(None)
            continue
        if height % args.window_size or width % args.window_size:
            masks.append(None)
            continue
        protected = protected_cell_mask(
            boxes,
            height,
            width,
            cell=args.window_size,
            padding=args.padding,
        )
        keep = ~sample_erase_mask(protected, 1.0 - args.keep_ratio)
        if bool(keep.all()):
            masks.append(None)
            continue
        masks.append(keep)
        keep_fractions.append(float(keep.float().mean().item()))
    mean_keep = (
        sum(keep_fractions) / len(keep_fractions) if keep_fractions else 1.0
    )
    return masks, mean_keep

def _load_teacher_from_export(path) -> torch.nn.Module:
    """Rebuild a task model from an AutoVision exported ``.pt`` (doc 23 D2).

    The engine's export payload (``engine.export_weights``) carries the task
    model's ``model_class_path``/``model_init_args`` and a wrapper-level
    state dict; the wrapper prefixes (``model.`` for a bare export,
    ``ema_model.model.`` when EMA was on) are stripped so the checkpoint's
    best weights — EMA weights whenever EMA ran — land on a bare task
    model. Current exports load through the pickle-free path. Legacy files
    that embed a ``__class__`` object are rejected unless the operator
    explicitly opts into the full unpickler for a trusted local artifact
    (``AUTOVISION_TRUST_TEACHER_CHECKPOINT=1``).
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        if not Env.TRUST_TEACHER_CHECKPOINT.value:
            raise CheckpointError(
                f"Teacher checkpoint '{path}' cannot be read with the "
                "pickle-free loader (legacy export or unsafe writer). "
                "Re-export it from the current framework, or — only for a "
                "trusted local artifact — set "
                "AUTOVISION_TRUST_TEACHER_CHECKPOINT=1 to use the full "
                "unpickler."
            ) from error
        logger.warning(
            "distill teacher_checkpoint '%s' requires the full unpickler; "
            "AUTOVISION_TRUST_TEACHER_CHECKPOINT=1 is set, loading as a "
            "trusted local artifact.",
            path,
        )
        # Reached only behind the explicit trusted-local
        # AUTOVISION_TRUST_TEACHER_CHECKPOINT opt-in.
        payload = torch.load(path, map_location="cpu", weights_only=False)  # nosec B614
    for key in ("model_class_path", "model_init_args", "train_model"):
        if key not in payload:
            raise ValueError(
                f"distill teacher_checkpoint '{path}' is not an AutoVision "
                f"export (missing '{key}')."
            )
    module_path, _, class_name = str(payload["model_class_path"]).rpartition(".")
    task_cls = getattr(importlib.import_module(module_path), class_name)
    teacher = task_cls(**payload["model_init_args"])
    state = payload["train_model"]
    for prefix in ("ema_model.model.", "model.", "ema_model."):
        if any(key.startswith(prefix) for key in state):
            state = {
                key[len(prefix):]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            break
    teacher.load_state_dict(state)
    return teacher


def _build_gt_to_teacher_query(
    teacher_indices: list[tuple[torch.Tensor, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    """Per-image GT→teacher-query map, −1 where the teacher missed the GT.

    The teacher runs its own Hungarian match against the same targets as
    the student (doc 23 D1); this map is the join key the criterion
    applies against the student's union match.
    """
    max_gt = max((int(t["boxes"].shape[0]) for t in targets), default=0)
    t_map = torch.full(
        (len(targets), max(max_gt, 1)), -1, dtype=torch.long, device=device
    )
    for batch_index, (src_t, dst_t) in enumerate(teacher_indices):
        t_map[batch_index, dst_t.to(device).long()] = src_t.to(device).long()
    return t_map


__all__ = ["DETRObjectDetectionTrain"]


class DETRObjectDetectionTrain(TrainModel):
    task = "object_detection"
    watch_metric = "val_metric/map"

    def _build_task_model(
        self, model_args: DETRObjectDetectionTrainArgs
    ) -> DETRObjectDetection:
        """Construct the task model from the train args.

        Extracted so task subclasses (e.g. the GTR instance-segmentation
        family) can override the construction without copying the ctor.
        """
        return DETRObjectDetection(
            num_classes=model_args.num_classes,
            model_name=model_args.backbone,
            image_size=model_args.image_size,
            decoder=model_args.decoder,
            reg_max=model_args.reg_max,
            reg_scale=model_args.reg_scale,
            enc_dim_feedforward=model_args.enc_dim_feedforward,
            enc_expansion=model_args.enc_expansion,
            enc_depth_mult=model_args.enc_depth_mult,
            hidden_dim=model_args.hidden_dim
            if model_args.hidden_dim != "auto"
            else None,
            decoder_hidden_dim=model_args.decoder_hidden_dim,
            num_queries=model_args.num_queries,
            dec_layers=model_args.dec_layers
            if model_args.dec_layers != "auto"
            else None,
            num_heads=model_args.num_heads if model_args.num_heads != "auto" else None,
            num_top_queries=model_args.num_top_queries,
            num_denoising=model_args.num_denoising,
            label_noise_ratio=model_args.label_noise_ratio,
            box_noise_scale=model_args.box_noise_scale,
            dn_max_gt_cap=model_args.dn_max_gt_cap,
            backbone_pretrained=model_args.backbone_pretrained,
            backbone_weights=model_args.backbone_weights,
            load_weights=model_args.backbone_pretrained,
            pyramid_levels=model_args.pyramid_levels,
            density=model_args.density,
            ssl=model_args.ssl,
            token_drop=model_args.token_drop,
            open_vocab=model_args.open_vocab,
        )

    def __init__(self, model_args: DETRObjectDetectionTrainArgs) -> None:
        task_model = self._build_task_model(model_args)
        super().__init__(task_model=task_model)
        self.model_args = model_args
        if model_args.warm_start is not None:
            from autovision.tasks.detection.detr.task_model import (
                load_warm_start_into_network,
            )

            summary = load_warm_start_into_network(task_model, model_args.warm_start)
            logger.info(
                "AVWarmStart '%s' loaded: %s", model_args.warm_start, summary
            )
        if model_args.detector_checkpoint is not None:
            from autovision.tasks.detection.detr.coco_checkpoint import (
                load_baseline_coco_checkpoint,
                resolve_detector_checkpoint,
            )

            checkpoint_path = resolve_detector_checkpoint(
                model_args.detector_checkpoint
            )
            summary = load_baseline_coco_checkpoint(task_model, checkpoint_path)
            logger.info(
                "baseline detector checkpoint '%s' loaded: %s",
                checkpoint_path, summary,
            )
        self.matcher = HungarianMatcher(
            cost_mask_ce=(
                model_args.mask_cost_ce_weight if model_args.mask_losses else 0.0
            ),
            cost_mask_dice=(
                model_args.mask_cost_dice_weight if model_args.mask_losses else 0.0
            ),
            mask_point_sample_ratio=(
                model_args.mask_sampling_ratio if model_args.mask_losses else None
            ),
        )
        # EMA is created before the criterion so the ``self:ema`` distill
        # teacher can alias this copy (design doc 23 §4.2); deepcopy draws
        # no RNG, so moving it changes nothing else about init.
        self.ema_model: ModelEMA | None = (
            ModelEMA(
                task_model,
                decay=model_args.ema_momentum,
                warmup_steps=model_args.ema_warmup_steps,
            )
            if model_args.use_ema_model
            else None
        )
        decoder = model_args.decoder
        if decoder == "auto":
            _, variant = resolve_detr_variant(f"detr/{model_args.backbone}")
            decoder = variant.decoder_default
        density = model_args.density
        ssl_args = model_args.ssl
        # The SSL criterion wiring only exists on the D-FINE criterion (v1,
        # design doc 14 §3.3); the batch-matched baseline (mode="off") needs
        # no wiring anywhere.
        ssl_params = (
            ssl_args.to_ssl_params()
            if ssl_args is not None and ssl_args.mode != "off"
            else None
        )
        if ssl_args is not None and decoder != "dfine":
            raise ValueError(
                "ssl (per-object contrastive head) requires the D-FINE "
                "decoder in v1; pass decoder='dfine' (design doc 14 §3.3)."
            )
        # The SIoU swap (design doc 15) lives in the same D-FINE criterion;
        # a non-dfine decoder with the flag set would silently train giou.
        if model_args.box_regression_iou == "siou" and decoder not in (
            "dfine",
            "dfine_official",
        ):
            raise ValueError(
                "box_regression_iou='siou' requires the D-FINE decoder "
                "in v1; pass decoder='dfine' (design doc 15 §4.5)."
            )
        # Cross-model distillation (design doc 23). The teacher is built
        # BEFORE the criterion so the class-space rule — which may zero the
        # classification leg — lands in the ctor's DistillParams. The
        # teacher is stored via object.__setattr__ (never a registered
        # submodule): it must stay out of state_dict / optimizer groups /
        # EMA / export for the whole run.
        self._distill_args: DETRDistillArgs | None = model_args.distill
        self._distill_teacher: torch.nn.Module | None = None
        self._distill_matcher: HungarianMatcher | None = None
        self._distill_ddf_warned = False
        distill_params = None
        if self._distill_args is not None:
            if decoder != "dfine":
                raise ValueError(
                    "distill requires the D-FINE decoder in v1 (the DDF leg "
                    "rides the fork decoder's side distributions); pass "
                    "decoder='dfine' (design doc 23 §4.1)."
                )
            self._distill_matcher = HungarianMatcher()
            teacher = self._build_distill_teacher(
                self._distill_args, student_num_classes=model_args.num_classes
            )
            object.__setattr__(self, "_distill_teacher", teacher)
            effective_cls = self._distill_args.resolved_cls_coef(
                int(teacher.num_classes), model_args.num_classes
            )
            distill_params = self._distill_args.to_distill_params()
            if effective_cls != distill_params.cls_coef:
                distill_params = replace(distill_params, cls_coef=effective_cls)

        if decoder in ("dfine", "dfine_official"):
            self.criterion: RTDETRCriterionV2 | DFINECriterionV2 = DFINECriterionV2(
                self.matcher,
                reg_max=model_args.reg_max,
                reg_scale=model_args.reg_scale,
                dn_fgl_loss_weight=model_args.dn_fgl_loss_weight,
                defe_density_map_weight=(
                    density.density_map_weight if density else 1.0
                ),
                defe_count_loss_weight=density.count_loss_weight if density else 1.0,
                defe_underestimation_penalty=(
                    density.density_underestimation_penalty if density else 0.3
                ),
                defe_count_underestimation_weight=(
                    density.count_underestimation_weight if density else 2.0
                ),
                ssl_params=ssl_params,
                box_regression_iou=model_args.box_regression_iou,
                siou_loss_coef=model_args.siou_loss_coef,
                siou_theta=model_args.siou_theta,
                flat_losses=model_args.criterion_flat_losses,
                # distill_params carries the class-space-adjusted cls_coef
                # (the teacher is built above, before this ctor).
                distill_params=distill_params,
                mask_losses=model_args.mask_losses,
                mask_sampling_ratio=model_args.mask_sampling_ratio,
                mask_ce_weight=model_args.mask_ce_weight,
                mask_dice_weight=model_args.mask_dice_weight,
            )
        else:
            self.criterion = RTDETRCriterionV2(self.matcher)
        self.categories: dict[int, str] = {}
        self._max_dets = model_args.eval_max_dets
        self._evaluators: dict[str, COCOMAPEvaluator] = {}
        # Route B (SSL-only pretraining) flag, resolved once (design doc 14 §4.10).
        self._ssl_only = (
            ssl_args is not None and ssl_args.instance_source == "encoder_candidates"
        )
        if self._ssl_only:
            # Best-checkpoint signal for stage 1: "val_ssl_loss" is a key the
            # val metrics never carry, so the engine's
            # ``get(watch_key, -val_loss)`` fallback watches the NEGATED
            # world-averaged validation loss (the held-out contrastive
            # loss) — exactly the lower-is-better signal needed. The class
            # attribute stays "val_metric/map" for stage 2 / joint runs.
            self.watch_metric = "val_ssl_loss"
        # Density-guided query diagnostics accumulated over the val split
        # (per-image (query_count, gt_count)); consumed by compute_metrics
        # so the engine persists them (validation log_dict is dropped).
        self._val_query_stats: list[tuple[int, int]] = []
        # Fraction of MWAS windows the adaptive mask selected, per val
        # image — the gate-tightness readout: 1.0 means the density gate
        # passes every window (random-init DeFE head), shrinking toward
        # the dense-region fraction as the head converges (doc 13 §4.2).
        self._val_window_ratios: list[float] = []

    # ── steps ───────────────────────────────────────────────────────────
    def _build_targets(
        self, batch: dict[str, Any], x: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        """Batch -> criterion targets (boxes/labels on the input device).

        Task subclasses add their extra target keys (masks) by overriding.
        """
        return [
            {"boxes": boxes.to(x.device), "labels": labels.to(x.device)}
            for boxes, labels in zip(batch["bboxes"], batch["classes"])
        ]

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        # In-backbone background token dropping (design doc 17): masks are
        # derived here on the final canvas and ride the targets into the
        # model forward. Built from the CPU-side collate boxes BEFORE the
        # device transfer (host-side per-sample math; on-device boxes would
        # cost 2-4 GPU→CPU syncs per sample per step). Step gate mirrors the
        # transform's erasure gate: an unresolved "auto" or None keeps it
        # on, a resolved int closes the window (Dome aug-stop cadence —
        # training ends dense).
        token_keep_ratio: float | None = None
        window_masks: list[torch.Tensor | None] | None = None
        token_drop = self.model_args.token_drop
        if token_drop is not None:
            stop = token_drop.step_stop
            if not isinstance(stop, int) or step < stop:
                window_masks, token_keep_ratio = _build_window_keep_masks(
                    list(batch["bboxes"]),
                    int(x.shape[-2]),
                    int(x.shape[-1]),
                    token_drop,
                )
        targets = self._build_targets(batch, x)
        if window_masks is not None and any(m is not None for m in window_masks):
            for target, mask in zip(targets, window_masks):
                if mask is not None:
                    target["window_keep_mask"] = mask
        out = self.model._forward_train(x, targets=targets)
        outputs_list = list(out.get("aux_outputs", [])) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        # Density-guided extras (design doc 13 §4.6): the query-validity
        # mask keeps padded queries out of the classification loss; the
        # defe dict carries the density map / GT heatmap / count estimate.
        query_mask = out.get("query_valid_mask")
        defe_info = out.get("defe_info")
        # Cross-model distillation (design doc 23): the teacher pass runs
        # under inference_mode and hands the criterion raw tensors + the
        # GT→teacher-query join map. Gated steps carry the `gated_out`
        # marker instead of a teacher payload so the criterion can tell a
        # schedule miss from a wiring error.
        distill_targets = None
        if self._distill_teacher is not None:
            distill_targets = self._build_teacher_targets(x, targets, step)
        if isinstance(self.criterion, DFINECriterionV2):
            # SSL stride state lives on the criterion, written per step
            # (design doc 14 §4.4); the features come from the model's
            # train-only projection head.
            self.criterion.ssl_step = step
            loss_dict = self.criterion(
                outputs_list,
                targets,
                world_size=1,
                dn_outputs=out.get("dn_aux_outputs", []),
                dn_meta=out.get("dn_meta"),
                dn_pre_outputs=out.get("dn_pre_outputs"),
                corners_list=out.get("pred_corners_list"),
                ref_boxes_list=out.get("ref_boxes_list"),
                dn_corners_list=out.get("dn_pred_corners_list"),
                dn_ref_boxes_list=out.get("dn_ref_boxes_list"),
                query_mask=query_mask,
                defe_info=defe_info,
                pre_outputs=out.get("pre_outputs"),
                enc_aux_outputs=out.get("enc_aux_outputs"),
                enc_meta=out.get("enc_meta"),
                ssl_z1=out.get("ssl_z1"),
                ssl_z2=out.get("ssl_z2"),
                ssl_s1=out.get("ssl_s1"),
                ssl_s2=out.get("ssl_s2"),
                distill_targets=distill_targets,
            )
        else:
            loss_dict = self.criterion(
                outputs_list,
                targets,
                world_size=1,
                dn_outputs=out.get("dn_aux_outputs", []),
                dn_meta=out.get("dn_meta"),
                query_mask=query_mask,
                enc_aux_outputs=out.get("enc_aux_outputs"),
                enc_meta=out.get("enc_meta"),
            )
        # Detached GPU tensors: the engine floats them only when logging
        # (every log_every_n_steps); per-item float() here would sync the
        # GPU once per loss term on every step.
        log_dict = {key: value.detach() for key, value in loss_dict.items()}
        if token_keep_ratio is not None:
            # Doc 17 §4.4: kept-window fraction over the masks applied this
            # step — the online sparsity readout, and the VisDrone
            # droppability meter behind gate G1 (dome_window_ratio twin).
            log_dict["token_keep_ratio"] = token_keep_ratio
        # SSL pool diagnostics (design doc 14 D10): paired_ratio is the
        # fraction of view-1 instances that found a validated cross-view
        # partner — a two_crop pair consumes two pool entries, so the
        # denominator is pool/2 (1.0 = perfect pairing); under the dropout
        # view every instance pairs trivially (denominator = pool).
        ssl_stats = getattr(self.criterion, "last_ssl_stats", None)
        if ssl_stats:
            pool = float(ssl_stats.get("pool_size", 0.0))
            paired = float(ssl_stats.get("paired_size", 0.0))
            ssl_view = (
                self.criterion.ssl_params.view
                if self.criterion.ssl_params
                else "two_crop"
            )
            denominator = max(pool / 2.0 if ssl_view == "two_crop" else pool, 1.0)
            log_dict["ssl_pool_size"] = pool
            log_dict["ssl_paired_ratio"] = paired / denominator
            if "candidate_size" in ssl_stats:
                # D14: paired + unpaired-reclaimed candidate columns.
                log_dict["ssl_candidate_size"] = float(ssl_stats["candidate_size"])
        # Distillation coverage diagnostics (design doc 23 §3.4): the
        # fraction of union-matched pairs whose GT the teacher also
        # matched. Sustained < 0.7 means the teacher under-covers the
        # domain — a signal to lower λ or pick a better teacher.
        distill_stats = getattr(self.criterion, "last_distill_stats", None)
        if distill_stats:
            union = float(distill_stats.get("union", 0))
            covered = float(distill_stats.get("covered", 0))
            log_dict["distill_union"] = union
            log_dict["distill_coverage"] = covered / max(union, 1.0)
        # Open-vocab diagnostics (design doc 27 §3.9): the learnable
        # similarity scale in log-space (CLIP convention — exp() it to
        # read the actual cosine multiplier).
        decoder = getattr(self.model, "decoder", None)
        if getattr(decoder, "_ov_text_embed_dim", None) is not None:
            first_head = decoder.dec_score_head[0]
            if hasattr(first_head, "logit_scale"):
                log_dict["ov_logit_scale"] = first_head.logit_scale.detach()
        return TaskStepResult(loss=loss_dict["loss"], log_dict=log_dict)

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        # NOTE: the D-FINE criterion receives no corners here, so the val
        # loss covers the base VFL/bbox/GIoU terms only (FGL/DDF are
        # train-time distributional losses). Val and train losses are
        # therefore not directly comparable for decoder="dfine".
        x = batch["image"]
        model = self.ema_model.model if self.ema_model is not None else self.model
        query_mask = None
        queries_num = None
        ssl_z1 = ssl_z2 = None
        if self._ssl_only:
            # Route B: the decoder emits candidate features in eval too, so
            # the validation loss is the held-out contrastive loss (design
            # doc 14 §4.10). The mAP eval below still runs — its curve is
            # expected to be noise until stage-2 finetuning.
            #
            # The projection head is switched to train mode for the forward:
            # in eval its dropout is disabled and the two draws coincide
            # exactly, degrading the held-out loss to a uniformity probe
            # instead of the train objective's two-view estimate. Restored
            # immediately after so no state leaks.
            head = getattr(model, "ssl_head", None)
            if head is not None:
                head.train()
            try:
                raw = model._forward_train(x)
            finally:
                if head is not None:
                    head.eval()
            outputs = (raw["pred_logits"], raw["pred_boxes"])
            ssl_z1 = raw.get("ssl_z1")
            ssl_z2 = raw.get("ssl_z2")
        elif getattr(model, "_density_enabled", False):
            # Dynamic-query models: the raw forward dict carries the
            # per-image query counts (dome metrics) and the padded-tail
            # query mask (design doc 13 §4.6).
            raw = model._forward_train(x)
            outputs = (raw["pred_logits"], raw["pred_boxes"])
            query_mask = raw.get("query_valid_mask")
            queries_num = raw.get("batch_queries_num")
            defe_info = raw.get("defe_info")
            if isinstance(defe_info, dict) and "window_mask" in defe_info:
                window_mask = defe_info["window_mask"]
                self._val_window_ratios.extend(
                    window_mask.float().mean(dim=(1, 2)).detach().tolist()
                )
        else:
            outputs = model(x)
        decoded = model.postprocess(
            outputs,
            [{"orig_w": w, "orig_h": h} for w, h in batch["original_size"]],
            # Baseline/COCO protocol: NO score gate before the mAP eval —
            # pycocotools ranks everything up to maxDets. A 0.05 gate drops
            # the low-confidence detections that early-epoch recall lives
            # on and understates mAP.
            threshold=0.0,
        )
        evaluator = self._evaluators.setdefault(
            "val", COCOMAPEvaluator(self.categories, max_dets=self._max_dets)
        )
        evaluator.update(decoded)
        evaluator.update_targets(
            [
                {"boxes": boxes, "labels": labels}
                for boxes, labels in zip(batch["bboxes"], batch["classes"])
            ],
            original_sizes=batch["original_size"],
        )
        if isinstance(outputs, dict):
            outputs_list = [outputs]
        elif isinstance(outputs, tuple):
            outputs_list = [{"pred_logits": outputs[0], "pred_boxes": outputs[1]}]
        else:  # ObjectDetectionOutput dataclass
            outputs_list = [
                {"pred_logits": outputs.logits, "pred_boxes": outputs.boxes}
            ]
        criterion_kwargs: dict[str, Any] = {}
        if isinstance(self.criterion, DFINECriterionV2):
            # Route B: the SSL-only branch consumes the candidate features;
            # the auxiliary path ignores them (absent in eval).
            criterion_kwargs = {"ssl_z1": ssl_z1, "ssl_z2": ssl_z2}
        losses = self.criterion(
            outputs_list,
            [
                {"boxes": b.to(x.device), "labels": c.to(x.device)}
                for b, c in zip(batch["bboxes"], batch["classes"])
            ],
            world_size=1,
            query_mask=query_mask,
            **criterion_kwargs,
        )
        log_dict = {key: float(value) for key, value in losses.items()}
        if queries_num is not None:
            # Dome diagnostics (baseline "defe Ample Rate" / "Average
            # Anchor Number", design doc 13 §1.3): how often the dynamic
            # selection keeps at least one query per GT, and the mean
            # query count (efficiency readout against the fixed baseline).
            # Accumulated per image and surfaced through compute_metrics.
            gt_counts = [int(boxes.shape[0]) for boxes in batch["bboxes"]]
            self._val_query_stats.extend(zip(queries_num, gt_counts))
            log_dict["dome_avg_queries"] = float(
                sum(queries_num) / max(len(queries_num), 1)
            )
            log_dict["dome_ample_rate"] = float(
                sum(n >= g for n, g in zip(queries_num, gt_counts))
                / max(len(queries_num), 1)
            )
        return TaskStepResult(loss=losses["loss"], log_dict=log_dict)

    def set_categories(self, categories: dict[int, str]) -> None:
        self.categories = dict(categories)
        # Open-vocab cross-check (design doc 27 §3.3 / risk R7): the label
        # space IS the class_names order, so a dataset whose categories
        # disagree with the prompt vocabulary would silently mislabel every
        # box. Fail at build time instead.
        open_vocab = getattr(self.model_args, "open_vocab", None)
        if open_vocab is not None and categories:
            dataset_names = [categories[cid] for cid in sorted(categories)]
            if dataset_names != list(open_vocab.class_names):
                raise ValueError(
                    "open_vocab class_names do not match the dataset "
                    f"categories: class_names={list(open_vocab.class_names)} "
                    f"vs dataset={dataset_names}. The vocabulary order must "
                    f"reproduce the dataset label ids exactly."
                )

    def validation_report(self, metrics: dict[str, float]) -> str | None:
        """Box tables (overall + per-class) for the engine's val log line.

        ``metrics`` is the prefixed ``val_metric/*`` dict the engine just
        computed; the per-class rows come from the evaluator's last
        ``compute()``. Returning ``None`` is never expected here, but the
        engine treats any exception as "no report" and falls back to the
        one-line summary.
        """
        from autovision.metrics.report import format_detection_validation_report

        scalars = {
            key.split("/", 1)[1]: value
            for key, value in metrics.items()
            if key.startswith("val_metric/")
        }
        evaluator = self._evaluators.get("val")
        per_class = getattr(evaluator, "last_per_class", []) if evaluator else []
        footer = None
        if isinstance(self.criterion, DFINECriterionV2):
            # Mirrors the note on validation_step: the val loss skips the
            # train-time distributional terms, so the two totals in the
            # metrics file are different scopes by construction. The IoU
            # term name follows the criterion's box-loss swap (doc 15) so
            # the note never goes stale again.
            iou_name = (
                "SIoU"
                if getattr(self.criterion, "box_regression_iou", "giou") == "siou"
                else "GIoU"
            )
            footer = (
                f"note: val_loss covers VFL/bbox/{iou_name} only (FGL/DDF and "
                "the denoising/aux branches are train-time) — not comparable "
                "to train_loss"
            )
        return format_detection_validation_report(
            "Validation (EMA weights)" if self.ema_model is not None else "Validation",
            scalars,
            per_class,
            self._max_dets,
            footer=footer,
        )

    def compute_metrics(self, split: str) -> dict[str, float]:
        evaluator = self._evaluators.get(split)
        if evaluator is None:
            return {}
        metrics = evaluator.compute()
        open_vocab = getattr(self.model_args, "open_vocab", None)
        if open_vocab is not None and open_vocab.novel_eval_classes:
            # Leave-out protocol (design doc 27 §3.7): per-class-AP subset
            # aggregates over the novel classes and their base complement.
            # Read while ``last_per_class`` holds the just-computed rows.
            metrics.update(self._base_novel_ap(evaluator, open_vocab))
        evaluator.reset()
        if split == "val" and self._val_query_stats:
            # Density-guided diagnostics (design doc 13 §1.3), consumed
            # here so the engine's val logging / metrics.jsonl carry them.
            stats = self._val_query_stats
            metrics["dome_avg_queries"] = float(
                sum(n for n, _ in stats) / max(len(stats), 1)
            )
            metrics["dome_ample_rate"] = float(
                sum(n >= g for n, g in stats) / max(len(stats), 1)
            )
            self._val_query_stats.clear()
        if split == "val" and self._val_window_ratios:
            metrics["dome_window_ratio"] = float(
                sum(self._val_window_ratios) / max(len(self._val_window_ratios), 1)
            )
            self._val_window_ratios.clear()
        return {f"{split}_metric/{name}": value for name, value in metrics.items()}

    @staticmethod
    def _base_novel_ap(
        evaluator: COCOMAPEvaluator, open_vocab: Any
    ) -> dict[str, float]:
        """Base/novel per-class-AP aggregates (design doc 27 §3.7).

        ``last_per_class`` only carries classes with ground truth, and a
        row's ``ap`` is ``-1.0`` when the class has no valid precision —
        both are skipped. An empty subset aggregates to ``-1.0`` (the
        pycocotools "undefined" sentinel, NOT a zero score).
        """
        novel = set(open_vocab.novel_eval_classes)
        per_class = {row["name"]: row["ap"] for row in evaluator.last_per_class}

        def _mean_subset(names: list[str]) -> float:
            values = [
                per_class[name]
                for name in names
                if name in per_class and per_class[name] > -1.0
            ]
            return float(sum(values) / len(values)) if values else -1.0

        base_names = [
            name for name in open_vocab.class_names if name not in novel
        ]
        return {
            "base_ap": _mean_subset(base_names),
            "novel_ap": _mean_subset(sorted(novel)),
        }

    def reset_metrics(self, split: str) -> None:
        evaluator = self._evaluators.get(split)
        if evaluator is not None:
            evaluator.reset()
        if split == "val":
            self._val_query_stats.clear()
            self._val_window_ratios.clear()

    def export_metric_state(self, split: str) -> dict | None:
        evaluator = self._evaluators.get(split)
        evaluator_state = evaluator.export_state() if evaluator is not None else None
        if split == "val":
            # Fold the dome query stats into the gathered payload so the
            # rank-zero merge sees every rank's shard (DDP path).
            return {
                "evaluator": evaluator_state,
                "dome_query_stats": list(self._val_query_stats),
                "dome_window_ratios": list(self._val_window_ratios),
            }
        return evaluator_state

    def merge_metric_state(self, split: str, state: dict | None) -> None:
        if state is None:
            return
        evaluator_state = state
        if split == "val" and isinstance(state, dict) and "evaluator" in state:
            evaluator_state = state.get("evaluator")
            self._val_query_stats.extend(state.get("dome_query_stats", []))
            self._val_window_ratios.extend(state.get("dome_window_ratios", []))
        evaluator = self._evaluators.setdefault(
            split, COCOMAPEvaluator(self.categories, max_dets=self._max_dets)
        )
        if evaluator_state is not None:
            evaluator.merge_state(evaluator_state)

    # ── cross-model distillation (design doc 23) ────────────────────────
    def _build_distill_teacher(
        self, args: DETRDistillArgs, *, student_num_classes: int
    ) -> torch.nn.Module:
        if args.teacher_checkpoint is not None:
            teacher = _load_teacher_from_export(args.teacher_checkpoint)
        elif args.teacher_model == "self:ema":
            if self.ema_model is None:
                raise ValueError(
                    "distill teacher_model='self:ema' requires "
                    "use_ema_model=True (the teacher IS the run's EMA copy)."
                )
            teacher = self.ema_model.model
        else:
            raise NotImplementedError(
                f"distill teacher_model={args.teacher_model!r}: v1 supports "
                "'self:ema' and teacher_checkpoint (an AutoVision exported "
                ".pt — the recommended same-domain teacher, design doc 23 "
                "§2.1 arm a); public-weight teachers (e.g. deimv2 HF) need "
                "the class-space retile question settled first (arm b)."
            )
        teacher.requires_grad_(False)
        teacher.eval()
        # Class-count attribute for the class-space rule; every task model
        # in the framework carries self.num_classes.
        if not hasattr(teacher, "num_classes"):
            raise ValueError(
                "distill teacher lacks a num_classes attribute; the "
                "class-space rule (design doc 23 §4.1) cannot be applied."
            )
        return teacher

    def _distill_gate(self, step: int) -> bool:
        cfg = self._distill_args
        return step >= cfg.start_step and (
            cfg.every_n_steps <= 1 or step % cfg.every_n_steps == 0
        )

    def _ensure_teacher_device(self, x: torch.Tensor) -> None:
        # Unregistered modules don't follow the engine's train_model.to();
        # move lazily on first use (idempotent — a no-op afterwards).
        teacher = self._distill_teacher
        if teacher is not None:
            parameter = next(teacher.parameters(), None)
            if parameter is not None and parameter.device != x.device:
                teacher.to(x.device)

    def _build_teacher_targets(
        self,
        x: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        step: int,
    ) -> dict:
        """Teacher forward + its own Hungarian match → criterion join inputs.

        Runs under ``torch.inference_mode`` (zero autograd cost). The
        DETR-family teacher's decoder is flipped to train mode for the
        forward — the side distributions (``pred_corners_list``) are a
        train-mode emission, and the fork decoder carries neither BatchNorm
        nor dropout, so the pass is deterministic — with ``num_denoising``
        temporarily zeroed so the CDN group (which consumes RNG) stays out
        of the teacher pass. The teacher's backbone/encoder remain in eval:
        their BN statistics must not drift.

        The teacher runs its OWN Hungarian match against the same targets
        as the student (GT-anchored alignment, doc 23 D1); the resulting
        GT→teacher-query map is what the criterion joins against the
        student's union match.
        """
        cfg = self._distill_args
        if not self._distill_gate(step):
            return {"gated_out": True}
        self._ensure_teacher_device(x)
        teacher = self._distill_teacher
        x_t = x
        if cfg.teacher_image_size is not None and tuple(cfg.teacher_image_size) != tuple(
            x.shape[-2:]
        ):
            # Normalized cxcywh targets/predictions are resolution-
            # independent, so a smaller teacher canvas is valid (doc 23 D6).
            x_t = torch.nn.functional.interpolate(
                x,
                size=tuple(cfg.teacher_image_size),
                mode="bilinear",
                align_corners=False,
            )
        with torch.inference_mode():
            decoder = getattr(teacher, "decoder", None)
            saved_dn = None
            if decoder is not None:
                saved_dn = decoder.num_denoising
                decoder.num_denoising = 0
                decoder.train()
            try:
                out = teacher._forward_train(x_t)
            finally:
                if decoder is not None:
                    decoder.num_denoising = saved_dn
                    decoder.eval()
            t_logits = out["pred_logits"]
            t_boxes = out["pred_boxes"]
            raw_corners = out.get("pred_corners_list")
            t_corners = raw_corners[-1] if raw_corners else None
            t_indices = self._distill_matcher(
                {"pred_logits": t_logits, "pred_boxes": t_boxes}, targets
            )
        bin_space = 4 * (self.model_args.reg_max + 1)
        if t_corners is not None and t_corners.shape[-1] != bin_space:
            if not self._distill_ddf_warned:
                logger.warning(
                    "distill: teacher side distributions are %d-wide but the "
                    "student's W(n) space is %d-wide — dropping the DDF leg "
                    "(cls/box legs are bin-space free and unaffected; "
                    "design doc 23 G3).",
                    t_corners.shape[-1],
                    bin_space,
                )
                self._distill_ddf_warned = True
            t_corners = None
        return {
            "teacher_logits": t_logits,
            "teacher_boxes": t_boxes,
            "teacher_corners": t_corners,
            "gt_to_teacher_query": _build_gt_to_teacher_query(
                t_indices, targets, x.device
            ),
        }

    # ── optimization ────────────────────────────────────────────────────
    def get_optimizer(
        self,
        total_steps: int,
        global_batch_size: int,
        train_num_batches: int = 0,
        gradient_accumulation_steps: int = 1,
    ) -> tuple[torch.optim.Optimizer, Any]:
        return build_detr_optimizer(
            self.model,
            self.model_args,
            total_steps,
            global_batch_size,
            train_num_batches,
            gradient_accumulation_steps,
        )

    def clip_gradients(self, optimizer) -> float | None:
        max_norm = self.model_args.gradient_clip_val
        if max_norm <= 0:
            # Clipping disabled: report the total norm without mutating
            # grads (baseline contract — clip_grad_norm_ with a
            # nonpositive max_norm would ZERO every gradient).
            grads = [
                p.grad
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            if not grads:
                return 0.0
            # One grouped kernel instead of one per tensor (hundreds on a
            # detector): _foreach_norm + the outer norm produce the same
            # total the per-tensor loop did.
            return float(
                torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
            )
        params = [p for group in optimizer.param_groups for p in group["params"]]
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm))

    def on_train_batch_end(self) -> None:
        if self.ema_model is not None:
            self.ema_model.update(self.model)

    # ── checkpoint hooks ────────────────────────────────────────────────
    @staticmethod
    def _is_ssl_head_key(key: str) -> bool:
        """Match SSL-head keys in every state-dict layout: bare
        (``ssl_head.*`` inside an EMA copy), train-wrapper
        (``model.ssl_head.*``) or export (``ema_model.ssl_head.*``)."""
        return key.startswith("ssl_head.") or ".ssl_head." in key

    @staticmethod
    def _is_ov_text_embeds_key(key: str) -> bool:
        """Match the open-vocab text buffer in every state-dict layout
        (bare / ``model.`` / ``ema_model.`` prefixed)."""
        return key == "ov_text_embeds" or key.endswith(".ov_text_embeds")

    def _fill_missing_ov_keys(self, incoming: dict[str, Any]) -> dict[str, Any]:
        """Fill open-vocab text-buffer keys absent from *incoming*.

        Adding the open-vocab branch on top of a baseline (pre-doc-27)
        checkpoint is a supported workflow: the buffer keeps the values
        the constructor just encoded from the run's own vocabulary
        instead of failing the strict load. Only text-buffer keys are
        filled — everything else stays loud (contrastive-head weights
        genuinely cannot come from a Linear-head checkpoint).
        """
        own = self.state_dict()
        missing = {
            key: value
            for key, value in own.items()
            if self._is_ov_text_embeds_key(key) and key not in incoming
        }
        if missing:
            incoming = {**incoming, **missing}
        return incoming

    def _fill_missing_ssl_keys(self, incoming: dict[str, Any]) -> dict[str, Any]:
        """Fill SSL-head keys absent from *incoming* with their current values.

        Adding the SSL branch on top of a detection-pretrained checkpoint is
        a supported workflow: the head keeps its fresh init instead of
        failing the strict load. Only keys that belong to the SSL head are
        filled — any other missing key, and any unexpected key, stays loud.
        """
        own = self.state_dict()
        missing_ssl = {
            key: value
            for key, value in own.items()
            if self._is_ssl_head_key(key) and key not in incoming
        }
        if missing_ssl:
            incoming = {**incoming, **missing_ssl}
        return incoming

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs: Any):
        incoming = dict(state_dict)
        own_keys = set(self.state_dict().keys())
        # Stage-2 workflow (design doc 14 §4.10): a stage-1 SSL-only
        # checkpoint carries the pretraining-only projection head — drop it
        # when the new config has no head, instead of failing the strict
        # load. (The engine's ``checkpoint=`` init path lands here; resume
        # with a mismatched ssl config stays loud for every non-SSL key.)
        incoming = {
            key: value
            for key, value in incoming.items()
            if not (self._is_ssl_head_key(key) and key not in own_keys)
        }
        return super().load_state_dict(
            self._fill_missing_ov_keys(self._fill_missing_ssl_keys(incoming)),
            strict=strict,
            **kwargs,
        )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_model is not None:
            checkpoint["ema"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        ema_state = checkpoint.get("ema")
        if ema_state is not None and self.ema_model is not None:
            ema_weights = dict(ema_state.get("model", {}))
            own_ema = self.ema_model.model.state_dict()
            missing_ssl = {
                key: value
                for key, value in own_ema.items()
                if self._is_ssl_head_key(key) and key not in ema_weights
            }
            if missing_ssl:
                ema_state = {**ema_state, "model": {**ema_weights, **missing_ssl}}
            ema_weights = dict(ema_state["model"])
            missing_ov = {
                key: value
                for key, value in own_ema.items()
                if self._is_ov_text_embeds_key(key) and key not in ema_weights
            }
            if missing_ov:
                ema_state = {**ema_state, "model": {**ema_weights, **missing_ov}}
            self.ema_model.load_state_dict(ema_state)

    # ── export state ────────────────────────────────────────────────────
    def get_export_state_dict(self) -> dict[str, Any]:
        if self.ema_model is not None:
            return {
                f"ema_model.{key}": value
                for key, value in self.ema_model.model.state_dict().items()
            }
        return self.state_dict()
