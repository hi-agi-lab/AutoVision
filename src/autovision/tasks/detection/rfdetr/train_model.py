# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RF-DETR object detection training wrapper.

Consumes the HF ``RfDetrForObjectDetection`` criterion (the official
IA-BCE + L1 + GIoU port) instead of an AutoVision-side criterion, and
replicates the official rf-detr optimizer recipe: AdamW with
encoder-layer-wise LR decay (``lr_encoder * 0.8^layer``), a decoder
component decay (``lr * 0.7``), weight-decay exemptions for
bias/norm/gamma/pos-embed parameters, and EMA 0.993 with a tau-style
warmup.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Any

import torch

from autovision.augment.detection.box_aware_eraser import (
    erase_background,
)
from autovision.metrics.map_evaluator import COCOMAPEvaluator
from autovision.tasks.detection.rfdetr.config import (
    RFDETRObjectDetectionTrainArgs,
)
from autovision.tasks.detection.rfdetr.task_model import (
    RFDETRObjectDetection,
)
from autovision.tasks.detection.shared.ssl_contrastive import (
    SSLParams,
    ssl_contrastive_loss,
)
from autovision.tasks.shared.ema import ModelEMA
from autovision.tasks.train_model import TaskStepResult, TrainModel

logger = logging.getLogger("autovision")

#: Row cap for the SIGReg pool: the Epps–Pulley intermediates scale as
#: (num_slices, N, num_knots), and an uncapped two_crop pool (8 x 300 rows)
#: allocates ~4GB per temporary on top of the step's autograd graph. The
#: statistic is a sketch — 1024 rows keep it bounded with no practical
#: signal loss.
_SIGREG_POOL_CAP = 1024

#: Second-eval split key: box-aware erased views (doc 16 P1.6). Metrics
#: surface as ``val_erased_metric/*``; the dense ``val_metric/*`` channel
#: and the monitored val loss are untouched.
_ERASED_EVAL_SPLIT = "val_erased"


def _eval_erase_seed(image_path: str) -> int:
    """Stable per-image seed: the same val image draws the same cells at
    every validation and checkpoint, keeping ``val_erased_metric/*``
    comparable across runs (a redrawn mask would add sampling noise)."""
    digest = hashlib.blake2b(image_path.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFF_FFFF


__all__ = ["RFDETRObjectDetectionTrain"]

# rf-detr param_groups.py: weight-decay exemptions by name fragment.
_NO_WD_FRAGMENTS = ("bias", "norm", "gamma", "pos_embed", "rel_pos")
_VIT_LAYER_PATTERN = re.compile(r"encoder\.layer\.(\d+)\.")


def _rfdetr_param_buckets(
    task_model: RFDETRObjectDetection, args: RFDETRObjectDetectionTrainArgs
) -> dict[tuple[str, bool], list[torch.nn.Parameter]]:
    """Split HF parameters into (lr-bucket, weight-decay) groups.

    Buckets: ``encoder_<layer>`` for the DINOv2 ViT (embeddings are layer
    0, blocks are 1-based — matching the official backbone.py scheme),
    ``decoder`` for the detection transformer, and ``other`` for the
    projector, queries, and heads.
    """
    buckets: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in task_model._model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_wd = any(fragment in name for fragment in _NO_WD_FRAGMENTS)
        match = _VIT_LAYER_PATTERN.search(name)
        if ".backbone.backbone." in name:
            layer = int(match.group(1)) + 1 if match else 0
            bucket = f"encoder_{layer}"
        elif ".decoder." in name:
            bucket = "decoder"
        else:
            bucket = "other"
        buckets.setdefault((bucket, no_wd), []).append(parameter)
    return buckets


def _bucket_lr(
    bucket: str, args: RFDETRObjectDetectionTrainArgs, lr: float, lr_encoder: float
) -> float:
    if bucket.startswith("encoder_"):
        layer = int(bucket.split("_")[1])
        return lr_encoder * args.lr_vit_layer_decay**layer
    if bucket == "decoder":
        return lr * args.lr_component_decay
    return lr


class RFDETRObjectDetectionTrain(TrainModel):
    task = "object_detection"
    watch_metric = "val_metric/map"

    def __init__(self, model_args: RFDETRObjectDetectionTrainArgs) -> None:
        model_args.resolve_image_size()
        task_model = RFDETRObjectDetection(
            num_classes=model_args.num_classes,
            model=model_args.model,
            image_size=model_args.image_size,
            num_queries=model_args.num_queries,
            num_select=model_args.num_select,
            load_weights=model_args.load_weights,
            box_regression_iou=model_args.box_regression_iou,
            siou_theta=model_args.siou_theta,
            ssl=model_args.ssl,
        )
        super().__init__(task_model=task_model)
        self.model_args = model_args
        self.ema_model: ModelEMA | None = (
            ModelEMA(
                task_model,
                decay=model_args.ema_momentum,
                warmup_steps=model_args.ema_warmup_steps,
            )
            if model_args.use_ema_model
            else None
        )
        self.categories: dict[int, str] = {}
        self._evaluators: dict[str, COCOMAPEvaluator] = {}
        # Per-object contrastive SSL (design doc 14, rfdetr port — doc 18):
        # the pool is the final-layer Hungarian match recomputed with the
        # SAME matcher and costs the HF box loss consumes (matching is
        # deterministic, so the SSL instances ARE the box-loss instances).
        # mode="off" is the batch-matched baseline: the doubled two_crop
        # pipeline runs, but neither head nor loss is wired (doc 14 §3.3).
        self.ssl_params: SSLParams | None = (
            model_args.ssl.to_ssl_params()
            if model_args.ssl is not None and model_args.ssl.mode != "off"
            else None
        )
        self._ssl_matcher = None
        if self.ssl_params is not None:
            from transformers.loss.loss_lw_detr import LwDetrHungarianMatcher

            hf_config = task_model._model.config
            self._ssl_matcher = LwDetrHungarianMatcher(
                class_cost=hf_config.class_cost,
                bbox_cost=hf_config.bbox_cost,
                giou_cost=hf_config.giou_cost,
            )

    # ── steps ───────────────────────────────────────────────────────────
    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        targets = [
            {"boxes": boxes.to(x.device), "labels": labels.to(x.device)}
            for boxes, labels in zip(batch["bboxes"], batch["classes"])
        ]
        out = self.model._forward_train(x, targets=targets)
        loss = out["loss"]
        if loss is None:
            raise RuntimeError(
                "RfDetrForObjectDetection returned no loss; the HF forward "
                "requires labels to compute the criterion."
            )
        log_dict: dict[str, Any] = {}
        if self.ssl_params is not None and self.model.training:
            loss, ssl_log = self._ssl_loss(out, targets, step)
            log_dict.update(ssl_log)
        # The engine aggregates the log window into a top-level
        # "train_loss" from the plain "loss" key (criterion convention).
        # Values stay as 0-dim tensors: the engine's log window sums them
        # on-device and only syncs at log points.
        log_dict["loss"] = loss.detach()
        for key, value in out["loss_dict"].items():
            log_dict[f"train_loss/{key}"] = (
                value.detach() if torch.is_tensor(value) else value
            )
        return TaskStepResult(loss=loss, log_dict=log_dict)

    def _ssl_loss(
        self,
        out: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
        step: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Per-object contrastive objective on the final-layer match.

        The instances are the SAME Hungarian match the HF box loss
        consumes: the matcher is re-run with the identical costs/config
        (deterministic), so no supervision drift is possible. The stride
        gate mirrors the D-FINE criterion (doc 14 §4.4).
        """
        params = self.ssl_params
        assert params is not None and self._ssl_matcher is not None
        stride = params.every_n_steps
        if stride > 1 and step % stride != 0:
            return out["loss"], {}
        z1, z2 = out["ssl_z1"], out["ssl_z2"]
        matcher_outputs = {
            "logits": out["pred_logits"],
            "pred_boxes": out["pred_boxes"],
        }
        matcher_targets = [
            {"class_labels": t["labels"], "boxes": t["boxes"]} for t in targets
        ]
        indices = self._ssl_matcher(
            matcher_outputs, matcher_targets, self.model._model.config.group_detr
        )
        # Group-DETR matching assigns EVERY query group to the GT set (the
        # box loss supervises all groups); the contrastive pool follows the
        # reference and uses the GROUP-0 match only — one anchor per GT
        # keeps the pool at instance scale instead of 13x (rf-detr-ssl-v2
        # ssl-classification-head, "group-0 matched pool").
        group_queries = out["pred_logits"].shape[1] // int(
            self.model._model.config.group_detr
        )
        indices = [
            (src[src < group_queries], tgt[src < group_queries]) for src, tgt in indices
        ]
        ssl_stats: dict[str, float] = {}
        loss_ssl = ssl_contrastive_loss(
            z1,
            z2,
            {"pred_logits": out["pred_logits"]},
            targets,
            indices,
            params,
            stats_sink=ssl_stats,
        )
        total = out["loss"] + loss_ssl * params.loss_coef
        log: dict[str, Any] = {"train_loss/loss_ssl": loss_ssl.detach()}
        # SIGReg distribution regularizer (doc 16 R1-A1): the RAW projector
        # outputs of the FULL query pool (batch-flattened; rfdetr has a
        # fixed query count, no padding to filter). Same formula as the
        # D-FINE criterion — per-view Epps–Pulley statistic, averaged.
        # Weighted leg when sigreg_coef>0; at coef=0 the unweighted
        # statistic still logs as `loss_sigreg_stat` — the always-on
        # read-only calibration readout for choosing λ_sr (doc 18 §2).
        s1_tap = out.get("ssl_s1")
        s2_tap = out.get("ssl_s2")
        if s1_tap is not None and s2_tap is not None:
            from autovision.tasks.detection.shared.ssl_contrastive import (
                epps_pulley_sigreg,
            )

            s1_pool = s1_tap.reshape(-1, s1_tap.shape[-1])
            s2_pool = s2_tap.reshape(-1, s2_tap.shape[-1])
            # The Epps–Pulley intermediates are (num_slices, N, knots) —
            # at two_crop's 2400-row pool that is a ~4GB single allocation
            # on top of the step's autograd graph (measured OOM at micro 8).
            # The statistic is a sketch: a fixed-size random row subsample
            # keeps it bounded with no practical signal loss.
            if s1_pool.shape[0] > _SIGREG_POOL_CAP:
                pick = torch.randperm(s1_pool.shape[0], device=s1_pool.device)[
                    :_SIGREG_POOL_CAP
                ]
                s1_pool = s1_pool[pick]
                s2_pool = s2_pool[pick]
            sigreg_stat = 0.5 * (
                epps_pulley_sigreg(
                    s1_pool,
                    num_slices=params.sigreg_slices,
                    num_knots=params.sigreg_knots,
                )
                + epps_pulley_sigreg(
                    s2_pool,
                    num_slices=params.sigreg_slices,
                    num_knots=params.sigreg_knots,
                )
            )
            if params.sigreg_coef > 0:
                total = total + sigreg_stat * params.sigreg_coef
                log["train_loss/loss_sigreg"] = sigreg_stat.detach()
            else:
                log["train_loss/loss_sigreg_stat"] = sigreg_stat.detach()
        elif params.sigreg_coef > 0:
            raise ValueError(
                "ssl sigreg_coef>0 is configured but the model outputs "
                "carry no ssl_s1/ssl_s2 raw taps; build the task model "
                "with the same ssl config (doc 16 R1-A1)."
            )
        pool = float(ssl_stats.get("pool_size", 0.0))
        paired = float(ssl_stats.get("paired_size", 0.0))
        denominator = max(pool / 2.0 if params.view == "two_crop" else pool, 1.0)
        log["train_loss/ssl_pool_size"] = pool
        log["train_loss/ssl_paired_ratio"] = paired / denominator
        return total, log

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        targets = [
            {"boxes": boxes.to(x.device), "labels": labels.to(x.device)}
            for boxes, labels in zip(batch["bboxes"], batch["classes"])
        ]
        model = self.ema_model.model if self.ema_model is not None else self.model
        # Eval mode keeps a single query group, so the same forward serves
        # both the decode and the monitored val loss.
        outputs = model._forward_train(x, targets=targets)
        metadata = [{"orig_w": w, "orig_h": h} for w, h in batch["original_size"]]
        decoded = model.postprocess(
            (outputs["pred_logits"], outputs["pred_boxes"]),
            metadata,
            threshold=0.0,  # COCO protocol: cap by top-k / maxDets, not scores
        )
        evaluator = self._evaluators.setdefault(
            "val",
            COCOMAPEvaluator(self.categories, max_dets=self.model_args.eval_max_dets),
        )
        evaluator.update(decoded)
        evaluator.update_targets(targets, original_sizes=batch["original_size"])
        if self.model_args.eval_erase is not None:
            self._erased_eval_step(x, targets, batch, model, metadata)
        loss = outputs["loss"]
        log_dict: dict[str, float] = {}
        if loss is not None:
            log_dict = {
                f"val_loss/{key}": float(value)
                for key, value in outputs["loss_dict"].items()
            }
            log_dict["val_loss"] = float(loss)
        return TaskStepResult(
            loss=loss if loss is not None else x.new_zeros(()), log_dict=log_dict
        )

    def _erased_eval_step(
        self,
        x: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        batch: dict,
        model: Any,
        metadata: list[dict[str, int]],
    ) -> None:
        """Second eval channel on box-aware erased views (doc 16 P1.6).

        Validation carries GT boxes — the protection source deployment
        inference lacks — so the train-time eraser applies here unchanged:
        the batch is already ImageNet-normalized (exactly the eraser's
        workspace, fill 0.0 = dataset-mean gray), every image is erased
        (no train-style coin flip — this is a fixed test condition), and
        decoded predictions are scored against the UNCHANGED dense GT. The
        forward runs without labels: the monitored val loss stays
        dense-only. Per-image masks are path-seeded (see
        :func:`_eval_erase_seed`); no-annotation images pass through
        untouched (the eraser's conservative skip).
        """
        erase_args = self.model_args.eval_erase
        image_paths = batch.get("image_path") or []
        erased_images = []
        for index in range(x.shape[0]):
            path = image_paths[index] if index < len(image_paths) else str(index)
            generator = torch.Generator().manual_seed(_eval_erase_seed(path))
            erased_images.append(
                erase_background(
                    x[index],
                    targets[index]["boxes"],
                    ratio=erase_args.ratio,
                    padding=erase_args.padding,
                    cell=erase_args.cell,
                    interior=erase_args.interior,
                    min_keep=erase_args.min_keep,
                    generator=generator,
                )
            )
        outputs = model._forward_train(torch.stack(erased_images))
        decoded = model.postprocess(
            (outputs["pred_logits"], outputs["pred_boxes"]),
            metadata,
            threshold=0.0,
        )
        evaluator = self._evaluators.setdefault(
            _ERASED_EVAL_SPLIT,
            COCOMAPEvaluator(self.categories, max_dets=self.model_args.eval_max_dets),
        )
        evaluator.update(decoded)
        evaluator.update_targets(targets, original_sizes=batch["original_size"])

    def set_categories(self, categories: dict[int, str]) -> None:
        self.categories = dict(categories)

    def compute_metrics(self, split: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        evaluator = self._evaluators.get(split)
        if evaluator is not None:
            metrics.update(
                {
                    f"{split}_metric/{name}": value
                    for name, value in evaluator.compute().items()
                }
            )
            evaluator.reset()
        if split == "val" and self.model_args.eval_erase is not None:
            erased = self._evaluators.get(_ERASED_EVAL_SPLIT)
            if erased is not None:
                metrics.update(
                    {
                        f"{_ERASED_EVAL_SPLIT}_metric/{name}": value
                        for name, value in erased.compute().items()
                    }
                )
                erased.reset()
        return metrics

    def reset_metrics(self, split: str) -> None:
        keys = [split]
        if split == "val" and self.model_args.eval_erase is not None:
            keys.append(_ERASED_EVAL_SPLIT)
        for key in keys:
            evaluator = self._evaluators.get(key)
            if evaluator is not None:
                evaluator.reset()

    def export_metric_state(self, split: str) -> dict | None:
        evaluator = self._evaluators.get(split)
        dense = evaluator.export_state() if evaluator is not None else None
        if split == "val" and self.model_args.eval_erase is not None:
            erased = self._evaluators.get(_ERASED_EVAL_SPLIT)
            return {
                "dense": dense,
                "erased": erased.export_state() if erased is not None else None,
            }
        return dense

    def merge_metric_state(self, split: str, state: dict | None) -> None:
        if state is None:
            return
        dense_state, erased_state = state, None
        if (
            split == "val"
            and self.model_args.eval_erase is not None
            and "dense" in state
        ):
            dense_state, erased_state = state["dense"], state.get("erased")
        if dense_state is not None:
            evaluator = self._evaluators.setdefault(
                split,
                COCOMAPEvaluator(
                    self.categories, max_dets=self.model_args.eval_max_dets
                ),
            )
            evaluator.merge_state(dense_state)
        if erased_state is not None:
            erased = self._evaluators.setdefault(
                _ERASED_EVAL_SPLIT,
                COCOMAPEvaluator(
                    self.categories, max_dets=self.model_args.eval_max_dets
                ),
            )
            erased.merge_state(erased_state)

    # ── optimization ────────────────────────────────────────────────────
    def get_optimizer(
        self,
        total_steps: int,
        global_batch_size: int,
        train_num_batches: int = 0,
        gradient_accumulation_steps: int = 1,
    ) -> tuple[torch.optim.Optimizer, Any]:
        args = self.model_args
        args.resolve_auto_fields(
            total_steps=total_steps,
            train_num_batches=train_num_batches or total_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        scale = math.sqrt(global_batch_size / args.default_batch_size)
        lr = args.lr * scale
        lr_encoder = args.lr_encoder * scale

        param_groups = []
        for (bucket, no_wd), params in sorted(
            _rfdetr_param_buckets(self.model, args).items()
        ):
            group: dict[str, Any] = {
                "name": f"{bucket}_no_wd" if no_wd else bucket,
                "params": params,
                "lr": _bucket_lr(bucket, args, lr, lr_encoder),
            }
            if no_wd:
                group["weight_decay"] = 0.0
            param_groups.append(group)

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=args.optimizer_betas,
            weight_decay=args.weight_decay,
        )

        warmup = int(args.lr_warmup_steps)
        milestones = [fraction * total_steps for fraction in args.lr_milestones]

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            if args.lr_scheduler == "step":
                factor = 1.0
                for milestone in milestones:
                    if step >= milestone:
                        factor *= args.lr_gamma
                return factor
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

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

    # ── export state ────────────────────────────────────────────────────
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_model is not None:
            checkpoint["ema"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        ema_state = checkpoint.get("ema")
        if ema_state is not None and self.ema_model is not None:
            self.ema_model.load_state_dict(ema_state)

    def get_export_state_dict(self) -> dict[str, Any]:
        if self.ema_model is not None:
            return {
                f"ema_model.{key}": value
                for key, value in self.ema_model.model.state_dict().items()
            }
        return self.state_dict()
