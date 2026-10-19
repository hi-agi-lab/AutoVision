# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR pose training wrapper (roadmap doc 31 §11 S5b).

Criterion/matcher/CDN are the faithfully re-ported official pieces
(``shared/gtr/pose_criterion.py`` + ``pose_dn.py``); the decoder runs the
pose CDN internally in training mode (upstream passes the image for the
per-keypoint noise scaling). Validation computes the official COCO
keypoint AP (OKS, pycocotools ``keypoints``) against the raw
``person_keypoints`` annotation file and selects on it
(``watch_metric = "val_metric/kp_map"``); the mean matched OKS remains a
diagnostic (``val_metric/oks``).
"""

from __future__ import annotations

from pathlib import Path

import torch

from autovision.metrics.pose_estimation import COCOKeypointAPEvaluator
from autovision.optim.optimizer_helpers import (
    build_adamw,
    get_weight_decay_parameters,
)
from autovision.optim.schedulers import CosineWarmupScheduler
from autovision.tasks.detection.shared.gtr.pose_criterion import (
    GTRPoseCriterion,
    GTRPoseHungarianMatcher,
    _sigmas_for,
    oks_overlaps,
)
from autovision.tasks.pose_estimation.gtr.config import (
    GTRPoseEstimationTrainArgs,
)
from autovision.tasks.pose_estimation.gtr.task_model import GTRPoseEstimation
from autovision.tasks.train_model import TaskStepResult, TrainModel
from autovision.types import PathLike

__all__ = ["GTRPoseEstimationTrain"]


class GTRPoseEstimationTrain(TrainModel):
    task = "pose_estimation"
    watch_metric = "val_metric/kp_map"

    def __init__(self, model_args: GTRPoseEstimationTrainArgs) -> None:
        model_args.resolve_auto_fields()
        task_model = GTRPoseEstimation(
            num_classes=model_args.num_classes,
            variant=model_args.variant,
            image_size=model_args.image_size,
            num_queries=model_args.num_queries,
            num_decoder_layers=model_args.num_decoder_layers,
            dim_feedforward=model_args.dim_feedforward,
            num_body_points=model_args.num_body_points,
            load_checkpoint=model_args.load_checkpoint,
            checkpoint=model_args.checkpoint,
        )
        super().__init__(task_model=task_model)
        self.model_args = model_args
        self.matcher = GTRPoseHungarianMatcher(
            cost_class=model_args.cost_class,
            cost_keypoints=model_args.cost_keypoints,
            cost_oks=model_args.cost_oks,
            num_body_points=model_args.num_body_points,
        )
        self.criterion = GTRPoseCriterion(
            num_classes=model_args.num_classes,
            matcher=self.matcher,
            weight_dict=model_args.loss_weight_dict,
            losses=model_args.losses,
            num_body_points=model_args.num_body_points,
        )
        self.categories: dict[int, str] = {}
        self._val_annotations: Path | None = None
        self._evaluators: dict[str, COCOKeypointAPEvaluator] = {}
        self._oks_sum = 0.0
        self._oks_count = 0

    def set_categories(self, categories: dict[int, str]) -> None:
        self.categories = dict(categories)

    def set_val_annotations(self, annotations: PathLike) -> None:
        """Raw ``person_keypoints`` JSON powering the official AP evaluator."""
        self._val_annotations = Path(annotations)

    # ── targets ─────────────────────────────────────────────────────────
    def _build_targets(
        self, batch: dict, x: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        """Targets from the batch contract (official layout):

        ``keypoints`` (N, 3K) normalized with the xy block first and the
        visibility block last (``cat([xy (2K), v (K)])`` — the official
        ``PoseNormalize`` layout, not interleaved triplets), ``bboxes``
        (N, 4) normalized cxcywh person boxes (their centers anchor the
        CDN noise), ``area`` (N,) frame-normalized person areas, and
        ``classes`` (N,).
        """
        targets = []
        for keypoints, boxes, labels in zip(
            batch["keypoints"], batch["bboxes"], batch["classes"]
        ):
            entry = {
                "keypoints": keypoints.to(x.device).float(),
                "boxes": boxes.to(x.device).float(),
                "labels": labels.to(x.device).long(),
            }
            if "area" in batch:
                entry["area"] = batch["area"][len(targets)].to(x.device).float()
            else:
                entry["area"] = boxes.to(x.device).float()[:, 2:4].prod(-1)
            targets.append(entry)
        return targets

    # ── steps ───────────────────────────────────────────────────────────
    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        targets = self._build_targets(batch, x)
        out = self.model._forward_train(x, targets=targets)
        loss_dict = self.criterion(out, targets)
        loss = sum(loss_dict.values())
        log_dict = {key: float(value.detach()) for key, value in loss_dict.items()}
        return TaskStepResult(loss=loss, log_dict=log_dict)

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        model = self.model
        out = model._forward_train(x)

        # Official-protocol AP: postprocess to the original frame (no score
        # gate — the official postprocessor keeps the plain top-k) and feed
        # the pycocotools keypoint evaluator.
        if self._val_annotations is not None:
            decoded = model.postprocess(
                (out["pred_logits"], out["pred_keypoints"]),
                [
                    {"orig_w": width, "orig_h": height}
                    for width, height in batch["original_size"]
                ],
            )
            evaluator = self._evaluators.setdefault(
                "val",
                COCOKeypointAPEvaluator(
                    self._val_annotations, categories=self.categories
                ),
            )
            image_ids = batch.get("index")
            evaluator.update(
                decoded,
                [int(image_id) for image_id in image_ids] if image_ids is not None else None,
            )

        # Mean matched OKS (diagnostic; matcher-based, training-domain units).
        targets = self._build_targets(batch, x)
        indices = self.matcher(
            {"pred_logits": out["pred_logits"], "pred_keypoints": out["pred_keypoints"]},
            targets,
        )
        batch_index = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        ) if indices else torch.zeros(0, dtype=torch.int64)
        src_keypoints = out["pred_keypoints"].flatten(0, 1)
        if len(batch_index):
            src = src_keypoints.reshape(-1, out["pred_keypoints"].shape[-1])[
                batch_index * out["pred_keypoints"].shape[1]
                + torch.cat([src for (src, _) in indices])
            ]
        else:
            # A validation batch can be entirely person-free (val keeps
            # every image); the matched tensors are then empty.
            src = src_keypoints.new_zeros(0, out["pred_keypoints"].shape[-1])
        tgt = torch.cat([t["keypoints"][j] for t, (_, j) in zip(targets, indices)], dim=0) \
            if len(batch_index) else torch.zeros(0, 3 * self.model_args.num_body_points)
        areas = torch.cat([t["area"][j] for t, (_, j) in zip(targets, indices)], dim=0) \
            if len(batch_index) else torch.zeros(0)
        if len(batch_index):
            k = self.model_args.num_body_points
            oks = oks_overlaps(
                src[:, : 2 * k],
                tgt[:, : 2 * k],
                tgt[:, 2 * k:],
                areas,
                _sigmas_for(k),
            )
            self._oks_sum += float(oks.sum())
            self._oks_count += int(oks.numel())
        zero = out["pred_logits"].sum() * 0.0
        return TaskStepResult(loss=zero, log_dict={"val_loss": 0.0})

    def compute_metrics(self, split: str) -> dict[str, float]:
        if split != "val":
            return {}
        metrics: dict[str, float] = {}
        evaluator = self._evaluators.get(split)
        if evaluator is not None:
            metrics.update(evaluator.compute())
            evaluator.reset()
        mean_oks = self._oks_sum / max(self._oks_count, 1)
        self._oks_sum = 0.0
        self._oks_count = 0
        metrics["oks"] = mean_oks
        return {f"{split}_metric/{name}": value for name, value in metrics.items()}

    def reset_metrics(self, split: str) -> None:
        evaluator = self._evaluators.get(split)
        if evaluator is not None:
            evaluator.reset()
        self._oks_sum = 0.0
        self._oks_count = 0

    def export_metric_state(self, split: str) -> dict | None:
        state: dict = {"oks_sum": self._oks_sum, "oks_count": self._oks_count}
        evaluator = self._evaluators.get(split)
        if evaluator is not None:
            state["evaluator"] = evaluator.export_state()
        return state

    def merge_metric_state(self, split: str, state: dict | None) -> None:
        if state is None:
            return
        self._oks_sum += float(state.get("oks_sum", 0.0))
        self._oks_count += int(state.get("oks_count", 0))
        evaluator_state = state.get("evaluator")
        if evaluator_state is not None:
            evaluator = self._evaluators.setdefault(
                split,
                COCOKeypointAPEvaluator(
                    self._val_annotations, categories=self.categories
                ),
            )
            evaluator.merge_state(evaluator_state)

    # ── optimization ────────────────────────────────────────────────────
    def get_optimizer(
        self,
        total_steps: int,
        global_batch_size: int,
        train_num_batches: int = 0,
        gradient_accumulation_steps: int = 1,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        args = self.model_args
        args.resolve_auto_fields(total_steps=total_steps)
        param_groups: list[dict] = []
        for name, modules, lr in (
            ("backbone", [self.model.backbone], args.lr * args.backbone_lr_factor),
            ("detector", [self.model.encoder, self.model.decoder], args.lr),
        ):
            with_decay, no_decay = get_weight_decay_parameters(modules)
            if with_decay:
                param_groups.append({"name": name, "params": with_decay, "lr": lr})
            if no_decay:
                param_groups.append(
                    {
                        "name": f"{name}_no_wd",
                        "params": no_decay,
                        "lr": lr,
                        "weight_decay": 0.0,
                    }
                )
        optimizer = build_adamw(
            param_groups,
            lr=args.lr,
            betas=args.optimizer_betas,
            weight_decay=args.weight_decay,
            fused=args.fused_optimizer,
        )
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_epochs=args.lr_warmup_steps, max_epochs=total_steps
        )
        return optimizer, scheduler

    def get_export_state_dict(self) -> dict:
        return self.model.state_dict()
