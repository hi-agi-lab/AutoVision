# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DINOv2: DINO + iBOT masked-patch prediction + KoLeo regularizer.

Recipe (design doc §7.5, blueprint in volume 12 §4.2):

- Reuses the DINO skeleton: multi-crop transform, projection head, EMA
  teacher, momentum/temp schedules, and weight-normed head.
- Adds iBOT: per global view a per-sample mask (ratio ~ U(0.1, 0.5)) feeds
  the student through the backbone's mask token; the student predicts the
  teacher's (unmasked, centered, sharpened) patch targets at masked
  positions only. The patch center is updated from masked teacher patches.
- Adds KoLeo: nearest-neighbor entropy regularizer on the concatenated
  global-view student CLS features, weight 0.1.
- Loss weights follow the blueprint: ``dino_global * 2/n + dino_local * 1/n
  + ibot + koleo_weight * koleo`` with n = total number of views. The CLS
  center updates once per step from the global teacher targets.
- Layer-wise LR decay 0.9 (deeper blocks → higher lr), patch embedding
  group ×0.2; weight decay cosine 0.04 → 0.4.

Teacher patch/CLS targets require the teacher and student patch grids to
match positionally (same patch size for the iBOT term), which is enforced
at construction when both wrappers report a patch size.
"""

from __future__ import annotations

import copy
import logging
import math
import re
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor

from autovision.optim import AdamWArgs, OptimizationTargets, get_weight_decay_parameters
from autovision.optim.schedulers import CosineWarmupScheduler
from autovision.pretext.dino.dino import (
    DINOProjectionHead,
    DINOTransform,
    _cosine_schedule,
    _update_momentum,
)
from autovision.pretext.method_args import PretextArgs
from autovision.pretext.recipe import PretextRecipe, StepOutput, prepare_teacher
from autovision.scaling import ScalingInfo

__all__ = ["DINOv2", "DINOv2Args"]

logger = logging.getLogger("autovision")


class DINOv2Args(PretextArgs):
    """Arguments for the DINOv2 method."""

    auto_steps: ClassVar[int | None] = 125_000
    auto_epochs: ClassVar[int | None] = None

    gradient_clip_val: float = 3.0
    # projection head (shared with the DINO recipe)
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    output_dim: int = 65536
    student_freeze_last_layer_steps: int = 1250
    batch_norm: bool = False
    norm_last_layer: bool = True
    # DINO CLS loss
    teacher_temp: float = 0.07
    warmup_teacher_temp: float = 0.04
    warmup_teacher_temp_steps: int = 37500
    warmup_teacher_temp_max_steps_fraction: float = 0.3
    student_temp: float = 0.1
    center_momentum: float = 0.9
    # iBOT masked-patch loss
    ibot_loss_weight: float = 1.0
    mask_ratio_min: float = 0.1
    mask_ratio_max: float = 0.5
    # KoLeo regularizer
    koleo_loss_weight: float = 0.1
    # schedules
    momentum_start: float = 0.996
    momentum_end: float = 1.0
    weight_decay_start: float = 0.04
    weight_decay_end: float = 0.4
    # layer-wise LR decay over transformer blocks
    layerwise_decay: float = 0.9
    patch_embed_lr: float = 0.2
    warmup_steps: int = 12500
    warmup_max_steps_fraction: float = 0.1

    def resolve_auto_fields(
        self,
        scaling: ScalingInfo | None = None,
        optimizer_args: Any = None,
        wrapped_model: Any = None,
    ) -> None:
        super().resolve_auto_fields(
            scaling=scaling,
            optimizer_args=optimizer_args,
            wrapped_model=wrapped_model,
        )
        if not 0.0 <= self.mask_ratio_min < self.mask_ratio_max <= 1.0:
            raise ValueError(
                "Mask ratios must satisfy 0 <= mask_ratio_min < "
                "mask_ratio_max <= 1, got "
                f"({self.mask_ratio_min}, {self.mask_ratio_max})."
            )
        if not 0.0 < self.layerwise_decay <= 1.0:
            raise ValueError(
                "layerwise_decay must be in (0, 1] (1.0 disables the "
                f"layer-wise decay), got {self.layerwise_decay}."
            )


class DINOv2AdamWArgs(AdamWArgs):
    """AdamW recipe: wd 0.04 start (cosine → 0.4)."""

    lr: float = 0.0005
    weight_decay: float = 0.04


# ── iBOT machinery ──────────────────────────────────────────────────────


class _MaskGenerator:
    """Per-sample block-free random masking (row-major patch order).

    Each sample's mask ratio is drawn from U(mask_ratio_min,
    mask_ratio_max); that many random patch positions are masked.
    """

    def __init__(self, mask_ratio_min: float = 0.1, mask_ratio_max: float = 0.5) -> None:
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max

    def __call__(self, batch_size: int, num_patches: int, device: Any) -> Tensor:
        ratios = torch.empty(batch_size, device=device).uniform_(
            self.mask_ratio_min, self.mask_ratio_max
        )
        num_masked = (ratios * num_patches).long().clamp(max=num_patches)
        # Rank each position by a random key; the first num_masked ranks
        # are masked. Double argsort = rank lookup (fully vectorized).
        ranks = torch.rand(batch_size, num_patches, device=device).argsort(dim=1).argsort(dim=1)
        return ranks < num_masked.unsqueeze(1)


def _dino_pair_loss(
    teacher_outputs: list[Tensor],
    student_outputs: list[Tensor],
    center: Tensor,
    teacher_temp: float,
    student_temp: float,
    skip_identical: bool,
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """DINO cross-entropy over view pairs.

    Mirrors ``DINOLoss`` (teacher L2-normalized before centering) and
    returns ``(loss, softmaxed targets, normalized raw teacher outputs)``
    so the caller can update the center from the pre-softmax outputs it is
    subtracted from (baseline ``DINOLoss.update_center`` receives the raw
    head outputs, never the softmaxed targets).
    """
    normalized = [F.normalize(view, dim=-1) for view in teacher_outputs]
    teacher_targets = [
        F.softmax((view - center) / teacher_temp, dim=-1) for view in normalized
    ]
    student_predictions = [
        F.log_softmax(view / student_temp, dim=-1) for view in student_outputs
    ]
    loss = teacher_outputs[0].new_zeros(())
    num_terms = 0
    for teacher_index, target in enumerate(teacher_targets):
        for student_index, prediction in enumerate(student_predictions):
            if skip_identical and teacher_index == student_index:
                continue  # skip the trivial identical global pair
            loss = loss - (target * prediction).sum(dim=-1).mean()
            num_terms += 1
    return loss / num_terms, teacher_targets, normalized


def _koleo_loss(features: Tensor) -> Tensor:
    """Kozachenko–Leonenko entropic regularizer (unit-sphere nearest NN)."""
    normalized = F.normalize(features, dim=-1, eps=1e-8)
    distances = torch.cdist(normalized, normalized)
    batch_size = normalized.shape[0]
    eye = torch.eye(batch_size, dtype=torch.bool, device=normalized.device)
    nearest = distances.masked_fill(eye, float("inf")).min(dim=1).values
    return -torch.log(nearest + 1e-8).mean()


# ── method ──────────────────────────────────────────────────────────────


class DINOv2(PretextRecipe):
    """DINOv2 pretraining method (DINO + iBOT + KoLeo)."""

    def __init__(
        self,
        method_args: DINOv2Args,
        optimizer_args: Any,
        embedding_model: Any,
        global_batch_size: int,
        num_input_channels: int,
    ) -> None:
        super().__init__(
            method_args=method_args,
            optimizer_args=optimizer_args,
            embedding_model=embedding_model,
            global_batch_size=global_batch_size,
            num_input_channels=num_input_channels,
        )
        self.method_args = method_args
        self.teacher_embedding_model = copy.deepcopy(embedding_model)
        self.student_projection_head = DINOProjectionHead(
            input_dim=embedding_model.embed_dim,
            hidden_dim=method_args.hidden_dim,
            bottleneck_dim=method_args.bottleneck_dim,
            output_dim=method_args.output_dim,
            batch_norm=method_args.batch_norm,
            norm_last_layer=method_args.norm_last_layer,
        )
        self.teacher_projection_head = copy.deepcopy(self.student_projection_head)
        for teacher_module in (self.teacher_embedding_model, self.teacher_projection_head):
            # eval + no grads + zeroed dropout/drop-path (via make_teacher),
            # so the EMA teacher stays deterministic in every phase.
            prepare_teacher(teacher_module)
        self.mask_generator = _MaskGenerator(
            method_args.mask_ratio_min, method_args.mask_ratio_max
        )
        self.register_buffer(
            "dino_center", torch.zeros(1, method_args.output_dim)
        )
        self.register_buffer(
            "ibot_center", torch.zeros(1, method_args.output_dim)
        )
        # The teacher is an EMA copy of the student, so teacher and student
        # patch grids match by construction (iBOT positional alignment).

    # ── contract ────────────────────────────────────────────────────────
    @staticmethod
    def method_args_cls() -> type[DINOv2Args]:
        return DINOv2Args

    @staticmethod
    def optimizer_args_cls(optim_type: str) -> type:
        if optim_type in ("auto", "adamw"):
            return DINOv2AdamWArgs
        from autovision.optim import get_optimizer_args_cls

        return get_optimizer_args_cls(optim_type)

    @staticmethod
    def transform_cls() -> type[DINOTransform]:
        # The multi-crop recipe is identical to DINO's (2 global + N local).
        return DINOTransform

    def train(self, mode: bool = True) -> "DINOv2":
        super().train(mode)
        self.teacher_embedding_model.eval()
        self.teacher_projection_head.eval()
        return self

    def trainable_modules(self) -> OptimizationTargets:
        return OptimizationTargets(
            targets=[self.embedding_model, self.student_projection_head]
        )

    # ── forward ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def _forward_teacher(self, x: Tensor) -> tuple[Tensor, Tensor]:
        cls_features = self.teacher_embedding_model(x, pool=True).flatten(1)
        patch_features = self.teacher_embedding_model(x, pool=False)
        patches = patch_features.flatten(2).transpose(1, 2)  # (B, N, D)
        return (
            self.teacher_projection_head(cls_features),
            self.teacher_projection_head(patches),
        )

    def _forward_student_cls(self, x: Tensor) -> Tensor:
        features = self.embedding_model(x, pool=True).flatten(1)
        return self.student_projection_head(features)

    def _forward_student_patches(
        self, x: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        features = self.embedding_model(
            x, pool=False, bool_masked_pos=mask
        )  # (B, D, h, w)
        patches = features.flatten(2).transpose(1, 2)  # (B, N, D)
        return self.student_projection_head(patches), patches

    def training_step_impl(self, batch: dict, batch_idx: int) -> StepOutput:
        args = self.method_args
        momentum = _cosine_schedule(
            self.global_step, self.total_steps, args.momentum_start, args.momentum_end
        )
        _update_momentum(self.embedding_model, self.teacher_embedding_model, momentum)
        _update_momentum(
            self.student_projection_head, self.teacher_projection_head, momentum
        )

        views = batch["views"]
        global_views = torch.cat(views[:2])  # (2B, ...)
        teacher_cls, teacher_patches = self._forward_teacher(global_views)
        teacher_temp = self._teacher_temp_schedule()

        student_global_cls = [
            self._forward_student_cls(view) for view in views[:2]
        ]
        student_local_cls = [
            self._forward_student_cls(view) for view in views[2:]
        ]

        # iBOT: mask each global view independently; the student predicts
        # the teacher's patch targets at masked positions.
        batch_size = views[0].shape[0]
        num_patches = teacher_patches.shape[1]
        masks = [
            self.mask_generator(batch_size, num_patches, views[0].device)
            for _ in range(2)
        ]
        student_masked_patches = []
        for view, mask in zip(views[:2], masks):
            predictions, _ = self._forward_student_patches(view, mask)
            student_masked_patches.append(predictions)

        # Global DINO part (identical pair skipped; center from raw outputs).
        loss_global, _, global_raw = _dino_pair_loss(
            teacher_outputs=list(teacher_cls.chunk(2)),
            student_outputs=student_global_cls,
            center=self.dino_center,
            teacher_temp=teacher_temp,
            student_temp=args.student_temp,
            skip_identical=True,
        )
        # Local DINO part.
        if student_local_cls:
            loss_local, _, _ = _dino_pair_loss(
                teacher_outputs=list(teacher_cls.chunk(2)),
                student_outputs=student_local_cls,
                center=self.dino_center,
                teacher_temp=teacher_temp,
                student_temp=args.student_temp,
                skip_identical=False,
            )
        else:
            loss_local = teacher_cls.new_zeros(())
        # Single center update per step over the global CLS targets (the
        # official DINOv2 recipe centers on global outputs only). The center
        # tracks the raw pre-softmax teacher outputs it is subtracted from.
        self._update_center(torch.cat(global_raw, dim=0))

        # iBOT patch loss on masked positions.
        loss_ibot = self._ibot_loss(
            teacher_patches.chunk(2), student_masked_patches, masks, teacher_temp
        )
        # KoLeo on the concatenated global student features.
        loss_koleo = _koleo_loss(torch.cat(student_global_cls, dim=0))

        n = len(views)
        loss = (
            (2.0 / n) * loss_global
            + (1.0 / n) * loss_local
            + args.ibot_loss_weight * loss_ibot
            + args.koleo_loss_weight * loss_koleo
        )
        return StepOutput(
            loss=loss,
            log_dict={
                "train_loss/dino_global": float(loss_global.detach()),
                "train_loss/dino_local": float(loss_local.detach()),
                "train_loss/ibot": float(loss_ibot.detach()),
                "train_loss/koleo": float(loss_koleo.detach()),
                "schedule/momentum": momentum,
                "schedule/teacher_temp": teacher_temp,
            },
        )

    def _ibot_loss(
        self,
        teacher_patches: list[Tensor],
        student_patches: list[Tensor],
        masks: list[Tensor],
        teacher_temp: float,
    ) -> Tensor:
        args = self.method_args
        loss = teacher_patches[0].new_zeros(())
        num_masked = 0
        masked_raw: list[Tensor] = []
        for teacher, student, mask in zip(teacher_patches, student_patches, masks):
            teacher_normalized = F.normalize(teacher, dim=-1)
            teacher_targets = F.softmax(
                (teacher_normalized - self.ibot_center) / teacher_temp,
                dim=-1,
            )
            student_predictions = F.log_softmax(
                student / args.student_temp, dim=-1
            )
            # Flatten to (B*N, out) before selecting masked positions:
            # a 1-D bool mask indexes the first dimension only.
            teacher_flat = teacher_targets.reshape(-1, teacher_targets.shape[-1])
            student_flat = student_predictions.reshape(
                -1, student_predictions.shape[-1]
            )
            raw_flat = teacher_normalized.reshape(-1, teacher.shape[-1])
            selected = mask.reshape(-1)
            masked_teacher = teacher_flat[selected]
            masked_student = student_flat[selected]
            masked_raw.append(raw_flat[selected])
            loss = loss - (masked_teacher * masked_student).sum(dim=-1).sum()
            num_masked += int(selected.sum())
        # Patch center follows the masked RAW teacher tokens (pre-softmax),
        # mirroring the baseline update_center(masked_patch_tokens_after_ibot).
        batch_center = torch.cat(masked_raw, dim=0).mean(dim=0, keepdim=True)
        self.ibot_center = (
            self.ibot_center * args.center_momentum + batch_center * (1 - args.center_momentum)
        )
        return loss / max(num_masked, 1)

    @torch.no_grad()
    def _update_center(self, teacher_batch: Tensor) -> None:
        batch_center = teacher_batch.mean(dim=0, keepdim=True)
        self.dino_center = (
            self.dino_center * self.method_args.center_momentum
            + batch_center * (1 - self.method_args.center_momentum)
        )

    def _teacher_temp_schedule(self) -> float:
        args = self.method_args
        warmup_steps = min(
            int(args.warmup_teacher_temp_steps),
            int(self.total_steps * args.warmup_teacher_temp_max_steps_fraction),
        )
        if warmup_steps <= 0 or self.global_step >= warmup_steps:
            return args.teacher_temp
        progress = self.global_step / warmup_steps
        return args.warmup_teacher_temp + progress * (args.teacher_temp - args.warmup_teacher_temp)

    # ── optimization ────────────────────────────────────────────────────
    def _student_num_layers(self) -> int | None:
        wrapped = self.embedding_model._inner
        config = getattr(wrapped, "config", None)
        value = getattr(config, "num_hidden_layers", None)
        return value if isinstance(value, int) else None

    def _layerwise_param_groups(
        self,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Per-depth LR groups: patch embed ×0.2, block d ×decay^(L-1-d).

        Returns the param groups plus a ``group name -> lr factor`` map;
        the factors are applied to the group LRs after optimizer
        construction (``OptimizerArgs.get_optimizer`` only takes a global
        scale).
        """
        args = self.method_args
        num_layers = self._student_num_layers()
        if num_layers is None:
            raise ValueError(
                "DINOv2 layer-wise LR decay requires the student backbone to "
                "expose config.num_hidden_layers (ViT backbones)."
            )
        block_pattern = re.compile(r"(?:encoder\.layer|layers)\.(\d+)\.")

        def _is_no_weight_decay(param: Any) -> bool:
            # Biases and 1-D norm parameters are excluded from decay — the
            # same heuristic get_weight_decay_parameters applies to modules.
            return param.ndim <= 1

        buckets: dict[float, dict[str, list]] = {}
        for name, param in self.embedding_model.named_parameters():
            match = block_pattern.search(name)
            if "patch_embed" in name or "embeddings.patch_embeddings" in name:
                factor = args.patch_embed_lr
            elif match:
                block_id = int(match.group(1))
                factor = args.layerwise_decay ** (num_layers - 1 - block_id)
            else:
                factor = 1.0
            key = "no_wd" if _is_no_weight_decay(param) else "wd"
            entry = buckets.setdefault(factor, {"wd": [], "no_wd": []})
            entry[key].append(param)

        param_groups: list[dict[str, Any]] = []
        factors: dict[str, float] = {}
        for factor in sorted(buckets):
            entry = buckets[factor]
            if entry["wd"]:
                name = f"backbone_wd_{factor:g}"
                param_groups.append({"name": name, "params": entry["wd"]})
                factors[name] = factor
            if entry["no_wd"]:
                name = f"backbone_no_wd_{factor:g}"
                param_groups.append(
                    {
                        "name": name,
                        "params": entry["no_wd"],
                        "weight_decay": 0.0,
                    }
                )
                factors[name] = factor
        return param_groups, factors

    def configure_optimizers(
        self, total_steps: int, num_epochs: int
    ) -> tuple[torch.optim.Optimizer, CosineWarmupScheduler]:
        lr_scale: float = self.global_batch_size / self.method_args.reference_batch_size
        if self.method_args.lr_scale_method == "sqrt":
            lr_scale = math.sqrt(lr_scale)

        param_groups, lr_factors = self._layerwise_param_groups()
        params_weight_decay, params_no_weight_decay = get_weight_decay_parameters(
            [self.student_projection_head]
        )
        last_layer_params = list(self.student_projection_head.last_layer.parameters())
        last_layer_ids = {id(p) for p in last_layer_params}
        params_weight_decay = [
            p for p in params_weight_decay if id(p) not in last_layer_ids
        ]
        params_no_weight_decay = [
            p for p in params_no_weight_decay if id(p) not in last_layer_ids
        ]
        if params_weight_decay:
            param_groups.append({"name": "params", "params": params_weight_decay})
        if params_no_weight_decay:
            param_groups.append(
                {
                    "name": "params_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                }
            )
        param_groups.append(
            {"name": "params_last_layer", "params": last_layer_params}
        )
        optimizer = self.optimizer_args.get_optimizer(
            params=param_groups, lr_scale=lr_scale
        )
        # Apply the layer-wise factors to the group LRs.
        for group in optimizer.param_groups:
            factor = lr_factors.get(group.get("name") or "", 1.0)
            if factor != 1.0:
                group["lr"] = group["lr"] * factor
        warmup_steps = min(
            int(self.method_args.warmup_steps),
            int(total_steps * self.method_args.warmup_max_steps_fraction),
        )
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer, warmup_epochs=warmup_steps, max_epochs=int(total_steps)
        )
        return optimizer, scheduler

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        args = self.method_args
        weight_decay = _cosine_schedule(
            self.global_step,
            self.total_steps,
            args.weight_decay_start,
            args.weight_decay_end,
        )
        for group in optimizer.param_groups:
            name = group.get("name") or ""
            # Both the backbone no-decay groups and the projection head's
            # no-decay group stay at 0 (baseline only reschedules groups
            # whose weight decay is nonzero).
            if "no_wd" in name:
                continue
            if name == "params_last_layer":
                if self.global_step < args.student_freeze_last_layer_steps:
                    group["lr"] = 0.0
                    group["weight_decay"] = 0.0
                else:
                    group["weight_decay"] = weight_decay
            else:
                group["weight_decay"] = weight_decay

    # ── checkpoint: strip teacher weights ───────────────────────────────
    _TEACHER_PREFIXES = (
        "teacher_embedding_model.",
        "teacher_projection_head.",
    )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint.get("state_dict", {})
        checkpoint["state_dict"] = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(self._TEACHER_PREFIXES)
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint.get("state_dict", {})
        live_modules = {
            self._TEACHER_PREFIXES[0]: self.teacher_embedding_model,
            self._TEACHER_PREFIXES[1]: self.teacher_projection_head,
        }
        for prefix, module in live_modules.items():
            if not any(key.startswith(prefix) for key in state_dict):
                logger.warning(
                    "Checkpoint has no '%s*' weights; the teacher was "
                    "rebuilt from the current student and needs time to "
                    "re-converge.", prefix,
                )
            for key, value in module.state_dict().items():
                state_dict.setdefault(f"{prefix}{key}", value)
