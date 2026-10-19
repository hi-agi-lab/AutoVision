# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DINO: self-distillation with a momentum teacher over multi-crop views.

Recipe (design doc §7.4, mirroring the baseline):

- Views: 2 global crops 224² (scale 0.14–1.0; view 1 with light blur +
  solarization) + 6 local crops 96² (scale 0.05–0.14, blur p=0.5).
- Student: embedding model + DINO head; teacher: frozen EMA copies, updated
  each step with a cosine momentum schedule (0.99–0.996 → 1.0 by dataset
  size).
- DINOLoss: cross-entropy between sharpened teacher distributions
  (centered, temperature-warped 0.02→0.07 by dataset size) and student
  predictions (τ=0.1); the identical global pair is skipped.
- Prototypes ("output_dim") bucketed by dataset size 1024 → 65536; the head
  bottleneck is 256.
- Last head layer frozen (lr=0) for the first 1250 steps; weight decay
  follows a cosine schedule; gradient clip norm 3.0.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init

from autovision.augment.method_transform import MultiViewTransform
from autovision.augment.transform_args import (
    ColorJitterArgs,
    GaussianBlurArgs,
    MethodTransformArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomResizeArgs,
    SolarizeArgs,
)
from autovision.augment.view_transform import ViewTransform
from autovision.optim import (
    AdamWArgs,
    OptimizationTargets,
    SGDArgs,
    get_weight_decay_parameters,
)
from autovision.optim.schedulers import CosineWarmupScheduler
from autovision.pretext.method_args import PretextArgs
from autovision.pretext.recipe import PretextRecipe, StepOutput, prepare_teacher
from autovision.scaling import (
    IMAGENET_SIZE,
    ScalingInfo,
    get_bucket_value,
    interpolate,
)

__all__ = ["DINO", "DINOArgs", "DINOLoss", "DINOProjectionHead"]

logger = logging.getLogger("autovision")

# (dataset_size <, prototypes) — small datasets get small output dims.
_OUTPUT_DIM_BUCKETS: tuple[tuple[float, int], ...] = (
    (20_000, 1024),
    (50_000, 2048),
    (100_000, 4096),
    (200_000, 16384),
    (500_000, 32768),
    (float("inf"), 65536),
)


class DINOArgs(PretextArgs):
    """Arguments for the DINO method."""

    gradient_clip_val: float = 3.0
    # projection head
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    output_dim: int | Literal["auto"] = "auto"
    student_freeze_last_layer_steps: int | Literal["auto"] = "auto"
    batch_norm: bool = False
    norm_last_layer: bool = True
    # loss
    teacher_temp: float | Literal["auto"] = "auto"
    warmup_teacher_temp: float | Literal["auto"] = "auto"
    warmup_teacher_temp_steps: int | Literal["auto"] = "auto"
    warmup_teacher_temp_max_steps_fraction: float = 0.3
    student_temp: float = 0.1
    center_momentum: float = 0.9
    # teacher momentum schedule
    momentum_start: float | Literal["auto"] = "auto"
    momentum_end: float = 1.0
    # weight decay schedule
    weight_decay_start: float | Literal["auto"] = "auto"
    weight_decay_end: float | Literal["auto"] = "auto"
    # LR warmup (steps): ImageNet default 12500, capped at 10% of the run.
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
        dataset_size = scaling.dataset_size if scaling is not None else None

        if self.output_dim == "auto":
            if dataset_size is None:
                raise ValueError("output_dim='auto' requires the dataset size.")
            # The default 65536 prototypes are far too many for small sets.
            self.output_dim = get_bucket_value(
                input=dataset_size, buckets=_OUTPUT_DIM_BUCKETS
            )
        if self.teacher_temp == "auto":
            if dataset_size is None:
                raise ValueError("teacher_temp='auto' requires the dataset size.")
            # Sharper (lower) temperatures avoid collapse on small datasets.
            self.teacher_temp = interpolate(
                dataset_size, 20_000, IMAGENET_SIZE, 0.02, 0.07, round_ndigits=2
            )
        if self.warmup_teacher_temp == "auto":
            self.warmup_teacher_temp = min(
                self.teacher_temp,
                interpolate(self.teacher_temp, 0.02, 0.07, 0.02, 0.04, round_ndigits=2),
            )
        if self.warmup_teacher_temp_steps == "auto":
            # ImageNet default: 30 epochs * 1.28M / 1024 ~= 37500 steps.
            self.warmup_teacher_temp_steps = 37500
        if self.student_freeze_last_layer_steps == "auto":
            # ImageNet default: 1 epoch * 1.28M / 1024 ~= 1250 steps.
            self.student_freeze_last_layer_steps = 1250
        if self.momentum_start == "auto":
            if dataset_size is None:
                raise ValueError("momentum_start='auto' requires the dataset size.")
            # Slower teacher updates stabilize early training on small sets.
            self.momentum_start = interpolate(
                dataset_size, 20_000, IMAGENET_SIZE, 0.99, 0.996, round_ndigits=3
            )
        weight_decay = getattr(optimizer_args, "weight_decay", None)
        if not isinstance(weight_decay, (int, float)):
            raise ValueError(
                "DINO requires a numeric weight decay on the optimizer args; "
                f"got {weight_decay!r}."
            )
        if self.weight_decay_start == "auto":
            self.weight_decay_start = weight_decay
        if self.weight_decay_end == "auto":
            self.weight_decay_end = weight_decay


class DINOAdamWArgs(AdamWArgs):
    lr: float = 0.0005
    weight_decay: float = 0.04


class DINOSGDArgs(SGDArgs):
    lr: float = 0.03
    weight_decay: float = 0.0001


# ── transform ───────────────────────────────────────────────────────────


class DINOTransformArgs(MethodTransformArgs):
    """Multi-crop recipe: 2 global views + 6 local views."""

    image_size: tuple[int, int] = (224, 224)
    num_local_views: int = 6
    local_view_size: tuple[int, int] = (96, 96)
    random_resize: RandomResizeArgs = RandomResizeArgs(min_scale=0.14, max_scale=1.0)
    local_resize: RandomResizeArgs = RandomResizeArgs(min_scale=0.05, max_scale=0.14)
    random_flip: RandomFlipArgs | None = RandomFlipArgs()
    color_jitter: ColorJitterArgs | None = ColorJitterArgs(
        prob=0.8, strength=0.5, brightness=0.8, contrast=0.8, saturation=0.4, hue=0.2
    )
    random_gray_scale: float | None = 0.2
    gaussian_blur: GaussianBlurArgs | None = GaussianBlurArgs(
        prob=1.0, sigmas=(0.1, 2.0)
    )
    # Global view 1 gets light blur plus solarization.
    global_view_1_blur: GaussianBlurArgs | None = GaussianBlurArgs(
        prob=0.1, sigmas=(0.1, 2.0)
    )
    global_view_1_solarize: SolarizeArgs | None = SolarizeArgs()
    # Local views get medium blur and no solarization.
    local_view_blur: GaussianBlurArgs | None = GaussianBlurArgs(
        prob=0.5, sigmas=(0.1, 2.0)
    )
    solarize: None = None
    normalize: NormalizeArgs = NormalizeArgs()


class DINOTransform(MultiViewTransform):
    """2 global 224² crops + ``num_local_views`` 96² crops."""

    def __init__(self, transform_args: DINOTransformArgs) -> None:
        global_view_0 = ViewTransform(transform_args)
        global_view_1_args = transform_args.model_copy(
            update={
                "gaussian_blur": transform_args.global_view_1_blur,
                "solarize": transform_args.global_view_1_solarize,
            }
        )
        global_view_1 = ViewTransform(global_view_1_args)
        local_args = transform_args.model_copy(
            update={
                "image_size": transform_args.local_view_size,
                "random_resize": transform_args.local_resize,
                "gaussian_blur": transform_args.local_view_blur,
                "solarize": None,
            }
        )
        local_view = ViewTransform(local_args)
        transforms = [global_view_0, global_view_1] + [
            local_view
        ] * transform_args.num_local_views
        super().__init__(transforms=transforms, transform_args=transform_args)

    @staticmethod
    def transform_args_cls() -> type[DINOTransformArgs]:
        return DINOTransformArgs


# ── modules ─────────────────────────────────────────────────────────────


class _WeightNormLinear(nn.Module):
    """Linear layer with weight normalization: ``w = g * v / ||v||``.

    Explicit re-implementation instead of ``torch.nn.utils`` weight_norm:
    the parameters stay plain graph leaves (deepcopy-safe for the teacher
    copy on torch >= 2) and ``weight_g`` remains an ordinary attribute.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight_v = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_g = nn.Parameter(torch.ones(out_features, 1))

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight_g * F.normalize(self.weight_v, dim=1)
        return F.linear(x, weight)


class DINOProjectionHead(nn.Module):
    """DINO head: MLP to a 256-d bottleneck + weight-norm prototype layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        num_layers: int = 3,
        batch_norm: bool = False,
        norm_last_layer: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = _WeightNormLinear(bottleneck_dim, output_dim)
        if norm_last_layer:
            # Freeze the norm vector: no collapse direction can grow freely.
            self.last_layer.weight_g.requires_grad = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.last_layer(self.mlp(x))


class DINOLoss(nn.Module):
    """DINO cross-entropy between centered/sharpened teacher targets and
    student predictions over all view pairs (identical pair skipped).

    Teacher outputs are L2-normalized before centering (following
    facebookresearch/dino): the norm keeps the centering/temperature
    semantics well-defined even when the head's last-layer weight norm is
    unfrozen and the output scale drifts.
    """

    def __init__(
        self,
        output_dim: int,
        teacher_temp: float,
        warmup_teacher_temp: float,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.warmup_teacher_temp = warmup_teacher_temp
        self.register_buffer("center", torch.zeros(1, output_dim))

    def forward(
        self,
        teacher_outputs: list[Tensor],
        student_outputs: list[Tensor],
        teacher_temp: float,
    ) -> Tensor:
        student_predictions = [
            F.log_softmax(view / self.student_temp, dim=-1)
            for view in student_outputs
        ]
        # The center must track the same quantity it is subtracted from:
        # the raw (pre-softmax) teacher outputs (baseline DINOLoss feeds the
        # unsoftmaxed head outputs to update_center).
        normalized = [F.normalize(view, dim=-1) for view in teacher_outputs]
        teacher_targets = [
            F.softmax((view - self.center) / teacher_temp, dim=-1)
            for view in normalized
        ]
        loss = teacher_outputs[0].new_zeros(())
        num_terms = 0
        for teacher_index, target in enumerate(teacher_targets):
            for student_index, prediction in enumerate(student_predictions):
                if teacher_index == student_index:
                    continue  # skip the trivial identical global pair
                loss = loss - (target * prediction).sum(dim=-1).mean()
                num_terms += 1
        self._update_center(torch.cat(normalized, dim=0))
        return loss / num_terms

    @torch.no_grad()
    def _update_center(self, teacher_batch: Tensor) -> None:
        batch_center = teacher_batch.mean(dim=0, keepdim=True)
        self.center = (
            self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        )


@torch.no_grad()
def _update_momentum(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for student_param, teacher_param in zip(
        student.parameters(), teacher.parameters()
    ):
        teacher_param.mul_(momentum).add_(student_param.detach(), alpha=1 - momentum)
    # Buffers follow too, else a BN teacher would run forever on its
    # initialization stats while the student's live batch stats drift away.
    for student_buffer, teacher_buffer in zip(student.buffers(), teacher.buffers()):
        if student_buffer.dtype.is_floating_point:
            teacher_buffer.mul_(momentum).add_(
                student_buffer.detach(), alpha=1 - momentum
            )
        else:
            teacher_buffer.copy_(student_buffer)


def _cosine_schedule(step: int, max_steps: int, start: float, end: float) -> float:
    """Cosine anneal from ``start`` (step 0) to ``end`` (step ``max_steps``)."""
    if max_steps <= 0:
        return start
    progress = min(max(step / max_steps, 0.0), 1.0)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── method ──────────────────────────────────────────────────────────────


class DINO(PretextRecipe):
    """DINO pretraining method (Siamese network with momentum teacher)."""

    def __init__(
        self,
        method_args: DINOArgs,
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
        # Teacher: frozen EMA copies of the student's backbone and head.
        # Parameters MUST have requires_grad=False: under DDP the engine
        # wraps the whole method and every registered parameter that never
        # produces a gradient kills the reduction on the next forward.
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
        for teacher_module in (
            self.teacher_embedding_model,
            self.teacher_projection_head,
        ):
            # eval + no grads + zeroed dropout/drop-path (via make_teacher),
            # so the EMA teacher stays deterministic in every phase.
            prepare_teacher(teacher_module)
        self.criterion = DINOLoss(
            output_dim=method_args.output_dim,
            teacher_temp=method_args.teacher_temp,
            warmup_teacher_temp=method_args.warmup_teacher_temp,
            student_temp=method_args.student_temp,
            center_momentum=method_args.center_momentum,
        )

    # ── contract ────────────────────────────────────────────────────────
    @staticmethod
    def method_args_cls() -> type[DINOArgs]:
        return DINOArgs

    @staticmethod
    def optimizer_args_cls(optim_type: str) -> type:
        if optim_type in ("auto", "sgd"):
            return DINOSGDArgs
        if optim_type == "adamw":
            return DINOAdamWArgs
        from autovision.optim import get_optimizer_args_cls

        return get_optimizer_args_cls(optim_type)

    @staticmethod
    def transform_cls() -> type[DINOTransform]:
        return DINOTransform

    def train(self, mode: bool = True) -> "DINO":
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
    def _forward_teacher(self, x: Tensor) -> Tensor:
        features = self.teacher_embedding_model(x, pool=True).flatten(1)
        return self.teacher_projection_head(features)

    def _forward_student(self, x: Tensor) -> Tensor:
        features = self.embedding_model(x, pool=True).flatten(1)
        return self.student_projection_head(features)

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
        global_views = torch.cat(views[:2])
        teacher_outputs = self._forward_teacher(global_views)
        if len(views) > 2:
            local_views = torch.cat(views[2:])
            student_outputs = torch.cat(
                [self._forward_student(global_views), self._forward_student(local_views)]
            )
        else:
            student_outputs = self._forward_student(global_views)

        teacher_temp = self._teacher_temp_schedule()
        loss = self.criterion(
            teacher_outputs=list(teacher_outputs.chunk(2)),
            student_outputs=list(student_outputs.chunk(len(views))),
            teacher_temp=teacher_temp,
        )
        return StepOutput(
            loss=loss,
            log_dict={
                "schedule/momentum": momentum,
                "schedule/teacher_temp": teacher_temp,
            },
        )

    def _teacher_temp_schedule(self) -> float:
        args = self.method_args
        warmup_steps = args.warmup_teacher_temp_steps
        warmup_steps = min(
            int(warmup_steps), int(self.total_steps * args.warmup_teacher_temp_max_steps_fraction)
        )
        if warmup_steps <= 0 or self.global_step >= warmup_steps:
            return args.teacher_temp
        warmup_temp = args.warmup_teacher_temp
        progress = self.global_step / warmup_steps
        return warmup_temp + progress * (args.teacher_temp - warmup_temp)

    # ── optimization ────────────────────────────────────────────────────
    def configure_optimizers(
        self, total_steps: int, num_epochs: int
    ) -> tuple[torch.optim.Optimizer, CosineWarmupScheduler]:
        lr_scale: float = self.global_batch_size / self.method_args.reference_batch_size
        if self.method_args.lr_scale_method == "sqrt":
            lr_scale = math.sqrt(lr_scale)

        params_weight_decay, params_no_weight_decay = get_weight_decay_parameters(
            [self.embedding_model, self.student_projection_head]
        )
        last_layer_params = list(self.student_projection_head.last_layer.parameters())
        last_layer_ids = {id(p) for p in last_layer_params}
        params_weight_decay = [
            p for p in params_weight_decay if id(p) not in last_layer_ids
        ]
        params_no_weight_decay = [
            p for p in params_no_weight_decay if id(p) not in last_layer_ids
        ]

        param_groups: list[dict[str, Any]] = [
            {"name": "params", "params": params_weight_decay},
            {"name": "params_last_layer", "params": last_layer_params},
        ]
        if params_no_weight_decay:
            param_groups.append(
                {
                    "name": "params_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                }
            )
        optimizer = self.optimizer_args.get_optimizer(
            params=param_groups, lr_scale=lr_scale
        )
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
        freeze_steps = args.student_freeze_last_layer_steps
        for group in optimizer.param_groups:
            name = group.get("name")
            if name == "params":
                group["weight_decay"] = weight_decay
            elif name == "params_last_layer":
                if self.global_step < freeze_steps:
                    group["lr"] = 0.0
                    group["weight_decay"] = 0.0
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
                # Checkpoints strip the teacher for size; rebuilding it from
                # the initial student means a long EMA re-convergence tail.
                logger.warning(
                    "Checkpoint has no '%s*' weights; the teacher was "
                    "rebuilt from the current student and needs time to "
                    "re-converge.", prefix,
                )
            for key, value in module.state_dict().items():
                state_dict.setdefault(f"{prefix}{key}", value)
