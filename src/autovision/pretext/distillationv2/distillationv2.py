# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Distillation V2: intermediate-layer feature MSE.

Recipe (design doc §7.3, blueprint in volume 12 §4.2):

- Student and teacher both tap their last two transformer blocks; the two
  feature maps are flattened to patch tokens and concatenated channel-wise.
- Loss: plain MSE between student and teacher features — no queue, no
  normalization, no projection head (a documented contrast to V3).
- Single-view 224² transform (no mixup).
- LARS recipe: lr 9.0, momentum 0.9, wd 1e-6 at the 1536 reference batch.

Because there is no projection, the student embed dim must equal the
teacher feature dim; a mismatch raises at construction. Patch grids may
differ (teacher/student patch sizes): the student's feature map is
bilinearly aligned to the teacher's grid before the MSE.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from autovision.augment.method_transform import MultiViewTransform
from autovision.augment.transform_args import (
    ColorJitterArgs,
    GaussianBlurArgs,
    MethodTransformArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomResizeArgs,
)
from autovision.augment.view_transform import ViewTransform
from autovision.optim import AdamWArgs, LARSArgs, OptimizationTargets, OptimizerArgs
from autovision.pretext.method_args import PretextArgs
from autovision.pretext.recipe import PretextRecipe, StepOutput
from autovision.scaling import ScalingInfo

__all__ = ["DistillationV2", "DistillationV2Args"]


class DistillationV2Args(PretextArgs):
    """Arguments for the DistillationV2 method."""

    auto_steps: ClassVar[int | None] = None
    auto_epochs: ClassVar[int] = 100

    teacher: str | Any = "dinov2/vits14"
    teacher_args: dict | None = None
    teacher_weights: str | Any | None = None
    teacher_pretrained: bool = True
    # 0-based block indices (a, b) to tap; "auto" = the last two blocks.
    teacher_layers: tuple[int, int] | Literal["auto"] = "auto"
    lr_scale_method: Literal["linear", "sqrt"] = "sqrt"
    reference_batch_size: int = 1536

    def resolve_auto_fields(
        self,
        scaling: ScalingInfo | None = None,
        optimizer_args: OptimizerArgs | None = None,
        wrapped_model: Any = None,
    ) -> None:
        super().resolve_auto_fields(
            scaling=scaling,
            optimizer_args=optimizer_args,
            wrapped_model=wrapped_model,
        )
        if isinstance(self.teacher_layers, tuple) and len(self.teacher_layers) != 2:
            raise ValueError(
                "teacher_layers must be a (start, end) pair of 0-based block "
                f"indices, got {self.teacher_layers!r}."
            )
        if isinstance(self.teacher_layers, tuple) and self.teacher_layers[0] < 0:
            raise ValueError(
                "teacher_layers must be 0-based indices, got "
                f"{self.teacher_layers!r}."
            )
        if isinstance(self.teacher_layers, tuple) and self.teacher_layers[0] > self.teacher_layers[1]:
            raise ValueError(
                "teacher_layers must be ordered (start <= end), got "
                f"{self.teacher_layers!r}."
            )


class DistillationV2AdamWArgs(AdamWArgs):
    """AdamW alternative: lr tuned for distillation."""

    lr: float = 0.0005
    weight_decay: float | Literal["auto"] = "auto"


class DistillationV2LARSArgs(LARSArgs):
    """LARS recipe: lr 9.0 at the 1536 reference batch size."""

    lr: float = 9.0
    momentum: float = 0.9
    weight_decay: float = 1e-6


class DistillationV2TransformArgs(MethodTransformArgs):
    """Single-view 224² recipe."""

    image_size: tuple[int, int] = (224, 224)
    random_resize: RandomResizeArgs = RandomResizeArgs(min_scale=0.14, max_scale=1.0)
    random_flip: RandomFlipArgs | None = RandomFlipArgs()
    color_jitter: ColorJitterArgs | None = ColorJitterArgs(
        prob=0.8, strength=0.5, brightness=0.8, contrast=0.8, saturation=0.4, hue=0.2
    )
    random_gray_scale: float | None = 0.2
    gaussian_blur: GaussianBlurArgs | None = GaussianBlurArgs(
        prob=1.0, sigmas=(0.0, 0.1)
    )
    solarize: None = None
    normalize: NormalizeArgs = NormalizeArgs()


class DistillationV2Transform(MultiViewTransform):
    """Single-view transform: one 224² view per image."""

    def __init__(self, transform_args: DistillationV2TransformArgs) -> None:
        super().__init__(
            transforms=[ViewTransform(transform_args)], transform_args=transform_args
        )

    @staticmethod
    def transform_args_cls() -> type[DistillationV2TransformArgs]:
        return DistillationV2TransformArgs


def _num_hidden_layers(wrapped_model: Any) -> int | None:
    """Best-effort block count from the wrapper's config (ViT backbones)."""
    candidates = [wrapped_model]
    raw_module = getattr(wrapped_model, "raw_module", None)
    if callable(raw_module):
        candidates.append(raw_module())
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        value = getattr(config, "num_hidden_layers", None)
        if isinstance(value, int) and value > 1:
            return value
    return None


def _resolve_teacher_layers(teacher: Any, requested: Any) -> tuple[int, int]:
    if isinstance(requested, tuple):
        return requested
    num_layers = _num_hidden_layers(teacher)
    if num_layers is None:
        raise ValueError(
            "teacher_layers='auto' requires a teacher config with "
            "num_hidden_layers; pass teacher_layers=(a, b) explicitly for "
            f"'{type(teacher).__name__}'."
        )
    return (num_layers - 2, num_layers - 1)


def _validate_layers_within_depth(
    layers: tuple[int, int], wrapper: Any, name: str
) -> None:
    """Fail fast when explicit tap indices exceed the backbone depth."""
    num_layers = _num_hidden_layers(wrapper)
    if num_layers is not None and layers[1] > num_layers - 1:
        raise ValueError(
            f"{name} tap layers {layers} exceed the backbone depth "
            f"(num_hidden_layers={num_layers})."
        )


class DistillationV2(PretextRecipe):
    """Intermediate-layer MSE distillation (no queue, no normalization)."""

    gradient_clip_val: float | None = 1.0

    def __init__(
        self,
        method_args: DistillationV2Args,
        optimizer_args: OptimizerArgs,
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
        # Late import: the method registry (teachers) imports this
        # package, so module-level import would be circular.
        from autovision.pretext.teachers import build_teacher

        self.teacher_embedding_model = build_teacher(
            method_args, num_input_channels=num_input_channels
        )
        self.teacher_embedding_dim: int = self.teacher_embedding_model.encoder_dim()
        if embedding_model.embed_dim != self.teacher_embedding_dim:
            raise ValueError(
                "DistillationV2 has no projection head: the student embed dim "
                f"({embedding_model.embed_dim}) must equal the teacher feature "
                f"dim ({self.teacher_embedding_dim}). Choose a same-width "
                "teacher, or use DistillationV3."
            )
        self.teacher_layers: tuple[int, int] = _resolve_teacher_layers(
            self.teacher_embedding_model, method_args.teacher_layers
        )
        # The student taps its own last two blocks under "auto" (depths may
        # differ across architectures); explicit indices apply to both.
        self.student_layers: tuple[int, int] = (
            self.teacher_layers
            if isinstance(method_args.teacher_layers, tuple)
            else _resolve_teacher_layers(embedding_model._inner, "auto")
        )
        # Explicit indices must exist in both backbones (depths may differ).
        if isinstance(method_args.teacher_layers, tuple):
            _validate_layers_within_depth(
                self.teacher_layers, self.teacher_embedding_model, "teacher"
            )
            _validate_layers_within_depth(
                self.student_layers, embedding_model._inner, "student"
            )

    # ── contract ────────────────────────────────────────────────────────
    @staticmethod
    def method_args_cls() -> type[DistillationV2Args]:
        return DistillationV2Args

    @staticmethod
    def optimizer_args_cls(optim_type: str) -> type[OptimizerArgs]:
        if optim_type in ("auto", "lars"):
            return DistillationV2LARSArgs
        if optim_type == "adamw":
            return DistillationV2AdamWArgs
        from autovision.optim import get_optimizer_args_cls

        return get_optimizer_args_cls(optim_type)

    @staticmethod
    def transform_cls() -> type[DistillationV2Transform]:
        return DistillationV2Transform

    def train(self, mode: bool = True) -> "DistillationV2":
        super().train(mode)
        self.teacher_embedding_model.eval()
        return self

    def trainable_modules(self) -> OptimizationTargets:
        return OptimizationTargets(targets=[self.embedding_model])

    # ── forward ─────────────────────────────────────────────────────────
    @staticmethod
    def _concat_patches(
        outputs: list[dict[str, Tensor]],
    ) -> tuple[Tensor, tuple[int, int]]:
        """Concatenate tap features channel-wise into (B, N, sum(C)) tokens.

        All taps must share the same patch grid (true for transformer
        blocks of one backbone); a mismatch raises with a clear message.
        """
        grids = [tuple(output["features"].shape[-2:]) for output in outputs]
        if any(grid != grids[0] for grid in grids[1:]):
            raise ValueError(
                f"Tapped layers must share one patch grid, got {grids}."
            )
        tokens = [
            output["features"].flatten(start_dim=2).transpose(1, 2)
            for output in outputs
        ]
        return torch.cat(tokens, dim=-1), grids[0]

    @torch.no_grad()
    def _forward_teacher(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        outputs = self.teacher_embedding_model.stage_feature_maps(
            x, self.teacher_layers
        )
        tokens, grid = self._concat_patches(outputs)
        return tokens, grid

    def _forward_student(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        outputs = self.embedding_model.stage_feature_maps(
            x, self.student_layers
        )
        tokens, grid = self._concat_patches(outputs)
        return tokens, grid

    def training_step_impl(self, batch: dict, batch_idx: int) -> StepOutput:
        x: Tensor = batch["views"][0]
        student_tokens, student_grid = self._forward_student(x)
        teacher_tokens, teacher_grid = self._forward_teacher(x)
        if student_grid != teacher_grid:
            # Different patch sizes: align the student's grid to the
            # teacher's before the pixel-wise MSE.
            student_tokens = (
                F.interpolate(
                    student_tokens.transpose(1, 2)
                    .unflatten(-1, student_grid)
                    .float(),
                    size=teacher_grid,
                    mode="bilinear",
                    align_corners=False,
                )
                .flatten(start_dim=2)
                .transpose(1, 2)
                .to(student_tokens.dtype)
            )
        loss = F.mse_loss(student_tokens, teacher_tokens)
        return StepOutput(
            loss=loss,
            log_dict={"train_loss/mse": float(loss.detach())},
        )

    # ── checkpoint: strip teacher weights ───────────────────────────────
    _TEACHER_PREFIX = "teacher_embedding_model."

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint.get("state_dict", {})
        checkpoint["state_dict"] = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(self._TEACHER_PREFIX)
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint.get("state_dict", {})
        teacher_state = self.teacher_embedding_model.state_dict()
        for key, value in teacher_state.items():
            full_key = f"{self._TEACHER_PREFIX}{key}"
            if full_key not in state_dict:
                state_dict[full_key] = value
