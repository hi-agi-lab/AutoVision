# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Distillation V3: queue-based global KL + intra-image patch KL.

Recipe (design doc §7.3; evolves the baseline ``Distillation`` recipe by
adding the local loss and a second projection head):

- Student: any backbone wrapper + two linear projection heads (global, local).
- Teacher: any model wrapper (default per framework: ``torchvision/resnet18``;
  the baseline defaults to ``dinov2/vitb14-noreg`` — packages are pluggable),
  frozen, eval mode, weights stripped from checkpoints.
- Single-view 224² transform with in-step mixup (lambda ~ U(0, 1)).
- Loss: KL divergence with the teacher distribution as target (τ=0.07),
  over queue-based global similarities + 1.0 × intra-image patch
  self-similarities (``KLDivLoss(input=student_log, target=teacher)``
  minimizes ``KL(teacher ‖ student)``).
- Teacher feature queue: per-rank FIFO, size bucketed by dataset size
  (16 → 8192), updated with the current batch *before* the loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

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
)
from autovision.augment.view_transform import ViewTransform
from autovision.optim import AdamWArgs, LARSArgs, OptimizerArgs
from autovision.pretext.method_args import PretextArgs
from autovision.pretext.recipe import PretextRecipe, StepOutput
from autovision.scaling import QUEUE_SIZE_BUCKETS, ScalingInfo, get_bucket_value

__all__ = ["DistillationV3", "DistillationV3Args"]


class DistillationV3Args(PretextArgs):
    """Arguments for the DistillationV3 method."""

    auto_steps: ClassVar[int | None] = None
    auto_epochs: ClassVar[int] = 100

    queue_size: int | Literal["auto"] = "auto"
    temperature_global: float = 0.07
    temperature_local: float = 0.07
    teacher: str | Any = "torchvision/resnet18"
    teacher_args: dict | None = None
    teacher_weights: str | Path | None = None
    teacher_pretrained: bool = True
    loss_local_weight: float = 1.0
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
        if self.queue_size == "auto":
            if scaling is None:
                raise ValueError(
                    "queue_size='auto' requires scaling (dataset size)."
                )
            self.queue_size = get_bucket_value(
                input=scaling.dataset_size, buckets=QUEUE_SIZE_BUCKETS
            )
        elif scaling is not None and self.queue_size >= scaling.dataset_size:
            raise ValueError(
                f"The specified queue size ({self.queue_size}) cannot be larger "
                f"than the dataset size ({scaling.dataset_size})."
            )


class DistillationV3AdamWArgs(AdamWArgs):
    """AdamW recipe: lr tuned for distillation, wd resolved per architecture."""

    lr: float = 0.0005
    weight_decay: float | Literal["auto"] = "auto"


class DistillationV3LARSArgs(LARSArgs):
    """LARS recipe: lr = 1.8 = 0.3 * 1536 / 256 at the reference batch size."""

    lr: float = 1.8
    momentum: float = 0.9
    weight_decay: float = 1e-6


class DistillationV3TransformArgs(MethodTransformArgs):
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


class DistillationV3Transform(MultiViewTransform):
    """Single-view transform: one 224² view per image."""

    def __init__(self, transform_args: DistillationV3TransformArgs) -> None:
        super().__init__(
            transforms=[ViewTransform(transform_args)], transform_args=transform_args
        )

    @staticmethod
    def transform_args_cls() -> type[DistillationV3TransformArgs]:
        return DistillationV3TransformArgs


def _l2_normalize(x: Tensor) -> Tensor:
    return F.normalize(x, p=2, dim=-1)


def build_teacher(
    method_args: DistillationV3Args, num_input_channels: int = 3
) -> Any:
    """Backwards-compatible alias of the shared helper in teachers."""
    from autovision.pretext.teachers import build_teacher as _shared

    return _shared(method_args, num_input_channels=num_input_channels)


class DistillationV3(PretextRecipe):
    """Queue-based distillation: global inter-image KL + local intra-image KL."""

    gradient_clip_val: float | None = 1.0

    def __init__(
        self,
        method_args: DistillationV3Args,
        optimizer_args: OptimizerArgs,
        embedding_model,
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
        # The base class stores the student as ``self.embedding_model``;
        # expose the distillation-specific name as a property (avoiding a
        # duplicate module registration under two names).

        # Teacher: registered as a submodule so it moves with .to(device);
        # its weights are stripped from checkpoints on save.
        self.teacher_embedding_model = build_teacher(
            method_args, num_input_channels=num_input_channels
        )
        self.teacher_embedding_dim: int = self.teacher_embedding_model.encoder_dim()

        self.student_projection_head_global = nn.Linear(
            embedding_model.embed_dim, self.teacher_embedding_dim
        )
        self.student_projection_head_local = nn.Linear(
            embedding_model.embed_dim, self.teacher_embedding_dim
        )
        init.trunc_normal_(self.student_projection_head_global.weight, std=0.02)
        init.trunc_normal_(self.student_projection_head_local.weight, std=0.02)

        # Per-rank queue: no cross-rank gather (by design, like the baseline).
        queue_size: int = method_args.queue_size
        self.register_buffer(
            "teacher_queue", torch.zeros(queue_size, self.teacher_embedding_dim)
        )
        self.kl_divergence = nn.KLDivLoss(reduction="batchmean", log_target=False)

    # ── contract ────────────────────────────────────────────────────────
    @property
    def student_embedding_model(self):
        """The student (stored by the base class as ``embedding_model``)."""
        return self.embedding_model

    def train(self, mode: bool = True) -> "DistillationV3":
        """Keep the frozen teacher in eval mode in every phase."""
        super().train(mode)
        self.teacher_embedding_model.eval()
        return self

    @staticmethod
    def method_args_cls() -> type[DistillationV3Args]:
        return DistillationV3Args

    @staticmethod
    def optimizer_args_cls(optim_type: str) -> type[OptimizerArgs]:
        """Distillation recipes: auto/AdamW -> tuned AdamW, LARS -> tuned LARS."""
        if optim_type in ("auto", "adamw"):
            return DistillationV3AdamWArgs
        if optim_type == "lars":
            return DistillationV3LARSArgs
        from autovision.optim import get_optimizer_args_cls

        return get_optimizer_args_cls(optim_type)

    @staticmethod
    def transform_cls() -> type[DistillationV3Transform]:
        return DistillationV3Transform

    def trainable_modules(self):
        from autovision.optim import OptimizationTargets

        return OptimizationTargets(
            targets=[
                self.student_embedding_model,
                self.student_projection_head_global,
                self.student_projection_head_local,
            ]
        )

    # ── teacher forward ─────────────────────────────────────────────────
    @torch.no_grad()
    def _forward_teacher(self, x: Tensor) -> tuple[Tensor, Tensor, tuple[int, int]]:
        features = self.teacher_embedding_model.feature_maps(x)["features"]
        pooled = self.teacher_embedding_model.pool_features({"features": features})
        teacher_global = pooled["pooled_features"].flatten(start_dim=1)  # (B, D)
        teacher_patches = features.flatten(start_dim=2).transpose(1, 2)  # (B, N, D)
        grid = (int(features.shape[2]), int(features.shape[3]))
        return _l2_normalize(teacher_global), _l2_normalize(teacher_patches), grid

    @torch.no_grad()
    def _update_queue(self, teacher_global: Tensor) -> None:
        """Shift the queue right by B and insert the current batch first.

        Called *before* the loss so targets include the current batch.
        """
        batch_size = teacher_global.size(0)
        queue_size = self.teacher_queue.size(0)
        if batch_size >= queue_size:
            self.teacher_queue = teacher_global[:queue_size].clone()
        else:
            self.teacher_queue[batch_size:] = self.teacher_queue[:-batch_size].clone()
            self.teacher_queue[:batch_size] = teacher_global

    # ── mixup (in-step, on the batch) ───────────────────────────────────
    @staticmethod
    def _mixup_data(x: Tensor) -> Tensor:
        lambda_ = torch.empty(1).uniform_(0.0, 1.0).item()
        index = torch.randperm(x.size(0), device=x.device)
        return lambda_ * x + (1.0 - lambda_) * x[index]

    # ── training step ───────────────────────────────────────────────────
    def training_step_impl(self, batch: dict, batch_idx: int) -> StepOutput:
        x: Tensor = batch["views"][0]
        x = self._mixup_data(x)

        # Single student forward pass: the pooled and patch representations
        # must come from the same backbone evaluation (BN running stats and
        # dropout are sampled once). Mirrors FeatureExtractor.forward's
        # semantics: the embed head applies to pooled features (pool=True)
        # and to raw features (pool=False).
        embedding_model = self.student_embedding_model
        student_features_dict = embedding_model._inner.feature_maps(x)
        pooled = embedding_model._inner.pool_features(student_features_dict)[
            "pooled_features"
        ]
        student_pooled = embedding_model._embed(pooled).flatten(
            start_dim=1
        )  # (B, embed_dim)
        student_features = embedding_model._embed(
            student_features_dict["features"]
        )
        student_patches = student_features.flatten(start_dim=2).transpose(
            1, 2
        )  # (B, N_student, embed_dim)

        teacher_global, teacher_patches, teacher_grid = self._forward_teacher(x)
        self._update_queue(teacher_global)

        # ── global loss: queue similarity ──
        student_global = _l2_normalize(
            self.student_projection_head_global(student_pooled)
        )
        student_queue_similarity = torch.einsum(
            "b d, c d -> b c", student_global, self.teacher_queue
        )
        teacher_queue_similarity = torch.einsum(
            "b d, c d -> b c", teacher_global, self.teacher_queue
        )
        teacher_distribution_inter_image = F.softmax(
            teacher_queue_similarity / self.method_args.temperature_global, dim=-1
        )
        student_log_distribution_inter_image = F.log_softmax(
            student_queue_similarity / self.method_args.temperature_global, dim=-1
        )
        global_loss = self.kl_divergence(
            student_log_distribution_inter_image, teacher_distribution_inter_image
        )

        # ── local loss: intra-image patch self-similarity ──
        if student_patches.shape[1] != teacher_patches.shape[1]:
            # Different feature-map grids: bilinearly align the student's
            # patch features to the teacher's grid.
            batch, num, dim = student_patches.shape
            height, width = student_features.shape[2], student_features.shape[3]
            student_patches = (
                F.interpolate(
                    student_patches.transpose(1, 2)
                    .unflatten(-1, (height, width))
                    .float(),
                    size=teacher_grid,
                    mode="bilinear",
                    align_corners=False,
                )
                .flatten(start_dim=2)
                .transpose(1, 2)
                .to(student_patches.dtype)
            )
        student_local = _l2_normalize(
            self.student_projection_head_local(student_patches)
        )
        teacher_self_similarity = torch.einsum(
            "b m d, b n d -> b m n", teacher_patches, teacher_patches
        ).flatten(start_dim=0, end_dim=1)
        student_self_similarity = torch.einsum(
            "b m d, b n d -> b m n", student_local, student_local
        ).flatten(start_dim=0, end_dim=1)
        teacher_distribution_intra_image = F.softmax(
            teacher_self_similarity / self.method_args.temperature_local, dim=-1
        )
        student_log_distribution_intra_image = F.log_softmax(
            student_self_similarity / self.method_args.temperature_local, dim=-1
        )
        local_loss = self.kl_divergence(
            student_log_distribution_intra_image, teacher_distribution_intra_image
        )

        loss = global_loss + self.method_args.loss_local_weight * local_loss
        return StepOutput(
            loss=loss,
            log_dict={
                "train_loss/global_loss": float(global_loss.detach()),
                "train_loss/local_loss": float(local_loss.detach()),
            },
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
        # Re-insert teacher weights from the live module so load_state_dict
        # stays strict about everything except the (expected) teacher keys.
        state_dict = checkpoint.get("state_dict", {})
        teacher_state = self.teacher_embedding_model.state_dict()
        for key, value in teacher_state.items():
            full_key = f"{self._TEACHER_PREFIX}{key}"
            if full_key not in state_dict:
                state_dict[full_key] = value
