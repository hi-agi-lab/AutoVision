# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""SimCLR: NT-Xent over two augmented views with an MLP projection head.

Recipe (design doc §7.4, mirroring the baseline):

- Two-view 224² transform (jitter strength 1.0, blur p=0.5).
- 2-layer linear projection head (hidden 2048 → output 128) with BatchNorm.
- NT-Xent loss (τ=0.1); the similarity matrix stays per-rank (no cross-rank
  gather — DDP users get per-rank negatives, matching the queue design of
  the distillation methods).
- SGD, lr 0.3 scaled linearly with the global batch size.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

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
from autovision.optim import OptimizationTargets, SGDArgs
from autovision.pretext.method_args import PretextArgs
from autovision.pretext.recipe import PretextRecipe, StepOutput

__all__ = ["SimCLR", "SimCLRArgs", "NTXentLoss"]


class SimCLRArgs(PretextArgs):
    """Arguments for the SimCLR method."""

    hidden_dim: int = 2048
    output_dim: int = 128
    num_layers: int = 2
    batch_norm: bool = True
    temperature: float = 0.1


class SimCLRSGDArgs(SGDArgs):
    """SGD recipe: lr 0.3 at the reference batch size (linear scaling)."""

    lr: float = 0.3


class SimCLRTransformArgs(MethodTransformArgs):
    """Two-view SimCLR recipe (Chen et al. 2020 defaults)."""

    image_size: tuple[int, int] = (224, 224)
    random_resize: RandomResizeArgs = RandomResizeArgs()
    random_flip: RandomFlipArgs | None = RandomFlipArgs()
    color_jitter: ColorJitterArgs | None = ColorJitterArgs(
        prob=0.8, strength=1.0, brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
    )
    random_gray_scale: float | None = 0.2
    gaussian_blur: GaussianBlurArgs | None = GaussianBlurArgs(
        prob=0.5, sigmas=(0.1, 2.0)
    )
    solarize: None = None
    normalize: NormalizeArgs = NormalizeArgs()


class SimCLRTransform(MultiViewTransform):
    """Two independent views of the SimCLR augmentation pipeline."""

    def __init__(self, transform_args: SimCLRTransformArgs) -> None:
        view = ViewTransform(transform_args)
        super().__init__(
            transforms=[view, view], transform_args=transform_args
        )

    @staticmethod
    def transform_args_cls() -> type[SimCLRTransformArgs]:
        return SimCLRTransformArgs


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross-entropy loss (SimCLR).

    With ``z0``/``z1`` of shape (B, D) the two positives are matched across
    the concatenated (2B, D) embedding matrix; all other samples are
    negatives. Per-rank by design (no cross-rank gather).
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z0: Tensor, z1: Tensor) -> Tensor:
        batch_size = z0.size(0)
        z = F.normalize(torch.cat([z0, z1], dim=0), p=2, dim=-1)
        similarity = torch.einsum("b d, c d -> b c", z, z) / self.temperature
        similarity.fill_diagonal_(float("-inf"))
        # Positive of sample i < B is sample i + B, and vice versa.
        targets = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size),
                torch.arange(0, batch_size),
            ]
        ).to(z.device)
        return F.cross_entropy(similarity, targets)


class SimCLR(PretextRecipe):
    """SimCLR pretraining method."""

    def __init__(
        self,
        method_args: SimCLRArgs,
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
        self.projection_head = _SimCLRProjectionHead(
            input_dim=embedding_model.embed_dim,
            hidden_dim=method_args.hidden_dim,
            output_dim=method_args.output_dim,
            num_layers=method_args.num_layers,
            batch_norm=method_args.batch_norm,
        )
        self.criterion = NTXentLoss(temperature=method_args.temperature)

    @staticmethod
    def method_args_cls() -> type[SimCLRArgs]:
        return SimCLRArgs

    @staticmethod
    def optimizer_args_cls(optim_type: str) -> type:
        if optim_type in ("auto", "sgd"):
            return SimCLRSGDArgs
        from autovision.optim import get_optimizer_args_cls

        return get_optimizer_args_cls(optim_type)

    @staticmethod
    def transform_cls() -> type[SimCLRTransform]:
        return SimCLRTransform

    def trainable_modules(self) -> OptimizationTargets:
        return OptimizationTargets(
            targets=[self.embedding_model, self.projection_head]
        )

    def training_step_impl(self, batch: dict, batch_idx: int) -> StepOutput:
        views = batch["views"]
        # Shared forward: all views pass through the backbone in one batch.
        embeddings = self.embedding_model(torch.cat(views), pool=True).flatten(1)
        projections = self.projection_head(embeddings)
        chunks = projections.chunk(len(views))
        loss = self.criterion(chunks[0], chunks[1])
        return StepOutput(loss=loss)


class _SimCLRProjectionHead(nn.Module):
    """MLP projection head: [Linear + BN + ReLU] × (num_layers - 1) + Linear.

    ``batch_norm=True`` requires a per-device batch of at least 2 samples.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
