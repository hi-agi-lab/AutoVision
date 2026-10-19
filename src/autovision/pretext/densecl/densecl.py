# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DenseCL: single-image contrastive learning with dense correspondence.

Recipe (design doc §7.6, blueprint in volume 12 §4.2):

- Two-view transform shared with SimCLR (identical augmentation recipe).
- Backbone pass produces the feature map; a global projection head pools
  it while a 1×1-conv dense head projects every pixel.
- MoCo-style memory bank of global keys (FIFO, normalized). The dense
  loss contrasts each query pixel against the pixel at the same spatial
  position in the other view (the dense positive) plus the global queue.
- Loss: global InfoNCE + ``dense_loss_weight`` × dense InfoNCE (τ=0.2).
- Queue size is bucketed by dataset size from 16 up to 65536 (this table
  intentionally exceeds the shared ``QUEUE_SIZE_BUCKETS`` cap of 8192).

Design notes: single-encoder keys with stop-gradient (no momentum
encoder, no shuffle-BN — consistent with the queue design of the other
methods in this codebase); per-rank queue without cross-rank gather.
The dense logits tensor is (B, h, w, 1 + queue_size) — on memory-bound
GPUs pick a smaller queue (or dataset bucket) explicitly.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from autovision.optim import OptimizationTargets, SGDArgs
from autovision.pretext.method_args import PretextArgs
from autovision.pretext.recipe import PretextRecipe, StepOutput
from autovision.pretext.simclr.simclr import (
    SimCLRTransform,
    _SimCLRProjectionHead,
)
from autovision.scaling import ScalingInfo, get_bucket_value

__all__ = ["DenseCL", "DenseCLArgs"]

# (dataset_size <, queue size) — DenseCL keeps large queues on large sets
# (the shared QUEUE_SIZE_BUCKETS cap of 8192 is too small here).
_QUEUE_SIZE_BUCKETS: tuple[tuple[float, int], ...] = (
    (1_000, 16),
    (5_000, 64),
    (20_000, 256),
    (100_000, 1_024),
    (500_000, 4_096),
    (2_000_000, 16_384),
    (float("inf"), 65_536),
)


class DenseCLArgs(PretextArgs):
    """Arguments for the DenseCL method."""

    hidden_dim: int = 2048
    output_dim: int = 128
    num_layers: int = 2
    batch_norm: bool = True
    temperature: float = 0.2
    dense_loss_weight: float = 1.0
    queue_size: int | Literal["auto"] = "auto"

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
        if self.queue_size == "auto":
            if scaling is None:
                raise ValueError("queue_size='auto' requires the dataset size.")
            self.queue_size = get_bucket_value(
                input=scaling.dataset_size, buckets=_QUEUE_SIZE_BUCKETS
            )
        if self.queue_size < 1:
            raise ValueError(f"queue_size must be >= 1, got {self.queue_size}.")


class DenseCLSGDArgs(SGDArgs):
    """SGD recipe (MoCo-style): lr 0.03 at the reference batch size."""

    lr: float = 0.03


class _DenseProjectionHead(nn.Module):
    """1×1-conv MLP over the feature map (per-pixel projection)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, bias=False)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(hidden_dim, output_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _info_nce_with_queue(
    query: Tensor, positive: Tensor, queue: Tensor, temperature: float
) -> Tensor:
    """InfoNCE against one positive plus the queue negatives.

    ``query``/``positive``: (N, C) unnormalized; ``queue``: (C, K)
    normalized keys. Returns the mean CE with the positive as target 0.
    """
    query = F.normalize(query, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    pos_logits = (query * positive).sum(dim=-1, keepdim=True) / temperature
    neg_logits = query @ queue / temperature  # (N, K)
    logits = torch.cat([pos_logits, neg_logits], dim=-1)
    targets = logits.new_zeros(logits.shape[0], dtype=torch.long)
    return F.cross_entropy(logits, targets)


class DenseCL(PretextRecipe):
    """DenseCL pretraining method (dense pixel-wise contrast)."""

    def __init__(
        self,
        method_args: DenseCLArgs,
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
        self.global_projection_head = _SimCLRProjectionHead(
            input_dim=embedding_model.embed_dim,
            hidden_dim=method_args.hidden_dim,
            output_dim=method_args.output_dim,
            num_layers=method_args.num_layers,
            batch_norm=method_args.batch_norm,
        )
        self.dense_projection_head = _DenseProjectionHead(
            input_dim=embedding_model.embed_dim,
            hidden_dim=method_args.hidden_dim,
            output_dim=method_args.output_dim,
            batch_norm=method_args.batch_norm,
        )
        queue = F.normalize(
            torch.randn(method_args.output_dim, method_args.queue_size), dim=0
        )
        self.register_buffer("queue", queue)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # ── contract ────────────────────────────────────────────────────────
    @staticmethod
    def method_args_cls() -> type[DenseCLArgs]:
        return DenseCLArgs

    @staticmethod
    def optimizer_args_cls(optim_type: str) -> type:
        if optim_type in ("auto", "sgd"):
            return DenseCLSGDArgs
        from autovision.optim import get_optimizer_args_cls

        return get_optimizer_args_cls(optim_type)

    @staticmethod
    def transform_cls() -> type[SimCLRTransform]:
        # The augmentation recipe is identical to SimCLR's.
        return SimCLRTransform

    def trainable_modules(self) -> OptimizationTargets:
        return OptimizationTargets(
            targets=[
                self.embedding_model,
                self.global_projection_head,
                self.dense_projection_head,
            ]
        )

    # ── forward ─────────────────────────────────────────────────────────
    def _forward_heads(self, maps: Tensor) -> tuple[Tensor, Tensor]:
        """Project the feature map globally and per pixel."""
        global_feats = self.global_projection_head(
            F.adaptive_avg_pool2d(maps, 1).flatten(1)
        )
        dense_feats = self.dense_projection_head(maps)
        return global_feats, dense_feats

    @torch.no_grad()
    def _enqueue(self, keys: Tensor) -> None:
        """FIFO push of detached global keys (2B, C) into the queue."""
        batch_size = keys.shape[0]
        queue_size = self.queue.shape[1]
        keys = F.normalize(keys.detach(), p=2, dim=-1)
        ptr = int(self.queue_ptr.item())
        if ptr + batch_size <= queue_size:
            self.queue[:, ptr : ptr + batch_size] = keys.T
        else:  # wrap around
            first = queue_size - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, : batch_size - first] = keys[first:].T
        self.queue_ptr[0] = (ptr + batch_size) % queue_size

    def training_step_impl(self, batch: dict, batch_idx: int) -> StepOutput:
        args = self.method_args
        views = batch["views"]
        maps = self.embedding_model(torch.cat(views), pool=False)  # (2B, C, h, w)
        global_feats, dense_feats = self._forward_heads(maps)
        global0, global1 = global_feats.chunk(2)
        dense0, dense1 = dense_feats.chunk(2)

        # Snapshot the queue for the loss graph: _enqueue writes the buffer
        # in place below, which would otherwise invalidate the backward of
        # the matmul against the live buffer.
        queue = self.queue.clone()

        # Global contrast: each view queries the other (stop-grad key).
        loss_global = 0.5 * (
            _info_nce_with_queue(global0, global1.detach(), queue, args.temperature)
            + _info_nce_with_queue(global1, global0.detach(), queue, args.temperature)
        )

        # Dense contrast: pixel (i, j) in view 0 matches pixel (i, j) in
        # view 1 (same-image two-crop correspondence).
        q0 = dense0.flatten(2).transpose(1, 2).reshape(-1, dense0.shape[1])  # (B*h*w, C)
        q1 = dense1.flatten(2).transpose(1, 2).reshape(-1, dense1.shape[1])
        loss_dense = 0.5 * (
            _info_nce_with_queue(q0, q1.detach(), queue, args.temperature)
            + _info_nce_with_queue(q1, q0.detach(), queue, args.temperature)
        )

        loss = loss_global + args.dense_loss_weight * loss_dense
        self._enqueue(torch.cat([global1, global0], dim=0))
        return StepOutput(
            loss=loss,
            log_dict={
                "train_loss/global": float(loss_global.detach()),
                "train_loss/dense": float(loss_dense.detach()),
            },
        )
