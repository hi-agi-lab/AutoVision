# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DetCon: masked region-contrastive pretraining (S = SupCon / B = BYOL).

Recipe (design doc §4.2, blueprint in volume 12 §4.2; DetCon:
Henaff et al., arXiv 2103.10957; reference configuration values follow the
open-source DeepMind detcon setup, Apache-2.0):

- Region masks (H, W) of region ids are resized (NEAREST) to the feature
  map grid, then features are masked-pooled into one embedding per region.
- A fixed number of regions per image is subsampled (uniform over the
  regions present in the image) so the head tensors stay rectangular.
- **DetConS**: SimCLR-style MLP projection head over region embeddings +
  InfoNCE where the positives of an anchor are all regions with the same
  region id in the *other* image-view pairs (SupCon semantics; anchors
  inside the same source view are excluded). Per-rank negatives by design
  (matching the contrastive methods in this codebase).
- **DetConB**: BYOL-style projection + prediction heads with an EMA target
  branch; the loss aligns each region with the mean target embedding of
  the same region id in the *other view of the same image* (BYOL has no
  negatives, so cross-image pairing is not meaningful here). The loss is
  computed symmetrically (0→1 and 1→0) and averaged.

Implemented directly from the paper semantics (no third-party SSL library
is involved):

- The DetConB target branch here follows the student with a fixed BYOL
  momentum of 0.99 (a frozen target copy would never track the student,
  which the paper's EMA formulation requires).
- Region-pairing semantics are documented above.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode

from autovision.optim import OptimizationTargets, SGDArgs
from autovision.pretext.detcon.detcon_transform import (
    DetConBTransform,
    DetConSTransform,
)
from autovision.pretext.dino.dino import _update_momentum
from autovision.pretext.method_args import PretextArgs
from autovision.pretext.recipe import PretextRecipe, StepOutput
from autovision.pretext.simclr.simclr import _SimCLRProjectionHead

logger = logging.getLogger("autovision")

__all__ = [
    "DetConS",
    "DetConSArgs",
    "DetConSGDArgs",
    "DetConSLoss",
    "DetConB",
    "DetConBArgs",
    "DetConBSGDArgs",
    "DetConBLoss",
    "pool_masked",
]

# BYOL momentum for the DetConB target branch (BYOL paper initial value).
_MOMENTUM = 0.99


class DetConSArgs(PretextArgs):
    """Arguments for the DetConS method."""

    hidden_dim: int = 2048
    output_dim: int = 128
    num_layers: int = 2
    batch_norm: bool = True
    temperature: float = 0.1
    num_sampled_cls: int = 16
    num_total_cls: int = 256


class DetConBArgs(PretextArgs):
    """Arguments for the DetConB method."""

    proj_hidden_dim: int = 2048
    proj_output_dim: int = 256
    pred_hidden_dim: int = 2048
    pred_output_dim: int = 256
    temperature: float = 0.1
    num_sampled_cls: int = 16
    num_total_cls: int = 256


class DetConSGDArgs(SGDArgs):
    """SGD recipe for DetConS (baseline ImageNet bs=4096 values)."""

    lr: float = 0.3
    weight_decay: float = 0.0001


class DetConBSGDArgs(SGDArgs):
    """SGD recipe for DetConB (baseline ImageNet bs=4096 values)."""

    lr: float = 0.3
    weight_decay: float = 0.0001


# ── masked pooling and region subsampling ───────────────────────────────


def pool_masked(source: Tensor, mask: Tensor, num_cls: int) -> Tensor:
    """Average-pool features inside each region of a mask.

    Args:
        source: float tensor (N, C, h, w).
        mask: integer tensor (N, h, w) with region ids in ``[0, num_cls)``.
        num_cls: total number of region ids.

    Returns:
        Float tensor (N, C, num_cls); empty regions pool to zero.
    """
    n, c, h, w = source.shape
    bin_masks = F.one_hot(mask, num_classes=num_cls)  # (N, h, w, num_cls)
    bin_masks = bin_masks.permute(0, 3, 1, 2).to(source.dtype)  # (N, num_cls, h, w)
    counts = bin_masks.sum(dim=(2, 3))  # (N, num_cls)
    pooled = (source.unsqueeze(1) * bin_masks.unsqueeze(2)).sum(dim=(3, 4))
    # (N, num_cls, C) -> (N, C, num_cls), then average by region pixel count.
    pooled = pooled.permute(0, 2, 1)
    return pooled / counts.clamp(min=1).unsqueeze(1)


def _subsample_mask_indices(
    masks: Tensor,
    n_total_cls: int,
    n_sampled_cls: int,
    class_weighted_sampling: bool = False,
) -> Tensor:
    """Sample a fixed number of region ids per image, with repetition.

    Args:
        masks: integer tensor (B, H, W) with region ids.
        n_total_cls: total number of region ids in the dataset.
        n_sampled_cls: number of ids to sample per image.
        class_weighted_sampling: sample ids proportional to their pixel
            count instead of uniformly over the ids present.

    Returns:
        Integer tensor (B, n_sampled_cls) of sampled region ids.
    """
    b, h, w = masks.shape
    flat = masks.view(b, h * w)
    bin_masks = (
        F.one_hot(flat, num_classes=n_total_cls).permute(0, 2, 1).to(torch.bool)
    )  # (B, n_total_cls, h*w)
    if class_weighted_sampling:
        unnormed_prob = bin_masks.sum(dim=-1).to(torch.float32)  # (B, n_total_cls)
    else:
        # Uniform over the ids actually present in the image (+eps keeps the
        # Categorical valid for images without any id).
        unnormed_prob = (
            torch.greater(bin_masks.sum(dim=-1), 1e-3).to(torch.float32) + 1e-10
        )
    indices: Tensor = torch.distributions.Categorical(
        probs=unnormed_prob
    ).sample([n_sampled_cls])  # (n_sampled_cls, B)
    return indices.permute(1, 0)  # (B, n_sampled_cls)


def _subsample_pooled_features(x: Tensor, indices: Tensor) -> Tensor:
    """Gather the pooled features of the sampled region ids.

    Args:
        x: float tensor (B, n_total_cls, C) of per-region pooled features.
        indices: integer tensor (B, n_sampled_cls).

    Returns:
        Float tensor (B, n_sampled_cls, C). Use per view (not on
        concatenated double batches) — indices are per image.
    """
    return torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


# ── losses ──────────────────────────────────────────────────────────────


class DetConSLoss(nn.Module):
    """SupCon-style InfoNCE over region embeddings.

    All regions of both views form the anchor/label pool: anchors attract
    same-id regions from other source views and repel different-id regions
    (per-rank negatives). Anchors without positives are skipped. With
    multiple positives the loss is ``-log( sum_pos exp(s) / sum_all exp(s) )``.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        view0: Tensor,
        view1: Tensor,
        mask_view0: Tensor,
        mask_view1: Tensor,
    ) -> Tensor:
        b, p, d = view0.shape
        embeddings = F.normalize(torch.cat([view0, view1], dim=0), dim=-1)
        labels = torch.cat([mask_view0, mask_view1], dim=0)  # (2B, P)
        flat = embeddings.reshape(2 * b * p, d)
        flat_labels = labels.reshape(-1)

        logits = flat @ flat.T / self.temperature  # (2BP, 2BP)
        same_label = flat_labels.unsqueeze(0) == flat_labels.unsqueeze(1)
        source = torch.arange(2 * b, device=logits.device).repeat_interleave(p)
        same_source = source.unsqueeze(0) == source.unsqueeze(1)
        pos_mask = same_label & ~same_source
        neg_mask = ~same_label
        all_mask = pos_mask | neg_mask  # excludes the self-similarity diagonal

        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            # No positive pairs in the batch (degenerate masks): zero loss
            # that still carries the graph.
            return logits.sum() * 0.0

        pos_sum = logits.masked_fill(~pos_mask, float("-inf")).logsumexp(dim=1)
        denom = logits.masked_fill(~all_mask, float("-inf")).logsumexp(dim=1)
        loss = -(pos_sum - denom)
        return loss[has_pos].mean()


class DetConBLoss(nn.Module):
    """BYOL-style loss over region embeddings (no negatives).

    For each region of a view, the target is the mean EMA-branch embedding
    of the same region id in the other view of the same image; regions
    whose id does not appear in the other view are skipped. The loss is
    ``1 - cos(pred, sg(target))``, averaged over valid regions and both
    directions. ``temperature`` is accepted for parity with the baseline
    signature but unused (BYOL has no temperature).
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()

    def forward(
        self,
        pred_view0: Tensor,
        pred_view1: Tensor,
        target_view0: Tensor,
        target_view1: Tensor,
        mask_view0: Tensor,
        mask_view1: Tensor,
    ) -> Tensor:
        loss01 = self._half(pred_view0, mask_view0, target_view1, mask_view1)
        loss10 = self._half(pred_view1, mask_view1, target_view0, mask_view0)
        return (loss01 + loss10) / 2

    def _half(
        self, pred: Tensor, pred_mask: Tensor, target: Tensor, target_mask: Tensor
    ) -> Tensor:
        """pred (B,P,D) vs per-image same-id mean of target (B,P,D_tgt)."""
        # The id space must cover both sides: the prediction view may hold
        # ids that never appear in the target view (they are skipped, but
        # the one-hot/gather tables must not index out of bounds).
        num_cls = (
            int(torch.maximum(pred_mask.max(), target_mask.max()).item()) + 1
        )
        onehot = F.one_hot(target_mask, num_classes=num_cls).to(target.dtype)
        counts = onehot.sum(dim=1)  # (B, num_cls)
        sums = torch.einsum("bpc,bpd->bcd", onehot, target)  # (B, num_cls, D)
        present = counts > 0
        means = sums / counts.clamp(min=1).unsqueeze(-1)

        pred = F.normalize(pred, dim=-1)
        means = F.normalize(means, dim=-1)
        matched = present.gather(1, pred_mask)  # (B, P) bool
        target_per_region = means.gather(
            1, pred_mask.unsqueeze(-1).expand(-1, -1, means.shape[-1])
        )  # (B, P, D)
        cos = (pred * target_per_region).sum(dim=-1)  # (B, P)
        if not matched.any():
            return cos.sum() * 0.0
        return (1.0 - cos)[matched].mean()


# ── methods ─────────────────────────────────────────────────────────────


class _BYOLProjectionHead(nn.Module):
    """BYOL projection head: Linear-BN-ReLU-Linear."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _BYOLPredictionHead(_BYOLProjectionHead):
    """BYOL prediction head (same structure as the projection head)."""


def _resize_masks(masks_list: list[Tensor], l_h: int, l_w: int) -> Tensor:
    """Concatenate per-view masks and resize to the feature map grid."""
    return TF.resize(
        torch.cat(masks_list, dim=0),
        [l_h, l_w],
        interpolation=InterpolationMode.NEAREST,
    ).to(torch.int64)  # (2B, l_h, l_w)


def _region_pipeline(
    features: Tensor,
    masks: Tensor,
    n_total_cls: int,
    n_sampled_cls: int,
    indices: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Pool features per region and subsample to a fixed region count.

    Args:
        features: (2B, C, l_h, l_w) feature maps of both views.
        masks: (2B, l_h, l_w) resized region-id masks.
        indices: optional pre-sampled region ids (2B, n_sampled_cls) to
            reuse — the DetConB target branch must gather the SAME regions
            as the prediction branch, otherwise the pooled target columns
            would be misaligned with the region ids used for pairing.

    Returns:
        (pooled, indices) with pooled (2B, n_sampled_cls, C) and
        indices (2B, n_sampled_cls).
    """
    pooled = pool_masked(features, masks, num_cls=n_total_cls)  # (2B, C, num_cls)
    pooled = pooled.permute(0, 2, 1)  # (2B, num_cls, C)
    if indices is None:
        indices = _subsample_mask_indices(
            masks=masks, n_total_cls=n_total_cls, n_sampled_cls=n_sampled_cls
        )  # (2B, n_sampled_cls)
    indices0, indices1 = indices.chunk(2, dim=0)
    pooled0, pooled1 = pooled.chunk(2, dim=0)
    pooled0 = _subsample_pooled_features(pooled0, indices0)
    pooled1 = _subsample_pooled_features(pooled1, indices1)
    indices = torch.cat([indices0, indices1], dim=0)
    pooled = torch.cat([pooled0, pooled1], dim=0)  # (2B, n_sampled_cls, C)
    return pooled, indices


class DetConS(PretextRecipe):
    """DetConS pretraining method (SupCon over masked region embeddings)."""

    def __init__(
        self,
        method_args: DetConSArgs,
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
        self.criterion = DetConSLoss(temperature=method_args.temperature)

    @staticmethod
    def method_args_cls() -> type[DetConSArgs]:
        return DetConSArgs

    @staticmethod
    def optimizer_args_cls(optim_type: str) -> type:
        if optim_type in ("auto", "sgd"):
            return DetConSGDArgs
        from autovision.optim import get_optimizer_args_cls

        return get_optimizer_args_cls(optim_type)

    @staticmethod
    def transform_cls() -> type[DetConSTransform]:
        return DetConSTransform

    def trainable_modules(self) -> OptimizationTargets:
        return OptimizationTargets(
            targets=[self.embedding_model, self.projection_head]
        )

    def training_step_impl(self, batch: dict, batch_idx: int) -> StepOutput:
        args = self.method_args
        views: list[Tensor] = batch["views"]
        masks_list: list[Tensor] = batch["masks"]
        features = self.embedding_model(
            torch.cat(views, dim=0), pool=False
        )  # (2B, C, l_h, l_w)
        _, c, l_h, l_w = features.shape

        masks = _resize_masks(masks_list, l_h, l_w)
        pooled, indices = _region_pipeline(
            features, masks, args.num_total_cls, args.num_sampled_cls
        )  # (2B, P, C), (2B, P)

        # BatchNorm1d in the head needs the region dim folded into batch.
        b = features.shape[0] // 2
        n = args.num_sampled_cls
        projected = self.projection_head(pooled.view(2 * b * n, c)).view(
            2 * b, n, args.output_dim
        )
        view0, view1 = projected.chunk(2, dim=0)
        mask0, mask1 = indices.chunk(2, dim=0)
        loss = self.criterion(view0, view1, mask0, mask1)
        return StepOutput(loss=loss)


class DetConB(PretextRecipe):
    """DetConB pretraining method (BYOL over masked region embeddings).

    The EMA target branch (``*_momentum`` modules) follows the student with
    a fixed momentum of 0.99 — the baseline builds these modules but never
    updates them, which freezes the targets at random init.
    """

    _MOMENTUM_PREFIXES = (
        "embedding_model_momentum.",
        "projection_head_momentum.",
    )

    def __init__(
        self,
        method_args: DetConBArgs,
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
        self.projection_head = _BYOLProjectionHead(
            input_dim=embedding_model.embed_dim,
            hidden_dim=method_args.proj_hidden_dim,
            output_dim=method_args.proj_output_dim,
        )
        self.prediction_head = _BYOLPredictionHead(
            input_dim=method_args.proj_output_dim,
            hidden_dim=method_args.pred_hidden_dim,
            output_dim=method_args.pred_output_dim,
        )
        # EMA target branch (deepcopies at init, updated per step below).
        self.embedding_model_momentum = copy.deepcopy(embedding_model)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        for module in (self.embedding_model_momentum, self.projection_head_momentum):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.criterion = DetConBLoss(temperature=method_args.temperature)

    @staticmethod
    def method_args_cls() -> type[DetConBArgs]:
        return DetConBArgs

    @staticmethod
    def optimizer_args_cls(optim_type: str) -> type:
        if optim_type in ("auto", "sgd"):
            return DetConBSGDArgs
        from autovision.optim import get_optimizer_args_cls

        return get_optimizer_args_cls(optim_type)

    @staticmethod
    def transform_cls() -> type[DetConBTransform]:
        return DetConBTransform

    def trainable_modules(self) -> OptimizationTargets:
        return OptimizationTargets(
            targets=[
                self.embedding_model,
                self.projection_head,
                self.prediction_head,
            ]
        )

    def train(self, mode: bool = True) -> "DetConB":
        # Keep the EMA branch in eval mode even when Lightning flips the
        # whole method to train (same pattern as DINO).
        super().train(mode)
        self.embedding_model_momentum.eval()
        self.projection_head_momentum.eval()
        return self

    def training_step_impl(self, batch: dict, batch_idx: int) -> StepOutput:
        args = self.method_args
        views: list[Tensor] = batch["views"]
        masks_list: list[Tensor] = batch["masks"]
        prediction = self.embedding_model(
            torch.cat(views, dim=0), pool=False
        )  # (2B, C, l_h, l_w)
        with torch.no_grad():
            target = self.embedding_model_momentum(
                torch.cat(views, dim=0), pool=False
            )  # (2B, C, l_h, l_w)
        _, c, l_h, l_w = prediction.shape

        masks = _resize_masks(masks_list, l_h, l_w)
        prediction_pooled, indices = _region_pipeline(
            prediction, masks, args.num_total_cls, args.num_sampled_cls
        )
        # Reuse the SAME region ids for the target branch so target columns
        # stay aligned with the ids used for pairing.
        target_pooled, _ = _region_pipeline(
            target, masks, args.num_total_cls, args.num_sampled_cls, indices=indices
        )

        b = prediction.shape[0] // 2
        n = args.num_sampled_cls
        projected = self.projection_head(
            prediction_pooled.view(2 * b * n, c)
        )
        predicted = self.prediction_head(projected).view(
            2 * b, n, args.pred_output_dim
        )
        with torch.no_grad():
            target_projected = self.projection_head_momentum(
                target_pooled.view(2 * b * n, c)
            ).view(2 * b, n, args.proj_output_dim)

        pred0, pred1 = predicted.chunk(2, dim=0)
        target0, target1 = target_projected.chunk(2, dim=0)
        mask0, mask1 = indices.chunk(2, dim=0)
        loss = self.criterion(pred0, pred1, target0, target1, mask0, mask1)

        # EMA update AFTER the loss graph is built (targets are detached).
        _update_momentum(self.embedding_model, self.embedding_model_momentum, _MOMENTUM)
        _update_momentum(self.projection_head, self.projection_head_momentum, _MOMENTUM)
        return StepOutput(loss=loss)

    # ── checkpoint: strip target-branch weights ─────────────────────────
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            return
        for key in [
            key
            for key in state_dict
            if key.startswith(self._MOMENTUM_PREFIXES)
        ]:
            del state_dict[key]

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint.get("state_dict", {})
        live_modules = {
            self._MOMENTUM_PREFIXES[0]: self.embedding_model_momentum,
            self._MOMENTUM_PREFIXES[1]: self.projection_head_momentum,
        }
        for prefix, module in live_modules.items():
            if not any(key.startswith(prefix) for key in state_dict):
                logger.warning(
                    "Checkpoint has no '%s*' weights; the EMA target branch "
                    "was rebuilt from the current student and needs time to "
                    "re-converge.", prefix,
                )
            for key, value in module.state_dict().items():
                state_dict.setdefault(f"{prefix}{key}", value)
