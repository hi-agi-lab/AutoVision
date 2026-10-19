# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Per-object SimCSE-style contrastive components (design doc 14).

Port of the RF-DETR-SSL training-side building blocks: the train-only
projection head, the unsupervised InfoNCE and supervised multi-positive
losses, the distributed negative-pool gather, the IoU-aligned positive
logit bias, and the cross-view instance correspondence. All similarity
computations follow SimCSE's ``cos(z1, z2) / temp`` convention and
normalise in float32 for AMP stability (bf16 cosine matrices have
numerically unstable gradients).

The objective is anchored to the SAME Hungarian match the box loss
consumes (zero extra matching): one contrastive instance is one matched
``(query, GT box, label)`` triple from the final decoder layer. Cross-view
positives (``two_crop``) are recovered per image by a box-IoU assignment
validated by label agreement; failing pairs are dropped, never kept as
noisy positives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from scipy.optimize import (  # type: ignore[import-untyped,unused-ignore]
    linear_sum_assignment as linear_sum_assignment,
)
from torch import Tensor, nn
from torchvision.ops import box_iou

from autovision.tasks.detection.shared.denoising import (
    box_cxcywh_to_xyxy,
)

__all__ = [
    "CrossViewMatch",
    "PaddedGather",
    "SSLParams",
    "SSLProjectionHead",
    "cosine_logits",
    "epps_pulley_sigreg",
    "gather_distributed",
    "gather_distributed_padded",
    "match_cross_view",
    "positive_logit_bias",
    "simcse_infonce_loss",
    "ssl_contrastive_loss",
    "supervised_contrastive_loss",
]

#: Floor for IoU values inside ``log`` so that a zero-overlap positive
#: cannot emit ``-inf`` (design doc 14, A1 semantics).
_EPS = 1e-12

_PROJECTION_HEAD_KINDS = ("simcse", "mlp2")


@dataclass(frozen=True)
class SSLParams:
    """Immutable SSL contract shared by the model head and the criterion.

    Built from the task-family config (e.g. ``DETRSSLArgs``) so this
    components layer stays free of task-model imports. ``loss_coef`` rides
    the criterion's ``weight_dict`` (``"loss_ssl"``), applied once — the
    objective is computed at the final decoder layer only.
    """

    mode: str = "unsup"  # "unsup" | "sup" | "both"
    loss_coef: float = 0.1
    temp: float = 0.05
    view: str = "two_crop"  # "two_crop" | "dropout"
    iou_match: float = 0.55
    iou_bias: float = 0.5
    iou_gamma: float = 1.0
    same_image_positive_only: bool = False
    gather_distributed: bool = True
    every_n_steps: int = 1
    #: Try the horizontally-mirrored alignment when matching cross-view
    #: instances: the two views flip independently (p=0.5 each), so an
    #: off-center object lands on opposite sides in ~50% of pairs and its
    #: raw normalized-box IoU is ~0 even though it is the same object.
    flip_invariant_pairing: bool = True
    #: Feed matched instances whose cross-view counterpart failed validation
    #: back into the pool as extra candidate columns (negatives under
    #: ``unsup``, soft positives via label equality under ``sup``) instead
    #: of dropping them — reclaims hard instances for the contrastive pool
    #: (design doc 14, D14 / R9).
    use_unpaired_as_negatives: bool = True
    #: "matched_gt" builds the pool from the final Hungarian match (auxiliary
    #: objective alongside the detection losses). "encoder_candidates" builds
    #: it from the decoder's top-k candidate features — label-free, unsup +
    #: dropout view only, and it IS the whole loss (SSL-only pretraining,
    #: design doc 14 §4.10).
    instance_source: str = "matched_gt"
    #: SIGReg distribution regularizer (design doc 16, LeVJEPA R1-A1): a
    #: third loss leg applied to the RAW (pre-normalization) projector
    #: outputs of the matched pool. 0 disables it entirely. Its weight is
    #: ``sigreg_coef`` (the criterion scales ``loss_sigreg`` by it) — the
    #: reference finding is that SIGReg mainly buys λ_ssl robustness rather
    #: than raw mAP.
    sigreg_coef: float = 0.0
    sigreg_slices: int = 1024
    sigreg_knots: int = 17


class SSLProjectionHead(nn.Module):
    """Train-only projection head mapping query features into the contrastive space.

    The ``"simcse"`` kind faithfully mirrors SimCSE's ``MLPLayer``
    (``Linear + Tanh``); the ``"mlp2"`` kind follows the two-layer MLP
    ablated in the SimCSE paper. The separate ``input_dropout`` supplies
    the second view's noise in the ``dropout`` view mode (two stochastic
    draws through the same head). The head is applied only on the training
    path — inference, export tracing and the decode contract never touch
    it — and its outputs are L2-normalised in float32 regardless of the
    autocast dtype.
    """

    def __init__(
        self,
        hidden_dim: int,
        projection_dim: int = 256,
        kind: str = "simcse",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kind not in _PROJECTION_HEAD_KINDS:
            raise ValueError(
                f"Unknown projection head kind {kind!r}; expected one of {_PROJECTION_HEAD_KINDS}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {dropout}.")
        projector: nn.Module
        if kind == "simcse":
            projector = nn.Sequential(nn.Linear(hidden_dim, projection_dim), nn.Tanh())
        else:
            projector = nn.Sequential(
                nn.Linear(hidden_dim, projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim, projection_dim),
            )
        self.projector = projector
        self.input_dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor) -> Tensor:
        """Project query features; returns float32 unit-norm embeddings."""
        z = self.projector(self.input_dropout(features))
        return F.normalize(z.float(), dim=-1)

    def forward_with_raw(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Project query features, returning both the normalized and raw embeddings.

        The SIGReg regularizer operates on the pre-normalization output (a
        unit-sphere sample can never look Gaussian, so the distribution test
        must see the raw projector space), while the contrastive losses keep
        consuming the normalized view.

        Returns:
            Tuple ``(z, s)``: ``z`` is the L2-normalized float32 embedding
            (identical to :meth:`forward`), ``s`` is the raw pre-normalization
            float32 projector output of the same shape.
        """
        s = self.projector(self.input_dropout(features)).float()
        z = F.normalize(s, dim=-1)
        return z, s


def epps_pulley_sigreg(
    embeddings: Tensor,
    num_slices: int = 1024,
    num_knots: int = 17,
    t_max: float = 3.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """SIGReg: sketched Epps–Pulley distance between the embedding distribution and N(0, I).

    The LeJEPA regularizer: by the Cramér–Wold theorem, multivariate
    standard-Gaussianity reduces to standard normality of 1-D projections,
    so the statistic averages the Epps–Pulley functional (weighted L2
    distance between empirical and target characteristic functions) over
    ``num_slices`` random unit directions. No whitening or centering: the
    target is the standard Gaussian itself. Gradients are bounded and the
    cost is linear in the sample count.

    Args:
        embeddings: Raw (pre-normalization) embeddings of shape ``(N, K)``, float32.
        num_slices: Number of random unit directions, resampled every call.
        num_knots: Trapezoidal quadrature knots on ``[0, t_max]``.
        t_max: Upper integration bound.
        generator: Optional RNG so callers can keep directions consistent
            across DDP ranks.

    Returns:
        Scalar statistic (>= 0); an autograd-connected zero when fewer than
        two samples are given.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2-D (N, K) embedding tensor, got shape {tuple(embeddings.shape)}."
        )
    n = embeddings.shape[0]
    if n < 2:
        return embeddings.sum() * 0.0
    k = embeddings.shape[1]
    samples = embeddings.float()
    directions = F.normalize(
        torch.randn(num_slices, k, generator=generator, device=samples.device), dim=-1
    )
    projections = samples @ directions.T  # (N, num_slices)
    t = torch.linspace(
        0.0, t_max, num_knots, device=samples.device, dtype=torch.float32
    )
    # Empirical characteristic function of each projection on the grid: (num_slices, num_knots).
    angles = projections.T.unsqueeze(-1) * t
    phi = angles.cos().mean(dim=0) + 1j * angles.sin().mean(dim=0)
    target = torch.exp(-0.5 * t**2)
    integrand = (phi - target).abs().square() * torch.exp(-0.5 * t**2)
    return torch.trapezoid(integrand, t, dim=-1).mean()


class PaddedGather(NamedTuple):
    """Rank-major concatenation of padded per-rank blocks plus a validity mask."""

    tensor: Tensor
    valid: Tensor


def gather_distributed(tensor: Tensor) -> Tensor:
    """Concatenate *tensor* across all DDP ranks, keeping the local gradient slot.

    ``all_gather`` returns gradient-free copies from every rank; the current
    rank's slot is replaced with the original tensor so gradients flow only
    through local rows while the loss still sees the full negative pool
    (SimCSE semantics). Order is deterministic: rank-major concatenation.
    Every rank must pass an identical shape — use
    :func:`gather_distributed_padded` when per-rank counts may differ.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    gathered[dist.get_rank()] = tensor
    return torch.cat(gathered, dim=0)


def gather_distributed_padded(tensor: Tensor) -> PaddedGather:
    """Gather across ranks even when per-rank counts differ, padding to a common size.

    Pad embeddings are zero vectors and pad labels are 0 — callers must
    exclude pad rows/columns from the losses via :attr:`PaddedGather.valid`
    (the losses accept ``valid_rows``/``valid_cols`` for exactly this).
    """
    device = tensor.device
    if not (dist.is_available() and dist.is_initialized()):
        return PaddedGather(
            tensor=tensor,
            valid=torch.ones(tensor.shape[0], dtype=torch.bool, device=device),
        )
    size = torch.tensor([tensor.shape[0]], device=device, dtype=torch.long)
    sizes_buffer = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes_buffer, size)
    sizes = [int(s.item()) for s in sizes_buffer]
    if len(set(sizes)) == 1:
        return PaddedGather(
            tensor=gather_distributed(tensor),
            valid=torch.ones(
                dist.get_world_size() * sizes[0], dtype=torch.bool, device=device
            ),
        )
    max_n = max(sizes)
    padded = tensor.new_zeros((max_n, *tensor.shape[1:]))
    padded[: tensor.shape[0]] = tensor
    blocks = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(blocks, padded.contiguous())
    # Slot replacement (SimCSE semantics): the local block keeps its
    # autograd connection so gradients flow through this rank's real rows.
    blocks[dist.get_rank()] = padded
    valid = torch.cat([torch.arange(max_n, device=device) < n for n in sizes])
    return PaddedGather(tensor=torch.cat(blocks, dim=0), valid=valid)


def cosine_logits(z1: Tensor, z2: Tensor, temp: float) -> Tensor:
    """``cos(z1, z2) / temp`` logit matrix, normalised in float32."""
    return F.normalize(z1.float(), dim=-1) @ F.normalize(z2.float(), dim=-1).T / temp


def simcse_infonce_loss(
    z1: Tensor,
    z2: Tensor,
    temp: float = 0.05,
    *,
    logits: Tensor | None = None,
    valid_rows: Tensor | None = None,
    valid_cols: Tensor | None = None,
) -> Tensor:
    """Unsupervised SimCSE objective: diagonal positives, in-batch negatives.

    Row ``i`` of ``z1`` and column ``i`` of ``z2`` are two views of the same
    instance; every other pair is a negative. ``z2`` may carry EXTRA
    candidates beyond the paired ones (``z2.shape[0] >= z1.shape[0]``) —
    columns ``>= z1.shape[0]`` are then always negatives (the unpaired
    instances of design doc 14 §4.10/D14).

    Args:
        z1: Anchor embeddings of shape ``(N, D)``.
        z2: Candidate embeddings of shape ``(C, D)`` with ``C >= N``; the
            first ``N`` rows are the paired candidates.
        temp: Temperature applied to the cosine logits.
        logits: Optional precomputed ``(N, C)`` logits (e.g. after
            :func:`positive_logit_bias`); replaces the internally computed
            cosine logits.
        valid_rows: Optional boolean mask of shape ``(N,)`` — real anchors
            in a padded pool; non-valid rows never become anchors.
        valid_cols: Optional boolean mask of shape ``(C,)`` — real
            candidates; non-valid columns are excluded from the softmax
            denominator.

    Returns:
        Scalar cross-entropy loss.

    Raises:
        ValueError: On shape mismatches, fewer than 2 anchors (the
            objective degenerates for ``N == 1``; callers guard the pool
            size beforehand), or mismatched masks.
    """
    if (
        z1.ndim != 2
        or z2.ndim != 2
        or z2.shape[0] < z1.shape[0]
        or z2.shape[1] != z1.shape[1]
    ):
        raise ValueError(
            f"Expected 2-D views with z2 rows >= z1 rows and matching dims, "
            f"got {tuple(z1.shape)} and {tuple(z2.shape)}."
        )
    if z1.shape[0] < 2:
        raise ValueError(
            f"Contrastive pool needs at least 2 anchors, got {z1.shape[0]}."
        )
    if logits is None:
        logits = cosine_logits(z1, z2, temp)
    else:
        expected = (z1.shape[0], z2.shape[0])
        if logits.shape != expected:
            raise ValueError(
                f"logits {tuple(logits.shape)} must be {expected}-shaped (N, C)."
            )

    anchor_rows = torch.arange(z1.shape[0], device=z1.device)
    if valid_rows is not None:
        if valid_rows.shape != (z1.shape[0],):
            raise ValueError(
                f"valid_rows must have shape ({z1.shape[0]},), got {tuple(valid_rows.shape)}."
            )
        anchor_rows = anchor_rows[valid_rows]
    if valid_cols is not None:
        if valid_cols.shape != (z2.shape[0],):
            raise ValueError(
                f"valid_cols must have shape ({z2.shape[0]},), got {tuple(valid_cols.shape)}."
            )
        # masked_fill, not a multiply: pad columns must leave the softmax
        # denominator without producing -inf * 0 = NaN.
        logits = logits.masked_fill(~valid_cols[None, :], float("-inf"))
    return F.cross_entropy(logits[anchor_rows], anchor_rows)


def supervised_contrastive_loss(
    z1: Tensor,
    z2: Tensor,
    labels1: Tensor,
    labels2: Tensor,
    temp: float = 0.05,
    image_ids1: Tensor | None = None,
    image_ids2: Tensor | None = None,
    same_image_positive_only: bool = False,
    *,
    logits: Tensor | None = None,
    valid_rows: Tensor | None = None,
    valid_cols: Tensor | None = None,
) -> Tensor:
    """Multi-positive supervised objective (SupCon-style extension of SimCSE-sup).

    Anchor ``i`` is positive with its own view-2 instance (column ``i``) and
    with every candidate sharing its label. ``z2`` may carry EXTRA candidates
    beyond the paired ones (``z2.shape[0] >= z1.shape[0]``) — same-label
    extras become soft positives too, the rest stay negatives (design doc
    14, D14). This extends SimCSE-sup, whose positives are an explicit
    paired column and whose remaining in-batch entries stay negative.

    Pad rows/columns carry label 0, which may collide with a real class 0 —
    the ``valid_rows``/``valid_cols`` handling excludes pad columns from the
    positive mask so a real class-0 anchor can never adopt a pad column as
    a positive.
    """
    if (
        z1.ndim != 2
        or z2.ndim != 2
        or z2.shape[0] < z1.shape[0]
        or z2.shape[1] != z1.shape[1]
    ):
        raise ValueError(
            f"Expected 2-D views with z2 rows >= z1 rows and matching dims, "
            f"got {tuple(z1.shape)} and {tuple(z2.shape)}."
        )
    if labels1.shape != (z1.shape[0],) or labels2.shape != (z2.shape[0],):
        raise ValueError(
            f"labels1 must have shape ({z1.shape[0]},) and labels2 shape ({z2.shape[0]},), "
            f"got {tuple(labels1.shape)} and {tuple(labels2.shape)}."
        )
    if z1.shape[0] < 2:
        raise ValueError(
            f"Contrastive pool needs at least 2 anchors, got {z1.shape[0]}."
        )
    if logits is None:
        logits = cosine_logits(z1, z2, temp)
    else:
        expected = (z1.shape[0], z2.shape[0])
        if logits.shape != expected:
            raise ValueError(
                f"logits {tuple(logits.shape)} must be {expected}-shaped (N, C)."
            )

    positive_mask = labels1[:, None] == labels2[None, :]
    if same_image_positive_only:
        if image_ids1 is None or image_ids2 is None:
            raise ValueError(
                "same_image_positive_only=True requires image_ids1 and image_ids2."
            )
        positive_mask &= image_ids1[:, None] == image_ids2[None, :]
    # The paired candidate of anchor i is column i — always a positive
    # (rectangular-safe diagonal fill; extras have no forced diagonal).
    paired = torch.arange(z1.shape[0], device=z1.device)
    positive_mask[paired, paired] = True

    if valid_rows is not None:
        if valid_rows.shape != (z1.shape[0],):
            raise ValueError(
                f"valid_rows must have shape ({z1.shape[0]},), got {tuple(valid_rows.shape)}."
            )
    if valid_cols is not None:
        if valid_cols.shape != (z2.shape[0],):
            raise ValueError(
                f"valid_cols must have shape ({z2.shape[0]},), got {tuple(valid_cols.shape)}."
            )
        logits = logits.masked_fill(~valid_cols[None, :], float("-inf"))
        positive_mask &= valid_cols[None, :]

    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_counts = positive_mask.sum(dim=1)
    # masked_fill, not a multiply: masked pad columns hold -inf log-probs
    # and -inf * 0 = NaN (design doc 14, M2 hard constraint (c)).
    per_anchor = (
        -(log_probs.masked_fill(~positive_mask, 0.0)).sum(dim=1) / positive_counts
    )
    if valid_rows is not None:
        per_anchor = per_anchor[valid_rows]
    return per_anchor.mean()


def positive_logit_bias(
    logits: Tensor,
    positive_mask: Tensor,
    iou: Tensor | None = None,
    gamma: float = 1.0,
    bias: float = 0.5,
) -> Tensor:
    """Add the IoU-aligned bias to positive logits only (A1 semantics).

    Marked positions receive ``logits + bias * log(iou^gamma)`` so weaker
    alignments contribute less; unmarked positions get ``log(1) == 0``,
    i.e. no shift. Non-positive positions must never be marked with a
    sub-1 weight — that would push ordinary negatives toward ``-inf``
    and disable them.
    """
    if logits.shape != positive_mask.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} and positive_mask {tuple(positive_mask.shape)} must match."
        )
    if iou is not None and iou.shape != logits.shape:
        raise ValueError(
            f"iou {tuple(iou.shape)} must match logits {tuple(logits.shape)}."
        )
    weight = torch.ones_like(logits)
    if iou is not None:
        weight = torch.where(
            positive_mask, iou.float().clamp(min=_EPS).pow(gamma), weight
        )
    return logits + bias * weight.log()


class CrossViewMatch(NamedTuple):
    """Validated correspondences between two views' instance pools."""

    row_idx: Tensor
    col_idx: Tensor
    iou: Tensor


def _mirror_xyxy(boxes: Tensor) -> Tensor:
    """Mirror xyxy boxes horizontally within the normalized [0, 1] canvas."""
    mirrored = boxes.clone()
    mirrored[:, 0] = 1.0 - boxes[:, 2]
    mirrored[:, 2] = 1.0 - boxes[:, 0]
    return mirrored


def match_cross_view(
    boxes1: Tensor,
    boxes2: Tensor,
    labels1: Tensor,
    labels2: Tensor,
    iou_threshold: float = 0.55,
    flip_invariant: bool = False,
) -> CrossViewMatch:
    """Recover same-object correspondences between two views via maximal box IoU.

    Runs an optimal one-to-one assignment maximising total IoU and validates
    each pair twice: IoU must reach ``iou_threshold`` AND the matched labels
    must agree. Failing pairs are dropped rather than kept as noisy
    positives. With ``flip_invariant`` the assignment scores each candidate
    pair under BOTH alignments — identity and horizontal mirror — and keeps
    the stronger: the two views flip independently, so a mirrored object is
    the same object. Among equally optimal assignments scipy's tie-breaking
    is unspecified; callers must not rely on a specific assignment for
    degenerate (tied) inputs.
    """
    if (
        boxes1.ndim != 2
        or boxes1.shape[-1] != 4
        or boxes2.ndim != 2
        or boxes2.shape[-1] != 4
    ):
        raise ValueError(
            f"Expected (N, 4) xyxy boxes, got {tuple(boxes1.shape)} and {tuple(boxes2.shape)}."
        )
    if labels1.shape != (boxes1.shape[0],) or labels2.shape != (boxes2.shape[0],):
        raise ValueError("labels must be 1-D and match the leading box dimension.")
    device = boxes1.device
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        empty_idx = torch.empty(0, dtype=torch.long, device=device)
        return CrossViewMatch(
            row_idx=empty_idx, col_idx=empty_idx, iou=torch.empty(0, device=device)
        )
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    iou = box_iou(boxes1, boxes2)
    if flip_invariant:
        iou = torch.maximum(iou, box_iou(boxes1, _mirror_xyxy(boxes2)))
    rows, cols = linear_sum_assignment(iou.detach().cpu().numpy(), maximize=True)
    row_idx = torch.as_tensor(rows, dtype=torch.long, device=device)
    col_idx = torch.as_tensor(cols, dtype=torch.long, device=device)
    keep = (iou[row_idx, col_idx] >= iou_threshold) & (
        labels1[row_idx] == labels2[col_idx]
    )
    return CrossViewMatch(
        row_idx=row_idx[keep],
        col_idx=col_idx[keep],
        iou=iou[row_idx[keep], col_idx[keep]],
    )


def _flatten_indices(
    indices: list[tuple[Tensor, Tensor]], device: torch.device
) -> tuple[Tensor, Tensor]:
    """Expand per-image matcher indices into flat ``(batch_row, query)`` pools.

    The matcher's assignment may live on CPU (scipy) while the SSL features
    are on the accelerator — everything is aligned to the feature device
    before any indexing.
    """
    batch_parts, src_parts = [], []
    for image_index, (src_idx, _tgt_idx) in enumerate(indices):
        batch_parts.append(torch.full_like(src_idx, image_index))
        src_parts.append(src_idx)
    return torch.cat(batch_parts).to(device), torch.cat(src_parts).to(device)


def _connected_zero(z1: Tensor) -> Tensor:
    """A numerically zero loss that stays attached to the autograd graph.

    Every degenerate path (empty local pool, all cross-view pairs dropped)
    returns this so DDP still observes gradients on the SSL parameters and
    the all-reduce stays consistent across ranks.
    """
    return z1.sum() * 0.0


def _ssl_candidates_loss(
    z1: Tensor,
    z2: Tensor,
    params: SSLParams,
    stats_sink: dict[str, float] | None = None,
) -> Tensor:
    """Route B pool (design doc 14 §4.10): every decoder candidate is an instance.

    The instances are the top-k candidate features gathered from the encoder
    memory (``enc_output`` output rows — gradients flow back into the
    encoder). No matching, no labels, no boxes: the two views are two
    dropout draws over the same candidate features, so the objective is
    plain in-batch InfoNCE with no IoU bias. ``two_crop`` is meaningless
    here (candidate proposals from independent crops are not comparable)
    and is rejected.
    """
    if params.mode != "unsup":
        raise ValueError(
            "instance_source='encoder_candidates' is label-free and supports mode='unsup' only."
        )
    if params.view == "two_crop":
        raise ValueError(
            "instance_source='encoder_candidates' supports view='dropout' only "
            "(candidate proposals are not comparable across independent crops)."
        )
    device = z1.device
    view1 = z1.reshape(-1, z1.shape[-1])
    view2 = z2.reshape(-1, z2.shape[-1])
    if stats_sink is not None:
        stats_sink["pool_size"] = float(view1.shape[0])
        stats_sink["paired_size"] = float(view1.shape[0])

    use_distributed = (
        dist.is_available() and dist.is_initialized() and params.gather_distributed
    )
    if use_distributed:
        # Same global enter/skip consensus as the matched pool: ranks must
        # never diverge inside the collectives.
        has_pool = torch.tensor(
            [int(view1.shape[0] >= 2)], device=device, dtype=torch.int64
        )
        dist.all_reduce(has_pool, op=dist.ReduceOp.MAX)
        if int(has_pool.item()) == 0:
            return _connected_zero(z1)
        gathered1 = gather_distributed_padded(view1)
        gathered2 = gather_distributed_padded(view2)
        view1, view2 = gathered1.tensor, gathered2.tensor
        valid_rows: Tensor | None = gathered1.valid
        valid_cols: Tensor | None = gathered2.valid
    else:
        valid_rows = valid_cols = None
    if view1.shape[0] < 2:
        return _connected_zero(z1)
    return simcse_infonce_loss(
        view1, view2, params.temp, valid_rows=valid_rows, valid_cols=valid_cols
    )


def _pair_two_crop(
    batch_sel: Tensor,
    src_sel: Tensor,
    labels_all: Tensor,
    boxes_xyxy: Tensor,
    images_per_view: int,
    iou_match: float,
    flip_invariant: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Match instances across the two views of every image pair.

    View-1 rows of the doubled batch sit below ``images_per_view``; their
    view-2 counterparts are ``row + images_per_view`` (the boundary comes
    from the batch dimension, never from ``rows.max()``). Returns
    ``(rows1, srcs1, rows2, srcs2, pair_ious, unpaired)`` — the validated
    pairs as pool indices, each pair's alignment IoU (the caller scales the
    positive-logit bias with it), and the pool positions of matched
    instances that NO pair claimed (both views; the D14 extra-negative
    candidates). When no pair validates, the pair tensors are empty and
    ``unpaired`` is the FULL pool — the caller sees zero anchors and takes
    its degenerate/connected-zero path (single process) or contributes
    extras-only candidate columns (distributed).

    Grouping is segment-based (one stable sort + searchsorted boundaries,
    O(pool)) instead of a per-image boolean-mask scan (O(images × pool)).
    The whole loop runs on the CPU: pools are tiny (matched instances only)
    and one host transfer per STEP beats several per IMAGE — the int64
    tensors ride one transfer and the float boxes another, so the CUDA
    pipeline drains twice, not four times. scipy's assignment runs per
    image deliberately — a block-diagonal batched assignment would be
    asymptotically worse, O((Σn)³) vs Σ O(n³) (design doc 14, R8).
    """
    packed_int = torch.cat([batch_sel, src_sel, labels_all]).detach().cpu()
    n_pool = batch_sel.numel()
    batch_sel = packed_int[:n_pool]
    src_sel = packed_int[n_pool : 2 * n_pool]
    labels_all = packed_int[2 * n_pool :]
    boxes_xyxy = boxes_xyxy.detach().cpu()

    # Segment the pool by image row (stable sort keeps pool order per image).
    order = torch.argsort(batch_sel, stable=True)
    sorted_rows = batch_sel[order]
    uniq, counts = torch.unique_consecutive(sorted_rows, return_counts=True)
    ends = torch.cumsum(counts, dim=0)
    starts = ends - counts

    def _segment(image_id: int) -> tuple[slice, Tensor] | None:
        idx = torch.searchsorted(uniq, torch.tensor(image_id, dtype=uniq.dtype))
        if idx >= uniq.numel() or uniq[idx].item() != image_id:
            return None
        return slice(int(starts[idx]), int(ends[idx])), order[starts[idx] : ends[idx]]

    rows1_list: list[Tensor] = []
    srcs1_list: list[Tensor] = []
    rows2_list: list[Tensor] = []
    srcs2_list: list[Tensor] = []
    ious_list: list[Tensor] = []
    for pos, image_id in enumerate(uniq.tolist()):
        if image_id >= images_per_view:
            continue  # view-2 rows are visited through their view-1 counterpart
        seg1 = slice(int(starts[pos]), int(ends[pos]))
        seg2_info = _segment(image_id + images_per_view)
        if seg2_info is None:
            continue
        seg2, _ = seg2_info
        pos1 = order[seg1]
        pos2 = order[seg2]
        match = match_cross_view(
            boxes_xyxy[pos1],
            boxes_xyxy[pos2],
            labels_all[pos1],
            labels_all[pos2],
            iou_threshold=iou_match,
            flip_invariant=flip_invariant,
        )
        if match.row_idx.numel() == 0:
            continue
        rows1_list.append(pos1[match.row_idx])
        srcs1_list.append(src_sel[pos1][match.row_idx])
        rows2_list.append(pos2[match.col_idx])
        srcs2_list.append(src_sel[pos2][match.col_idx])
        ious_list.append(match.iou)
    empty_idx = torch.empty(0, dtype=torch.long)
    rows1 = torch.cat(rows1_list) if rows1_list else empty_idx
    rows2 = torch.cat(rows2_list) if rows2_list else empty_idx
    srcs1 = torch.cat(srcs1_list) if srcs1_list else empty_idx
    srcs2 = torch.cat(srcs2_list) if srcs2_list else empty_idx
    ious = torch.cat(ious_list) if ious_list else torch.empty(0)
    # Unpaired matched instances (either view) — D14 extra candidates. With
    # zero validated pairs the paired flags stay all-False, so this is the
    # full pool by construction.
    paired_flag = torch.zeros(batch_sel.numel(), dtype=torch.bool)
    paired_flag[rows1] = True
    paired_flag[rows2] = True
    unpaired = torch.nonzero(~paired_flag).squeeze(1)
    return (rows1, srcs1, rows2, srcs2, ious, unpaired)


def ssl_contrastive_loss(
    z1: Tensor,
    z2: Tensor,
    outputs_final: dict[str, Tensor],
    targets: list[dict[str, Tensor]],
    indices: list[tuple[Tensor, Tensor]],
    params: SSLParams,
    query_mask: Tensor | None = None,
    stats_sink: dict[str, float] | None = None,
) -> Tensor:
    """Per-object contrastive loss on final-layer matched query features.

    Args:
        z1: Projection of the final-layer query features, view-1 draw,
            shape ``(B_total, Q, D)`` — ``B_total`` is ``2B`` under the
            ``two_crop`` view (row ``r`` and row ``r + B`` are the same
            source image), else ``B``.
        z2: Second independent projection draw, same shape as ``z1``.
        outputs_final: Final-layer output dict; ``pred_logits`` provides
            the batch boundary for the two-view split.
        targets: Per-image targets with ``"labels"`` and normalized
            cxcywh ``"boxes"`` (the latter only read under ``two_crop``).
        indices: The final-layer Hungarian match — the same assignment the
            box loss consumes; no extra matching happens here.
        params: The frozen SSL contract.
        query_mask: Optional ``(B_total, Q)`` bool mask of valid (non-pad)
            queries — padded queries of dynamic-query decoders are filtered
            defensively (their logits are already prohibitive at the source).
        stats_sink: Optional dict receiving ``pool_size`` (matched instances
            on this rank), ``paired_size`` (anchors with a validated
            cross-view positive) and ``candidate_size`` (paired + D14
            unpaired extras entering the contrastive pool).

    Returns:
        Scalar loss. Degenerate paths return a zero loss connected to the
        autograd graph (:func:`_connected_zero`) so DDP gradient all-reduce
        stays consistent; under DDP the enter/skip decision is synchronised
        with an ``all_reduce`` so ranks never diverge inside the gather
        collectives.
    """
    two_crop = params.view == "two_crop"
    device = z1.device
    if params.same_image_positive_only and not two_crop:
        # Without two_crop rows every instance shares one image, and the
        # rank-offset image-id arithmetic below would fabricate cross-rank
        # same-image positives — reject instead (mirrors the config
        # validator for hand-built SSLParams).
        raise ValueError("same_image_positive_only requires view='two_crop'.")
    if params.instance_source == "encoder_candidates":
        # Route B: the pool is the candidate features, independent of the
        # Hungarian match — targets/indices are ignored on this path.
        return _ssl_candidates_loss(z1, z2, params, stats_sink)
    images_per_view = outputs_final["pred_logits"].shape[0] // 2 if two_crop else 0

    # ── instance pool: final-layer match ∩ valid queries ────────────────
    batch_sel, src_sel = _flatten_indices(indices, device)
    keep = torch.ones(batch_sel.shape[0], dtype=torch.bool, device=device)
    if query_mask is not None:
        keep &= query_mask[batch_sel, src_sel]
    batch_sel, src_sel = batch_sel[keep], src_sel[keep]
    labels_all = torch.cat(
        [t["labels"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)]
    )[keep].to(device)
    boxes_all: Tensor | None = None
    if two_crop:
        boxes_all = box_cxcywh_to_xyxy(
            torch.cat(
                [t["boxes"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)]
            )[keep]
        ).to(device)
    if stats_sink is not None:
        stats_sink["pool_size"] = float(batch_sel.numel())
        stats_sink["paired_size"] = 0.0  # completed after pairing (0 on early exits)

    # ── distributed enter/skip consensus (never a per-rank decision) ────
    use_distributed = (
        dist.is_available() and dist.is_initialized() and params.gather_distributed
    )
    local_has_pool = batch_sel.numel() >= 2
    if use_distributed:
        has_pool = torch.tensor([int(local_has_pool)], device=device, dtype=torch.int64)
        torch.distributed.all_reduce(has_pool, op=dist.ReduceOp.MAX)
        if int(has_pool.item()) == 0:
            return _connected_zero(z1)
    elif not local_has_pool:
        return _connected_zero(z1)

    # ── view selection / cross-view correspondence ──────────────────────
    pool_positions = torch.arange(batch_sel.numel(), device=device)
    rows1 = rows2 = pool_positions
    srcs1 = srcs2 = src_sel
    pair_ious: Tensor | None = None
    extras_z: Tensor | None = None
    extras_labels: Tensor | None = None
    extras_boxes: Tensor | None = None
    extras_ids: Tensor | None = None
    if two_crop:
        assert boxes_all is not None
        rows1, srcs1, rows2, srcs2, pair_ious_cpu, unpaired_cpu = _pair_two_crop(
            batch_sel,
            src_sel,
            labels_all,
            boxes_all,
            images_per_view,
            params.iou_match,
            params.flip_invariant_pairing,
        )
        rows1, srcs1 = rows1.to(device), srcs1.to(device)
        rows2, srcs2 = rows2.to(device), srcs2.to(device)
        pair_ious = pair_ious_cpu.to(device)
        if params.use_unpaired_as_negatives:
            # D14: matched instances whose counterpart failed validation
            # rejoin the pool as extra candidate columns. Their embedding
            # comes from the draw of their own view. Built even when the
            # local unpaired set is empty so the gather collectives below
            # stay symmetric across ranks (the flag is global config;
            # emptiness is per-rank data).
            unpaired = unpaired_cpu.to(device)
            extras_is_view2 = batch_sel[unpaired] >= images_per_view
            unpaired_srcs = src_sel[unpaired]
            extras_z = torch.where(
                extras_is_view2[:, None],
                z2[batch_sel[unpaired], unpaired_srcs],
                z1[batch_sel[unpaired], unpaired_srcs],
            )
            extras_labels = labels_all[unpaired]
            extras_boxes = boxes_all[unpaired]

    # Anchors = paired view-1 instances (possibly zero on this rank);
    # candidates = their paired view-2 counterparts FIRST (columns i < N are
    # the diagonal positives), then the D14 unpaired extras.
    view1 = z1[batch_sel[rows1], srcs1]
    view2 = z2[batch_sel[rows2], srcs2]
    labels1 = labels_all[rows1]
    labels2 = labels_all[rows2]
    boxes1_local: Tensor | None = None
    boxes2_local: Tensor | None = None
    if two_crop:
        assert boxes_all is not None
        boxes1_local = boxes_all[rows1]
        boxes2_local = boxes_all[rows2]
    if stats_sink is not None:
        stats_sink["paired_size"] = float(view1.shape[0])
        stats_sink["candidate_size"] = float(
            view2.shape[0] + (extras_z.shape[0] if extras_z is not None else 0)
        )

    # Same-image positives need globally unique image ids across the
    # gathered pool — rank r offsets its image space by r * B (two_crop row
    # arithmetic makes ``row % B`` the source image; view-2 rows fold back
    # by subtracting ``images_per_view``).
    image_ids1: Tensor | None = None
    image_ids2: Tensor | None = None
    if params.mode in ("sup", "both") and params.same_image_positive_only:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        offset = rank * images_per_view
        image_ids1 = batch_sel[rows1] + offset
        image_ids2 = batch_sel[rows2] - images_per_view + offset
        if extras_z is not None:
            # Same fold-back for the extras; gathered separately from the
            # paired ids so the paired-first candidate order is preserved.
            extras_rows = batch_sel[unpaired]
            extras_ids = (
                torch.where(
                    extras_is_view2,
                    extras_rows - images_per_view,
                    extras_rows,
                )
                + offset
            )

    # ── gather the pool (z / labels / boxes / ids in the same rank order) ─
    # Paired candidates and D14 extras are padded-gathered SEPARATELY and
    # concatenated paired-first. The paired block then has the exact block
    # structure of the anchor rows (same per-rank sizes → same padding), so
    # the ``arange`` diagonal IS the correct pairing for every rank.
    # Pre-concatenating extras into each rank's candidate block before the
    # gather would interleave them per rank and break that diagonal under
    # per-rank count asymmetry — the multi-GPU failure this layout prevents.
    boxes1: Tensor | None = None
    if use_distributed:
        gathered1 = gather_distributed_padded(view1)
        gathered2 = gather_distributed_padded(view2)
        view1 = gathered1.tensor
        valid_rows = gathered1.valid
        paired_valid = gathered2.valid
        if extras_z is not None:
            gathered_extras = gather_distributed_padded(extras_z)
            cand_z = torch.cat([gathered2.tensor, gathered_extras.tensor])
            valid_cols = torch.cat([paired_valid, gathered_extras.valid])
        else:
            cand_z = gathered2.tensor
            valid_cols = paired_valid
        labels1 = gather_distributed_padded(labels1).tensor
        cand_labels = gather_distributed_padded(labels2).tensor
        if extras_z is not None:
            cand_labels = torch.cat(
                [cand_labels, gather_distributed_padded(extras_labels).tensor]
            )
        if boxes_all is not None:
            boxes1 = gather_distributed_padded(boxes1_local).tensor
            cand_boxes = gather_distributed_padded(boxes2_local).tensor
            if extras_z is not None:
                cand_boxes = torch.cat(
                    [cand_boxes, gather_distributed_padded(extras_boxes).tensor]
                )
            if pair_ious is not None:
                pair_ious = gather_distributed_padded(pair_ious).tensor
        if image_ids1 is not None and image_ids2 is not None:
            image_ids1 = gather_distributed_padded(image_ids1).tensor
            paired_ids2 = gather_distributed_padded(image_ids2).tensor
            image_ids2 = (
                torch.cat(
                    [paired_ids2, gather_distributed_padded(extras_ids).tensor]
                )
                if extras_ids is not None
                else paired_ids2
            )
    else:
        valid_rows = valid_cols = None
        boxes1 = boxes1_local
        cand_boxes = boxes2_local
        cand_z, cand_labels = view2, labels2
        if extras_z is not None:
            cand_z = torch.cat([view2, extras_z])
            cand_labels = torch.cat([labels2, extras_labels])
            if cand_boxes is not None:
                cand_boxes = torch.cat([cand_boxes, extras_boxes])

    # Global post-pairing guard (reached identically on every rank — no
    # collectives below this line).
    if view1.shape[0] < 2:
        return _connected_zero(z1)

    # two_crop composes the IoU soft-alignment bias into the logits before
    # the losses; the dropout view has identical boxes across views
    # (IoU 1 → log 1 = 0, no bias). The bias lands only on positive
    # positions (the diagonal paired set, plus same-label entries in sup).
    biased_logits: Tensor | None = None
    if two_crop and boxes1 is not None and cand_boxes is not None:
        positive_mask = torch.zeros(
            (view1.shape[0], cand_z.shape[0]), dtype=torch.bool, device=view1.device
        )
        paired_diag = torch.arange(view1.shape[0], device=view1.device)
        positive_mask[paired_diag, paired_diag] = True
        if params.mode in ("sup", "both"):
            positive_mask |= labels1[:, None] == cand_labels[None, :]
        bias_iou = box_iou(boxes1.float(), cand_boxes.float()).nan_to_num(0.0)
        if pair_ious is not None and pair_ious.shape[0] == view1.shape[0]:
            # The bias must reflect the ALIGNMENT strength of each paired
            # instance. Raw cross-frame IoU misjudges flip-recovered pairs
            # (the match was found under the mirrored alignment), so the
            # pairing's own IoU takes precedence on the diagonal.
            bias_iou[paired_diag, paired_diag] = pair_ious
        biased_logits = positive_logit_bias(
            cosine_logits(view1, cand_z, params.temp),
            positive_mask=positive_mask,
            # Degenerate (zero-area) GT boxes give 0/0 = NaN IoU at their
            # diagonal position; fold them to 0 so the positive is softly
            # disabled instead of poisoning the loss with NaN.
            iou=bias_iou,
            gamma=params.iou_gamma,
            bias=params.iou_bias,
        )

    loss: Tensor | None = None
    if params.mode in ("unsup", "both"):
        loss = simcse_infonce_loss(
            view1,
            cand_z,
            params.temp,
            logits=biased_logits,
            valid_rows=valid_rows,
            valid_cols=valid_cols,
        )
    if params.mode in ("sup", "both"):
        sup_loss = supervised_contrastive_loss(
            view1,
            cand_z,
            labels1,
            cand_labels,
            params.temp,
            image_ids1=image_ids1,
            image_ids2=image_ids2,
            same_image_positive_only=params.same_image_positive_only,
            # R11: in "both" mode the two objectives get SEPARATE logits — the
            # IoU alignment bias is meaningful for the InfoNCE diagonal but
            # would simultaneously down-weight SupCon's same-label soft
            # positives on low-IoU pairs (opposing gradients over one shared
            # matrix). Pure "sup" keeps the biased logits (its documented
            # contract, unchanged).
            logits=biased_logits if params.mode == "sup" else None,
            valid_rows=valid_rows,
            valid_cols=valid_cols,
        )
        loss = sup_loss if loss is None else loss + sup_loss
    assert loss is not None  # params.mode is validated to unsup/sup/both
    return loss
