# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Representation-quality probes for the per-object SSL branch (design doc 14 §4.5).

The NLP analogue is SimCSE's STS-B/SICK-R development selection: these
probes play the same role for detection embeddings. All functions are
pure-torch, run on CPU, and operate on embeddings together with
ground-truth labels/boxes supplied by the caller — they are NOT wired into
the training loop; run them offline on extracted embeddings:

- :func:`linear_probe_accuracy` — frozen-feature logistic-regression top-1
  (class separability);
- :func:`retrieval_map_at_k` — same-class precision@K over cosine nearest
  neighbours (retrieval);
- :func:`alignment_uniformity` — Wang & Isola alignment/uniformity (collapse
  detectors);
- :func:`cross_view_nn_accuracy` — nearest-neighbour recovery of the
  two-view correspondence (two_crop embedding quality);
- :func:`position_leakage_r2` — how linearly predictable box coordinates
  are from embeddings (R² > 0.6 flags geometry memorisation over
  appearance, design doc 14 R10).

Extract embeddings by enabling ``ModelConfig``-style query-feature emission
during a short instrumented run and collecting ``query_features`` /
projected ``ssl_z1`` per matched instance.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

__all__ = [
    "alignment_uniformity",
    "cross_view_nn_accuracy",
    "linear_probe_accuracy",
    "position_leakage_r2",
    "retrieval_map_at_k",
]


def linear_probe_accuracy(
    train_z: Tensor,
    train_y: Tensor,
    val_z: Tensor,
    val_y: Tensor,
    *,
    epochs: int = 200,
    lr: float = 0.1,
    seed: int = 0,
) -> float:
    """Top-1 accuracy of a linear classifier trained on frozen embeddings.

    Args:
        train_z: Training embeddings of shape ``(N, D)``.
        train_y: Training labels of shape ``(N,)`` in ``[0, num_classes)``.
        val_z: Validation embeddings of shape ``(M, D)``.
        val_y: Validation labels of shape ``(M,)``.
        epochs: Full-batch optimisation steps.
        lr: Adam learning rate for the probe.
        seed: Seed for probe weight initialisation.

    Returns:
        Top-1 accuracy in ``[0, 1]`` on the validation split.
    """
    classifier = torch.nn.Linear(train_z.shape[1], int(train_y.max().item()) + 1)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        classifier.reset_parameters()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    features = train_z.detach().float()
    targets = train_y.detach().long()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss: Tensor = F.cross_entropy(classifier(features), targets)
        torch.autograd.backward(loss)
        optimizer.step()
    with torch.no_grad():
        predictions = classifier(val_z.detach().float()).argmax(dim=-1)
    accuracy = (predictions == val_y.detach().long()).float().mean()
    return float(accuracy.item())


def retrieval_map_at_k(z: Tensor, labels: Tensor, k: int = 10) -> float:
    """Mean same-class precision@K over cosine nearest neighbours, self excluded.

    Args:
        z: Embeddings of shape ``(N, D)``.
        labels: Instance labels of shape ``(N,)``.
        k: Neighbour count; capped at ``N - 1``.

    Returns:
        Mean precision@K in ``[0, 1]``; 0.0 when fewer than two instances exist.
    """
    if z.shape[0] < 2:
        return 0.0
    effective_k = min(k, z.shape[0] - 1)
    similarities = F.normalize(z.float(), dim=-1) @ F.normalize(z.float(), dim=-1).T
    similarities.fill_diagonal_(float("-inf"))
    neighbours = similarities.topk(effective_k, dim=1).indices
    hits = (labels[neighbours] == labels[:, None]).float().sum(dim=1)
    return (hits / effective_k).mean().item()


def alignment_uniformity(z1: Tensor, z2: Tensor) -> tuple[float, float]:
    """Wang & Isola alignment and uniformity (Wang & Isola, 2020).

    Alignment is the mean squared L2 distance between matched positives
    (lower is tighter); uniformity is ``log mean exp(-2 * ||x - y||^2)`` over
    all pairwise distances of view 1 (lower means the embeddings spread more
    uniformly — the direct collapse guard). Requires z1 to have at least two
    rows for a defined uniformity; a single row yields uniformity 0.

    Args:
        z1: View-1 embeddings of shape ``(N, D)``.
        z2: View-2 embeddings of shape ``(N, D)`` (matched rows).

    Returns:
        ``(alignment, uniformity)`` as floats.
    """
    alignment = (z1.float() - z2.float()).norm(dim=-1).pow(2).mean().item()
    pairwise = torch.pdist(z1.float(), p=2).pow(2)
    if pairwise.numel() == 0:
        uniformity = 0.0
    else:
        # log-mean-exp via logsumexp: numerically stable where
        # exp(-2 d^2) underflows to 0.
        uniformity = (torch.logsumexp(-2.0 * pairwise, dim=0) - math.log(pairwise.numel())).item()
    return alignment, uniformity


def cross_view_nn_accuracy(z1: Tensor, z2: Tensor) -> float:
    """Fraction of view-1 instances whose nearest view-2 neighbour is their own.

    The direct "box-aligned embedding" quality measure: if the contrastive
    objective worked, nearest-neighbour search alone recovers the two-view
    correspondence.

    Args:
        z1: View-1 embeddings of shape ``(N, D)``.
        z2: View-2 embeddings of shape ``(N, D)`` (matched rows).

    Returns:
        Top-1 correspondence accuracy in ``[0, 1]``.
    """
    if z1.shape[0] == 0:
        return 0.0
    similarities = F.normalize(z1.float(), dim=-1) @ F.normalize(z2.float(), dim=-1).T
    neighbours = similarities.argmax(dim=1)
    return (neighbours == torch.arange(z1.shape[0], device=z1.device)).float().mean().item()


def position_leakage_r2(z: Tensor, boxes: Tensor) -> float:
    """R² of predicting normalized box coordinates linearly from embeddings.

    High R² means the embedding memorises geometry rather than appearance —
    the position-leakage probe (design doc 14, R10). Fits a closed-form
    least-squares linear map (with bias) from embeddings to ``(cx, cy, w, h)``.

    Args:
        z: Embeddings of shape ``(N, D)``.
        boxes: Normalized ``cxcywh`` boxes of shape ``(N, 4)``.

    Returns:
        Aggregate R² across the four coordinate outputs; 0.0 when the
        coordinates have no variance or the pool is too small.
    """
    if z.shape[0] < 2 or boxes.var(dim=0).sum().item() == 0.0:
        return 0.0
    design = torch.cat([z.float(), torch.ones(z.shape[0], 1, device=z.device)], dim=1)
    solution = torch.linalg.lstsq(design, boxes.float()).solution
    predictions = design @ solution
    residual = ((boxes.float() - predictions) ** 2).sum()
    total = ((boxes.float() - boxes.float().mean(dim=0)) ** 2).sum()
    if total.item() == 0.0:
        return 0.0
    return float((1.0 - residual / total).item())
