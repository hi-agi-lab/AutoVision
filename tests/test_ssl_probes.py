# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the SSL representation-quality probes (design doc 14 §4.5).

All cases are hermetic: synthetic well-separated clusters with fixed seeds,
hand-computed invariants, and the documented thresholds (position-leakage
R² > 0.6 = geometry memorisation).
"""

import pytest
import torch

from autovision.metrics.ssl_probes import (
    alignment_uniformity,
    cross_view_nn_accuracy,
    linear_probe_accuracy,
    position_leakage_r2,
    retrieval_map_at_k,
)


def _cluster_embeddings(num_per_class: int = 20, dim: int = 16, num_classes: int = 4,
                        spread: float = 0.05, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.eye(dim)[:num_classes] * 10.0
    labels = torch.arange(num_classes).repeat_interleave(num_per_class)
    z = centers[labels] + spread * torch.randn(num_classes * num_per_class, dim, generator=generator)
    return z, labels


def test_linear_probe_separates_clean_clusters():
    z, labels = _cluster_embeddings()
    # Shuffle so both splits see every class (the fixture is class-sorted).
    perm = torch.randperm(len(labels), generator=torch.Generator().manual_seed(7))
    z, labels = z[perm], labels[perm]
    train_z, train_y = z[:60], labels[:60]
    val_z, val_y = z[60:], labels[60:]
    accuracy = linear_probe_accuracy(train_z, train_y, val_z, val_y, epochs=150)
    assert accuracy > 0.95
    assert 0.0 <= accuracy <= 1.0


def test_retrieval_map_perfect_on_tight_clusters():
    z, labels = _cluster_embeddings()
    assert retrieval_map_at_k(z, labels, k=5) == pytest.approx(1.0)


def test_retrieval_map_small_pool_and_k_cap():
    single = torch.randn(1, 4)
    assert retrieval_map_at_k(single, torch.tensor([0])) == 0.0
    z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])
    score = retrieval_map_at_k(z, labels, k=10)  # k capped at N-1 = 2
    assert 0.0 <= score <= 1.0


def test_alignment_uniformity_directions():
    torch.manual_seed(1)
    z1 = torch.randn(8, 8)
    alignment, _ = alignment_uniformity(z1, z1.clone())
    assert alignment == 0.0  # identical views are perfectly aligned
    tight = torch.randn(8, 8) * 0.1
    spread = torch.randn(8, 8) * 10.0
    _, uni_tight = alignment_uniformity(tight, tight)
    _, uni_spread = alignment_uniformity(spread, spread)
    # Uniformity decreases (more negative) as the cloud spreads — the
    # collapse guard direction pinned.
    assert uni_spread < uni_tight


def test_cross_view_nn_accuracy():
    torch.manual_seed(2)
    z1 = torch.randn(6, 8)
    assert cross_view_nn_accuracy(z1, z1.clone()) == 1.0
    rolled = torch.roll(z1, shifts=1, dims=0)
    assert cross_view_nn_accuracy(z1, rolled) < 1.0
    assert cross_view_nn_accuracy(torch.zeros(0, 4), torch.zeros(0, 4)) == 0.0


def test_position_leakage_thresholds():
    torch.manual_seed(3)
    # Embeddings that ARE the boxes: perfect linear readout → leakage.
    boxes = torch.rand(50, 4)
    assert position_leakage_r2(boxes, boxes) > 0.99
    # Random embeddings carry no box information.
    random_z = torch.randn(50, 16)
    assert position_leakage_r2(random_z, boxes) < 0.6
    # Degenerate inputs define R² = 0.
    assert position_leakage_r2(torch.randn(1, 4), torch.rand(1, 4)) == 0.0
    constant = torch.ones(10, 4)
    assert position_leakage_r2(torch.randn(10, 4), constant) == 0.0
