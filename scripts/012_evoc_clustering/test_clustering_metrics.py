"""Single-responsibility tests for Experiment 012 metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clustering_metrics import (
    external_metrics,
    knn_neighbor_indices,
    knn_purity,
    mean_pairwise_ari,
    pairwise_ari,
    partition_summary,
)


def test_partition_summary_keeps_noise_separate() -> None:
    summary = partition_summary(np.array([-1, 0, 0, 1, 1, 1]))
    assert summary.n == 6
    assert summary.clusters == 2
    assert summary.noise == 1
    assert summary.noise_fraction == pytest.approx(1 / 6)
    assert summary.min_size == 2
    assert summary.max_size == 3
    assert summary.largest_share == pytest.approx(0.5)


def test_external_metrics_perfect_partition() -> None:
    labels = np.array([0, 0, 1, 1, -1])
    truth = np.array(["a", "a", "b", "b", "c"])
    metrics = external_metrics(labels, truth)
    assert metrics["ari"] == pytest.approx(1.0)
    assert metrics["ami"] == pytest.approx(1.0)
    assert metrics["v_measure"] == pytest.approx(1.0)


def test_external_metrics_noise_excluded_sensitivity() -> None:
    labels = np.array([0, 0, 1, 1, -1])
    truth = np.array(["a", "a", "b", "b", "c"])
    kept = external_metrics(labels, truth)
    dropped = external_metrics(labels, truth, drop_noise=True)
    assert dropped["n"] == 4
    assert dropped["ari"] >= kept["ari"]


def test_knn_purity_separates_two_blobs() -> None:
    vectors = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
    )
    labels = np.array([0, 0, 1, 1])
    assert knn_purity(vectors, labels, k=1) == pytest.approx(1.0)


def test_neighbor_indices_exclude_self() -> None:
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=np.float32
    )
    neighbors = knn_neighbor_indices(vectors, k=2)
    for idx, row in enumerate(neighbors):
        assert idx not in row


def test_pairwise_ari_identity_and_disagreement() -> None:
    left = np.array([0, 0, 1, 1])
    relabeled = np.array([1, 1, 0, 0])
    crossed = np.array([0, 1, 0, 1])
    assert pairwise_ari({"a": left, "b": relabeled})["a|b"] == pytest.approx(1.0)
    assert pairwise_ari({"a": left, "b": crossed})["a|b"] == pytest.approx(-0.5)


def test_mean_pairwise_ari_on_subset_intersection() -> None:
    full = np.column_stack([np.arange(5), np.array([0, 0, 1, 1, 1])])
    same = np.column_stack([np.arange(5), np.array([0, 0, 1, 1, 1])])
    assert mean_pairwise_ari([full, same]) == pytest.approx(1.0)
