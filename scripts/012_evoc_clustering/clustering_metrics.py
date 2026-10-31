"""Clustering metrics for Experiment 012.

Pure metric helpers: no EVōC import, no I/O. The caller supplies partitions,
labels, and vectors. Functions keep noise (``-1``) as a distinct label unless a
function explicitly says otherwise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_completeness_v_measure,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class PartitionSummary:
    """Size and noise summary for one hard partition."""

    n: int
    clusters: int
    noise: int
    noise_fraction: float
    min_size: int
    median_size: float
    p90_size: float
    max_size: int
    largest_share: float


def partition_summary(labels: np.ndarray) -> PartitionSummary:
    """Summarize a partition, treating ``-1`` as noise."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    n = int(labels.size)
    if n == 0:
        raise ValueError("labels must not be empty")
    noise = int(np.sum(labels == -1))
    counts = np.fromiter(
        (c for label, c in Counter(labels.tolist()).items() if label != -1),
        dtype=np.int64,
    )
    if counts.size == 0:
        return PartitionSummary(
            n=n,
            clusters=0,
            noise=noise,
            noise_fraction=noise / n,
            min_size=0,
            median_size=0.0,
            p90_size=0.0,
            max_size=0,
            largest_share=0.0,
        )
    return PartitionSummary(
        n=n,
        clusters=int(counts.size),
        noise=noise,
        noise_fraction=noise / n,
        min_size=int(counts.min()),
        median_size=float(np.median(counts)),
        p90_size=float(np.quantile(counts, 0.9)),
        max_size=int(counts.max()),
        largest_share=float(counts.max() / n),
    )


def external_metrics(
    labels: np.ndarray,
    truth: np.ndarray,
    *,
    drop_noise: bool = False,
) -> dict[str, float]:
    """Return external partition metrics against single-label truth.

    Noise is kept as its own label by default. ``drop_noise=True`` is a
    sensitivity check only.
    """
    labels = np.asarray(labels)
    truth = np.asarray(truth)
    if labels.shape != truth.shape:
        raise ValueError("labels and truth must have the same shape")
    if labels.ndim != 1:
        raise ValueError("labels and truth must be one-dimensional")
    if drop_noise:
        mask = labels != -1
        labels = labels[mask]
        truth = truth[mask]
    if labels.size == 0:
        raise ValueError("no non-noise points remain")
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        truth, labels
    )
    return {
        "ami": float(adjusted_mutual_info_score(truth, labels)),
        "ari": float(adjusted_rand_score(truth, labels)),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "n": float(labels.size),
    }


def cluster_purity(labels: np.ndarray, truth: np.ndarray) -> float:
    """Return size-weighted majority-label purity over non-noise clusters."""
    labels = np.asarray(labels)
    truth = np.asarray(truth)
    if labels.shape != truth.shape:
        raise ValueError("labels and truth must have the same shape")
    total = 0
    correct = 0
    for cluster in sorted(set(labels.tolist()) - {-1}):
        members = truth[labels == cluster]
        if members.size == 0:
            continue
        total += int(members.size)
        correct += int(Counter(members.tolist()).most_common(1)[0][1])
    return correct / total if total else 0.0


def knn_neighbor_indices(
    vectors: np.ndarray,
    *,
    k: int = 15,
) -> np.ndarray:
    """Return an ``(n, k)`` matrix of neighbor indices, self excluded."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("vectors must be two-dimensional")
    if not 1 <= k < vectors.shape[0]:
        raise ValueError("k must be in [1, n-1]")
    nn = NearestNeighbors(n_neighbors=min(k + 1, vectors.shape[0]), metric="cosine")
    nn.fit(vectors)
    _, raw = nn.kneighbors(vectors)
    out = np.empty((vectors.shape[0], k), dtype=np.int64)
    for idx, row in enumerate(raw):
        neighbors = [int(j) for j in row if int(j) != idx][:k]
        if len(neighbors) < k:
            raise ValueError("not enough non-self neighbors")
        out[idx] = neighbors
    return out


def knn_purity_from_neighbors(
    neighbors: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Mean k-NN purity from a precomputed neighbor matrix."""
    labels = np.asarray(labels)
    if neighbors.ndim != 2:
        raise ValueError("neighbors must be two-dimensional")
    if neighbors.shape[0] != labels.shape[0]:
        raise ValueError("neighbors and labels must agree")
    mask = labels != -1
    if not np.any(mask):
        return 0.0
    same = labels[neighbors] == labels[:, None]
    return float(np.mean(same[mask]))


def knn_purity_per_point(
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 15,
) -> np.ndarray:
    """Return per-point k-NN purity; ``nan`` for noise query points."""
    labels = np.asarray(labels)
    neighbors = knn_neighbor_indices(vectors, k=k)
    out = np.full(labels.shape, np.nan, dtype=np.float64)
    same = labels[neighbors] == labels[:, None]
    for idx in range(labels.shape[0]):
        if labels[idx] != -1:
            out[idx] = float(np.mean(same[idx]))
    return out


def knn_purity(
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 15,
) -> float:
    """Mean fraction of a point's k nearest neighbors sharing its cluster.

    Self is excluded. Noise points are excluded as query points; a noise
    neighbor is counted as a mismatch because it has no shared cluster.
    """
    scores = knn_purity_per_point(vectors, labels, k=k)
    return float(np.nanmean(scores)) if np.any(~np.isnan(scores)) else 0.0


def centroid_metrics(
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int | None = 1000,
    random_state: int = 42,
) -> dict[str, float | None]:
    """Return centroid internal metrics on non-noise points.

    These are comparative-only on high-dimensional normalized embeddings.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    labels = np.asarray(labels)
    mask = labels != -1
    x = vectors[mask]
    y = labels[mask]
    if len(set(y.tolist())) < 2:
        return {
            "silhouette": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
        }
    return {
        "silhouette": float(
            silhouette_score(
                x,
                y,
                metric="cosine",
                sample_size=min(sample_size, len(y)) if sample_size else None,
                random_state=random_state,
            )
        ),
        "calinski_harabasz": float(calinski_harabasz_score(x, y)),
        "davies_bouldin": float(davies_bouldin_score(x, y)),
    }


def pairwise_ari(labels_by_arm: dict[str, np.ndarray]) -> dict[str, float]:
    """Return ARI for every unordered pair of named partitions."""
    names = sorted(labels_by_arm)
    out: dict[str, float] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            out[f"{left}|{right}"] = float(
                adjusted_rand_score(labels_by_arm[left], labels_by_arm[right])
            )
    return out


def mean_pairwise_ari(runs: list[np.ndarray]) -> float:
    """Return mean ARI across all run pairs on their common points."""
    if len(runs) < 2:
        return 0.0
    values = []
    for i, left in enumerate(runs):
        for right in runs[i + 1 :]:
            common = np.intersect1d(left[:, 0], right[:, 0], assume_unique=True)
            if common.size < 2:
                continue
            left_map = dict(zip(left[:, 0].tolist(), left[:, 1].tolist(), strict=True))
            right_map = dict(zip(right[:, 0].tolist(), right[:, 1].tolist(), strict=True))
            values.append(
                adjusted_rand_score(
                    [left_map[int(x)] for x in common],
                    [right_map[int(x)] for x in common],
                )
            )
    return float(np.mean(values)) if values else 0.0
