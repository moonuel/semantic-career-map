"""Run Experiment 012 EVōC clustering over the Experiment 011 embeddings.

Usage:
    python scripts/012_evoc_clustering/run_clustering.py --stage primary
    python scripts/012_evoc_clustering/run_clustering.py --stage all
    python scripts/012_evoc_clustering/run_clustering.py --stage metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import evoc
import numpy as np
from clustering_metrics import (
    centroid_metrics,
    cluster_purity,
    external_metrics,
    knn_neighbor_indices,
    knn_purity,
    knn_purity_from_neighbors,
    knn_purity_per_point,
    mean_pairwise_ari,
    pairwise_ari,
    partition_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
PLOTS_DIR = EXPERIMENT_DIR / "plots"
EXP011_DIR = PROJECT_ROOT / "scripts" / "011_encoder_coverage" / "data"
RUNS_DIR = DATA_DIR / "runs"
LABELS_PATH = DATA_DIR / "labels_primary_function.jsonl"

ARMS: dict[str, Path] = {
    "a_minilm_truncated": EXP011_DIR / "embeddings_a_minilm_truncated.npy",
    "b_minilm_pooled": EXP011_DIR / "embeddings_b_minilm_pooled.npy",
    "b2_minilm_pooled_overlap": EXP011_DIR / "embeddings_b2_minilm_pooled_overlap.npy",
    "c_titan_v2": EXP011_DIR / "embeddings_c_titan_v2.npy",
}

PRIMARY_CONFIG: dict[str, Any] = {
    "n_neighbors": 15,
    "noise_level": 0.5,
    "base_min_cluster_size": 10,
    "base_n_clusters": None,
    "approx_n_clusters": None,
    "min_samples": 5,
    "node_embedding_dim": None,
    "neighbor_scale": 1.0,
    "min_similarity_threshold": 0.2,
    "max_layers": 10,
    "n_epochs": 50,
    "node_embedding_init": "label_prop",
    "symmetrize_graph": True,
    "random_state": 42,
}

SENSITIVITY_CONFIGS: dict[str, dict[str, Any]] = {
    "primary": {},
    "n_neighbors_10": {"n_neighbors": 10},
    "n_neighbors_30": {"n_neighbors": 30},
    "base_min_cluster_size_5": {"base_min_cluster_size": 5},
    "base_min_cluster_size_20": {"base_min_cluster_size": 20},
    "min_samples_1": {"min_samples": 1},
    "min_samples_10": {"min_samples": 10},
}

SEEDS = [42, 43, 44, 45, 46]
SUBSAMPLE_SEEDS = list(range(42, 52))
SUBSAMPLE_FRACTION = 0.8
KNN_K = 15
BOOTSTRAP = 2000
PERMUTATIONS = 2000


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except subprocess.SubprocessError:
        return "unknown"


def _load_sample() -> list[dict[str, Any]]:
    path = EXP011_DIR / "sample_1000.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_truth() -> np.ndarray | None:
    if not LABELS_PATH.exists():
        return None
    rows = [json.loads(line) for line in LABELS_PATH.read_text().splitlines() if line.strip()]
    by_hash = {row["source_hash"]: row for row in rows}
    sample = _load_sample()
    if len(by_hash) != len(sample):
        return None
    if any(row["source_hash"] not in by_hash for row in sample):
        return None
    return np.array([by_hash[row["source_hash"]]["primary_function"] for row in sample])


def _summary_from_clusterer(clusterer: Any, arm: str, run_id: str, stage: str, config: dict[str, Any], seconds: float) -> dict[str, Any]:
    labels = np.asarray(clusterer.labels_).astype(int)
    layers = [np.asarray(layer).astype(int) for layer in clusterer.cluster_layers_]
    primary_index = next(
        (index for index, layer in enumerate(layers) if np.array_equal(layer, labels)),
        None,
    )
    summary = partition_summary(labels)
    layer_rows = []
    for index, layer in enumerate(layers):
        layer_summary = partition_summary(layer)
        persistence = float(clusterer.persistence_scores_[index]) if index < len(clusterer.persistence_scores_) else 0.0
        strengths = (
            np.asarray(clusterer.membership_strength_layers_[index])
            if index < len(clusterer.membership_strength_layers_)
            else np.array([])
        )
        layer_rows.append(
            {
                "layer": index,
                "clusters": layer_summary.clusters,
                "noise": layer_summary.noise,
                "noise_fraction": layer_summary.noise_fraction,
                "min_size": layer_summary.min_size,
                "median_size": layer_summary.median_size,
                "p90_size": layer_summary.p90_size,
                "max_size": layer_summary.max_size,
                "largest_share": layer_summary.largest_share,
                "persistence": persistence,
                "mean_membership": float(np.mean(strengths)) if strengths.size else 0.0,
            }
        )
    duplicates = sorted(
        [[int(i), int(j)] for i, j in getattr(clusterer, "duplicates_", set())]
    )
    return {
        "arm": arm,
        "run_id": run_id,
        "stage": stage,
        "config": config,
        "seconds": seconds,
        "git_commit": _git_commit(),
        "n": summary.n,
        "clusters": summary.clusters,
        "noise": summary.noise,
        "noise_fraction": summary.noise_fraction,
        "min_size": summary.min_size,
        "median_size": summary.median_size,
        "p90_size": summary.p90_size,
        "max_size": summary.max_size,
        "largest_share": summary.largest_share,
        "primary_layer": primary_index,
        "layers": layer_rows,
        "duplicate_pairs": len(duplicates),
        "duplicate_rate": len(duplicates) / summary.n,
        "duplicates": duplicates,
    }


def _run_dir(arm: str, stage: str, run_id: str) -> Path:
    return RUNS_DIR / arm / stage / run_id


def _fit(
    vectors: np.ndarray,
    arm: str,
    stage: str,
    run_id: str,
    config: dict[str, Any],
    *,
    force: bool,
) -> dict[str, Any]:
    out_dir = _run_dir(arm, stage, run_id)
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not force:
        try:
            return json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            pass
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    clusterer = evoc.EVoC(**config).fit(vectors)
    seconds = time.time() - started
    summary = _summary_from_clusterer(clusterer, arm, run_id, stage, config, seconds)
    labels = np.asarray(clusterer.labels_).astype(int)
    layers = np.stack([np.asarray(layer).astype(int) for layer in clusterer.cluster_layers_])
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "layers.npy", layers)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{arm:28s} {stage:12s} {run_id:22s} {seconds:7.2f}s clusters={summary['clusters']:2d} noise={summary['noise_fraction']:.3f}")
    return summary


def _run_primary(arms: list[str], force: bool) -> None:
    for arm in arms:
        vectors = np.load(ARMS[arm]).astype(np.float32)
        _fit(vectors, arm, "primary", "primary", PRIMARY_CONFIG, force=force)


def _run_sensitivity(arms: list[str], force: bool) -> None:
    for arm in arms:
        vectors = np.load(ARMS[arm]).astype(np.float32)
        for name, overrides in SENSITIVITY_CONFIGS.items():
            if name == "primary":
                continue
            config = {**PRIMARY_CONFIG, **overrides}
            _fit(vectors, arm, "sensitivity", name, config, force=force)


def _run_stability(arms: list[str], force: bool) -> None:
    for arm in arms:
        vectors = np.load(ARMS[arm]).astype(np.float32)
        for seed in SEEDS:
            if seed == PRIMARY_CONFIG["random_state"]:
                continue
            config = {**PRIMARY_CONFIG, "random_state": seed}
            _fit(vectors, arm, "seeds", f"seed_{seed}", config, force=force)

    n = len(_load_sample())
    for arm in ("b2_minilm_pooled_overlap", "c_titan_v2"):
        if arm not in arms:
            continue
        vectors = np.load(ARMS[arm]).astype(np.float32)
        for seed in SUBSAMPLE_SEEDS:
            out_dir = _run_dir(arm, "subsamples", f"seed_{seed}")
            summary_path = out_dir / "summary.json"
            if summary_path.exists() and not force:
                continue
            rng = np.random.default_rng(seed)
            indices = np.sort(
                rng.choice(n, size=round(n * SUBSAMPLE_FRACTION), replace=False)
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            config = {**PRIMARY_CONFIG, "random_state": seed}
            started = time.time()
            clusterer = evoc.EVoC(**config).fit(vectors[indices])
            seconds = time.time() - started
            summary = _summary_from_clusterer(
                clusterer, arm, f"seed_{seed}", "subsamples", config, seconds
            )
            summary["subsample_size"] = int(indices.size)
            np.save(out_dir / "indices.npy", indices.astype(np.int64))
            np.save(out_dir / "labels.npy", np.asarray(clusterer.labels_).astype(int))
            np.save(
                out_dir / "layers.npy",
                np.stack(
                    [np.asarray(layer).astype(int) for layer in clusterer.cluster_layers_]
                ),
            )
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            print(
                f"{arm:28s} subsample    seed_{seed:<17d} {seconds:7.2f}s "
                f"clusters={summary['clusters']:2d} noise={summary['noise_fraction']:.3f}"
            )


def _load_run(arm: str, stage: str, run_id: str) -> tuple[dict[str, Any] | None, np.ndarray | None, np.ndarray | None]:
    out_dir = _run_dir(arm, stage, run_id)
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return None, None, None
    summary = json.loads(summary_path.read_text())
    labels = np.load(out_dir / "labels.npy") if (out_dir / "labels.npy").exists() else None
    layers = np.load(out_dir / "layers.npy") if (out_dir / "layers.npy").exists() else None
    return summary, labels, layers


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_external(labels: np.ndarray, truth: np.ndarray, rng: np.random.Generator, n_boot: int, drop_noise: bool = False) -> dict[str, tuple[float, float, float]]:
    n = len(labels)
    values: dict[str, list[float]] = {"ami": [], "ari": [], "v_measure": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        metrics = external_metrics(labels[idx], truth[idx], drop_noise=drop_noise)
        for key, value in values.items():
            value.append(metrics[key])
    observed = external_metrics(labels, truth, drop_noise=drop_noise)
    return {
        key: (observed[key], float(np.quantile(values[key], 0.025)), float(np.quantile(values[key], 0.975)))
        for key in values
    }


def _bootstrap_diff(labels_a: np.ndarray, labels_b: np.ndarray, truth: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict[str, tuple[float, float, float, float]]:
    n = len(truth)
    values: dict[str, list[float]] = {"ami": [], "ari": [], "v_measure": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = external_metrics(labels_a[idx], truth[idx])
        b = external_metrics(labels_b[idx], truth[idx])
        for key, value in values.items():
            value.append(b[key] - a[key])
    observed_a = external_metrics(labels_a, truth)
    observed_b = external_metrics(labels_b, truth)
    return {
        key: (
            observed_b[key] - observed_a[key],
            float(np.quantile(values[key], 0.025)),
            float(np.quantile(values[key], 0.975)),
            float(min(1.0, 2 * min(np.mean(np.array(values[key]) <= 0), np.mean(np.array(values[key]) >= 0)))),
        )
        for key in values
    }


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (0.0, 0.0, 0.0)
    boots = np.array([np.mean(values[rng.integers(0, values.size, size=values.size)]) for _ in range(n_boot)])
    return (float(np.mean(values)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for rank, (key, value) in enumerate(ordered, start=1):
        running = max(running, min(1.0, (m - rank + 1) * value))
        adjusted[key] = running
    return adjusted


def _run_metrics(arms: list[str], bootstrap: int, permutations: int) -> None:
    truth = _load_truth()
    metric_rows = []
    layer_rows = []
    stability_rows = []
    sensitivity_rows = []
    run_labels: dict[str, np.ndarray] = {}
    run_summaries: dict[str, dict[str, Any]] = {}
    knn_per_point: dict[str, np.ndarray] = {}
    knn_neighbors: dict[str, np.ndarray] = {}

    for arm in arms:
        summary, labels, layers = _load_run(arm, "primary", "primary")
        if summary is None or labels is None or layers is None:
            print(f"missing primary run for {arm}; skipping metrics")
            continue
        run_summaries[arm] = summary
        run_labels[arm] = labels
        vectors = np.load(ARMS[arm]).astype(np.float32)
        partition = partition_summary(labels)
        point_purity = knn_purity_per_point(vectors, labels, k=KNN_K)
        knn_per_point[arm] = point_purity
        knn_neighbors[arm] = knn_neighbor_indices(vectors, k=KNN_K)
        row: dict[str, Any] = {
            "arm": arm,
            "clusters": partition.clusters,
            "noise": partition.noise,
            "noise_fraction": partition.noise_fraction,
            "min_size": partition.min_size,
            "median_size": partition.median_size,
            "p90_size": partition.p90_size,
            "max_size": partition.max_size,
            "largest_share": partition.largest_share,
            "knn_purity": float(np.nanmean(point_purity)),
            "duplicate_pairs": summary["duplicate_pairs"],
            "duplicate_rate": summary["duplicate_rate"],
            "primary_layer": summary.get("primary_layer"),
        }
        if truth is not None:
            external = external_metrics(labels, truth)
            external_no_noise = external_metrics(labels, truth, drop_noise=True)
            centroids = centroid_metrics(vectors, labels)
            row.update(
                {
                    "ami": external["ami"],
                    "ari": external["ari"],
                    "v_measure": external["v_measure"],
                    "homogeneity": external["homogeneity"],
                    "completeness": external["completeness"],
                    "purity": cluster_purity(labels, truth),
                    "ami_no_noise": external_no_noise["ami"],
                    "ari_no_noise": external_no_noise["ari"],
                    "v_measure_no_noise": external_no_noise["v_measure"],
                    **centroids,
                }
            )
        metric_rows.append(row)

        for layer_index in range(layers.shape[0]):
            layer = layers[layer_index]
            layer_summary = partition_summary(layer)
            layer_row = {
                "arm": arm,
                "layer": layer_index,
                "clusters": layer_summary.clusters,
                "noise_fraction": layer_summary.noise_fraction,
                "median_size": layer_summary.median_size,
                "largest_share": layer_summary.largest_share,
                "persistence": summary["layers"][layer_index]["persistence"],
                "mean_membership": summary["layers"][layer_index]["mean_membership"],
                "knn_purity": knn_purity(vectors, layer, k=KNN_K),
            }
            if truth is not None:
                layer_external = external_metrics(layer, truth)
                layer_row.update(
                    {
                        "ami": layer_external["ami"],
                        "ari": layer_external["ari"],
                        "v_measure": layer_external["v_measure"],
                    }
                )
            layer_rows.append(layer_row)

        for config_dir in sorted((RUNS_DIR / arm / "sensitivity").glob("*")):
            config_summary, config_labels, _ = _load_run(
                arm, "sensitivity", config_dir.name
            )
            if config_summary is None or config_labels is None:
                continue
            config_partition = partition_summary(config_labels)
            config_row = {
                "arm": arm,
                "config": config_dir.name,
                "clusters": config_partition.clusters,
                "noise_fraction": config_partition.noise_fraction,
                "knn_purity": knn_purity(vectors, config_labels, k=KNN_K),
            }
            if truth is not None:
                config_external = external_metrics(config_labels, truth)
                config_row.update(
                    {
                        "ami": config_external["ami"],
                        "ari": config_external["ari"],
                        "v_measure": config_external["v_measure"],
                    }
                )
            sensitivity_rows.append(config_row)

        seed_runs = []
        seed_cluster_counts = []
        for seed in SEEDS:
            if seed == PRIMARY_CONFIG["random_state"]:
                seed_runs.append(np.column_stack([np.arange(len(labels)), labels]))
                seed_cluster_counts.append(run_summaries[arm]["clusters"])
                continue
            seed_summary, seed_labels, _ = _load_run(arm, "seeds", f"seed_{seed}")
            if seed_labels is not None and seed_summary is not None:
                seed_runs.append(
                    np.column_stack([np.arange(len(seed_labels)), seed_labels])
                )
                seed_cluster_counts.append(seed_summary["clusters"])
        if seed_runs:
            stability_rows.append(
                {
                    "arm": arm,
                    "kind": "seed",
                    "runs": len(seed_runs),
                    "mean_pairwise_ari": mean_pairwise_ari(seed_runs),
                    "clusters_min": min(seed_cluster_counts),
                    "clusters_max": max(seed_cluster_counts),
                }
            )

        if arm in {"b2_minilm_pooled_overlap", "c_titan_v2"}:
            subset_runs = []
            cluster_counts = []
            for seed in SUBSAMPLE_SEEDS:
                out_dir = _run_dir(arm, "subsamples", f"seed_{seed}")
                if not (out_dir / "indices.npy").exists():
                    continue
                indices = np.load(out_dir / "indices.npy")
                sub_labels = np.load(out_dir / "labels.npy")
                subset_runs.append(np.column_stack([indices, sub_labels]))
                sub_summary = json.loads((out_dir / "summary.json").read_text())
                cluster_counts.append(sub_summary["clusters"])
            if subset_runs:
                stability_rows.append(
                    {
                        "arm": arm,
                        "kind": "subsample_80pct",
                        "runs": len(subset_runs),
                        "mean_pairwise_ari": mean_pairwise_ari(subset_runs),
                        "clusters_min": min(cluster_counts),
                        "clusters_max": max(cluster_counts),
                    }
                )

    _write_csv(DATA_DIR / "metrics.csv", metric_rows)
    _write_csv(DATA_DIR / "layer_metrics.csv", layer_rows)
    _write_csv(DATA_DIR / "stability.csv", stability_rows)
    _write_csv(DATA_DIR / "sensitivity_metrics.csv", sensitivity_rows)

    if len(run_labels) >= 2:
        pairwise = pairwise_ari(run_labels)
        _write_csv(
            DATA_DIR / "pairwise_ari.csv",
            [{"pair": key, "ari": value} for key, value in pairwise.items()],
        )

    if truth is None:
        print("labels unavailable or incomplete; wrote partition/internal metrics only")
        return

    rng = np.random.default_rng(20261030)
    bootstrap_rows = []
    for arm, labels in run_labels.items():
        for key, (estimate, lower, upper) in _bootstrap_external(labels, truth, rng, bootstrap).items():
            bootstrap_rows.append(
                {
                    "arm": arm,
                    "metric": key,
                    "contrast": "vs_label",
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "p_value": "",
                }
            )
        point_purity = knn_per_point[arm]
        estimate, lower, upper = _bootstrap_mean(point_purity, rng, bootstrap)
        bootstrap_rows.append(
            {
                "arm": arm,
                "metric": "knn_purity",
                "contrast": "vs_label",
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value": "",
            }
        )

    contrasts = {}
    if "b2_minilm_pooled_overlap" in run_labels and "a_minilm_truncated" in run_labels:
        contrasts["B2-A"] = ("a_minilm_truncated", "b2_minilm_pooled_overlap")
    if "c_titan_v2" in run_labels and "b2_minilm_pooled_overlap" in run_labels:
        contrasts["C-B2"] = ("b2_minilm_pooled_overlap", "c_titan_v2")
    p_values = {}
    for name, (left, right) in contrasts.items():
        for key, (estimate, lower, upper, p_value) in _bootstrap_diff(
            run_labels[left], run_labels[right], truth, rng, bootstrap
        ).items():
            bootstrap_rows.append(
                {
                    "arm": right,
                    "metric": key,
                    "contrast": name,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "p_value": p_value,
                }
            )
            if key == "ami":
                p_values[name] = p_value
    holm = _holm(p_values)
    for row in bootstrap_rows:
        if row["contrast"] in holm and row["metric"] == "ami":
            row["holm_p"] = holm[row["contrast"]]
        else:
            row["holm_p"] = ""
    _write_csv(DATA_DIR / "bootstrap_ci.csv", bootstrap_rows)

    permutation_rows = []
    for arm, labels in run_labels.items():
        observed = external_metrics(labels, truth)
        observed_knn = float(np.nanmean(knn_per_point[arm]))
        null_ami = []
        null_ari = []
        null_knn = []
        neighbors = knn_neighbors[arm]
        for _ in range(permutations):
            shuffled = truth[rng.permutation(len(truth))]
            null_ami.append(external_metrics(labels, shuffled)["ami"])
            null_ari.append(external_metrics(labels, shuffled)["ari"])
            null_knn.append(
                knn_purity_from_neighbors(neighbors, labels[rng.permutation(len(labels))])
            )
        for key, nulls, value in (
            ("ami", null_ami, observed["ami"]),
            ("ari", null_ari, observed["ari"]),
            ("knn_purity", null_knn, observed_knn),
        ):
            nulls = np.asarray(nulls)
            permutation_rows.append(
                {
                    "arm": arm,
                    "metric": key,
                    "observed": value,
                    "null_mean": float(np.mean(nulls)),
                    "null_sd": float(np.std(nulls)),
                    "p_value": float((1 + np.sum(nulls >= value)) / (1 + len(nulls))),
                    "z_score": float((value - np.mean(nulls)) / np.std(nulls)) if np.std(nulls) else 0.0,
                }
            )
    _write_csv(DATA_DIR / "permutation_null.csv", permutation_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["primary", "sensitivity", "stability", "metrics", "all"],
        default="all",
    )
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = parser.parse_args()

    valid_arms = [arm for arm in args.arms if arm in ARMS]
    if not valid_arms:
        raise SystemExit(f"no valid arms; choose from {sorted(ARMS)}")
    if args.stage in {"primary", "all"}:
        _run_primary(valid_arms, args.force)
    if args.stage in {"sensitivity", "all"}:
        _run_sensitivity(valid_arms, args.force)
    if args.stage in {"stability", "all"}:
        _run_stability(valid_arms, args.force)
    if args.stage in {"metrics", "all"}:
        _run_metrics(valid_arms, args.bootstrap, args.permutations)


if __name__ == "__main__":
    main()
