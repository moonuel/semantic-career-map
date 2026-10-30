"""Experiment 009 run harness: PCA informativeness benchmark on the production corpus.

Implements the preregistered design in ``docs/experiments/009-pca-informativeness.md``
(motivation, hypotheses, candidate grid, four metric families, decision rule).
Vectors come from ``semantic_career_map.postings_embeddings`` — 9,354 rows x 384
dims float32, one row per ``source_hash``, deterministically ordered — fetched
through the shared Athena runner exactly like ``scripts/map_umap_probe.py`` and
cached locally so every stage after the first fetch runs offline.

Stages (each recorded separately in the raw results JSON, so an interrupted run
resumes from ``--stage``):

* ``cache``     fetch or reload the production vectors, freeze the config header
* ``spectrum``  family A: thin SVD, EVR curve, participation ratio, scree knee, permutation noise floor
* ``retention`` family B: exact cosine 15/50-NN retention forward through the candidate grid
* ``fidelity``  family C: sampled-pair distance correlation and neighbor-vs-random contrast
* ``labels``    family D: golden-label separation gap (skipped cleanly when no bridge exists)
* ``summary``   pre-registered decision rule applied to the recorded metrics

All stochastic parts (permutation shuffles, pair sampling, random-other
sampling) draw from three reproducible seed substreams — 42, 43, 44 — so staged
runs are bit-reproducible independent of how many stages run in one process.
``U/S/Vt`` are recomputed deterministically whenever a downstream stage needs
them, so a resume never depends on in-memory state. Original-space ground truth
ranks are 1..g positions by construction (the ground-truth top-50 is sorted by
original similarity), so rank displacement in a reduced space measures against
those positions directly; strict-greater counting keeps identical tie handling
in every compared space.

Spike, not production: hardcoded parameters, no modules, no logging framework.
Frozen as an immutable artifact once the first corpus-wide run is recorded.

Usage:
    python scripts/009_pca_informativeness/run_benchmark.py --self-check
    python scripts/009_pca_informativeness/run_benchmark.py                 # all stages
    python scripts/009_pca_informativeness/run_benchmark.py --stage retention
    python scripts/009_pca_informativeness/run_benchmark.py --refresh-cache
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

STAGES = ("cache", "spectrum", "retention", "fidelity", "labels", "summary")

CONFIGURATION_HASH = "a29a35e37a0d7025a961b5241ff278904500ed43edf022ac7304dacdb97e0c98"
DATABASE = "semantic_career_map"
WORKGROUP = "semantic_career_map"
FULL_ROWS = 9354
EXPECTED_DIMS = 384

K_GRID: tuple[int, ...] = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384)
KNN_PRIMARY = 15
KNN_SECONDARY = 50
JACCARD_THRESHOLD = 0.95
PAIR_COUNT = 200_000
RANDOM_OTHERS = 94
PERMUTATION_REPLICAS = 10
SEED_PERMUTATION = 42
SEED_PAIRS = 43
SEED_RANDOM_OTHERS = 44
BLOCK_ROWS = 2048
RANK_CHUNK = 256
TRACKED_NEIGHBOURS = KNN_SECONDARY

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_RAW = DEFAULT_DATA_DIR / "009_raw_results.json"
DEFAULT_SUMMARY = Path("data/009_pca_benchmark.json")

_VECTORS_SQL = f"""
SELECT source_hash, embedding
FROM {DATABASE}.postings_embeddings
WHERE configuration_hash = '{CONFIGURATION_HASH}'
ORDER BY source_hash
LIMIT {FULL_ROWS}
"""


def _athena():
    """Build an Athena client pinned to the lake bucket's region (repo rule)."""
    import boto3
    from botocore.config import Config

    from backend.pipeline.athena import aws_region

    return boto3.Session().client(
        "athena",
        region_name=aws_region(),
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def _run(athena, sql: str) -> list[list[str]]:
    """Submit, poll, and drain an Athena query to a row list (header first)."""
    from backend.pipeline.athena import (
        iter_query_results,
        poll_query_completion,
        results_location,
        submit_query,
    )

    execution_id = submit_query(
        athena,
        sql,
        database=DATABASE,
        workgroup=WORKGROUP,
        output_location=results_location(),
    )
    poll_query_completion(athena, execution_id)
    return list(iter_query_results(athena, execution_id))


def fetch_vectors() -> tuple[list[str], np.ndarray]:
    """Fetch the full production vector set as ``(hashes, float32 matrix)``.

    The query orders by ``source_hash`` for stable row order, and the result
    must satisfy the proven vector contract exactly: all 9,354 rows, 384 dims
    each, no duplicate hashes, strictly ascending key order.

    Returns:
        Keys in query order and a ``(n, 384)`` float32 matrix whose row order
        matches the keys.

    Raises:
        SystemExit: when the result violates the contract.
    """
    rows = _run(_athena(), _VECTORS_SQL)
    body = rows[1:]
    if len(body) != FULL_ROWS:
        raise SystemExit(f"expected {FULL_ROWS} rows, got {len(body)}")

    keys: list[str] = []
    vectors = np.empty((len(body), EXPECTED_DIMS), dtype=np.float32)
    for i, row in enumerate(body):
        dims = json.loads(row[1])
        if len(dims) != EXPECTED_DIMS:
            raise SystemExit(f"row {i} has {len(dims)} dims, expected {EXPECTED_DIMS}")
        keys.append(row[0])
        vectors[i] = np.asarray(dims, dtype=np.float32)

    if len(set(keys)) != len(keys):
        raise SystemExit(f"{len(keys) - len(set(keys))} duplicate source_hash rows")
    if keys != sorted(keys):
        raise SystemExit("row order is not ascending source_hash")
    return keys, vectors


def load_vectors(data_dir: Path, refresh: bool) -> tuple[list[str], np.ndarray, dict]:
    """Load the cached corpus, fetching from Athena only when needed.

    Args:
        data_dir: Directory holding ``vectors_cache.npy`` and ``source_hashes.json``.
        refresh: Refetch from Athena even when a valid cache exists.

    Returns:
        Keys, the float32 matrix, and fetch metadata for the raw results.

    Raises:
        SystemExit: when the cache loads empty or the contract fails.
    """
    cache_npy = data_dir / "vectors_cache.npy"
    cache_hashes = data_dir / "source_hashes.json"
    fetched = False
    started = time.monotonic()
    if refresh or not (cache_npy.exists() and cache_hashes.exists()):
        data_dir.mkdir(parents=True, exist_ok=True)
        keys, vectors = fetch_vectors()
        np.save(cache_npy, vectors)
        cache_hashes.write_text(json.dumps(keys), encoding="utf-8")
        fetched = True
    else:
        keys = list(json.loads(cache_hashes.read_text(encoding="utf-8")))
        vectors = np.load(cache_npy)
        if vectors.shape != (len(keys), EXPECTED_DIMS):
            raise SystemExit(
                f"cache shape {vectors.shape} disagrees with {len(keys)} keys"
            )
    meta = {
        "rows": int(vectors.shape[0]),
        "dims": int(vectors.shape[1]),
        "dtype": str(vectors.dtype),
        "fetched_this_run": fetched,
        "seconds": round(time.monotonic() - started, 2),
    }
    return keys, vectors, meta


def git_commit() -> str:
    """Return the current HEAD sha, or ``unknown`` when git is unavailable."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except OSError, subprocess.SubprocessError:
        return "unknown"


def build_config() -> dict:
    """Return the frozen config header for the preregistered design."""
    return {
        "git_commit": git_commit(),
        "frozen_note": "run timestamps live in the cache stage payload, not here",
        "configuration_hash": CONFIGURATION_HASH,
        "full_rows": FULL_ROWS,
        "dims": EXPECTED_DIMS,
        "knn_primary": KNN_PRIMARY,
        "knn_secondary": KNN_SECONDARY,
        "k_grid": list(K_GRID),
        "jaccard_threshold": JACCARD_THRESHOLD,
        "pair_count": PAIR_COUNT,
        "random_others": RANDOM_OTHERS,
        "permutation_replicas": PERMUTATION_REPLICAS,
        "seeds": {
            "permutation": SEED_PERMUTATION,
            "pairs": SEED_PAIRS,
            "random_others": SEED_RANDOM_OTHERS,
        },
        "block_rows": BLOCK_ROWS,
        "tracked_neighbours": TRACKED_NEIGHBOURS,
    }


def freeze_config(data_dir: Path) -> dict:
    """Write the config header once; on resume, verify it byte-for-byte.

    Args:
        data_dir: Directory holding the frozen ``009_config.json``.

    Returns:
        The verified config header.

    Raises:
        SystemExit: when an existing header drifts from the frozen design.
    """
    target = data_dir / "009_config.json"
    config = build_config()
    payload = json.dumps(config, indent=2) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != payload:
            raise SystemExit("009_config.json drifted from the frozen header; aborting")
    else:
        data_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
    return config


def record_stage(raw_path: Path, stage: str, payload: dict) -> None:
    """Merge one stage payload into the raw results JSON (read-modify-write)."""
    raw = {}
    if raw_path.exists():
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw[stage] = payload
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(f"stage={stage} recorded -> {raw_path}")


def row_normalize(x: np.ndarray) -> np.ndarray:
    """Scale ``x`` to unit L2 rows; zero rows pass through unchanged."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def knee_index(evr: np.ndarray) -> int:
    """Index of the scree knee by maximum chord distance on the log-EVR curve."""
    y = np.log(np.maximum(evr, 1e-300))
    t = np.linspace(0.0, 1.0, len(y))
    line = y[0] + (y[-1] - y[0]) * t
    return int(np.argmax(np.abs(y - line)))


def run_spectrum(
    x64: np.ndarray,
) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Family A: exact thin-SVD spectrum with a permutation noise floor.

    Args:
        x64: Full ``(n, d)`` float64 corpus matrix (uncentered input).

    Returns:
        The spectrum payload for the raw results and the ``(U, S, Vt)`` triplet
        every candidate truncation slices from.
    """
    n, d = x64.shape
    started = time.monotonic()
    centered = x64 - x64.mean(axis=0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    svd_seconds = time.monotonic() - started

    started = time.monotonic()
    lam = s**2
    evr = lam / lam.sum()
    permuted = np.empty((PERMUTATION_REPLICAS, d))
    rng = np.random.default_rng(SEED_PERMUTATION)
    for r in range(PERMUTATION_REPLICAS):
        xp = x64.copy()
        for col in np.arange(d):
            xp[:, col] = x64[rng.permutation(n), col]
        lam_p = (
            np.linalg.svd(xp - xp.mean(axis=0), full_matrices=False, compute_uv=False)
            ** 2
        )
        permuted[r] = lam_p / lam_p.sum()
    perm_seconds = time.monotonic() - started

    payload = {
        "singular_values": [float(v) for v in s],
        "evr": [float(v) for v in evr],
        "cumulative_evr": [float(v) for v in np.cumsum(evr)],
        "participation_ratio": float((lam.sum() ** 2) / (lam**2).sum()),
        "knee_dims": knee_index(evr) + 1,
        "permuted_evr_mean": [float(v) for v in permuted.mean(axis=0)],
        "signal_count": int(np.sum(evr > permuted.mean(axis=0))),
        "svd_seconds": round(svd_seconds, 2),
        "permutation_seconds": round(perm_seconds, 2),
    }
    return payload, (u, s, vt)


def strict_greater_ranks(sim_rows: np.ndarray, ref_sims: np.ndarray) -> np.ndarray:
    """1-based rank of each tracked neighbour inside full similarity rows.

    Rank counts strictly-greater similarities; float32 cosine ties break
    identically in every compared space, so retention and displacement reads
    stay internally consistent.

    Args:
        sim_rows: ``(b, n)`` similarity rows against the whole space.
        ref_sims: ``(b, g)`` similarities of the tracked neighbours.

    Returns:
        ``(b, g)`` int64 ranks starting at 1.
    """
    b, g = ref_sims.shape
    ranks = np.empty((b, g), dtype=np.int64)
    for start in range(0, b, RANK_CHUNK):
        stop = min(start + RANK_CHUNK, b)
        ranks[start:stop] = (
            sim_rows[start:stop][:, None, :] > ref_sims[start:stop][:, :, None]
        ).sum(axis=2)
    return ranks + 1


def shape_tracker(n: int) -> np.ndarray:
    """Dummy tracked-index matrix with the ground-truth shape.

    The original-space pass discovers its own top-50 ordering, so the tracked
    content only has to be a valid index matrix for that first pass; every
    metric of record uses the returned ground truth afterward.

    Args:
        n: Number of points.

    Returns:
        ``(n, TRACKED_NEIGHBOURS)`` int64 array of shifted row indices.
    """
    base = np.arange(n)[:, None]
    shifts = np.arange(1, TRACKED_NEIGHBOURS + 1)
    return ((base + shifts) % n).astype(np.int64)


def space_pass(
    xn: np.ndarray, tracked_idx: np.ndarray, rand_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Exact-cosine pass over one space.

    The query point is excluded from its own similarity row, so every tracked
    neighbour and candidate selection is a genuine other-point neighbour.
    Per point: the top-``g`` neighbours sorted by similarity (descending), 1-based
    ranks of the tracked neighbours in the full ordering, the mean top-15
    similarity, and the mean similarity to the tracked random others.

    Args:
        xn: Row-normalized ``(n, d)`` matrix for the space.
        tracked_idx: ``(n, g)`` tracked neighbour indices (ground truth).
        rand_idx: ``(n, r)`` tracked random-other indices.

    Returns:
        Top-``g`` index matrix, 1-based rank matrix, mean top-15 similarity,
        mean random-other similarity.
    """
    n = xn.shape[0]
    topk = tracked_idx.shape[1]
    out_top = np.empty((n, topk), dtype=np.int64)
    out_ranks = np.empty((n, topk), dtype=np.int64)
    sum15 = 0.0
    sum_rand = 0.0
    for start in range(0, n, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, n)
        sims = xn[start:stop] @ xn.T
        sims[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        tracked_sims = np.take_along_axis(sims, tracked_idx[start:stop], axis=1)
        part = np.argpartition(-sims, topk - 1, axis=1)[:, :topk]
        part_sims = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_sims, axis=1)
        out_top[start:stop] = np.take_along_axis(part, order, axis=1)
        out_ranks[start:stop] = strict_greater_ranks(sims, tracked_sims)
        sum15 += float(
            np.take_along_axis(sims, out_top[start:stop][:, :KNN_PRIMARY], axis=1).sum()
        )
        sum_rand += float(np.take_along_axis(sims, rand_idx[start:stop], axis=1).sum())
    return (
        out_top,
        out_ranks,
        sum15 / (n * KNN_PRIMARY),
        sum_rand / (n * rand_idx.shape[1]),
    )


def jaccard_per_row(a_idx: np.ndarray, b_idx: np.ndarray) -> np.ndarray:
    """Per-row Jaccard similarity of two equal-width index selections."""
    g = a_idx.shape[1]
    inter = (a_idx[:, :, None] == b_idx[:, None, :]).sum(axis=(1, 2))
    return inter.astype(np.float64) / (2 * g - inter)


def truncation(usv: tuple, k: int) -> np.ndarray:
    """Return the ``k``-dim PCA truncation ``U[:, :k] * S[:k] @ Vt[:k]`` (float32)."""
    u, s, vt = usv
    return ((u[:, :k] * s[:k]) @ vt[:k]).astype(np.float32)


def make_rand_and_pairs(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the tracked random-other matrix and the fidelity pair sample.

    Fixed seed substreams guarantee identical indices on every call and across
    processes, so all cross-space reads are paired.

    Args:
        n: Number of points.

    Returns:
        ``(rand_idx, pair_i, pair_j)`` with ``rand_idx`` shaped ``(n,
        RANDOM_OTHERS)`` and the pair arrays shaped ``(PAIR_COUNT,)`` where
        ``pair_i != pair_j`` always.
    """
    rand_rng = np.random.default_rng(SEED_RANDOM_OTHERS)
    rand_idx = rand_rng.integers(0, n, size=(n, RANDOM_OTHERS))
    row_this = np.arange(n)[:, None]
    bump = (row_this + np.arange(1, RANDOM_OTHERS + 1)[None, :]) % n
    rand_idx = np.where(rand_idx == row_this, bump, rand_idx)
    rand_idx = np.where(rand_idx == row_this, (row_this + RANDOM_OTHERS) % n, rand_idx)

    pair_rng = np.random.default_rng(SEED_PAIRS)
    pair_i = pair_rng.integers(0, n, size=PAIR_COUNT)
    pair_j = (pair_i + pair_rng.integers(1, n, size=PAIR_COUNT)) % n
    return rand_idx, pair_i, pair_j


def run_retention(x32: np.ndarray, usv: tuple) -> dict:
    """Family B: exact cosine 15/50-NN retention forward through the grid.

    Computes the original-space ground truth (top-50 neighbours per point,
    sorted by original similarity, so their true ranks are 1..50 positions),
    then re-passes the same points in every ``k_d`` truncation.

    Args:
        x32: Full ``(n, 384)`` float32 corpus matrix.
        usv: ``(U, S, Vt)`` triplet from the exact spectrum pass.

    Returns:
        Payload with one metrics row per candidate ``k_d`` plus the per-space
        mean similarities the fidelity stage needs as contrast inputs.
    """
    n = x32.shape[0]
    rand_idx, _, _ = make_rand_and_pairs(n)
    started = time.monotonic()

    xn = row_normalize(x32)
    gt_start = time.monotonic()
    gt_top, _, mean15_orig, rand_mean_orig = space_pass(xn, shape_tracker(n), rand_idx)
    pass_seconds = time.monotonic() - gt_start
    positions = np.arange(1, TRACKED_NEIGHBOURS + 1)[None, :].astype(np.int64)

    gt15 = gt_top[:, :KNN_PRIMARY]
    space_means: dict[str, dict] = {
        "original": {"mean_sims15": mean15_orig, "mean_rand": rand_mean_orig}
    }
    print(
        f"retention: original exact pass done ({pass_seconds:.1f}s), "
        f"mean15={mean15_orig:.4f} rand={rand_mean_orig:.4f}"
    )

    rows: list[dict] = []
    for k_d in K_GRID:
        k_start = time.monotonic()
        xkn = row_normalize(truncation(usv, k_d))
        red_top, red_ranks, mean15_k, rand_mean_k = space_pass(xkn, gt_top, rand_idx)
        pass_seconds += time.monotonic() - k_start
        jac15 = float(jaccard_per_row(gt15, red_top[:, :KNN_PRIMARY]).mean())
        jac50 = float(jaccard_per_row(gt_top, red_top).mean())
        nn1 = float((red_top[:, 0] == gt_top[:, 0]).mean())
        disp15 = red_ranks[:, :KNN_PRIMARY] - positions[:, :KNN_PRIMARY]
        rows.append(
            {
                "k_d": k_d,
                "jaccard_15": round(jac15, 6),
                "jaccard_50": round(jac50, 6),
                "nn1_rate": round(nn1, 6),
                "mean_rank_disp_15": round(float(disp15.mean()), 6),
                "max_rank_disp_15": int(disp15.max()),
            }
        )
        space_means[str(k_d)] = {"mean_sims15": mean15_k, "mean_rand": rand_mean_k}
        print(f"retention: k_d={k_d} jaccard15={jac15:.4f} nn1={nn1:.4f}")

    return {
        "grid": list(K_GRID),
        "rows": rows,
        "space_means": space_means,
        "tracked_neighbours": TRACKED_NEIGHBOURS,
        "pass_seconds": round(pass_seconds, 2),
        "total_seconds": round(time.monotonic() - started, 2),
    }


def run_fidelity(x32: np.ndarray, usv: tuple, space_means: dict) -> dict:
    """Family C: sampled-pair distance fidelity and neighbour-vs-random contrast.

    Args:
        x32: Full ``(n, 384)`` float32 corpus matrix.
        usv: ``(U, S, Vt)`` triplet from the exact spectrum pass.
        space_means: Per-space ``mean_sims15``/``mean_rand`` means from family B.

    Returns:
        Payload with sampled-pair cosine statistics per candidate ``k_d`` and
        the original-space contrast used as the survival baseline.
    """
    _, pair_i, pair_j = make_rand_and_pairs(x32.shape[0])
    started = time.monotonic()
    xn = row_normalize(x32)
    cos_orig = np.sum(xn[pair_i] * xn[pair_j], axis=1).astype(np.float64)
    ct_orig = (
        space_means["original"]["mean_sims15"] / space_means["original"]["mean_rand"]
    )

    rows: list[dict] = []
    for k_d in K_GRID:
        xkn = row_normalize(truncation(usv, k_d))
        cos_k = np.sum(xkn[pair_i] * xkn[pair_j], axis=1).astype(np.float64)
        pear = float(stats.pearsonr(cos_orig, cos_k).statistic)
        spear = float(stats.spearmanr(cos_orig, cos_k).statistic)
        mae = float(np.mean(np.abs(cos_orig - cos_k)))
        ct_k = space_means[str(k_d)]["mean_sims15"] / space_means[str(k_d)]["mean_rand"]
        rows.append(
            {
                "k_d": k_d,
                "pearson_r": round(pear, 6),
                "spearman_r": round(spear, 6),
                "mae": round(mae, 8),
                "contrast_ratio": round(ct_k, 8),
                "contrast_survival": round(ct_k / ct_orig, 6),
            }
        )
        print(f"fidelity: k_d={k_d} pearson={pear:.4f} spearman={spear:.4f}")
    return {
        "rows": rows,
        "contrast_original": round(ct_orig, 8),
        "total_seconds": round(time.monotonic() - started, 2),
    }


def run_labels() -> dict:
    """Family D: golden-label separation gap, skipped when no bridge exists.

    The golden 27 rows key by slug ``posting_id``; local data carries no
    slug-to-64-hex ``source_hash`` bridge (sample manifests key the lake by the
    ``sha256`` of source files; ``data/jobs.json`` predates hashing). The join
    therefore resolves to 0 labelled corpus rows — far below the preregistered
    20-point minimum — and skips cleanly.
    """
    return {
        "status": "skipped",
        "reason": (
            "no posting_id -> source_hash bridge in local data "
            "(golden slugs vs 64-hex lake hashes); preregistered <20 join -> skip"
        ),
    }


def apply_decision(retention_rows: list[dict], signal_count: int) -> dict:
    """Pre-registered decision rule applied to the recorded retention rows.

    A candidate ``k_d`` gates when its mean Jaccard@15 clears the threshold AND
    it is at least the permutation-informed signal count; ``k_d*`` is the
    smallest gated candidate, or ``None`` when nothing on the grid clears.

    Args:
        retention_rows: Family-B rows, one per ``k_d``, in grid order.
        signal_count: Permutation-informed signal dimension count.

    Returns:
        Decision payload with ``k_d_star`` (or None) and per-candidate gates.
    """
    gates = {
        row["k_d"]: bool(
            row["jaccard_15"] >= JACCARD_THRESHOLD and row["k_d"] >= signal_count
        )
        for row in retention_rows
    }
    gated = [k for k in K_GRID if gates.get(k)]
    return {
        "k_d_star": min(gated) if gated else None,
        "signal_count": signal_count,
        "jaccard_threshold": JACCARD_THRESHOLD,
        "gates": gates,
    }


def build_summary(config: dict, raw: dict, contrast_orig: float) -> dict:
    """Assemble the canonical summary from the recorded stage payloads.

    Args:
        config: The frozen config header.
        raw: Every stage payload recorded so far.
        contrast_orig: Original-space neighbour-vs-random contrast (family C).

    Returns:
        The canonical summary payload written beside the repo's ``data/`` dir.
    """
    spectrum, retention, fidelity, labels = (
        raw["spectrum"],
        raw["retention"],
        raw["fidelity"],
        raw["labels"],
    )
    by_k = {row["k_d"]: row for row in fidelity["rows"]}
    joined = [{**row, **by_k[row["k_d"]]} for row in retention["rows"]]
    decision = apply_decision(retention["rows"], spectrum["signal_count"])
    return {
        "experiment": "009",
        "design_doc": "docs/experiments/009-pca-informativeness.md",
        "config": config,
        "spectrum": {
            "participation_ratio": spectrum["participation_ratio"],
            "knee_dims": spectrum["knee_dims"],
            "signal_count": spectrum["signal_count"],
            "cumulative_evr_at_grid": {
                str(k_d): spectrum["cumulative_evr"][k_d - 1] for k_d in K_GRID
            },
        },
        "retention": joined,
        "fidelity": fidelity["rows"],
        "contrast_original": contrast_orig,
        "labels": labels,
        "decision": decision,
        "runtimes": {
            "cache_s": raw["cache"]["seconds"],
            "svd_s": spectrum["svd_seconds"],
            "permutation_s": spectrum["permutation_seconds"],
            "retention_s": retention["total_seconds"],
            "fidelity_s": fidelity["total_seconds"],
        },
    }


def self_check() -> None:
    """Single-responsibility checks of every metric kernel on synthetic data."""
    rng = np.random.default_rng(7)

    x = rng.normal(size=(24, 8))
    centered = x - x.mean(axis=0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    assert np.allclose(truncation((u, s, vt), 8), centered, atol=2e-6), (
        "k=d rebuilds the centered matrix"
    )
    evr = (s**2) / (s**2).sum()
    assert abs(np.cumsum(evr)[-1] - 1.0) < 1e-9, "cumEVR[-1] must equal 1"
    pr = float(((s**2).sum() ** 2) / ((s**2) ** 2).sum())
    assert 1.0 <= pr <= 8.0, "participation ratio must sit in [1, d]"

    n, d = 40, 8
    xb = rng.normal(size=(n, d))
    xbn = row_normalize(xb)
    tracked = ((np.arange(n)[:, None] + np.arange(1, 11)) % n).astype(np.int64)
    rand_idx = ((np.arange(n)[:, None] + np.arange(1, 6)) % n).astype(np.int64)
    top, ranks, mean15, rand_mean = space_pass(xbn, tracked, rand_idx)
    for i in range(n):
        sims = xbn @ xbn[i]
        sims[i] = -np.inf
        brute = np.argsort(-sims, kind="stable")[:10]
        assert list(brute) == list(top[i]), f"top-k mismatch at row {i}"
        brute_ranks = 1 + np.sum(sims[:, None] > sims[tracked[i]][None, :], axis=0)
        assert np.array_equal(np.asarray(brute_ranks), ranks[i]), (
            f"rank mismatch row {i}"
        )
    assert -1.0 <= mean15 <= 1.0 and -1.0 <= rand_mean <= 1.0, "cosine means in range"

    same = np.arange(10)[None, :].repeat(5, axis=0)
    assert np.allclose(jaccard_per_row(same, same), 1.0), "jaccard(A,A) must be 1"
    assert np.allclose(jaccard_per_row(same, same[:, ::-1]), 1.0), "order cannot matter"
    left = np.tile(np.arange(5)[None, :], (2, 1))
    right = np.tile(np.arange(5, 10)[None, :], (2, 1))
    assert np.allclose(jaccard_per_row(left, right), 0.0), "disjoint scores 0"

    steplike = np.array([0.30, 0.25, 0.20, 0.02, 0.018, 0.015, 0.012, 0.01])
    assert knee_index(steplike) <= 3, "knee must land near the cliff"
    linear = np.linspace(0.2, 0.05, 8)
    assert 0 <= knee_index(linear) < len(linear), "knee needs a valid index"

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    assert abs(stats.spearmanr(a, b).statistic - 1.0) < 1e-12, "monotone pairs"
    assert abs(stats.spearmanr(a, -b).statistic + 1.0) < 1e-12, "inverse pairs"

    rows = [
        {"k_d": 24, "jaccard_15": 0.97},
        {"k_d": 32, "jaccard_15": 0.949},
        {"k_d": 48, "jaccard_15": 0.96},
    ]
    decision = apply_decision(rows, signal_count=32)
    assert decision["gates"] == {24: False, 32: False, 48: True}, "gate logic"
    assert decision["k_d_star"] == 48, "smallest gated candidate wins"
    assert apply_decision(rows, signal_count=64)["k_d_star"] is None, "no gate -> None"

    noisy = np.tile([0.3, 0.02, 0.001, 0.0009, 0.001, 0.0008, 0.001, 0.0009], (30, 1))
    noisy = noisy + rng.normal(scale=1e-5, size=(30, 8))
    payload, _ = run_spectrum(noisy)
    assert 1 <= payload["signal_count"] <= 4, (
        f"signal count sane: {payload['signal_count']}"
    )

    print("self-check: all metric kernels passed")


def main(argv: list[str] | None = None) -> int:
    """Freeze config, run requested stages, and write raw + canonical output.

    Args:
        argv: CLI arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 unless a stage raises).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=STAGES, default="cache", help="run from this stage onward"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--refresh-cache", action="store_true", help="refetch the corpus"
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="verify metric kernels on synthetic data",
    )
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        return 0

    from dotenv import load_dotenv

    load_dotenv()
    config = freeze_config(args.data_dir)
    raw_path = args.raw
    raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {}
    stage_idx = STAGES.index(args.stage)
    stage_set = STAGES[stage_idx:]

    total_started = time.monotonic()
    if "cache" in stage_set:
        keys, vectors, meta = load_vectors(args.data_dir, args.refresh_cache)
        record_stage(raw_path, "cache", {**meta, "git_commit": config["git_commit"]})
        raw["cache"] = meta
    else:
        keys, vectors, meta = load_vectors(args.data_dir, refresh=False)
        raw.setdefault("cache", meta)
    print(
        f"vectors={len(keys)}x{meta['dims']} fetched_this_run={meta['fetched_this_run']}"
    )

    x32 = vectors
    x64 = vectors.astype(np.float64)

    need_usv = any(s in stage_set for s in ("retention", "fidelity", "summary"))
    if "spectrum" in stage_set:
        spectrum_payload, usv = run_spectrum(x64)
        record_stage(raw_path, "spectrum", spectrum_payload)
        raw["spectrum"] = spectrum_payload
    elif need_usv:
        if "spectrum" not in raw:
            raise SystemExit("spectrum not recorded in raw results; run it first")
        spectrum_payload = raw["spectrum"]
        _, usv = run_spectrum(x64)

    if "retention" in stage_set:
        payload = run_retention(x32, usv)
        record_stage(raw_path, "retention", payload)
        raw["retention"] = payload

    if "fidelity" in stage_set:
        if "retention" not in raw:
            raise SystemExit("retention not recorded in raw results; run it first")
        payload = run_fidelity(x32, usv, raw["retention"]["space_means"])
        record_stage(raw_path, "fidelity", payload)
        raw["fidelity"] = payload

    if "labels" in stage_set:
        payload = run_labels()
        record_stage(raw_path, "labels", payload)
        raw["labels"] = payload

    if "summary" in stage_set:
        if "fidelity" not in raw:
            raise SystemExit("fidelity not recorded in raw results; run it first")
        summary = build_summary(config, raw, raw["fidelity"]["contrast_original"])
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        record_stage(raw_path, "summary", {"k_d_star": summary["decision"]["k_d_star"]})
        print(f"canonical summary -> {args.summary}")
        print(
            f"decision: k_d*={summary['decision']['k_d_star']} "
            f"signal_count={summary['spectrum']['signal_count']}"
        )

    print(f"total wall time: {time.monotonic() - total_started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
