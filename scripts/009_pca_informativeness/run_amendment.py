"""Experiment 009A amendment harness: frame-corrected PCA pre-reduction gate.

Preregistered amendment to the recorded 009 run (see the "Experiment 009A"
section of ``docs/experiments/009-pca-informativeness.md``). The 009 gate
measured centered-PCA truncations against the *sphere-frame* cosine ground
truth, and its k_d=384 control proved the resulting ceiling (Jaccard 0.5752
self-inclusive, 0.5537 self-excluded) is the centering frame shift, not information loss. 009A measures the same
truncations inside the frame the deployment actually lives in:

* **Ground truth (arm A target):** exact cosine 15-NN of the *centered*
  full-rank space, re-normalized onto the sphere — the deployment frame.
* **Arm A (primary):** centered PCA top-``k_d`` basis coordinates
  ``rn(U_k S_k)`` — the coherent in-frame pipeline (truncate, renormalize,
  cosine).
* **Arm B (reference):** truncated reconstruction ``rn(U_k S_k Vt_k + mu)``
  scored against the *sphere-frame* ground truth — quantifies what in-frame
  deployment gives up relative to the original geometry.

Fully deterministic: exact matmul k-NN in both frames, no stochastic
component. Gate (inherited verbatim from 009, preregistered): smallest
``k_d`` with arm-A Jaccard@15 >= 0.95 AND ``k_d`` >= 009's recorded signal
count (71). Exploratory check with a self-excluded harness: 256 clears
(0.9529), 192 misses (0.8879); 009A confirms or refutes on the record.

Self-contained: constants re-declared, nothing imported from
``run_benchmark.py`` (no-cross-experiment-imports rule). Frozen as an
immutable artifact once its run is recorded.

Usage:
    python scripts/009_pca_informativeness/run_amendment.py --self-check
    python scripts/009_pca_informativeness/run_amendment.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_RAW = DEFAULT_DATA_DIR / "009_raw_results.json"
DEFAULT_SUMMARY = Path("data/009_pca_benchmark.json")

K_GRID: tuple[int, ...] = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384)
KNN = 15
KNN_SECONDARY = 50
JACCARD_THRESHOLD = 0.95
SIGNAL_COUNT_009 = 71
BLOCK_ROWS = 2048

FRAME_AGREEMENT_009 = 0.5537
FRAME_AGREEMENT_TOLERANCE = 0.02
SPHERE_KD384_JACCARD_MIN = 0.99


def topk_idx(a: np.ndarray, k: int) -> np.ndarray:
    """Exact cosine top-``k`` neighbour indices per row, sorted best-first.

    The query point is excluded from its own row before selection.

    Args:
        a: Row-normalized ``(n, d)`` matrix.
        k: Neighbourhood width.

    Returns:
        ``(n, k)`` int64 neighbour indices, descending similarity.
    """
    n = len(a)
    out = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, n)
        sims = a[start:stop] @ a.T
        sims[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        part_sims = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_sims, axis=1)
        out[start:stop] = np.take_along_axis(part, order, axis=1)
    return out


def row_normalize(x: np.ndarray) -> np.ndarray:
    """Scale ``x`` to unit L2 rows; zero rows pass through unchanged."""
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def jaccard_per_row(a_idx: np.ndarray, b_idx: np.ndarray) -> np.ndarray:
    """Per-row Jaccard similarity of two index selections.

    The union denominator uses the two selection widths, so comparing a
    top-15 ground truth with a top-50 candidate is valid.
    """
    g_a = a_idx.shape[1]
    g_b = b_idx.shape[1]
    inter = (a_idx[:, :, None] == b_idx[:, None, :]).sum(axis=(1, 2))
    return inter.astype(np.float64) / (g_a + g_b - inter)


def arm_row(gt15: np.ndarray, gt50: np.ndarray, cand: np.ndarray) -> dict:
    """Compute the inherited 009 retention metrics for one arm/k_d pair.

    Args:
        gt15: Ground-truth top-15 neighbour indices ``(n, KNN)``.
        gt50: Ground-truth top-50 neighbour indices ``(n, 50)``.
        cand: Candidate-space top-50 indices; the top-15 slice carries the
            gate metrics.

    Returns:
        Metrics row: Jaccard@15 (gate), Jaccard@50, 1-NN rate.
    """
    top50 = cand
    jac15 = float(jaccard_per_row(gt15, top50[:, :KNN]).mean())
    jac50 = float(jaccard_per_row(gt50, top50).mean())
    nn1 = float((top50[:, 0] == gt15[:, 0]).mean())
    return {
        "jaccard_15": round(jac15, 6),
        "jaccard_50": round(jac50, 6),
        "nn1_rate": round(nn1, 6),
    }


def apply_decision(rows: list[dict]) -> dict:
    """Preregistered gate: smallest k_d clearing Jaccard@15 >= 0.95 and k_d >= 71.

    Args:
        rows: Arm-A metrics rows, one per candidate ``k_d``, in grid order.

    Returns:
        Decision payload with ``k_d_star`` (or None) and per-candidate gates.
    """
    gates = {
        row["k_d"]: bool(
            row["jaccard_15"] >= JACCARD_THRESHOLD and row["k_d"] >= SIGNAL_COUNT_009
        )
        for row in rows
    }
    gated = [k for k in K_GRID if gates.get(k)]
    return {
        "k_d_star": min(gated) if gated else None,
        "signal_count_009": SIGNAL_COUNT_009,
        "jaccard_threshold": JACCARD_THRESHOLD,
        "gates": gates,
    }


def self_check() -> None:
    """Synthetic-geometry checks of every kernel before any corpus work."""
    rng = np.random.default_rng(11)

    # topk_idx: brute-force agreement on a small random cloud
    x = row_normalize(rng.normal(size=(30, 6)))
    top = topk_idx(x, 7)
    for i in (0, 7, 29):
        sims = x @ x[i]
        sims[i] = -np.inf
        brute = np.argsort(-sims, kind="stable")[:7]
        assert list(brute) == list(top[i]), f"topk mismatch row {i}"

    # jaccard_per_row: identity, order, disjoint, unequal widths
    same = np.arange(5)[None, :].repeat(3, axis=0)
    assert np.allclose(jaccard_per_row(same, same), 1.0), "jaccard(A,A) = 1"
    assert np.allclose(jaccard_per_row(same, same[:, ::-1]), 1.0), "order irrelevant"
    left = np.tile(np.arange(3)[None, :], (2, 1))
    right = np.tile(np.arange(3, 6)[None, :], (2, 1))
    assert np.allclose(jaccard_per_row(left, right), 0.0), "disjoint = 0"
    wide = np.arange(10)[None, :].repeat(3, axis=0)
    assert np.allclose(jaccard_per_row(same, wide), 5.0 / 10.0), (
        "unequal-width union denominator"
    )

    # apply_decision: gate semantics incl. the >= signal-count clause
    rows = [
        {"k_d": 64, "jaccard_15": 0.97},
        {"k_d": 96, "jaccard_15": 0.96},
    ]
    dec = apply_decision(rows)
    assert dec["gates"] == {64: False, 96: True}, "signal-count clause must bind"
    assert dec["k_d_star"] == 96, "smallest gated candidate wins"

    # cone geometry sanity: a short arc on a hypersphere — the centered top-1
    # basis coordinate must trace the arc monotonically (corr ~ 1), while the
    # mean direction (uncentered top variance direction) carries almost no
    # within-arc ordering signal.
    t = np.linspace(-0.2, 0.2, 40)
    arc = np.stack(
        [np.cos(t), np.sin(t) * 0.05, np.ones_like(t) * 0.9, np.full_like(t, 0.02)],
        axis=1,
    )
    arc = row_normalize(arc)
    mu = arc.mean(axis=0)
    u_a, _s_a, _ = np.linalg.svd(arc - mu, full_matrices=False)
    corr_basis = float(np.corrcoef(u_a[:, 0], t)[0, 1])
    assert abs(corr_basis) > 0.99, f"centered top basis traces the arc: {corr_basis}"
    corr_mean = float(np.corrcoef(arc @ (mu / np.linalg.norm(mu)), t)[0, 1])
    assert abs(corr_mean) < 0.6, f"mean direction must not order the arc: {corr_mean}"

    print("self-check: all amendment kernels passed")


def main(argv: list[str] | None = None) -> int:
    """Run both amendment arms on the cached corpus and record raw + summary.

    Args:
        argv: CLI arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 unless a stage raises).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--self-check", action="store_true", help="verify amendment kernels and exit"
    )
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        return 0

    started = time.monotonic()
    x = np.load(args.data_dir / "vectors_cache.npy").astype(np.float64)
    print(f"vectors={x.shape[0]}x{x.shape[1]}")

    mu = x.mean(axis=0)
    xc = x - mu
    u_c, s_c, vt_c = np.linalg.svd(xc, full_matrices=False)
    del x

    gt_sphere50 = topk_idx(row_normalize(xc + mu), KNN_SECONDARY)
    gt_centered50 = topk_idx(row_normalize(xc), KNN_SECONDARY)
    gt_sphere = gt_sphere50[:, :KNN]
    gt_centered = gt_centered50[:, :KNN]
    frame_agreement = float(jaccard_per_row(gt_sphere, gt_centered).mean())
    print(
        f"frame agreement sphere-vs-centered Jaccard@{KNN} = {frame_agreement:.4f} "
        f"(self-excluded 009 control is {FRAME_AGREEMENT_009})"
    )
    if abs(frame_agreement - FRAME_AGREEMENT_009) > FRAME_AGREEMENT_TOLERANCE:
        raise SystemExit(
            "frame agreement drifted from the 009 control reading; investigate before gating"
        )

    rows_a: list[dict] = []
    rows_b: list[dict] = []
    for k_d in K_GRID:
        basis = row_normalize(u_c[:, :k_d] * s_c[:k_d])
        recon = row_normalize((u_c[:, :k_d] * s_c[:k_d]) @ vt_c[:k_d] + mu)
        row_a = {
            "k_d": k_d,
            **arm_row(gt_centered, gt_centered50, topk_idx(basis, KNN_SECONDARY)),
        }
        row_b = {
            "k_d": k_d,
            **arm_row(gt_sphere, gt_sphere50, topk_idx(recon, KNN_SECONDARY)),
        }
        rows_a.append(row_a)
        rows_b.append(row_b)
        print(
            f"k_d={k_d}: A(in-frame)={row_a['jaccard_15']:.4f} "
            f"B(sphere-recon)={row_b['jaccard_15']:.4f}"
        )

    # k=384 arm-A sanity: in-frame full-rank must reproduce its own ground truth
    full_rank = rows_a[-1]["jaccard_15"]
    if full_rank < SPHERE_KD384_JACCARD_MIN:
        raise SystemExit(f"arm-A k=384 must be ~1.0, got {full_rank}")

    decision = apply_decision(rows_a)
    print(
        f"decision: k_d*={decision['k_d_star']} "
        f"(jaccard15>={JACCARD_THRESHOLD} AND k_d>={SIGNAL_COUNT_009})"
    )

    payload = {
        "frame_agreement": round(frame_agreement, 6),
        "arm_a_in_frame": rows_a,
        "arm_b_sphere_reconstruction": rows_b,
        "decision": decision,
        "total_seconds": round(time.monotonic() - started, 2),
    }
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    raw["amendment_009a"] = payload
    args.raw.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(f"stage=amendment_009a recorded -> {args.raw}")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    summary["amendment"] = {
        "design_note": "preregistered amendment run 2026-10-29; see 009A section",
        "k_d_star_in_frame": decision["k_d_star"],
        "k_d_star_sphere_reconstruction": apply_decision(rows_b)["k_d_star"],
        "frame_agreement": payload["frame_agreement"],
        "arm_a": rows_a,
        "arm_b": rows_b,
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"canonical summary updated -> {args.summary}")
    print(f"total wall time: {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
