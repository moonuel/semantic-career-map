"""Experiment 009 analyzer: result tables and the two preregistered plots.

Offline companion to ``run_benchmark.py`` — reads the canonical summary JSON
plus the raw results payload, prints the spectrum and retention/fidelity
tables, and renders the scree and retention curves into the experiments
folder (``docs/experiments/assets/``), where the experiment page embeds them.
No corpus math is redone here: cache and raw JSON are the only inputs.

Usage:
    python scripts/009_pca_informativeness/analyze_results.py
    python scripts/009_pca_informativeness/analyze_results.py --summary data/009_pca_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmark import DEFAULT_RAW, DEFAULT_SUMMARY, JACCARD_THRESHOLD

DEFAULT_PLOTS = Path(__file__).resolve().parents[2] / "docs" / "experiments" / "assets"


def print_table(title: str, header: list[str], rows: list[list]) -> None:
    """Print a fixed-width aligned table (space columns, one line per row).

    Args:
        title: Table title printed above the rows.
        header: Column header strings.
        rows: Cell values per row, stringified on print.
    """
    args = [str(cell) for cell in header]
    widths = [
        max(len(args[i]), *(len(str(r[i])) for r in rows)) if rows else len(args[i])
        for i in range(len(args))
    ]
    print(f"\n{title}")
    print(" | ".join(args[i].ljust(widths[i]) for i in range(len(args))))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))))


def spectrum_table(summary: dict) -> None:
    """Print cumulative EVR at every preregistered candidate cut."""
    spectrum = summary["spectrum"]
    rows = []
    for k_d in summary["config"]["k_grid"]:
        cum = spectrum["cumulative_evr_at_grid"][str(k_d)]
        rows.append([str(k_d), f"{cum:.4f}"])
    rows.append([f"signal={spectrum['signal_count']}", ""])
    print_table("Spectrum — cumulative EVR at candidate cuts", ["k_d", "cumEVR"], rows)
    print(
        f"\nparticipation_ratio={spectrum['participation_ratio']:.2f} "
        f"knee_dims={spectrum['knee_dims']} signal_count={spectrum['signal_count']}"
    )


def metric_table(summary: dict) -> None:
    """Print the joined per-``k_d`` metric rows with gate markers."""
    decision = summary["decision"]
    gates = decision["gates"]
    header = [
        "k_d",
        "jacc15",
        "jacc50",
        "nn1",
        "disp15",
        "pear",
        "spear",
        "mae",
        "contrast",
        "gate",
    ]
    rows = []
    for ret, fid in zip(summary["retention"], summary["fidelity"]):
        rows.append(
            [
                ret["k_d"],
                f"{ret['jaccard_15']:.4f}",
                f"{ret['jaccard_50']:.4f}",
                f"{ret['nn1_rate']:.4f}",
                f"+{ret['mean_rank_disp_15']:.2f}",
                f"{fid['pearson_r']:.5f}",
                f"{fid['spearman_r']:.5f}",
                f"{fid['mae']:.6f}",
                f"{fid['contrast_survival']:.3f}",
                "YES" if gates[str(ret["k_d"])] else ".",
            ]
        )
    print_table(
        "Per-k_d metrics (prefix + = mean over points; contrast = survival vs original)",
        header,
        rows,
    )
    print(
        f"\ndecision: k_d*={decision['k_d_star']} "
        f"(jaccard15>={decision['jaccard_threshold']} AND k_d>={decision['signal_count']})"
    )


def plot_scree(summary: dict, raw: dict, target: Path) -> Path:
    """Render the scree figure: log EVR + permutation floor, cumulative EVR.

    Reads the full-text spectrum arrays (per-dimension series) from the raw
    payload; the canonical summary stores only cuts at the candidate grid.
    """
    spectrum = raw["spectrum"]
    dims = list(range(1, len(spectrum["evr"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(dims, spectrum["evr"], lw=1.4, label="EVR (corpus)")
    axes[0].plot(
        dims,
        spectrum["permuted_evr_mean"],
        lw=1.2,
        ls="--",
        label="EVR mean (column-shuffled)",
    )
    axes[0].axvline(
        spectrum["signal_count"],
        color="tab:red",
        lw=1,
        label=f"signal count = {spectrum['signal_count']}",
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("dimension")
    axes[0].set_ylabel("explained variance ratio")
    axes[0].set_title("Scree + permutation noise floor")
    axes[0].legend(fontsize=8)

    axes[1].plot(dims, spectrum["cumulative_evr"], lw=1.4, label="cumulative EVR")
    for k_d in summary["config"]["k_grid"]:
        axes[1].axvline(k_d, color="grey", lw=0.3, alpha=0.5)
    axes[1].axhline(0.95, color="tab:red", lw=0.8, ls=":")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("dimension retained $k_d$")
    axes[1].set_ylabel("cumulative EVR")
    axes[1].set_title("Cumulative EVR at candidate cuts")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_retention(summary: dict, target: Path) -> Path:
    """Render the retention figure: Jaccard/1-NN retention, then fidelity."""
    ks = [row["k_d"] for row in summary["retention"]]
    decision = summary["decision"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(
        ks,
        [r["jaccard_15"] for r in summary["retention"]],
        marker="o",
        ms=3,
        lw=1.4,
        label="Jaccard@15",
    )
    axes[0].plot(
        ks,
        [r["jaccard_50"] for r in summary["retention"]],
        marker="s",
        ms=3,
        lw=1.0,
        label="Jaccard@50",
    )
    axes[0].plot(
        ks,
        [r["nn1_rate"] for r in summary["retention"]],
        marker="^",
        ms=3,
        lw=1.0,
        label="1-NN retention",
    )
    axes[0].axhline(
        JACCARD_THRESHOLD,
        color="tab:red",
        lw=0.8,
        ls=":",
        label=f"gate {JACCARD_THRESHOLD}",
    )
    if decision["k_d_star"] is not None:
        axes[0].axvline(
            decision["k_d_star"],
            color="tab:green",
            lw=1.2,
            label=f"k_d* = {decision['k_d_star']}",
        )
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_xlabel("PCA retained dims $k_d$")
    axes[0].set_ylabel("raw vs PCA-$k_d$ cosine k-NN")
    axes[0].set_title("Neighborhood retention through the grid")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        ks,
        [f["pearson_r"] for f in summary["fidelity"]],
        marker="o",
        ms=3,
        lw=1.4,
        label="Pearson r",
    )
    axes[1].plot(
        ks,
        [f["spearman_r"] for f in summary["fidelity"]],
        marker="s",
        ms=3,
        lw=1.0,
        label="Spearman r",
    )
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xlabel("PCA retained dims $k_d$")
    axes[1].set_ylabel("pairwise cosine correlation")
    axes[1].set_title("Sampled-pair distance fidelity (200k pairs)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return target


def main(argv: list[str] | None = None) -> int:
    """Load the recorded results, print tables, and render both plots.

    Args:
        argv: CLI arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS)
    args = parser.parse_args(argv)

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    raw = json.loads(args.raw.read_text(encoding="utf-8")) if args.raw.exists() else {}

    print(
        f"experiment {summary['experiment']} — config hash {summary['config']['configuration_hash'][:12]}…"
    )
    print(f"cache: rows={summary['config']['full_rows']} (raw stages: {sorted(raw)})")
    if summary["labels"].get("status") == "skipped":
        print(f"family D skipped: {summary['labels']['reason']}")

    spectrum_table(summary)
    metric_table(summary)

    if not raw:
        print("raw results unavailable; skipping plots")
        return 1
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    scree_path = plot_scree(summary, raw, args.plots_dir / "009_scree.png")
    retention_path = plot_retention(summary, args.plots_dir / "009_retention_curve.png")
    print(f"\nplots -> {scree_path}")
    print(f"plots -> {retention_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
