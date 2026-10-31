"""Render Experiment 012 overlay copies of the Experiment 011 maps.

This is the Experiment 012 adaptation of
``scripts/011_encoder_coverage/render_maps.py``. It reuses the cached
Experiment 011 UMAP coordinates and detail files, copies each map into the
Experiment 012 plot directory, and applies the function-label and EVōC-cluster
overlays from ``map_overlay.py``.

Usage::

    python scripts/012_evoc_clustering/render_overlay_plots.py
    python scripts/012_evoc_clustering/render_overlay_plots.py --arm c_titan_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from map_overlay import apply_overlay

PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
EXP011_DIR = PROJECT_ROOT / "scripts" / "011_encoder_coverage"
SAMPLE_PATH = EXP011_DIR / "data" / "sample_1000.jsonl"
LABELS_PATH = EXPERIMENT_DIR / "data" / "labels_primary_function.jsonl"
PLOTS_DIR = EXPERIMENT_DIR / "plots"

DATASETS = (
    ("a_minilm_truncated", "A — MiniLM truncated"),
    ("b_minilm_pooled", "B — MiniLM pooled"),
    ("b2_minilm_pooled_overlap", "B2 — MiniLM pooled, overlap"),
    ("c_titan_v2", "C — Titan V2"),
)


def _paths(arm: str) -> dict[str, Path]:
    return {
        "source_html": EXP011_DIR / "plots" / arm / "index.html",
        "clusters": EXPERIMENT_DIR / "data" / "runs" / arm / "primary" / "primary" / "labels.npy",
        "cluster_summary": EXPERIMENT_DIR / "data" / "runs" / arm / "primary" / "primary" / "summary.json",
        "out_html": PLOTS_DIR / f"{arm}_overlay" / "index.html",
    }


def main(argv: list[str] | None = None) -> int:
    """Render overlay maps for all arms or selected arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        choices=[slug for slug, _ in DATASETS],
        help="render one arm; repeat to render several; default all",
    )
    args = parser.parse_args(argv)

    selected = [item for item in DATASETS if not args.arm or item[0] in args.arm]
    missing = []
    for slug, _ in selected:
        paths = _paths(slug)
        for key in ("source_html", "clusters", "cluster_summary"):
            if not paths[key].exists():
                missing.append(str(paths[key]))
    if missing:
        raise SystemExit("missing required input(s):\n" + "\n".join(missing))

    summaries = []
    for slug, label in selected:
        paths = _paths(slug)
        print(f"render {slug} -> {paths['out_html'].parent}")
        summary = apply_overlay(
            html_path=paths["source_html"],
            sample_path=SAMPLE_PATH,
            labels_path=LABELS_PATH,
            clusters_path=paths["clusters"],
            cluster_summary_path=paths["cluster_summary"],
            arm=slug,
            out_html=paths["out_html"],
        )
        summary["dataset"] = slug
        summary["label"] = label
        summaries.append(summary)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "012",
        "sample": str(SAMPLE_PATH),
        "labels": str(LABELS_PATH),
        "maps": summaries,
    }
    manifest_path = PLOTS_DIR / "overlay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
