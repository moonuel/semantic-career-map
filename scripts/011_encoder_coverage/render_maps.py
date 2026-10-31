"""Render the four Experiment 011 interactive maps from local artifacts.

This is the second script of Experiment 011. It orchestrates the local
``map_local.py`` renderer over the four cached embedding datasets produced by
``run_experiment.py`` and writes four interactive HTML maps under
``scripts/011_encoder_coverage/plots/``. Nothing is fetched from Athena or the
object store; the sample cache is the only input.

Usage::

    python scripts/011_encoder_coverage/render_maps.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from map_local import render_map

DATA_DIR = EXPERIMENT_DIR / "data"
PLOTS_DIR = EXPERIMENT_DIR / "plots"
SAMPLE_PATH = DATA_DIR / "sample_1000.jsonl"

DATASETS = (
    (
        "a_minilm_truncated",
        "A — MiniLM truncated (256 tokens)",
        DATA_DIR / "embeddings_a_minilm_truncated.npy",
    ),
    (
        "b_minilm_pooled",
        "B — MiniLM pooled, no overlap",
        DATA_DIR / "embeddings_b_minilm_pooled.npy",
    ),
    (
        "b2_minilm_pooled_overlap",
        "B2 — MiniLM pooled, 32-token overlap",
        DATA_DIR / "embeddings_b2_minilm_pooled_overlap.npy",
    ),
    (
        "c_titan_v2",
        "C — Titan V2 normalized (1024-d)",
        DATA_DIR / "embeddings_c_titan_v2.npy",
    ),
)


def main(argv: list[str] | None = None) -> int:
    """Render every available dataset and write a map manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("2d", "3d", "both"), default="2d")
    args = parser.parse_args(argv)

    if not SAMPLE_PATH.exists():
        raise SystemExit(
            f"missing sample cache {SAMPLE_PATH}; run run_experiment.py first"
        )

    summaries = []
    missing = []
    for slug, label, vectors_path in DATASETS:
        if not vectors_path.exists():
            missing.append(str(vectors_path))
            print(f"skip {slug}: missing {vectors_path}")
            continue
        out_dir = PLOTS_DIR / slug
        print(f"render {slug} -> {out_dir}")
        summary = render_map(
            vectors_path,
            SAMPLE_PATH,
            label,
            out_dir,
            only=args.only,
        )
        summary["dataset"] = slug
        summary["label"] = label
        summaries.append(summary)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "011",
        "sample": str(SAMPLE_PATH),
        "maps": summaries,
        "missing": missing,
    }
    manifest_path = PLOTS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest_path}")
    if missing:
        raise SystemExit(f"missing {len(missing)} embedding dataset(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
