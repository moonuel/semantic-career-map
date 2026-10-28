"""Layer 1b step 1: project the embedded vectors with UMAP, 2D and 3D.

Feature spike for the job-map deliverable (research report 020). Takes the
vector contract proven by ``scripts/map_sql_probe.py`` — 9354 rows, 384
dimensions, one row per ``source_hash`` — and produces the two coordinate
artifacts the map renderers need:

* **2D** → DataMapPlot's interactive Deck.GL map (the Stage-1 deliverable).
* **3D** → a coordinate table for inspection and any later 3D viewer.

Both projections run on the same matrix with the same UMAP parameters, so the
only difference between them is ``n_components``. Coordinates are written as
compact JSON keyed by ``source_hash`` so every downstream consumer joins on
the same identity the embeddings use, never on row position.

The 2D half also renders a static PNG through DataMapPlot's matplotlib
renderer, which is the cheapest way to eyeball whether the projection is
worth rendering interactively before committing to the HTML step.

Spike, not production: hardcoded parameters, no modules, no logging
framework. It is deleted once the feature graduates to ``scripts/map.py``.

Usage:
    python scripts/map_umap_probe.py                  # both projections, 100 postings
    python scripts/map_umap_probe.py --only 2d
    python scripts/map_umap_probe.py --limit 9354     # the full corpus
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import datamapplot
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_sql_probe import _athena, _run

from backend.pipeline.athena import aws_region

CONFIGURATION_HASH = "a29a35e37a0d7025a961b5241ff278904500ed43edf022ac7304dacdb97e0c98"
DATABASE = "semantic_career_map"
FULL_ROWS = 9354
EXPECTED_DIMS = 384

N_NEIGHBORS = 15
MIN_DIST = 0.1
METRIC = "cosine"
RANDOM_STATE = 42

DEFAULT_LIMIT = 100
DEFAULT_OUT_DIR = Path("data/plots/map_probe")

_VECTORS_SQL = f"""
SELECT source_hash, embedding
FROM {DATABASE}.postings_embeddings
WHERE configuration_hash = '{CONFIGURATION_HASH}'
ORDER BY source_hash
LIMIT {{limit}}
"""


def fetch_vectors(limit: int) -> tuple[list[str], np.ndarray]:
    """Load up to ``limit`` production vectors as ``(hashes, matrix)``.

    The query is deterministically ordered so a bounded run selects the same
    postings every time, and every run is a prefix of any larger run.

    Args:
        limit: Maximum rows to pull; clamped to the full dataset size.

    Returns:
        Keys in query order and an ``(n, 384)`` float32 matrix whose row order
        matches ``keys``.

    Raises:
        SystemExit: when the result violates the proven vector contract.
    """
    limit = max(1, min(limit, FULL_ROWS))
    rows, _ = _run(_athena(), _VECTORS_SQL.format(limit=limit))
    body = rows[1:]
    if not body:
        raise SystemExit("query returned no vectors")
    if len(body) > limit:
        raise SystemExit(f"expected at most {limit} rows, got {len(body)}")

    keys: list[str] = []
    vectors = np.empty((len(body), EXPECTED_DIMS), dtype=np.float32)
    for i, row in enumerate(body):
        keys.append(row[0])
        vectors[i] = np.asarray(json.loads(row[1]), dtype=np.float32)

    if len(set(keys)) != len(keys):
        raise SystemExit(f"{len(keys) - len(set(keys))} duplicate source_hash rows")
    return keys, vectors


def project(vectors: np.ndarray, n_components: int) -> np.ndarray:
    """Run UMAP on the vectors with the pinned parameter set.

    ``low_memory=True`` and ``n_jobs=1`` keep peak RSS to a couple of
    float32 copies of the input; the default exact neighbour search built
    dense distance matrices big enough to exhaust memory at a few hundred
    rows on this machine.
    """
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(N_NEIGHBORS, len(vectors) - 1),
        min_dist=MIN_DIST,
        metric=METRIC,
        random_state=RANDOM_STATE,
        low_memory=True,
        n_jobs=1,
    )
    return reducer.fit_transform(vectors)


def write_coords(keys: list[str], coords: np.ndarray, target: Path) -> Path:
    """Write ``{source_hash: [c0, c1, …]}`` as JSON and return the path."""
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: [round(float(v), 6) for v in coords[i]] for i, key in enumerate(keys)
    }
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def render_static_2d(keys: list[str], coords: np.ndarray, target: Path) -> Path:
    """Render a static PNG of the 2D projection through DataMapPlot."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = datamapplot.create_plot(
        coords,
        np.array([f"{k[:8]}…" for k in keys]),
        title="Job map — 2D UMAP (no labels yet)",
        sub_title=f"{len(keys)} postings · all-MiniLM-L6-v2 · cosine · seed {RANDOM_STATE}",
    )
    fig.savefig(target, dpi=160, bbox_inches="tight")
    return target


def main(argv: list[str] | None = None) -> int:
    """Project, write coordinates, and optionally render the static preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=("2d", "3d", "both"), default="both", help="projection set"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"max postings to project (default {DEFAULT_LIMIT}, {FULL_ROWS} = all)",
    )
    parser.add_argument(
        "--no-static", action="store_true", help="skip the 2D PNG preview"
    )
    args = parser.parse_args(argv)

    load_dotenv()
    print(f"region={aws_region()}")

    keys, vectors = fetch_vectors(args.limit)
    print(f"vectors={vectors.shape[0]} dims={vectors.shape[1]} (limit={args.limit})")

    if args.only in ("2d", "both"):
        started = time.monotonic()
        coords_2d = project(vectors, 2)
        path = write_coords(keys, coords_2d, args.out_dir / "coords_2d.json")
        print(
            f"umap2d shape={coords_2d.shape} seconds={time.monotonic() - started:.1f} "
            f"-> {path}"
        )
        if not args.no_static:
            png = render_static_2d(keys, coords_2d, args.out_dir / "umap_2d.png")
            print(f"static preview -> {png} ({png.stat().st_size} bytes)")

    if args.only in ("3d", "both"):
        started = time.monotonic()
        coords_3d = project(vectors, 3)
        path = write_coords(keys, coords_3d, args.out_dir / "coords_3d.json")
        print(
            f"umap3d shape={coords_3d.shape} seconds={time.monotonic() - started:.1f} "
            f"-> {path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
