"""Job-map projection: production vectors to UMAP 2D/3D coordinates and renders.

Graduation of the ``map_sql_probe.py`` / ``map_umap_probe.py`` spikes into the
production local script for the job-map deliverable (research report 020,
research report 022 §7.1). Produces, per run:

* ``coords_2d.json`` / ``coords_3d.json`` — ``{source_hash: [c0, c1, …]}``,
  joined on the embeddings' content-addressed identity, never row position.
* ``umap_2d.png`` — static DataMapPlot preview (opt-in ``--static``; the label
  overlap solver is memory-hungry and off by default).
* ``index.html`` — the interactive Deck.GL map with per-point hover
  (title + employer) resolved from the raw lake (unless ``--no-interactive``).

Vector source: Athena ``semantic_career_map.postings_embeddings`` filtered by
the frozen configuration hash. Locally, the 009 vector cache
(``scripts/009_pca_informativeness/data/``) is reused when it matches the
configuration hash and covers the requested limit — cloud runs have no cache
and always fetch. Optional PCA pre-reduction (``--n-dims 256``) applies the
009A deployment layout: centered PCA → top-k components → renormalize.

Phase/step: Layer 1b, job-map deliverable (plan: ``.kilo/plans/2026-10-29-live-map-batch.md`` T2).

Usage:
    python scripts/map.py                          # full corpus, coords + interactive
    python scripts/map.py --limit 1000             # smoke run
    python scripts/map.py --only 2d --no-interactive
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv

from backend.pipeline.athena import aws_region
from backend.storage.raw import parse_raw_key
from backend.storage.store import ObjectStore

CONFIGURATION_HASH = "a29a35e37a0d7025a961b5241ff278904500ed43edf022ac7304dacdb97e0c98"
DATABASE = "semantic_career_map"
WORKGROUP = "semantic_career_map"
FULL_ROWS = 9354
EXPECTED_DIMS = 384
CACHE_DIR = Path("scripts/009_pca_informativeness/data")

N_NEIGHBORS = 15
MIN_DIST = 0.1
METRIC = "cosine"
RANDOM_STATE = 42

DEFAULT_LIMIT = FULL_ROWS
DEFAULT_OUT_DIR = Path("data/plots/map")
HOVER_TEXT_MAX = 80

_VECTORS_SQL = f"""
SELECT source_hash, embedding
FROM {DATABASE}.postings_embeddings
WHERE configuration_hash = '{CONFIGURATION_HASH}'
ORDER BY source_hash
LIMIT {{limit}}
"""

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _athena():
    """An Athena client pinned to the lake bucket's region (repo rule)."""
    import boto3

    return boto3.Session().client(
        "athena",
        region_name=aws_region(),
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def _run(athena, sql: str) -> list[list[str]]:
    """Submit, poll, drain, and return the result rows."""
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


def _fetch_vectors_athena(limit: int) -> tuple[list[str], np.ndarray]:
    """Pull up to ``limit`` vectors from Athena as ``(hashes, matrix)``.

    The query is deterministically ordered so a bounded run selects the same
    postings every time, and every run is a prefix of any larger run.

    Args:
        limit: Maximum rows to pull; clamped to the full dataset size.

    Returns:
        Keys in query order and an ``(n, 384)`` float32 matrix whose row
        order matches ``keys``.

    Raises:
        SystemExit: when the result violates the proven vector contract.
    """
    rows = _run(_athena(), _VECTORS_SQL.format(limit=limit))
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


def _load_cache(limit: int) -> tuple[list[str], np.ndarray] | None:
    """Reuse the 009 vector cache when it matches the run's contract.

    The cache is ``(n, 384)`` float32 rows aligned with a sorted
    ``source_hashes.json`` — the same order as ``ORDER BY source_hash`` — so
    any ``limit`` slice is the identical prefix Athena would return.

    Args:
        limit: Requested row count.

    Returns:
        ``(hashes, matrix)`` when usable, else ``None``.
    """
    hashes_path = CACHE_DIR / "source_hashes.json"
    matrix_path = CACHE_DIR / "vectors_cache.npy"
    config_path = CACHE_DIR / "009_config.json"
    if not (hashes_path.exists() and matrix_path.exists() and config_path.exists()):
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if config.get("configuration_hash") != CONFIGURATION_HASH:
        return None
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    matrix = np.load(matrix_path)
    if matrix.dtype != np.float32 or matrix.shape != (len(hashes), EXPECTED_DIMS):
        return None
    if hashes != sorted(hashes):
        return None
    if limit > len(hashes):
        return None
    hashes = hashes[:limit]
    if any(not _HASH_RE.match(h) for h in hashes):
        return None
    return hashes, np.ascontiguousarray(matrix[:limit])


def fetch_vectors(limit: int) -> tuple[list[str], np.ndarray]:
    """Load up to ``limit`` production vectors as ``(hashes, matrix)``.

    Args:
        limit: Maximum rows; clamped to the full dataset size.

    Returns:
        Keys in deterministic order and an ``(n, 384)`` float32 matrix.
    """
    limit = max(1, min(limit, FULL_ROWS))
    cached = _load_cache(limit)
    if cached is not None:
        print(f"vector source=cache rows={len(cached[0])}")
        return cached
    print(f"vector source=athena rows<={limit}")
    return _fetch_vectors_athena(limit)


def pre_reduce(vectors: np.ndarray, n_dims: int) -> np.ndarray:
    """Apply the 009A deployment pre-reduction: centered PCA → renormalize.

    Args:
        vectors: ``(n, 384)`` float32 input.
        n_dims: Component count for the truncated basis.

    Returns:
        ``(n, n_dims)`` float32, rows re-normalized to the unit sphere.
    """
    centered = vectors.astype(np.float64) - vectors.mean(axis=0, dtype=np.float64)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    reduced = centered @ vt[:n_dims].T
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (reduced / norms).astype(np.float32)


def project(vectors: np.ndarray, n_components: int) -> np.ndarray:
    """Run UMAP with the pinned parameter set.

    ``low_memory=True`` and ``n_jobs=1`` keep peak RSS bounded (measured
    636 MB at the full 9,354 rows); the default exact neighbour search built
    dense distance matrices big enough to exhaust memory.

    Args:
        vectors: ``(n, d)`` input matrix.
        n_components: Target dimensionality of the embedding.

    Returns:
        ``(n, n_components)`` coordinates.
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


def fetch_metadata(store: ObjectStore, keys: list[str]) -> dict[str, dict[str, str]]:
    """Resolve ``source_hash`` → raw-lake posting metadata for the map panel.

    Lists each allowlisted source partition, keeps only the keys this run
    needs, then reads those objects. A missing or unreadable payload yields
    empty strings — metadata gaps never fail a point. ``source`` comes from
    the key's partition (the payload dict carries none).

    Args:
        store: Object store holding the raw lake.
        keys: The run's ``source_hash`` values.

    Returns:
        Map of hash → posting fields (title, employer, url, location,
        country, employment_type, date_posted, salary, source_id, source);
        only matched hashes present.
    """
    from backend.storage.keys import RAW_SOURCES

    wanted = set(keys)
    found: dict[str, tuple[str, str]] = {}
    for source in sorted(RAW_SOURCES):
        prefix = f"lake/raw/source={source}/"
        for key in store.list_keys(prefix):
            try:
                _, source_hash = parse_raw_key(key)
            except ValueError:
                continue
            if source_hash in wanted:
                found[source_hash] = (source, key)

    meta: dict[str, dict[str, str]] = {}
    for source_hash, (source, key) in found.items():
        record: dict[str, str] = {"source": source}
        try:
            data = json.loads(store.get(key).decode("utf-8"))
        except OSError, ValueError, json.JSONDecodeError, AttributeError:
            meta[source_hash] = record
            continue
        for name in (
            "title",
            "employer",
            "url",
            "location",
            "country",
            "employment_type",
            "date_posted",
            "source_id",
        ):
            value = data.get(name)
            record[name] = "" if value is None else str(value)
        record["salary"] = _format_salary(data)
        meta[source_hash] = record

    missing = len(keys) - len(meta)
    if missing:
        print(f"metadata: {missing}/{len(keys)} hashes unresolved (empty panel)")
    return meta


def _format_salary(data: dict) -> str:
    """Render the payload's salary bounds as one display string."""
    low, high = data.get("salary_min"), data.get("salary_max")
    currency = str(data.get("salary_currency") or "")
    if low is None and high is None:
        return ""
    bounds = " – ".join(
        f"{value:,.0f}" for value in (low, high) if isinstance(value, int | float)
    )
    return f"{bounds} {currency}".strip()


def fetch_cleaned_text(keys: list[str]) -> dict[str, str]:
    """Pull the LLM-cleaned posting text from Athena for the run's keys.

    The table is fetched whole (9.6k rows, ~35 MB of text) and filtered to
    the run's keys client-side — an ``IN`` list of thousands of hashes would
    exceed Athena's query-size limit.

    Args:
        keys: The run's ``source_hash`` values.

    Returns:
        Map of hash → cleaned text (only hashes present in the artifact
        table appear).
    """
    wanted = set(keys)
    sql = f"""
SELECT source_hash, cleaned_text
FROM {DATABASE}.artifacts_cleaned
WHERE variant = 'llm' AND version = 'v1'
"""
    rows = _run(_athena(), sql)
    text: dict[str, str] = {}
    for row in rows[1:]:
        if len(row) >= 2 and row[0] in wanted:
            text[row[0]] = row[1]
    print(f"cleaned_text: {len(text)}/{len(keys)} hashes matched")
    return text


def write_details(
    keys: list[str],
    meta: dict[str, dict[str, str]],
    cleaned: dict[str, str],
    out_dir: Path,
) -> list[Path]:
    """Write hash-bucketed click-detail JSON files beside the map.

    Each file ``details/<first two hex chars>.json`` holds
    ``{source_hash: record}`` for the postings sharing that prefix, so the
    page fetches ~330 KB per bucket on demand instead of a 35 MB blob.

    Args:
        keys: The run's ``source_hash`` values.
        meta: Per-hash posting metadata from :func:`fetch_metadata`.
        cleaned: Per-hash cleaned text from :func:`fetch_cleaned_text`.
        out_dir: Map output directory; buckets land in ``out_dir/details``.

    Returns:
        Written bucket paths.
    """
    buckets: dict[str, dict[str, dict[str, str]]] = {}
    for source_hash in keys:
        record = dict(meta.get(source_hash, {}))
        record["hash"] = source_hash
        record["cleaned_text"] = cleaned.get(source_hash, "")
        buckets.setdefault(source_hash[:2], {})[source_hash] = record

    details_dir = out_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for prefix, payload in sorted(buckets.items()):
        target = details_dir / f"{prefix}.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        written.append(target)
    total = sum(p.stat().st_size for p in written)
    print(f"details: {len(written)} buckets, {total / 2**20:.1f} MB")
    return written


_ARROW_ESM = "https://cdn.jsdelivr.net/npm/apache-arrow@20/+esm"

_DETAILS_JS = """
    const hoverHashes = hoverData.getChild('hash').toArray();
    const detailBuckets = {};
    async function loadDetail(hash) {
      const bucket = hash.slice(0, 2);
      if (!(bucket in detailBuckets)) {
        const resp = await fetch(`details/${bucket}.json`);
        detailBuckets[bucket] = resp.ok ? await resp.json() : {};
      }
      return detailBuckets[bucket][hash] || null;
    }
    const panel = document.createElement('div');
    panel.style.cssText = 'position:fixed;top:0;right:0;width:min(480px,92vw);height:100%;' +
      'overflow:auto;background:#fff;box-shadow:-2px 0 10px rgba(0,0,0,.25);' +
      'padding:18px;box-sizing:border-box;font-family:arial;font-size:14px;' +
      'line-height:1.45;display:none;z-index:20';
    document.body.appendChild(panel);
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '\\u00d7';
    closeBtn.style.cssText = 'float:right;border:0;background:none;font-size:24px;' +
      'cursor:pointer;padding:0 4px';
    closeBtn.onclick = () => { panel.style.display = 'none'; };
    async function showDetails(info) {
      if (!info || info.index == null) return;
      const detail = await loadDetail(hoverHashes[info.index]);
      if (!detail) return;
      panel.replaceChildren(closeBtn);
      const title = document.createElement('div');
      title.textContent = detail.title || '(untitled)';
      title.style.cssText = 'font-size:18px;font-weight:bold;margin:0 28px 4px 0';
      panel.appendChild(title);
      for (const line of [
        [detail.employer, detail.location].filter(Boolean).join(' \\u00b7 '),
        [detail.salary, detail.employment_type, detail.date_posted]
          .filter(Boolean).join(' \\u00b7 '),
        [detail.source, detail.source_id].filter(Boolean).join(' \\u00b7 '),
      ]) {
        if (!line) continue;
        const d = document.createElement('div');
        d.textContent = line;
        d.style.color = '#555';
        panel.appendChild(d);
      }
      if (detail.url) {
        const a = document.createElement('a');
        a.href = detail.url;
        a.textContent = 'original posting';
        a.target = '_blank';
        a.rel = 'noopener';
        a.style.display = 'block';
        a.style.margin = '6px 0';
        panel.appendChild(a);
      }
      panel.appendChild(document.createElement('hr'));
      const body = document.createElement('div');
      body.textContent = detail.cleaned_text || '(no cleaned text)';
      body.style.whiteSpace = 'pre-wrap';
      panel.appendChild(body);
      panel.scrollTop = 0;
      panel.style.display = 'block';
    }
"""

_HTML_PATCHES: list[tuple[str, str]] = [
    (
        (
            "    import { ArrowLoader } from "
            "'https://cdn.jsdelivr.net/npm/@loaders.gl/arrow@4.1.0-alpha.10/+esm'\n"
            "    import { JSONLoader } from "
            "'https://cdn.jsdelivr.net/npm/@loaders.gl/json@4.0.5/+esm'"
        ),
        f"    import {{ tableFromIPC }} from '{_ARROW_ESM}'",
    ),
    (
        (
            "    import * as loaders from "
            "'https://cdn.jsdelivr.net/npm/@loaders.gl/core@4.1.0-alpha.10/+esm'\n"
        ),
        "",
    ),
    (
        "const pointData = await loaders.parse(pointDataBuffer, ArrowLoader);",
        "const pointData = await tableFromIPC(pointDataBuffer);",
    ),
    (
        "const hoverData = await loaders.parse(unzippedHoverData, ArrowLoader);",
        "const hoverData = await tableFromIPC(unzippedHoverData);",
    ),
    (
        "const labelData = await loaders.parse(unzippedLabelData, JSONLoader);",
        "const labelData = JSON.parse(fflate.strFromU8(unzippedLabelData));",
    ),
    (
        "const DATA = {src: pointData.data, length: pointData.data.x.length}",
        (
            "const DATA = {src: {"
            "x: pointData.getChild('x').toArray(), "
            "y: pointData.getChild('y').toArray(), "
            "r: pointData.getChild('r').toArray(), "
            "g: pointData.getChild('g').toArray(), "
            "b: pointData.getChild('b').toArray()}, "
            "length: pointData.numRows}"
        ),
    ),
    (
        "({index}) => hoverData.data.hover_text[index]",
        "({index}) => hoverData.getChild('hover_text').toArray()[index]",
    ),
    (
        "autoHighlight: true,",
        "autoHighlight: true,\n        onClick: showDetails,",
    ),
    (
        'document.getElementById("loading").style.display = "none";',
        'document.getElementById("loading").style.display = "none";' + _DETAILS_JS,
    ),
]


def patch_interactive_html(html: str) -> tuple[str, int]:
    """Rewrite datamapplot's deck.gl template to drop the broken loaders.gl chain.

    The vendored template calls ``loaders.parse(...)`` without importing
    ``loaders``, and jsdelivr's ``+esm`` bundles for the loaders.gl 4.1
    alphas resolve transitive dependencies unversioned, so the graph errors
    (``makeBatchFromTable`` missing) even when patched in the import. The
    data blobs are standard pyarrow output — Arrow IPC file bytes (gzip for
    hover) and a gzip JSON array — so the plain ``apache-arrow`` ESM build
    and ``JSON.parse`` replace the loaders.gl stack wholesale.

    Args:
        html: Generated interactive HTML holding the template's module
            script.

    Returns:
        The rewritten HTML and how many replacements of
        :data:`_HTML_PATCHES` applied.
    """
    applied = 0
    for old, new in _HTML_PATCHES:
        if old in html:
            html = html.replace(old, new, 1)
            applied += 1
    return html, applied


def render_interactive(
    keys: list[str],
    coords_2d: np.ndarray,
    meta: dict[str, dict[str, str]],
    target: Path,
) -> Path | None:
    """Render the interactive Deck.GL map with hover data.

    Args:
        keys: Row-order ``source_hash`` values.
        coords_2d: ``(n, 2)`` coordinates.
        meta: Per-hash metadata from :func:`fetch_metadata`.
        target: HTML file to write.

    Returns:
        The written path, or ``None`` when the render failed (the coordinate
        artifacts still stand; the HTML step is retried separately).
    """
    import datamapplot

    def hover(h: str) -> str:
        record = meta.get(h, {})
        title = record.get("title", "") or "…"
        return (
            title if len(title) <= HOVER_TEXT_MAX else title[: HOVER_TEXT_MAX - 1] + "…"
        )

    titles = [hover(h) for h in keys]
    extra = pd.DataFrame(
        [
            {
                "title": meta.get(h, {}).get("title", ""),
                "employer": meta.get(h, {}).get("employer", ""),
                "source": meta.get(h, {}).get("source", ""),
                "hash": h,
            }
            for h in keys
        ]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig = datamapplot.create_interactive_plot(
            coords_2d,
            hover_text=titles,
            extra_point_data=extra,
            inline_data=True,
            title="Semantic job map",
            sub_title=(
                f"{len(keys)} postings · all-MiniLM-L6-v2 · UMAP(cosine) · "
                f"seed {RANDOM_STATE}"
            ),
        )
        fig.save(str(target))
        html = Path(target).read_text(encoding="utf-8")
        html, _applied = patch_interactive_html(html)
        Path(target).write_text(html, encoding="utf-8")
        problems = [
            marker
            for marker in ("tableFromIPC", "onClick: showDetails", "loadDetail")
            if marker not in html
        ]
        if problems:
            print(
                f"interactive template patch incomplete (missing {problems}); "
                "map may not load",
                file=sys.stderr,
            )
    except Exception as exc:  # noqa: BLE001 — render failure must not kill coords
        print(f"interactive render failed: {exc}", file=sys.stderr)
        return None
    return target


def render_static_2d(
    keys: list[str],
    coords: np.ndarray,
    meta: dict[str, dict[str, str]],
    target: Path,
) -> Path:
    """Render a static PNG preview through DataMapPlot's matplotlib renderer."""
    import datamapplot

    def label(h: str) -> str:
        record = meta.get(h, {})
        title = record.get("title", "") or h[:8]
        return (
            title if len(title) <= HOVER_TEXT_MAX else title[: HOVER_TEXT_MAX - 1] + "…"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = datamapplot.create_plot(
        coords,
        np.array([label(h) for h in keys]),
        title="Job map — 2D UMAP",
        sub_title=f"{len(keys)} postings · all-MiniLM-L6-v2 · cosine · seed {RANDOM_STATE}",
    )
    fig.savefig(target, dpi=160, bbox_inches="tight")
    return target


def main(argv: list[str] | None = None) -> int:
    """Fetch, project, write coordinates, and render the artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=("2d", "3d", "both"), default="both", help="projection set"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"max postings to project (default {DEFAULT_LIMIT} = all)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="render the 2D PNG (expensive: datamapplot's label-overlap "
        "solver peaked >9 GB RSS at 500 labels; off by default)",
    )
    parser.add_argument(
        "--no-interactive", action="store_true", help="skip the HTML render"
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="skip the Athena cleaned-text join and the click-detail buckets",
    )
    parser.add_argument(
        "--n-dims",
        type=int,
        default=0,
        choices=(0, 256),
        help="009A PCA pre-reduction (0 = off; 256 at scale)",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    print(f"region={aws_region()}")

    keys, vectors = fetch_vectors(args.limit)
    print(f"vectors={vectors.shape[0]} dims={vectors.shape[1]} (limit={args.limit})")
    if args.n_dims:
        vectors = pre_reduce(vectors, args.n_dims)
        print(f"pre-reduced dims={vectors.shape[1]} (009A)")

    meta: dict[str, dict[str, str]] = {}
    needs_meta = not (args.no_interactive and not args.static) and args.only in (
        "2d",
        "both",
    )
    if needs_meta:
        try:
            meta = fetch_metadata(ObjectStore.from_env(), keys)
        except RuntimeError as exc:
            print(f"metadata unavailable ({exc}); hovers empty", file=sys.stderr)

    want_details = needs_meta and not args.no_text
    if want_details:
        cleaned = fetch_cleaned_text(keys)
        write_details(keys, meta, cleaned, args.out_dir)

    coords_2d: np.ndarray | None = None
    if args.only in ("2d", "both"):
        started = time.monotonic()
        coords_2d = project(vectors, 2)
        path = write_coords(keys, coords_2d, args.out_dir / "coords_2d.json")
        print(
            f"umap2d shape={coords_2d.shape} seconds={time.monotonic() - started:.1f} "
            f"-> {path}"
        )
        if args.static:
            png = render_static_2d(keys, coords_2d, meta, args.out_dir / "umap_2d.png")
            print(f"static preview -> {png} ({png.stat().st_size} bytes)")
        if not args.no_interactive:
            html = render_interactive(
                keys, coords_2d, meta, args.out_dir / "index.html"
            )
            if html is not None:
                print(f"interactive map -> {html} ({html.stat().st_size} bytes)")

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
