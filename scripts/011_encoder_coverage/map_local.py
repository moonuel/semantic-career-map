"""Local interactive map renderer for Experiment 011.

A task-local copy of the load-bearing parts of ``scripts/map.py``: the pinned
UMAP parameters, coordinate writing, DataMapPlot interactive rendering, and
the Deck.GL template patches that power hover and click-detail panels. Unlike
``scripts/map.py`` it reads vectors and metadata from local files, accepts any
vector dimension, and takes the model label as an argument.

Usage::

    python scripts/011_encoder_coverage/map_local.py \
        --vectors data/embeddings_a_minilm_truncated.npy \
        --sample data/sample_1000.jsonl \
        --model-label "MiniLM truncated" \
        --out-dir plots/a_minilm_truncated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_NEIGHBORS = 15
MIN_DIST = 0.1
METRIC = "cosine"
RANDOM_STATE = 42
HOVER_TEXT_MAX = 80

_ARROW_ESM = "https://cdn.jsdelivr.net/npm/apache-arrow@20.0.0/+esm"

_DECK_GL_SCRIPT = "https://unpkg.com/deck.gl@9.1.14/dist.min.js"
_DECK_GL_INTEGRITY = (
    "sha384-9Qz9Vg68oxSJ8alpsDVG3eFNLSZ8P16UxKNQyVheKTcnTlbnueGTD7D0MRd6Na0z"
)

_DETAILS_JS = """
    const hoverText = hoverData.getChild('hover_text').toArray();
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
        [detail.source, detail.source_id].filter(Boolean).join(' \\u00b7 '),
      ]) {
        if (!line) continue;
        const d = document.createElement('div');
        d.textContent = line;
        d.style.color = '#555';
        panel.appendChild(d);
      }
      if (detail.url) {
        let parsed = null;
        try { parsed = new URL(detail.url); } catch (e) {}
        if (parsed && (parsed.protocol === 'http:' || parsed.protocol === 'https:')) {
          const a = document.createElement('a');
          a.href = parsed.href;
          a.textContent = 'original posting';
          a.target = '_blank';
          a.rel = 'noopener';
          a.style.display = 'block';
          a.style.margin = '6px 0';
          panel.appendChild(a);
        }
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
        'src="https://unpkg.com/deck.gl@latest/dist.min.js"',
        (
            f'src="{_DECK_GL_SCRIPT}" integrity="{_DECK_GL_INTEGRITY}" '
            'crossorigin="anonymous"'
        ),
    ),
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
        "({index}) => hoverText[index]",
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


def project(vectors: np.ndarray, n_components: int) -> np.ndarray:
    """Run UMAP with the production script's pinned parameters."""
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
    """Write ``{source_hash: [c0, c1, ...]}`` coordinates."""
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: [round(float(v), 6) for v in coords[i]] for i, key in enumerate(keys)
    }
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def write_details(sample: list[dict], out_dir: Path) -> list[Path]:
    """Write hash-bucketed click-detail JSON files."""
    buckets: dict[str, dict[str, dict]] = {}
    for row in sample:
        record = dict(row)
        source_hash = record["source_hash"]
        buckets.setdefault(source_hash[:2], {})[source_hash] = record

    details_dir = out_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    for stale in details_dir.glob("*.json"):
        stale.unlink()
    written: list[Path] = []
    for prefix, payload in sorted(buckets.items()):
        target = details_dir / f"{prefix}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        written.append(target)
    return written


def patch_interactive_html(html: str) -> tuple[str, int]:
    """Apply the Deck.GL template patches used by ``scripts/map.py``."""
    applied = 0
    for old, new in _HTML_PATCHES:
        if old in html:
            html = html.replace(old, new, 1)
            applied += 1
    return html, applied


def render_interactive(
    sample: list[dict],
    coords_2d: np.ndarray,
    model_label: str,
    target: Path,
) -> Path:
    """Render one interactive Deck.GL map from local sample data."""
    import datamapplot

    keys = [row["source_hash"] for row in sample]
    titles = []
    for row in sample:
        title = row.get("title") or "…"
        titles.append(
            title if len(title) <= HOVER_TEXT_MAX else title[: HOVER_TEXT_MAX - 1] + "…"
        )

    extra = pd.DataFrame(
        [
            {
                "title": row.get("title", ""),
                "employer": row.get("employer", ""),
                "source": row.get("source", ""),
                "hash": row["source_hash"],
            }
            for row in sample
        ]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    fig = datamapplot.create_interactive_plot(
        coords_2d,
        hover_text=titles,
        extra_point_data=extra,
        inline_data=True,
        title="Semantic job map — Experiment 011",
        sub_title=(
            f"{len(keys)} postings · {model_label} · UMAP(cosine) · seed {RANDOM_STATE}"
        ),
    )
    fig.save(str(target))
    html = target.read_text(encoding="utf-8")
    html, _applied = patch_interactive_html(html)
    target.write_text(html, encoding="utf-8")
    problems = [
        marker
        for marker in ("tableFromIPC", "onClick: showDetails", "loadDetail")
        if marker not in html
    ]
    if problems:
        raise RuntimeError(f"interactive template patch incomplete: {problems}")
    return target


def render_map(
    vectors_path: Path,
    sample_path: Path,
    model_label: str,
    out_dir: Path,
    *,
    only: str = "2d",
) -> dict:
    """Load local vectors, project, and write the map artifacts."""
    vectors = np.load(vectors_path)
    sample = [
        json.loads(line)
        for line in sample_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if vectors.shape[0] != len(sample):
        raise ValueError(
            f"{vectors_path} has {vectors.shape[0]} rows, sample has {len(sample)}"
        )
    keys = [row["source_hash"] for row in sample]
    if len(set(keys)) != len(keys):
        raise ValueError("sample has duplicate source_hash values")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_details(sample, out_dir)
    summary: dict = {
        "vectors": str(vectors_path),
        "rows": int(vectors.shape[0]),
        "dimensions": int(vectors.shape[1]),
        "model_label": model_label,
        "umap": {
            "n_neighbors": min(N_NEIGHBORS, len(sample) - 1),
            "min_dist": MIN_DIST,
            "metric": METRIC,
            "random_state": RANDOM_STATE,
        },
    }

    if only in ("2d", "both"):
        coords_2d = project(vectors, 2)
        path = write_coords(keys, coords_2d, out_dir / "coords_2d.json")
        html = render_interactive(
            sample, coords_2d, model_label, out_dir / "index.html"
        )
        summary["coords_2d"] = str(path)
        summary["html"] = str(html)
    if only in ("3d", "both"):
        coords_3d = project(vectors, 3)
        path = write_coords(keys, coords_3d, out_dir / "coords_3d.json")
        summary["coords_3d"] = str(path)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Render one map from local vectors and the experiment sample cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--only", choices=("2d", "3d", "both"), default="2d")
    args = parser.parse_args(argv)

    summary = render_map(
        args.vectors,
        args.sample,
        args.model_label,
        args.out_dir,
        only=args.only,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
