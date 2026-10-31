"""Experiment 012 overlay renderer for the Experiment 011 interactive maps.

This is the Experiment 012 adaptation of ``scripts/011_encoder_coverage/
map_local.py``. It does not re-run UMAP. It copies an existing Experiment 011
plot directory, then patches the copied map with two mutually exclusive
overlays:

- function labels from ``labels_primary_function.jsonl``
- primary EVōC clusters from ``runs/<arm>/primary/primary/labels.npy``

Usage::

    python scripts/012_evoc_clustering/map_overlay.py \
        --html scripts/011_encoder_coverage/plots/c_titan_v2/index.html \
        --sample scripts/011_encoder_coverage/data/sample_1000.jsonl \
        --labels scripts/012_evoc_clustering/data/labels_primary_function.jsonl \
        --clusters scripts/012_evoc_clustering/data/runs/c_titan_v2/primary/primary/labels.npy \
        --cluster-summary scripts/012_evoc_clustering/data/runs/c_titan_v2/primary/primary/summary.json \
        --arm c_titan_v2 \
        --out scripts/012_evoc_clustering/plots/c_titan_v2_overlay/index.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from taxonomy import VALID_FUNCTIONS

FUNCTION_COLORS: dict[str, str] = {
    "agentic-ai": "#e6194b",
    "llm-fine-tuning": "#3cb44b",
    "llm-information-retrieval": "#4363d8",
    "classical-ml": "#f58231",
    "mlops-production": "#911eb4",
    "data-engineering": "#00a6a6",
    "analytics-storytelling": "#f032e6",
    "computer-vision": "#8f9b00",
    "ml-platform": "#fabed4",
    "reinforcement-learning": "#469990",
    "research": "#7f5af0",
    "ai-safety-governance": "#9a6324",
    "deep-learning": "#800000",
    "other": "#9ca3af",
}

CLUSTER_PALETTE: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
)

NOISE_COLOR = "#a0a0a0"

_MARKER_POINT_LAYER = "const pointLayer = new deck.ScatterplotLayer({"
_MARKER_STYLE_CLOSE = "    </style>"
_MARKER_BODY_CLOSE = "  </body>"
_MARKER_SCRIPT_CLOSE = "    </script>"
_MARKER_DETAIL_HR = "      panel.appendChild(document.createElement('hr'));"

_FILL_COLOR_PATTERN = re.compile(
    r"        getFillColor: \(object, \{index, data\}\) => \{.*?\n        \},",
    re.DOTALL,
)

_CSS = """
        #overlay-bar {
          position: fixed;
          left: 0;
          right: 0;
          bottom: 0;
          z-index: 10;
          display: flex;
          align-items: center;
          gap: 6px;
          overflow-x: auto;
          padding: 8px 12px;
          background: rgba(255, 255, 255, 0.96);
          border-top: 1px solid #ddd;
          box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.08);
          font-family: arial;
        }

        #overlay-bar .overlay-separator {
          width: 1px;
          align-self: stretch;
          background: #ccc;
          margin: 0 4px;
        }

        .overlay-button {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          flex: 0 0 auto;
          border: 1px solid #ccc;
          border-radius: 6px;
          background: #fff;
          color: #222;
          cursor: pointer;
          font: 13px arial;
          padding: 6px 9px;
          white-space: nowrap;
        }

        .overlay-button:hover {
          border-color: #888;
        }

        .overlay-button[aria-pressed="true"] {
          border-color: #111;
          box-shadow: inset 0 0 0 1px #111;
          font-weight: bold;
        }

        .overlay-swatch {
          display: inline-block;
          width: 12px;
          height: 12px;
          border-radius: 50%;
          border: 1px solid rgba(0, 0, 0, 0.25);
          flex: 0 0 auto;
        }

        #cluster-legend {
          position: fixed;
          right: 12px;
          bottom: 64px;
          z-index: 11;
          display: none;
          align-items: center;
          gap: 8px;
          max-width: calc(100vw - 24px);
          overflow-x: auto;
          padding: 7px 10px;
          background: rgba(255, 255, 255, 0.97);
          border: 1px solid #ddd;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
          font: 12px arial;
          color: #333;
        }

        .cluster-legend-item {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          flex: 0 0 auto;
          white-space: nowrap;
        }

        .cluster-legend-swatch {
          display: inline-block;
          width: 10px;
          height: 10px;
          border-radius: 2px;
          border: 1px solid rgba(0, 0, 0, 0.2);
        }
"""

_BAR_HTML = """
    <div id="cluster-legend"></div>
    <div id="overlay-bar"></div>
"""

_UI_JS = r"""
    const overlayBar = document.getElementById('overlay-bar');
    const clusterLegend = document.getElementById('cluster-legend');

    function hexToCss(color) {
      return color || '#9ca3af';
    }

    function makeSwatch(color) {
      const swatch = document.createElement('span');
      swatch.className = 'overlay-swatch';
      swatch.style.background = hexToCss(color);
      return swatch;
    }

    function makeButton(label, color, pressed, title, onClick, swatchStyle) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'overlay-button';
      button.setAttribute('aria-pressed', pressed ? 'true' : 'false');
      if (title) button.title = title;
      if (swatchStyle) {
        const swatch = document.createElement('span');
        swatch.className = 'overlay-swatch';
        swatch.style.background = swatchStyle;
        button.appendChild(swatch);
      } else if (color) {
        button.appendChild(makeSwatch(color));
      }
      const text = document.createElement('span');
      text.textContent = label;
      button.appendChild(text);
      button.addEventListener('click', onClick);
      return button;
    }

    const allButton = makeButton(
      'All',
      null,
      true,
      'Primary-function colors',
      () => setMode('all')
    );
    overlayBar.appendChild(allButton);

    for (const item of functionSummary) {
      const button = makeButton(
        `${item.id} · ${item.primary_count}`,
        item.color,
        false,
        `primary ${item.primary_count} · secondary ${item.secondary_count}`,
        () => {
          if (selectedMode === 'function' && selectedFunction === item.id) {
            setMode('all');
          } else {
            setMode('function', item.id);
          }
        }
      );
      button.dataset.mode = 'function';
      button.dataset.function = item.id;
      overlayBar.appendChild(button);
    }

    const separator = document.createElement('span');
    separator.className = 'overlay-separator';
    overlayBar.appendChild(separator);

    const clusterGradient = clusterLegendItems
      .filter((item) => !item.is_noise)
      .map((item) => item.color)
      .join(', ');
    const clusterButton = makeButton(
      'EVōC clusters',
      null,
      false,
      `EVōC primary · ${clusterSummary.clusters} clusters · ${clusterSummary.noise} noise`,
      () => {
        if (selectedMode === 'clusters') {
          setMode('all');
        } else {
          setMode('clusters');
        }
      },
      clusterGradient
        ? `linear-gradient(90deg, ${clusterGradient})`
        : NOISE_COLOR
    );
    clusterButton.dataset.mode = 'clusters';
    overlayBar.appendChild(clusterButton);

    function updateBarState() {
      for (const button of overlayBar.querySelectorAll('.overlay-button')) {
        const mode = button.dataset.mode || 'all';
        const fn = button.dataset.function || null;
        const active =
          selectedMode === 'all'
            ? mode === 'all'
            : selectedMode === 'function'
              ? mode === 'function' && fn === selectedFunction
              : mode === 'clusters';
        button.setAttribute('aria-pressed', active ? 'true' : 'false');
      }
    }

    function updateClusterLegend() {
      clusterLegend.replaceChildren();
      if (selectedMode !== 'clusters') {
        clusterLegend.style.display = 'none';
        return;
      }
      for (const item of clusterLegendItems) {
        const wrapper = document.createElement('span');
        wrapper.className = 'cluster-legend-item';
        const swatch = document.createElement('span');
        swatch.className = 'cluster-legend-swatch';
        swatch.style.background = item.color;
        const text = document.createElement('span');
        text.textContent = `${item.label} · ${item.count}`;
        wrapper.appendChild(swatch);
        wrapper.appendChild(text);
        clusterLegend.appendChild(wrapper);
      }
      clusterLegend.style.display = 'flex';
    }

    function refreshColors() {
      const activeLayer = pointLayer.clone({
        updateTriggers: { getFillColor: [selectedMode, selectedFunction] }
      });
      deckgl.setProps({ layers: [activeLayer, labelLayer] });
    }

    function setMode(mode, fn = null) {
      selectedMode = mode;
      selectedFunction = mode === 'function' ? fn : null;
      updateBarState();
      updateClusterLegend();
      refreshColors();
    }

    function appendOverlayDetails(panel, hash) {
      const label = functionByHash[hash];
      const cluster = clusterByHash[hash];
      const line = (text) => {
        const row = document.createElement('div');
        row.textContent = text;
        panel.appendChild(row);
      };
      if (label) {
        line(`Primary function: ${label.primary}`);
        line(
          `Additional tags: ${
            label.tags.length ? label.tags.join(', ') : '(none)'
          }`
        );
      }
      if (cluster !== undefined) {
        const clusterText = cluster === -1 ? 'noise (-1)' : String(cluster);
        line(`EVōC cluster: ${clusterText} (arm ${clusterSummary.arm})`);
      }
    }

    updateBarState();
    updateClusterLegend();
"""

_GET_FILL_COLOR_JS = """        getFillColor: (object, {index, data}) => {
            const hash = pointHashes[index];
            if (selectedMode === 'clusters') {
                const cluster = clusterByHash[hash];
                if (cluster === -1) {
                    return [160, 160, 160, 80];
                }
                return [...clusterColors[String(cluster)], 220];
            }
            const label = functionByHash[hash];
            if (selectedMode === 'all') {
                return [...functionColors[label.primary], 220];
            }
            if (label.primary === selectedFunction) {
                return [...functionColors[selectedFunction], 230];
            }
            if (label.tags.includes(selectedFunction)) {
                return [...functionColors[selectedFunction], 110];
            }
            return [190, 190, 190, 35];
        },"""


def _hex_to_rgb(color: str) -> list[int]:
    """Convert ``#rrggbb`` to an RGB triplet."""
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"invalid hex color: {color!r}")
    return [int(value[index : index + 2], 16) for index in (0, 2, 4)]


def _golden_angle_color(index: int) -> str:
    """Generate a deterministic fallback color for large cluster counts."""
    hue = (index * 137.508) % 360.0
    saturation = 0.65
    lightness = 0.52
    chroma = (1.0 - abs(2.0 * lightness - 1.0)) * saturation
    x = chroma * (1.0 - abs((hue / 60.0) % 2.0 - 1.0))
    m = lightness - chroma / 2.0
    if hue < 60:
        red, green, blue = chroma, x, 0.0
    elif hue < 120:
        red, green, blue = x, chroma, 0.0
    elif hue < 180:
        red, green, blue = 0.0, chroma, x
    elif hue < 240:
        red, green, blue = 0.0, x, chroma
    elif hue < 300:
        red, green, blue = x, 0.0, chroma
    else:
        red, green, blue = chroma, 0.0, x
    return "#" + "".join(
        f"{round((channel + m) * 255):02x}" for channel in (red, green, blue)
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sample(path: Path) -> list[dict[str, Any]]:
    """Load the fixed sample and require unique hashes."""
    rows = _read_jsonl(path)
    hashes = [row.get("source_hash") for row in rows]
    if not rows or any(not value for value in hashes):
        raise ValueError(f"{path}: missing source_hash values")
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"{path}: duplicate source_hash values")
    return rows


def load_labels(path: Path, sample: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Load and validate weak function labels against the sample."""
    rows = _read_jsonl(path)
    by_hash: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_hash = row.get("source_hash")
        primary = row.get("primary_function")
        tags = row.get("function_tags", [])
        if not source_hash:
            raise ValueError(f"{path}: row missing source_hash")
        if source_hash in by_hash:
            raise ValueError(f"{path}: duplicate source_hash {source_hash}")
        if primary not in VALID_FUNCTIONS:
            raise ValueError(f"{path}: invalid primary_function {primary!r}")
        if not isinstance(tags, list):
            raise TypeError(f"{path}: function_tags is not a list for {source_hash}")
        if any(tag not in VALID_FUNCTIONS for tag in tags):
            raise ValueError(f"{path}: invalid function_tags for {source_hash}")
        if primary == "other" and tags:
            raise ValueError(f"{path}: other has function_tags for {source_hash}")
        if primary != "other" and primary not in tags:
            raise ValueError(f"{path}: primary missing from function_tags for {source_hash}")
        by_hash[source_hash] = {
            "primary": primary,
            "tags": list(dict.fromkeys(tags)),
        }

    sample_hashes = [row["source_hash"] for row in sample]
    if len(by_hash) != len(sample_hashes):
        raise ValueError(
            f"{path}: {len(by_hash)} label rows, sample has {len(sample_hashes)}"
        )
    missing = [source_hash for source_hash in sample_hashes if source_hash not in by_hash]
    extra = [source_hash for source_hash in by_hash if source_hash not in set(sample_hashes)]
    if missing or extra:
        raise ValueError(
            f"{path}: label/sample mismatch (missing={len(missing)}, extra={len(extra)})"
        )
    return by_hash


def load_clusters(
    path: Path,
    sample: list[dict[str, Any]],
    summary_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load primary EVōC labels and validate them against their summary."""
    labels = np.load(path)
    if labels.ndim != 1:
        raise ValueError(f"{path}: expected a 1-D label array, got {labels.shape}")
    if labels.shape[0] != len(sample):
        raise ValueError(f"{path}: {labels.shape[0]} labels, sample has {len(sample)}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{path}: labels must be integers, got {labels.dtype}")
    labels = labels.astype(int)
    if np.any(labels < -1):
        raise ValueError(f"{path}: labels below -1 are invalid")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cluster_ids = sorted(int(value) for value in np.unique(labels) if value >= 0)
    if len(cluster_ids) != int(summary.get("clusters", -1)):
        raise ValueError(
            f"{path}: {len(cluster_ids)} cluster IDs, summary says {summary.get('clusters')}"
        )
    noise = int(np.sum(labels == -1))
    if noise != int(summary.get("noise", -1)):
        raise ValueError(f"{path}: noise={noise}, summary says {summary.get('noise')}")
    if cluster_ids != list(range(len(cluster_ids))):
        raise ValueError(f"{path}: cluster IDs are not contiguous from 0: {cluster_ids}")
    if labels.shape[0] != int(summary.get("n", labels.shape[0])):
        raise ValueError(f"{path}: summary n does not match label count")
    return labels, summary


def build_function_summary(
    labels: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    """Build button metadata and RGB colors for function categories."""
    counts: dict[str, int] = {}
    secondary: dict[str, int] = {}
    for row in labels.values():
        primary = row["primary"]
        counts[primary] = counts.get(primary, 0) + 1
        for tag in row["tags"]:
            if tag != primary:
                secondary[tag] = secondary.get(tag, 0) + 1

    present = [name for name in VALID_FUNCTIONS if name in counts]
    summary = [
        {
            "id": name,
            "color": FUNCTION_COLORS[name],
            "primary_count": counts[name],
            "secondary_count": secondary.get(name, 0),
        }
        for name in present
    ]
    colors = {name: _hex_to_rgb(FUNCTION_COLORS[name]) for name in present}
    if not summary:
        raise ValueError("no function labels found")
    return summary, colors


def build_cluster_summary(
    labels: np.ndarray,
    summary: dict[str, Any],
    arm: str,
) -> tuple[dict[str, Any], dict[str, list[int]], list[dict[str, Any]]]:
    """Build cluster metadata, RGB colors, and legend entries."""
    cluster_ids = sorted(int(value) for value in np.unique(labels) if value >= 0)
    counts = {
        str(cluster_id): int(np.sum(labels == cluster_id))
        for cluster_id in cluster_ids
    }
    colors: dict[str, str] = {}
    for index, cluster_id in enumerate(cluster_ids):
        color = (
            CLUSTER_PALETTE[index]
            if index < len(CLUSTER_PALETTE)
            else _golden_angle_color(index)
        )
        colors[str(cluster_id)] = color

    noise = int(np.sum(labels == -1))
    payload = {
        "arm": arm,
        "clusters": len(cluster_ids),
        "noise": noise,
        "counts": counts,
        "primary_layer": summary.get("primary_layer"),
        "summary_clusters": summary.get("clusters"),
        "summary_noise": summary.get("noise"),
    }
    legend = [
        {
            "id": cluster_id,
            "label": f"cluster {cluster_id}",
            "count": counts[str(cluster_id)],
            "color": colors[str(cluster_id)],
            "is_noise": False,
        }
        for cluster_id in cluster_ids
    ]
    if noise:
        legend.append(
            {
                "id": -1,
                "label": "noise",
                "count": noise,
                "color": NOISE_COLOR,
                "is_noise": True,
            }
        )
    rgb_colors = {key: _hex_to_rgb(value) for key, value in colors.items()}
    return payload, rgb_colors, legend


def build_payload(
    sample: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    cluster_labels: np.ndarray,
    cluster_summary: dict[str, Any],
    arm: str,
) -> dict[str, Any]:
    """Build the JSON payload injected into the copied map."""
    function_summary, function_colors = build_function_summary(labels)
    cluster_payload, cluster_colors, cluster_legend = build_cluster_summary(
        cluster_labels, cluster_summary, arm
    )
    cluster_by_hash = {
        row["source_hash"]: int(cluster_labels[index])
        for index, row in enumerate(sample)
    }
    return {
        "functionByHash": labels,
        "functionSummary": function_summary,
        "functionColors": function_colors,
        "clusterByHash": cluster_by_hash,
        "clusterColors": cluster_colors,
        "clusterSummary": cluster_payload,
        "clusterLegendItems": cluster_legend,
    }


def _injection_before_layer(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    return f"""const pointHashes = hoverData.getChild('hash').toArray();
    const overlayData = {data};
    const functionByHash = overlayData.functionByHash;
    const functionSummary = overlayData.functionSummary;
    const functionColors = overlayData.functionColors;
    const clusterByHash = overlayData.clusterByHash;
    const clusterColors = overlayData.clusterColors;
    const clusterSummary = overlayData.clusterSummary;
    const clusterLegendItems = overlayData.clusterLegendItems;
    const NOISE_COLOR = {json.dumps(NOISE_COLOR)};
    let selectedMode = 'all';
    let selectedFunction = null;

    """


def patch_html(html: str, payload: dict[str, Any]) -> tuple[str, dict[str, int]]:
    """Inject overlay data, styles, controls, and JS into a map HTML string."""
    counts = {
        "pointLayer": html.count(_MARKER_POINT_LAYER),
        "style": html.count(_MARKER_STYLE_CLOSE),
        "body": html.count(_MARKER_BODY_CLOSE),
        "script": html.count(_MARKER_SCRIPT_CLOSE),
        "detail": html.count(_MARKER_DETAIL_HR),
        "fillColor": len(_FILL_COLOR_PATTERN.findall(html)),
    }
    expected = {key: 1 for key in counts}
    if counts != expected:
        raise ValueError(f"unexpected map template markers: {counts} != {expected}")

    html = html.replace(
        _MARKER_POINT_LAYER,
        _injection_before_layer(payload) + _MARKER_POINT_LAYER,
        1,
    )
    html, replaced = _FILL_COLOR_PATTERN.subn(_GET_FILL_COLOR_JS, html, count=1)
    if replaced != 1:
        raise ValueError("getFillColor block was not replaced exactly once")

    html = html.replace(_MARKER_STYLE_CLOSE, _CSS + _MARKER_STYLE_CLOSE, 1)
    html = html.replace(_MARKER_BODY_CLOSE, _BAR_HTML + _MARKER_BODY_CLOSE, 1)
    html = html.replace(
        _MARKER_DETAIL_HR,
        """      const overlayInfo = document.createElement('div');
      overlayInfo.style.cssText = 'margin:8px 0;padding:8px;background:#f7f7f7;border-radius:6px';
      appendOverlayDetails(overlayInfo, hoverHashes[info.index]);
      panel.appendChild(overlayInfo);
      panel.appendChild(document.createElement('hr'));""",
        1,
    )

    script_close = html.rfind(_MARKER_SCRIPT_CLOSE)
    if script_close == -1:
        raise ValueError("module script close marker not found")
    html = (
        html[:script_close]
        + _UI_JS
        + html[script_close:]
    )
    return html, counts


def _copy_plot(source_html: Path, target_html: Path) -> Path:
    source_dir = source_html.parent
    target_dir = target_html.parent
    if source_dir.resolve() == target_dir.resolve():
        raise ValueError("source and target plot directories must differ")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    return target_dir / source_html.name


def apply_overlay(
    *,
    html_path: Path,
    sample_path: Path,
    labels_path: Path,
    clusters_path: Path,
    cluster_summary_path: Path,
    arm: str,
    out_html: Path,
) -> dict[str, Any]:
    """Copy one 011 plot and apply function + EVōC cluster overlays."""
    sample = load_sample(sample_path)
    labels = load_labels(labels_path, sample)
    cluster_labels, cluster_summary = load_clusters(
        clusters_path, sample, cluster_summary_path
    )
    payload = build_payload(sample, labels, cluster_labels, cluster_summary, arm)
    patched, markers = patch_html(html_path.read_text(encoding="utf-8"), payload)
    target_html = _copy_plot(html_path, out_html)
    target_html.write_text(patched, encoding="utf-8")

    overlay_summary = {
        "experiment": "012",
        "arm": arm,
        "rows": len(sample),
        "sample": str(sample_path),
        "labels": str(labels_path),
        "labels_sha256": _sha256(labels_path),
        "clusters": str(clusters_path),
        "clusters_sha256": _sha256(clusters_path),
        "cluster_summary": str(cluster_summary_path),
        "cluster_summary_sha256": _sha256(cluster_summary_path),
        "cluster_primary_layer": cluster_summary.get("primary_layer"),
        "function_summary": payload["functionSummary"],
        "cluster_summary_data": payload["clusterSummary"],
        "cluster_legend": payload["clusterLegendItems"],
        "function_colors": payload["functionColors"],
        "cluster_colors": payload["clusterColors"],
        "markers": markers,
        "source_html": str(html_path),
        "html": str(out_html),
    }
    summary_path = target_html.parent / "overlay_summary.json"
    summary_path.write_text(json.dumps(overlay_summary, indent=2), encoding="utf-8")
    return overlay_summary


def main(argv: list[str] | None = None) -> int:
    """Apply the overlay to one Experiment 011 map."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--cluster-summary", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = apply_overlay(
        html_path=args.html,
        sample_path=args.sample,
        labels_path=args.labels,
        clusters_path=args.clusters,
        cluster_summary_path=args.cluster_summary,
        arm=args.arm,
        out_html=args.out,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
