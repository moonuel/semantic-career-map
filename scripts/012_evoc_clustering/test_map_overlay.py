"""Single-responsibility tests for the Experiment 012 map overlay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from map_overlay import (
    apply_overlay,
    build_cluster_summary,
    build_function_summary,
    load_clusters,
    load_labels,
    load_sample,
    patch_html,
)

SAMPLE = [
    {"source_hash": "aa" + "0" * 62, "title": "Data role"},
    {"source_hash": "bb" + "0" * 62, "title": "Sales role"},
]

LABELS = [
    {
        "source_hash": SAMPLE[0]["source_hash"],
        "primary_function": "data-engineering",
        "function_tags": ["data-engineering", "mlops-production"],
    },
    {
        "source_hash": SAMPLE[1]["source_hash"],
        "primary_function": "other",
        "function_tags": [],
    },
]

CLUSTER_SUMMARY = {
    "n": 2,
    "clusters": 1,
    "noise": 1,
    "primary_layer": 0,
}

MAP_HTML = """<!DOCTYPE html>
<html>
  <head>
    <style>
        body { margin: 0; }
    </style>
  </head>
  <body>
    <div id="deck-container"></div>
  </body>
  <script type="module">
    const hoverData = {
      getChild: () => ({ toArray: () => ['hash-a', 'hash-b'] })
    };
    const DATA = {src: {x: [0, 1], y: [0, 1]}, length: 2};
    const pointLayer = new deck.ScatterplotLayer({
        id: 'dataPointLayer',
        data: DATA,
        getFillColor: (object, {index, data}) => {
            return [
                data.src.r[index],
                data.src.g[index],
                data.src.b[index],
                180
            ]
        },
        getLineColor: [250, 250, 250, 128],
        highlightColor: [170, 0, 0, 187],
        autoHighlight: true,
        onClick: showDetails,
    });
    const labelLayer = new deck.TextLayer({});
    const deckgl = new deck.DeckGL({
        layers: [pointLayer, labelLayer],
        getTooltip: ({index}) => hoverText[index]
    });
    const hoverHashes = hoverData.getChild('hash').toArray();
    async function showDetails(info) {
      const panel = document.createElement('div');
      panel.appendChild(document.createElement('hr'));
      panel.style.display = 'block';
    }
    </script>
</html>
"""


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def _payload() -> dict:
    labels = {
        row["source_hash"]: {
            "primary": row["primary_function"],
            "tags": row["function_tags"],
        }
        for row in LABELS
    }
    function_summary, function_colors = build_function_summary(labels)
    clusters = np.array([0, -1])
    cluster_summary, cluster_colors, cluster_legend = build_cluster_summary(
        clusters, CLUSTER_SUMMARY, "a_minilm_truncated"
    )
    return {
        "functionByHash": labels,
        "functionSummary": function_summary,
        "functionColors": function_colors,
        "clusterByHash": {
            SAMPLE[0]["source_hash"]: 0,
            SAMPLE[1]["source_hash"]: -1,
        },
        "clusterColors": cluster_colors,
        "clusterSummary": cluster_summary,
        "clusterLegendItems": cluster_legend,
    }


def test_build_function_summary_counts_secondary_tags() -> None:
    labels = {
        row["source_hash"]: {
            "primary": row["primary_function"],
            "tags": row["function_tags"],
        }
        for row in LABELS
    }
    labels["cc"] = {
        "primary": "mlops-production",
        "tags": ["mlops-production"],
    }
    summary, colors = build_function_summary(labels)
    by_id = {row["id"]: row for row in summary}
    assert by_id["data-engineering"]["primary_count"] == 1
    assert by_id["data-engineering"]["secondary_count"] == 0
    assert by_id["mlops-production"]["primary_count"] == 1
    assert by_id["mlops-production"]["secondary_count"] == 1
    assert by_id["other"]["primary_count"] == 1
    assert all(len(value) == 3 for value in colors.values())


def test_build_cluster_summary_colors_clusters_and_noise() -> None:
    labels = np.array([0, 0, 1, -1, -1])
    summary = {"clusters": 2, "noise": 2, "primary_layer": 3}
    payload, colors, legend = build_cluster_summary(
        labels, summary, "c_titan_v2"
    )
    assert payload["clusters"] == 2
    assert payload["noise"] == 2
    assert payload["counts"] == {"0": 2, "1": 1}
    assert set(colors) == {"0", "1"}
    assert legend[-1]["is_noise"] is True
    assert legend[-1]["count"] == 2


def test_patch_html_injects_both_modes() -> None:
    patched, counts = patch_html(MAP_HTML, _payload())
    assert counts == {
        "pointLayer": 1,
        "style": 1,
        "body": 1,
        "script": 1,
        "detail": 1,
        "fillColor": 1,
        "highlight": 1,
        "onClick": 1,
        "tooltip": 1,
    }
    for marker in (
        "pointHashes",
        "functionByHash",
        "clusterByHash",
        "clusterColors",
        "selectedMode",
        "selectedFunction",
        "overlay-bar",
        "EVōC clusters",
        "cluster-legend",
        "getFillColor",
        "isInteractable",
        "onHover",
    ):
        assert marker in patched
    assert "data.src.r[index]" not in patched
    assert "highlightColor: [170, 0, 0, 187]" not in patched
    assert "onClick: showDetails" not in patched
    assert "getTooltip: ({index}) => hoverText[index]" not in patched


def test_patch_html_rejects_missing_marker() -> None:
    with pytest.raises(ValueError, match="unexpected map template markers"):
        patch_html(MAP_HTML.replace("  </body>", ""), _payload())


def test_load_labels_rejects_primary_not_in_tags(tmp_path: Path) -> None:
    sample = [{"source_hash": "aa"}]
    labels = {
        "source_hash": "aa",
        "primary_function": "data-engineering",
        "function_tags": ["mlops-production"],
    }
    path = tmp_path / "bad_labels.jsonl"
    _write_jsonl(path, [labels])
    with pytest.raises(ValueError, match="primary missing from function_tags"):
        load_labels(path, sample)


def test_apply_overlay_copies_plot_and_writes_summary(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    clusters_path = tmp_path / "labels.npy"
    cluster_summary_path = tmp_path / "summary.json"
    source_html = tmp_path / "source" / "index.html"
    source_html.parent.mkdir()
    (source_html.parent / "details").mkdir()
    (source_html.parent / "details" / "aa.json").write_text("{}", encoding="utf-8")

    _write_jsonl(sample_path, SAMPLE)
    _write_jsonl(labels_path, LABELS)
    np.save(clusters_path, np.array([0, -1]))
    cluster_summary_path.write_text(json.dumps(CLUSTER_SUMMARY), encoding="utf-8")
    source_html.write_text(MAP_HTML, encoding="utf-8")

    out_html = tmp_path / "out" / "a_minilm_truncated_overlay" / "index.html"
    summary = apply_overlay(
        html_path=source_html,
        sample_path=sample_path,
        labels_path=labels_path,
        clusters_path=clusters_path,
        cluster_summary_path=cluster_summary_path,
        arm="a_minilm_truncated",
        out_html=out_html,
    )

    assert out_html.exists()
    assert (out_html.parent / "details" / "aa.json").exists()
    assert (out_html.parent / "overlay_summary.json").exists()
    assert summary["rows"] == 2
    assert summary["cluster_summary_data"]["noise"] == 1
    assert "EVōC clusters" in out_html.read_text(encoding="utf-8")


def test_load_clusters_rejects_summary_mismatch(tmp_path: Path) -> None:
    clusters_path = tmp_path / "labels.npy"
    summary_path = tmp_path / "summary.json"
    np.save(clusters_path, np.array([0, 0, -1]))
    summary_path.write_text(
        json.dumps({"n": 3, "clusters": 2, "noise": 1}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cluster IDs"):
        load_clusters(clusters_path, SAMPLE + [{"source_hash": "cc"}], summary_path)


def test_load_sample_rejects_duplicate_hashes(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    _write_jsonl(path, [{"source_hash": "aa"}, {"source_hash": "aa"}])
    with pytest.raises(ValueError, match="duplicate"):
        load_sample(path)
