"""Tests for the SQL-selection adapter (D12)."""

from __future__ import annotations

import hashlib
import json

import pytest

from backend.pipeline import selection
from backend.storage import keys
from tests.conftest import MemorySink


def _hash(tag: str) -> str:
    """Deterministic 64-char hex digest for a tag."""
    return hashlib.sha256(tag.encode()).hexdigest()


def _csv(*rows: tuple[str, str, str]) -> str:
    """Render selection rows as Athena's CSV output."""
    lines = ["source,source_hash,source_id"] + [",".join(r) for r in rows]
    return "\n".join(lines) + "\n"


def test_configuration_hash_matches_task_33_definition() -> None:
    h = selection.configuration_hash_for("m", "sys", "user")
    assert h == hashlib.sha256(b"m\nsys\nuser").hexdigest()
    assert h != selection.configuration_hash_for("m2", "sys", "user")


def test_parse_query_results_validates_rows() -> None:
    rows = selection.parse_query_results(
        _csv(("greenhouse", _hash("a"), "greenhouse:x:1"))
    )
    assert rows == [
        {
            "source": "greenhouse",
            "source_hash": _hash("a"),
            "source_id": "greenhouse:x:1",
        }
    ]


def test_parse_query_results_rejects_bad_header() -> None:
    with pytest.raises(ValueError, match="unexpected query-result header"):
        selection.parse_query_results("src,hash,id\ngreenhouse,x,y\n")


def test_parse_query_results_rejects_unknown_source_and_bad_hash() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        selection.parse_query_results(_csv(("mars", _hash("a"), "")))
    with pytest.raises(ValueError, match="bad source_hash"):
        selection.parse_query_results(_csv(("greenhouse", "shorthash", "")))


def test_empty_result_csv_raises() -> None:
    with pytest.raises(ValueError, match="no header row"):
        selection.parse_query_results("")


def test_build_selection_manifest_deterministic() -> None:
    rows = [
        {"source": "ashby", "source_hash": _hash("b"), "source_id": "ashby:x:2"},
        {
            "source": "greenhouse",
            "source_hash": _hash("a"),
            "source_id": "greenhouse:x:1",
        },
    ]
    first = selection.build_selection_manifest(rows, configuration_hash=_hash("c1"))
    second = selection.build_selection_manifest(rows, configuration_hash=_hash("c1"))
    assert first == second
    manifest = json.loads(first)
    # sorted by (source, source_hash); key reconstructed from the builders
    assert [p["source"] for p in manifest["postings"]] == ["ashby", "greenhouse"]
    assert manifest["postings"][1]["key"] == keys.raw_payload("greenhouse", _hash("a"))
    assert manifest["params"]["configuration_hash"] == _hash("c1")


def test_build_selection_manifest_filters_failed_ledger() -> None:
    store = MemorySink()
    run_key = "lake/manifests/runs/run_id="
    # run.json keys need a canonical UUIDv7 partition; build via the builder.
    run_id = "0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d70"
    store.objects[f"{run_key}{run_id}/run.json"] = json.dumps(
        {
            "run_id": run_id,
            "status": "partial",
            "failed_source_hashes": [_hash("b")],
        }
    ).encode()
    rows = [
        {"source": "ashby", "source_hash": _hash("b"), "source_id": "ashby:x:2"},
        {
            "source": "greenhouse",
            "source_hash": _hash("a"),
            "source_id": "greenhouse:x:1",
        },
    ]
    manifest = json.loads(
        selection.build_selection_manifest(
            rows, configuration_hash=_hash("c"), store=store
        )
    )
    assert [p["source_hash"] for p in manifest["postings"]] == [_hash("a")]


def test_ledger_reader_ignores_malformed_records() -> None:
    store = MemorySink()
    store.objects[
        "lake/manifests/runs/run_id=0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d70/run.json"
    ] = b"{bad"
    store.objects[
        "lake/manifests/runs/run_id=0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d71/flush=00001.json"
    ] = b"{}"
    assert selection.failed_source_hashes(store) == set()


def test_selected_rows_use_valid_raw_keys() -> None:
    rows = [{"source": "linkedin", "source_hash": _hash("k"), "source_id": ""}]
    manifest = json.loads(
        selection.build_selection_manifest(rows, configuration_hash=_hash("c2"))
    )
    # linkedin has no source_id — empty string, key still well-formed
    keys.validate_key(manifest["postings"][0]["key"])
    assert manifest["postings"][0]["source_id"] == ""


def test_select_uncleaned_sql_contains_all_sources() -> None:
    sql = selection.select_uncleaned_sql()
    for source in sorted(keys.RAW_SOURCES):
        assert f"raw_{source} " in sql, f"missing table for {source}"
    assert ":config" in sql
    assert "ORDER BY source, source_hash" in sql
