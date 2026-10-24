"""Tests for the eval-results writer + orchestration (tasks 4.3/4.4).

Round-trip lineage is asserted on the written Parquet (identity fields,
rubric hash, evaluation version, creation timestamp); a second evaluation
under a different version lands a new prefix part without overwriting the
first. Mixed cleaning configurations must surface as an error unless
explicitly pinned. The summary table renders the exp-003-style layout.
"""

import hashlib
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.pipeline.eval import results as r
from backend.pipeline.eval import run as eval_run
from backend.storage import ids, keys
from tests.conftest import MemorySink

GOLDEN_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("source_hash", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("golden_text", pa.string()),
        pa.field("annotator", pa.string()),
        pa.field("created_at", pa.string()),
    ]
)

CLEANED_SCHEMA = pa.schema(
    [
        pa.field("source_hash", pa.string()),
        pa.field("cleaned_text", pa.string()),
        pa.field("configuration_hash", pa.string()),
    ]
)


def _results_prefix(variant: str = "exp003", version: str = "v1") -> str:
    """The results dataset prefix (same composition ``write_results`` uses)."""
    return (
        f"lake/eval/results/type=cleaning_quality/variant={variant}/version={version}/"
    )


def _seed_raw(store: MemorySink, source: str, source_hash: str, text: str) -> None:
    """Put the raw payload the hallucination metric reads."""
    payload = {"description_html": text}
    key = f"lake/raw/source={source}/sha256={source_hash}.json"
    store.put(key, json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())


def _seed_golden(store: MemorySink, rows: list[dict], *, version: str = "v1") -> None:
    """Write one golden dataset part with the given reference rows."""
    run_id = ids.new_id()
    key = keys.eval_dataset("cleaning_golden", "author", version, run_id, 1)
    table = pa.Table.from_pylist(rows, schema=GOLDEN_SCHEMA)
    sink = io.BytesIO()
    pq.write_table(table, sink)
    store.put(key, sink.getvalue())


def _seed_cleaned(store: MemorySink, rows: list[dict], *, version: str = "v1") -> None:
    """Write cleaned artifact rows (one part, like the cleaning runner's)."""
    run_id = ids.new_id()
    key = keys.artifact_part("cleaned_text", "llm", version, run_id, 1)
    table = pa.Table.from_pylist(rows, schema=CLEANED_SCHEMA)
    sink = io.BytesIO()
    pq.write_table(table, sink)
    store.put(key, sink.getvalue())


def _hash(label: str) -> str:
    """A stable 64-hex stand-in source/configuration hash."""
    return hashlib.sha256(label.encode()).hexdigest()


def _seed_two_postings(store: MemorySink) -> list[tuple[str, str]]:
    """Seed golden + cleaned + raw for two postings; return their hashes."""
    g1, g2, c1, _c2 = _hash("g1"), _hash("g2"), _hash("c1"), _hash("c2")
    _seed_golden(
        store,
        [
            {
                "source": "ashby",
                "source_hash": g1,
                "source_id": "ashby:board:1",
                "golden_text": "alpha beta",
                "annotator": "manual",
                "created_at": "2026-10-01T00:00:00Z",
            },
            {
                "source": "lever",
                "source_hash": g2,
                "source_id": "lever:board:2",
                "golden_text": "one two",
                "annotator": "manual",
                "created_at": "2026-10-01T00:00:00Z",
            },
        ],
    )
    _seed_cleaned(
        store,
        [
            {"source_hash": g1, "cleaned_text": "alpha beta", "configuration_hash": c1},
            {"source_hash": g2, "cleaned_text": "one", "configuration_hash": c1},
        ],
    )
    _seed_raw(store, "ashby", g1, "alpha beta raw words")
    _seed_raw(store, "lever", g2, "one two raw words")
    return [(g1, c1), (g2, c1)]


def test_evaluation_rows_carry_lineage(tmp_path) -> None:
    """Round-trip: the written part reads back with full lineage fields."""
    store = MemorySink()
    (g1, _c1), (g2, _) = _seed_two_postings(store)
    summary = eval_run.run_evaluation(
        store,
        golden=eval_run.load_golden(store),
        cleaned=eval_run.load_cleaned(store),
        run_id=ids.new_id(),
        created_at="2026-10-22T01:00:00Z",
    )
    part = summary.part_key
    assert part.startswith(_results_prefix(variant="exp003", version="v1"))
    table = pq.read_table(io.BytesIO(store.get(part)))
    rows = table.to_pylist()
    by_hash = {row["source_hash"]: row for row in rows}
    assert set(by_hash) == {g1, g2}
    for row in rows:
        assert row["evaluation_configuration_hash"] == r.evaluation_hash("v1")
        assert row["evaluation_version"] == "v1"
        assert row["created_at"] == "2026-10-22T01:00:00Z"
    assert by_hash[g1]["source_id"] == "ashby:board:1"
    assert by_hash[g1]["jaccard"] == 1.0
    assert by_hash[g1]["hallucination_count"] == 0
    assert by_hash[g1]["over_deletion_pct"] == 0.0
    assert not by_hash[g1]["over_deletion_flagged"]
    assert by_hash[g2]["jaccard"] == 0.5
    assert by_hash[g2]["over_deletion_pct"] == pytest.approx(50.0)
    assert by_hash[g2]["over_deletion_flagged"]


def test_second_evaluation_version_writes_new_prefix() -> None:
    """v1 and v2 results coexist; versions append, never overwrite."""
    store = MemorySink()
    (_g1, _c1), (_g2, _) = _seed_two_postings(store)
    first = eval_run.run_evaluation(
        store,
        run_id=ids.new_id(),
        created_at="2026-10-22T01:00:00Z",
    )
    second = eval_run.run_evaluation(
        store,
        run_id=ids.new_id(),
        created_at="2026-10-22T02:00:00Z",
        evaluation_version="v2",
    )
    assert first.part_key.startswith(_results_prefix(version="v1"))
    assert second.part_key.startswith(_results_prefix(version="v2"))
    v1_parts = store.list_keys(_results_prefix(version="v1"))
    v2_parts = store.list_keys(_results_prefix(version="v2"))
    assert len(v1_parts) == 1 and len(v2_parts) == 1
    rows2 = pq.read_table(io.BytesIO(store.get(second.part_key))).to_pylist()
    assert all(row["evaluation_version"] == "v2" for row in rows2)
    assert rows2[0]["evaluation_configuration_hash"] == r.evaluation_hash("v2")
    assert r.evaluation_hash("v2") != r.evaluation_hash("v1")


def test_dry_run_writes_nothing(tmp_path) -> None:
    store = MemorySink()
    _seed_two_postings(store)
    summary = eval_run.run_evaluation(store, write=False)
    assert summary.part_key == ""
    assert not store.list_keys(_results_prefix())
    assert len(summary.rows) == 2


def test_mixed_configurations_require_pin() -> None:
    """A cleaned version mixing configurations raises without a pin."""
    store = MemorySink()
    (g1, c1), (g2, _) = _seed_two_postings(store)
    _seed_cleaned(
        store,
        [
            {"source_hash": g1, "cleaned_text": "alpha beta", "configuration_hash": c1},
            {"source_hash": g2, "cleaned_text": "one", "configuration_hash": c1},
        ],
        version="v2",
    )
    # same version, different cleaning configuration — the mixed case
    _seed_cleaned(
        store,
        [
            {
                "source_hash": g1,
                "cleaned_text": "alpha",
                "configuration_hash": _hash("c8"),
            }
        ],
        version="v2",
    )
    with pytest.raises(ValueError, match="mixes configurations"):
        eval_run.load_cleaned(store, version="v2")
    pinned = eval_run.load_cleaned(store, version="v2", configuration_hash=c1)
    assert [row.source_hash for row in pinned] == [g1, g2]


def test_render_table_exp003_layout() -> None:
    store = MemorySink()
    _seed_two_postings(store)
    summary = eval_run.run_evaluation(store, write=False)
    table = eval_run.render_table(summary)
    assert "CLEANING QUALITY EVALUATION — 2 evaluated postings" in table
    assert "jaccard" in table
    assert "means  jaccard=" in table
    # The lever row over-deletes half the golden words → flagged
    assert "FLAG" in table
