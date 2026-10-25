"""Tests for the single-writer flush-watermark writer (task 3.4, D10).

Covers: every trigger (rows, bytes, dwell, drained, stop) commits
exactly one part + one checkpoint; part → checkpoint ordering; round-trip
Parquet (schema, row-per-posting, source_hash joins back to raw keys);
checkpoint accounting fields; row validation; distinct part keys per
flush.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

import pyarrow.parquet as pq
import pytest

from backend.pipeline.cleaning.writer import ROW_COLUMNS, FlushWriter
from backend.storage import ids, keys
from tests.conftest import JournalSink


def _hash(tag: str) -> str:
    """Deterministic 64-char hex digest for a tag."""
    return hashlib.sha256(tag.encode()).hexdigest()


def _row(tag: str, *, text: str = "clean text", **overrides: str) -> dict[str, str]:
    """A schema-complete row with deterministic identities."""
    row = {
        "source_hash": _hash(tag),
        "variant": "llm",
        "version": "v1",
        "cleaned_text": text,
        "configuration_hash": _hash("config"),
        "pipeline_version": "dev",
    }
    row.update(overrides)
    return row


def _writer(store: JournalSink, run_id: str, **kw: Any) -> FlushWriter:
    return FlushWriter(store, run_id=run_id, pipeline_version="dev", **kw)


def _make_raw(store: JournalSink, source: str, tag: str) -> str:
    """Place a raw payload object under its content key; return the hash."""
    source_hash = _hash(tag)
    store.objects[keys.raw_payload(source, source_hash)] = b'{"k": 1}'
    return source_hash


def test_rows_trigger_flushes_on_threshold() -> None:
    store = JournalSink()
    run_id = ids.new_id()
    w = _writer(store, run_id, flush_rows=3)
    h1 = _make_raw(store, "ashby", "a")
    h2 = _make_raw(store, "ashby", "b")
    h3 = _make_raw(store, "greenhouse", "c")
    assert w.add(_row("a", source_hash=h1), batch_id="b1") is None
    assert w.add(_row("b", source_hash=h2), batch_id="b2") is None
    cp = w.add(_row("c", source_hash=h3), batch_id="b3")
    assert cp is not None and cp["trigger"] == "rows"
    assert w.results == [cp]
    assert cp["row_count"] == 3
    assert cp["part_key"] == keys.artifact_part("cleaned_text", "llm", "v1", run_id, 1)


def test_bytes_trigger_flushes_immediately() -> None:
    store = JournalSink()
    w = _writer(store, ids.new_id(), flush_bytes=64)
    cp = w.add(_row("big", text="x" * 80), batch_id="b1")
    assert cp is not None and cp["trigger"] == "bytes"
    assert store.exists(cp["part_key"])


def test_dwell_trigger_uses_elapsed_time() -> None:
    store = JournalSink()
    w = _writer(store, ids.new_id(), flush_seconds=0.0)
    w.add(_row("t"), batch_id="b1")
    cp = w.maybe_flush()
    assert cp is not None and cp["trigger"] == "dwell"
    assert w.maybe_flush() is None  # no buffer → nothing fires


def test_dwell_does_not_fire_before_threshold() -> None:
    store = JournalSink()
    w = _writer(store, ids.new_id(), flush_seconds=3600.0)
    w.add(_row("t"), batch_id="b1")
    assert w.maybe_flush() is None
    w._buffer[0].added_at -= 3601.0  # simulate elapsed dwell
    cp = w.maybe_flush()
    assert cp is not None and cp["trigger"] == "dwell"


def test_drained_trigger_commits_buffer() -> None:
    store = JournalSink()
    w = _writer(store, ids.new_id())
    w.add(_row("d"), batch_id="b1")
    cp = w.maybe_flush(queue_empty=True)
    assert cp is not None and cp["trigger"] == "drained"
    assert w.flush_final() is None  # drained buffer not double-committed


def test_stop_trigger_flushes_remainder_idempotently() -> None:
    store = JournalSink()
    w = _writer(store, ids.new_id())
    assert w.flush_final() is None  # nothing buffered
    w.add(_row("s"), batch_id="b1")
    cp = w.flush_final()
    assert cp is not None and cp["trigger"] == "stop"
    assert w.flush_final() is None  # buffer empty again


def test_round_trip_and_join_back_to_raw() -> None:
    store = JournalSink()
    w = _writer(store, ids.new_id(), flush_rows=2)
    h1 = _make_raw(store, "ashby", "a")
    h2 = _make_raw(store, "greenhouse", "c")
    w.add(_row("a", source_hash=h1, text="Ashby cleaned text"), batch_id="b1")
    cp = w.add(
        _row("c", source_hash=h2, text="Greenhouse cleaned text"),
        batch_id="b0",
    )
    assert cp is not None
    table = pq.read_table(io.BytesIO(store.get(cp["part_key"])))
    assert table.column_names == list(ROW_COLUMNS)
    assert table.num_rows == 2  # one row per posting
    rows = table.to_pylist()
    # every artifact row joins back to a raw payload object in the store
    source_of = {h1: "ashby", h2: "greenhouse"}
    for r in rows:
        raw_key = keys.raw_payload(source_of[r["source_hash"]], r["source_hash"])
        assert store.exists(raw_key)
    assert [r["source_hash"] for r in rows] == [h1, h2]  # completion order
    by_hash = {r["source_hash"]: r for r in rows}
    assert by_hash[h1]["cleaned_text"] == "Ashby cleaned text"
    assert by_hash[h2]["configuration_hash"] == _hash("config")


def test_schema_fields_populated() -> None:
    store = JournalSink()
    w = _writer(store, ids.new_id(), flush_rows=1)
    h = _make_raw(store, "lever", "a")
    cp = w.add(_row("a", source_hash=h), batch_id="b1")
    assert cp is not None
    r = pq.read_table(io.BytesIO(store.get(cp["part_key"]))).to_pylist()[0]
    assert r["variant"] == "llm" and r["version"] == "v1"
    assert r["pipeline_version"] == "dev"
    assert len(r["configuration_hash"]) == 64
    assert r["created_at"].endswith("+00:00")  # UTC-stamped ISO-8601
    assert r["cleaned_text"] == "clean text"


def test_checkpoint_accounting_fields() -> None:
    store = JournalSink()
    w = _writer(store, ids.new_id(), flush_rows=2)
    h1 = _make_raw(store, "ashby", "a")
    h2 = _make_raw(store, "ashby", "b")
    w.add(
        _row("a", source_hash=h1),
        batch_id="b1",
        duration_s=1.0,
        usage={"prompt_tokens": 10, "completion_tokens": 2},
        cost_usd=0.0001,
    )
    cp = w.add(
        _row("b", source_hash=h2),
        batch_id="b2",
        duration_s=3.0,
        usage={"prompt_tokens": 5, "completion_tokens": 1},
        cost_usd=0.0002,
    )
    assert cp is not None
    assert cp["counters"] == {"cleaned": 2}
    assert cp["latency"] == {
        "n_calls": 2,
        "median_s": 2.0,
        "p90_s": 3.0,
        "max_s": 3.0,
        "durations": [1.0, 3.0],
    }
    assert cp["usage"] == {"prompt_tokens": 15, "completion_tokens": 3}
    assert cp["cost_usd"] == 0.0003
    assert cp["batch_range"] == ["b1", "b2"]
    assert cp["source_hashes"] == sorted([h1, h2])
    assert cp["part_key"].endswith(".parquet")


def test_part_then_checkpoint_write_order_per_flush() -> None:
    store = JournalSink()
    run_id = ids.new_id()
    w = _writer(store, run_id)
    for i in range(2):
        h = _make_raw(store, "ashby", f"t{i}")
        w.add(_row("t", source_hash=h), batch_id=f"b{i}", duration_s=0.5)
        w.flush_final()
    w.flush_final()
    # every checkpoint appears in the journal strictly after its part
    for seq, cp in enumerate(w.results, start=1):
        part_idx = store.journal.index(cp["part_key"])
        cp_idx = store.journal.index(keys.manifest_run_checkpoint(run_id, seq))
        assert part_idx < cp_idx
    assert len(store.journal) == 4  # 2 parts + 2 checkpoints


def test_row_schema_validation_fails_loudly_at_add() -> None:
    store = JournalSink()
    w = _writer(store, ids.new_id())
    good = _row("x")
    bad_missing = {k: v for k, v in good.items() if k != "cleaned_text"}
    bad_extra = {**good, "posting_id": "nope"}
    bad_hash = {**good, "source_hash": "short"}
    for bad in (bad_missing, bad_extra, bad_hash):
        with pytest.raises(ValueError):
            w.add(bad, batch_id="b")
    assert store.journal == []  # nothing ever buffered/committed


def test_distinct_part_keys_across_flushes() -> None:
    store = JournalSink()
    w = _writer(store, ids.new_id(), flush_rows=1)
    h1 = _make_raw(store, "ashby", "a")
    h2 = _make_raw(store, "ashby", "b")
    cp1 = w.add(_row("a", source_hash=h1), batch_id="b1")
    cp2 = w.add(_row("b", source_hash=h2), batch_id="b2")
    assert cp1 is not None and cp2 is not None
    assert cp1["part_key"] != cp2["part_key"]
    assert cp2["part_key"].endswith("-00002.parquet")
    keys.validate_key(cp2["part_key"])
