"""Tests for run-record merging + orphan healing (task 3.6, D10)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from backend.pipeline.cleaning import runs
from backend.pipeline.cleaning.writer import FlushWriter
from backend.storage import ids, keys
from tests.conftest import JournalSink


def _row(tag: str) -> dict[str, str]:
    return {
        "source_hash": hashlib.sha256(tag.encode()).hexdigest(),
        "variant": "llm",
        "version": "v1",
        "cleaned_text": f"cleaned {tag}",
        "configuration_hash": hashlib.sha256(b"config").hexdigest(),
        "pipeline_version": "dev",
    }


def _writer(store: JournalSink, **kw: Any) -> FlushWriter:
    return FlushWriter(store, run_id=ids.new_id(), pipeline_version="dev", **kw)


def test_merge_run_record_from_two_checkpoints() -> None:
    store = JournalSink()
    w = _writer(store, flush_rows=1)
    w.add(_row("a"), batch_id="b1", duration_s=1.0, cost_usd=0.001)
    w.add(_row("b"), batch_id="b2", duration_s=3.0, cost_usd=0.002)
    record = runs.merge_run_record(
        store,
        w._run_id,
        status="completed",
        input_dataset="data/sample_manifests/x.json",
        output_dataset_id=runs.dataset_id_for(),
        model="m",
        configuration_hash=hashlib.sha256(b"config").hexdigest(),
        pipeline_version="dev",
        started_at="2026-10-23T00:00:00+00:00",
        completed_at="2026-10-23T00:01:00+00:00",
        worker_count=4,
        cache_hits=7,
        failed=0,
    )
    assert record["counters"] == {"cleaned": 2, "cache_hits": 7, "failed": 0}
    assert record["latency"]["n_calls"] == 2
    assert record["latency"]["median_s"] == 2.0  # even-n median
    assert record["latency"]["max_s"] == 3.0
    assert record["latency"]["durations"] == [1.0, 3.0]
    assert len(record["flush_refs"]) == 2
    assert record["cost_usd"] == 0.003
    # run.json durable + re-readable from the store
    stored = json.loads(store.get(keys.manifest_run(w._run_id)).decode())
    assert stored["run_id"] == w._run_id
    assert stored["status"] == "completed"
    assert stored["failed_source_hashes"] == []


def test_merge_rejects_unknown_status() -> None:
    store = JournalSink()
    w = _writer(store)
    run_id = w._run_id
    with pytest.raises(ValueError, match="status"):
        runs.merge_run_record(
            store,
            run_id,
            status="running",
            input_dataset="m",
            output_dataset_id="d",
            model="m",
            configuration_hash="x",
            pipeline_version="dev",
            started_at="s",
            completed_at="c",
        )


def test_dataset_manifest_written() -> None:
    store = JournalSink()
    manifest = runs.write_dataset_manifest(
        store,
        dataset_id=runs.dataset_id_for(),
        input_dataset="data/sample_manifests/x.json",
        model="google/gemini-2.5-flash-lite",
        configuration_hash=hashlib.sha256(b"config").hexdigest(),
        pipeline_version="abc1234",
        record_count=42,
    )
    assert manifest["record_count"] == 42
    stored = json.loads(
        store.get(keys.manifest_dataset("cleaned_text-llm-v1")).decode()
    )
    assert stored["dataset_id"] == "cleaned_text-llm-v1"
    assert stored["pipeline_version"] == "abc1234"


def test_orphan_part_backfilled_as_recovered() -> None:
    store = JournalSink()
    w = _writer(store, flush_rows=1)
    w.add(_row("solo"), batch_id="b1", duration_s=2.0)
    cp = w.results[-1]
    checkpoint_key = keys.manifest_run_checkpoint(w._run_id, 1)
    # simulate the crash between the two writer puts
    store.objects.pop(checkpoint_key)
    assert not store.exists(checkpoint_key)

    healed = runs.heal_orphan_checkpoints(store)
    assert len(healed) == 1
    rec = healed[0]
    assert rec["recovered"] is True
    assert rec["trigger"] == "recovered"
    assert rec["row_count"] == 1
    assert rec["source_hashes"] == [cp["source_hashes"][0]]
    assert rec["latency"]["n_calls"] == 0  # rows carry no telemetry
    assert store.exists(checkpoint_key)

    # the healed checkpoint merges into the run record like any other
    record = runs.merge_run_record(
        store,
        w._run_id,
        status="completed",
        input_dataset="m",
        output_dataset_id=runs.dataset_id_for(),
        model="m",
        configuration_hash=hashlib.sha256(b"config").hexdigest(),
        pipeline_version="dev",
        started_at="2026-10-23T00:00:00+00:00",
        completed_at="2026-10-23T00:01:00+00:00",
    )
    assert record["counters"]["cleaned"] == 1
    assert record["latency"]["n_calls"] == 0

    # healing is idempotent: a checkpoint now exists
    assert runs.heal_orphan_checkpoints(store) == []


def test_no_false_positive_healing_on_healthy_runs() -> None:
    store = JournalSink()
    w = _writer(store, flush_rows=2)
    w.add(_row("a"), batch_id="b1", duration_s=0.5)
    w.add(_row("b"), batch_id="b2", duration_s=0.5)
    assert runs.heal_orphan_checkpoints(store) == []
