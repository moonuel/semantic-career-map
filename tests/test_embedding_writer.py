"""Flush-writer tests for the embedding tier (tasks 3.3/3.4): trigger
vocabulary, part→checkpoint commit ordering, merged-batch parts, checkpoint
document shape. Recovery paths (orphan healing) live in
``tests/test_embedding_recovery.py``."""

from __future__ import annotations

import io
from datetime import UTC, datetime

import numpy as np
import pyarrow.parquet as pq
import pytest

from backend.pipeline.embedding.config import EmbeddingConfig
from backend.pipeline.embedding.writer import (
    FlushWriter,
    build_flush_checkpoint,
)
from backend.storage import keys
from tests.conftest import JournalSink, MemorySink


def _config(**kw) -> EmbeddingConfig:
    defaults = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_revision": "rev-1",
        "model_segment": "all-minilm-l6-v2",
        "variant": "minilm",
        "text_variant": "variant_text",
        "batch_size": 2,
        "version": "v1",
        "sql_text": "SELECT 1",  # identity-neutral for the writer
    }
    defaults.update(kw)
    return EmbeddingConfig(**defaults)


def _vectors(n: int, fill: float) -> np.ndarray:
    return np.full((n, 8), fill, dtype=np.float32)


class _Clock:
    """Deterministic monotonic clock (dwell bookkeeping)."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


CREATED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
RUN_ID = "01020304-0000-7f7e-a000-000000000001"
RUN_ID_2 = "01020304-0000-7f7e-a000-000000000002"


def _writer(store, **kw) -> FlushWriter:
    config = _config()
    return FlushWriter(
        store,
        run_id=kw.pop("run_id", RUN_ID),
        config=config,
        configuration_hash=kw.pop("digest", "b" * 64),
        created_at=kw.pop("created_at", None) or CREATED_AT,
        **kw,
    )


def test_row_threshold_flushes_merged_batches(memory_sink: MemorySink) -> None:
    writer = _writer(memory_sink, flush_rows=4)
    assert (
        writer.add_batch(
            batch_id="encode-0001",
            source_hashes=["a" * 64, "b" * 64],
            vectors=_vectors(2, 1.0),
        )
        is None
    )  # 2 rows buffered
    checkpoint = writer.add_batch(
        batch_id="encode-0002",
        source_hashes=["c" * 64, "d" * 64],
        vectors=_vectors(2, 2.0),
    )
    assert checkpoint is not None  # rows trigger fired at 4 buffered
    assert checkpoint["trigger"] == "rows"
    assert checkpoint["row_count"] == 4
    # both batches merged into one part, vectors order-aligned
    part_key = checkpoint["part_key"]
    table = pq.read_table(io.BytesIO(memory_sink.get(part_key)))
    assert table["source_hash"].to_pylist() == ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
    vectors = np.stack(table["embedding"].to_pylist())
    assert vectors[0, 0] == 1.0 and vectors[3, 0] == 2.0
    assert writer.results == [checkpoint] and writer._buffer == []


def test_commit_order_is_part_then_checkpoint(journal: JournalSink) -> None:
    writer = _writer(journal, flush_rows=1)
    writer.add_batch(
        batch_id="encode-0001", source_hashes=["a" * 64], vectors=_vectors(1, 1.0)
    )
    part_position = journal.journal.index(
        keys.embedding_part("all-minilm-l6-v2", "minilm", "v1", RUN_ID, 1)
    )
    check_position = journal.journal.index(keys.manifest_run_checkpoint(RUN_ID, 1))
    assert part_position < check_position  # strict commit-order invariant


def test_bytes_trigger_fires_before_rows(memory_sink: MemorySink) -> None:
    writer = _writer(memory_sink, flush_rows=100, flush_bytes=32)
    checkpoint = writer.add_batch(
        batch_id="encode-0001",
        source_hashes=["a" * 64, "b" * 64],
        vectors=_vectors(2, 1.0),  # 2*8*4 = 64 payload bytes >= 32
    )
    assert checkpoint is not None and checkpoint["trigger"] == "bytes"


def test_dwell_trigger_fires_on_old_buffer(memory_sink: MemorySink) -> None:
    clock = _Clock()
    writer = _writer(memory_sink, flush_seconds=60.0, monotonic=clock)
    assert (
        writer.add_batch(
            batch_id="encode-0001",
            source_hashes=["a" * 64],
            vectors=_vectors(1, 1.0),
        )
        is None
    )
    clock.now += 61.0  # oldest batch older than the dwell
    checkpoint = writer.maybe_flush()
    assert checkpoint is not None and checkpoint["trigger"] == "dwell"


def test_drained_and_stop_are_distinct_triggers(memory_sink: MemorySink) -> None:
    drained_writer = _writer(memory_sink)
    assert (
        drained_writer.add_batch(
            batch_id="encode-0001",
            source_hashes=["a" * 64],
            vectors=_vectors(1, 1.0),
        )
        is None
    )
    flushed = drained_writer.maybe_flush(queue_empty=True)
    assert flushed is not None and flushed["trigger"] == "drained"

    stop_writer = _writer(memory_sink, run_id=RUN_ID_2)
    assert (
        stop_writer.add_batch(
            batch_id="encode-0001",
            source_hashes=["b" * 64],
            vectors=_vectors(1, 1.0),
        )
        is None
    )
    final = stop_writer.flush_final()
    assert final is not None and final["trigger"] == "stop"
    assert stop_writer.flush_final() is None  # empty buffer stops cleanly
    assert stop_writer.maybe_flush(queue_empty=True) is None


def test_misaligned_batch_is_rejected(memory_sink: MemorySink) -> None:
    writer = _writer(memory_sink)
    with pytest.raises(ValueError, match="align"):
        writer.add_batch(
            batch_id="encode-0001",
            source_hashes=["a" * 64, "b" * 64],
            vectors=_vectors(1, 1.0),
        )


def test_checkpoint_document_shape() -> None:
    checkpoint = build_flush_checkpoint(
        run_id="01020304-0000-7f7e-a000-000000000003",
        seq=2,
        trigger="rows",
        part_key="lake/embeddings/model=all-minilm-l6-v2/variant=minilm/"
        "version=v1/part-01020304-0000-7f7e-a000-000000000003-00002.parquet",
        source_hashes=["d" * 64, "c" * 64],
        configuration_hash="b" * 64,
        batch_range=["encode-0003", "encode-0004"],
        written_at="2026-01-02T03:04:05+00:00",
    )
    assert checkpoint == {
        "run_id": "01020304-0000-7f7e-a000-000000000003",
        "seq": 2,
        "trigger": "rows",
        "part_key": "lake/embeddings/model=all-minilm-l6-v2/variant=minilm/"
        "version=v1/part-01020304-0000-7f7e-a000-000000000003-00002.parquet",
        "row_count": 2,
        "batch_range": ["encode-0003", "encode-0004"],
        "source_hashes": ["c" * 64, "d" * 64],  # sorted
        "configuration_hash": "b" * 64,
        "counters": {"embedded": 2},
        "written_at": "2026-01-02T03:04:05+00:00",
    }
