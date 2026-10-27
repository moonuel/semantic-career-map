"""Buffered flush writer for the embedding tier (tasks 3.3/3.4).

Same discipline as the cleaning stage's flush writer
(``backend/pipeline/cleaning/writer.py``), stage-native payload: completed
encode batches are buffered in the writer and committed as **one Parquet
part per flush** —
``lake/embeddings/model=<m>/variant=<v>/version=<vN>/part-<run-id>-<seq>.parquet``
— immediately followed by **one flush checkpoint** under
``manifests/runs/run_id=<run-id>/flush=<seq>.json`` covering exactly that
part's rows. Commit order is part → checkpoint (per-key PutObject
atomicity; no cross-key transaction assumed). The part→checkpoint gap is
the only place an orphan part can originate; the next run back-fills a
missing checkpoint from the part's contents (:mod:`reconcile`) — rows are
durable either way.

Flush triggers, any-of (a flush commits on the first that fires):

- ``rows`` — buffer holds ``flush_rows`` completed rows (``add_batch``)
- ``bytes`` — buffered ``embedding`` payload bytes reach ``flush_bytes``
  (``add_batch``)
- ``dwell`` — the oldest buffered batch has waited ``flush_seconds``
  (``maybe_flush``; the driver paces between CPU encode windows)
- ``drained`` — the runner's input queue is empty (``maybe_flush`` argument;
  the writer does not track the queue)
- ``stop`` — run ending (final ``flush_final`` of the remainder)

The buffer decouples part size from the encode batch size: small CPU
batches accumulate into warehouse-sized parts (defaults mirror the cleaning
stage: 128 rows, 60 s dwell, 2 MiB of payload bytes).

**Reuse decision (as-built, task 3.3):** the two stages' writers share the
key builders (``keys.embedding_part`` / ``keys.manifest_run_checkpoint``),
the trigger vocabulary, the single-seq numbering, and the part → checkpoint
commit-order invariant — and that is the honest reuse surface. A parametric
skeleton extracted from both proved to be adapter ceremony: the shared
bookkeeping is ~25 lines while the stage payloads diverge completely (the
cleaning writer carries per-row LLM latency/usage/cost and per-row schema
validation; this writer carries one typed vector matrix per batch). Drift
of the shared invariants is guarded instead by the lake shape pinners and
the two stages' pinned commit-order tests.

Checkpoint schema (stage-native): the cleaning checkpoint's fields minus
the LLM-only telemetry (latency, usage, cost) — ``run_id``, ``seq``,
``trigger``, ``part_key``, ``row_count``, ``batch_range``,
``source_hashes`` (sorted), ``configuration_hash``, ``counters.embedded``,
``written_at``. The recovery paths (:mod:`reconcile`) reuse this builder so
healed, reconciled, and live-flush checkpoints share one schema.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from backend.pipeline.embedding.config import EmbeddingConfig
from backend.storage import keys

FLUSH_ROWS_DEFAULT = 128
FLUSH_SECONDS_DEFAULT = 60.0
FLUSH_BYTES_DEFAULT = 2 * 1024 * 1024

# The embedding part schema (data.md §7 + provenance), in writer column
# order. This is the single source of truth for the row layout: the Glue
# registration derives its (name, Athena-type) column list from it so a
# rename/retype in part_bytes cannot silently drift the query layer.
PART_COLUMNS: list[tuple[str, str]] = [
    ("source_hash", "string"),
    ("embedding", "array<float>"),
    ("model", "string"),
    ("model_version", "string"),
    ("input_artifact", "string"),
    ("input_version", "string"),
    ("embedding_dimension", "bigint"),
    ("normalization", "string"),
    ("created_at", "timestamp"),
    ("configuration_hash", "string"),
    ("run_id", "string"),
]


def part_bytes(
    source_hashes: list[str],
    vectors: np.ndarray,  # (N, D) float32
    config: EmbeddingConfig,
    run_id: str,
    created_at: datetime,
    configuration_hash: str,
) -> bytes:
    """Serialize one provenance-complete Parquet part (data.md §7 columns).

    Args:
        source_hashes: Row source hashes in encode order.
        vectors: The ``(N, D)`` float32 matrix (order-aligned).
        config: The run's configuration.
        run_id: The minted run id.
        created_at: The run's wall-clock commit time.
        configuration_hash: The run's configuration hash (row column).

    Returns:
        The Parquet part bytes.
    """
    dimension = int(vectors.shape[1])
    # Build the embedding column from the flat float32 buffer + row offsets
    # (all rows share `dimension`), avoiding a numpy→Python-list round-trip
    # that would box every float on large batches.
    flat = np.ascontiguousarray(vectors, dtype=np.float32).reshape(-1)
    offsets = pa.array(range(0, len(flat) + 1, dimension), type=pa.int32())
    embedding = pa.ListArray.from_arrays(offsets, pa.array(flat, type=pa.float32()))
    table = pa.table(
        {
            "source_hash": pa.array(source_hashes, type=pa.string()),
            "embedding": embedding,
            "model": pa.array(
                [config.model_name] * len(source_hashes), type=pa.string()
            ),
            "model_version": pa.array(
                [config.model_revision] * len(source_hashes), type=pa.string()
            ),
            "input_artifact": pa.array(
                [config.input_artifact] * len(source_hashes), type=pa.string()
            ),
            "input_version": pa.array(
                [config.input_version] * len(source_hashes), type=pa.string()
            ),
            "embedding_dimension": pa.array(
                [dimension] * len(source_hashes), type=pa.int64()
            ),
            "normalization": pa.array(
                ["l2" if config.normalize else "none"] * len(source_hashes),
                type=pa.string(),
            ),
            "created_at": pa.array(
                [created_at] * len(source_hashes), type=pa.timestamp("us")
            ),
            "configuration_hash": pa.array(
                [configuration_hash] * len(source_hashes), type=pa.string()
            ),
            "run_id": pa.array([run_id] * len(source_hashes), type=pa.string()),
        }
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def build_flush_checkpoint(
    *,
    run_id: str,
    seq: int,
    trigger: str,
    part_key: str,
    source_hashes: list[str],
    configuration_hash: str,
    batch_range: list[str] | None,
    written_at: str,
) -> dict[str, Any]:
    """Build the embedding flush-checkpoint document (its sole definition).

    Args:
        run_id: The producing run's UUIDv7.
        seq: 1-based flush sequence within the run.
        trigger: The trigger label recorded in the checkpoint (also
            ``recovered`` / ``reconciled`` from the recovery paths).
        part_key: The committed part key (this checkpoint's subject).
        source_hashes: The rows' ``source_hash`` values (sorted here).
        configuration_hash: The run's configuration digest — rows of one
            part share it by construction.
        batch_range: The flush's first/last encode-batch ids (``None``
            for recovered/reconciled checkpoints).
        written_at: The commit wall clock (ISO-8601 UTC string).

    Returns:
        The checkpoint dict, also serialized by callers under
        :func:`backend.storage.keys.manifest_run_checkpoint`.
    """
    rows = len(source_hashes)
    return {
        "run_id": run_id,
        "seq": seq,
        "trigger": trigger,
        "part_key": part_key,
        "row_count": rows,
        "batch_range": batch_range,
        "source_hashes": sorted(source_hashes),
        "configuration_hash": configuration_hash,
        "counters": {"embedded": rows},
        "written_at": written_at,
    }


class _Pending:
    """One buffered encode batch."""

    __slots__ = ("added_at", "batch_id", "source_hashes", "vectors")

    def __init__(
        self,
        batch_id: str,
        source_hashes: list[str],
        vectors: np.ndarray,
        added_at: float,
    ) -> None:
        self.batch_id = batch_id
        self.source_hashes = source_hashes
        self.vectors = vectors
        self.added_at = added_at


class FlushWriter:
    """Buffers completed encode batches; commits part-then-checkpoint.

    Single-writer discipline: methods are lock-guarded; the driver funnels
    every completed batch through one writer instance (the lock is misuse
    insurance, as in the cleaning stage).

    Args:
        store: Put-capable object store (``ObjectStore`` or testing double).
        run_id: The owning run's canonical UUIDv7.
        config: The run's configuration (key + row fields).
        configuration_hash: The run's configuration digest (row column).
        created_at: The run's wall-clock commit time (row column).
        flush_rows: Row-threshold trigger.
        flush_bytes: Embedding-payload-bytes-threshold trigger.
        flush_seconds: Dwell-threshold trigger for the oldest buffered batch.
        now: Wall clock for checkpoint ``written_at`` stamps; ``None`` uses
            now (UTC). Deterministic tests inject.
        monotonic: Monotonic clock for dwell bookkeeping; ``None`` uses
            :func:`time.monotonic`. Deterministic tests inject.
    """

    def __init__(
        self,
        store,
        *,
        run_id: str,
        config: EmbeddingConfig,
        configuration_hash: str,
        created_at: datetime,
        flush_rows: int = FLUSH_ROWS_DEFAULT,
        flush_bytes: int = FLUSH_BYTES_DEFAULT,
        flush_seconds: float = FLUSH_SECONDS_DEFAULT,
        now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._store = store
        keys.embedding_part(  # validate run_id/layout up front
            config.model_segment, config.variant, config.version, run_id, 1
        )
        keys.manifest_run_checkpoint(run_id, 1)
        self._run_id = run_id
        self._config = config
        self._digest = configuration_hash
        self._created_at = created_at
        self._flush_rows = flush_rows
        self._flush_bytes = flush_bytes
        self._flush_seconds = flush_seconds
        self._now = now if now is not None else (lambda: datetime.now(UTC))
        self._monotonic = monotonic if monotonic is not None else time.monotonic
        self._lock = threading.Lock()
        self._buffer: list[_Pending] = []
        self._buffer_rows = 0
        self._buffer_bytes = 0
        self._seq = 0
        self.results: list[dict[str, Any]] = []

    def add_batch(
        self, *, batch_id: str, source_hashes: list[str], vectors: np.ndarray
    ) -> dict[str, Any] | None:
        """Buffer one completed encode batch; auto-flush on thresholds.

        Args:
            batch_id: This encode window's id (checkpoint provenance).
            source_hashes: The batch's row hashes (order-aligned).
            vectors: The ``(N, D)`` float32 matrix (order-aligned).

        Returns:
            The flush checkpoint (part written) if this add fired the
            rows/bytes trigger, else ``None``.

        Raises:
            ValueError: if hashes and vectors misalign.
        """
        with self._lock:
            if len(source_hashes) != vectors.shape[0]:
                raise ValueError(
                    f"source_hashes ({len(source_hashes)}) and vectors "
                    f"({vectors.shape[0]}) must align"
                )
            self._buffer.append(
                _Pending(
                    batch_id,
                    list(source_hashes),
                    np.asarray(vectors, dtype=np.float32),
                    self._monotonic(),
                )
            )
            self._buffer_rows += len(source_hashes)
            self._buffer_bytes += int(vectors.nbytes)
            if self._buffer_rows >= self._flush_rows:
                return self._flush("rows")
            if self._buffer_bytes >= self._flush_bytes:
                return self._flush("bytes")
            return None

    def maybe_flush(self, *, queue_empty: bool = False) -> dict[str, Any] | None:
        """Commit on the dwell trigger, or ``drained`` when run input ends.

        Args:
            queue_empty: Whether the run's input queue is drained.

        Returns:
            The flush checkpoint, or ``None`` when no trigger fired.
        """
        with self._lock:
            if not self._buffer:
                return None
            if self._monotonic() - self._buffer[0].added_at >= self._flush_seconds:
                return self._flush("dwell")
            if queue_empty:
                return self._flush("drained")
            return None

    def flush_final(self) -> dict[str, Any] | None:
        """Commit the remainder on stop (interrupt-safe final flush)."""
        with self._lock:
            if not self._buffer:
                return None
            return self._flush("stop")

    def _flush(self, trigger: str) -> dict[str, Any]:
        """Commit one part and its checkpoint, part → checkpoint order.

        Args:
            trigger: The trigger label recorded in the checkpoint.

        Returns:
            The checkpoint dict (also appended to :attr:`results`).
        """
        pending, self._buffer = self._buffer, []
        self._buffer_rows = 0
        self._buffer_bytes = 0
        self._seq += 1
        seq = self._seq
        source_hashes = [
            source_hash for batch in pending for source_hash in batch.source_hashes
        ]
        vectors = (
            np.concatenate([batch.vectors for batch in pending])
            if len(pending) > 1
            else pending[0].vectors
        )
        part_key = keys.embedding_part(
            self._config.model_segment,
            self._config.variant,
            self._config.version,
            self._run_id,
            seq,
        )
        self._store.put(  # part first…
            part_key,
            part_bytes(
                source_hashes,
                vectors,
                self._config,
                self._run_id,
                self._created_at,
                self._digest,
            ),
        )
        checkpoint = build_flush_checkpoint(
            run_id=self._run_id,
            seq=seq,
            trigger=trigger,
            part_key=part_key,
            source_hashes=source_hashes,
            configuration_hash=self._digest,
            batch_range=[pending[0].batch_id, pending[-1].batch_id],
            written_at=self._now().isoformat(),
        )
        payload = json.dumps(checkpoint, indent=2, ensure_ascii=False).encode("utf-8")
        self._store.put(  # …checkpoint after
            keys.manifest_run_checkpoint(self._run_id, seq), payload
        )
        self.results.append(checkpoint)
        return checkpoint
