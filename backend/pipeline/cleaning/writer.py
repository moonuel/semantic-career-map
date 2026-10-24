"""Single-writer flush-watermark Parquet writer (task 3.4, design D10).

The durability unit of the cleaning stage: completed posting results are
buffered in the writer and committed as **one Parquet part per flush** —
``artifacts/type=<type>/variant=<variant>/version=<vN>/part-<run-id>-<seq>.parquet``
— immediately followed by **one flush checkpoint** under
``manifests/runs/run_id=<run-id>/flush=<seq>.json`` covering exactly that
part's rows. Commit order is part → checkpoint (S3 per-key PutObject
atomicity + strong read-after-write; no cross-key transaction assumed —
D10). A part without a checkpoint (crash between the two writes) is an
orphan the next run's aggregation back-fills (task 3.6); the two writes
here are the only place that can create one.

Flush triggers, any-of (a flush commits on the first that fires):

- ``rows`` — buffer holds ``flush_rows`` completed rows (checked in
  :meth:`FlushWriter.add`)
- ``bytes`` — buffered ``cleaned_text`` bytes reach ``flush_bytes``
  (checked in :meth:`FlushWriter.add`)
- ``dwell`` — the oldest buffered row has waited ``flush_seconds``
  (checked in :meth:`FlushWriter.maybe_flush`)
- ``drained`` — the runner's input queue is empty (a
  :meth:`FlushWriter.maybe_flush` argument; runners know when they are
  drained, the writer does not track the queue)
- ``stop`` — run ending (a final :meth:`FlushWriter.flush_final` of the
  remainder)

Single-writer discipline: these methods are lock-guarded and may be
called from any thread, but the runner (task 3.7) funnels all results
through one writer context — the discipline is a runner invariant, the
lock is misuse insurance, matching the exp-008 lesson that two writers
corrupt flat caches (design D10 keeps the lesson; the corruption vector
here is interleaved parts, not a JSON file). Loss window = in-flight LLM
calls: a flush of the remainder on stop keeps completed-not-yet-flushed
work durable, and replay of unflushed postings is idempotent under the
artifact-row cache (task 3.5).

Checkpoint fields (design D10): ``batch_range``, ``source_hashes``,
``row_count``, counters, latency aggregates, usage, cost, ``part_key``.
Cache hits never reach the writer — a hit resolves to an existing row in
an existing part (task 3.5), so every row here is one LLM-cleaned
posting (``counters.cleaned == row_count``; cache-hit/failed counters
live at run level, task 3.6).
"""

from __future__ import annotations

import datetime
import io
import json
import statistics
import threading
import time
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from backend.storage import keys

# The §5–6 row schema for ``type=cleaned_text`` (cleaned-artifacts spec).
ROW_COLUMNS: tuple[str, ...] = (
    "source_hash",
    "variant",
    "version",
    "cleaned_text",
    "configuration_hash",
    "pipeline_version",
    "created_at",
)

# D10 defaults: 128 rows, 60 s dwell, 2 MiB of cleaned bytes.
FLUSH_ROWS_DEFAULT = 128
FLUSH_SECONDS_DEFAULT = 60.0
FLUSH_BYTES_DEFAULT = 2 * 1024 * 1024


class PutSink(Protocol):
    """The put-only storage surface the writer needs."""

    def put(self, key: str, data: bytes) -> None: ...


def latency_aggregates(durations: list[float]) -> dict[str, Any]:
    """Compute call-latency aggregates over LLM-cleaned rows only.

    Args:
        durations: Per-row LLM call durations (cache-hit rows contribute
            none; every row here is LLM-cleaned).

    Returns:
        ``n_calls`` with median/p90/max ``duration_s`` in seconds, plus
        the sorted ``durations`` samples — per-flush aggregates alone
        cannot be merged into exact run-level aggregates (task 3.6), so
        the raw samples ride the checkpoint (n ≤ flush_rows, trivial).
    """
    if not durations:
        return {
            "n_calls": 0,
            "median_s": None,
            "p90_s": None,
            "max_s": None,
            "durations": [],
        }
    ordered = sorted(durations)
    p90 = ordered[min(len(ordered) - 1, round(0.9 * len(ordered)) - 1)]
    return {
        "n_calls": len(ordered),
        "median_s": statistics.median(ordered),
        "p90_s": p90,
        "max_s": ordered[-1],
        "durations": ordered,
    }


def build_flush_checkpoint(
    *,
    run_id: str,
    seq: int,
    trigger: str,
    part_key: str,
    source_hashes: list[str],
    configuration_hash: str,
    batch_range: list[str] | None,
    durations: list[float],
    usage: dict[str, int],
    cost_usd: float,
) -> dict[str, Any]:
    """Build the flush-checkpoint document (the sole definition of its schema).

    Args:
        run_id: The producing run's UUIDv7.
        seq: 1-based flush sequence within the run.
        trigger: The trigger label recorded in the checkpoint.
        part_key: The committed part key (this checkpoint's subject).
        source_hashes: The rows' ``source_hash`` values (sorted here).
        configuration_hash: The run's configuration digest — rows of one
            part share it by construction; the cache index (task 3.5)
            reads it here instead of downloading the part.
        batch_range: The flush's first/last batch ids (``None`` for a
            recovered checkpoint).
        durations: Per-row LLM call durations (recovered checkpoints pass
            ``[]`` — rows carry no call telemetry).
        usage: Token usage sums (``prompt_tokens`` / ``completion_tokens``;
            zeros for recovered checkpoints).
        cost_usd: The flush's total LLM cost (rounded internally).

    Returns:
        The checkpoint dict, also serialized by callers under
        :func:`keys.manifest_run_checkpoint`. The orphan-heal path
        (:mod:`runs`) reuses this builder so healed and live-flush
        checkpoints share one schema.
    """
    cleaned = len(source_hashes)
    return {
        "run_id": run_id,
        "seq": seq,
        "trigger": trigger,
        "part_key": part_key,
        "row_count": cleaned,
        "batch_range": batch_range,
        "source_hashes": sorted(source_hashes),
        "configuration_hash": configuration_hash,
        "counters": {"cleaned": cleaned},
        "latency": latency_aggregates(durations),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        },
        "cost_usd": round(cost_usd, 8),
        "written_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }


class _Pending:
    """One buffered row plus its scheduling/provenance annotations."""

    __slots__ = ("added_at", "batch_id", "cost_usd", "duration_s", "row", "usage")

    def __init__(
        self,
        row: dict[str, str],
        *,
        batch_id: str,
        duration_s: float | None,
        usage: dict[str, int] | None,
        cost_usd: float | None,
        added_at: float,
    ) -> None:
        self.row = row
        self.batch_id = batch_id
        self.duration_s = duration_s
        self.usage = usage
        self.cost_usd = cost_usd
        self.added_at = added_at


class FlushWriter:
    """Buffers completed posting results; commits part-then-checkpoint.

    Args:
        store: Put-capable object store (``ObjectStore`` or a testing
            double).
        run_id: Canonical UUIDv7 minter identity of the owning run.
        pipeline_version: Git sha (or dev tag) of the producing code.
        artifact_type: Governed artifact family (default ``cleaned_text``).
        variant: Producing method (default ``llm``).
        version: Version token (default ``v1``).
        flush_rows: Row-threshold trigger.
        flush_bytes: Cleaned-bytes-threshold trigger.
        flush_seconds: Dwell-threshold trigger for the oldest row.
    """

    def __init__(
        self,
        store: PutSink,
        *,
        run_id: str,
        pipeline_version: str,
        artifact_type: str = "cleaned_text",
        variant: str = "llm",
        version: str = "v1",
        flush_rows: int = FLUSH_ROWS_DEFAULT,
        flush_bytes: int = FLUSH_BYTES_DEFAULT,
        flush_seconds: float = FLUSH_SECONDS_DEFAULT,
    ) -> None:
        self._store = store
        self._run_id = run_id
        keys.manifest_run_checkpoint(run_id, 1)  # validate run_id up front
        keys.artifact_part(artifact_type, variant, version, run_id, 1)
        self._artifact_type = artifact_type
        self._variant = variant
        self._version = version
        self._pipeline_version = pipeline_version
        self._flush_rows = flush_rows
        self._flush_bytes = flush_bytes
        self._flush_seconds = flush_seconds
        self._lock = threading.Lock()
        self._buffer: list[_Pending] = []
        self._buffer_bytes = 0
        self._seq = 0
        self.results: list[dict[str, Any]] = []

    def add(
        self,
        row: dict[str, str],
        *,
        batch_id: str,
        duration_s: float | None = None,
        usage: dict[str, int] | None = None,
        cost_usd: float | None = None,
    ) -> dict[str, Any] | None:
        """Buffer one completed LLM-cleaned posting; auto-flush on thresholds.

        Args:
            row: The row dict — exactly the ``ROW_COLUMNS`` fields except
                ``created_at``, which the writer stamps on buffer entry.
            batch_id: Runner-assigned scheduling unit this posting came in
                with (checkpoint provenance; never part of the row).
            duration_s: LLM call latency for the row (every row here is
                LLM-cleaned, so runners pass a real measurement).
            usage: The call's token counts (``prompt_tokens`` /
                ``completion_tokens``).
            cost_usd: The call's priced cost, when the caller computes one.

        Returns:
            The flush checkpoint (part written) if this add fired the
            row/bytes trigger, else ``None``.

        Raises:
            ValueError: if the row's fields do not match the row schema or
                a hash field is not 64-char lowercase hex.
        """
        with self._lock:
            self._validate_row(row)
            stamped = dict(row)
            stamped["created_at"] = datetime.datetime.now(datetime.UTC).isoformat()
            self._buffer.append(
                _Pending(
                    stamped,
                    batch_id=batch_id,
                    duration_s=duration_s,
                    usage=usage,
                    cost_usd=cost_usd,
                    added_at=time.monotonic(),
                )
            )
            self._buffer_bytes += len(stamped["cleaned_text"].encode("utf-8"))
            if len(self._buffer) >= self._flush_rows:
                return self._flush("rows")
            if self._buffer_bytes >= self._flush_bytes:
                return self._flush("bytes")
            return None

    def maybe_flush(self, *, queue_empty: bool = False) -> dict[str, Any] | None:
        """Commit on the dwell or drained trigger (runner-side pacing hook).

        Args:
            queue_empty: Whether the runner's input queue is drained.

        Returns:
            The flush checkpoint, or ``None`` when no trigger fired.
        """
        with self._lock:
            if not self._buffer:
                return None
            if time.monotonic() - self._buffer[0].added_at >= self._flush_seconds:
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

    def _validate_row(self, row: dict[str, str]) -> None:
        """Fail loudly at buffer time, not at flush time.

        Args:
            row: The candidate row.

        Raises:
            ValueError: on a field mismatch (type or name) or a bad hex
                hash: a drifted runner must surface the moment it hands a
                row over.
        """
        if not isinstance(row, dict):
            raise TypeError(f"row must be a dict, got {type(row).__name__}")
        expected = set(ROW_COLUMNS) - {"created_at"}
        if set(row) != expected:
            raise ValueError(
                f"row fields {sorted(row)} do not match the cleaned_text "
                f"schema {sorted(expected)}"
            )
        if not keys.is_sha256_hex(row["source_hash"]):
            raise ValueError(f"row source_hash {row['source_hash']!r} is not 64-hex")
        if not keys.is_sha256_hex(row["configuration_hash"]):
            raise ValueError(
                f"row configuration_hash {row['configuration_hash']!r} is not 64-hex"
            )

    def _flush(self, trigger: str) -> dict[str, Any]:
        """Commit one part and its checkpoint, part → checkpoint order.

        Args:
            trigger: The trigger label recorded in the checkpoint.

        Returns:
            The checkpoint dict (also appended to :attr:`results`).
        """
        pending, self._buffer = self._buffer, []
        self._buffer_bytes = 0
        self._seq += 1
        seq = self._seq
        rows = [p.row for p in pending]
        part_key = keys.artifact_part(
            self._artifact_type,
            self._variant,
            self._version,
            self._run_id,
            seq,
        )
        table = pa.table({col: [r[col] for r in rows] for col in ROW_COLUMNS})
        sink = io.BytesIO()
        pq.write_table(table, sink)
        self._store.put(part_key, sink.getvalue())  # part first…
        checkpoint = build_flush_checkpoint(
            run_id=self._run_id,
            seq=seq,
            trigger=trigger,
            part_key=part_key,
            source_hashes=[r["source_hash"] for r in rows],
            configuration_hash=rows[0]["configuration_hash"],
            batch_range=[pending[0].batch_id, pending[-1].batch_id],
            durations=[p.duration_s for p in pending if p.duration_s is not None],
            usage={
                "prompt_tokens": sum(
                    (p.usage or {}).get("prompt_tokens", 0) for p in pending
                ),
                "completion_tokens": sum(
                    (p.usage or {}).get("completion_tokens", 0) for p in pending
                ),
            },
            cost_usd=sum(p.cost_usd for p in pending if p.cost_usd is not None),
        )
        payload = json.dumps(checkpoint, indent=2, ensure_ascii=False).encode("utf-8")
        checkpoint_key = keys.manifest_run_checkpoint(self._run_id, seq)
        self._store.put(checkpoint_key, payload)  # …checkpoint after
        self.results.append(checkpoint)
        return checkpoint
