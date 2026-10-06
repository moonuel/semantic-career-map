"""Batch-scrape ingestion manifest builder.

One Parquet manifest per batched scrape run (data.md §2.1, §3, §4): the run
manifest and the ingestion-event log merged into a single append-only object
placed by :func:`backend.storage.keys.manifest_ingestion`. Each row records one
payload observation; run-level fields repeat on every row because Parquet's
dictionary/RLE encoding makes repetition near-free, and flat rows keep Athena
trivial. Computing the manifest is pure — a ``(table -> bytes)`` builder with
no store access — so it is unit-testable without S3; the caller puts the bytes
at the key minted for its ingestion id.

The schema follows the approved batch-manifest design:

- Run-level: ``schema_version``, ``ingestion_id``, ``ingested_on``,
  ``pipeline_version``, ``configuration_hash``, ``run_started_at``,
  ``completed_at``, ``status``, ``boards_attempted``, ``boards_failed``,
  ``board_status`` (board slug -> ``ok`` or error summary).
- Observation-level: ``source``, ``action`` (``stored`` vs ``duplicate`` —
  write-time dedup made observable, §4), ``content_sha256`` (the raw object's
  key leaf and join handle), ``source_id``, ``board``, ``employer``,
  ``observed_at``.

Payload content itself is deliberately not copied: it lives in
``lake/raw`` and joins back via ``content_sha256``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import pyarrow as pa
import pyarrow.parquet as pq

from backend.storage.ids import is_canonical_uuid7, is_sha256_hex

SCHEMA_VERSION = 1

ACTIONS = frozenset({"stored", "duplicate"})
STATUSES = frozenset({"completed", "partial", "failed"})

_SCHEMA = pa.schema(
    [
        ("schema_version", pa.uint8()),
        ("ingestion_id", pa.string()),
        ("ingested_on", pa.date32()),
        ("source", pa.string()),
        ("pipeline_version", pa.string()),
        ("configuration_hash", pa.string()),
        ("run_started_at", pa.timestamp("ms", tz="UTC")),
        ("completed_at", pa.timestamp("ms", tz="UTC")),
        ("status", pa.string()),
        ("boards_attempted", pa.int32()),
        ("boards_failed", pa.int32()),
        ("board_status", pa.map_(pa.string(), pa.string())),
        ("action", pa.string()),
        ("content_sha256", pa.string()),
        ("source_id", pa.string()),
        ("board", pa.string()),
        ("employer", pa.string()),
        ("observed_at", pa.timestamp("ms", tz="UTC")),
    ]
)


@dataclass(frozen=True)
class BatchObservation:
    """One payload observed during a batch scrape.

    Attributes:
        source: Raw source allowlist value; joins the raw key partition.
        action: ``stored`` if the run put a new raw object, ``duplicate`` if
            the content hash was already present (write-time dedup, §4).
        content_sha256: SHA-256 of the verbatim payload; ``sha256=<hash>``
            leaf of the raw key this observation refers to.
        source_id: Native posting id as ``<ats>:<board>:<id>``, when the
            source provides one.
        board: Board slug the batch fetched the payload from.
        employer: Employer name from the batch spec.
        observed_at: When this payload was fetched (UTC).
    """

    source: str
    action: str
    content_sha256: str
    source_id: str | None
    board: str
    employer: str | None
    observed_at: datetime


def build_ingestion_manifest(
    observations: Sequence[BatchObservation],
    *,
    ingestion_id: str,
    ingested_on: date,
    pipeline_version: str,
    configuration_hash: str,
    run_started_at: datetime,
    completed_at: datetime,
    status: str,
    boards_attempted: int,
    boards_failed: int,
    board_status: Mapping[str, str],
) -> bytes:
    """Render one batch scrape as Parquet manifest bytes.

    Args:
        observations: One :class:`BatchObservation` per payload seen this run.
        ingestion_id: Canonical UUIDv7 minted once per run; also names the
            object key.
        ingested_on: Run date (UTC); also the key's day partition.
        pipeline_version: Code identity of the run (e.g. git sha).
        configuration_hash: 64-char hex SHA-256 of the run spec (the board
            list and builder), tying the run to §3-style configs.
        run_started_at: UTC timestamp the run started.
        completed_at: UTC timestamp the run finished.
        status: ``completed``, ``partial`` (finished with board failures), or
            ``failed`` (crashed with zero or partial observations).
        boards_attempted: Boards the batch tried to fetch.
        boards_failed: Boards whose fetch failed.
        board_status: Board slug -> ``ok`` or a one-line error summary, so
            partial runs are observable without extra files.

    Returns:
        The Parquet bytes to put at ``keys.manifest_ingestion(...)``.

    Raises:
        ValueError: if any field violates the manifest contract.
    """
    if not is_canonical_uuid7(ingestion_id):
        raise ValueError(f"ingestion_id {ingestion_id!r} must be a canonical UUIDv7")
    if not is_sha256_hex(configuration_hash):
        raise ValueError(
            f"configuration_hash {configuration_hash!r} must be a 64-char hex string"
        )
    if not pipeline_version or not pipeline_version.strip():
        raise ValueError("pipeline_version must be non-empty")
    if status not in STATUSES:
        raise ValueError(f"status {status!r} must be one of {sorted(STATUSES)}")
    if boards_attempted < 0 or boards_failed < 0:
        raise ValueError("board counts must be non-negative")

    timestamps: list[tuple[str, datetime]] = [
        ("run_started_at", run_started_at),
        ("completed_at", completed_at),
        *[
            (f"observation[{i}].observed_at", obs.observed_at)
            for i, obs in enumerate(observations)
        ],
    ]
    for name, value in timestamps:
        if value.tzinfo is None:
            raise ValueError(f"{name} must be timezone-aware (naive got {value!r})")
    if completed_at < run_started_at:
        raise ValueError(
            f"completed_at {completed_at!r} precedes run_started_at {run_started_at!r}"
        )

    for obs in observations:
        if obs.action not in ACTIONS:
            raise ValueError(f"action {obs.action!r} must be one of {sorted(ACTIONS)}")
        if not is_sha256_hex(obs.content_sha256):
            raise ValueError(
                f"content_sha256 {obs.content_sha256!r} must be a 64-char hex string"
            )
        for join_column in ("source", "board", "employer"):
            value = getattr(obs, join_column)
            if value is not None and not value:
                raise ValueError(f"observation {join_column} must be non-empty")

    rows = len(observations)
    table = pa.Table.from_pydict(
        {
            "schema_version": [SCHEMA_VERSION] * rows,
            "ingestion_id": [ingestion_id] * rows,
            "ingested_on": [ingested_on] * rows,
            "source": [obs.source for obs in observations],
            "pipeline_version": [pipeline_version] * rows,
            "configuration_hash": [configuration_hash] * rows,
            "run_started_at": [run_started_at] * rows,
            "completed_at": [completed_at] * rows,
            "status": [status] * rows,
            "boards_attempted": [boards_attempted] * rows,
            "boards_failed": [boards_failed] * rows,
            "board_status": [dict(board_status)] * rows,
            "action": [obs.action for obs in observations],
            "content_sha256": [obs.content_sha256 for obs in observations],
            "source_id": [obs.source_id for obs in observations],
            "board": [obs.board for obs in observations],
            "employer": [obs.employer for obs in observations],
            "observed_at": [obs.observed_at for obs in observations],
        },
        schema=_SCHEMA,
    )

    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="snappy")
    return sink.getvalue().to_pybytes()
