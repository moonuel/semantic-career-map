"""Batch ingestion manifests: the append-only record of what a run observed.

Every raw object is content-addressed and silent on ingestion time (no date
in its key — layout §2.1), so provenance lives here: one Parquet file per
run under ``lake/manifests/ingestion/<ingested_on>/``, keyed by an
``ingestion_id`` minted once per run. The builder is ``build_ingestion_manifest``:
a run-level header (identity, pipeline, configuration, board outcomes) is
repeated on every row so each row is self-describing, and observation rows
join back to raw objects via ``content_sha256``.

``tests/test_manifest.py`` is the source of truth for this contract; this
module implements it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime

import pyarrow as pa
import pyarrow.parquet as pq

from backend.storage import ids, keys

# Manifest schema version, bumped on any breaking change to the columns.
SCHEMA_VERSION = "v1"

# Row-level outcome of one observation: new content stored (the payload was
# written to the lake) or content seen before (a content-addressed no-op).
_ACTIONS: frozenset[str] = frozenset({"stored", "duplicate"})

# Overall run status derived from the board outcomes.
_STATUSES: frozenset[str] = frozenset({"ok", "partial", "failed"})

_HEADER_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("ingestion_id", pa.string()),
        ("ingested_on", pa.date32()),
        ("pipeline_version", pa.string()),
        ("configuration_hash", pa.string()),
        ("run_started_at", pa.timestamp("us", tz="UTC")),
        ("completed_at", pa.timestamp("us", tz="UTC")),
        ("status", pa.string()),
        ("boards_attempted", pa.int64()),
        ("boards_failed", pa.int64()),
        ("board_status", pa.map_(pa.string(), pa.string())),
    ]
)

_OBSERVATION_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("action", pa.string()),
        ("content_sha256", pa.string()),
        ("source_id", pa.string()),
        ("board", pa.string()),
        ("employer", pa.string()),
        ("observed_at", pa.timestamp("us", tz="UTC")),
    ]
)

_SCHEMA = pa.schema(list(_HEADER_SCHEMA) + list(_OBSERVATION_SCHEMA))


@dataclass(frozen=True)
class BatchObservation:
    """One posting observation inside a batch run manifest row.

    Attributes:
        source: Allowlisted raw source the posting was ingested from.
        action: ``stored`` (new content written) or ``duplicate`` (content
            hash already in the raw layer — the idempotent no-op).
        content_sha256: SHA-256 of the canonical payload bytes — the join to
            the raw object key ``raw_payload(source, content_sha256)``.
        source_id: Native ATS posting id when the source provides one; the
            curated linkedin corpus instead carries its filename stem.
        board: Non-empty board the posting was scraped from.
        employer: Organization name, when the source provided one.
        observed_at: TZ-aware UTC moment the observation was made.

    Raises:
        ValueError: from :meth:`validate` when any field violates its
            invariant; board-level and storage failures are header concerns,
            not observation rows.
    """

    source: str
    action: str
    content_sha256: str
    source_id: str | None
    board: str
    employer: str | None
    observed_at: datetime

    def validate(self) -> None:
        """Raise ``ValueError`` if any field violates the manifest contract."""
        if self.source not in keys.RAW_SOURCES:
            raise ValueError(
                f"unknown raw source {self.source!r}; "
                f"expected one of {sorted(keys.RAW_SOURCES)}"
            )
        if self.action not in _ACTIONS:
            raise ValueError(
                f"unknown action {self.action!r}; expected one of {sorted(_ACTIONS)}"
            )
        if not ids.is_sha256_hex(self.content_sha256):
            raise ValueError(
                f"content_sha256 {self.content_sha256!r} must be 64-char hex"
            )
        if not self.board:
            raise ValueError("board must be non-empty")
        if self.source_id is not None and not isinstance(self.source_id, str):
            raise ValueError("source_id must be a string or None")
        if self.employer is not None and not isinstance(self.employer, str):
            raise ValueError("employer must be a string or None")
        _check_tz(self.observed_at, name="observed_at")


def _check_tz(value: datetime, *, name: str) -> None:
    """Raise unless ``value`` is a datetime carrying a timezone."""
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be a TZ-aware datetime")


def build_ingestion_manifest(
    observations: list[BatchObservation],
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
    board_status: dict[str, str],
) -> bytes:
    """Serialize one run's manifest as a single Parquet batch.

    Args:
        observations: One per posting observed by the run, in encounter
            order; an empty list is a well-formed zero-row manifest.
        ingestion_id: Canonical UUIDv7 minted once for the run; identifies
            every row and the manifest object itself.
        ingested_on: Calendar date of the run (the manifest object's day
            partition; raw object keys deliberately carry no date).
        pipeline_version: Version of the pipeline code (e.g. a git sha).
        configuration_hash: 64-char hex content hash of the run
            configuration (layout §3).
        run_started_at, completed_at: TZ-aware UTC run boundaries.
        status: Overall run status — ``ok``, ``partial`` (some boards
            failed), or ``failed`` (all boards failed).
        boards_attempted, boards_failed: Board outcome counts; failed never
            exceeds attempted.
        board_status: Per-board outcome map, e.g. ``{"lever/palantir":
            "ok", "lever/ghost": "HTTP 404"}``.

    Returns:
        Parquet bytes matching :data:`_SCHEMA`, snappy-compressed.

    Raises:
        ValueError: if any header or observation field violates its
            invariant.
    """
    _check_header(
        ingestion_id=ingestion_id,
        ingested_on=ingested_on,
        pipeline_version=pipeline_version,
        configuration_hash=configuration_hash,
        run_started_at=run_started_at,
        completed_at=completed_at,
        status=status,
        boards_attempted=boards_attempted,
        boards_failed=boards_failed,
        board_status=board_status,
    )
    for observation in observations:
        observation.validate()

    header: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "ingestion_id": ingestion_id,
        "ingested_on": ingested_on,
        "pipeline_version": pipeline_version,
        "configuration_hash": configuration_hash,
        "run_started_at": run_started_at,
        "completed_at": completed_at,
        "status": status,
        "boards_attempted": boards_attempted,
        "boards_failed": boards_failed,
        "board_status": board_status,
    }
    rows = [{**header, **vars(observation)} for observation in observations]
    sink = io.BytesIO()
    pq.write_table(
        pa.Table.from_pylist(rows, schema=_SCHEMA), sink, compression="snappy"
    )
    return sink.getvalue()


def _check_header(
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
    board_status: dict[str, str],
) -> None:
    """Raise ``ValueError`` if any run-header field violates its invariant."""
    if not ids.is_canonical_uuid7(ingestion_id):
        raise ValueError(f"ingestion_id {ingestion_id!r} must be a canonical UUIDv7")
    if not ids.is_sha256_hex(configuration_hash):
        raise ValueError(
            f"configuration_hash {configuration_hash!r} must be 64-char hex"
        )
    if not pipeline_version:
        raise ValueError("pipeline_version must be non-empty")
    if status not in _STATUSES:
        raise ValueError(
            f"unknown status {status!r}; expected one of {sorted(_STATUSES)}"
        )
    if isinstance(boards_attempted, bool) or boards_attempted < 0:
        raise ValueError("boards_attempted must be a non-negative int")
    if (
        isinstance(boards_failed, bool)
        or boards_failed < 0
        or boards_failed > boards_attempted
    ):
        raise ValueError(
            f"boards_failed {boards_failed!r} must be between 0 and "
            f"boards_attempted ({boards_attempted})"
        )
    if not isinstance(board_status, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in board_status.items()
    ):
        raise ValueError("board_status must be a {str: str} map")
    _check_tz(run_started_at, name="run_started_at")
    _check_tz(completed_at, name="completed_at")
    if completed_at < run_started_at:
        raise ValueError("completed_at must not precede run_started_at")
    if not ingested_on:
        raise ValueError("ingested_on must be a date")


def write_ingestion_manifest(
    sink, manifest_bytes: bytes, *, ingested_on: date, ingestion_id: str
) -> str:
    """Store a built manifest at its batch key; return the key.

    Kept as a one-liner over :func:`keys.manifest_ingestion` so callers never
    string-interpolate manifest keys.

    Args:
        sink: The storage to write through (``ObjectStore`` or ``FileSink``).
        manifest_bytes: Output of :func:`build_ingestion_manifest`.
        ingested_on: The run date the manifest was built with.
        ingestion_id: The run id the manifest was built with.

    Returns:
        The well-formed manifest key the bytes were written to.
    """
    key = keys.manifest_ingestion(ingested_on.isoformat(), ingestion_id)
    keys.validate_key(key)
    sink.put(key, manifest_bytes)
    return key
