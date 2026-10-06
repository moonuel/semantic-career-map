"""Unit tests for :mod:`backend.ingestion.manifest` (batch manifest builder).

Covers the schema contract of the batched-scrape ingestion manifest proposed
in the manifest-schema discussion: one Parquet file per run, run-level header
fields repeated per row, observation rows joining raw objects via
``content_sha256``. Round-trips through pyarrow to verify bytes, not just the
builder's return shape.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.ingestion.manifest import (
    SCHEMA_VERSION,
    BatchObservation,
    build_ingestion_manifest,
)

ID = "0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d70"
HASH1 = "a" * 64
HASH2 = "b" * 64
RUN_DATE = date(2026, 10, 6)
STARTED = datetime(2026, 10, 6, 21, 58, 0, tzinfo=UTC)
COMPLETED = datetime(2026, 10, 6, 22, 3, 0, tzinfo=UTC)

TWO_ROWS = [
    BatchObservation(
        source="linkedin",
        action="stored",
        content_sha256=HASH1,
        source_id="ashby:acme:7458d4e9",
        board="acme",
        employer="Acme Corp",
        observed_at=datetime(2026, 10, 6, 21, 58, 40, tzinfo=UTC),
    ),
    BatchObservation(
        source="greenhouse",
        action="duplicate",
        content_sha256=HASH2,
        source_id="greenhouse:beta:991",
        board="beta",
        employer=None,
        observed_at=datetime(2026, 10, 6, 22, 1, 12, tzinfo=UTC),
    ),
]

HEADER = {
    "ingestion_id": ID,
    "ingested_on": RUN_DATE,
    "pipeline_version": "65ff0f9",
    "configuration_hash": "c" * 64,
    "run_started_at": STARTED,
    "completed_at": COMPLETED,
    "status": "partial",
    "boards_attempted": 2,
    "boards_failed": 0,
    "board_status": {"acme": "ok", "beta": "ok"},
}


def _build(
    observations: list[BatchObservation] | None = None, **overrides: object
) -> bytes:
    kwargs: dict[str, object] = {**HEADER, **overrides}
    args: list[object] = [TWO_ROWS if observations is None else observations]
    return build_ingestion_manifest(*args, **kwargs)  # type: ignore[arg-type]


# --- 1. Round trip -----------------------------------------------------------


class TestRoundTrip:
    """The built bytes read back as one row per observation."""

    def test_reads_back_with_row_per_observation(self) -> None:
        table = pq.read_table(pa.BufferReader(_build()))
        assert table.num_rows == 2

    def test_observation_columns(self) -> None:
        table = pq.read_table(pa.BufferReader(_build()))
        row0 = {c: table.column(c)[0].as_py() for c in table.column_names}
        row1 = {c: table.column(c)[1].as_py() for c in table.column_names}
        assert row0["source"] == "linkedin"
        assert row0["action"] == "stored"
        assert row0["content_sha256"] == HASH1
        assert row0["source_id"] == "ashby:acme:7458d4e9"
        assert row0["board"] == "acme"
        assert row0["employer"] == "Acme Corp"
        assert row0["observed_at"] == TWO_ROWS[0].observed_at
        assert row1["action"] == "duplicate"
        assert row1["source_id"] == "greenhouse:beta:991"
        assert row1["employer"] is None

    def test_header_fields_repeat_on_every_row(self) -> None:
        table = pq.read_table(pa.BufferReader(_build()))
        for name in (
            "schema_version",
            "ingestion_id",
            "ingested_on",
            "pipeline_version",
            "configuration_hash",
            "run_started_at",
            "completed_at",
            "status",
            "boards_attempted",
            "boards_failed",
            "board_status",
        ):
            column = table.column(name).to_pylist()
            assert column[0] == column[1], name

    def test_header_values(self) -> None:
        table = pq.read_table(pa.BufferReader(_build()))
        row = {c: table.column(c)[0].as_py() for c in table.column_names}
        assert row["schema_version"] == SCHEMA_VERSION
        assert row["ingestion_id"] == ID
        assert row["ingested_on"] == RUN_DATE
        assert row["status"] == "partial"
        assert row["boards_attempted"] == 2
        assert row["boards_failed"] == 0
        assert dict(row["board_status"]) == {"acme": "ok", "beta": "ok"}
        assert row["run_started_at"] == STARTED
        assert row["completed_at"] == COMPLETED

    def test_row_order_is_preserved(self) -> None:
        table = pq.read_table(pa.BufferReader(_build()))
        assert table.column("source").to_pylist() == ["linkedin", "greenhouse"]

    def test_zero_rows_manifest(self) -> None:
        table = pq.read_table(pa.BufferReader(_build(observations=[])))
        assert table.num_rows == 0

    def test_is_snappy_parquet(self) -> None:
        meta = pq.read_metadata(pa.BufferReader(_build()))
        assert meta.num_columns > 0
        assert meta.row_group(0).column(0).compression == "SNAPPY"


# --- 2. Contract violations --------------------------------------------------


class TestRejectsBadInput:
    """Validation at the batch boundary: run identity and observation shape."""

    def test_rejects_bad_ingestion_id(self) -> None:
        with pytest.raises(ValueError):
            _build(ingestion_id="0190f3a27b417c9e8a3d5f6e1b2c4d70")

    def test_rejects_bad_configuration_hash(self) -> None:
        with pytest.raises(ValueError):
            _build(configuration_hash="xyz")

    def test_rejects_empty_pipeline_version(self) -> None:
        with pytest.raises(ValueError):
            _build(pipeline_version="")

    def test_rejects_unknown_status(self) -> None:
        with pytest.raises(ValueError):
            _build(status="success")

    def test_rejects_negative_board_counts(self) -> None:
        with pytest.raises(ValueError):
            _build(boards_failed=-1)

    def test_rejects_completed_before_started(self) -> None:
        with pytest.raises(ValueError):
            _build(completed_at=STARTED, run_started_at=COMPLETED)

    def test_rejects_naive_timestamps(self) -> None:
        # Deliberately naive: the tzinfo=None strip is the point of this case.
        naive = datetime(2026, 10, 6, 21, 58, 0)  # noqa: DTZ001
        with pytest.raises(ValueError):
            _build(run_started_at=naive, completed_at=COMPLETED)

    def test_rejects_bad_action(self) -> None:
        obs = BatchObservation(
            source="linkedin",
            action="overwrote",
            content_sha256=HASH1,
            source_id=None,
            board="acme",
            employer="Acme Corp",
            observed_at=STARTED,
        )
        with pytest.raises(ValueError):
            build_ingestion_manifest([obs], **HEADER)  # type: ignore[arg-type]

    def test_rejects_bad_content_hash(self) -> None:
        obs = BatchObservation(
            source="linkedin",
            action="stored",
            content_sha256="a" * 63,
            source_id=None,
            board="acme",
            employer=None,
            observed_at=STARTED,
        )
        with pytest.raises(ValueError):
            build_ingestion_manifest([obs], **HEADER)  # type: ignore[arg-type]

    def test_rejects_empty_join_columns(self) -> None:
        obs = BatchObservation(
            source="",
            action="stored",
            content_sha256=HASH1,
            source_id=None,
            board="",
            employer="Acme Corp",
            observed_at=STARTED,
        )
        with pytest.raises(ValueError):
            build_ingestion_manifest([obs], **HEADER)  # type: ignore[arg-type]
