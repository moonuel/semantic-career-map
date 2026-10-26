"""Concurrent-run reconciliation: self-healing row dedupe for one configuration.

The concurrent-run gap learned live in Layer 2: two runs sharing a
``configuration_hash`` can both stream the same anti-join delta (neither
sees the other's uncommitted parts), so duplicate
``(source_hash, configuration_hash)`` rows land across parts of different
run ids — 309 orphan parts / 9,404 duplicate rows in the observed incident,
pruned then by hand.

This module automates that remediation: the driver calls
:func:`reconcile_configuration` **before** every selection (the next run
heals the lake), scanning the run's version prefix, grouping parts by their
minting run (the run id rides in every part leaf), and keeping — per
duplicated source hash — only the canonical copy from the earliest run
(UUIDv7 ids compare chronologically).

In-flight safety: a run still encoding (its parts are younger than
:attr:`DEFAULT_MIN_RUN_AGE`) is left entirely untouched — pruning a live
writer's parts would strand its manifest and counters; its duplicate rows
heal once it ages out. Parts of other configurations sharing the version
prefix (same model/variant/version, different SQL or batch) are never
touched: only rows whose ``configuration_hash`` equals the healing config's
participate.

Immutability deviation (deliberate): the lake is append-only, but duplicate
rows are a defect state, so a fully-duplicate part is deleted and a
partially-duplicate part is rewritten in place with the surviving row set
(schema and provenance columns preserved verbatim). This mirrors the manual
prune the Layer 2 incident required and keeps the table (and the parity
gate) truthful. The dataset manifest is not rewritten here — manifest
semantics across multiple runs of one configuration are the flush-checkpoint
design's surface (the change's later tasks).
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from backend.pipeline.embedding.config import EmbeddingConfig, configuration_hash
from backend.pipeline.embedding.writer import build_flush_checkpoint
from backend.storage import keys

# Runs younger than this are treated as possibly in-flight and never pruned.
# A scheduled CPU run over the bounded delta (<= 5000 rows) finishes in
# minutes; six hours is far past any legitimate wall time and short enough
# that the next daily run waits for nothing.
DEFAULT_MIN_RUN_AGE = timedelta(hours=6)


@dataclass(frozen=True)
class ReconcileReport:
    """What one reconcile pass saw and did.

    Attributes:
        configuration_hash: The healed configuration's hash.
        scanned_parts: Parts under the version prefix carrying this
            configuration's rows.
        scanned_rows: Rows of this configuration observed.
        duplicate_row_count: Rows removed this pass (deferred in-flight
            duplicates are not counted here).
        kept_rows: Rows of this configuration surviving the pass.
        canonical_run_id: Earliest run id whose copies are authoritative,
            ``None`` when the configuration has no rows.
        deferred_runs: Run ids left untouched by the in-flight guard.
        rewritten_parts: Part keys rewritten with the surviving subset.
        deleted_parts: Fully-duplicate part keys deleted.
    """

    configuration_hash: str
    scanned_parts: int
    scanned_rows: int
    duplicate_row_count: int
    kept_rows: int
    canonical_run_id: str | None
    deferred_runs: list[str]
    rewritten_parts: list[str]
    deleted_parts: list[str]


@dataclass
class _PartScan:
    """One scanned part of the healing configuration."""

    key: str
    run_id: str
    seq: int
    table: pa.Table  # the part's full table (schema + all rows)
    ours: list[int]  # row positions of this configuration's rows


# Placeholder run id only to reach the key builder's validation path; a
# valid UUIDv7 shape so validation passes. Never written anywhere.
_PROBE_RUN_ID = "00000000-0000-7f7e-a000-000000000000"


def version_prefix(config: EmbeddingConfig) -> str:
    """Return the dataset version prefix this configuration writes under.

    Built through :func:`backend.storage.keys.embedding_part` so the prefix
    stays derived from the layout rather than re-interpolated.

    Args:
        config: The embedding configuration.

    Returns:
        The ``lake/embeddings/model=…/variant=…/version=…/`` prefix.
    """
    probe = keys.embedding_part(
        config.model_segment, config.variant, config.version, _PROBE_RUN_ID, 1
    )
    return probe.rsplit("/", 1)[0] + "/"


def reconcile_configuration(
    store,
    config: EmbeddingConfig,
    *,
    as_of: datetime | None = None,
    min_run_age: timedelta | None = None,
) -> ReconcileReport:
    """Heal duplicate rows of one configuration before a run selects.

    Args:
        store: The object store (real or testing double).
        config: The healing configuration.
        as_of: Reference wall clock; ``None`` uses now (UTC).
        min_run_age: Runs (by their parts' latest ``created_at``) younger
            than this are deferred as possibly in-flight; ``None`` uses
            :attr:`DEFAULT_MIN_RUN_AGE`, ``timedelta(0)`` disables the guard.

    Returns:
        The :class:`ReconcileReport`.

    Raises:
        ValueError: on a part whose schema drifts (missing
            ``configuration_hash``/``created_at``/``source_hash``) or whose
            leaf is not run-minted.
    """
    if as_of is None:
        as_of = datetime.now(UTC)
    if min_run_age is None:
        min_run_age = DEFAULT_MIN_RUN_AGE
    digest = configuration_hash(config)
    prefix = version_prefix(config)

    scans: list[_PartScan] = []
    run_created: dict[str, datetime | None] = {}
    for key in store.list_keys(prefix):
        if not key.endswith(".parquet"):
            raise ValueError(f"unexpected non-Parquet object in {prefix}: {key!r}")
        run_id, seq = keys.parse_part_leaf(key.rsplit("/", 1)[-1])
        table = pq.read_table(io.BytesIO(store.get(key)))
        for column in ("configuration_hash", "created_at", "source_hash"):
            if column not in table.column_names:
                raise ValueError(f"part {key!r} is missing column {column!r}")
        config_column = table.column("configuration_hash").to_pylist()
        ours = [index for index, value in enumerate(config_column) if value == digest]
        if not ours:
            continue  # another configuration's part — untouched territory
        created = table.column("created_at").to_pylist()
        max_created = max(_utc(created[index]) for index in ours)
        scans.append(_PartScan(key=key, run_id=run_id, seq=seq, table=table, ours=ours))
        previous = run_created.get(run_id)
        run_created[run_id] = (
            max_created if previous is None else max(previous, max_created)
        )

    eligible = {
        run_id
        for run_id, max_created in run_created.items()
        if max_created is not None and max_created <= as_of - min_run_age
    }
    deferred = sorted(set(run_created) - eligible)

    # Canonical copy per source hash: earliest (run_id, seq, row position).
    canonical: dict[str, tuple[str, int, int]] = {}
    for scan in scans:
        if scan.run_id not in eligible:
            continue
        hashes = scan.table.column("source_hash").to_pylist()
        for position in scan.ours:
            mark = (scan.run_id, scan.seq, position)
            existing = canonical.get(hashes[position])
            if existing is None or mark < existing:
                canonical[hashes[position]] = mark

    scanned_rows = sum(len(scan.ours) for scan in scans)
    duplicate_row_count = 0
    rewritten: list[str] = []
    deleted: list[str] = []
    for scan in scans:
        if scan.run_id not in eligible:
            continue
        hashes = scan.table.column("source_hash").to_pylist()
        checkpoint_key = keys.manifest_run_checkpoint(scan.run_id, scan.seq)
        keep = [
            position
            for position in scan.ours
            if canonical[hashes[position]] == (scan.run_id, scan.seq, position)
        ]
        removed = len(scan.ours) - len(keep)
        if removed == 0:
            continue
        duplicate_row_count += removed
        if keep:
            store.put(scan.key, _take_bytes(scan.table, keep))
            rewritten.append(scan.key)
            # the part's checkpoint book follows its content — rebuilt from
            # the surviving rows so books never reference rows that a part
            # no longer carries
            checkpoint = build_flush_checkpoint(
                run_id=scan.run_id,
                seq=scan.seq,
                trigger="reconciled",
                part_key=scan.key,
                source_hashes=[hashes[position] for position in keep],
                configuration_hash=digest,
                batch_range=None,
                written_at=as_of.isoformat(),
            )
            checkpoint["reconciled"] = True
            store.put(
                checkpoint_key,
                json.dumps(checkpoint, indent=2, ensure_ascii=False).encode("utf-8"),
            )
        else:
            store.delete(scan.key)
            store.delete(checkpoint_key)  # books follow the part into the void
            deleted.append(scan.key)

    canonical_run_id = min(eligible) if eligible else None
    return ReconcileReport(
        configuration_hash=digest,
        scanned_parts=len(scans),
        scanned_rows=scanned_rows,
        duplicate_row_count=duplicate_row_count,
        kept_rows=scanned_rows - duplicate_row_count,
        canonical_run_id=canonical_run_id,
        deferred_runs=deferred,
        rewritten_parts=rewritten,
        deleted_parts=deleted,
    )


def heal_orphan_checkpoints(
    store,
    config: EmbeddingConfig,
    *,
    as_of: datetime | None = None,
    min_run_age: timedelta | None = None,
) -> list[dict[str, Any]]:
    """Back-fill checkpoints for parts whose checkpoint never landed.

    A part without its checkpoint is the crash-between-puts orphan (the
    part → checkpoint gap is the only place books can lag rows); its rows
    are durable either way, so healing restates the books from the part's
    contents as ``trigger=recovered`` checkpoints with
    ``recovered: true`` and no batch telemetry — exactly the cleaning
    stage's recovery shape
    (:func:`backend.pipeline.cleaning.runs.heal_orphan_checkpoints`),
    stage-native builder. Runs younger than the guard are deferred as
    possibly in-flight: racing a live writer's bookkeeping could overwrite
    its real checkpoint with a ``recovered`` mislabel, and the deferral
    costs nothing (the next run heals them then).

    Args:
        store: The object store (real or testing double).
        config: The healing configuration (its version prefix is scanned).
        as_of: Reference wall clock; ``None`` uses now (UTC).
        min_run_age: Orphan parts younger than this (by their rows'
            ``created_at``) are left for a later run; ``None`` uses
            :attr:`DEFAULT_MIN_RUN_AGE`, ``timedelta(0)`` disables the guard.

    Returns:
        The recovery checkpoints written this call (empty when nothing
        was orphaned).

    Raises:
        ValueError: on a part whose schema drifts or whose leaf is not
            run-minted.
    """
    if as_of is None:
        as_of = datetime.now(UTC)
    if min_run_age is None:
        min_run_age = DEFAULT_MIN_RUN_AGE
    prefix = version_prefix(config)
    existing = _existing_checkpoints(store)
    recovered: list[dict[str, Any]] = []
    for part_key in store.list_keys(prefix):
        if not part_key.endswith(".parquet"):
            raise ValueError(f"unexpected non-Parquet object in {prefix}: {part_key!r}")
        run_id, seq = keys.parse_part_leaf(part_key.rsplit("/", 1)[-1])
        if (run_id, seq) in existing:
            continue
        table = pq.read_table(io.BytesIO(store.get(part_key)))
        for column in ("source_hash", "configuration_hash", "created_at"):
            if column not in table.column_names:
                raise ValueError(f"part {part_key!r} is missing column {column!r}")
        if table.num_rows == 0:
            raise ValueError(f"part {part_key!r} carries no rows to heal from")
        max_created = max(
            _utc(value) for value in table.column("created_at").to_pylist()
        )
        if max_created > as_of - min_run_age:
            continue  # possibly in-flight; never contest a live writer's books
        checkpoint = build_flush_checkpoint(
            run_id=run_id,
            seq=seq,
            trigger="recovered",
            part_key=part_key,
            source_hashes=table.column("source_hash").to_pylist(),
            configuration_hash=table.column("configuration_hash")[0].as_py(),
            batch_range=None,
            written_at=as_of.isoformat(),
        )
        checkpoint["recovered"] = True
        store.put(
            keys.manifest_run_checkpoint(run_id, seq),
            json.dumps(checkpoint, indent=2, ensure_ascii=False).encode("utf-8"),
        )
        recovered.append(checkpoint)
    return recovered


def _existing_checkpoints(store) -> set[tuple[str, int]]:
    """Collect all ``(run_id, seq)`` checkpoint pairs under the run root.

    One listing instead of one probe per part — the set covers every
    historical run, and malformed leaves contribute nothing rather than
    failing a healing pass (same walk as the cleaning stage's runs module).
    """
    existing: set[tuple[str, int]] = set()
    for key in store.list_keys("lake/manifests/runs/"):
        parent, leaf = key.rstrip("/").rsplit("/", 1)
        if not leaf.startswith("flush=") or not leaf.endswith(".json"):
            continue
        run_segment = parent.rsplit("/", 1)[-1]
        if not run_segment.startswith("run_id="):
            continue
        try:
            seq = int(leaf.removeprefix("flush=").removesuffix(".json"))
        except ValueError:
            continue
        existing.add((run_segment.removeprefix("run_id="), seq))
    return existing


def _utc(value: datetime) -> datetime:
    """Normalize a stored (naive-UTC wall) timestamp to aware UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _take_bytes(table: pa.Table, keep: list[int]) -> bytes:
    """Serialize the surviving rows of one part, schema preserved."""
    sink = pa.BufferOutputStream()
    pq.write_table(table.take(keep), sink)
    return sink.getvalue().to_pybytes()
