"""Raw-layer ingestion: fetched postings become content-addressed lake objects.

The write path of ``docs/architecture/data.md``: a batch of postings is
normalized (:class:`~backend.ingestion.scrapers.ScrapedJob` or a linkedin
markdown payload), serialized to canonical JSON, and stored once under its
content hash via ``keys.raw_payload``. Re-observed content hashes to the same
key, so ingestion is idempotent (layout §4): identical runs are no-ops beyond
a ``skipped`` event each.

This module owns the storage-facing half of ingestion — sinks, payload
canonicalization, hash minting, per-batch events, and the per-board loop. The
live-facing half (HTTP fetch) stays in :mod:`backend.ingestion.scrapers`, and
scripts stay thin CLIs that compose the two. Every ingestion observation is
returned as an :class:`IngestionEvent` so callers can stream results into a
manifest (:mod:`backend.ingestion.manifest`) instead of silent side effects.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from backend.ingestion.scrapers import (
    ScrapedJob,
    _ashby_board_fetch,
    _dataroles_board_fetch,
    _greenhouse_board_fetch,
    _lever_board_fetch,
)
from backend.storage import ids, keys


class Sink(Protocol):
    """The put/exists subset of a storage backend the ingestion path needs.

    ``backend.storage.store.ObjectStore`` satisfies this directly;
    :class:`FileSink` is the local-mirror variant. Keys arrive pre-validated,
    and mints of :func:`keys.raw_payload` are the only keys ever passed in.
    """

    def put(self, key: str, data: bytes) -> None: ...

    def exists(self, key: str) -> bool: ...


class FileSink:
    """A :class:`Sink` that mirrors keys as files under ``root``.

    The local-cache variant of the store for pipeline development and store
    downtime; keys are safe path fragments (no traversal) because they are
    validated by :func:`keys.validate_key` before reaching a sink.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def put(self, key: str, data: bytes) -> None:
        """Write ``data`` at ``key`` beneath the mirror root (overwrite)."""
        keys.validate_key(key)
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def exists(self, key: str) -> bool:
        """Return whether ``key`` exists beneath the mirror root."""
        keys.validate_key(key)
        return (self._root / key).exists()


@dataclass(frozen=True)
class IngestionEvent:
    """One posting observation, convertible to a manifest row.

    Attributes:
        source: Allowlisted source partition the payload was ingested under.
        board: Board the posting was scraped from (non-empty always, so the
            manifest rows stack up against the run header's board outcomes).
        event: ``stored`` (new content written to the raw layer) or
            ``duplicate`` (content hash already present — the idempotent
            no-op). Board-level failures are run-header concerns, not events.
        content_sha256: SHA-256 of the canonical payload bytes.
        raw_key: The object key written; empty for duplicates (whose payload
            maps to the same key as the already-stored object).
        source_id: Native ATS posting id (``<ats>:<board>:<id>``) when known.
        employer: In-memory provenance for caller summaries.
    """

    source: str
    board: str
    event: str
    content_sha256: str = ""
    raw_key: str = ""
    source_id: str = ""
    employer: str = ""


def canonical_json(payload: dict[str, Any]) -> bytes:
    """Serialize a raw payload to the canonical object bytes.

    Canonical form (sorted keys, no ASCII escaping) is the whole content
    contract: the hash of these bytes is the payload's permanent identity, so
    the serialization must stay byte-stable across runs.
    """
    return json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")


def job_payload(job: ScrapedJob) -> bytes:
    """Serialize a scraped posting to its canonical raw-payload bytes.

    The payload is the full :class:`ScrapedJob` record — exactly what the
    source normalized through :mod:`backend.ingestion.scrapers` gave us, with
    no ingestion-time metadata mixed in (that belongs to manifests), so
    identical posting content is byte-identical across runs.
    """
    return canonical_json(dataclasses.asdict(job))


def ingest_payload(
    source: str,
    payload: bytes,
    *,
    sink: Sink,
    existing_hashes: set[str],
    board: str = "",
    source_id: str = "",
    employer: str = "",
) -> IngestionEvent:
    """Ingest one canonical payload; return its :class:`IngestionEvent`.

    The key is minted from the payload's hash — content-addressed identity —
    and a hash already in the raw layer (previous runs, via the sink's
    existence check; this run, via the shared set) is skipped before any
    write. Returns the event for the caller's manifest rather than touching
    storage beyond the single put.

    Args:
        source: Allowlisted raw source partition.
        payload: Canonical payload bytes; hashed for identity.
        sink: Where the object is written (store or file mirror).
        existing_hashes: Hashes seen this run, mutated when the payload is
            written so a caller loop stays dedup-correct without repeat
            existence calls.
        board, source_id, employer: Provenance columns for the returned
            event.

    Returns:
        A ``stored`` or ``duplicate`` :class:`IngestionEvent`.
    """
    content_hash = ids.content_sha256(payload)
    key = keys.raw_payload(source, content_hash)
    keys.validate_key(key)
    if content_hash in existing_hashes or sink.exists(key):
        return IngestionEvent(
            source=source,
            board=board,
            event="duplicate",
            content_sha256=content_hash,
            source_id=source_id,
            employer=employer,
        )
    sink.put(key, payload)
    existing_hashes.add(content_hash)
    return IngestionEvent(
        source=source,
        board=board,
        event="stored",
        content_sha256=content_hash,
        raw_key=key,
        source_id=source_id,
        employer=employer,
    )


def usable_jobs(jobs: Iterable[ScrapedJob], *, limit: int | None) -> list[ScrapedJob]:
    """Filter, order, and bound the intake from one board fetch.

    A posting must carry a title and non-empty description (the raw layer is
    the permanent corpus; empty shells are not stored), and intake order is
    stable ``source_id`` order so a ``limit`` slices deterministically
    regardless of how the board happened to order its response.

    Raises:
        ValueError: if ``limit`` is negative.
    """
    if limit is not None and limit < 0:
        raise ValueError(f"limit {limit!r} must be non-negative")
    usable = sorted(
        (job for job in jobs if job.title and job.description_html),
        key=lambda job: job.source_id or "",
    )
    return usable if limit is None else usable[:limit]


def ingest_board(
    jobs: Iterable[ScrapedJob],
    *,
    source: str,
    board: str,
    sink: Sink,
    existing_hashes: set[str],
    limit: int | None = None,
) -> list[IngestionEvent]:
    """Ingest one board fetch and return its manifest events.

    Args:
        jobs: Postings from one board fetch.
        source: Allowlisted raw source partition.
        board: The board slug scraped.
        sink: Where raw objects are written.
        existing_hashes: Dedup set, updated in place.
        limit: Per-board posting bound; ``None`` takes the whole board.

    Returns:
        One :class:`IngestionEvent` per posting ingested.
    """
    events = []
    for job in usable_jobs(jobs, limit=limit):
        events.append(
            ingest_payload(
                source,
                job_payload(job),
                sink=sink,
                existing_hashes=existing_hashes,
                board=board,
                source_id=job.source_id or "",
                employer=job.employer,
            )
        )
    return events


def board_fetch(source: str, board: str, limit: int | None = None) -> list[ScrapedJob]:
    """Fetch one board through the per-source scraper function dispatch.

    Fetchers share the uniform ``(board, limit)`` signature. The ATS sources
    take their whole board in one call and treat ``limit`` as a no-op — the
    intake bound applies post-fetch at :func:`usable_jobs` — while the
    dataroles curator fetches page-by-page and honors ``limit`` as its
    page-fetch budget.
    """
    fetcher = _FETCHERS[source]
    return fetcher(board, limit)


_FETCHERS: dict[str, Callable[[str, int | None], list[ScrapedJob]]] = {
    "ashby": _ashby_board_fetch,
    "greenhouse": _greenhouse_board_fetch,
    "lever": _lever_board_fetch,
    "dataroles": _dataroles_board_fetch,
}
