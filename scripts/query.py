"""On-command query pulls against the official HiringCafe MCP endpoint.

The query path of ``docs/specs/hiringcafe-mcp-query-integration.md`` — the
manual counterpart to :mod:`scripts.scrape`. Unlike the scheduled scraper,
queries are enumerated on the command line, as named :data:`_QUERY_SETS`
(curated role-search vocabulary this project's taxonomy targets), or both —
never in the source registry, so the scheduler can never reach this entry
point. Each invocation runs one MCP handshake, then per query one search
call plus details calls for the top-k results; detail-expanded postings are
ingested into ``lake/raw/source=hiringcafe/`` (content-addressed, idempotent
re-runs) and every run writes one ``lake/manifests/ingestion/`` manifest
recording each stored/duplicate observation plus per-query outcomes — the
run's provenance record.

Intake is details-expanded postings only: search rows carry no description
field, and the raw layer stores postings, not shells (the ``usable_jobs``
contract of :mod:`backend.ingestion.raw`). ``--details 0`` is survey mode:
windows are fetched and printed, nothing is ingested, and the run manifest
is a zero-row record (still observable).

Rate posture (spec C1): human-triggered only, one POST per call with a delay
between calls, total calls bounded by ``queries * (1 + k) + 1``.

    python scripts/query.py --query "ml engineer" --limit 10
    python scripts/query.py --set-all --details 0   # cheapest full survey
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from backend.ingestion import mcp, raw
from backend.ingestion.manifest import (
    BatchObservation,
    build_ingestion_manifest,
    event_to_observation,
    pipeline_version,
    write_ingestion_manifest,
)
from backend.ingestion.raw import Sink
from backend.storage import ids
from backend.storage.store import ObjectStore

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Named query sets: the curated role-search vocabulary this project sources
# against HiringCafe (spec §1 — query-driven windows over their index). Sets
# are an observation lens, not a registry: they name what to ask, never a
# board the scheduler would poll. Grow per need; every query costs
# 1 + details calls per run.
_QUERY_SETS: dict[str, tuple[str, ...]] = {
    "core": ("machine learning engineer", "mlops", "data platform"),
    "ai-engineering": ("ai engineer", "llm engineer", "computer vision engineer"),
    "platform-infra": (
        "platform engineer",
        "backend engineer",
        "data engineer",
        "site reliability engineer",
    ),
    "research": ("data scientist", "research scientist", "applied scientist"),
}

_DEFAULT_LIMIT = 25
_DEFAULT_DETAILS = 10
_CALL_DELAY_S = 0.5


def _error_text(exc: Exception) -> str:
    """Compact single-line error message for a query status note."""
    return " ".join(str(exc).split())[:200] or exc.__class__.__name__


def _ingest_record(
    record: dict[str, Any],
    *,
    sink: Sink,
    board: str,
    existing_hashes: set[str],
) -> raw.IngestionEvent | None:
    """Ingest one detail-expanded MCP record into the raw layer.

    Intake filter: a posting must carry a non-empty title and description
    (the raw layer is the permanent corpus). Returns ``None`` when the
    record is skipped as a shell, with a one-line stderr note.

    Args:
        record: The ``structuredContent.job`` record from
            :meth:`mcp.HiringCafeClient.details`.
        sink: Where the canonical payload is written.
        board: The run's slugified board label for this query.
        existing_hashes: Run-shared dedup set, updated in place.

    Returns:
        The ``stored``/``duplicate`` event, or ``None`` for a skipped shell.

    Raises:
        mcp.MCPError: propagates from nothing here; normalization and
            hashing failures raise ``ValueError``.
    """
    payload_record = mcp.normalize_job(record)
    if not payload_record.get("title") or not payload_record.get("description"):
        print(
            f"{board}: skipping {payload_record['source_id']}: no title/description",
            file=sys.stderr,
        )
        return None
    return raw.ingest_payload(
        mcp.SOURCE,
        raw.canonical_json(payload_record),
        sink=sink,
        existing_hashes=existing_hashes,
        board=board,
        source_id=payload_record["source_id"],
        employer=payload_record["employer"] or "",
    )


def _ingest_query(
    client: mcp.HiringCafeClient,
    query: str,
    *,
    board: str,
    limit: int,
    details: int,
    sink: Sink,
    existing_hashes: set[str],
) -> list[raw.IngestionEvent]:
    """Fetch one query's window and ingest its detail-expanded postings.

    One search call plus up to ``details`` detail calls, with a politeness
    delay between calls. Failures raise; the run loop isolates them per
    query. Only detail-expanded postings are ingestable — search rows have
    no description, and only postings (not shells) enter the raw layer.

    Args:
        client: The MCP client (handshake already done).
        query: The search query text.
        board: The slugified board label for the manifest.
        limit: Search result bound (clamped by the client to the server cap).
        details: How many top results to expand to full details (``0`` =
            survey mode — fetch and report only).
        sink: Where raw objects are written.
        existing_hashes: Run-shared dedup set.

    Returns:
        The query's ``stored``/``duplicate`` events, in encounter order.

    Raises:
        mcp.MCPError: on any fetch failure.
        ValueError: on normalization/intake failures.
    """
    jobs = client.search(search=query, limit=limit, posted_salary_only=True)
    if not details:
        print(f"{query}: survey {len(jobs)} postings (no ingestion)")
        return []
    print(f"{query}: window {len(jobs)} postings; ingesting top {details}")
    events: list[raw.IngestionEvent] = []
    for job in jobs[:details]:
        time.sleep(_CALL_DELAY_S)
        record = client.details(job["id"])
        event = _ingest_record(
            record, sink=sink, board=board, existing_hashes=existing_hashes
        )
        if event is not None:
            events.append(event)
            description = record.get("description") or ""
            print(
                f"  {job['id']} | {(job['title'] or '')[:60]} | "
                f"{job.get('company')} | {len(description)} chars"
            )
    return events


def _select_queries(
    sets: list[str], queries: list[str], *, set_all: bool = False
) -> list[str]:
    """Resolve the effective query list, order-preserving and de-duplicated.

    Args:
        sets: Named sets from ``--set``, in argv order.
        queries: Explicit queries from ``--query``, in argv order.
        set_all: ``--set-all`` — iterate every named set.

    Returns:
        Union of the named sets' members and the explicit queries; with
        no set selection given, the ``core`` set.
    """
    selected: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        if value not in seen:
            seen.add(value)
            selected.append(value)

    if set_all:
        for set_queries in _QUERY_SETS.values():
            for query in set_queries:
                _add(query)
    elif not sets and not queries:
        for query in _QUERY_SETS["core"]:
            _add(query)
        return selected
    else:
        for name in sets:
            for query in _QUERY_SETS[name]:
                _add(query)
    for query in queries:
        _add(query)
    return selected


def run(
    *,
    queries: list[str],
    limit: int,
    details: int,
    client: mcp.HiringCafeClient,
    sink: Sink,
    configuration: dict[str, Any],
) -> int:
    """Run every query, ingest detail-expanded postings, write one manifest.

    Query isolation mirrors ``scrape.run``: one query failing (transport,
    protocol, or normalization) becomes a status note and the run continues.
    Queries slugify to collision scores: two text-wise distinct queries that
    slugify identically record one window (the second is skipped with a
    note) — they are the same request to the server.

    Args:
        queries: Query texts, in invocation order.
        limit: Search result bound per query.
        details: Detail-expansion count per query.
        client: An initialized MCP client.
        sink: The ingestion sink (the object store).
        configuration: The run configuration for the manifest's
            configuration hash (verbatim ``--set``/``--query``/bounds argv).

    Returns:
        Process exit code (0 unless every query failed).
    """
    run_started_at = datetime.now(UTC)
    ingestion_id = ids.new_id()
    observations: list[BatchObservation] = []
    board_status: dict[str, str] = {}
    existing_hashes: set[str] = set()
    seen_boards: set[str] = set()
    failed = 0

    for query in queries:
        try:
            board = mcp.board_slug(query)
            if board in seen_boards:
                print(
                    f"{query}: board {board!r} already covered — skipped",
                    file=sys.stderr,
                )
                continue
            seen_boards.add(board)
            time.sleep(_CALL_DELAY_S)
            events = _ingest_query(
                client,
                query,
                board=board,
                limit=limit,
                details=details,
                sink=sink,
                existing_hashes=existing_hashes,
            )
            stored = sum(e.event == "stored" for e in events)
            duplicates = sum(e.event == "duplicate" for e in events)
            for event in events:
                observations.append(
                    event_to_observation(event, observed_at=datetime.now(UTC))
                )
            board_status[f"{mcp.SOURCE}/{board}"] = "ok"
            print(
                f"{query}: written={stored} duplicates={duplicates}"
                if events or not details
                else f"{query}: written=0 duplicates=0 (nothing intakeable)"
            )
        except (mcp.MCPError, ValueError) as exc:
            failed += 1
            try:
                label = mcp.board_slug(query) or query
            except ValueError:
                label = query
            board_status[f"{mcp.SOURCE}/{label}"] = _error_text(exc)
            print(f"{query}: failed — {_error_text(exc)}", file=sys.stderr)

    boards_attempted = len(board_status)
    boards_failed = sum(v != "ok" for v in board_status.values())
    status = (
        "failed"
        if boards_attempted and boards_failed == boards_attempted
        else "partial"
        if boards_failed
        else "ok"
    )
    manifest = build_ingestion_manifest(
        sorted(observations, key=lambda o: (o.source, o.board, o.content_sha256)),
        ingestion_id=ingestion_id,
        ingested_on=run_started_at.date(),
        pipeline_version=pipeline_version(),
        configuration_hash=ids.content_sha256(raw.canonical_json(configuration)),
        run_started_at=run_started_at,
        completed_at=datetime.now(UTC),
        status=status,
        boards_attempted=boards_attempted,
        boards_failed=boards_failed,
        board_status=dict(sorted(board_status.items())),
    )
    manifest_key = write_ingestion_manifest(
        sink, manifest, ingested_on=run_started_at.date(), ingestion_id=ingestion_id
    )
    total_stored = sum(o.action == "stored" for o in observations)
    total_duplicates = sum(o.action == "duplicate" for o in observations)
    print(
        f"queries: {len(queries) - failed}/{len(queries)} ok; "
        f"boards failed: {boards_failed}/{boards_attempted}; "
        f"written={total_stored} duplicates={total_duplicates}; "
        f"manifest={manifest_key}"
    )
    return 1 if failed == len(queries) else 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        metavar="QUERY",
        help="explicit search query; repeatable, unioned after --set",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="sets",
        choices=sorted(_QUERY_SETS),
        default=None,
        metavar="SET",
        help="named query set (one of: %(choices)s); repeatable; defaults "
        "to the 'core' set",
    )
    parser.add_argument(
        "--set-all",
        action="store_true",
        help="iterate every named query set",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=_DEFAULT_LIMIT,
        help=f"search results per query, capped at {mcp.SEARCH_CAP} "
        f"(default: {_DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--details",
        type=int,
        default=_DEFAULT_DETAILS,
        help=f"top results expanded to full details and ingested "
        f"(default: {_DEFAULT_DETAILS}); 0 = survey mode",
    )
    args = parser.parse_args(argv)

    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.details < 0:
        parser.error("--details must be non-negative")
    if args.limit > mcp.SEARCH_CAP:
        print(
            f"warning: --limit {args.limit} exceeds the endpoint's per-call "
            f"cap of {mcp.SEARCH_CAP}; clamping to {mcp.SEARCH_CAP}",
            file=sys.stderr,
        )
        args.limit = mcp.SEARCH_CAP

    load_dotenv(_REPO_ROOT / ".env")
    sink: Sink = ObjectStore.from_env()

    queries = _select_queries(args.sets or [], args.query or [], set_all=args.set_all)
    configuration = {
        "sets": args.sets or [],
        "set_all": args.set_all,
        "queries": args.query or [],
        "limit": args.limit,
        "details": args.details,
    }

    client = mcp.HiringCafeClient()
    try:
        client.initialize()
    except mcp.MCPError as exc:
        print(f"error: mcp handshake failed: {_error_text(exc)}", file=sys.stderr)
        return 1

    try:
        return run(
            queries=queries,
            limit=args.limit,
            details=args.details,
            client=client,
            sink=sink,
            configuration=configuration,
        )
    except (BotoCoreError, ClientError, OSError) as exc:
        # Storage unavailable: the runner cannot work, manifest cannot land.
        print(f"error: storage failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
