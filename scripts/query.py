"""On-command query pulls against the official HiringCafe MCP endpoint.

The query path of ``docs/specs/hiringcafe-mcp-query-integration.md`` — the
manual counterpart to :mod:`scripts.scrape`. Unlike the scheduled scraper,
queries are enumerated on the command line, as named :data:`_QUERY_SETS`
(curated role-search vocabulary this project's taxonomy targets), or both —
never in the source registry, so the scheduler can never reach this entry
point. Each invocation runs one MCP handshake, then per query one search
call plus details calls for the top-k results, and prints per-query
summaries. Lake writes, raw-layer ingestion, and the run manifest land in a
later slice; this slice is fetch-and-report.

Rate posture (spec C1): human-triggered only, one POST per call with a delay
between calls, total calls bounded by ``queries * (1 + k) + 1``.

    python scripts/query.py --query "ml engineer" --limit 10
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime

from backend.ingestion import mcp

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


def _run_query(
    client: mcp.HiringCafeClient,
    query: str,
    *,
    limit: int,
    details: int,
) -> None:
    """Fetch one query's window and print its summary.

    One search call plus up to ``details`` detail calls, with a politeness
    delay between calls. Failures raise; the run loop isolates them per query.

    Args:
        client: The MCP client (handshake already done).
        query: The search query text.
        limit: Search result bound (clamped by the client to the server cap).
        details: How many top results to expand to full details.

    Raises:
        mcp.MCPError: on any call failure (isolated by the caller).
    """
    jobs = client.search(search=query, limit=limit, posted_salary_only=True)
    print(f"{query}: showing {len(jobs)} of live index")
    for job in jobs[:details]:
        time.sleep(_CALL_DELAY_S)
        record = client.details(job["id"])
        description = record.get("description") or ""
        print(
            f"  {job['id']} | {job['title'][:60]} | {job['company']} | "
            f"detail: {len(description)} chars"
        )


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
) -> int:
    """Run every query through the client, isolating failures per query.

    Args:
        queries: Query texts, in invocation order.
        limit: Search result bound per query.
        details: Detail-expansion count per query.
        client: An initialized MCP client.

    Returns:
        Process exit code (0 unless every query failed).
    """
    started_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    failed = 0
    for query in queries:
        time.sleep(_CALL_DELAY_S)
        try:
            _run_query(client, query, limit=limit, details=details)
        except mcp.MCPError as exc:
            failed += 1
            print(f"{query}: failed — {_error_text(exc)}", file=sys.stderr)
    print(f"queries: {len(queries) - failed}/{len(queries)} ok; started={started_at}")
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
        help=f"top results expanded to full details (default: {_DEFAULT_DETAILS})",
    )
    args = parser.parse_args(argv)

    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.details < 0:
        parser.error("--details must be non-negative")

    client = mcp.HiringCafeClient()
    try:
        client.initialize()
    except mcp.MCPError as exc:
        print(f"error: mcp handshake failed: {_error_text(exc)}", file=sys.stderr)
        return 1

    queries = _select_queries(args.sets or [], args.query or [], set_all=args.set_all)
    return run(queries=queries, limit=args.limit, details=args.details, client=client)


if __name__ == "__main__":
    raise SystemExit(main())
