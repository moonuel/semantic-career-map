"""On-command query pulls against the official HiringCafe MCP endpoint.

The query path of ``docs/specs/hiringcafe-mcp-query-integration.md`` — the
manual counterpart to :mod:`scripts.scrape`. Unlike the scheduled scraper,
queries are enumerated on the command line (or fall back to a small declared
set here), never in the source registry, so the scheduler can never reach
this entry point. Each invocation runs one MCP handshake, then per query one
search call plus details calls for the top-k results, and prints per-query
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

_DEFAULT_QUERIES = ("ml engineer", "mlops", "data platform")
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
        help="search query; repeatable, defaults to the module's default set",
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

    queries = args.query if args.query else list(_DEFAULT_QUERIES)
    return run(queries=queries, limit=args.limit, details=args.details, client=client)


if __name__ == "__main__":
    raise SystemExit(main())
