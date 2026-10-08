"""Scheduled scraper entry point: every source site, one run apiece.

The single process the scheduler knows. No arguments runs the full declared
picture — every source in :data:`~backend.ingestion.sources.SOURCE_BOARDS`
with its declared boards, the default ``--limit`` cap per board — through
``backend.ingestion.scrapers`` (the live fetch) and
``backend.ingestion.raw`` (the content-addressed write). Per-source flags
narrow a run for ad-hoc work:

    python scripts/scrape.py                     # the whole registry
    python scripts/scrape.py --source lever --boards palantir --limit-all

One fetch per board; one ``lake/manifests/ingestion/`` manifest for the run,
recording every stored / duplicate observation plus board-level outcomes.
Storage dedup is content-hash level, so scheduled re-runs are no-ops while
staying observable — the manifest is that observability.

Sources are isolated: a board that fails is recorded in the run header and
the run continues; the exit code is nonzero only if every run source failed
entirely (or storage itself went away, making the manifest fail with it).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from backend.ingestion import raw, sources
from backend.ingestion.manifest import (
    BatchObservation,
    build_ingestion_manifest,
    write_ingestion_manifest,
)
from backend.ingestion.raw import FileSink, Sink
from backend.ingestion.sources import SOURCE_BOARDS
from backend.storage import ids
from backend.storage.store import ObjectStore

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"
_DEFAULT_LIMIT = 25


def _pipeline_version() -> str:
    """Version of the pipeline code: short git sha, or ``unknown``.

    Args:
        None.

    Returns:
        A non-empty git short sha when available, else ``"unknown"`` (the
        manifest contract requires a non-empty version).
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            check=True,
        )
    except OSError, subprocess.SubprocessError:
        return "unknown"
    return out.stdout.strip() or "unknown"


def _scrape_source(
    source: str,
    *,
    boards: list[str],
    sink: Sink,
    limit: int | None,
    observed_at: datetime,
) -> tuple[list[BatchObservation], dict[str, str]]:
    """Scrape one source across ``boards``; return its observations and statuses.

    Dedup is seeded per source (an empty set seen by
    :func:`raw.ingest_board`) and the store's content-hash existence check
    covers prior runs; within one run, cross-board duplicates resolve through
    the shared set inside :func:`raw.ingest_board`. Board isolation happens
    here: one board raising leaves the others running, with the failure on
    the returned status map for the run header.

    Args:
        source: Allowlisted raw source.
        boards: The board slugs to scrape.
        sink: Where raw objects and the eventual manifest are written.
        limit: Per-board posting bound; ``None`` takes whole boards.
        observed_at: Clock for observation rows (one per run, wall-stamped).

    Returns:
        ``(observations, board_status)`` — manifest rows and the per-board
        ``"ok"``/error map.
    """
    observations: list[BatchObservation] = []
    board_status: dict[str, str] = {}
    for board in boards:
        try:
            jobs = raw.board_fetch(source, board)
        except requests.RequestException as exc:
            board_status[f"{source}/{board}"] = _error_text(exc)
            continue
        existing_hashes: set[str] = set()
        for event in raw.ingest_board(
            jobs,
            source=source,
            board=board,
            sink=sink,
            existing_hashes=existing_hashes,
            limit=limit,
        ):
            observations.append(
                BatchObservation(
                    source=event.source,
                    action=event.event,
                    content_sha256=event.content_sha256,
                    source_id=event.source_id or None,
                    board=event.board,
                    employer=event.employer or None,
                    observed_at=observed_at,
                )
            )
        board_status[f"{source}/{board}"] = "ok"
    return observations, board_status


def _error_text(exc: Exception) -> str:
    """Compact single-line error message for a board status entry."""
    return " ".join(str(exc).split())[:200] or exc.__class__.__name__


def run(
    *,
    run_sources: list[str],
    boards_override: dict[str, list[str]],
    sink: Sink,
    limit: int | None,
) -> int:
    """Scrape every source, then write the run's single batch manifest.

    Args:
        run_sources: Sources to scrape, in registry order.
        boards_override: Per-source board list when ``--boards`` was given;
            registry defaults otherwise.
        sink: The ingestion sink (store or file mirror).
        limit: Per-board posting bound; ``None`` takes whole boards.

    Returns:
        Process exit code (0 unless every run source failed).
    """
    run_started_at = datetime.now(UTC)
    ingestion_id = ids.new_id()
    all_observations: list[BatchObservation] = []
    all_board_status: dict[str, str] = {}
    boards_attempted = 0
    failed_sources = 0

    for source in run_sources:
        boards = boards_override.get(source) or list(SOURCE_BOARDS[source])
        observations, board_status = _scrape_source(
            source,
            boards=boards,
            sink=sink,
            limit=limit,
            observed_at=datetime.now(UTC),
        )
        all_observations.extend(observations)
        all_board_status.update(board_status)
        boards_attempted += len(boards)

        written = sum(o.action == "stored" for o in observations)
        duplicates = sum(o.action == "duplicate" for o in observations)
        failed = sum(v != "ok" for v in board_status.values())
        if written == duplicates == 0 and failed == len(boards):
            failed_sources += 1
            print(
                f"{source}: written=0 duplicates=0 failed={failed} boards — source down"
            )
            continue
        print(f"{source}: written={written} duplicates={duplicates}")

    boards_failed = sum(v != "ok" for v in all_board_status.values())
    status = (
        "failed"
        if boards_attempted and boards_failed == boards_attempted
        else "partial"
        if boards_failed
        else "ok"
    )

    manifest = build_ingestion_manifest(
        all_observations,
        ingestion_id=ingestion_id,
        ingested_on=run_started_at.date(),
        pipeline_version=_pipeline_version(),
        configuration_hash=ids.content_sha256(
            raw.canonical_json(
                {
                    "sources": run_sources,
                    "boards_override": boards_override,
                    "limit": limit,
                }
            )
        ),
        run_started_at=run_started_at,
        completed_at=datetime.now(UTC),
        status=status,
        boards_attempted=boards_attempted,
        boards_failed=boards_failed,
        board_status=all_board_status,
    )
    manifest_key = write_ingestion_manifest(
        sink, manifest, ingested_on=run_started_at.date(), ingestion_id=ingestion_id
    )
    total = sum(o.action == "stored" for o in all_observations)
    print(
        f"boards failed: {boards_failed}/{boards_attempted}; "
        f"manifest={manifest_key}; total written={total}"
    )
    return 1 if failed_sources == len(run_sources) else 0


class _BoardOverride(argparse.Action):
    """Bind one ``--boards`` list to the most recently named ``--source``.

    ``action="append"`` loses argv ordering between the two flags, so the
    bind must happen at parse time: a ``--boards`` value joins the source
    named before it, and a value named before any ``--source`` is an error.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if not namespace.source:
            parser.error("--boards requires --source (boards vary per source)")
        overrides = dict(getattr(namespace, self.dest, None) or {})
        overrides[namespace.source[-1]] = list(values)
        setattr(namespace, self.dest, overrides)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(sources.SOURCE_BOARDS),
        help="source to scrape; repeatable, defaults to the full registry",
    )
    parser.add_argument(
        "--boards",
        nargs="+",
        action=_BoardOverride,
        dest="boards_override",
        default=None,
        metavar="BOARD",
        help="boards for the most recently named --source; repeatable to "
        "override several sources; defaults to registry boards",
    )
    parser.add_argument(
        "--target",
        choices=("store", "files"),
        default="store",
        help="write to the S3-compatible store or the local mirror (default: store)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help=f"mirror root for --target files (default: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=_DEFAULT_LIMIT,
        help=f"postings to ingest per board (default: {_DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--limit-all",
        action="store_true",
        help="ingest whole boards, ignoring --limit",
    )
    args = parser.parse_args(argv)

    if args.limit < 0:
        parser.error("--limit must be non-negative")

    sink: Sink = FileSink(args.data_dir)
    if args.target == "store":
        load_dotenv(_REPO_ROOT / ".env")
        sink = ObjectStore.from_env()

    # _BoardOverride binds each --boards to the --source named before it;
    # sources riding registry defaults need nothing here.
    run_sources = args.source or sorted(sources.SOURCE_BOARDS)
    boards_override: dict[str, list[str]] = args.boards_override or {}

    try:
        return run(
            run_sources=run_sources,
            boards_override=boards_override,
            sink=sink,
            limit=None if args.limit_all else args.limit,
        )
    except (BotoCoreError, ClientError, OSError) as exc:
        # Storage unavailable: the runner cannot work, manifest cannot land.
        print(f"error: storage failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
