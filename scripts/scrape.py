"""Scheduled scraper entry point: every source site, one run apiece.

The single process the scheduler knows. No arguments runs the full declared
picture — every source in :data:`~backend.ingestion.sources.SOURCE_BOARDS`
with its declared boards — through ``backend.ingestion.scrapers`` (the live
fetch) and ``backend.ingestion.raw`` (the content-addressed write).

Boards run in parallel: one worker per ``(source, board)`` on a thread pool
(fetches are pure I/O), bounded by ``--max-workers`` and by per-source
politeness gates so no host sees more than its declared concurrency — the
dataroles.co curator never multiplexes, since its page pacing lives inside
its own fetcher. A board fetch that fails transiently retries with backoff
before the board is failed. Per-source flags narrow a run for ad-hoc work:

    python scripts/scrape.py                     # the whole registry
    python scripts/scrape.py --source lever --boards palantir --limit-all

One fetch per board; one ``lake/manifests/ingestion/`` manifest for the run,
recording every stored / duplicate observation plus board-level outcomes in
a stable (source, board, content hash) order so identical data builds
byte-identical manifests. Storage dedup is content-hash level, so scheduled
re-runs are no-ops while staying observable — the manifest is that
observability.

Sources are isolated: a board that fails is recorded in the run header and
the run continues. Ctrl+C cancels boards that have not started, waits for
the in-flight fetches, and still writes the manifest for whatever completed.
The exit code is nonzero only if every run source failed entirely (or
storage itself went away, making the manifest fail with it); an interrupted
run exits 130.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from backend.ingestion import raw, sources
from backend.ingestion.manifest import (
    BatchObservation,
    build_ingestion_manifest,
    event_to_observation,
    write_ingestion_manifest,
)
from backend.ingestion.raw import FileSink, Sink
from backend.ingestion.scrapers import ScrapedJob
from backend.ingestion.sources import SOURCE_BOARDS
from backend.storage import ids
from backend.storage.store import ObjectStore

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"
_DEFAULT_LIMIT = 25
_DEFAULT_MAX_WORKERS = 4
_MAX_WORKERS_CAP = 16

# Politeness: boards of one source hit the same ATS host. ATS fetches are a
# single GET per board, so two concurrent boards per host stay well inside
# what these public, key-free read APIs tolerate; dataroles.co is a full
# page-by-page walk whose pacing already lives inside its fetcher, so it
# must never multiplex.
_PER_SOURCE_CONCURRENCY = {"dataroles": 1}
_DEFAULT_CONCURRENCY = 2

_FETCH_ATTEMPTS = 3
_BACKOFF_S = 0.5
_RETRY_STATUSES = frozenset({429, 502, 503, 504})
_RETRY_AFTER_CAP_S = 60.0


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


@dataclass(frozen=True)
class _BoardTask:
    """One unit of parallel work: fetch and ingest a single board."""

    source: str
    board: str


def _board_tasks(
    run_sources: list[str], boards_override: dict[str, list[str]]
) -> list[_BoardTask]:
    """Flatten the run registry into per-board tasks, in registry order.

    Duplicate board slugs (a repeated ``--boards`` value) collapse so a
    board is never scraped twice in one run.
    """
    tasks: list[_BoardTask] = []
    for source in run_sources:
        boards = boards_override.get(source) or SOURCE_BOARDS[source]
        tasks.extend(_BoardTask(source, board) for board in dict.fromkeys(boards))
    return tasks


def _error_text(exc: Exception) -> str:
    """Compact single-line error message for a board status entry."""
    return " ".join(str(exc).split())[:200] or exc.__class__.__name__


def _retryable(exc: requests.RequestException) -> bool:
    """Whether a failed board fetch is worth retrying.

    Transport failures (connection, timeout) and the throttling / gateway
    statuses mean "try again later"; a 404 on an unknown board slug, a JSON
    decode error, or any other deterministic failure does not.
    """
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    response = getattr(exc, "response", None)
    return response is not None and response.status_code in _RETRY_STATUSES


def _retry_delay(attempt: int, exc: requests.RequestException) -> float:
    """Seconds to wait before fetch attempt ``attempt + 1``.

    Honors a numeric ``Retry-After`` header (capped) over the exponential
    backoff floor, and jitters so concurrent boards do not re-sync.

    Args:
        attempt: Zero-based index of the attempt that just failed.
        exc: The failed attempt's exception.

    Returns:
        A positive sleep duration in seconds.
    """
    delay = _BACKOFF_S * 2**attempt
    response = getattr(exc, "response", None)
    after = response.headers.get("Retry-After") if response is not None else None
    if after is not None:
        try:
            delay = max(delay, min(float(after), _RETRY_AFTER_CAP_S))
        except ValueError:
            pass
    return delay + random.uniform(0, 0.25 * _BACKOFF_S)


def _fetch_with_retries(source: str, board: str, limit: int | None) -> list[ScrapedJob]:
    """Fetch one board, retrying transient failures with backoff.

    A board's retries are its own: they run on that board's worker and never
    delay the other boards. After ``_FETCH_ATTEMPTS`` tries the exception
    propagates to :func:`_scrape_board`, which records the board as failed.

    Args:
        source: Allowlisted raw source.
        board: The board slug to scrape.
        limit: Per-board posting bound; ``None`` takes whole boards.

    Returns:
        The board's postings, as from :func:`raw.board_fetch`.

    Raises:
        requests.RequestException: after the final attempt still fails, or
            on a non-retryable failure.
    """
    for attempt in range(_FETCH_ATTEMPTS):
        try:
            return raw.board_fetch(source, board, limit)
        except requests.RequestException as exc:
            if attempt == _FETCH_ATTEMPTS - 1 or not _retryable(exc):
                raise
            wait = _retry_delay(attempt, exc)
            print(
                f"{source}/{board}: transient fetch failure "
                f"({_error_text(exc)}); retrying in {wait:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise AssertionError("unreachable: retry loop either returns or raises")


def _scrape_board(
    task: _BoardTask,
    *,
    sink: Sink,
    limit: int | None,
    gates: dict[str, threading.Semaphore],
    dedup_locks: dict[str, threading.Lock],
) -> tuple[list[BatchObservation], str]:
    """Fetch and ingest one board; run on a thread pool worker.

    Board isolation lives here: a transport or registry failure becomes the
    returned status string, while storage failures propagate — a run whose
    store is down cannot pretend it ingested anything. The fetch runs under
    the source's politeness gate; ingestion takes the source partition's
    dedup lock per payload so concurrent boards of one source cannot both
    report the same content as newly stored.

    Args:
        task: The ``(source, board)`` to scrape.
        sink: Where raw objects are written (shared, thread-safe).
        limit: Per-board posting bound; ``None`` takes whole boards.
        gates: Per-source fetch concurrency semaphores.
        dedup_locks: Per-source dedup locks passed into
            :func:`raw.ingest_board`.

    Returns:
        ``(observations, status)`` — the board's manifest rows and its
        ``"ok"``/error status for the run header.
    """
    try:
        with gates[task.source]:
            jobs = _fetch_with_retries(task.source, task.board, limit)
        events = raw.ingest_board(
            jobs,
            source=task.source,
            board=task.board,
            sink=sink,
            existing_hashes=set(),
            limit=limit,
            dedup_lock=dedup_locks[task.source],
        )
    except (requests.RequestException, ValueError) as exc:
        return [], _error_text(exc)
    observed_at = datetime.now(UTC)
    observations = [
        event_to_observation(event, observed_at=observed_at) for event in events
    ]
    return observations, "ok"


def run(
    *,
    run_sources: list[str],
    boards_override: dict[str, list[str]],
    sink: Sink,
    limit: int | None,
    max_workers: int,
) -> int:
    """Scrape every board in parallel, then write the run's single batch manifest.

    Workers return their observations; aggregation, printing, and the manifest
    stay on the main thread, so no worker prints and no manifest state is
    shared. On KeyboardInterrupt, boards that have not started cancel and
    in-flight ones drain, then the (partial) manifest is still written.

    Args:
        run_sources: Sources to scrape, in registry order.
        boards_override: Per-source board list when ``--boards`` was given;
            registry defaults otherwise.
        sink: The ingestion sink (store or file mirror).
        limit: Per-board posting bound; ``None`` takes whole boards.
        max_workers: Upper bound on concurrent board workers.

    Returns:
        Process exit code: 130 after an interrupt, else 1 only if every run
        source failed entirely.
    """
    run_started_at = datetime.now(UTC)
    ingestion_id = ids.new_id()
    tasks = _board_tasks(run_sources, boards_override)
    gates = {
        source: threading.Semaphore(
            _PER_SOURCE_CONCURRENCY.get(source, _DEFAULT_CONCURRENCY)
        )
        for source in run_sources
    }
    dedup_locks = {source: threading.Lock() for source in run_sources}

    all_observations: list[BatchObservation] = []
    board_status: dict[str, str] = {}
    boards_attempted = 0
    total_written = 0
    total_duplicates = 0
    interrupted = False

    executor = ThreadPoolExecutor(max_workers=min(max_workers, len(tasks) or 1))
    absorbed: set[_BoardTask] = set()

    def absorb(
        task: _BoardTask, observations: list[BatchObservation], status: str
    ) -> None:
        """Merge one finished board into the run's aggregates and print its line."""
        nonlocal total_written, total_duplicates, boards_attempted
        absorbed.add(task)
        all_observations.extend(observations)
        board_status[f"{task.source}/{task.board}"] = status
        boards_attempted += 1
        written = sum(o.action == "stored" for o in observations)
        duplicates = sum(o.action == "duplicate" for o in observations)
        total_written += written
        total_duplicates += duplicates
        print(
            f"{task.source}/{task.board}: {status}"
            if status != "ok"
            else f"{task.source}/{task.board}: written={written} duplicates={duplicates}"
        )

    try:
        futures: dict[Future[tuple[list[BatchObservation], str]], _BoardTask] = {
            executor.submit(
                _scrape_board,
                task,
                sink=sink,
                limit=limit,
                gates=gates,
                dedup_locks=dedup_locks,
            ): task
            for task in tasks
        }
        try:
            for future in as_completed(futures):
                observations, status = future.result()
                absorb(futures[future], observations, status)
        except KeyboardInterrupt:
            interrupted = True
            # Cancel not-yet-started workers, wait out the in-flight ones.
            executor.shutdown(wait=True, cancel_futures=True)
            for future, task in futures.items():
                if not future.cancelled() and task not in absorbed:
                    observations, status = future.result()
                    absorb(task, observations, status)
            print("interrupted: cancelled remaining boards, in-flight ones drained")
    finally:
        executor.shutdown(wait=True)

    boards_failed = sum(v != "ok" for v in board_status.values())
    status = (
        "failed"
        if boards_attempted and boards_failed == boards_attempted
        else "partial"
        if boards_failed
        else "ok"
    )

    failed_sources = 0
    for source in run_sources:
        attempted_statuses = [
            v for k, v in board_status.items() if k.startswith(f"{source}/")
        ]
        if not attempted_statuses:  # interrupted before this source ran
            continue
        source_failed = sum(v != "ok" for v in attempted_statuses)
        source_written = sum(
            o.action == "stored" for o in all_observations if o.source == source
        )
        source_duplicates = sum(
            o.action == "duplicate" for o in all_observations if o.source == source
        )
        if source_written == source_duplicates == 0 and source_failed == len(
            attempted_statuses
        ):
            failed_sources += 1
            print(
                f"{source}: written=0 duplicates=0 failed={source_failed} "
                "boards — source down"
            )
        else:
            print(f"{source}: written={source_written} duplicates={source_duplicates}")

    manifest = build_ingestion_manifest(
        sorted(all_observations, key=lambda o: (o.source, o.board, o.content_sha256)),
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
        board_status=dict(sorted(board_status.items())),
    )
    manifest_key = write_ingestion_manifest(
        sink, manifest, ingested_on=run_started_at.date(), ingestion_id=ingestion_id
    )
    print(
        f"boards failed: {boards_failed}/{boards_attempted}; "
        f"manifest={manifest_key}; total written={total_written}"
    )
    return 130 if interrupted else (1 if failed_sources == len(run_sources) else 0)


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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=_DEFAULT_MAX_WORKERS,
        metavar="N",
        help=f"boards scraped concurrently (default: {_DEFAULT_MAX_WORKERS})",
    )
    args = parser.parse_args(argv)

    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")

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
            max_workers=min(args.max_workers, _MAX_WORKERS_CAP),
        )
    except (BotoCoreError, ClientError, OSError) as exc:
        # Storage unavailable: the runner cannot work, manifest cannot land.
        print(f"error: storage failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
