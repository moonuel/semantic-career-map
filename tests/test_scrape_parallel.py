"""Tests for the parallel board orchestration in :mod:`scripts.scrape`.

Worker isolation, cross-board dedup under the shared partition lock,
per-source politeness gates, retry/backoff behavior, deterministic manifest
row order, and the interrupt drain path — all against a file-mirror sink and
stub fetchers (no real HTTP).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import requests

from backend.ingestion.raw import FileSink
from backend.ingestion.scrapers import ScrapedJob
from scripts import scrape

_FETCH = Callable[[str, str, int | None], list[ScrapedJob]]


def _job(source_id: str, body: str = "body") -> ScrapedJob:
    """One synthetic posting; differing ``source_id`` differing payload."""
    return ScrapedJob(
        url=f"https://example.co/{source_id}",
        title=f"Role {source_id}",
        description_html=body,
        employer="example",
        location=None,
        country=None,
        employment_type=None,
        date_posted=None,
        salary_min=None,
        salary_max=None,
        salary_currency=None,
        source_id=source_id,
    )


def _defer_fetch(counts: dict[str, int], body: _FETCH) -> _FETCH:
    """A fetch stub that counts calls per ``source/board`` before delegating."""

    def fetch(source: str, board: str, limit: int | None) -> list[ScrapedJob]:
        counts[f"{source}/{board}"] = counts.get(f"{source}/{board}", 0) + 1
        return body(source, board, limit)

    return fetch


def _manifest(root: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Read the run's single manifest back; return (rows, board_status)."""
    paths = list(root.glob("lake/manifests/ingestion/**/*.parquet"))
    assert len(paths) == 1
    rows = pq.read_table(paths[0]).to_pylist()
    board_status = dict(rows[0]["board_status"])
    return rows, board_status


def test_boards_run_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three dry-board fetches overlap on the pool, or the barrier breaks."""
    calls: list[tuple[str, str]] = []
    barrier = threading.Barrier(3)

    def fetch(source: str, board: str, limit: int | None) -> list[ScrapedJob]:
        calls.append((source, board))
        # Sequential execution would strand every waiter past the timeout.
        barrier.wait(timeout=5)
        return [_job(f"{source}:{board}")]

    monkeypatch.setattr("backend.ingestion.raw.board_fetch", fetch)
    code = scrape.run(
        run_sources=["ashby", "greenhouse"],
        boards_override={"ashby": ["supabase"], "greenhouse": ["blee", "bletch"]},
        sink=FileSink(tmp_path),
        limit=None,
        max_workers=3,
    )
    assert code == 0
    assert len(calls) == 3
    rows, board_status = _manifest(tmp_path)
    assert sorted(board_status.values()) == ["ok", "ok", "ok"]
    assert sum(row["action"] == "stored" for row in rows) == 3


def test_cross_board_identical_payload_yields_one_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two boards of one source ingesting the same payload at once: one stored."""
    payload = _job("shared:1", body="identical")

    def fetch(source: str, board: str, limit: int | None) -> list[ScrapedJob]:
        return [payload]

    monkeypatch.setattr("backend.ingestion.raw.board_fetch", fetch)
    code = scrape.run(
        run_sources=["lever"],
        boards_override={"lever": ["one", "two", "three"]},
        sink=FileSink(tmp_path),
        limit=None,
        max_workers=3,
    )
    assert code == 0
    rows, _ = _manifest(tmp_path)
    assert sum(row["action"] == "stored" for row in rows) == 1
    assert sum(row["action"] == "duplicate" for row in rows) == 2
    content_shas = {row["content_sha256"] for row in rows}
    assert len(content_shas) == 1


def test_failed_board_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One board failing leaves siblings ingested; status lands in the header."""

    def fetch(source: str, board: str, limit: int | None) -> list[ScrapedJob]:
        if board == "dead":
            raise requests.ConnectionError("connection severed")
        return [_job(f"{source}:{board}")]

    counts: dict[str, int] = {}
    monkeypatch.setattr(
        "backend.ingestion.raw.board_fetch", _defer_fetch(counts, fetch)
    )
    monkeypatch.setattr("scripts.scrape.time.sleep", lambda _s: None)
    code = scrape.run(
        run_sources=["lever"],
        boards_override={"lever": ["dead", "alive"]},
        sink=FileSink(tmp_path),
        limit=None,
        max_workers=2,
    )
    assert code == 0
    rows, board_status = _manifest(tmp_path)
    assert "severed" in board_status["lever/dead"]
    assert board_status["lever/alive"] == "ok"
    assert sum(row["action"] == "stored" for row in rows) == 1
    # Transient transport failures are retried to exhaustion.
    assert counts["lever/dead"] == scrape._FETCH_ATTEMPTS


def test_retry_then_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two transient failures then success: backoff ran, board is ok."""
    calls: list[str] = []
    sleeps: list[float] = []

    def fetch(source: str, board: str, limit: int | None) -> list[ScrapedJob]:
        calls.append(f"{source}/{board}")
        if calls.count(f"{source}/{board}") <= 2:
            raise requests.ConnectionError("flapped")
        return [_job(f"{source}:{board}")]

    monkeypatch.setattr("backend.ingestion.raw.board_fetch", fetch)
    monkeypatch.setattr("scripts.scrape.time.sleep", lambda s: sleeps.append(s))
    code = scrape.run(
        run_sources=["lever"],
        boards_override={"lever": ["palantir"]},
        sink=FileSink(tmp_path),
        limit=None,
        max_workers=1,
    )
    assert code == 0
    rows, board_status = _manifest(tmp_path)
    assert board_status["lever/palantir"] == "ok"
    assert sum(row["action"] == "stored" for row in rows) == 1
    assert len(sleeps) == 2
    assert all(s > 0 for s in sleeps)
    assert calls == ["lever/palantir"] * 3


def test_non_retryable_http_error_is_single_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP 404 on an unknown board slug: failed once, immediately."""
    sleeps: list[float] = []
    response = requests.models.Response()
    response.status_code = 404

    def fetch(source: str, board: str, limit: int | None) -> list[ScrapedJob]:
        if board == "ghost":
            raise requests.HTTPError("HTTP 404", response=response)
        return [_job(f"{source}:{board}")]

    monkeypatch.setattr("backend.ingestion.raw.board_fetch", fetch)
    monkeypatch.setattr("scripts.scrape.time.sleep", lambda s: sleeps.append(s))
    code = scrape.run(
        run_sources=["lever"],
        boards_override={"lever": ["ghost", "alive"]},
        sink=FileSink(tmp_path),
        limit=None,
        max_workers=1,
    )
    assert code == 0
    rows, board_status = _manifest(tmp_path)
    assert board_status["lever/ghost"] != "ok"
    assert board_status["lever/alive"] == "ok"
    assert sleeps == []
    assert sum(row["action"] == "stored" for row in rows) == 1


def test_manifest_rows_stable_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observation rows are sorted (source, board, sha) regardless of completion order."""

    def fetch(source: str, board: str, limit: int | None) -> list[ScrapedJob]:
        return [_job(f"{source}:{board}")]

    monkeypatch.setattr("backend.ingestion.raw.board_fetch", fetch)
    code = scrape.run(
        run_sources=["greenhouse", "ashby"],
        boards_override={},
        sink=FileSink(tmp_path),
        limit=None,
        max_workers=4,
    )
    assert code == 0
    rows, _ = _manifest(tmp_path)
    keys = [(row["source"], row["board"], row["content_sha256"]) for row in rows]
    assert keys == sorted(keys)


def test_interrupt_drains_and_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KeyboardInterrupt during collection: workers drain, manifest still lands."""

    def fetch(source: str, board: str, limit: int | None) -> list[ScrapedJob]:
        return [_job(f"{source}:{board}")]

    def interrupting_as_completed(
        _futures: dict, *args: object, **kwargs: object
    ) -> Iterator[None]:
        raise KeyboardInterrupt
        yield

    monkeypatch.setattr("backend.ingestion.raw.board_fetch", fetch)
    monkeypatch.setattr("scripts.scrape.as_completed", interrupting_as_completed)
    code = scrape.run(
        run_sources=["lever"],
        boards_override={"lever": ["one", "two", "three"]},
        sink=FileSink(tmp_path),
        limit=None,
        max_workers=3,
    )
    assert code == 130
    rows, board_status = _manifest(tmp_path)
    assert sorted(board_status.values()) == ["ok", "ok", "ok"]
    assert sum(row["action"] == "stored" for row in rows) == 3


def test_board_tasks_collapse_duplicate_slugs() -> None:
    """A repeated ``--boards`` value never schedules the same board twice."""
    tasks = scrape._board_tasks(["lever"], {"lever": ["palantir", "palantir"]})
    assert [t.board for t in tasks] == ["palantir"]
    tasks = scrape._board_tasks(["greenhouse"], {})
    assert [(t.source, t.board) for t in tasks] == [
        ("greenhouse", "anthropic"),
        ("greenhouse", "vercel"),
    ]


def test_dataroles_gate_is_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The curator's fetch concurrency is 1, even under a parallel pool."""

    def fetch(source: str, board: str, limit: int | None) -> list[ScrapedJob]:
        return [_job(f"{source}:{board}")]

    monkeypatch.setattr("backend.ingestion.raw.board_fetch", fetch)
    code = scrape.run(
        run_sources=["dataroles"],
        boards_override={},
        sink=FileSink(tmp_path),
        limit=1,
        max_workers=4,
    )
    assert code == 0
    rows, board_status = _manifest(tmp_path)
    assert board_status == {"dataroles/curated": "ok"}
    assert sum(row["action"] == "stored" for row in rows) == 1
    assert scrape._PER_SOURCE_CONCURRENCY["dataroles"] == 1
    assert scrape._DEFAULT_CONCURRENCY == 2
