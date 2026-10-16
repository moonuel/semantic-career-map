"""Unit tests for :mod:`backend.ingestion.raw` (the raw-layer write path).

Scope: canonicalization, content-hash dedup at the sink boundary, the
FileSink mirror layout, and the per-board intake policy. Live fetch is
``backend/ingestion/scrapers.py``'s concern (its own tests); storage-level
behaviour beyond put/exists is :mod:`backend.storage.store`'s.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.ingestion import raw
from backend.ingestion.scrapers import ScrapedJob

# --- Fixtures ----------------------------------------------------------------


def _job(
    source_id: str,
    *,
    title: str = "Senior Data Scientist",
    employer: str = "Acme",
    description: str = "<p>Do ML things.</p>",
) -> ScrapedJob:
    return ScrapedJob(
        url=f"https://jobs.example.com/{source_id}",
        title=title,
        description_html=description,
        employer=employer,
        location=None,
        country=None,
        employment_type="FULL_TIME",
        date_posted=None,
        salary_min=None,
        salary_max=None,
        salary_currency=None,
        source_id=source_id,
    )


class CountingSink(raw.FileSink):
    """A FileSink that also counts puts, to observe skip-before-write."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.puts = 0

    def put(self, key: str, data: bytes) -> None:
        self.puts += 1
        super().put(key, data)


# --- 1. Canonicalization -----------------------------------------------------


class TestCanonicalJson:
    def test_sorted_keys_ensure_ascii_off(self) -> None:
        assert raw.canonical_json({"b": 1, "a": "é"}) == '{"a": "é", "b": 1}'.encode()

    def test_byte_stable_across_calls(self) -> None:
        payload = {"x": [1, 2], "y": {"k": None}}
        assert raw.canonical_json(payload) == raw.canonical_json(dict(**payload))

    def test_job_payload_hashes_stably(self, tmp_path: Path) -> None:
        payload = raw.job_payload(_job("ashby:acme:1"))
        assert payload == raw.job_payload(_job("ashby:acme:1"))


# --- 2. Ingest dedup ---------------------------------------------------------


class TestIngestPayload:
    def test_first_write_stored_at_content_address(self, tmp_path: Path) -> None:
        event = raw.ingest_payload(
            "ashby",
            raw.canonical_json({"k": "v"}),
            sink=CountingSink(tmp_path),
            existing_hashes=set(),
        )
        assert event.event == "stored"
        from backend.storage import keys

        assert event.raw_key == keys.raw_payload("ashby", event.content_sha256)

    def test_written_key_holds_payload_bytes(self, tmp_path: Path) -> None:
        sink: raw.FileSink = raw.FileSink(tmp_path)
        payload = raw.canonical_json({"k": "v"})
        event = raw.ingest_payload("ashby", payload, sink=sink, existing_hashes=set())
        assert (tmp_path / event.raw_key).read_bytes() == payload

    def test_repeat_run_skips_via_sink_exists(self, tmp_path: Path) -> None:
        payload = raw.canonical_json({"k": "v"})
        first = raw.ingest_payload(
            "ashby", payload, sink=raw.FileSink(tmp_path), existing_hashes=set()
        )
        second = raw.ingest_payload(
            "ashby", payload, sink=raw.FileSink(tmp_path), existing_hashes=set()
        )
        assert first.event == "stored"
        assert second.event == "duplicate"
        assert second.raw_key == ""

    def test_in_run_set_skips_before_write(self, tmp_path: Path) -> None:
        sink = CountingSink(tmp_path)
        hashes: set[str] = set()
        payload = raw.canonical_json({"k": "v"})
        raw.ingest_payload("ashby", payload, sink=sink, existing_hashes=hashes)
        puts_after_first = sink.puts
        event = raw.ingest_payload("ashby", payload, sink=sink, existing_hashes=hashes)
        assert event.event == "duplicate"
        assert sink.puts == puts_after_first


# --- 3. Board intake ---------------------------------------------------------


class TestUsableJobs:
    def test_drop_empty_title_or_description(self) -> None:
        jobs = [
            _job("ashby:a:1", title=""),
            _job("ashby:a:2", description=""),
            _job("ashby:a:3"),
        ]
        kept = raw.usable_jobs(jobs, limit=None)
        assert [j.source_id for j in kept] == ["ashby:a:3"]

    def test_stable_source_id_order(self) -> None:
        jobs = [_job(f"ashby:a:{i}") for i in ["c9", "a1", "b4"]]
        kept = raw.usable_jobs(jobs, limit=None)
        assert [j.source_id for j in kept] == ["ashby:a:a1", "ashby:a:b4", "ashby:a:c9"]

    def test_limit_slices_sorted_order(self) -> None:
        jobs = [_job(f"ashby:a:{i}") for i in [5, 1, 3]]
        kept = raw.usable_jobs(jobs, limit=2)
        assert [j.source_id for j in kept] == ["ashby:a:1", "ashby:a:3"]

    def test_rejects_negative_limit(self) -> None:
        jobs = [_job("ashby:a:1")]
        with pytest.raises(ValueError, match="non-negative"):
            raw.usable_jobs(jobs, limit=-1)


class TestIngestBoard:
    def test_every_usable_posting_ingests(self, tmp_path: Path) -> None:
        sink = raw.FileSink(tmp_path)
        hashes: set[str] = set()
        events = raw.ingest_board(
            [_job("ashby:a:1"), _job("ashby:a:2")],
            source="ashby",
            board="acme",
            sink=sink,
            existing_hashes=hashes,
        )
        assert [e.event for e in events] == ["stored", "stored"]
        assert {e.content_sha256 for e in events} == hashes

    def test_limit_bounds_intake(self, tmp_path: Path) -> None:
        sink = raw.FileSink(tmp_path)
        events = raw.ingest_board(
            [_job(f"ashby:a:{i}") for i in range(5)],
            source="ashby",
            board="acme",
            sink=sink,
            existing_hashes=set(),
            limit=2,
        )
        assert len(events) == 2

    def test_cross_board_duplicates_share_source_hashes(self, tmp_path: Path) -> None:
        sink = raw.FileSink(tmp_path)
        hashes: set[str] = set()
        first = raw.ingest_board(
            [_job("ashby:a:1")],
            source="ashby",
            board="acme",
            sink=sink,
            existing_hashes=hashes,
        )
        second = raw.ingest_board(
            [_job("ashby:a:1")],
            source="ashby",
            board="beta",
            sink=sink,
            existing_hashes=hashes,
        )
        assert first[0].event == "stored"
        assert second[0].event == "duplicate"


# --- 3b. Board-fetch dispatch -------------------------------------------------


class TestBoardFetch:
    """``board_fetch`` — the uniform ``(board, limit)`` dispatch contract."""

    def test_forwards_board_and_limit_to_fetcher(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, int | None]] = []

        def fake_fetcher(board: str, limit: int | None) -> list[ScrapedJob]:
            calls.append((board, limit))
            return [_job("ashby:a:1")]

        monkeypatch.setitem(raw._FETCHERS, "ashby", fake_fetcher)
        jobs = raw.board_fetch("ashby", "acme", 7)
        assert calls == [("acme", 7)]
        assert [j.source_id for j in jobs] == ["ashby:a:1"]

    def test_dispatch_covers_registry(self) -> None:
        from backend.ingestion import sources

        assert set(raw._FETCHERS) == set(sources.SOURCE_BOARDS)


# --- 4. FileSink mirror ------------------------------------------------------


class TestFileSink:
    def test_mirror_file_at_validated_key(self, tmp_path: Path) -> None:
        from backend.storage import ids, keys

        key = keys.raw_payload("ashby", ids.content_sha256(b"x"))
        sink = raw.FileSink(tmp_path)
        sink.put(key, b"x")
        assert (tmp_path / key).read_bytes() == b"x"

    def test_exists_reflects_put(self, tmp_path: Path) -> None:
        from backend.storage import ids, keys

        key = keys.raw_payload("ashby", ids.content_sha256(b"x"))
        sink = raw.FileSink(tmp_path)
        assert not sink.exists(key)
        sink.put(key, b"x")
        assert sink.exists(key)

    def test_rejects_invalid_key(self, tmp_path: Path) -> None:
        sink = raw.FileSink(tmp_path)
        with pytest.raises(ValueError):
            sink.put("lake/raw/source=nope/sha256=bad.json", b"x")
        with pytest.raises(ValueError):
            sink.exists("../escape")
