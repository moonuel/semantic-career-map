"""Tests for :mod:`scripts.query`.

All hermetic: a fake MCP client records calls, can fail selected queries,
and serves synthetic records; politeness sleeps are patched out; no network
happens (the live endpoint was verified separately — spec Appendix B). The
runner also writes provenance: raw objects under the temporary file mirror
and one Parquet manifest per run, asserted through the mirror.

Covered: query-set selection (``--set``/``--query``/``--set-all`` union,
``core`` default, de-duplication), per-query failure isolation and exit
codes, detail-expansion capping, raw-layer intake (shells skipped,
duplicates idempotent), the zero-row survey manifest, and slug collision
skipping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from backend.ingestion import mcp
from backend.storage import ids
from scripts import query


class _FakeClient:
    """Stand-in for :class:`mcp.HiringCafeClient`: records calls, may fail.

    Args:
        fail_on: Query texts whose search call raises.
        rows_per_search: Synthetic rows returned per successful search.
        shell_details: Serve detail records without title/description, so
            the runner's intake filter skips every posting as a shell.
    """

    def __init__(
        self,
        fail_on: set[str] | None = None,
        rows_per_search: int = 2,
        shell_details: bool = False,
    ) -> None:
        self.fail_on = fail_on or set()
        self.rows_per_search = rows_per_search
        self.shell_details = shell_details
        self.initialized = False
        self.search_calls: list[dict[str, Any]] = []
        self.details_calls: list[str] = []

    def initialize(self) -> dict[str, Any]:
        self.initialized = True
        return {"serverInfo": {"name": "fake", "version": "0"}}

    def search(self, **arguments: Any) -> list[dict[str, Any]]:
        self.search_calls.append(arguments)
        if arguments["search"] in self.fail_on:
            raise mcp.MCPError(f"mcp search failed: {arguments['search']}")
        return [
            {
                "id": f"{arguments['search']}-{i}",
                "title": f"Role {i} of {arguments['search']}",
                "company": "Example Analytics",
            }
            for i in range(self.rows_per_search)
        ]

    def details(self, job_id: str) -> dict[str, Any]:
        self.details_calls.append(job_id)
        if self.shell_details:
            return {"id": job_id}
        return {
            "id": job_id,
            "title": f"Role {job_id}",
            "company": "Example Analytics",
            "description": f"Full posting body for {job_id}.",
        }


@pytest.fixture()
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Patch out politeness sleeps; return the recorded delays."""
    delays: list[float] = []
    monkeypatch.setattr(query.time, "sleep", delays.append)
    return delays


def _install(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> None:
    monkeypatch.setattr(mcp, "HiringCafeClient", lambda **_: client)


def _main(
    monkeypatch: pytest.MonkeyPatch, client: _FakeClient, tmp_path: Path, *argv: str
) -> int:
    """Run the CLI against the fake client with a file mirror sink."""
    _install(monkeypatch, client)
    return query.main(["--target", "files", "--data-dir", str(tmp_path), *argv])


def _raw_keys(mirror: Path) -> list[Path]:
    """Raw objects written under ``lake/raw/source=hiringcafe/``."""
    return list(mirror.glob("lake/raw/source=hiringcafe/sha256=*.json"))


def _manifest_rows(mirror: Path) -> list[dict[str, Any]]:
    """The run's single manifest; rows for header+observation columns."""
    paths = list(mirror.glob("lake/manifests/ingestion/**/*.parquet"))
    assert len(paths) == 1, f"expected one manifest, got {paths}"
    return pq.read_table(paths[0]).to_pylist()


def _all_manifests(mirror: Path) -> list[dict[str, Any]]:
    """Every run's manifest rows, newest (uuid7-ordered) manifest first."""
    paths = sorted(mirror.glob("lake/manifests/ingestion/**/*.parquet"))
    return [row for path in paths for row in pq.read_table(path).to_pylist()]


def _board_status(row: dict[str, Any]) -> dict[str, str]:
    """The header's per-board outcome map off a manifest row."""
    return dict(row["board_status"])


# --- selection ----------------------------------------------------------------


def test_default_set_is_core(
    monkeypatch: pytest.MonkeyPatch, no_sleep, tmp_path: Path
) -> None:
    client = _FakeClient(rows_per_search=0)
    assert _main(monkeypatch, client, tmp_path) == 0
    queries_run = [c["search"] for c in client.search_calls]
    assert queries_run == list(query._QUERY_SETS["core"])
    assert client.initialized


def test_query_flag_overrides_default(
    monkeypatch: pytest.MonkeyPatch, no_sleep, tmp_path: Path
) -> None:
    client = _FakeClient(rows_per_search=0)
    assert _main(monkeypatch, client, tmp_path, "--query", "rust developer") == 0
    assert [c["search"] for c in client.search_calls] == ["rust developer"]


def test_set_selection(
    monkeypatch: pytest.MonkeyPatch, no_sleep, tmp_path: Path
) -> None:
    client = _FakeClient(rows_per_search=0)
    assert _main(monkeypatch, client, tmp_path, "--set", "research") == 0
    assert [c["search"] for c in client.search_calls] == list(
        query._QUERY_SETS["research"]
    )


def test_set_all_iterates_every_set(
    monkeypatch: pytest.MonkeyPatch, no_sleep, tmp_path: Path
) -> None:
    client = _FakeClient(rows_per_search=0)
    assert _main(monkeypatch, client, tmp_path, "--set-all") == 0
    expected = [q for members in query._QUERY_SETS.values() for q in members]
    assert [c["search"] for c in client.search_calls] == expected


def test_set_union_with_queries_and_dedup(
    monkeypatch: pytest.MonkeyPatch, no_sleep, tmp_path: Path
) -> None:
    client = _FakeClient(rows_per_search=0)
    argv = ("--set", "research", "--set", "core", "--query", "mlops")
    assert _main(monkeypatch, client, tmp_path, *argv) == 0
    queries_run = [c["search"] for c in client.search_calls]
    expected = list(
        dict.fromkeys([*query._QUERY_SETS["research"], *query._QUERY_SETS["core"]])
    )
    assert queries_run == expected


def test_selection_pure_function() -> None:
    assert query._select_queries([], []) == list(query._QUERY_SETS["core"])
    assert query._select_queries([], ["dup", "fresh"]) == ["dup", "fresh"]
    assert query._select_queries(["core"], ["mlops"]) == list(query._QUERY_SETS["core"])
    all_sets = query._select_queries([], [], set_all=True)
    assert all_sets == [q for members in query._QUERY_SETS.values() for q in members]


# --- failure isolation and exit codes ------------------------------------------


def test_failed_query_isolated(
    monkeypatch: pytest.MonkeyPatch,
    no_sleep,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    second = query._QUERY_SETS["core"][1]
    client = _FakeClient(fail_on={second}, rows_per_search=1)
    assert _main(monkeypatch, client, tmp_path) == 0
    assert "failed" in capsys.readouterr().err
    assert len(client.search_calls) == len(query._QUERY_SETS["core"])
    rows = _manifest_rows(tmp_path)
    assert rows[0]["boards_attempted"] == len(query._QUERY_SETS["core"])
    assert rows[0]["boards_failed"] == 1
    assert rows[0]["status"] == "partial"
    assert sum(row["action"] == "stored" for row in rows) == (
        len(query._QUERY_SETS["core"]) - 1
    )


def test_all_failed_exit_one(
    monkeypatch: pytest.MonkeyPatch, no_sleep, tmp_path: Path
) -> None:
    client = _FakeClient(fail_on=set(query._QUERY_SETS["core"]), rows_per_search=0)
    assert _main(monkeypatch, client, tmp_path) == 1
    # every query failed: nothing ingested, the record is a zero-row manifest
    assert _manifest_rows(tmp_path) == []


def test_handshake_failure_exit_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    class _HandshakeFail(_FakeClient):
        def initialize(self) -> dict[str, Any]:
            raise mcp.MCPError("mcp initialize: down")

    monkeypatch.setattr(mcp, "HiringCafeClient", lambda **_: _HandshakeFail())
    assert query.main(["--target", "files", "--data-dir", str(tmp_path)]) == 1
    assert "handshake" in capsys.readouterr().err


def test_limit_validation_rejects_non_positive(monkeypatch, no_sleep, tmp_path) -> None:
    with pytest.raises(SystemExit):
        _main(monkeypatch, _FakeClient(), tmp_path, "--limit", "0")


def test_unknown_set_rejected(monkeypatch, no_sleep, tmp_path) -> None:
    with pytest.raises(SystemExit):
        _main(monkeypatch, _FakeClient(), tmp_path, "--set", "not-a-set")


# --- fetch semantics ------------------------------------------------------------


def test_details_expansion_capped(
    monkeypatch: pytest.MonkeyPatch, no_sleep, tmp_path: Path
) -> None:
    client = _FakeClient()
    assert (
        _main(monkeypatch, client, tmp_path, "--query", "one role", "--details", "1")
        == 0
    )
    assert len(client.details_calls) == 1
    assert client.details_calls[0] == "one role-0"


def test_search_arguments_pass_through(
    monkeypatch: pytest.MonkeyPatch, no_sleep, tmp_path: Path
) -> None:
    client = _FakeClient(rows_per_search=0)
    assert (
        _main(
            monkeypatch,
            client,
            tmp_path,
            "--query",
            "one role",
            "--limit",
            "5",
            "--details",
            "0",
        )
        == 0
    )
    assert client.search_calls[0] == {
        "search": "one role",
        "limit": 5,
        "posted_salary_only": True,
    }
    assert client.details_calls == []


def test_politeness_delay_per_call(
    monkeypatch: pytest.MonkeyPatch, no_sleep, tmp_path: Path
) -> None:
    client = _FakeClient(rows_per_search=1)
    assert (
        _main(monkeypatch, client, tmp_path, "--query", "one role", "--details", "1")
        == 0
    )
    # one sleep before the search, one before the single details call
    assert len(no_sleep) == 2
    assert query._CALL_DELAY_S in no_sleep


# --- provenance writes -----------------------------------------------------------


def test_ingest_stores_raw_and_manifest_row(
    monkeypatch: pytest.MonkeyPatch, no_sleep, tmp_path: Path
) -> None:
    client = _FakeClient(rows_per_search=1)
    assert (
        _main(monkeypatch, client, tmp_path, "--query", "one role", "--details", "1")
        == 0
    )

    keys = _raw_keys(tmp_path)
    assert len(keys) == 1
    assert keys[0].name.startswith("sha256=")

    rows = _manifest_rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "hiringcafe"
    assert row["board"] == "one-role"
    assert row["action"] == "stored"
    assert row["source_id"] == "hiringcafe:one role-0"
    assert row["employer"] == "Example Analytics"
    assert ids.is_sha256_hex(row["content_sha256"])
    assert row["boards_attempted"] == 1
    assert row["boards_failed"] == 0
    assert row["status"] == "ok"
    assert _board_status(row) == {"hiringcafe/one-role": "ok"}
    assert isinstance(row["pipeline_version"], str) and row["pipeline_version"]
    assert ids.is_sha256_hex(row["configuration_hash"])


def test_rerun_is_all_duplicate(
    monkeypatch: pytest.MonkeyPatch, no_sleep, tmp_path: Path
) -> None:
    client = _FakeClient(rows_per_search=1)
    argv = ("--query", "one role", "--details", "1")
    assert _main(monkeypatch, client, tmp_path, *argv) == 0
    # same client, same synthetic content: the second run must observe duplicates
    assert _main(monkeypatch, client, tmp_path, *argv) == 0
    manifests = list(tmp_path.glob("lake/manifests/ingestion/**/*.parquet"))
    assert len(manifests) == 2
    oldest_path, newest_path = min(manifests), max(manifests)
    first = pq.read_table(oldest_path).to_pylist()
    newest = pq.read_table(newest_path).to_pylist()
    assert first and any(row["action"] == "stored" for row in first)
    assert newest and all(row["action"] == "duplicate" for row in newest)
    # still exactly one raw object: content-addressed, idempotent
    assert len(_raw_keys(tmp_path)) == 1


def test_survey_mode_writes_zero_row_manifest(
    monkeypatch: pytest.MonkeyPatch, no_sleep, tmp_path: Path
) -> None:
    client = _FakeClient(rows_per_search=2)
    assert (
        _main(monkeypatch, client, tmp_path, "--query", "one role", "--details", "0")
        == 0
    )
    assert client.details_calls == []
    assert not _raw_keys(tmp_path)
    rows = _manifest_rows(tmp_path)
    assert rows == []


def test_shell_records_skipped(
    monkeypatch: pytest.MonkeyPatch,
    no_sleep,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    client = _FakeClient(rows_per_search=1, shell_details=True)
    assert (
        _main(monkeypatch, client, tmp_path, "--query", "one role", "--details", "1")
        == 0
    )
    assert not _raw_keys(tmp_path)
    assert _manifest_rows(tmp_path) == []
    assert "no title/description" in capsys.readouterr().err


def test_slug_collision_skips_second_query(
    monkeypatch: pytest.MonkeyPatch,
    no_sleep,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    client = _FakeClient(rows_per_search=0)
    argv = ("--query", "One Role", "--query", "one role")
    assert _main(monkeypatch, client, tmp_path, *argv) == 0
    assert len(client.search_calls) == 1  # second query is the same window, skipped
    assert "already covered" in capsys.readouterr().err


def test_unsluggable_query_fails_that_query_only(
    monkeypatch: pytest.MonkeyPatch,
    no_sleep,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    client = _FakeClient(rows_per_search=1)
    argv = ("--query", "!!!", "--query", "valid query")
    exit_code = _main(monkeypatch, client, tmp_path, *argv)
    assert exit_code == 0
    assert "failed" in capsys.readouterr().err
    rows = _manifest_rows(tmp_path)
    assert rows[0]["boards_attempted"] == 2
    assert rows[0]["boards_failed"] == 1
    assert rows[0]["status"] == "partial"
    status_map = _board_status(rows[0])
    assert status_map["hiringcafe/valid-query"] == "ok"
    assert status_map["hiringcafe/!!!"] != "ok"
