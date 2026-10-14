"""Tests for :mod:`scripts.query`.

All hermetic: a fake MCP client records calls and can fail selected queries,
politeness sleeps are patched out, and no network happens (the live endpoint
was verified separately — spec Appendix B). Covered: query-set selection
(``--set``/``--query`` union, ``core`` default, de-duplication), per-query
failure isolation and exit codes, detail-expansion capping, and that runner
arguments reach the client.search keyword body unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.ingestion import mcp
from scripts import query


class _FakeClient:
    """Stand-in for :class:`mcp.HiringCafeClient`: records calls, may fail.

    Args:
        fail_on: Query texts whose search call raises.
        rows_per_search: Synthetic rows returned per successful search.
    """

    def __init__(
        self, fail_on: set[str] | None = None, rows_per_search: int = 2
    ) -> None:
        self.fail_on = fail_on or set()
        self.rows_per_search = rows_per_search
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
                "id": f"row{i}",
                "title": f"Role {i}",
                "company": "Example Analytics",
            }
            for i in range(self.rows_per_search)
        ]

    def details(self, job_id: str) -> dict[str, Any]:
        self.details_calls.append(job_id)
        return {"id": job_id, "description": "A synthetic full posting body."}


@pytest.fixture()
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Patch out politeness sleeps; return the recorded delays."""
    delays: list[float] = []
    monkeypatch.setattr(query.time, "sleep", delays.append)
    return delays


def _install(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> None:
    monkeypatch.setattr(mcp, "HiringCafeClient", lambda **_: client)


def test_default_set_is_core(monkeypatch: pytest.MonkeyPatch, no_sleep) -> None:
    client = _FakeClient()
    _install(monkeypatch, client)
    assert query.main([]) == 0
    queries_run = [c["search"] for c in client.search_calls]
    assert queries_run == list(query._QUERY_SETS["core"])
    assert client.initialized


def test_query_flag_overrides_default(
    monkeypatch: pytest.MonkeyPatch, no_sleep
) -> None:
    client = _FakeClient()
    _install(monkeypatch, client)
    assert query.main(["--query", "rust developer"]) == 0
    queries_run = [c["search"] for c in client.search_calls]
    assert queries_run == ["rust developer"]


def test_set_selection(monkeypatch: pytest.MonkeyPatch, no_sleep) -> None:
    client = _FakeClient()
    _install(monkeypatch, client)
    assert query.main(["--set", "research"]) == 0
    queries_run = [c["search"] for c in client.search_calls]
    assert queries_run == list(query._QUERY_SETS["research"])


def test_set_union_with_queries_and_dedup(
    monkeypatch: pytest.MonkeyPatch, no_sleep
) -> None:
    client = _FakeClient()
    _install(monkeypatch, client)
    assert query.main(["--set", "research", "--set", "core", "--query", "mlops"]) == 0
    queries_run = [c["search"] for c in client.search_calls]
    expected = list(
        dict.fromkeys([*query._QUERY_SETS["research"], *query._QUERY_SETS["core"]])
    )
    assert queries_run == expected


def test_selection_pure_function() -> None:
    assert query._select_queries([], []) == list(query._QUERY_SETS["core"])
    assert query._select_queries([], ["dup", "fresh"]) == ["dup", "fresh"]
    selected = query._select_queries(["core"], ["mlops"])
    assert selected == list(query._QUERY_SETS["core"])


def test_set_all_iterates_every_set() -> None:
    selected = query._select_queries([], [], set_all=True)
    assert selected == [q for members in query._QUERY_SETS.values() for q in members]


def test_set_all_with_extra_query_appended() -> None:
    selected = query._select_queries([], ["extra"], set_all=True)
    assert selected[-1] == "extra"
    assert (
        len(selected)
        == len({q for members in query._QUERY_SETS.values() for q in members}) + 1
    )


def test_set_all_by_flag(monkeypatch: pytest.MonkeyPatch, no_sleep) -> None:
    client = _FakeClient(rows_per_search=0)
    _install(monkeypatch, client)
    assert query.main(["--set-all", "--details", "0"]) == 0
    expected = query._select_queries([], [], set_all=True)
    queries_run = [c["search"] for c in client.search_calls]
    assert queries_run == expected


def test_failed_query_isolated(
    monkeypatch: pytest.MonkeyPatch, no_sleep, capsys: pytest.CaptureFixture
) -> None:
    second = query._QUERY_SETS["core"][1]
    client = _FakeClient(fail_on={second})
    _install(monkeypatch, client)
    assert query.main([]) == 0
    assert "failed" in capsys.readouterr().err
    assert len(client.search_calls) == len(query._QUERY_SETS["core"])


def test_all_failed_exit_one(monkeypatch: pytest.MonkeyPatch, no_sleep) -> None:
    client = _FakeClient(fail_on=set(query._QUERY_SETS["core"]))
    _install(monkeypatch, client)
    assert query.main([]) == 1


def test_details_expansion_capped(monkeypatch: pytest.MonkeyPatch, no_sleep) -> None:
    client = _FakeClient(rows_per_search=2)
    _install(monkeypatch, client)
    assert query.main(["--query", "one role", "--details", "1"]) == 0
    assert len(client.details_calls) == 1
    assert client.details_calls[0] == "row0"


def test_search_arguments_pass_through(
    monkeypatch: pytest.MonkeyPatch, no_sleep
) -> None:
    client = _FakeClient(rows_per_search=0)
    _install(monkeypatch, client)
    assert query.main(["--query", "one role", "--limit", "5", "--details", "0"]) == 0
    call = client.search_calls[0]
    assert call == {"search": "one role", "limit": 5, "posted_salary_only": True}
    assert client.details_calls == []


def test_politeness_delay_per_call(monkeypatch: pytest.MonkeyPatch, no_sleep) -> None:
    client = _FakeClient(rows_per_search=1)
    _install(monkeypatch, client)
    assert query.main(["--query", "one role", "--details", "1"]) == 0
    # one sleep before the search, one before the single details call
    assert len(no_sleep) == 2
    assert query._CALL_DELAY_S in no_sleep


def test_limit_validation_rejects_non_positive(
    monkeypatch: pytest.MonkeyPatch, no_sleep
) -> None:
    with pytest.raises(SystemExit):
        query.main(["--query", "one role", "--limit", "0"])


def test_unknown_set_rejected(monkeypatch: pytest.MonkeyPatch, no_sleep) -> None:
    with pytest.raises(SystemExit):
        query.main(["--set", "not-a-set"])
