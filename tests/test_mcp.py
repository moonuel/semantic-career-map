"""Tests for :mod:`backend.ingestion.mcp`.

Hermetic tests against the JSON-RPC response shapes verified live on
2026-10-14 (spec Appendix B). Fixtures under ``tests/fixtures/hiringcafe/``
are **synthetic minimal records** shaped exactly like the verified
``structuredContent`` payloads (dual ``content`` text + sibling
``structuredContent`` blocks); no real posting data. The transport is a fake
``requests.Session`` recording request bodies, since the endpoint is
stateless (one POST per method) and the client owns session construction.

Not covered here: live behavior (a ``network``-marked live check can follow
the ``test_scrapers.py`` pattern once the query path is wired end-to-end).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from backend.ingestion import mcp, raw
from backend.storage import ids, keys

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hiringcafe"


def load_fixture(name: str) -> str:
    """Return the raw text of a stored MCP response fixture."""
    return (_FIXTURES / name).read_text()


class _FakeResponse:
    """Stand-in for ``requests.Response`` serving a stored JSON body."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    """Stand-in for ``requests.Session`` recording POST bodies per route.

    ``routes`` maps method name → list of payloads served in order; a method
    without a route serves a generic error body.
    """

    def __init__(self, routes: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.routes = routes or {}
        self.headers: dict[str, str] = {}
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, json: dict[str, Any], timeout: float) -> _FakeResponse:
        self.requests.append(json)
        method = json["method"]
        queue = self.routes.get(method)
        if not queue:
            return _FakeResponse(
                {"jsonrpc": "2.0", "id": json["id"], "error": {"message": "boom"}}
            )
        return _FakeResponse(
            {"jsonrpc": "2.0", "id": json["id"], "result": queue.pop(0)}
        )


def make_client(session: _FakeSession) -> mcp.HiringCafeClient:
    """Build a client over a fake session (patched in)."""
    original = mcp.requests.Session
    mcp.requests.Session = lambda: session  # type: ignore[assignment]
    try:
        return mcp.HiringCafeClient()
    finally:
        mcp.requests.Session = original


def test_initialize_handshake() -> None:
    session = _FakeSession(
        {
            "initialize": [
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "hiringcafe-jobs", "version": "0.1.0"},
                }
            ]
        }
    )
    client = make_client(session)
    result = client.initialize()
    assert result["serverInfo"]["name"] == "hiringcafe-jobs"
    body = session.requests[0]
    assert body["method"] == "initialize"
    assert body["params"]["protocolVersion"] == "2025-06-18"
    assert body["params"]["clientInfo"]["name"] == "semantic-career-map"


def test_search_parses_structured_content() -> None:
    session = _FakeSession(
        {"tools/call": [json.loads(load_fixture("search_call.json"))["result"]]}
    )
    client = make_client(session)
    jobs = client.search(search="ml platform engineer", limit=10)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["id"] == "ex1mpla1c0de16ch"
    assert job["compensation"]["minimum"] == 180000
    assert job["seniorityLevel"] == "Senior Level"


def test_search_clamps_limit_locally() -> None:
    session = _FakeSession(
        {"tools/call": [json.loads(load_fixture("search_call.json"))["result"]]}
    )
    client = make_client(session)
    client.search(search="ml platform engineer", limit=30)
    body = session.requests[-1]
    assert body["params"]["arguments"]["limit"] == mcp.SEARCH_CAP


def test_search_tool_error_raises() -> None:
    session = _FakeSession(
        {
            "tools/call": [
                {"content": [{"type": "text", "text": "bad filter"}], "isError": True}
            ]
        }
    )
    client = make_client(session)
    with pytest.raises(mcp.MCPError, match="tool error: bad filter"):
        client.search(search="ml platform engineer")


def test_search_transport_error_raises() -> None:
    class _DownSession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def post(self, url: str, json: Any, timeout: float) -> Any:
            raise requests.ConnectionError("down")

    original = mcp.requests.Session
    mcp.requests.Session = lambda: _DownSession()  # type: ignore[assignment]
    try:
        client = mcp.HiringCafeClient()
    finally:
        mcp.requests.Session = original
    with pytest.raises(mcp.MCPError, match="mcp tools/call"):
        client.search(search="ml platform engineer")


def test_details_parses_job_record() -> None:
    session = _FakeSession(
        {"tools/call": [json.loads(load_fixture("details_call.json"))["result"]]}
    )
    client = make_client(session)
    job = client.details("ex1mpla1c0de16ch")
    assert job["title"] == "Staff ML Platform Engineer"
    assert "ML platform lifecycle" in job["description"]
    body = session.requests[-1]
    assert body["params"]["arguments"] == {"job_id": "ex1mpla1c0de16ch"}


def test_details_missing_job_raises() -> None:
    session = _FakeSession({"tools/call": [{"content": [], "structuredContent": {}}]})
    client = make_client(session)
    with pytest.raises(mcp.MCPError, match="no job record"):
        client.details("doesnotexist1")


class TestNormalizeJob:
    """``normalize_job``: the MCP record → ``source=hiringcafe`` payload mapping."""

    def _search_record(self) -> dict[str, Any]:
        return json.loads(load_fixture("search_call.json"))["result"][
            "structuredContent"
        ]["jobs"][0]

    def _details_record(self) -> dict[str, Any]:
        return json.loads(load_fixture("details_call.json"))["result"][
            "structuredContent"
        ]["job"]

    def test_search_row_maps_normalized_fields(self) -> None:
        payload = mcp.normalize_job(self._search_record())
        assert payload["source_id"] == "hiringcafe:ex1mpla1c0de16ch"
        assert payload["title"] == "Staff ML Platform Engineer"
        assert payload["employer"] == "Example Analytics"
        assert payload["url"].startswith("https://hiringcafe.com/job/")
        assert payload["workplace_type"] == "Remote"
        assert payload["employment_types"] == ["Full Time"]
        assert payload["date_posted"] == "2026-10-01T12:00:00.000Z"
        assert payload["location"] == "United States"
        # search rows carry no description field: payload keeps it None, the
        # writer (later slice) decides intake.
        assert payload["description"] is None

    def test_salary_flattened_from_compensation(self) -> None:
        payload = mcp.normalize_job(self._search_record())
        assert payload["salary"] == {
            "currency": "USD",
            "frequency": "Yearly",
            "minimum": 180000,
            "maximum": 240000,
        }

    def test_salary_none_for_non_dict_compensation(self) -> None:
        record = self._search_record() | {"compensation": "bad"}
        assert mcp.normalize_job(record)["salary"] is None

    def test_extraction_layer_kept_whole(self) -> None:
        payload = mcp.normalize_job(self._details_record())
        extracted = payload["extracted"]
        assert extracted["requirements_summary"] == (
            "Requires 5+ years building ML platforms and data pipelines."
        )
        assert extracted["seniority_level"] == "Senior Level"
        assert extracted["minimum_years_experience"] == 5
        assert extracted["responsibilities"] == [
            "defining architecture",
            "mentoring engineers",
        ]
        assert extracted["technologies"] == ["Apache Spark", "Kubernetes"]

    def test_details_row_carries_description(self) -> None:
        payload = mcp.normalize_job(self._details_record())
        assert "ML platform lifecycle" in payload["description"]

    def test_unknown_fields_preserved_under_extra(self) -> None:
        record = self._search_record() | {
            "jobCategory": "Engineering",
            "isRemote": False,
        }
        payload = mcp.normalize_job(record)
        assert payload["extra"] == {"jobCategory": "Engineering", "isRemote": False}

    def test_known_fields_never_leak_into_extra(self) -> None:
        payload = mcp.normalize_job(self._details_record())
        assert payload["extra"] == {}

    def test_record_without_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="without id"):
            mcp.normalize_job({"title": "no id"})

    def test_payload_is_canonical_deterministic(self) -> None:
        record = self._details_record()
        first = raw.canonical_json(mcp.normalize_job(record))
        second = raw.canonical_json(mcp.normalize_job(json.loads(json.dumps(record))))
        assert first == second
        # the payload is key-mintable: accepts its hash placement unrounded
        content_hash = ids.content_sha256(first)
        key = keys.raw_payload("hiringcafe", content_hash)
        assert keys.validate_key(key) == key
        assert key.startswith("lake/raw/source=hiringcafe/sha256=")
