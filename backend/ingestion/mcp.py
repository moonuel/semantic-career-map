"""Official HiringCafe MCP client: stateless JSON-RPC over Streamable HTTP.

The live fetch for the query path of ``docs/specs/hiringcafe-mcp-query-integration.md``.
The official endpoint — ``https://hiringcafe.com/api/chatgpt/mcp``, no auth,
read-only by design — answers one JSON-RPC method per POST with no session
state (verified 2026-10-14, spec Appendix B), so "connected" is a
:class:`requests.Session` for TCP/TLS reuse plus a one-time ``initialize``
handshake per run. :class:`HiringCafeClient` is the only MCP knowledge in the
codebase; callers treat its failures exactly like a board-fetch failure
(:class:`MCPError` subclasses :class:`requests.RequestException`).

Server caps are enforced locally for behavior determinism: the endpoint
silently clamps ``limit`` (verified), but the client clamps to
:data:`SEARCH_CAP` first so a run's request bodies are deterministic regardless
of server behavior changes.
"""

from __future__ import annotations

from typing import Any

import requests

OFFICIAL_ENDPOINT = "https://hiringcafe.com/api/chatgpt/mcp"
_PROTOCOL_VERSION = "2025-06-18"
_CLIENT_NAME = "semantic-career-map"

# The endpoint's documented per-call result cap (1-25); the client clamps
# to this before sending so request bodies stay deterministic.
SEARCH_CAP = 25


def normalize_job(record: dict[str, Any]) -> dict[str, Any]:
    """Map one MCP job record to the canonical ``source=hiringcafe`` payload.

    Pure and byte-deterministic: fields map to flat normalized names
    (:data:`_PAYLOAD_FIELDS`), the LLM extraction layer stays whole under
    ``extracted`` (:data:`_PAYLOAD_EXTRACTED_FIELDS`), unrecognized structured
    fields are preserved under ``extra``. Together with the full plain-text
    ``description``, the payload is exactly spec F3/N2 — what the source gave,
    verbatim-where-it-shifts, no observation metadata mixed in. Identity is
    content-addressed downstream, so this dict must hash identically across
    runs for identical server records.

    Args:
        record: One ``structuredContent`` job row (search result) or
            ``structuredContent.job`` record (details result) verbatim from
            the MCP response. A details record is a search row plus
            ``description``; either is accepted.

    Returns:
        The payload dict to canonicalize with ``raw.canonical_json``.

    Raises:
        ValueError: if ``record`` lacks the job id token.
    """
    job_id = record.get("id")
    if not job_id:
        raise ValueError(f"mcp job record without id: {job_id!r}")
    payload: dict[str, Any] = {"source_id": f"hiringcafe:{job_id}"}
    mapped = _PAYLOAD_FIELDS | _PAYLOAD_EXTRACTED_FIELDS
    for field, name in _PAYLOAD_FIELDS.items():
        payload[name] = record.get(field)
    salary = record.get("compensation")
    payload["salary"] = dict(salary) if isinstance(salary, dict) and salary else None
    extracted = {
        name: record.get(field) for field, name in _PAYLOAD_EXTRACTED_FIELDS.items()
    }
    extra = {
        key: value
        for key, value in record.items()
        if key not in mapped and key not in ("description", "compensation", "id")
    }
    payload["description"] = record.get("description")
    payload["extracted"] = extracted
    payload["extra"] = extra
    return payload


# MCP record field -> payload field, for the lakes normalized posting fields.
_PAYLOAD_FIELDS: dict[str, str] = {
    "applyUrl": "url",
    "title": "title",
    "company": "employer",
    "location": "location",
    "workplaceType": "workplace_type",
    "employmentTypes": "employment_types",
    "datePosted": "date_posted",
}

# MCP record field -> payload field, for the LLM extraction layer (kept whole
# under ``extracted``): this is the schema-benchmark material (spec F3/N2).
_PAYLOAD_EXTRACTED_FIELDS: dict[str, str] = {
    "requirementsSummary": "requirements_summary",
    "seniorityLevel": "seniority_level",
    "minimumYearsExperience": "minimum_years_experience",
    "responsibilities": "responsibilities",
    "technologies": "technologies",
}


class MCPError(requests.RequestException):
    """One MCP transport or protocol failure, typed like a fetch failure.

    Raised for non-2xx transport outcomes, JSON-RPC error bodies, and
    ``isError`` tool results alike, so callers can catch it alongside
    :class:`requests.RequestException` (the ``_scrape_source`` pattern).
    """


class HiringCafeClient:
    """A stateless MCP client: one POST per JSON-RPC method, one session.

    Args:
        endpoint: The MCP endpoint URL; default is the official one.
        timeout: Per-POST timeout in seconds.
    """

    def __init__(
        self, endpoint: str = OFFICIAL_ENDPOINT, timeout: float = 30.0
    ) -> None:
        self._endpoint = endpoint
        self._timeout = timeout
        self._http = requests.Session()
        self._http.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": _PROTOCOL_VERSION,
            }
        )
        self._next_id = 0

    def initialize(self) -> dict[str, Any]:
        """Run the one-time MCP handshake; one POST.

        The server does not require it (stateless, verified), but the
        protocol-correct call keeps the client honest about server
        capabilities and version drift.

        Returns:
            The ``initialize`` result object (server info, capabilities).

        Raises:
            MCPError: if the handshake fails at transport or protocol level.
        """
        return self._call(
            "initialize",
            params={
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": _CLIENT_NAME, "version": "0.1"},
            },
        )

    def search(self, **arguments: Any) -> list[dict[str, Any]]:
        """Search live postings via ``search_hiringcafe_jobs``; one POST.

        Keyword arguments are the tool's input schema fields verbatim
        (``search``, ``location``, ``posted_salary_only``, ...). ``limit`` is
        clamped to :data:`SEARCH_CAP` before sending.

        Args:
            **arguments: Tool input-schema fields.

        Returns:
            The ``structuredContent`` job rows (possibly empty).

        Raises:
            MCPError: if the call fails at transport or protocol level.
        """
        if "limit" in arguments and isinstance(arguments["limit"], int):
            arguments = {**arguments, "limit": min(arguments["limit"], SEARCH_CAP)}
        result = self._tool_call("search_hiringcafe_jobs", arguments)
        structured = result.get("structuredContent") or {}
        jobs = structured.get("jobs", []) if isinstance(structured, dict) else []
        return [job for job in jobs if isinstance(job, dict)]

    def details(self, job_id: str) -> dict[str, Any]:
        """Fetch one posting's full details via ``get_hiringcafe_job_details``.

        Args:
            job_id: The job id token from search results (the 16-char
                URL-stable slug token).

        Returns:
            The ``structuredContent`` job record, including the full
            plain-text ``description`` field.

        Raises:
            MCPError: if the call fails or no job record comes back.
        """
        result = self._tool_call("get_hiringcafe_job_details", {"job_id": job_id})
        structured = result.get("structuredContent") or {}
        job = structured.get("job") if isinstance(structured, dict) else None
        if not isinstance(job, dict):
            raise MCPError("mcp get_hiringcafe_job_details: no job record")
        return job

    def _tool_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """One ``tools/call`` POST; returns the tool result object."""
        return self._call("tools/call", params={"name": name, "arguments": arguments})

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """One JSON-RPC POST; returns the result object or raises ``MCPError``."""
        self._next_id += 1
        body = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": params,
        }
        try:
            response = self._http.post(self._endpoint, json=body, timeout=self._timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise MCPError(f"mcp {method}: {exc}") from exc
        if not isinstance(payload, dict):
            raise MCPError(f"mcp {method}: non-object response body")
        error = payload.get("error")
        if isinstance(error, dict):
            raise MCPError(f"mcp {method}: {error.get('message') or error}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise MCPError(f"mcp {method}: no result object in response")
        if result.get("isError"):
            text = " ".join(
                block.get("text", "")
                for block in result.get("content", [])
                if isinstance(block, dict)
            )
            raise MCPError(f"mcp {method}: tool error: {text[:200]}")
        return result
