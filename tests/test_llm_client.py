"""Tests for the extracted Kilo Gateway client (task 3.1).

Retry/empty-response/missing-key semantics mirror
``scripts/008_boilerplate_only_cleaning/llm_client.py`` over
``httpx.MockTransport``; sleeps are stubbed so tests are instant.
"""

from __future__ import annotations

import json

import httpx
import pytest

from backend.pipeline.cleaning import llm_client
from backend.pipeline.cleaning.llm_client import GatewayClient

_OK_RESPONSE = {
    "choices": [{"message": {"content": "clean text", "finish_reason": "stop"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}


def _transport(
    outcomes: list[httpx.Response | Exception],
) -> tuple[httpx.MockTransport, list[dict[str, object]]]:
    """Build a MockTransport yielding the queued outcomes per attempt."""
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        outcome = outcomes[len(bodies) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return httpx.MockTransport(handler), bodies


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)


def test_complete_returns_content_and_usage() -> None:
    transport, bodies = _transport([httpx.Response(200, json=_OK_RESPONSE)])
    client = GatewayClient(api_key="k", transport=transport)
    content, usage = client.complete(
        "sys", "user", model="google/gemini-2.5-flash-lite"
    )
    assert content == "clean text"
    assert usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert len(bodies) == 1


def test_request_payload_matches_gateway_shape() -> None:
    transport, bodies = _transport([httpx.Response(200, json=_OK_RESPONSE)])
    client = GatewayClient(api_key="k", transport=transport)
    client.complete("sys", "user", model="m", max_tokens=64)
    body = bodies[0]
    assert body["model"] == "m"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 64
    assert body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1] == {"role": "user", "content": "user"}


def test_retry_sleeps_backoff_between_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[float] = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: recorded.append(s))
    transport, bodies = _transport(
        [httpx.Response(429, text="rate"), httpx.Response(200, json=_OK_RESPONSE)]
    )
    client = GatewayClient(api_key="k", transport=transport)
    content, _ = client.complete("sys", "user", model="m")
    assert content == "clean text"
    assert len(bodies) == 2
    assert recorded == [llm_client.RETRY_BACKOFF_S]


def test_transport_error_then_success_retries_once() -> None:
    transport, bodies = _transport(
        [
            httpx.ConnectError("boom"),
            httpx.Response(200, json=_OK_RESPONSE),
        ]
    )
    client = GatewayClient(api_key="k", transport=transport)
    content, _ = client.complete("sys", "user", model="m")
    assert content == "clean text"
    assert len(bodies) == 2


def test_retry_budget_exhaustion_raises_last_fault() -> None:
    transport, bodies = _transport([httpx.Response(503, text="ovr")] * 3)
    client = GatewayClient(api_key="k", transport=transport)
    with pytest.raises(httpx.HTTPStatusError, match="503"):
        client.complete("sys", "user", model="m", tries=3)
    assert len(bodies) == 3


def test_non_retryable_status_retried_without_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Original Exp 008 loop shape: a non-retryable fault still exhausts the
    # attempt budget — it only skips the backoff sleep.
    recorded: list[float] = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: recorded.append(s))
    transport, bodies = _transport([httpx.Response(400, text="bad")] * 2)
    client = GatewayClient(api_key="k", transport=transport)
    with pytest.raises(httpx.HTTPStatusError, match="400"):
        client.complete("sys", "user", model="m", tries=2)
    assert len(bodies) == 2
    assert recorded == []


def test_empty_response_retries_without_backoff_then_raises() -> None:
    transport, bodies = _transport(
        [
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "  "}}],
                    "usage": {},
                },
            )
        ]
        * 2
    )
    client = GatewayClient(api_key="k", transport=transport)
    with pytest.raises(RuntimeError, match="empty response"):
        client.complete("sys", "user", model="m")
    assert len(bodies) == 2


def test_usage_defaults_to_zero_when_gateway_omits_usage() -> None:
    transport, _ = _transport(
        [httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})]
    )
    client = GatewayClient(api_key="k", transport=transport)
    _, usage = client.complete("sys", "user", model="m")
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0}


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KILO_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="KILO_API_KEY is not set"):
        GatewayClient()


def test_missing_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KILO_API_KEY", "env-key")
    transport, _ = _transport([httpx.Response(200, json=_OK_RESPONSE)])
    client = GatewayClient(transport=transport)
    assert client.api_key == "env-key"
