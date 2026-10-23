"""Kilo Gateway chat-completion client (task 3.1, extracted from Exp 008).

Non-streaming completion caller with the fault classes and recovery the
experiments hardened: retryable HTTP statuses (429/500/502/503/504) and
transport errors retry with a fixed backoff; an empty LLM response raises
``RuntimeError`` on the spot and still gets retried without backoff (Exp
007 lesson: the empty-response fault class exists even on non-streaming
models, and is posting-content-dependent — every failed attempt is
re-raised verbatim for the caller to log/count).

Fidelity provenance: logic and constants verbatim from
``scripts/008_boilerplate_only_cleaning/llm_client.py`` (unchanged
experiment artifact), with the experiment-local pieces deliberately not
brought over:

- The JSON call cache: production cache resolution is lake artifact rows
  (task 3.5, design D-cache) — ``data/.llm_clean_cache.json`` is never read
  or written by production code. Runner-side pacing (per-worker jittered
  sleep, task 3.7) consumes :data:`RATE_LIMIT_SECONDS`; the client itself
  counts neither rate nor cost.
- The per-token pricing loader (Exp 007's frozen gateway-probe table):
  cost accounting flows from usage records into run manifests (task 3.6).

The CLI entry point (task 3.7) loads the project-root ``.env`` before
constructing the client (003 convention, imported library modules do not
mutate the environment).
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

BASE_URL = "https://api.kilo.ai/api/gateway/chat/completions"
TEMPERATURE = 0.0
MAX_TOKENS = 2048
RATE_LIMIT_SECONDS = 1.0  # runner-side per-worker pacing budget (task 3.7)
REQUEST_TIMEOUT_S = 120.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
RETRY_BACKOFF_S = 2.0


def _is_retriable(exc: Exception) -> bool:
    """Return whether one failed attempt is worth retrying.

    Args:
        exc: The raised exception (transport, HTTP status, or response
            shaped like an httpx error).

    Returns:
        True for transport errors and 429/5xx statuses, False otherwise —
        notably False for the client's own empty-response ``RuntimeError``,
        which retries without backoff instead.
    """
    if isinstance(exc, httpx.TransportError):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in RETRYABLE_STATUS


class GatewayClient:
    """Non-streaming Kilo Gateway chat-completion caller.

    Config-identity (``configuration_hash`` over model + prompts) and
    cache-before-call discipline live around this client (tasks 3.3/3.5);
    this class is only the wire call with exp-hardened retry semantics.
    """

    def __init__(
        self,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Construct the client.

        Args:
            api_key: Gateway bearer token; defaults to the
                ``KILO_API_KEY`` environment variable.
            transport: Optional httpx transport override (test
                injection via ``httpx.MockTransport``).

        Raises:
            RuntimeError: if no API key is available.
        """
        self.api_key = api_key or os.getenv("KILO_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "KILO_API_KEY is not set. Add it to .env or pass api_key= explicitly."
            )
        self._client = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(REQUEST_TIMEOUT_S),
        )

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str,
        max_tokens: int = MAX_TOKENS,
        tries: int = 2,
    ) -> tuple[str, dict[str, Any]]:
        """Run one (system prompt, user message) completion, retrying faults.

        Args:
            system_prompt: System prompt text.
            user_message: Filled user template (the posting text).
            model: Gateway model identifier.
            max_tokens: Completion token cap.
            tries: Total attempts (first call + retries).

        Returns:
            ``(cleaned_text, usage)`` where usage carries
            ``prompt_tokens`` / ``completion_tokens`` counts.

        Raises:
            RuntimeError: on an empty LLM response after ``tries`` attempts.
            Exception: the last fault verbatim after the attempt budget is
                exhausted (429/5xx/transport failures back off
                :data:`RETRY_BACKOFF_S` between tries; any other fault is
                re-attempted without backoff, per the original Exp 008
                loop shape).
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        last_error: Exception | None = None
        for attempt in range(1, tries + 1):
            try:
                response = self._client.post(BASE_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    raise RuntimeError("empty response")
                usage = data.get("usage", {}) or {}
                usage_record = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                }
                return content, usage_record
            except Exception as exc:  # noqa: BLE001 — every fault class re-raised for the caller
                last_error = exc
                if attempt < tries and _is_retriable(exc):
                    time.sleep(RETRY_BACKOFF_S)
        raise last_error  # type: ignore[misc]
