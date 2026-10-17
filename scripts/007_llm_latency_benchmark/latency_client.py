"""Streaming latency client for the exp-007 response-time benchmark.

Tracks, per call, the arrival timestamp of every content delta so ITL
(inter-token latency) and tail time can be reported alongside TTFT, E2E
latency, and token counts. Calls are always live — no cache — because cached
responses would corrupt the latency measurement.

Every call yields exactly one record; nothing is dropped silently. Transport
faults retry once; the exhausted retry yields an ``ok=False`` row with the
error taxonomy.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import time
from collections.abc import Sequence
from typing import Any

import httpx

BASE_URL = "https://api.kilo.ai/api/gateway/chat/completions"
TEMPERATURE = 0.0
MAX_TOKENS = 4096
REQUEST_TIMEOUT_S = 120.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
RETRY_BACKOFF_S = 2.0


def quantile(values: Sequence[float], q: float) -> float:
    """Return the linearly interpolated q-quantile of a value sequence."""
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def compute_stream_metrics(
    chunk_times_s: list[float],
    latency_s: float,
    completion_tokens: int,
) -> dict[str, Any]:
    """Derive TTFT/TPOT/ITL/tail from content-chunk arrival timestamps.

    Args:
        chunk_times_s: Arrival time of each content delta, seconds since start.
        latency_s: Total wall time to stream end.
        completion_tokens: Server-reported output token count.

    Returns:
        Metric fields with None where the denominator is unavailable.
    """
    n_chunks = len(chunk_times_s)
    itls = [later - earlier for earlier, later in itertools.pairwise(chunk_times_s)]
    decode_span = chunk_times_s[-1] - chunk_times_s[0] if n_chunks > 1 else None
    tpot = (
        decode_span / (completion_tokens - 1)
        if decode_span is not None and completion_tokens > 1
        else None
    )
    tail = latency_s - chunk_times_s[-1] if n_chunks else None
    return {
        "ttft_s": chunk_times_s[0] if n_chunks else None,
        "tpot_s": tpot,
        "itl_p50_s": quantile(itls, 0.50) if itls else None,
        "itl_p95_s": quantile(itls, 0.95) if itls else None,
        "itl_iqr_s": quantile(itls, 0.75) - quantile(itls, 0.25) if itls else None,
        "tail_s": tail,
        "content_chunks": n_chunks,
    }


def base_record(model: str, prompt_index: int, pass_index: int) -> dict[str, Any]:
    """Return the identity block shared by every raw protocol row."""
    return {
        "pass_index": pass_index,
        "prompt_index": prompt_index,
        "model": model,
        "attempt": 1,
        "ok": False,
        "error": None,
        "latency_s": None,
        "ttft_s": None,
        "tpot_s": None,
        "itl_p50_s": None,
        "itl_p95_s": None,
        "itl_iqr_s": None,
        "tail_s": None,
        "content_chunks": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "finish_reason": None,
        "usage_missing": False,
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


class LatencyClient:
    """One model's live streaming client over the Kilo Gateway."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key if api_key is not None else os.getenv("KILO_API_KEY")
        if not self.api_key:
            raise RuntimeError("KILO_API_KEY not set")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, user_prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": user_prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    def _is_retriable(self, exc: Exception) -> bool:
        """Return whether a fault is transient and worth one retry."""
        if isinstance(exc, httpx.TransportError):
            return True
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status in RETRYABLE_STATUS

    def _measure_once(self, user_prompt: str, record: dict[str, Any]) -> None:
        """Execute one streamed call and fill ``record`` in place.

        Raises:
            httpx.HTTPError: transport or HTTP fault (caller's taxonomy).
        """
        start = time.perf_counter()
        chunk_times: list[float] = []
        content_seen = False
        prompt_tokens = 0
        completion_tokens = 0
        with httpx.stream(
            "POST",
            BASE_URL,
            headers=self.headers,
            json=self._payload(user_prompt),
            timeout=httpx.Timeout(REQUEST_TIMEOUT_S),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choice = chunk.get("choices", [{}])[0]
                if choice.get("delta", {}).get("content"):
                    chunk_times.append(time.perf_counter() - start)
                    content_seen = True
                if choice.get("finish_reason"):
                    record["finish_reason"] = choice["finish_reason"]
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
        record["latency_s"] = time.perf_counter() - start
        record.update(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usage_missing=not (prompt_tokens and completion_tokens),
        )
        if chunk_times:
            record.update(
                compute_stream_metrics(
                    chunk_times, record["latency_s"], completion_tokens
                )
            )
        if not content_seen:
            record["error"] = "empty response"

    def measure(
        self, user_prompt: str, prompt_index: int, pass_index: int, tries: int = 2
    ) -> dict[str, Any]:
        """Measure one live streaming call with a single transport retry.

        Args:
            user_prompt: Prompt text.
            prompt_index: Prompt slot recorded in the raw row.
            pass_index: Repetition pass recorded in the raw row.
            tries: Total attempts across transient faults.

        Returns:
            One complete raw protocol row; ``ok=False`` rows carry ``error``.
        """
        record = base_record(self.model, prompt_index, pass_index)
        for attempt in range(1, tries + 1):
            record["attempt"] = attempt
            try:
                self._measure_once(user_prompt, record)
                record["ok"] = not record.get("error")
                return record
            except Exception as exc:  # noqa: BLE001 — every fault class logged
                record["error"] = f"{type(exc).__name__}: {exc}"
                if attempt < tries and self._is_retriable(exc):
                    time.sleep(RETRY_BACKOFF_S)
                    continue
                return record
        raise AssertionError("unreachable")
