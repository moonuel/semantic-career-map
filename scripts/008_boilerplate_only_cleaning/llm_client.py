"""Shared Kilo Gateway client + append-only cache for Exp 008.

Phase 1.4c — Experiment 008. One cache serves all 7 arms plus the golden
drafting helper. Cache keys are composite:

    "<prompt_sha256>:<posting_id>:<model>"

and byte-identical inputs additionally check ``input_hash``. Unlike Exp
003's single-slot cache, a prompt-hash mismatch never invalidates anything —
a changed prompt simply writes new entries beside the old ones. Exp 003's
cache (data/.llm_clean_cache.json) is untouched; this module owns
scripts/008_boilerplate_only_cleaning/data/.008_llm_cache.json only.

Pricing for cost accounting is loaded from Exp 007's frozen gateway probe
(scripts/007_llm_latency_benchmark/data/007_config.json) with a fallback
entry for gpt-5.4-nano sourced from Exp 003's MODEL_PRICING table.

All 008 data artifacts live inside scripts/008_boilerplate_only_cleaning/
(experiment-local convention, as in Exp 007).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()  # KILO_API_KEY lives in .env at the project root (003 convention)

BASE_URL = "https://api.kilo.ai/api/gateway/chat/completions"
TEMPERATURE = 0.0
MAX_TOKENS = 2048
RATE_LIMIT_SECONDS = 1.0
REQUEST_TIMEOUT_S = 120.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
RETRY_BACKOFF_S = 2.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_PATH = SCRIPT_DATA_DIR / ".008_llm_cache.json"
PRICING_007_PATH = (
    PROJECT_ROOT / "scripts" / "007_llm_latency_benchmark" / "data" / "007_config.json"
)

# gpt-5.4-nano was the 003/004 needle, not in 007's 007_config probe table.
PRICE_GPT_5_4_NANO = (1.5e-07, 6.0e-07)  # per-token, ($/M 0.15 in, $/M 0.60 out)


def price_per_token(model: str) -> tuple[float, float]:
    """Return (prompt, completion) per-token price for a model id."""
    if model == "openai/gpt-5.4-nano":
        return PRICE_GPT_5_4_NANO
    with open(PRICING_007_PATH) as f:
        config = json.load(f)
    prices = config.get("pricing", {}).get(model)
    if not prices:
        raise KeyError(f"No frozen pricing for {model}")
    return float(prices[0]), float(prices[1])


def pricing_source() -> str:
    return (
        "scripts/007_llm_latency_benchmark/data/007_config.json gateway_probe "
        "(2026-10-17) + openai/gpt-5.4-nano from Exp 003 MODEL_PRICING"
    )


def build_prompt_hash(system_prompt: str, user_template: str) -> str:
    """SHA-256 over system prompt + user template — the cache namespace key."""
    return hashlib.sha256((system_prompt + user_template).encode("utf-8")).hexdigest()


def load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            data = json.load(f)
        data.setdefault("entries", {})
        return data
    return {"meta": {}, "entries": {}}


def save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache["meta"]["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    cache["meta"]["n_entries"] = len(cache["entries"])
    # Atomic replace — a torn or concatenated file is worse than a lost write
    # (two concurrent writers produced exactly that once; see .json.corrupt.bak).
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    tmp.replace(CACHE_PATH)


def cache_key(prompt_hash: str, posting_id: str, model: str) -> str:
    return f"{prompt_hash}:{posting_id}:{model}"


def _is_retriable(exc: Exception) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in RETRYABLE_STATUS


class GatewayClient:
    """Non-streaming chat-completion caller backed by the append-only cache."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("KILO_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "KILO_API_KEY is not set. Add it to .env or pass api_key= explicitly."
            )

    def cached_call(
        self,
        prompt_hash: str,
        posting_id: str,
        model: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int = MAX_TOKENS,
        tries: int = 2,
    ) -> tuple[str, dict[str, Any], bool]:
        """Return (cleaned_text, usage, cache_hit) for one (prompt, posting, model) call.

        Cache hits are byte-identical (same input_hash). Empty responses and
        retryable transports/5xx retry once (Exp 007 lesson: the ``empty
        response`` fault class exists even on non-streaming-friendly models).
        """
        input_hash = hashlib.sha256(user_message.encode("utf-8")).hexdigest()
        key = cache_key(prompt_hash, posting_id, model)
        cache = load_cache()
        entry = cache["entries"].get(key)
        if entry and entry.get("input_hash") == input_hash:
            return entry["output"], entry.get("usage", {}), True

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
                response = httpx.post(
                    BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=httpx.Timeout(REQUEST_TIMEOUT_S),
                )
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
                cache["entries"][key] = {
                    "input_hash": input_hash,
                    "output": content,
                    "usage": usage_record,
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                save_cache(cache)
                return content, usage_record, False
            except Exception as exc:  # noqa: BLE001 — every fault class logged by caller
                last_error = exc
                if attempt < tries and _is_retriable(exc):
                    time.sleep(RETRY_BACKOFF_S)
        raise last_error  # type: ignore[misc]
