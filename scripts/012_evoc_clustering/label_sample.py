"""Generate cached weak function labels for Experiment 012.

Usage:
    python scripts/012_evoc_clustering/label_sample.py
    python scripts/012_evoc_clustering/label_sample.py --limit 5
    python scripts/012_evoc_clustering/label_sample.py --force --audit-size 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from taxonomy import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE, VALID_FUNCTIONS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "011_encoder_coverage"
    / "data"
    / "sample_1000.jsonl"
)
LABELS_PATH = DATA_DIR / "labels_primary_function.jsonl"
MANIFEST_PATH = DATA_DIR / "label_manifest.json"
AUDIT_PATH = DATA_DIR / "audit_100.jsonl"

BASE_URL = "https://api.kilo.ai/api/gateway/chat/completions"
MODEL = "openai/gpt-5.4-nano"
TEMPERATURE = 0.0
MAX_TOKENS = 400
SLEEP_SECONDS = 0.2
MAX_ATTEMPTS = 3


def _prompt_hash() -> str:
    return hashlib.sha256(
        (SYSTEM_PROMPT + USER_MESSAGE_TEMPLATE).encode("utf-8")
    ).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except subprocess.SubprocessError:
        return "unknown"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )
    tmp.replace(path)


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in response: {content[:200]!r}")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise TypeError("response JSON is not an object")
    return parsed


def _normalize(parsed: dict[str, Any]) -> dict[str, Any]:
    primary = str(parsed.get("primary_function", "")).strip()
    if primary not in VALID_FUNCTIONS:
        raise ValueError(f"invalid primary_function: {primary!r}")
    raw_tags = parsed.get("function_tags", [])
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    if not isinstance(raw_tags, list):
        raw_tags = []
    tags = []
    for tag in raw_tags:
        tag = str(tag).strip()
        if tag in VALID_FUNCTIONS and tag != "other" and tag not in tags:
            tags.append(tag)
    if primary == "other":
        tags = []
    elif primary not in tags:
        tags = [primary, *tags]
    tags = tags[:3]
    confidence = str(parsed.get("label_confidence", "low")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {
        "primary_function": primary,
        "function_tags": tags,
        "label_confidence": confidence,
        "rationale": str(parsed.get("rationale", "")).strip(),
    }


def _request(
    client: httpx.Client,
    api_key: str,
    title: str,
    text: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    user_message = USER_MESSAGE_TEMPLATE.format(title=title, text=text)
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.post(
                BASE_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not content.strip():
                raise RuntimeError("empty response")
            parsed = _normalize(_parse_json(content))
            usage = data.get("usage", {})
            return parsed, {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            }
        except (httpx.HTTPError, KeyError, ValueError, RuntimeError, TypeError) as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(2**attempt)
    raise RuntimeError(f"label request failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _label_one(
    api_key: str,
    source: dict[str, Any],
    prompt_hash: str,
) -> dict[str, Any]:
    with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
        parsed, usage = _request(
            client,
            api_key,
            source.get("title", ""),
            source.get("cleaned_text", ""),
        )
    time.sleep(SLEEP_SECONDS)
    return {
        "source_hash": source["source_hash"],
        "title": source.get("title", ""),
        **parsed,
        "model": MODEL,
        "prompt_hash": prompt_hash,
        "usage": usage,
        "labeled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _build_audit(
    sample_by_hash: dict[str, dict[str, Any]],
    labels: list[dict[str, Any]],
    audit_size: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_function: dict[str, list[dict[str, Any]]] = {}
    for row in labels:
        by_function.setdefault(row["primary_function"], []).append(row)
    selected: list[dict[str, Any]] = []
    for function in sorted(by_function):
        pool = sorted(by_function[function], key=lambda row: row["source_hash"])
        rng.shuffle(pool)
        selected.extend(pool[:8])
    selected_hashes = {row["source_hash"] for row in selected}
    remaining = [
        row for row in labels if row["source_hash"] not in selected_hashes
    ]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, audit_size - len(selected))])
    if len(selected) > audit_size:
        rng.shuffle(selected)
        selected = selected[:audit_size]
    audit = []
    for row in selected:
        sample = sample_by_hash[row["source_hash"]]
        audit.append(
            {
                "source_hash": row["source_hash"],
                "title": sample.get("title", ""),
                "cleaned_text": sample.get("cleaned_text", ""),
                "model_primary_function": row["primary_function"],
                "model_function_tags": row["function_tags"],
                "audit_primary_function": "",
                "audit_function_tags": [],
                "audit_notes": "",
            }
        )
    audit.sort(key=lambda row: row["source_hash"])
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="label at most N rows")
    parser.add_argument("--force", action="store_true", help="ignore existing labels")
    parser.add_argument("--audit-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("KILO_API_KEY")
    if not api_key:
        raise SystemExit("KILO_API_KEY is not set in the environment or .env")

    sample = _load_rows(SAMPLE_PATH)
    if len(sample) != 1000:
        raise SystemExit(f"expected 1000 sample rows, found {len(sample)}")
    sample_by_hash = {row["source_hash"]: row for row in sample}
    if len(sample_by_hash) != len(sample):
        raise SystemExit("sample contains duplicate source_hash values")

    prompt_hash = _prompt_hash()
    existing: dict[str, dict[str, Any]] = {}
    if LABELS_PATH.exists() and not args.force:
        for row in _load_rows(LABELS_PATH):
            if row.get("prompt_hash") == prompt_hash and row.get("model") == MODEL:
                existing[row["source_hash"]] = row

    todo = [row for row in sample if row["source_hash"] not in existing]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("labels already complete")
    else:
        print(f"labeling {len(todo)} of {len(sample)} postings with {args.workers} workers")
        rows = list(existing.values())
        completed = 0
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(_label_one, api_key, source, prompt_hash): source
                for source in todo
            }
            for future in as_completed(futures):
                source = futures[future]
                try:
                    row = future.result()
                except (httpx.HTTPError, KeyError, ValueError, RuntimeError, TypeError) as exc:
                    errors.append(f"{source['source_hash']}: {exc}")
                    print(f"  ERROR {source['source_hash']}: {exc}")
                    continue
                rows.append(row)
                existing[row["source_hash"]] = row
                completed += 1
                if completed % 25 == 0 or completed == len(todo):
                    _write_rows(LABELS_PATH, sorted(rows, key=lambda r: r["source_hash"]))
                    print(f"  {completed}/{len(todo)}")
        rows = sorted(existing.values(), key=lambda row: row["source_hash"])
        _write_rows(LABELS_PATH, rows)
        if errors:
            print(f"label errors: {len(errors)}")

    labels = _load_rows(LABELS_PATH) if LABELS_PATH.exists() else []
    by_hash = {row["source_hash"]: row for row in labels}
    missing = [key for key in sample_by_hash if key not in by_hash]
    if missing:
        print(f"incomplete labels: {len(missing)} missing")
    else:
        print("labels complete: 1000/1000")

    manifest = {
        "experiment": "012",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "sample": str(SAMPLE_PATH.relative_to(PROJECT_ROOT)),
        "labels": str(LABELS_PATH.relative_to(PROJECT_ROOT)),
        "model": MODEL,
        "temperature": TEMPERATURE,
        "prompt_hash": prompt_hash,
        "rows": len(labels),
        "missing": len(missing),
        "complete": not missing,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    if not missing and args.audit_size > 0 and (
        args.force or not AUDIT_PATH.exists()
    ):
        audit = _build_audit(sample_by_hash, labels, args.audit_size)
        _write_rows(AUDIT_PATH, audit)
        print(f"wrote audit sheet: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
