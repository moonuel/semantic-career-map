"""LLM-as-judge audit of the Experiment 012 label audit sheet.

Feeds every row of ``data/audit_100.jsonl`` through GLM 5.3 on Amazon Bedrock
and fills the ``audit_primary_function`` / ``audit_function_tags`` / ``audit_notes``
fields, then computes Cohen's kappa and tag micro-F1 against the labeler.

Usage:
    python scripts/012_evoc_clustering/audit_llm.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from taxonomy import TAXONOMY_DEFINITIONS, VALID_FUNCTIONS

EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
AUDIT_PATH = DATA_DIR / "audit_100.jsonl"
SCORED_PATH = DATA_DIR / "audit_100_scored.jsonl"
CACHE_PATH = DATA_DIR / ".audit_cache.json"
SUMMARY_PATH = DATA_DIR / "audit_summary.json"

REGION = "us-east-2"
MODEL_ID = "zai.glm-5"
TEMPERATURE = 0.0
MAX_TOKENS = 400
SLEEP_SECONDS = 0.1
MAX_ATTEMPTS = 3

AUDIT_SYSTEM = f"""You are an independent job-posting function auditor. Classify \
the primary function of each posting using the 14-category taxonomy below. \
Ignore seniority, location, schedule, eligibility, and salary.

{TAXONOMY_DEFINITIONS}

Assign primary_function: exactly one category from the 14 categories above.

Assign function_tags: a list of 0-3 categories selected STRICTLY from the 14 \
categories above. If primary_function is "other", function_tags must be empty.

Assign rationale: a 1-2 sentence explanation referencing specific duties.

Output ONLY a JSON object with three fields:
- primary_function: string
- function_tags: list of strings
- rationale: string

No markdown, no explanation, no code fences."""

USER_TEMPLATE = "Title: {title}\n\nCleaned posting:\n\n{text}"


def _prompt_hash() -> str:
    return hashlib.sha256(
        (AUDIT_SYSTEM + USER_TEMPLATE).encode("utf-8")
    ).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=EXPERIMENT_DIR.parents[1], text=True
        ).strip()
    except subprocess.SubprocessError:
        return "unknown"


def _client():
    return boto3.Session().client(
        "bedrock-runtime",
        region_name=REGION,
        config=Config(
            retries={"max_attempts": 5, "mode": "standard"},
            read_timeout=120,
            connect_timeout=10,
        ),
    )


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise TypeError(f"no JSON object in response: {content[:200]!r}")
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
    return {
        "primary_function": primary,
        "function_tags": tags,
        "rationale": str(parsed.get("rationale", "")).strip(),
    }


def _audit_one(
    client: Any,
    title: str,
    text: str,
) -> dict[str, Any]:
    user_message = USER_TEMPLATE.format(title=title, text=text)
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.converse(
                modelId=MODEL_ID,
                system=[{"text": AUDIT_SYSTEM}],
                messages=[{"role": "user", "content": [{"text": user_message}]}],
                inferenceConfig={
                    "temperature": TEMPERATURE,
                    "maxTokens": MAX_TOKENS,
                },
            )
            content = response["output"]["message"]["content"]
            text_out = "".join(block.get("text", "") for block in content)
            if not text_out.strip():
                raise RuntimeError("empty response")
            return _normalize(_parse_json(text_out))
        except (ClientError, BotoCoreError, KeyError, ValueError, TypeError, RuntimeError) as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(2**attempt)
    raise RuntimeError(f"audit request failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {"meta": {}, "entries": {}}


def _save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2) + "\n")
    tmp.replace(CACHE_PATH)


def cohen_kappa(model: list[str], judge: list[str]) -> float:
    labels = sorted(set(model) | set(judge))
    n = len(model)
    index = {label: i for i, label in enumerate(labels)}
    observed = [[0] * len(labels) for _ in range(len(labels))]
    for m, j in zip(model, judge, strict=True):
        observed[index[m]][index[j]] += 1
    row_sum = [sum(row) for row in observed]
    col_sum = [sum(observed[i][j] for i in range(len(labels))) for j in range(len(labels))]
    agreement = sum(observed[i][i] for i in range(len(labels))) / n
    chance = sum(row_sum[i] * col_sum[i] for i in range(len(labels))) / (n * n)
    if chance == 1.0:
        return 1.0
    return (agreement - chance) / (1 - chance)


def tag_micro_f1(model_tags: list[set[str]], judge_tags: list[set[str]]) -> float:
    tp = fp = fn = 0
    for m, j in zip(model_tags, judge_tags, strict=True):
        tp += len(m & j)
        fp += len(j - m)
        fn += len(m - j)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = _rows(AUDIT_PATH)
    if args.limit:
        rows = rows[: args.limit]

    prompt_hash = _prompt_hash()
    cache = _load_cache()
    if cache.get("meta", {}).get("prompt_hash") != prompt_hash or cache.get(
        "meta", {}
    ).get("model") != MODEL_ID:
        cache = {"meta": {"prompt_hash": prompt_hash, "model": MODEL_ID}, "entries": {}}

    client = _client()
    for index, row in enumerate(rows, start=1):
        key = row["source_hash"]
        entry = cache["entries"].get(key)
        if entry is None:
            result = _audit_one(client, row["title"], row["cleaned_text"])
            entry = {"judge": result}
            cache["entries"][key] = entry
            _save_cache(cache)
        row["audit_primary_function"] = entry["judge"]["primary_function"]
        row["audit_function_tags"] = entry["judge"]["function_tags"]
        row["audit_notes"] = entry["judge"]["rationale"]
        if index % 10 == 0 or index == len(rows):
            print(f"{index}/{len(rows)}")
        time.sleep(SLEEP_SECONDS)

    _write_rows(SCORED_PATH, rows)

    model_primary = [row["model_primary_function"] for row in rows]
    judge_primary = [row["audit_primary_function"] for row in rows]
    model_tags = [set(row["model_function_tags"]) for row in rows]
    judge_tags = [set(row["audit_function_tags"]) for row in rows]

    kappa = cohen_kappa(model_primary, judge_primary)
    micro_f1 = tag_micro_f1(model_tags, judge_tags)
    exact_match = sum(m == j for m, j in zip(model_primary, judge_primary, strict=True)) / len(rows)

    summary = {
        "experiment": "012",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "model_id": MODEL_ID,
        "region": REGION,
        "temperature": TEMPERATURE,
        "prompt_hash": prompt_hash,
        "rows": len(rows),
        "cohen_kappa": kappa,
        "tag_micro_f1": micro_f1,
        "primary_exact_match": exact_match,
        "labeler_model": "openai/gpt-5.4-nano",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"kappa={kappa:.4f} tag_micro_f1={micro_f1:.4f} exact_match={exact_match:.4f}")


if __name__ == "__main__":
    main()
