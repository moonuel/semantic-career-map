"""Run the exp-007 response-time benchmark: interleaved, paced, resumable.

Protocol: one fixed recitation prompt (the first five paragraphs of Lorem
Ipsum) sent to the expanded candidate roster over the Kilo Gateway. Prompt
slots x passes execute in a seeded order with the model list rotated
round-robin per slot so provider drift is spread evenly across candidates.
Every call is appended to the raw JSONL as it completes (interrupt-safe) and
``(pass_index, prompt_index, model)`` keys make reruns idempotent.

Data products land beside the code: ``data/007_config.json`` and
``data/007_raw_results.jsonl``.

Usage:
    python scripts/007_llm_latency_benchmark/run_benchmark.py \
        [--passes 3] [--pace 1.0] [--seed 8008] \
        [--models meta-llama/llama-3.1-8b-instruct ...]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from latency_client import LatencyClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
CONFIG_PATH = DATA_DIR / "007_config.json"
RAW_PATH = DATA_DIR / "007_raw_results.jsonl"

DEFAULT_PASSES = 3
DEFAULT_PACE_S = 1.0
DEFAULT_SEED = 8008

LOREM_PROMPT = (
    "Recite the first five paragraphs of Lorem Ipsum. Reproduce the "
    "standard Lorem Ipsum text verbatim."
)

# Expanded roster of cheap, lightweight instruct models, probed 2026-10-16.
DEFAULT_ROSTER = [
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemma-3-4b-it",
    "mistralai/mistral-small-24b-instruct-2501",
    "nex-agi/nex-n2.5-mini",
    "ibm-granite/granite-4.0-h-micro",
    "mistralai/ministral-3b-2512",
    "microsoft/phi-4",
    "qwen/qwen3.7-flash",
    "google/gemma-3-12b-it",
    "amazon/nova-micro-v1",
    "cohere/command-r7b-12-2024",
    "google/gemma-3-27b-it",
    "inclusionai/ling-3.0-flash",
    "deepseek/deepseek-v4-flash",
    "nvidia/nemotron-3-nano-30b-a3b",
    "mistralai/ministral-8b-2512",
    "mistralai/mistral-nemo",
    "tencent/hy-mt2-7b",
    "google/gemini-2.5-flash-lite",
    "openai/gpt-5-nano",
]


def probe_pricing(api_key: str) -> tuple[dict[str, tuple[float, float]], str]:
    """Probe per-token pricing from the Gateway catalog endpoint.

    Args:
        api_key: Kilo Gateway bearer token.

    Returns:
        (pricing map slug -> (price_in, price_out) per token, source label).
        Missing models or a failed probe leave those models priced at zero
        and the source label reflects that, so cost columns stay auditable.
    """
    try:
        response = httpx.get(
            "https://api.kilo.ai/api/gateway/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        response.raise_for_status()
        probed: dict[str, tuple[float, float]] = {}
        for entry in response.json().get("data", []):
            pricing = entry.get("pricing", {})
            try:
                probed[str(entry["id"])] = (
                    float(pricing["prompt"]),
                    float(pricing["completion"]),
                )
            except KeyError, TypeError, ValueError:
                continue
        return probed, "gateway_probe"
    except Exception as exc:  # noqa: BLE001 — fallback must catch any probe fault
        print(f"pricing probe failed ({exc}); cost columns will be zero")
        return {}, "probe_failed"


def git_commit() -> str:
    """Return the repo commit hash recorded in the config header."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — any git fault degrades to "unknown"
        return "unknown"


def rotation(models: list[str], offset: int) -> list[str]:
    """Return the model list rotated by ``offset`` (round-robin schedule)."""
    cut = offset % len(models)
    return models[cut:] + models[:cut]


def load_raw_keys(path: Path) -> set[tuple[int, int, str]]:
    """Return already-measured (pass, prompt slot, model) keys for resumability."""
    keys: set[tuple[int, int, str]] = set()
    if not path.exists():
        return keys
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((row["pass_index"], row["prompt_index"], row["model"]))
    return keys


def run_benchmark(
    models: list[str],
    passes: int,
    pace_s: float,
    seed: int,
    raw_path: Path,
) -> list[dict[str, Any]]:
    """Execute the interleaved timing protocol; append new rows only.

    Returns:
        The rows appended in this invocation (already persisted in
        ``raw_path``).
    """
    slots = [(pass_index, 0) for pass_index in range(passes)]
    planned_total = len(slots) * len(models)
    done_keys = load_raw_keys(raw_path)
    clients = {model: LatencyClient(model) for model in models}
    appended: list[dict[str, Any]] = []
    counter = 0
    random.Random(seed).shuffle(slots)
    for pass_index, prompt_index in slots:
        for model in rotation(models, counter):
            counter += 1
            key = (pass_index, prompt_index, model)
            if key in done_keys:
                continue
            print(
                f"[{len(appended) + 1}/{planned_total}] "
                f"p{pass_index + 1} slot{prompt_index} -> {model}",
                flush=True,
            )
            record = clients[model].measure(LOREM_PROMPT, prompt_index, pass_index)
            record["rotation_offset"] = counter
            appended.append(record)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["ok"]:
                print(
                    f"    ttft={record['ttft_s']:.2f}s "
                    f"e2e={record['latency_s']:.2f}s "
                    f"tok={record['prompt_tokens']}+{record['completion_tokens']}"
                )
            else:
                print(f"    FAULT: {record['error']}")
            time.sleep(pace_s)
    return appended


def write_config(args: argparse.Namespace, models: list[str]) -> dict[str, Any]:
    """Freeze the run config header (pricing, commit) to disk."""
    api_key = os.environ.get("KILO_API_KEY", "")
    pricing, pricing_source = probe_pricing(api_key)
    config = {
        "experiment": "007_llm_latency_benchmark",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": git_commit(),
        "models": models,
        "passes": args.passes,
        "pace_s": args.pace,
        "seed": args.seed,
        "prompt": LOREM_PROMPT,
        "pricing": {m: list(pricing[m]) for m in models if m in pricing},
        "pricing_source": pricing_source,
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    return config


def config_summary(config: dict[str, Any]) -> str:
    """Return the one-line run header describing the frozen config."""
    return (
        f"exp007: models={len(config['models'])} passes={config['passes']} "
        f"pace={config['pace_s']}s seed={config['seed']} "
        f"pricing={config['pricing_source']}"
    )


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return parsed rows from a JSONL file (empty when absent)."""
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def quick_summary(raw_path: Path) -> None:
    """Print a coarse per-model median table (full stats: analyze_results)."""
    buckets: dict[str, dict[str, list[float]]] = {}
    for row in iter_jsonl(raw_path):
        if not row.get("ok"):
            continue
        bucket = buckets.setdefault(row["model"], {"ttft": [], "e2e": [], "tpot": []})
        if row.get("ttft_s") is not None:
            bucket["ttft"].append(row["ttft_s"])
        if row.get("latency_s") is not None:
            bucket["e2e"].append(row["latency_s"])
        if row.get("tpot_s") is not None:
            bucket["tpot"].append(row["tpot_s"])

    header = f"{'model':<44}{'n':>5}{'ttft_med':>10}{'e2e_med':>10}{'tpot_med':>10}"
    print(f"\n{header}\n{'-' * len(header)}")
    for model, bucket in sorted(buckets.items()):
        cells = "".join(
            f"{st.median(vals):>10.3f}" if vals else f"{'-':>10}"
            for vals in (bucket["ttft"], bucket["e2e"], bucket["tpot"])
        )
        print(f"{model:<44}{len(bucket['ttft']):>5}{cells}")


def main() -> None:
    """CLI entry point: config freeze, then the interleaved timing run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_ROSTER)
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES)
    parser.add_argument("--pace", type=float, default=DEFAULT_PACE_S)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force", action="store_true", help="overwrite a mismatching frozen config"
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    if CONFIG_PATH.exists() and not args.force:
        with open(CONFIG_PATH) as f:
            frozen = json.load(f)
        if frozen.get("models") != args.models or frozen.get("passes") != args.passes:
            sys.exit("frozen config mismatch (models/passes); use --force to re-freeze")
    config = write_config(args, args.models)
    print(config_summary(config))
    appended = run_benchmark(args.models, args.passes, args.pace, args.seed, RAW_PATH)
    print(f"timing: appended {len(appended)} rows to {RAW_PATH}")
    quick_summary(RAW_PATH)


if __name__ == "__main__":
    main()
