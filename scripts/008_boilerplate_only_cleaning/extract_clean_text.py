"""Exp 008 8-arm boilerplate-only extraction runner with cache/resume.

Phase 1.4c — Experiment 008, Step 3. Runs every (arm, posting) pair once
through the Kilo Gateway at temperature 0.0, non-streaming, storing cleaned
text per arm in
scripts/008_boilerplate_only_cleaning/data/008_clean_outputs.json
(jobs.json is NOT mutated — unlike Exp 003's in-place clean_batch).

Arms (top-5 survivors of Exp 007 + 003-model controls + v3 prompt ablation):
    A1 nvidia/nemotron-3-nano-30b-a3b      new prompt
    A2 google/gemma-3-4b-it                new prompt
    A3 google/gemini-2.5-flash-lite        new prompt
    A4 meta-llama/llama-3.1-8b-instruct    new prompt
    A5 inclusionai/ling-3.0-flash          new prompt
    C1 openai/gpt-5.4-nano                 new prompt   (003 model, fixed prompt)
    C2 openai/gpt-5.4-nano                 old prompt   (003 baseline control)
    C3 openai/gpt-5.4-nano                 v3 prompt    (workpad-taxonomy refinement)

Cache lives in scripts/008_boilerplate_only_cleaning/data/.008_llm_cache.json
keyed per (prompt_sha256, posting_id, model); reruns hit cache and perform
zero API calls. A failed/empty arm-posting pair is recorded as a per-arm fault and
the runner continues other arms (gateway route drift must not crash the run).

Run ONE extractor process at a time — cache and outputs are whole-file
JSON writes without locking, and two concurrent runs corrupt each other.

Usage:
    python scripts/008_boilerplate_only_cleaning/extract_clean_text.py
    python scripts/008_boilerplate_only_cleaning/extract_clean_text.py --arms A1,C2 --max-postings 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import prompts
from llm_client import (
    CACHE_PATH,
    RATE_LIMIT_SECONDS,
    GatewayClient,
    build_prompt_hash,
    pricing_source,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = SCRIPT_DATA_DIR / "008_corpus.json"
OUTPUTS_PATH = SCRIPT_DATA_DIR / "008_clean_outputs.json"

ARM_ROSTER: dict[str, dict[str, str]] = {
    "A1": {
        "model": "nvidia/nemotron-3-nano-30b-a3b",
        "prompt": prompts.PROMPT_VARIANT_NEW,
    },
    "A2": {"model": "google/gemma-3-4b-it", "prompt": prompts.PROMPT_VARIANT_NEW},
    "A3": {
        "model": "google/gemini-2.5-flash-lite",
        "prompt": prompts.PROMPT_VARIANT_NEW,
    },
    "A4": {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "prompt": prompts.PROMPT_VARIANT_NEW,
    },
    "A5": {"model": "inclusionai/ling-3.0-flash", "prompt": prompts.PROMPT_VARIANT_NEW},
    "C1": {"model": "openai/gpt-5.4-nano", "prompt": prompts.PROMPT_VARIANT_NEW},
    "C2": {"model": "openai/gpt-5.4-nano", "prompt": prompts.PROMPT_VARIANT_OLD},
    "C3": {"model": "openai/gpt-5.4-nano", "prompt": prompts.PROMPT_VARIANT_V3},
}


def load_outputs() -> dict[str, Any]:
    if OUTPUTS_PATH.exists():
        outputs = json.loads(OUTPUTS_PATH.read_text())
    else:
        outputs = {"experiment": "008_boilerplate_only_cleaning", "arms": {}}
    for arm_id, spec in ARM_ROSTER.items():
        if arm_id not in outputs["arms"]:
            outputs["arms"][arm_id] = {
                "model": spec["model"],
                "prompt_variant": spec["prompt"],
                "temperature": 0.0,
                "max_tokens": 2048,
                "streaming": False,
                "outputs": {},
                "faults": {},
            }
    return outputs


def run_arm(
    arm_id: str,
    postings: list[dict],
    client: GatewayClient,
    outputs: dict[str, Any],
) -> None:
    """Run one arm across `postings`; append outputs/faults in place."""
    spec = ARM_ROSTER[arm_id]
    arm = outputs["arms"][arm_id]
    system_prompt = prompts.PROMPTS[spec["prompt"]]
    prompt_hash = build_prompt_hash(system_prompt, prompts.USER_MESSAGE_TEMPLATE)
    arm["prompt_sha256"] = prompt_hash

    total = len(postings)
    done_old = len(arm["outputs"])
    print(
        f"\n[{arm_id}] {spec['model']} + {spec['prompt']} (cached {done_old}/{total})"
    )
    for i, posting in enumerate(postings, start=1):
        posting_id = posting["posting_id"]
        user_message = prompts.USER_MESSAGE_TEMPLATE.format(
            raw_full_text=posting["raw_full_text"]
        )
        if i < total:
            time.sleep(RATE_LIMIT_SECONDS)
        try:
            content, usage, cache_hit = client.cached_call(
                prompt_hash,
                posting_id,
                spec["model"],
                system_prompt,
                user_message,
            )
        except Exception as e:  # noqa: BLE001 — every fault class logged by caller
            fault = arm["faults"].get(posting_id) or {}
            fault[time.strftime("%Y-%m-%dT%H:%M:%S%z")] = f"{type(e).__name__}: {e}"
            arm["faults"][posting_id] = fault
            print(f"  FAULT {posting_id}: {type(e).__name__}: {e}")
            continue
        arm["outputs"][posting_id] = {
            "text": content,
            "usage": usage,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        marker = "cache hit" if cache_hit else "api call"
        print(
            f"  [{i}/{total}] [{marker}] {posting_id} ({usage['completion_tokens']} out)"
        )
        with open(OUTPUTS_PATH, "w") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
    print(f"[{arm_id}] done: {len(arm['outputs'])} ok, {len(arm['faults'])} faulted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        default="A1,A2,A3,A4,A5,C1,C2,C3",
        help="Comma-separated arm ids to run (default: all 8)",
    )
    parser.add_argument(
        "--max-postings",
        type=int,
        default=None,
        help="Smoke flag: only the first N corpus postings",
    )
    args = parser.parse_args()

    arm_ids = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arm_ids if a not in ARM_ROSTER]
    if unknown:
        raise SystemExit(f"Unknown arm ids: {unknown} (valid: {', '.join(ARM_ROSTER)})")

    corpus = json.loads(CORPUS_PATH.read_text())
    postings = corpus["postings"]
    if args.max_postings is not None:
        postings = postings[: args.max_postings]

    outputs = load_outputs()
    client = GatewayClient()

    n_calls_before = (
        len(json.loads(CACHE_PATH.read_text())["entries"]) if CACHE_PATH.exists() else 0
    )

    for arm_id in arm_ids:
        run_arm(arm_id, postings, client, outputs)
        with open(OUTPUTS_PATH, "w") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    cache = json.loads(CACHE_PATH.read_text())
    print(f"\nCache entries: {n_calls_before} before → {len(cache['entries'])} now")
    print(f"Pricing source: {pricing_source()}")


if __name__ == "__main__":
    main()
