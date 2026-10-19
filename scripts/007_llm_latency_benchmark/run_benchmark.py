"""Run the exp-007 two-track response-time benchmark: interleaved, paced, resumable.

Two tracks share one protocol over the expanded candidate roster:

- Track A (``ttft``): streaming micro-prompt ``"Say hello world."`` — clean
  time-to-first-token, decode negligible (~2 output tokens).
- Track B (``throughput``): non-streaming copy task — the first five
  paragraphs of Lorem Ipsum are embedded verbatim in the prompt and the model
  must repeat them, so output length is a constant and E2E latency and
  tokens/s compare directly across models.

Each track's prompt slots x passes execute in a seeded order with the model
list rotated round-robin per slot so provider drift is spread evenly across
candidates. Every call is appended to the raw JSONL as it completes
(interrupt-safe) and ``(track, pass_index, prompt_index, model)`` keys make
reruns idempotent.

Data products land beside the code: ``data/007_config.json`` and
``data/007_raw_results.jsonl``.

Usage:
    python scripts/007_llm_latency_benchmark/run_benchmark.py \
        [--track all] [--ttft-passes 10] [--throughput-passes 3] \
        [--pace 1.0] [--seed 8008] \
        [--models meta-llama/llama-3.1-8b-instruct ...]
"""

from __future__ import annotations

import argparse
import hashlib
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
from latency_client import MAX_TOKENS, LatencyClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
CONFIG_PATH = DATA_DIR / "007_config.json"
RAW_PATH = DATA_DIR / "007_raw_results.jsonl"

DEFAULT_PACE_S = 1.0
DEFAULT_SEED = 8008

# Track A — streaming micro-prompt: TTFT floor, decode cost bounded by 32
# tokens so a runaway reply cannot pollute passes (content detection works
# with any delta, truncation safe).
TTFT_PROMPT = "Say hello world."
TTFT_MAX_TOKENS = 32

# Track B — non-streaming copy task. Reference text supplied by the user,
# five paragraphs verbatim, separated by blank lines. Output length is then
# a constant across models, killing the length confound of round 1.
LOREM_REFERENCE = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis lobortis "
    "nunc sed fermentum ullamcorper. Lorem ipsum dolor sit amet, consectetur "
    "adipiscing elit. Maecenas convallis finibus vehicula. Maecenas sapien "
    "leo, consequat dapibus diam vitae, accumsan dignissim nisl. Fusce "
    "tristique, massa vel feugiat condimentum, velit mauris varius tortor, "
    "non viverra justo felis ac dolor. Interdum et malesuada fames ac ante "
    "ipsum primis in faucibus. Cras efficitur tempor interdum. Curabitur nec "
    "scelerisque neque.\n"
    "\n"
    "Class aptent taciti sociosqu ad litora torquent per conubia nostra, per "
    "inceptos himenaeos. Donec pretium dolor eros, sed ullamcorper dui "
    "elementum nec. Duis in mattis velit. Pellentesque iaculis elit dolor, "
    "sit amet porttitor ex facilisis eget. Nunc vehicula venenatis ligula "
    "sed lobortis. Morbi dictum diam non mauris suscipit scelerisque. Ut "
    "imperdiet suscipit ipsum, eu porttitor ipsum. Quisque accumsan justo "
    "sed euismod sagittis. Nullam tincidunt nec nisl in tristique. Sed "
    "semper diam ac enim gravida, eget pharetra elit rhoncus. Curabitur "
    "quis risus neque. Phasellus nec quam lorem. Nullam a nulla eget nisl "
    "eleifend iaculis nec pulvinar ex.\n"
    "\n"
    "Mauris efficitur, tellus vitae ultrices bibendum, ante neque lacinia "
    "purus, a scelerisque est sapien a mi. Etiam molestie accumsan tortor, "
    "at ornare nisl ultricies non. In volutpat neque id ipsum fringilla "
    "ultricies. Vivamus nec magna sodales, posuere elit eu, rhoncus tortor. "
    "Etiam tincidunt, erat ac pharetra varius, ipsum justo tincidunt arcu, "
    "placerat tempus eros nisi tincidunt nulla. Donec placerat luctus quam "
    "quis eleifend. Mauris hendrerit eget purus in aliquet. Cras vitae "
    "ultrices tellus. Nunc consectetur venenatis dictum. Nullam venenatis "
    "turpis a nisl faucibus, sed euismod ipsum imperdiet.\n"
    "\n"
    "Quisque pretium risus sit amet accumsan pulvinar. Cras aliquam "
    "fermentum tristique. Quisque a est sit amet ante dictum volutpat. "
    "Nullam luctus nunc et turpis vehicula mollis. Sed pellentesque "
    "tincidunt mi sit amet volutpat. Suspendisse potenti. Fusce lacinia in "
    "lacus vitae vehicula.\n"
    "\n"
    "In hac habitasse platea dictumst. Duis non sapien porta, semper tellus "
    "ac, lobortis eros. Nullam ullamcorper at augue vitae elementum. Mauris "
    "lobortis commodo vulputate. Sed turpis ligula, elementum vel pretium "
    "sit amet, luctus vitae risus. Sed ut libero tincidunt, ullamcorper "
    "lorem ut, mollis felis. Suspendisse facilisis dui ac ligula porta "
    "maximus. Curabitur euismod consectetur dui, ac rhoncus ante mattis ut. "
    "Ut sit amet arcu quis ipsum congue scelerisque. Quisque pretium erat "
    "magna, ut accumsan arcu convallis et. Etiam tellus neque, aliquam id "
    "pulvinar non, pharetra vitae urna. Integer dictum enim eget nibh "
    "pharetra, ac facilisis mauris pulvinar. Donec eget lectus in nisl "
    "semper pellentesque."
)

THROUGHPUT_PROMPT_TEMPLATE = (
    "Repeat the following 5 paragraphs of Lorem Ipsum:\n\n{lorem_ipsum}"
)
THROUGHPUT_PROMPT = THROUGHPUT_PROMPT_TEMPLATE.format(lorem_ipsum=LOREM_REFERENCE)
THROUGHPUT_MAX_TOKENS = MAX_TOKENS

# Track registry. ttft runs first: its calls are cheap and short, so a
# partial failure of the expensive throughput track still leaves its data.
TRACKS: dict[str, dict[str, Any]] = {
    "ttft": {
        "prompt": TTFT_PROMPT,
        "max_tokens": TTFT_MAX_TOKENS,
        "stream": True,
        "default_passes": 10,
    },
    "throughput": {
        "prompt": THROUGHPUT_PROMPT,
        "max_tokens": THROUGHPUT_MAX_TOKENS,
        "stream": False,
        "default_passes": 3,
    },
}

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


def load_raw_keys(path: Path) -> set[tuple[str, int, int, str]]:
    """Return already-measured (track, pass, prompt slot, model) keys.

    Pre-two-track rows have no ``track`` field and are keyed as
    ``("legacy", ...)`` so their old keys can never collide with a new
    run's 4-tuples (they are preserved on disk, never re-executed).
    """
    keys: set[tuple[str, int, int, str]] = set()
    if not path.exists():
        return keys
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add(
                (
                    str(row.get("track", "legacy")),
                    row["pass_index"],
                    row["prompt_index"],
                    row["model"],
                )
            )
    return keys


def run_track(
    models: list[str],
    track: str,
    passes: int,
    pace_s: float,
    seed: int,
    raw_path: Path,
) -> list[dict[str, Any]]:
    """Execute one track's interleaved timing protocol; append new rows only.

    Args:
        models: Candidate model slugs.
        track: Track name, a key of ``TRACKS``.
        passes: Repetitions of the single prompt slot.
        pace_s: Seconds slept between calls (1 req/s pacing).
        seed: Shuffle seed for the execution-slot order.
        raw_path: Raw JSONL target, appended-to per call.

    Returns:
        The rows appended in this invocation (already persisted in
        ``raw_path``).
    """
    spec = TRACKS[track]
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
            key = (track, pass_index, prompt_index, model)
            if key in done_keys:
                continue
            print(
                f"[{len(appended) + 1}/{planned_total}] {track} "
                f"p{pass_index + 1} slot{prompt_index} -> {model}",
                flush=True,
            )
            if spec["stream"]:
                record = clients[model].measure(
                    spec["prompt"],
                    prompt_index,
                    pass_index,
                    max_tokens=spec["max_tokens"],
                )
            else:
                record = clients[model].measure_nonstream(
                    spec["prompt"],
                    prompt_index,
                    pass_index,
                    max_tokens=spec["max_tokens"],
                )
            record["track"] = track
            record["rotation_offset"] = counter
            appended.append(record)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["ok"]:
                if spec["stream"]:
                    print(
                        f"    ttft={record['ttft_s']:.2f}s "
                        f"e2e={record['latency_s']:.2f}s "
                        f"tok={record['prompt_tokens']}+{record['completion_tokens']}"
                    )
                else:
                    print(
                        f"    e2e={record['latency_s']:.2f}s "
                        f"tok={record['prompt_tokens']}+{record['completion_tokens']} "
                        f"finish={record['finish_reason']}"
                    )
            else:
                print(f"    FAULT: {record['error']}")
            time.sleep(pace_s)
    return appended


def run_benchmark(
    models: list[str],
    tracks: list[str],
    passes_by_track: dict[str, int],
    pace_s: float,
    seed: int,
    raw_path: Path,
) -> list[dict[str, Any]]:
    """Execute the selected tracks sequentially (ttft before throughput).

    Returns:
        The rows appended in this invocation (already persisted in
        ``raw_path``).
    """
    appended: list[dict[str, Any]] = []
    for track in tracks:
        appended.extend(
            run_track(models, track, passes_by_track[track], pace_s, seed, raw_path)
        )
    return appended


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of a fully formatted prompt text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_config(args: argparse.Namespace, models: list[str]) -> dict[str, Any]:
    """Freeze the run config header (prompts, pricing, commit) to disk."""
    api_key = os.environ.get("KILO_API_KEY", "")
    pricing, pricing_source = probe_pricing(api_key)
    config = {
        "experiment": "007_llm_latency_benchmark",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": git_commit(),
        "models": models,
        "ttft_passes": args.ttft_passes,
        "throughput_passes": args.throughput_passes,
        "pace_s": args.pace,
        "seed": args.seed,
        "ttft_prompt_sha256": sha256_text(TTFT_PROMPT),
        "throughput_prompt_sha256": sha256_text(THROUGHPUT_PROMPT),
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
        f"exp007: models={len(config['models'])} "
        f"ttft_passes={config['ttft_passes']} "
        f"throughput_passes={config['throughput_passes']} "
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
    """Print a coarse per-track, per-model median table (full: analyze_results)."""
    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in iter_jsonl(raw_path):
        if not row.get("ok"):
            continue
        bucket = buckets.setdefault(
            (str(row.get("track", "legacy")), str(row["model"])),
            {"ttft": [], "e2e": [], "tpot": []},
        )
        if row.get("ttft_s") is not None:
            bucket["ttft"].append(row["ttft_s"])
        if row.get("latency_s") is not None:
            bucket["e2e"].append(row["latency_s"])
        if row.get("tpot_s") is not None:
            bucket["tpot"].append(row["tpot_s"])

    header = (
        f"{'model':<44}{'track':<12}{'n':>5}"
        f"{'ttft_med':>10}{'e2e_med':>10}{'tpot_med':>10}"
    )
    print(f"\n{header}\n{'-' * len(header)}")
    for (track, model), bucket in sorted(buckets.items()):
        cells = "".join(
            f"{st.median(vals):>10.3f}" if vals else f"{'-':>10}"
            for vals in (bucket["ttft"], bucket["e2e"], bucket["tpot"])
        )
        print(f"{model:<44}{track:<12}{len(bucket['ttft']):>5}{cells}")


def main() -> None:
    """CLI entry point: config freeze, then both tracks in order (A then B)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_ROSTER)
    parser.add_argument(
        "--track",
        choices=("ttft", "throughput", "all"),
        default="all",
        help="run one track or both (default: all)",
    )
    parser.add_argument(
        "--ttft-passes", type=int, default=TRACKS["ttft"]["default_passes"]
    )
    parser.add_argument(
        "--throughput-passes",
        type=int,
        default=TRACKS["throughput"]["default_passes"],
    )
    parser.add_argument("--pace", type=float, default=DEFAULT_PACE_S)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force", action="store_true", help="overwrite a mismatching frozen config"
    )
    args = parser.parse_args()

    passes_by_track = {"ttft": args.ttft_passes, "throughput": args.throughput_passes}
    tracks = list(TRACKS) if args.track == "all" else [args.track]

    load_dotenv(PROJECT_ROOT / ".env")
    if CONFIG_PATH.exists() and not args.force:
        with open(CONFIG_PATH) as f:
            frozen = json.load(f)
        current: dict[str, Any] = {
            "models": args.models,
            "ttft_passes": args.ttft_passes,
            "throughput_passes": args.throughput_passes,
            "pace_s": args.pace,
            "seed": args.seed,
        }
        mismatched = [f for f, v in current.items() if frozen.get(f) != v]
        if mismatched:
            sys.exit(
                "frozen config mismatch ("
                + ", ".join(mismatched)
                + "); use --force to re-freeze"
            )
    config = write_config(args, args.models)
    print(config_summary(config))
    appended = run_benchmark(
        args.models, tracks, passes_by_track, args.pace, args.seed, RAW_PATH
    )
    print(f"timing: appended {len(appended)} rows to {RAW_PATH}")
    quick_summary(RAW_PATH)


if __name__ == "__main__":
    main()
