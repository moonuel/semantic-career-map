"""Analyze exp-007 two-track raw results: response-time aggregates per track.

Inputs: ``007_config.json`` and ``007_raw_results.jsonl`` beside the code.
Tracks: ``ttft`` — streaming micro-prompt, TTFT stats; ``throughput`` —
non-streaming Lorem Ipsum copy task, E2E stats, tokens/s, cost from probed
gateway pricing. Rows written before the two-track revision carry no
``track`` field; they are reported only with ``--include-legacy`` as the
streaming-buffering diagnostic table.

Mean / median / p90 / p95 only — no inferential statistics.

Usage:
    python scripts/007_llm_latency_benchmark/analyze_results.py \
        [--raw data/007_raw_results.jsonl] [--out data/007_analysis.json] \
        [--include-legacy]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

from latency_client import quantile
from run_benchmark import LOREM_REFERENCE

DATA_DIR = Path(__file__).resolve().parent / "data"
CONFIG_PATH = DATA_DIR / "007_config.json"
DEFAULT_RAW = DATA_DIR / "007_raw_results.jsonl"
DEFAULT_OUT = DATA_DIR / "007_analysis.json"

TRACK_ROOTS = ("ttft", "throughput", "legacy")
REFERENCE_TOKENS = len(LOREM_REFERENCE.split())

METRICS = ("ttft_s", "latency_s", "tpot_s", "tail_s", "itl_p50_s", "itl_p95_s")

# Metrics worth blocking per track: ttft track keys on TTFT (latency kept
# for the cross-check), throughput keys on E2E, legacy keeps the full
# pre-revision diagnostic set.
TRACK_METRICS: dict[str, tuple[str, ...]] = {
    "ttft": ("ttft_s", "latency_s"),
    "throughput": ("latency_s",),
    "legacy": METRICS,
}


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return parsed rows from a JSONL file (empty when absent)."""
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def row_track(row: dict[str, Any]) -> str:
    """Return a row's track name; pre-two-track rows default to legacy."""
    return str(row.get("track", "legacy"))


def stream_tokens_per_second(row: dict[str, Any]) -> float | None:
    """Return the per-stream decode speed of one streaming record, None when undefined."""
    latency = row.get("latency_s")
    ttft = row.get("ttft_s")
    tokens = row.get("completion_tokens", 0)
    if latency is None or ttft is None or latency - ttft <= 0 or tokens < 1:
        return None
    return tokens / (latency - ttft)


def copy_tokens_per_second(row: dict[str, Any]) -> float | None:
    """Return completion_tokens / e2e for one non-streaming row, None when undefined.

    Truncated outputs (``finish_reason == "length"``) are excluded — their
    duration is bounded by the token cap, not by completing the task.
    """
    latency = row.get("latency_s")
    tokens = row.get("completion_tokens", 0)
    if row.get("finish_reason") == "length":
        return None
    if latency is None or latency <= 0 or tokens < 1:
        return None
    return tokens / latency


def median_key(block: dict[str, Any], metric: str) -> float:
    """Return a block's metric median, infinite when the metric is absent."""
    stats = block.get(metric)
    return float(stats["median"]) if stats else float("inf")


def aggregate_model(
    model: str, rows: list[dict[str, Any]], track: str, pricing: list[Any]
) -> dict[str, Any]:
    """Reduce one model's raw rows of one track to its summary block."""
    ok = [r for r in rows if r.get("ok")]
    faults = [r for r in rows if not r.get("ok")]
    if len(pricing) == 2:
        price_in, price_out = float(pricing[0]), float(pricing[1])
    else:
        price_in, price_out = 0.0, 0.0
    costs = [
        r["prompt_tokens"] * price_in + r["completion_tokens"] * price_out
        for r in ok
        if not r.get("usage_missing")
    ]
    tps: list[float] = []
    if track == "throughput":
        tps = [t for t in (copy_tokens_per_second(r) for r in ok) if t is not None]
    elif track == "legacy":
        tps = [t for t in (stream_tokens_per_second(r) for r in ok) if t is not None]
    fault_tally: dict[str, int] = {}
    for r in faults:
        key = str(r.get("error") or "unknown")
        fault_tally[key] = fault_tally.get(key, 0) + 1
    block: dict[str, Any] = {
        "track": track,
        "model": model,
        "calls": len(rows),
        "ok": len(ok),
        "faults": len(faults),
        "fault_tally": fault_tally,
        "fault_examples": [r["error"] for r in faults[:3]],
        "usage_missing": sum(1 for r in ok if r.get("usage_missing")),
        "finish_reason_length": sum(
            1 for r in ok if r.get("finish_reason") == "length"
        ),
        "prompt_tokens_avg": st.mean([r["prompt_tokens"] for r in ok]) if ok else None,
        "completion_tokens_avg": (
            st.mean([r["completion_tokens"] for r in ok]) if ok else None
        ),
        "tokens_per_second_mean": st.mean(tps) if tps else None,
        "cost_per_request_usd": (st.mean(costs) if costs else 0.0) or None,
    }
    for metric in TRACK_METRICS[track]:
        values = [r[metric] for r in ok if r.get(metric) is not None]
        block[metric] = (
            {
                "mean": st.mean(values),
                "median": st.median(values),
                "p90": quantile(values, 0.90),
                "p95": quantile(values, 0.95),
            }
            if values
            else None
        )
    return block


def main() -> None:
    """CLI entry point: aggregate raw rows per track and persist the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="also summarize pre-two-track streaming rows (buffering diagnostic)",
    )
    args = parser.parse_args()

    config: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = json.load(f)
    raw_rows = iter_jsonl(args.raw)
    if not raw_rows:
        sys.exit(f"no timing rows in {args.raw} — run run_benchmark.py first")

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in raw_rows:
        groups.setdefault(row_track(row), {}).setdefault(str(row["model"]), []).append(
            row
        )
    pricing_all = config.get("pricing", {})
    pricing_all = {model: prices for model, prices in pricing_all.items() if prices}

    timing: dict[str, list[dict[str, Any]]] = {}
    reported: list[str] = []
    for track in TRACK_ROOTS:
        if track == "legacy" and not args.include_legacy:
            continue
        if not groups.get(track):
            continue
        rows_count = sum(len(rows) for rows in groups[track].values())
        timing[track] = [
            aggregate_model(model, rows, track, pricing_all.get(model, []))
            for model, rows in sorted(groups[track].items())
        ]
        reported.append(f"{track}: {rows_count} rows")

    if not timing:
        sys.exit(
            f"no two-track rows in {args.raw} — pass --include-legacy to inspect "
            "the pre-revision streaming run"
        )
    print(f"analyzed {'; '.join(reported)}")

    output = {
        "experiment": "007_llm_latency_benchmark",
        "pricing_source": config.get("pricing_source"),
        "reference_tokens": REFERENCE_TOKENS,
        "timing": timing,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if "ttft" in timing:
        print_track_a_markdown(timing["ttft"])
    if "throughput" in timing:
        print_track_b_markdown(timing["throughput"])
    if "legacy" in timing:
        print_legacy_markdown(timing["legacy"])


def print_track_a_markdown(blocks: list[dict[str, Any]]) -> None:
    """Print the Track A table: TTFT responsiveness, sorted by TTFT median."""
    ranked = sorted(blocks, key=lambda b: median_key(b, "ttft_s"))
    print('\n## Track A — TTFT (streaming "Say hello world.")\n')
    print("| model | ok | TTFT mean | TTFT med | TTFT p90 | TTFT p95 | faults |")
    print("|---|---|---|---|---|---|---|")
    for block in ranked:
        ttft = block["ttft_s"] or {}
        if not ttft:
            continue
        print(
            f"| {block['model']} | {block['ok']}/{block['calls']} | "
            f"{ttft['mean']:.2f} | {ttft['median']:.2f} | "
            f"{ttft['p90']:.2f} | {ttft['p95']:.2f} | {block['faults']} |"
        )


def print_track_b_markdown(blocks: list[dict[str, Any]]) -> None:
    """Print the Track B table: throughput copy task, sorted by E2E median."""
    print(
        f"\nTrack B reference: LOREM_REFERENCE = {REFERENCE_TOKENS} words "
        "(dev column = avg output tokens minus this)\n"
    )
    ranked = sorted(blocks, key=lambda b: median_key(b, "latency_s"))
    print("## Track B — throughput (non-streaming lorem copy task, E2E)\n")
    print(
        "| model | ok | E2E mean | E2E med | E2E p90 | E2E p95 | out_tok avg "
        "| dev | tok/s | $/req | trunc | faults |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for block in ranked:
        e2e = block["latency_s"] or {}
        if not e2e:
            continue
        out_tok = block["completion_tokens_avg"]
        dev = out_tok - REFERENCE_TOKENS if out_tok is not None else None
        tps = (
            f"{block['tokens_per_second_mean']:.1f}"
            if block["tokens_per_second_mean"]
            else "-"
        )
        cost = (
            f"{block['cost_per_request_usd']:.5f}"
            if block["cost_per_request_usd"]
            else "-"
        )
        print(
            f"| {block['model']} | {block['ok']}/{block['calls']} | "
            f"{e2e['mean']:.2f} | {e2e['median']:.2f} | {e2e['p90']:.2f} | "
            f"{e2e['p95']:.2f} | {out_tok:.0f} | {dev:+.0f} | {tps} | {cost} | "
            f"{block['finish_reason_length']} | {block['faults']} |"
        )


def print_legacy_markdown(blocks: list[dict[str, Any]]) -> None:
    """Print the legacy streaming table: pre-revision buffering diagnostic."""
    ranked = sorted(blocks, key=lambda b: median_key(b, "latency_s"))
    print("\n## Legacy run (streaming lorem recitation, no track field) — diagnostic\n")
    print("| model | ok | E2E med | E2E p95 | TTFT med | TPOT med | tok/s | $/req |")
    print("|---|---|---|---|---|---|---|---|")
    for block in ranked:
        e2e = block["latency_s"] or {}
        ttft = block["ttft_s"] or {}
        tpot = block["tpot_s"] or {}
        cost = (
            f"{block['cost_per_request_usd']:.5f}"
            if block["cost_per_request_usd"]
            else "-"
        )
        tps = (
            f"{block['tokens_per_second_mean']:.1f}"
            if block["tokens_per_second_mean"]
            else "-"
        )
        print(
            f"| {block['model']} | {block['ok']}/{block['calls']} | "
            f"{e2e.get('median', 0):.2f} | {e2e.get('p95', 0):.2f} | "
            f"{ttft.get('median', 0):.2f} | {tpot.get('median', 0) if tpot else 0:.3f} "
            f"| {tps} | {cost} |"
        )


if __name__ == "__main__":
    main()
