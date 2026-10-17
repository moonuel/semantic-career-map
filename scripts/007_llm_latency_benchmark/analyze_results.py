"""Analyze exp-007 raw results: simple response-time aggregates per model.

Inputs: ``007_config.json`` and ``007_raw_results.jsonl`` beside the code.
Output: per-model mean / median / p90 / p95 for TTFT, E2E, TPOT, tail, ITL;
average tokens and tokens-per-stream; failures; cost per request from the
probed gateway pricing; a printed markdown summary sorted by E2E median.

Usage:
    python scripts/007_llm_latency_benchmark/analyze_results.py \
        [--raw data/007_raw_results.jsonl] [--out data/007_analysis.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

from latency_client import quantile

DATA_DIR = Path(__file__).resolve().parent / "data"
CONFIG_PATH = DATA_DIR / "007_config.json"
DEFAULT_RAW = DATA_DIR / "007_raw_results.jsonl"
DEFAULT_OUT = DATA_DIR / "007_analysis.json"

METRICS = ("ttft_s", "latency_s", "tpot_s", "tail_s", "itl_p50_s", "itl_p95_s")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return parsed rows from a JSONL file (empty when absent)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def tokens_per_second(row: dict[str, Any]) -> float | None:
    """Return the per-stream speed of one record, None when undefined."""
    latency = row.get("latency_s")
    ttft = row.get("ttft_s")
    tokens = row.get("completion_tokens", 0)
    if latency is None or ttft is None or latency - ttft <= 0 or tokens < 1:
        return None
    return tokens / (latency - ttft)


def aggregate_model(
    model: str, rows: list[dict[str, Any]], pricing: dict[str, Any]
) -> dict[str, Any]:
    """Reduce one model's raw rows to its mean/median/p90/p95 summary."""
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
    tps = [t for t in (tokens_per_second(r) for r in ok) if t is not None]
    block: dict[str, Any] = {
        "model": model,
        "calls": len(rows),
        "ok": len(ok),
        "faults": len(faults),
        "fault_examples": [r["error"] for r in faults[:3]],
        "usage_missing": sum(1 for r in ok if r.get("usage_missing")),
        "prompt_tokens_avg": st.mean([r["prompt_tokens"] for r in ok]) if ok else None,
        "completion_tokens_avg": (
            st.mean([r["completion_tokens"] for r in ok]) if ok else None
        ),
        "tokens_per_stream_mean": st.mean(tps) if tps else None,
        "cost_per_request_usd": (st.mean(costs) if costs else 0.0) or None,
    }
    for metric in METRICS:
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
    """CLI entry point: aggregate raw rows and persist the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = json.load(f)
    raw_rows = iter_jsonl(args.raw)
    if not raw_rows:
        sys.exit(f"no timing rows in {args.raw} — run run_benchmark.py first")

    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    pricing_all = config.get("pricing", {})
    pricing_all = {model: prices for model, prices in pricing_all.items() if prices}
    blocks = [
        aggregate_model(model, rows, pricing_all.get(model, []))
        for model, rows in sorted(by_model.items())
    ]
    output = {
        "experiment": "007_llm_latency_benchmark",
        "pricing_source": config.get("pricing_source"),
        "models": [
            {"model": block["model"], "calls": block["calls"], "ok": block["ok"]}
            for block in blocks
        ],
        "timing": blocks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print_markdown(blocks)


def print_markdown(blocks: list[dict[str, Any]]) -> None:
    """Print the per-model response-time summary sorted by E2E median."""
    ranked = sorted(
        blocks,
        key=lambda b: b["latency_s"]["median"] if b["latency_s"] else float("inf"),
    )
    print("\n## Response times (s)\n")
    print(
        "| model | ok | E2E med | E2E p95 | TTFT med | TTFT p95 | TPOT med "
        "| tok/s | $/req |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
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
            f"{block['tokens_per_stream_mean']:.1f}"
            if block["tokens_per_stream_mean"]
            else "-"
        )
        print(
            f"| {block['model']} | {block['ok']}/{block['calls']} | "
            f"{e2e.get('median', 0):.2f} | {e2e.get('p95', 0):.2f} | "
            f"{ttft.get('median', 0):.2f} | {ttft.get('p95', 0):.2f} | "
            f"{tpot.get('median', 0) if tpot else 0:.3f} | {tps} | {cost} |"
        )


if __name__ == "__main__":
    main()
