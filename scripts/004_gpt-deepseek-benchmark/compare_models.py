"""Benchmark round-trip latency and token speed between gpt-5.4-nano and deepseek-v4-flash.

Uses the Kilo Gateway (same codebase as project). Runs each model N times on the same
prompts and prints a comparison table.

Usage:
    python scripts/benchmark/compare_models.py [--runs 3] [--prompts-file data/golden_cleaned.json]
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("KILO_API_KEY")
BASE_URL = "https://api.kilo.ai/api/gateway/chat/completions"

MODELS = ["openai/gpt-5.4-nano", "deepseek/deepseek-v4-flash"]
MAX_TOKENS = 1024
TEMPERATURE = 0.0
DEFAULT_RUNS = 3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_FILE = PROJECT_ROOT / "data" / "golden_cleaned.json"

SYSTEM_PROMPT = (
    "Be a precise, helpful assistant. Answer concisely and accurately."
)

DEFAULT_PROMPTS = [
    "Explain what a Python context manager is and give a short code example.",
    "What is the difference between deep learning and traditional machine learning?",
    "Write a short paragraph summarizing the key features of the Rust programming language.",
    "Compare and contrast SQL and NoSQL databases in 3-4 sentences.",
    "Explain the CAP theorem and why it matters for distributed systems.",
]


class ModelBenchmark:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise RuntimeError("KILO_API_KEY not set")

    def _make_request(
        self, user_prompt: str, stream: bool = False
    ) -> tuple[dict, float, float | None, int, int]:
        """Send one request, return (data, latency_s, ttft_s, prompt_tokens, completion_tokens).

        When stream=False, TTFT is None.
        """
        start = time.perf_counter()

        if stream:
            return self._streaming_request(user_prompt, start)
        else:
            return self._non_streaming_request(user_prompt, start)

    def _non_streaming_request(
        self, user_prompt: str, start: float
    ) -> tuple[dict, float, float | None, int, int]:
        response = httpx.post(
            BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=httpx.Timeout(120.0),
        )
        response.raise_for_status()
        data = response.json()
        latency = time.perf_counter() - start

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return data, latency, None, prompt_tokens, completion_tokens

    def _streaming_request(
        self, user_prompt: str, start: float
    ) -> tuple[dict, float, float | None, int, int]:
        first_token_time = None
        full_content_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        with httpx.stream(
            "POST",
            BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": True,
            },
            timeout=httpx.Timeout(120.0),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]  # strip "data: "
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if first_token_time is None:
                    first_token_time = time.perf_counter() - start

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    full_content_parts.append(content)

                usage_chunk = chunk.get("usage")
                if usage_chunk:
                    prompt_tokens = usage_chunk.get("prompt_tokens", 0)
                    completion_tokens = usage_chunk.get("completion_tokens", 0)

        end = time.perf_counter()
        latency = end - start
        ttft = first_token_time if first_token_time is not None else latency
        data = {"choices": [{"message": {"content": "".join(full_content_parts)}}]}

        return data, latency, ttft, prompt_tokens, completion_tokens

    def run(
        self, prompts: list[str], runs: int, stream: bool = True
    ) -> list[dict]:
        """Run benchmark across prompts × runs and return per-request metrics."""
        results: list[dict] = []
        for i, prompt in enumerate(prompts):
            for r in range(runs):
                print(
                    f"  [{self.model}] prompt {i+1}/{len(prompts)}, "
                    f"run {r+1}/{runs} ...",
                    end=" ",
                    flush=True,
                )
                try:
                    data, latency, ttft, pt, ct = self._make_request(
                        prompt, stream=stream
                    )
                    results.append(
                        {
                            "model": self.model,
                            "prompt_index": i,
                            "run": r,
                            "latency_s": latency,
                            "ttft_s": ttft,
                            "prompt_tokens": pt,
                            "completion_tokens": ct,
                            "output_chars": len(
                                data.get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                            ),
                        }
                    )
                    tps = ct / latency if latency > 0 else 0
                    ttft_str = f"ttft={ttft:.2f}s, " if ttft is not None else ""
                    print(
                        f"latency={latency:.2f}s, "
                        f"{ttft_str}"
                        f"tokens={pt}+{ct}, "
                        f"tps={tps:.1f}"
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                    results.append(
                        {
                            "model": self.model,
                            "prompt_index": i,
                            "run": r,
                            "latency_s": 0.0,
                            "ttft_s": 0.0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "output_chars": 0,
                            "error": str(e),
                        }
                    )
        return results


def load_prompts(path: str) -> list[str]:
    filepath = Path(path)
    if not filepath.exists():
        print(f"Prompts file '{path}' not found, using defaults.")
        return DEFAULT_PROMPTS

    with open(filepath, "r") as f:
        data = json.load(f)

    prompts: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for key in ("raw_full_text", "clean_text", "about_role", "text"):
                    val = item.get(key)
                    if val and isinstance(val, str) and len(val) > 20:
                        prompts.append(f"Summarize this job description:\n\n{val[:3000]}")
                        break
            elif isinstance(item, str) and len(item) > 20:
                prompts.append(item)
    elif isinstance(data, dict):
        for value in data.values():
            if isinstance(value, str) and len(value) > 20:
                prompts.append(value)

    if not prompts:
        print("No suitable prompts found in file, using defaults.")
        return DEFAULT_PROMPTS

    # Cap at 5 prompts to keep benchmark time reasonable
    return prompts[:5]


MODEL_PRICING = {
    "openai/gpt-5.4-nano": (0.15, 0.60),
    "deepseek/deepseek-v4-flash": (0.20, 0.80),
}


def summarize(results: list[dict], model: str) -> dict:
    subset = [r for r in results if r["model"] == model and "error" not in r]
    errors = [r for r in results if r["model"] == model and "error" in r]

    if not subset:
        return {"model": model, "error": "no successful runs"}

    latencies = [r["latency_s"] for r in subset]
    ttfts = [r["ttft_s"] for r in subset if r.get("ttft_s")]
    prompt_tokens = [r["prompt_tokens"] for r in subset]
    completion_tokens = [r["completion_tokens"] for r in subset]
    output_chars = [r["output_chars"] for r in subset]
    total_latency = sum(latencies)
    total_completion = sum(completion_tokens)
    overall_tps = total_completion / total_latency if total_latency > 0 else 0

    price_in, price_out = MODEL_PRICING.get(model, (0.0, 0.0))
    total_prompt = sum(prompt_tokens)
    total_cost = (
        total_prompt * price_in / 1_000_000
        + total_completion * price_out / 1_000_000
    )

    return {
        "model": model,
        "runs": len(subset),
        "errors": len(errors),
        "latency_mean": statistics.mean(latencies),
        "latency_median": statistics.median(latencies),
        "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)],
        "ttft_mean": statistics.mean(ttfts) if ttfts else None,
        "ttft_median": statistics.median(ttfts) if ttfts else None,
        "tps_overall": overall_tps,
        "prompt_tokens_avg": statistics.mean(prompt_tokens),
        "completion_tokens_avg": statistics.mean(completion_tokens),
        "output_chars_avg": statistics.mean(output_chars),
        "total_cost": total_cost,
    }


def print_summary(all_results: list[dict]) -> None:
    summaries = {m: summarize(all_results, m) for m in MODELS}

    header = (
        f"{'Metric':<28} {'gpt-5.4-nano':>16} {'deepseek-v4-flash':>16} {'Winner':>12}"
    )
    print(header)
    print("-" * len(header))

    rows = [
        ("Latency (mean, s)", "latency_mean", "s", False),
        ("Latency (median, s)", "latency_median", "s", False),
        ("Latency (p95, s)", "latency_p95", "s", False),
        ("TTFT (mean, s)", "ttft_mean", "s", False),
        ("TTFT (median, s)", "ttft_median", "s", False),
        ("Tokens/s (overall)", "tps_overall", "tps", True),
        ("Prompt tokens avg", "prompt_tokens_avg", None, False),
        ("Completion tokens avg", "completion_tokens_avg", None, False),
        ("Output chars avg", "output_chars_avg", None, False),
        ("Total cost", "total_cost", "$", False),
        ("Success rate", None, None, False),
    ]

    for label, key, unit, higher_better in rows:
        if key is None and label == "Success rate":
            v0 = summaries[MODELS[0]]["runs"]
            v1 = summaries[MODELS[1]]["runs"]
            t0 = summaries[MODELS[0]]["runs"] + summaries[MODELS[0]]["errors"]
            t1 = summaries[MODELS[1]]["runs"] + summaries[MODELS[1]]["errors"]
            s0 = f"{v0}/{t0}" if t0 else "-"
            s1 = f"{v1}/{t1}" if t1 else "-"
        else:
            v0 = summaries[MODELS[0]].get(key)
            v1 = summaries[MODELS[1]].get(key)
            s0 = fmt_val(v0, unit)
            s1 = fmt_val(v1, unit)

        winner = ""
        if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
            winner = compare(v0, v1, higher_better)

        print(f"{label:<28} {s0:>16} {s1:>16} {winner:>12}")


def fmt_val(value, unit: str | None) -> str:
    if value is None:
        return "-"
    if unit == "s":
        return f"{value:.2f}s"
    if unit == "tps":
        return f"{value:.1f}"
    if unit == "$":
        return f"${value:.6f}"
    return f"{value:.1f}"


def compare(a: float, b: float, higher_better: bool) -> str:
    if higher_better:
        return "← Fast" if a > b else "→ Fast" if b > a else "Tie"
    else:
        return "← Fast" if a < b else "→ Fast" if b < a else "Tie"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark gpt-5.4-nano vs deepseek-v4-flash"
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS, help="Runs per prompt (default: 3)"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=str(DEFAULT_PROMPTS_FILE),
        help="JSON file of prompts to use",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Use non-streaming requests (default: streaming)",
    )
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: KILO_API_KEY not set in .env")
        return

    prompts = load_prompts(args.prompts_file)
    stream = not args.no_stream
    mode = "streaming" if stream else "non-streaming"
    print(
        f"Benchmarking {len(prompts)} prompts × {args.runs} runs each ({mode})\n"
    )

    all_results: list[dict] = []

    for model in MODELS:
        print(f"--- {model} ---")
        bench = ModelBenchmark(model)
        results = bench.run(prompts, args.runs, stream=stream)
        all_results.extend(results)

    result_path = PROJECT_ROOT / "data" / "benchmark_results.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results written to {result_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
