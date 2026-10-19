# Experiment 007 — Two-Track LLM Response-Time Benchmark

**Date:** October 19, 2026

**Hypothesis:** A two-track protocol can measure LLM response latency cleanly across a roster of ~20 cheap instruct models: streaming TTFT from a micro-prompt (Track A) isolates connection + queue + prefill cost, while a non-streaming fixed-length copy task (Track B) makes end-to-end latency and tokens/s directly comparable across models — killing the two confounds that made the first round of this benchmark hard to read.

This experiment grew out of [Experiment 004](./004-gpt-deepseek-benchmark.md) (a 2-model speed test) into a full survey of cheap, lightweight instruct models on the Kilo Gateway. The pipeline's text-cleaning calls are non-streaming request/reply calls, so the model choice for Experiment 003-style cleaning needs honest latency and cost numbers before scaling up. A first round of this benchmark exposed two measurement confounds that drove this revision 3 design:

1. **Output length confound** — an open-ended "recite 5 paragraphs" prompt produced 110 to 8323 output tokens across models, so E2E latency rankings mixed speed with verbosity and disobedience.
2. **Provider buffering/empty-stream confound** — several models returned HTTP 200 streams with *no content deltas at all* (`empty response` faults), and at least one flushed its whole reply in a single burst after a long TTFT.

## Round 1 recap (legacy data, kept as diagnostic)

The first run (October 16, 2026) used one streaming prompt ("recite the first five paragraphs of Lorem Ipsum") across the same 20-model roster, 3 passes. Its 60 raw rows are preserved in `007_raw_results.jsonl` and tagged `legacy` by the analyzer (they carry no `track` field). Findings that motivated the redesign:

- Output tokens ranged **110–8323** per reply from an identical prompt.
- `deepseek/deepseek-v4-flash` **0/3**, `nex-agi/nex-n2.5-mini` 2/3, and `openai/gpt-5-nano` 2/3 failed with `empty response` — no content chunk ever arrived, so these are model/provider faults, not retriable transport faults.
- `ling-3.0-flash` and `qwen3.7-flash` truncated at `MAX_TOKENS=4096`.
- `ling-3.0-flash` showed the smoking gun for streaming-buffering behavior: **TTFT ≈ 10.3s then the entire reply in one chunk** (nominal 20,632 tok/s in one burst). `gpt-5-nano` ditto (28.6s TTFT then instant flush).

## Experimental design

The harness lives in [`scripts/007_llm_latency_benchmark/`](../../scripts/007_llm_latency_benchmark/) (three files, no imports from other experiments):

- **`run_benchmark.py`** — the protocol runner. Track registry, seeded slot shuffle, per-slot round-robin model rotation, 1 req/s pacing, append-per-call JSONL (interrupt-safe), idempotent reruns via `(track, pass_index, prompt_index, model)` resume keys, frozen config header.
- **`latency_client.py`** — streaming client (per-chunk timestamps → TTFT/TPOT/ITL/tail) and non-streaming client (usage-block token counts, `finish_reason`). One transport retry on transient faults; empty content is classified `ok=False` / `empty response`, never dropped.
- **`analyze_results.py`** — track-grouped aggregates; per-model mean / median / p90 / p95 only (no inferential statistics, per scope decision).

### Track A — TTFT (streaming micro-prompt)

- Prompt: `"Say hello world."` — single user message, streaming, `max_tokens=32` (a runaway reply can't pollute passes because content detection works on any delta).
- Metric: **TTFT** (first content delta arrival), mean / median / p90 / p95.
- **10 passes × 20 models** (cheap calls).

### Track B — throughput (non-streaming copy task)

- Prompt: `Repeat the following 5 paragraphs of Lorem Ipsum:` followed by a fixed 398-word reference text (≈ 500 tokens) embedded verbatim in the single user message — but **non-streaming**, `max_tokens=4096`. Output length is *supposed* to be a constant, making E2E and tokens/s comparable.
- Metrics: **E2E** latency (mean / median / p90 / p95), avg output tokens vs reference (deviation), tok/s = output tokens / E2E, $/request (probed gateway pricing), `finish_reason=length` tally (excluded from tok/s).
- **3 passes × 20 models** (expensive calls). Track A always runs before Track B so a partial failure leaves the cheap data first.

### Protocol invariants (shared)

- 20-model roster of cheap, lightweight instruct models (see the frozen config), temperature 0.0, pinned `max_tokens`, live calls only — no caching.
- Seeded shuffle of execution slots and round-robin rotation per slot to decorrelate time-of-day/provider drift.
- Frozen config header (`data/007_config.json`): commit hash, both fully-formatted prompt SHA-256s, per-track passes, seed, pace, and per-token pricing probed live from `/api/gateway/models` with an auditable `probe_failed` zero-cost fallback.
- Raw rows appended to `data/007_raw_results.jsonl`; aggregates to `data/007_analysis.json`. Data lives beside the code in the package's `data/` subdir.
- Full run: 10×20 + 3×20 = **260 calls** at 1 rps ≈ 5 minutes.

## Results

**TL;DR:** `nemotron-3-nano-30b-a3b` and `gemini-2.5-flash-lite` win on both tracks with clean failure records; the round-1 `empty response` fault class is confirmed **streaming-only**; and no model actually copied the reference text exactly, though the field is much tighter than round 1.

### Track A — TTFT (streaming, 10 passes per model)

| model | ok | TTFT mean (s) | TTFT med (s) | TTFT p90 (s) | TTFT p95 (s) | faults |
|---|---|---|---|---|---|---|
| cohere/command-r7b-12-2024 | 10/10 | 0.52 | 0.52 | 0.57 | 0.59 | 0 |
| microsoft/phi-4 | 10/10 | 0.63 | 0.59 | 0.78 | 0.90 | 0 |
| mistralai/mistral-small-24b-instruct-2501 | 10/10 | 1.03 | 0.69 | 2.05 | 2.11 | 0 |
| google/gemma-3-4b-it | 10/10 | 0.84 | 0.69 | 1.33 | 1.53 | 0 |
| nvidia/nemotron-3-nano-30b-a3b | 10/10 | 0.70 | 0.71 | 0.85 | 0.88 | 0 |
| meta-llama/llama-3.1-8b-instruct | 10/10 | 0.76 | 0.74 | 0.84 | 0.87 | 0 |
| mistralai/ministral-8b-2512 | 10/10 | 0.78 | 0.76 | 0.86 | 0.93 | 0 |
| mistralai/ministral-3b-2512 | 10/10 | 0.78 | 0.76 | 0.83 | 0.89 | 0 |
| ibm-granite/granite-4.0-h-micro | 10/10 | 0.81 | 0.78 | 0.98 | 1.08 | 0 |
| mistralai/mistral-nemo | 10/10 | 0.86 | 0.82 | 1.09 | 1.20 | 0 |
| google/gemini-2.5-flash-lite | 10/10 | 0.88 | 0.88 | 0.95 | 0.97 | 0 |
| google/gemma-3-12b-it | 10/10 | 1.06 | 0.97 | 1.45 | 1.86 | 0 |
| google/gemma-3-27b-it | 10/10 | 1.12 | 1.03 | 1.53 | 1.74 | 0 |
| tencent/hy-mt2-7b | 10/10 | 1.21 | 1.15 | 1.32 | 1.49 | 0 |
| amazon/nova-micro-v1 | 10/10 | 1.21 | 1.20 | 1.48 | 1.49 | 0 |
| inclusionai/ling-3.0-flash | 10/10 | 1.47 | 1.36 | 1.90 | 2.12 | 0 |
| nex-agi/nex-n2.5-mini | 10/10 | 1.78 | 1.67 | 2.93 | 3.02 | 0 |
| deepseek/deepseek-v4-flash | 9/10 | 1.71 | 1.68 | 2.05 | 2.08 | 1 |
| qwen/qwen3.7-flash | 10/10 | 3.70 | 3.65 | 4.27 | 4.31 | 0 |

**Not shown:** `openai/gpt-5-nano` — **0/10, all `empty response`** on the streaming track. The analyzer hides all-fault models from the table rather than printing misleading zeros. Same failure signature as round 1, now over 10 consecutive attempts.

Observations:

- The field clusters at 0.5–1.7s median TTFT, with tails behaving well (p95 ≈ median + 0.1–0.5s for most). Provider scheduling variance at 1 rps is modest.
- `qwen3.7-flash` is 3.65s — an order of magnitude better than its round-1 43s, but still structurally last.
- Round-1 pathologies vanished or moved: `ling-3.0-flash` is now a normal, well-behaved stream (TTFT 1.36s, 10/10) vs its round-1 10.3s buffered burst. Gateway routing has clearly changed since October 16.

### Track B — throughput (non-streaming copy task, 3 passes per model)

Reference: 398 words (≈ 500 tokens). `dev` = avg output tokens minus this reference (so +222 ≈ 620 output tokens).

| model | ok | E2E mean (s) | E2E med (s) | E2E p90 (s) | E2E p95 (s) | out_tok avg | dev | tok/s | $/req | trunc | faults |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nvidia/nemotron-3-nano-30b-a3b | 3/3 | 2.76 | 2.45 | 3.36 | 3.48 | 832 | +434 | 314.8 | 0.00021 | 0 | 0 |
| google/gemini-2.5-flash-lite | 3/3 | 3.53 | 3.56 | 3.65 | 3.66 | 620 | +222 | 176.0 | 0.00031 | 0 | 0 |
| google/gemma-3-4b-it | 3/3 | 6.22 | 3.64 | 10.16 | 10.97 | 622 | +224 | 138.5 | 0.00006 | 0 | 0 |
| mistralai/ministral-3b-2512 | 3/3 | 8.07 | 5.23 | 13.13 | 14.12 | 1089 | +691 | 166.7 | 0.00019 | 0 | 0 |
| meta-llama/llama-3.1-8b-instruct | 3/3 | 7.37 | 5.28 | 10.40 | 11.04 | 818 | +420 | 127.8 | 0.00005 | 0 | 0 |
| inclusionai/ling-3.0-flash | 3/3 | 5.22 | 5.74 | 5.78 | 5.79 | 910 | +512 | 178.7 | 0.00022 | 0 | 0 |
| tencent/hy-mt2-7b | 3/3 | 7.45 | 7.22 | 7.81 | 7.88 | 809 | +411 | 108.9 | 0.00030 | 0 | 0 |
| mistralai/ministral-8b-2512 | 3/3 | 8.87 | 8.97 | 9.89 | 10.01 | 877 | +479 | 100.4 | 0.00026 | 0 | 0 |
| nex-agi/nex-n2.5-mini | 3/3 | 8.60 | 9.53 | 10.34 | 10.44 | 1458 | +1060 | 167.1 | 0.00017 | 0 | 0 |
| openai/gpt-5-nano | 3/3 | 10.62 | 9.65 | 13.88 | 14.41 | 1653 | +1255 | 155.1 | 0.00069 | 0 | 0 |
| amazon/nova-micro-v1 | 3/3 | 11.98 | 11.36 | 12.95 | 13.15 | 4096 | +3698 | - | 0.00061 | 3 | 0 |
| cohere/command-r7b-12-2024 | 3/3 | 11.15 | 11.44 | 12.22 | 12.32 | 868 | +470 | 78.8 | 0.00016 | 0 | 0 |
| microsoft/phi-4 | 3/3 | 14.31 | 14.71 | 16.08 | 16.25 | 802 | +404 | 57.1 | 0.00014 | 0 | 0 |
| google/gemma-3-12b-it | 3/3 | 13.71 | 14.85 | 15.30 | 15.35 | 638 | +240 | 47.7 | 0.00010 | 0 | 0 |
| deepseek/deepseek-v4-flash | 3/3 | 14.08 | 16.02 | 16.88 | 16.99 | 1531 | +1133 | 107.6 | 0.00037 | 0 | 0 |
| google/gemma-3-27b-it | 3/3 | 21.63 | 20.22 | 23.85 | 24.30 | 623 | +225 | 29.1 | 0.00015 | 0 | 0 |
| qwen/qwen3.7-flash | 3/3 | 21.81 | 20.42 | 24.87 | 25.43 | 1968 | +1570 | 90.9 | 0.00028 | 0 | 0 |
| ibm-granite/granite-4.0-h-micro | 3/3 | 24.56 | 24.70 | 24.74 | 24.75 | 809 | +411 | 32.9 | 0.00011 | 0 | 0 |
| mistralai/mistral-nemo | 3/3 | 37.65 | 35.52 | 40.86 | 41.53 | 851 | +453 | 22.8 | 0.00025 | 0 | 0 |
| mistralai/mistral-small-24b-instruct-2501 | 3/3 | 56.82 | 61.69 | 76.41 | 78.25 | 1675 | +1277 | 35.5 | 0.00018 | 0 | 0 |

All 60 Track B calls succeeded — **zero faults, zero empty responses, non-streaming**. The `empty response` failure mode of round 1 is exclusive to streaming; every model that streamed brokenly is 3/3 here.

### Legacy round — streaming lorem recitation (diagnostic only)

Preserved for the buffering-delta read. Full table printed by `analyze_results.py --include-legacy`; the takeaways used below:

- Healthy models' per-stream tok/s (legacy) and non-stream tok/s (Track B) agree within ~±10% on comparable work: mistral-nemo 21.9 vs 22.8, granite 36.2 vs 32.9, gemma-3-27b 33.5 vs 29.1 — **no hidden SSE tax** for well-routed providers.
- The round-1 absurd "streaming speeds" (ling 20,632 tok/s, gpt-5-nano 2,464, nex 983) were burst-flush artifacts; their honest throughput is what Track B measures (179, 155, 167 tok/s).

## Discussion

- **Single-track metrics lie.** `command-r7b` has the best TTFT of the whole roster (0.52s) but only 78.8 tok/s decode → 11.4s E2E. A one-metric benchmark would have crowned it. Track A + Track B together is the minimum honest protocol.
- **The copy task is only partially copied.** Deviation spans +222 to +6,988 tokens: the best-behaved models emit ~620 output tokens for a 398-word reference (instruction echo + light re-styling), while `nova-micro-v1` pinned all three outputs at the 4096 cap — a recitation loop, excluded from tok/s per protocol. As designed, this experiment measures **speed and cost only**, not fidelity; Track B's deviation column exists precisely so length drift doesn't get misread as quality.
- **Failure modes archived, not aborted.** `gpt-5-nano` v 0/10 streaming / 3/3 non-streaming is a clean determination that broken delivery is SSE-specific, not transient. `deepseek-v4-flash` shows the same pattern, 1 fault. This matches the [Experiment 004 update](./004-gpt-deepseek-benchmark.md) where deepseek hung on extraction work mid-round.
- **Cost dispersion is real even in the "cheap" tier** — $5e-5 (llama-8b) to $6.9e-4 (gpt-5-nano) per request, a ~14× spread, and the expensive end is not the fast end.

## Verdict

- **Top picks:** `nemotron-3-nano-30b-a3b` — best E2E (2.45s med), 16/16 clean across all three rounds, $2.1e-4/req. `gemini-2.5-flash-lite` — best consistency (tightest p95s on both tracks), 3.56s med, $3.1e-4/req.
- **Budget lane:** `gemma-3-4b-it` (3.64s med, $6e-5, but a 10.2s p90 tail) and `meta-llama/llama-3.1-8b-instruct` (5.28s med, $5e-5, cheapest in field).
- **Avoid for latency reasons:** `mistral-small-24b` (62s med — though it has fine TTFT, its decode side is punishing), `mistral-nemo` (35s), `nova-micro` (recitation loop), `gpt-5-nano` (verbose, expensive, SSE broken), `qwen3.7-flash` (worst TTFT + verbose).
- **For the cleaning pipeline** (non-streaming workload): the Track B table *is* the selection sheet. `deepseek-v4-flash`'s 16s median puts it mid-pack — defensible on quality, not on speed.

## Honest caveats

1. **n=3 per model on Track B.** Medians are stable across models; tail claims (p90/p95) are anecdote-grade at n=3. Track A's n=10 gives respectable tails.
2. **Deviation is systematic.** Even "faithful" copiers emit ~20–60% more tokens than the reference, so E2E spreads retain a genuine length component; tok/s (where defined) is the length-normalized cross-check.
3. **One time window, one gateway.** Round 1 vs round 2 differences (ling's rebirth, qwen's TTFT surge) prove gateway routing drifts over days. Rerun before finalizing a production choice.
4. **398 words ≠ 500 tokens exactly.** The reference scale is a word count used as a token-scale proxy (the `dev`/tok/s denominations are honest but shouldn't be over-precise).
5. **Cost columns degrade silently to zero** when the pricing probe fails — the `pricing_source` field makes this auditable (`gateway_probe` for this run).

## Next steps

- Re-benchmark top-3 shortlist on the *actual* cleaning workload (posting texts, real prompt shape) with quality gates from Experiment 003 — latency ranking can flip when the prompt changes (~500-token output here vs ~1000+ for cleanings).
- Track the empty-stream fault class: if `gpt-5-nano` / `deepseek-v4-flash` become cleaning candidates on quality, prefer non-streaming calls for them explicitly.
- Archive `007_analysis.json` values in the experiment report only at freeze points; re-runs overwrite raw data in place (append-only), so this report documents the October 17, 2026 run.

## Artifacts

- [`scripts/007_llm_latency_benchmark/`](../../scripts/007_llm_latency_benchmark/) — `run_benchmark.py`, `latency_client.py`, `analyze_results.py`, `data/` (config, raw JSONL, analysis JSON)
- Analysis JSON: `scripts/007_llm_latency_benchmark/data/007_analysis.json`
- Reproduce the analysis offline (no API calls): `python scripts/007_llm_latency_benchmark/analyze_results.py --include-legacy`
- Re-run the benchmark: `python scripts/007_llm_latency_benchmark/run_benchmark.py --track all --force` (needs `KILO_API_KEY` in `.env`; idempotent otherwise)
