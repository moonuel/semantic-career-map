"""Experiment 007 — standalone LLM response-time benchmark.

Pre-registered protocol: a fixed recitation prompt (first five paragraphs of
Lorem Ipsum) is sent to the expanded candidate roster over the Kilo Gateway
streaming API. Interleaved round-robin execution, seeded ordering, one
transport retry, and per-call ``stream_options`` usage blocks yield TTFT,
E2E latency, TPOT, ITL, tail time, and token counts. Analysis reports
mean / median / p90 / p95 response times plus cost per request from probed
gateway pricing.

Data products live beside the code in ``scripts/007_llm_latency_benchmark/data/``.
"""
