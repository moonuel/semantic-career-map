# Experiment 004 -- Benchmarking gpt-5.4-nano vs. deepseek-v4-flash speed

**Date:** July 27, 2026

This experiment benchmarks the performance of GPT-5.4 Nano against DeepSeek V4 Flash in terms of response time and throughput.
These models are both cheap, lightweight models, and I've had a lot of success with `deepseek-v4-flash` in coding work. 
`gpt-5.4-mini` is a personal favourite as well, so I thought I should compare how it's little cousin fares against it.

The objective is not to compare performance on accuracy or quality, but rather to measure speed and cost efficiency.
Kilo Code (my preferred coding agent tool) suggests that `deepseek-v4-flash` is less than half the price of `gpt-5.4-mini`, but the Deepseek model performed poorly on the [last experiment](./003-llm-extraction.md) due to very long delays. 

I was just wondering if that was an anomaly or if Deepseek V4 Flash is actually very slow.

## Experimental design

A test script, [`compare_models.py`](../../scripts/004_gpt-deepseek-benchmark/compare_models.py), was used to benchmark both models against a standardized set of prompts. The script sends identical requests to each model and measures response times across multiple runs to ensure statistical reliability.

The prompts include:

- "Explain what a Python context manager is and give a short code example."
- "What is the difference between deep learning and traditional machine learning?"
- "Write a short paragraph summarizing the key features of the Rust programming language."
- "Compare and contrast SQL and NoSQL databases in 3-4 sentences."
- "Explain the CAP theorem and why it matters for distributed systems."

The script sends identical requests to each model and measures response times across multiple runs to ensure statistical reliability.

The primary metrics of interest were:

- Time to first token
- Tokens per second
- Total response time

## Results

The following table enumerates the key results (three runs each):

| Metric                | gpt-5.4-nano | deepseek-v4-flash | Winner |
| --------------------- | ------------ | ----------------- | ------ |
| Latency (mean, s)     | 4.23s        | 4.27s             | ← Fast |
| Latency (median, s)   | 3.19s        | 4.13s             | ← Fast |
| Latency (p95, s)      | **9.02s**    | 5.65s             | → Fast |
| TTFT (mean, s)        | 1.26s        | 1.89s             | ← Fast |
| TTFT (median, s)      | **0.96s**    | 2.10s             | ← Fast |
| **Tokens/s (overall)**| 69.2         | 56.5              | ← Fast |
| Prompt tokens avg     | 502.4        | 495.0             | → Fast |
| Completion tokens avg | 292.7        | 241.1             | → Fast |
| Output chars avg      | **1435.3**   | **661.5**         | → Fast |

Wow! This is not a very powerful statistical test due to the small sample size (only three runs each). 
The 9 second 95th percentile latency for `gpt-5.4-nano` is very surprising though, despite the lower mean AND median. 
I won't hold it against it though, since again this is a very small and transient test.

`gpt-5.4-nano` had a comparatively very low median TTFT however, perhaps reflecting the longer internet delay between my local machine and the Deepseek server location. 
`gpt`'s mean TTFT was significantly higher than its median, but still lower than Deepseek's.
Probably an artifact of whatever caused the outlier latency spike.

Anyways `deepseek-v4-flash` performed well enough for me to want to potentially halve my costs. 

## Discussion

- `deepseek-v4-flash` performed comparably well with `gpt-5.4-nano` on these speed benchmarks. 
- It was consistently slower than `gpt-5.4-nano`, but still competitive in many metrics.
- The 9 second outlier for `gpt-5.4-nano` suggests some instability that I won't worry too much about due to the very small sample size. Could just be bad luck. 
- I will probably prefer `deepseek-v4-flash` for text extraction work due to its lower cost and competitive performance.