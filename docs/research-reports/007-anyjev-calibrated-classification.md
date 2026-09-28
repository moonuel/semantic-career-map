# AnyJev & Jev — Calibrated LLM Classification Without Generation

> Date: 2026-09-28
> Context: Research report on AnyJev (Nokia Applied Research) and the Jev model it is modeled after (TypeSafe AI), assessing relevance to the Semantic Career Mapping Platform — specifically to role/taxonomy classification and confidence-calibrated decision thresholds.

---

## Table of Contents

1. [Background: What Is AnyJev (and Jev)?](#1-background-what-is-anyjev-and-jev)
2. [How AnyJev Works: The Levels](#2-how-anyjev-works-the-levels)
3. [Results Summary](#3-results-summary)
4. [Jev — The Proprietary Counterpart](#4-jev--the-proprietary-counterpart)
5. [Relevance to Semantic Career Mapping](#5-relevance-to-semantic-career-mapping)
6. [Limitations & Caveats](#6-limitations--caveats)
7. [Sources](#7-sources)

---

## 1. Background: What Is AnyJev (and Jev)?

On 2026-09-15, TypeSafe AI released **Jev**, the first public "System One" model — a proprietary classifier that generates no text. Given a *state* (unstructured text/JSON) and one or more *questions*, it returns a **typed decision**: a chosen option, a probability per option, and a confidence score. Jev is trained with **RLCD** (Reinforcement Learning for Calibrated Decisions) so that its stated probabilities match observed frequencies.

Shortly after, researchers at Nokia (Jiamu Zhang, Tianze Yang, Yucheng Shi, Liang Wu) published **AnyJev**, an Apache-2.0 open-source library that pursues the same goal — a calibrated, typed decision model — using **any open LLM, with no fine-tuning**. AnyJev reads the model's next-token probability distribution over option letters (A, B, C…) in a single prefill pass: no generation, no parsing, no gradient descent.

The name references William Stanley Jevons and Kahneman's System 1 ("fast" judgment, as opposed to System 2 "slow" reasoning).

**Core contrast:** Jev is a trained proprietary model; AnyJev is a training-free layer over models you already run locally.

---

## 2. How AnyJev Works: The Levels

AnyJev addresses two distinct flaws in raw LLM classification, progressively, through three levels:

### The two flaws

1. **Position bias** — the model prefers certain letters/positions regardless of content. Reordering options changes the answer (raw flip rate 0.230 on Qwen3-8B / BANKING77).
2. **Poorly scaled confidence** — post-trained models (esp. RLHF) are overconfident; a "0.9" is not actually 90% correct. (Not universal: base models are often better calibrated, so "measure before correcting.")

### L0 — position bias, zero labels

- Present options in every cyclic rotation (A-B-C, B-C-A, C-A-B…) and average the scores, so each option passes through every position.
- Apply a **label-prior correction** (estimated without labels) to offset the model's leaning toward certain categories.
- Cost: K prefills for a K-option choice (reducible to ~7 of 18 via a rotation budget / `adaptive_shifts`).
- Effect on BANKING77: flip rate 0.230 → 0.073; accuracy 0.747 → 0.803.

### L1 — calibration, 100–500 labels per question

- **Temperature scaling**: divide logits by a temperature *T* learned on a validation set, before softmax.
- Does **not** change the ranking — only rescales confidence (accuracy moves marginally 0.803 → 0.807, which is not its purpose).
- Effect: ECE 0.184 (L0) → 0.095 (vs 0.240 raw).

### L2 — reading inside the model, 100–300 labels per question

- Stops computation at ~2/3 of layers (e.g. block 24 of 36 for Qwen3-8B) and retrieves the hidden state.
- A small **closed-form linear head** (shrinkage LDA or ridge, ~100 KB, solved in seconds on CPU, no gradients) maps that hidden state to an answer.
- Improves the *answer*, not just the confidence; costs ~0.68× of one full forward pass.
- The head **self-maintains**: only feature mean/scale are re-estimated from ~30 unlabeled requests to adapt to reworded questions.

Every `Decision` carries its `level`, and `require="L1"` makes downstream code refuse weaker outputs (`LevelError`).

---

## 3. Results Summary

All numbers from the AnyJev README (regenerated from committed JSON).

### BANKING77, Qwen3-8B, 20-way, 300 test items

| | raw logits | L0 (zero labels) | L1 (+temperature) |
|:--|:--:|:--:|:--:|
| Labels required | none | none | 100–500 |
| Answer flips when options reversed | 0.230 | 0.073 | 0.077 |
| Accuracy | 0.747 | 0.803 | 0.807 |
| Calibration error (ECE) | 0.240 | 0.184 | 0.095 |
| **Auto-decidable at ≤5% error** | **7.7%** | **46.3%** | **52.0%** |

The headline insight is the last row: accuracy improves only ~6 points, but the share of traffic that can be **safely automated** at ≤5% error rises **7.7% → 52.0% (6.8×)**. Most of the gain is already from L0 (46.3%) with zero labels.

### Typed-decisions benchmark (LocalLLaMA/typed-decisions), L2

| model | L0, zero labels | L2 | block | cost vs one forward |
|:--|:--:|:--:|:--:|:--:|
| Qwen3-1.7B | 0.494 | 0.730 | 18/28 | 0.70× |
| Qwen3-4B | 0.564 | 0.786 | 24/36 | 0.69× |
| Qwen3-8B | 0.647 | 0.771 | 24/36 | 0.68× |
| Qwen3-30B-A3B | 0.630 | 0.799 | 40/48 | — |
| Qwen3-32B | 0.700 | 0.798 | 52/64 | 0.84× |

Reference points published by their authors on the same set: Jev 0.727, fine-tuned Laya 0.768. Pooled ECE at L2 is 0.03–0.05.

---

## 4. Jev — The Proprietary Counterpart

Jev is a classifier that can be addressed like a language model, but returns typed values instead of strings.

- **API:** `POST /v1/systemone` (not a chat-completions interface); three question types — `choice`, `score`, `noul`.
- **Pricing:** $0.042 / 1M input tokens; output tokens free. Context 64k (32k state + longest question).
- **Training:** RLCD — rewards that stated probability matches actual hit rate.
- **Claims:** 70–500 ms per call; parallel (not sequential) output; 0% type errors (schema guaranteed by construction); "cannot hallucinate" in the narrow sense of inventing a schema value — but it *can* choose the wrong class.
- **Caveats (vendor-stated):** calibration is a population property, not a per-answer guarantee; price may be subsidized; latency measured from US West Coast; "193.6× faster / 444.6× cheaper" figures are internal evals against two frontier models; independent benchmarks not yet available.

The architectural takeaway from the innFactory review: **"a fast model decides and routes, a generative model drafts, a person takes over when confidence is not sufficient."** Jev/AnyJev are the "smart if-statement" building block — control flow stays in code, deterministic rules stay in code, and the calibrated model appears only where programmable common sense is needed.

---

## 5. Relevance to Semantic Career Mapping

The project already has a classification-style task: **taxonomy/role labeling** (Experiment 006 — taxonomy role labeling). AnyJev's primitives and calibration levels map directly onto it.

| Project need | AnyJev fit |
|---|---|
| **Role/taxonomy classification** of a job posting | `Question.choice` over `role_category` options (or `noul` for binary flags) |
| **Skill / field tagging** with confidence | `Question.choice` per tag, `Question.score` for seniority level |
| **Routing low-confidence items to manual review** | Calibrated probabilities make a "human-in-the-loop if p < threshold" rule meaningful — the exact 7.7% → 52.0% automatable-traffic story |
| **Avoiding brittle prompt-parse** of a free-text LLM answer | Typed decisions remove parsing/format drift entirely |

### Concrete fit points

- **Labeling pipeline (Experiment 006):** Today role labeling likely uses a generative LLM with structured-output parsing. AnyJev offers a faster, training-free alternative that also *quantifies* its own uncertainty — useful for deciding which postings need human review of their assigned role.
- **Golden-set / evaluation support:** The calibration metrics (reliability diagram, ECE, Brier score) and "choose threshold on held-out data" discipline align with the project's eval-first culture (`data/golden_set.json`).
- **CPU-first, local-model constraint:** AnyJev runs on the Hugging Face `transformers` backend and serves via vLLM — compatible with the project's CPU-first, Docker deployment bias. L2 requires local hidden-state access (not a closed API).

### Caveats for this project

- **L2 is per-question and per-model** — heads ship only for Qwen3; a new model or new question needs new labels + a re-solved head. This is significant friction for an evolving taxonomy.
- **"Accuracy" on typed decisions is agreement with a teacher LLM, not human ground truth** (teacher self-agreement ≈ 0.735) — the project's own golden set would be a better ground-truth source.
- **≤26 options** in the letter readout (span readout not yet implemented). A `role_category` taxonomy must fit within 26 classes for L0/L1; L2 is not subject to this ceiling in the same way.
- **Young project** (released late September 2026, days before this report): headline numbers are point estimates at n=300 with wide confidence intervals.

### Recommendation

Treat AnyJev as a **candidate for a spike experiment**, not an immediate dependency: benchmark its L0/L1 readout against the current role-labeling approach on the project's own golden set. The lowest-cost, highest-signal test is **L0** (zero labels, K-rotation readout) — it isolates whether position-bias correction alone improves the project's classification agreement. Defer L2 until the taxonomy is stable and Qwen-class models are in use.

---

## 6. Limitations & Caveats

- **Teacher-LLM ground truth:** typed-decisions "accuracy" is not human ground truth.
- **Per-question / per-model L2 heads:** no transfer across questions or models; only Qwen3 heads ship.
- **Model coverage:** headline results are Qwen models; L2 needs hidden states (local `transformers` / vLLM embed server; SGLang not yet).
- **Cannot fix a model that cannot answer:** on maze/Minesweeper tasks no readout beats trivial baselines.
- **L0 is not a free win everywhere:** the batch prior can cost accuracy when one label dominates.
- **High-variance estimates:** coverage at 5% risk measured at n=300.
- **No independent Jev comparison:** the repo cites TypeSafe's published figures without re-running them; AnyJev states it is not affiliated with TypeSafe AI.

---

## 7. Sources

- [Feriel Oulhadj — "Your LLM Classifiers Are Overconfident: Understanding Calibration with AnyJev" (Medium)](https://medium.com/@feriel.oulhadj/your-llm-classifiers-are-overconfident-understanding-calibration-with-anyjev-009d6b4f328b)
- [AnyJev — GitHub repository (Nokia Applied Research)](https://github.com/nokia-applied-research/AnyJev)
- [AnyJev — DeepWiki](https://deepwiki.com/nokia-applied-research/AnyJev)
- [AnyJev — PyPI](https://pypi.org/project/anyjev)
- [Jev — LLM Reference entry](https://www.llmreference.com/model/jev)
- [Jev by TypeSafe — innFactory](https://innfactory.ai/en/blog/jev-system-one-model-classifier-not-llm/)

---

## Decision Log

| Decision | Rationale | Date |
|---|---|---|
| Report AnyJev as a research item, not a dependency | Young project (days old), Qwen-centric, teacher-LLM ground truth; needs validation on our own golden set | 2026-09-28 |
| Recommend L0 spike experiment first | Zero labels, isolates position-bias correction; cheapest highest-signal test vs. current role-labeling approach | 2026-09-28 |
| Defer L2 | Requires stable taxonomy + Qwen-class local models; per-question/per-model head friction | 2026-09-28 |
