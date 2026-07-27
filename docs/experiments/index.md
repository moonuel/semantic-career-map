# Experiment Log

Experiment log tracking the development of the semantic embedding pipeline for job posting matching. Results inform the next step and final architecture. Hand-written with love.

**Experiments follow a consistent methodology:**

    1. Establish context and hypothesis
    2. Document experimental design
    3. Document results and discussion
    4. Write concrete conclusions and next-steps

<!-- ## Planned progression:

### Job posting embeddings
```
Raw text baseline → LLM text extraction → Feature engineering → Clustering evaluation 
```

### User input embeddings
```
Raw user input baseline → LLM feature engineering → Embedding optimization → Classification and ranking
``` -->

## Completed Experiments

<div class="grid cards" markdown>

-   [**001 — Baseline Embedding**](001-baseline-embedding.md){ .md-button }

    **2026-07-22**

    Initial pipeline established. 27 postings collected; weak embedding clusters; likely noisy text from boilerplate contamination.

-   [**002 — Regex boilerplate Removal**](002-boilerplate-removal.md){ .md-button }

    **2026-07-22**

    Improved separation of clusters (+0.0203, 3.4×) but brittle regex approach expected to fail on more general job posting formats.

-   [**003 — LLM Extraction**](003-llm-extraction.md){ .md-button }

    **2026-07-23**

    +0.0493 separation gap of clusters (8.2×), no embedding degeneracy, zero hallucinations on 5 datasets. Generality on further postings to be tested, as well as LLM evals. 

-   [**004 — GPT vs DeepSeek Speed Benchmark**](004-gpt-deepseek-benchmark.md){ .md-button }

    **2026-07-27**

    `deepseek-v4-flash` is competitive with `gpt-5.4-nano` on speed and cheaper. Likely top candidate for cost-sensitive text extraction work.

</div>

<!-- ## Results at a Glance

| Experiment | Separation Gap | Self-Retrieval | Key Finding |
|---|---|---|---|
| 001 — Baseline | +0.0060 | 100.0% | Boilerplate dilutes semantic signal |
| 002 — Boilerplate | +0.0203 | 100.0% | Regex too brittle for production |
| 003 — LLM | +0.0493 | 100.0% | LLM extraction is robust and effective |
| 004 — GPT vs DeepSeek | — | — | DeepSeek V4 Flash is competitive on speed at half cost | -->

## Planned Experiments

| # | Title | Variable | Step |
|---|---|---|---|
| 005 | Proxy Metrics | Formalize self-retrieval + separation gap, NN audit | 1.3 |
| ? | Skill Extraction | spaCy PhraseMatcher with 200-term ML vocabulary | 1.6 |
| ? | Weighted Concatenation | Title 3×, skills 2×, body 1× vs multi-field fusion | 1.7 |
| ? | Golden Set Evaluation | pytrec_eval: Precision@5, MRR, NDCG | 1.9 |
| ? | Data Augmentation | Template-based synthetic posting generation | 1.10 |

<!-- ## Methodology

Each experiment follows a strict protocol:

1. **Define hypothesis** — what change is being tested and why
2. **Implement change** — modify only one variable
3. **Re-embed** — run the same embedding pipeline with the changed input
4. **Measure delta** — compare before/after with self-retrieval and separation gap
5. **Document decision** — adopt, reject, or iterate

!!! info "Current Phase"
    The project is in Phase 1 (Preprocessing & Embedding Optimization). See the [project overview](../project.md) for full scope details. -->
