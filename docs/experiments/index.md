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

-   [**005 — Semantic Partitioning**](005-semantic-partitioning.md){ .md-button }

    **2026-07-27**

    Can LLMs partition text into distinct semantic dimensions?
    This experiment suggests a positive result, using manually-tuned test sets and LLM-generated partitions. 
    Job-context can be effectively extracted, but role-context struggles.
    Clustering analysis reinforces the partitioning results. 

-   [**006 — Taxonomy-Driven Role Labeling**](006-taxonomy-role-labeling.md){ .md-button }

    **2026-08-04**

    LLM assigns fine-grained function tags from a 13-category ML/AI responsibility taxonomy to 27 job postings. 22 unique tag sets with 80% golden recall. `deep-learning` + `mlops-production` is the dominant pair (10 co-occurrences). Qualitative UMAP evaluation shows tag-set-colored clusters align with embedding similarity.

-   [**007 — Two-Track LLM Latency Benchmark**](007-llm-latency-benchmark.md){ .md-button }

    **2026-10-19**

    20 cheap instruct models benchmarked on two tracks: streaming TTFT (micro-prompt) and non-streaming fixed-length copy task. The round-1 `empty response` fault class is confirmed streaming-only; `nemotron-3-nano-30b-a3b` and `gemini-2.5-flash-lite` lead on speed with clean failure records; no model actually copied the reference exactly. **Adoption finalized 2026-10-23:** the winning lane became the production cleaning configuration — see the adoption section (sample eval means: Jaccard 0.958, over-deletion 0.9%, zero hallucinations).

-   [**008 — Boilerplate-Only LLM Cleaning**](008-boilerplate-only-cleaning.md){ .md-button }

    **2026-10-20**

    Corrected boilerplate-only prompt fixes 003's over-deletion at scale (    22.8% → 2.3–5.2% on one model) on a 39-posting hand-curated golden set. Prompt v3, distilled from human strip behavior, lifts Jaccard 0.826 → 0.858. Deploy recommendation: `gemini-2.5-flash-lite` (Jaccard 0.894, $0.00036/posting). **Deployed as production 2026-10-23** — see the [landing update](../technical/data-processing.md#step-2-llm-cleaning).

-   [**009 — PCA Informativeness**](009-pca-informativeness.md){ .md-button }

    **2026-10-29**

    PCA pre-reduction benchmark on the full 9,354×384 production corpus: participation ratio 57.6, permutation signal count 71, cumEVR ≥ 0.95 only at ~192 dims. The pre-registered gate returns `k_d* = None` — the k_d=384 control isolated PCA's centering shift as the dominant turnover (42.5% of cosine 15-NN neighborhoods), dwarfing truncation loss (95% of surviving neighborhoods kept at 128 dims). Report 022's `n_dims` default stays unset pending a centered-frame/uncentered-truncation variant.

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
| 010 | UMAP Subsample-Fit Stability | Fit sample size m × seeds; trustworthiness vs full fit | 1.11 (job map) |
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
