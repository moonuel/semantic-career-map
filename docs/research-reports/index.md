# Research Reports

Research log for exploratory reports that fuel experiments and the final design. Usually AI-generated and used to guide early prototypes and downstream agents.


<div class="grid cards" markdown>

-   [**001 — Embedding optimization**](001-embedding-optimization-research.md){ .md-button }

    **2026-07-22**

    Feeding raw data into a model is almost never optimal. Approximate techniques for data cleaning inputs are sufficient for initial validation. 

-   [**002 — Tutte Institute Tools**](002-tutte-institute-tool-review.md){ .md-button }

    **2026-07-22**

    I saw a cool talk by Dr. Valerie Poulin detailing the usage of LLMs to annotate branches in hierarchically-clustered, interactive UMAP embeddings. Would love to try it. 

-   [**003 — Taxonomy of ML/AI Responsibilities**](003-ml-ai-responsibility-taxonomy.md){ .md-button }

    **2026-07-22**

    I'm working with the hypothesis that jobs are more accurately grouped by responsibilities rather than title. An audit of ML/AI responsibilities was conducted and a taxonomy of roles constructed from that.

</div>

<!-- ## Results at a Glance

| Experiment | Separation Gap | Self-Retrieval | Key Finding |
|---|---|---|---|
| 001 — Baseline | +0.0060 | 100.0% | Boilerplate dilutes semantic signal |
| 002 — Boilerplate | +0.0203 | 100.0% | Regex too brittle for production |
| 003 — LLM | +0.0493 | 100.0% | LLM extraction is robust and effective | -->

## Planned Experiments

| # | Title | Variable | Step |
|---|---|---|---|
| 004 | Proxy Metrics | Formalize self-retrieval + separation gap, NN audit | 1.3 |
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
