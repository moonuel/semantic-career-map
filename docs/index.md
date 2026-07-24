---
title: Introduction
---

<div style="text-align: center; padding: 3em 0 0.5em;" markdown="1">

# AI/ML Career Map

A semantic search tool for mapping your interests and experience to actual job families in AI and machine learning, based on skills and not title.

</div>

<!-- --- -->

<div class="grid cards" markdown>

-   **Problem**

    ---

    Traditional job searches struggle when *responsibilities don't match the title*. 
    
    A "Machine Learning Engineer" won't match an "Applied Scientist" role, even when the work is identical.

    How do you find the jobs families that are right for you? 

-   **Solution**

    ---

    Dense text embeddings provide *semantic clustering of job postings* based on responsibilities.

    Clustering and similarity rankings *map your interests to job families* based on skills and experience.

    Dimension-reduction tools provide *intuitive 2D visualizations* of high-dimensional spaces.

<!-- -   :material-pipe:{ .lg .middle } **System** -->

<!-- --- -->

<!-- Embedding pipeline → vector index → similarity search → API service. Precomputed embeddings with L2 normalization enable dot-product similarity at query time — no GPU required. -->

<!-- -   :material-chart-bar:{ .lg .middle } **Evaluation** -->

<!-- --- -->

<!-- Quantitative retrieval benchmarks: self-retrieval (100%), separation gap (+0.0493, LLM-cleaned). pytrec_eval integration planned for Precision@5, MRR, NDCG. -->

</div>

<div style="text-align: center; padding: 0.5em 0 0.5em;" markdown="1">

[Launch Demo](demo.md){ .md-button .md-button--primary }
<!-- [View Source Code](https://github.com/moonuel/semantic-career-map){ .md-button .md-button--warn } -->
</div>

<div style="text-align: center; padding: 0 0 2em;" markdown="1">

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-all--MiniLM--L6--v2-green)
![Scikit-learn](https://img.shields.io/badge/-scikit--learn-%23F7931E?logo=scikit-learn&logoColor=white)
![UMAP-learn](https://img.shields.io/badge/UMAP--learn-green)
![FastAPI](https://img.shields.io/badge/FastAPI-planned-lightgrey)
![Docker](https://img.shields.io/badge/Docker-planned-lightgrey)
![CPU-only](https://img.shields.io/badge/Inference-CPU--only-orange)

</div>
<!-- 
---

## Architecture Preview

```mermaid
graph LR
A[Documents] --> B[Preprocessing]
B --> C[Embedding Model<br/>all-MiniLM-L6-v2]
C --> D[Vector Storage<br/>numpy arrays]
D --> E[Query API<br/>FastAPI]
E --> F[Ranked Results]
```

[:material-arrow-right: Full architecture documentation](architecture.md)

---

## Current Status

| Phase | Description | Status |
|---|---|---|
| 1 | Preprocessing & Embedding Optimization | :material-progress-check: Baseline embedding + visualization complete (27 postings). Experiment loop next. |
| 2 | Data Collection & Ingestion | :material-progress-check: 27 postings collected. Non-ML postings planned for contrast. |
| 3 | Embedding Pipeline & Similarity Engine | :material-close: Not started |
| 4 | FastAPI Backend | :material-close: Not started |
| 5 | Docker Containerization | :material-close: Not started |
| 6 | Cloud Deployment | :material-close: Not started |
| 7 | Frontend | :material-close: Not started |
| 8 | CI/CD Pipeline | :material-close: Not started |
| 9 | Documentation & Polish | :material-close: Not started |

### Phase 1 Progress

- ✅ Steps 1.0–1.2: Parsed → embedded (all-MiniLM-L6-v2, L2-normalized) → PCA + UMAP visualized
- ✅ Steps 1.4–1.7: Experiment loop — baseline, boilerplate removal, LLM extraction complete
- 🔜 Step 1.3: Proxy metrics formalization, nearest-neighbor audit
- 🔜 Steps 1.6–1.7: Skill extraction, weighted concatenation experiments -->
