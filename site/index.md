---
title: Semantic Career Mapping
---

<div style="text-align: center; padding: 3em 0 2em;">

# Vector Embedding Search Platform

A production-oriented semantic search system built with modern embedding models, vector retrieval, and evaluation pipelines.

[Launch Demo](demo.md){ .md-button .md-button--primary }
[View Source Code](https://github.com/moonuel/semantic-career-map){ .md-button }

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-all--MiniLM--L6--v2-green)
![FastAPI](https://img.shields.io/badge/FastAPI-planned-lightgrey)
![Docker](https://img.shields.io/badge/Docker-planned-lightgrey)
![CPU-only](https://img.shields.io/badge/Inference-CPU--only-orange)

</div>

---

<div class="grid cards" markdown>

-   :material-magnify:{ .lg .middle } **Problem**

    ---

    Traditional keyword search fails when meaning differs from wording. A "machine learning engineer" won't match an "applied scientist" role, even when the work is identical.

-   :material-vector-line:{ .lg .middle } **Solution**

    ---

    Semantic retrieval using dense vector representations. Resumes and job postings are mapped into a shared 384-dimensional embedding space where cosine similarity captures meaning, not just keywords.

-   :material-pipeline:{ .lg .middle } **System**

    ---

    Embedding pipeline → vector index → similarity search → API service. Precomputed embeddings with L2 normalization enable dot-product similarity at query time — no GPU required.

-   :material-chart-bar:{ .lg .middle } **Evaluation**

    ---

    Quantitative retrieval benchmarks: self-retrieval (100%), separation gap (+0.0493, LLM-cleaned). pytrec_eval integration planned for Precision@5, MRR, NDCG.

</div>

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
- 🔜 Steps 1.6–1.7: Skill extraction, weighted concatenation experiments
