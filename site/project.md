# Project Overview

## Problem Statement

Most resume screening systems rely on keyword matching. A "machine learning engineer" won't match an "applied scientist" role, even when the work is identical. This project demonstrates end-to-end ML engineering by building a working semantic search system that maps resumes and job postings into a shared embedding space.

Given an arbitrary resume or text query, the system answers:

- Which job postings are most semantically similar?
- What does the ranking look like when scored by semantic similarity?

---

## Goals

### Primary

- [x] Build an end-to-end semantic search system
- [ ] Evaluate embedding quality empirically with proxy metrics and golden sets
- [ ] Deploy an interactive retrieval service via FastAPI
- [ ] Containerize and deploy to cloud

### Non-Goals

- [x] Training foundation models — out of scope
- [x] Building a general search engine — focused on job posting retrieval
- [x] User accounts or persistent storage — MVP is stateless
- [x] Multiple embedding model comparison — single model, documented rationale
- [x] GPU-accelerated inference — CPU latency is ~20 ms, well below threshold

---

## System Capabilities

<div class="grid cards" markdown>

-   :material-database-arrow-right-outline:{ .lg .middle } **Embedding Generation**

    ---

    Convert job posting text into dense 384-dimensional vector representations using `all-MiniLM-L6-v2`. L2 normalization ensures cosine similarity reduces to dot product — the fastest similarity metric.

-   :material-magnify-scan:{ .lg .middle } **Similarity Search**

    ---

    Retrieve semantically related job postings via cosine similarity on precomputed embeddings. Rank by match score regardless of keyword overlap.

-   :material-chart-box-outline:{ .lg .middle } **Evaluation**

    ---

    Proxy metrics (self-retrieval, separation gap) for fast iteration during experiment loop. IR metrics (Precision@K, MRR, NDCG) planned with pytrec_eval and golden set.

-   :material-cloud-upload-outline:{ .lg .middle } **Deployment**

    ---

    FastAPI backend with `/upload-resume`, `/search`, and `/jobs` endpoints. Docker containerization and cloud deployment planned.

</div>

---

## Project Evolution

```
Research prototype (2026-07-22)
        ↓
Baseline embedding + visualization
        ↓
Experiment loop: boilerplate → LLM extraction  [current]
        ↓
Production pipeline: skill extraction, weighted concatenation, augmentation
        ↓
FastAPI backend + Docker + Cloud deployment
```

---

## Project Scope

| # | Deliverable | Status |
|---|---|---|
| 1 | Data Ingestion Pipeline | 🟡 27 postings collected, expanding |
| 2 | Embedding Pipeline | ✅ all-MiniLM-L6-v2, L2-normalized |
| 3 | Embedding Optimization & Feature Engineering | 🟡 3 experiments complete, 4 planned |
| 4 | Similarity Scoring Engine | 🟡 Cosine distance implemented, pytrec_eval planned |
| 5 | FastAPI Backend | 🔴 Not started |
| 6 | Docker Containerization | 🔴 Not started |
| 7 | Cloud Deployment | 🔴 Not started |
| 8 | Frontend + Documentation | 🟡 Documentation site in progress |
| 9 | CI/CD Pipeline | 🔴 Not started |

## Data Status

- **Real job postings collected:** 27 (in `data/selected-job-postings/`)
- **Role breakdown:** Data Scientist (8), ML Engineer (5), AI Engineer (3), Applied Researcher (2), plus 9 additional roles
- **Target with augmentation:** ~350+ postings
