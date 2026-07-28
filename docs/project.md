# Project Overview

## Motivation

In the current AI/ML job market, the same job title can describe ***vastly different responsibilities***. It can be difficult to identify which roles are **truly relevant** to your skill set.

This tool is designed to help (me) bridge this gap by using semantic search to match my **interests, skills, and experiences** to job families in AI and machine learning, beyond title.

It also fills an experience gap I've been looking to fill, by deploying an **end-to-end ML engineering** project demonstrating my research experience, engineering skill, and product thinking through accessible artifacts and ***comprehensive documentation***.

---

## Hypothesis

The underlying hypothesis of this work is that **rigorous use of LLMs** for data cleaning and feature engineering can turn unstructured job postings and resumes into **strong semantic signals** that can be **effectively clustered** and used for classification and ranking with classical ML and dimensionality reduction.

---

## Goals

<!-- ### Primary -->

- [ ] Build an end-to-end **semantic search system** for navigating AI/ML career opportunities
- [ ] Develop a **robust data pipeline** to improve embedding quality and downstream unsupervised learning performance
- [x] Establish a **rigorous experimental workflow** with documented hypotheses, evaluations, results, and design decisions
- [ ] Deploy the system through a secure API layer using FastAPI while protecting proprietary implementation details
- [ ] Containerize the application and deploy the service to a cloud environment
- [ ] Publish an accessible demo for others to verify quality of work
- [x] Provide **comprehensive documentation** of research and development, system architecture, and engineering decisions

<!-- ### Non-Goals

- [x] Training foundation models — out of scope
- [x] Building a general search engine — focused on job posting retrieval
- [x] User accounts or persistent storage — MVP is stateless
- [x] Multiple embedding model comparison — single model, documented rationale
- [x] GPU-accelerated inference — CPU latency is ~20 ms, well below threshold -->

---

<!-- ## System Capabilities

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

--- -->

<!-- ## Planned Evolution

```
Research prototype: baseline embedding + visualization (2026-07-22)
        ↓
Experiment loop: text extraction, feature engineering, clustering [current]
        ↓
Production pipeline: pre-processing, embedding, similarity ranking, classification
        ↓
FastAPI backend + Docker + Cloud deployment
```

--- -->

## Current Status

| # | Deliverable | Status |
|---|---|---|
| 1 | Data Ingestion Pipeline | 🟡 27 postings collected, expanding |
| 2 | Embedding Pipeline | ✅ all-MiniLM-L6-v2, L2-normalized |
| 3 | Embedding Optimization & Feature Engineering | 🟡 4 experiments complete, more planned |
| 4 | Clustering and Similarity Scoring | 🟡 Cosine distance implemented |
| 5 | FastAPI Backend | 🔴 Not started |
| 6 | Docker Containerization | 🔴 Not started |
| 7 | Cloud Deployment | 🔴 Not started |
| 8 | Frontend + Documentation | 🟡 Documentation site in progress |
| 9 | CI/CD Pipeline | 🔴 Not started |

<!-- ## Data Status

- **Real job postings collected:** 27 (in `data/selected-job-postings/`)
- **Role breakdown:** Data Scientist (8), ML Engineer (5), AI Engineer (3), Applied Researcher (2), plus 9 additional roles
- **Target with augmentation:** ~350+ postings -->
