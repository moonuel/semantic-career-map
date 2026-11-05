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
- [x] Develop a **robust data pipeline** to improve embedding quality and downstream unsupervised learning performance
- [x] Establish a **rigorous experimental workflow** with documented hypotheses, evaluations, results, and design decisions
- [ ] Deploy the system through a secure API layer using FastAPI while protecting proprietary implementation details
- [ ] Containerize the web application and deploy the query service to a cloud environment
- [x] Publish an accessible demo for others to verify quality of work
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

## Current Status

Last code audit: **2026-11-05**. This table is an implementation index, not a live AWS probe.

| # | Deliverable | Status | Implemented surface |
|---|---|---|---|
| 1 | Data Ingestion & Lakehouse | ✅ Live | Ashby, Greenhouse, Lever, and dataroles registry; HiringCafe MCP and LinkedIn import are manual paths. Content-addressed raw JSON, ingestion manifests, and append-only run/dataset records. ATS Lambda runs at 04:00 UTC; curator Fargate task at 05:00 UTC. |
| 2 | Query Layer | ✅ Live | Glue database + Athena workgroup; `raw_<source>` tables, `manifests_ingestion`, `artifacts_cleaned`, and `postings_embeddings`; 256 MiB per-query scan cap; provision/gate commands in `scripts/provision_query_layer/`. |
| 3 | Cleaning & Evaluation | ✅ Live | Boilerplate-only Gemini 2.5 Flash Lite cleaning at 06:00 UTC; Athena anti-join selection bounded to 3 days / 5,000 rows; flush-watermark Parquet parts, artifact cache, orphan healing, fail-fast, and run records. Evaluation is a manual 39-reference golden gate over the exp-003 metrics. |
| 4 | Embedding Pipeline | ✅ Live | Amazon Titan V2 over Bedrock, 1,024-d and L2-normalized, scheduled at 07:00 UTC. One configuration per stable version via claims; `state/production.json` is the immutable-generation pointer; reconciliation heals duplicates and orphan checkpoints. MiniLM 384-d remains the rollback generation. |
| 5 | Clustering & Similarity | 🟡 Experimental | Cosine distance and the EVōC bake-off are implemented. Experiment 012 selected normalized Titan V2 and accepted its stability risk. There is no production clustering dataset or served cluster/label layer yet. |
| 6 | Map & Demo | ✅ Live | Interactive UMAP map, 3D coordinates, and lazy detail buckets are live. The published artifact contains **10,010 Titan V2 1,024-d** postings, generated from the production pointer's first embedding entry. |
| 7 | FastAPI Backend | 🔴 Not started | No resume upload, query, similarity-search, or jobs API. |
| 8 | Pipeline Containerization & AWS Deployment | ✅ Live | Three ARM64 images (`scrape`, `clean`, `embed`), Lambda ATS tier, three Fargate task definitions, four EventBridge schedules, SNS alerts, SQS DLQ, ECR, SSM secret, and IAM roles. This covers the pipeline, not a user-facing API. |
| 9 | Frontend & Documentation | 🟡 Partial | Docs site and embedded map are live. Resume-to-role search UI is not started. |
| 10 | CI/CD | 🟡 Partial | GitHub Actions runs `ruff` and `pytest` on Python 3.14, and docs builds/publishes to Pages. Image builds and AWS deployments remain manual. |

### System Index

| Subsystem | Primary code | Canonical docs |
|---|---|---|
| Ingestion and raw storage | `backend/ingestion/`, `backend/storage/raw.py`, `scripts/scrape.py` | [Data Architecture](architecture/data.md), [AWS Migration Roadmap](architecture/aws-migration-roadmap.md) |
| Cleaning and evaluation | `backend/pipeline/cleaning/`, `backend/pipeline/eval/` | [Cleaning Automation](architecture/cleaning-automation-report.md), [Data Processing](technical/data-processing.md) |
| Embedding and production control | `backend/pipeline/embedding/`, `backend/pipeline/production.py`, `scripts/embed.py` | [Embedding Pipeline](technical/embedding-pipeline.md), [Data Architecture](architecture/data.md) |
| Query layer | `backend/pipeline/athena.py`, `scripts/provision_query_layer/` | [Data Architecture](architecture/data.md), [Athena Lakehouse Layout](architecture/athena-layout.md) |
| Map and demo | `scripts/map.py`, `docs/demo.md`, `.github/workflows/docs.yml` | [Results](results.md), [Experiment 012](experiments/012-evoc-clustering-bakeoff.md) |
| Deployment | `scripts/provision_compute_stack/`, `deploy/` | [Deploy README](https://github.com/moonuel/semantic-career-map/blob/main/deploy/README.md) |

### Current Gaps

- Build the **FastAPI retrieval layer** and resume-to-role query surface.
- Turn the EVōC results into a production clustering/label dataset and serving contract.
- Make the cleaning **eval-before-promote gate** enforced rather than runbook prose ([#15](https://github.com/moonuel/semantic-career-map/issues/15)).
- Add automated map publication and provenance checks; the S3 sync is currently manual and the Pages workflow only checks that `index.html` and `details/` exist.
- The Glue `artifacts_cleaned` table is pinned to `version=v1`; changing the cleaning version requires re-registering the table.
