# Semantic Career Mapping Platform

> A production-oriented machine learning system that models the ML job market as a high-dimensional semantic space and maps resumes into that space using modern embedding models, vector search, and clustering.
>
> **Status:** Phase 1 — Baseline embedding + visualization complete. Proxy metrics + experiment loop next.

---

## What This Project Does

Given an arbitrary resume, the system answers:

- Which job postings are most semantically similar?
- What does the ranking look like when scored by semantic similarity?

Rather than relying on keyword matching, resumes and job postings are mapped into a shared embedding space where semantic similarity can be computed. The project demonstrates end-to-end ML engineering: data pipeline → embedding model → vector search → API → deployment.

---

## Current Status

| Phase | Description | Status |
|---|---|---|
<<<<<<< HEAD
| 1 | Preprocessing & Embedding Optimization | 🟡 Baseline embedding + visualization complete (27 postings). Experiment loop next. |
| 2 | Data Collection & Ingestion | 🟡 27 postings collected. LinkedIn saturated for ML roles. Will add non-ML postings later for contrast. |
=======
| 1 | Preprocessing & Embedding Optimization | 🔴 Not started |
| 2 | Data Collection & Ingestion | 🟡 In progress — collecting from LinkedIn |
>>>>>>> 26e55410ac3d7b39d7a733a4a20cb926d71c81a5
| 3 | Embedding Pipeline & Similarity Engine | 🔴 Not started |
| 4 | FastAPI Backend | 🔴 Not started |
| 5 | Docker Containerization | 🔴 Not started |
| 6 | Cloud Deployment | 🔴 Not started |
| 7 | Frontend | 🔴 Not started |
| 8 | CI/CD Pipeline | 🔴 Not started |
| 9 | Documentation & Polish | 🔴 Not started |

<<<<<<< HEAD
**Collecting:** 27 ML/DS/AI job postings collected. Stored in `data/selected-job-postings/`. 

**Phase 1 progress:**
- ✅ Steps 1.0–1.2: Parsed → embedded (all-MiniLM-L6-v2, L2-normalized) → PCA + UMAP visualized
- 🔜 Step 1.3: Proxy metrics (self-retrieval, separation gap, nearest-neighbor audit)
- 🔜 Steps 1.4–1.7: Preprocessing experiment loop (boilerplate → skills → weighted concat)

**Planned: function-based role labels.** Current `role_category` labels are derived from raw job titles (Data Scientist, ML Engineer, etc.), but titles are noisy — a "Data Scientist" might do pure analytics while another builds production ML systems. A second labeling pass will assign each posting to a **function category** based on the actual work described:
- `data-engineering` — pipelines, ETL, infrastructure
- `exploratory-analysis` — A/B testing, dashboards, SQL-heavy analytics
- `model-development` — training, fine-tuning, experimentation
- `model-production` — deployment, MLOps, serving, monitoring
- `research` — novel methods, publications, prototyping
- `applied-ai` — building AI-powered products/features end-to-end

Each posting may map to 1–2 categories. These labels will be used to judge cluster quality during the experiment loop — if preprocessing improvements bring same-function postings closer together, the embedding space is capturing *what people actually do*, not just what their title says.

**Future data expansion:** Non-ML postings (SWE, PM, DevOps) will be collected later to provide contrast in the embedding space.
=======
**Collecting:** Targeting 50–200 ML/DS/AI job postings from LinkedIn. Currently at 7 in `data/selected-job-postings/`. Format: markdown files per posting. After collection → `scripts/parse_postings.py` → `data/jobs.json` → baseline embedding via `scripts/embed_baseline.py`.
>>>>>>> 26e55410ac3d7b39d7a733a4a20cb926d71c81a5

### Data Status

- **Real job postings collected:** 27 (in `data/selected-job-postings/`)
- **Role breakdown:** Data Scientist (8), ML Engineer (5), AI Engineer (3), Applied Researcher (2), AI Solutions Engineer (1), Performance Benchmarking (1), Data Engineering (1), DS/Analyst (1), Data Analytics Intern (1), AI Intern (1), Consultant Analyst (1), ML Recruitment (1), Decision Scientist (1)
- **Target for Phase 2:** 27 (collected) — expanding to include non-ML roles in future
- **Target with augmentation:** ~350+

---

## Quick Start

```bash
# Create virtual environment and install dependencies (uses uv)
uv venv && uv sync

# Download spaCy model (one-time)
uv run spacy download en_core_web_sm

# Parse all postings, embed, and visualize
uv run python scripts/parse_and_embed_quickstart.py

# Run tests (after tests are written)
uv run pytest tests/ -v
```

---

## Architecture

```
semantic-career-map/
├── backend/                    # FastAPI app, embeddings, retrieval
├── data/                       # Job postings, embeddings, golden set
├── frontend/                   # Simple web UI
├── scripts/                    # Data pipeline, experiments, evaluation
├── tests/                      # pytest tests
├── docs/                       # Planning and research documents
├── AGENTS.md                   # Instructions for AI coding agents
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container definition
└── README.md                   # This file
```

See `docs/mvp-project-idea.md` for the full project scope and design decisions.

---

## Documentation Index

| Document | Purpose |
|---|---|---|
| `docs/mvp-project-idea.md` | Project scope, deliverables, finish line |
| `docs/implementation-plan.md` | Detailed phase-by-phase implementation plan |
| `docs/embedding-optimization-research.md` | Research report on IR embedding optimization |
| `docs/tutte-institute-tool-review.md` | Evaluation of Tutte Institute tools (UMAP, HDBSCAN, DataMapPlot, EVōC, Toponymy, etc.) for this project |
| `docs/initial-project-idea.md` | Original project vision (historical reference) |
| `AGENTS.md` | Operational instructions for AI coding agents |

---

## Key Design Decisions

1. **Single embedding model** (`all-MiniLM-L6-v2`) — reduces scope, focuses on shipping
2. **CPU-only inference** — avoids GPU complexity; latency is acceptable
3. **Precomputed embeddings** — generated once at setup, faster queries
4. **Stateless API** — no user accounts or persistence in MVP
5. **Template-based data augmentation** — densifies embedding space without collecting thousands of real postings
6. **Simple frontend** — HTML/CSS/JS, no framework overhead
7. **CI/CD from day one** — GitHub Actions with status badge in README
8. **No ONNX or GPU yet** — profiled the pipeline; PyTorch CPU is ~20ms per embedding and the Docker image is a one-time concern. ONNX export and CUDA are documented as high-impact future improvements with clear activation conditions (see Future Work).

---

## Future Work

**Retrieval & Model Quality**
- Clustering and job family discovery (planned: **EVōC** + **Toponymy** from Tutte Institute — see `docs/tutte-institute-tool-review.md`)
- Multiple embedding model comparison
- Hybrid BM25 + dense retrieval
- Cross-encoder reranker for improved top-k precision
- Resume skill extraction and gap analysis
- Classification layer for job family prediction

**Performance Optimization**

These were consciously deferred after profiling — current latency is acceptable:

- **ONNX Runtime export:** Convert the Sentence Transformer model from PyTorch to ONNX. Gains: 2–3× inference speedup, ~75% smaller Docker image (eliminates torch dependency). One-time export cost, then drop-in replacement via `onnxruntime`. C++ inference core with Python bindings — zero Rust needed at this scale.
- **GPU-accelerated inference:** Enable CUDA for batch encoding of 10K+ postings. Not justified at current scale — `all-MiniLM-L6-v2` encodes a posting in ~20ms on CPU, well below any user-perceptible threshold. Would matter if a cross-encoder reranker were added to the query path.

**User-Facing**
- Interactive UMAP visualization of embedding space (planned: **DataMapPlot** from Tutte Institute)
- Recruiter-facing search interface
- Salary estimation

---

## License

MIT
