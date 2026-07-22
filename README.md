# Semantic Career Mapping Platform

> A production-oriented machine learning system that models the ML job market as a high-dimensional semantic space and maps resumes into that space using modern embedding models, vector search, and clustering.
>
> **Status:** Phase 1 — Preprocessing & Embedding Optimization (Pre-Implementation)

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
| 1 | Preprocessing & Embedding Optimization | 🔴 Not started |
| 2 | Data Collection & Ingestion | 🟡 27 postings collected. LinkedIn saturated for ML roles. Will add non-ML postings later for contrast. |
| 3 | Embedding Pipeline & Similarity Engine | 🔴 Not started |
| 4 | FastAPI Backend | 🔴 Not started |
| 5 | Docker Containerization | 🔴 Not started |
| 6 | Cloud Deployment | 🔴 Not started |
| 7 | Frontend | 🔴 Not started |
| 8 | CI/CD Pipeline | 🔴 Not started |
| 9 | Documentation & Polish | 🔴 Not started |

**Collecting:** 27 ML/DS/AI job postings collected. LinkedIn no longer serves relevant roles at this volume. Stored in `data/selected-job-postings/`. Roles span: Data Scientist, ML Engineer, AI Engineer, Applied Researcher, AI Solutions Engineer, Performance Benchmarking Engineer, Data Engineering, and one banking DS/Analyst.

**Next:** `scripts/parse_postings.py` → `data/jobs.json` → baseline embedding via `scripts/embed_baseline.py`.

**Future data expansion:** Non-ML postings (SWE, PM, DevOps) will be collected later to provide contrast in the embedding space — the current all-ML dataset risks producing a homogeneous vector space where all postings are similar, making retrieval distinctions harder to evaluate.

### Data Status

- **Real job postings collected:** 27 (in `data/selected-job-postings/`)
- **Role breakdown:** Data Scientist (8), ML Engineer (5), AI Engineer (3), Applied Researcher (2), AI Solutions Engineer (1), Performance Benchmarking (1), Data Engineering (1), DS/Analyst (1), Data Analytics Intern (1), AI Intern (1), Consultant Analyst (1), ML Recruitment (1), Decision Scientist (1)
- **Target for Phase 2:** 27 (collected) — expanding to include non-ML roles in future
- **Target with augmentation:** ~350+

---

## Quick Start

> The project is in early pre-implementation. No code has been written yet.

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies (after requirements.txt is created)
pip install -r requirements.txt

# Run tests (after tests are written)
pytest tests/ -v
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
