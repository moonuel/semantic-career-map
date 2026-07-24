# Semantic Career Mapping Platform

> A production-oriented machine learning system that models the ML job market as a high-dimensional semantic space and maps resumes into that space using modern embedding models, vector search, and clustering.
>
> **Status:** Phase 1 — 3 experiments complete (baseline, boilerplate removal, LLM extraction). Integrated documentation site deployed. Experiment loop in progress.
>
> **Documentation site:** [semantic-career-map](https://moonuel.github.io/semantic-career-map/) — live via GitHub Pages

---

## What This Project Does

Given an arbitrary resume, the system answers:

- Which job postings are most semantically similar?
- What does the ranking look like when scored by semantic similarity?

Rather than relying on keyword matching, resumes and job postings are mapped into a shared embedding space where semantic similarity can be computed. The project demonstrates end-to-end ML engineering: data pipeline → embedding model → vector search → API → deployment.

---

## Current Status

| Phase | Description | Status |
|---|---|---|---|
| 1 | Preprocessing & Embedding Optimization | 🟡 3 experiments complete (baseline, boilerplate removal, LLM extraction). LLM-cleaned text achieves best separation gap (+0.0493, 8.2× over raw). Experiment loop in progress. |
| 2 | Data Collection & Ingestion | 🟡 27 postings collected. Golden set hand-cleaned (5 postings). LinkedIn saturated for ML roles. Non-ML postings planned for contrast. |
| 3 | Embedding Pipeline & Similarity Engine | 🟡 Cosine similarity on L2-normalized vectors functional. Self-retrieval benchmark implemented. pytrec_eval integration next. |
| 4 | FastAPI Backend | 🔴 Not started |
| 5 | Docker Containerization | 🔴 Not started |
| 6 | Cloud Deployment | 🔴 Not started |
| 7 | Frontend | 🔴 Not started |
| 8 | Documentation Site | ✅ Integrated documentation site ([live](https://moonuel.github.io/semantic-career-map/)) with experiment log, architecture docs, results, and demo page. Built with Zensical, deployed via GitHub Actions to GitHub Pages. |
| 9 | CI/CD Pipeline | 🟡 GitHub Actions workflow deploys docs site on push to main. Full test + Docker build pipeline not yet started. |

**Collecting:** 27 ML/DS/AI job postings collected. Stored in `data/selected-job-postings/`. 

**Phase 1 experiments complete:**
- ✅ Steps 1.0–1.2: Parsed → embedded (all-MiniLM-L6-v2, L2-normalized) → PCA + UMAP + t-SNE visualized
- ✅ Step 1.4: Boilerplate removal experiment — section-filtered clean text embedding
- ✅ Step 1.4b: LLM-based text extraction — gpt-5.4-nano strips company culture/benefits/EEO, evaluated against hand-cleaned golden set (5 postings). Best separation gap: +0.0493 (8.2× over raw baseline)
- 🔜 Step 1.3: Proxy metrics formalization (self-retrieval, separation gap, NN audit, HDBSCAN cluster check)
- 🔜 Steps 1.5–1.7: Title canonicalization, skill extraction, weighted concatenation

**Documentation site:** The [integrated documentation site](https://moonuel.github.io/semantic-career-map/) ([source](site/)) includes architecture documentation, experiment logs (3 experiments), results with before/after metrics, technology choices, and a project overview. Built with Zensical (MkDocs successor), deployed to GitHub Pages via GitHub Actions on push to main.

**Planned: function-based role labels.** Current `role_category` labels are derived from raw job titles (Data Scientist, ML Engineer, etc.), but titles are noisy — a "Data Scientist" might do pure analytics while another builds production ML systems. A second labeling pass will assign each posting to a **function category** based on the actual work described:
- `data-engineering` — pipelines, ETL, infrastructure
- `exploratory-analysis` — A/B testing, dashboards, SQL-heavy analytics
- `model-development` — training, fine-tuning, experimentation
- `model-production` — deployment, MLOps, serving, monitoring
- `research` — novel methods, publications, prototyping
- `applied-ai` — building AI-powered products/features end-to-end

Each posting may map to 1–2 categories. These labels will be used to judge cluster quality during the experiment loop — if preprocessing improvements bring same-function postings closer together, the embedding space is capturing *what people actually do*, not just what their title says.

**Future data expansion:** Non-ML postings (SWE, PM, DevOps) will be collected later to provide contrast in the embedding space.

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
├── scripts/                    # Data pipeline, experiments, evaluation
├── site/                       # Documentation site source (Zensical/MkDocs)
│   ├── experiments/            # Experiment log (3 experiments)
│   ├── technical/              # Tech choices, evaluation, data processing
│   └── assets/images/          # PCA, UMAP, t-SNE visualizations
├── _build/                     # Built documentation site (deployed to GitHub Pages)
├── tests/                      # pytest tests
├── docs/                       # Planning and research documents
├── AGENTS.md                   # Instructions for AI coding agents
├── requirements.txt            # Python dependencies
├── mkdocs.yml                  # Zensical/MkDocs site configuration
├── .github/workflows/          # CI/CD — docs deployment
├── Dockerfile                  # Container definition (planned)
└── README.md                   # This file
```

The [documentation site](https://moonuel.github.io/semantic-career-map/) is integrated into the repository under `site/`. It's built with Zensical and auto-deployed to GitHub Pages on every push to main via `.github/workflows/docs.yml`.

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
| `site/` | Documentation site source — architecture, experiments, results, demo page |
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
