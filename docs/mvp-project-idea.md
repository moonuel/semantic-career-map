# Semantic Career Mapping Platform — Hiring-Optimized MVP

> A production-ready ML system that maps resumes into a semantic embedding space of job postings, enabling fast similarity-based job retrieval. Designed for rapid completion and maximum hiring signal density.

---

# Vision

Most resume screening systems rely on keyword matching. This project demonstrates end-to-end ML engineering by building a working semantic search system that maps resumes and job postings into a shared embedding space.

Given an arbitrary resume, the system answers:
- Which job postings are most semantically similar?
- What does the ranking look like when scored by semantic similarity?

---

# Project Finish Line

The project is complete when all 9 core deliverables ship. Scope is explicitly locked to prevent feature creep.

---

# Core Deliverables

## 1. Data Ingestion Pipeline
- Collect 200–500 ML job postings from public sources
- Parse and extract: job title, company, description, required skills, seniority
- Normalize: whitespace, formatting, duplicate detection
- Store raw + processed documents in a structured format

## 2. Embedding Pipeline
- Integrate Sentence Transformers (single model: `all-MiniLM-L6-v2` or `bge-small`)
- Generate dense embeddings for all job postings
- Store embeddings efficiently (numpy arrays or lightweight vector DB)
- Document model selection rationale

## 3. Embedding Optimization & Feature Engineering
- **Text preprocessing before embedding:**
  - Skill keyword extraction (technologies, frameworks, tools)
  - Section segmentation: separate title, skills, and description text
  - Weighted concatenation: title appears 2×, skills section appears 1.5× before embedding
- **Embedding post-processing:**
  - L2-normalize all vectors for consistent cosine similarity
  - Optionally apply PCA whitening to decorrelate dimensions
- **Data augmentation (simulated postings):**
  - Template-based generation: take 50 real postings as seeds, swap technology stacks and seniority labels to produce 300+ synthetic variants
  - This densifies the embedding space without collecting thousands of real postings
  - Tag synthetic vs real postings in metadata for transparency
- **Evaluation metrics for optimization:**
  - Measure whether weighted concatenation improves pairwise similarity for same-job-family pairs
  - Compare retrieval quality (precision@k) with and without L2 normalization
- Document every optimization decision with before/after numbers

## 4. Similarity Scoring Engine
- Implement deterministic similarity scoring: cosine distance on embeddings
- Rank job postings by similarity to a query resume embedding
- Document scoring approach clearly in code

## 5. FastAPI Backend
- `/upload-resume` endpoint: accept resume (text or PDF), generate embedding, return ranked jobs
- `/search` endpoint: query by text, return top-k matches with scores
- `/jobs` endpoint: list all jobs with metadata
- Error handling and input validation
- Type hints throughout

## 6. Docker Containerization
- Dockerfile with reproducible Python environment
- Docker Compose if needed for multi-container setup
- Optimized image size (use slim base images)
- Test the build locally

## 7. Cloud Deployment
- Deploy to AWS (EC2 or Lightsail) OR GCP Cloud Run OR Azure Container Instances
- Expose public API with HTTPS
- Document deployment process step-by-step
- Verify endpoints work from outside the machine

## 8. Frontend + Documentation
- Simple single-page web interface (HTML/CSS/JS or lightweight React):
  - Resume upload form
  - Display ranked job results with similarity scores
  - Basic styling (functional, not polished)
- Comprehensive README including:
  - Architecture diagram (text-based or image)
  - Data sources and collection process
  - Model selection and rationale
  - API documentation (endpoints, examples)
  - Deployment instructions
  - Known limitations and future work

## 9. CI/CD Pipeline
- GitHub Actions workflow for automated testing and validation
- Run tests on every push and pull request
- Docker image build verification
- Automated deployment to cloud platform on main branch
- Status badge in README showing current CI status
- Optional: integrate with deployment (auto-deploy on successful build)

---

# Deferred to "Future Work" Section

These are explicitly listed as future improvements in the README, not blockers:

- Multiple embedding model comparison
- Clustering and job family discovery
- **Non-ML postings for contrast:** Collect 10–20 non-ML job postings (SWE, PM, DevOps, Data Analyst) to diversify the embedding space. An all-ML dataset risks homogeneity — all vectors cluster together, making retrieval distinctions between ML sub-roles less meaningful.
- Resume parsing and skill extraction
- Classification layer (job family prediction)
- **ONNX Runtime export:** Convert the PyTorch Sentence Transformer model to ONNX for 2–3× inference speedup and ~75% smaller Docker image (ONNX Runtime is C++ with Python bindings — eliminates torch as a dependency). Requires one-time model export and manual tokenization + pooling implementation. See `docs/embedding-optimization-research.md` for analysis.
- **GPU-accelerated inference:** Enable CUDA for batch encoding large datasets (10K+ postings). Query-time benefit is negligible at current scale (~20ms CPU vs ~5ms GPU) but becomes essential if a cross-encoder reranker is added to the retrieval pipeline.
- User accounts and persistent storage
- Resume improvement suggestions
- Skill gap analysis
- Advanced reranking strategies
- Interactive UMAP visualization
- Recruiter search interface
- Salary estimation
- Additional CI/CD features (e.g., load testing, performance benchmarking, staging environment)

---

# Implementation Plan

### Phase 1: Domain Exploration (1–2 days)
- Manually collect 200–500 job postings
- Identify common job titles, companies, skill clusters
- Create initial taxonomy (optional, for README context)

### Phase 2: Data Engineering (2–3 days)
- Build ingestion scripts
- Parse and normalize job postings
- Create versioned dataset snapshot
- Store in simple format (CSV, JSON, or SQLite)

### Phase 3: Embedding + Feature Engineering (3–4 days)
- Integrate Sentence Transformers
- Implement text preprocessing: skill extraction, section segmentation, weighted concatenation
- Generate base embeddings for dataset
- Apply L2 normalization and optionally PCA whitening
- Build data augmentation pipeline: template-based synthetic posting generation
- Build similarity search function
- Evaluate optimization decisions with before/after metrics
- Test with manual queries

### Phase 4: Backend API (1–2 days)
- Implement FastAPI endpoints
- Resume upload handling (text + PDF support)
- Ranking and response formatting
- Local testing

### Phase 5: Docker (1 day)
- Write Dockerfile
- Build and test locally
- Verify dependencies are correct

### Phase 6: Cloud Deployment (1–2 days)
- Choose cloud platform (AWS Lightsail recommended for simplicity)
- Deploy container
- Configure DNS if desired
- Test public endpoints

### Phase 7: Frontend + Documentation (2–3 days)
- Build simple web interface
- Write comprehensive README
- Document architecture decisions
- Create API examples
- List future improvements clearly

### Phase 8: CI/CD Pipeline (1 day)
- Create GitHub Actions workflow file (`.github/workflows/`)
- Configure tests to run on push/PR
- Set up Docker image build in CI
- Configure automated deployment (if using cloud platform with easy integration)
- Add CI status badge to README
- Test workflow end-to-end

**Total estimated time: 12–19 days of focused work**

---

# Hiring Signal Strategy

This project demonstrates:

1. **Data engineering** — sourcing, parsing, normalizing real-world data
2. **NLP fundamentals** — understanding embedding models and their strengths
3. **Feature engineering** — text preprocessing, section weighting, L2 normalization, data augmentation
4. **Information retrieval** — similarity scoring, ranking, evaluation metrics (precision@k)
5. **Backend API design** — RESTful endpoints, error handling, type safety
6. **Software architecture** — modular, maintainable code with clear abstractions
7. **DevOps fundamentals** — Docker, containerization, cloud deployment
8. **CI/CD and testing** — automated testing, build verification, deployment pipelines
9. **Documentation and communication** — clear README explaining decisions and trade-offs

Each capability appears in a **single shipped artifact**, which is stronger than isolated notebooks demonstrating each component separately.

The CI/CD pipeline is particularly visible: a **green status badge** in the README signals to recruiters that code is actively tested and production-ready, not just a weekend project.

---

# Success Criteria

- [ ] Data pipeline runs without errors and produces a normalized dataset
- [ ] Data augmentation produces 300+ synthetic postings with varied tech stacks and seniority
- [ ] Embeddings generate successfully for all postings (real + synthetic)
- [ ] L2 normalization is applied and verified (all vectors have unit norm)
- [ ] Feature engineering decisions are documented with before/after metrics
- [ ] API returns ranked results in <2 seconds (CPU acceptable)
- [ ] Frontend loads and allows resume upload
- [ ] Docker image builds and runs locally
- [ ] Deployed API is publicly accessible and works
- [ ] README includes architecture, decisions, and deployment instructions
- [ ] Code is readable, type-hinted, and reasonably tested
- [ ] CI/CD pipeline runs on push/PR and shows passing tests
- [ ] CI status badge displays in README and reflects current build state
- [ ] Deployment is automated or one-click from CI dashboard

---

# Concrete Architecture

```
semantic-career-map/
├── backend/
│   ├── api.py                 # FastAPI app, endpoints
│   ├── embeddings.py          # Embedding generation + L2 normalization
│   ├── retrieval.py           # Similarity search + precision@k
│   ├── preprocessing.py       # Text prep: skill extraction, weighting
│   ├── resume_parser.py       # Resume text extraction
│   └── config.py              # Configuration
│
├── data/
│   ├── jobs.json              # Normalized job postings
│   ├── jobs_augmented.json    # Real + synthetic job postings
│   ├── embeddings.npy         # Precomputed embeddings
│   └── metadata.json          # Job metadata (ID, title, company, etc)
│
├── scripts/
│   ├── collect_jobs.py        # Data collection
│   ├── augment_jobs.py        # Synthetic posting generation
│   ├── generate_embeddings.py # Precompute embeddings
│   └── test_api.py            # API testing script
│
├── tests/
│   ├── test_retrieval.py
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions workflow
│
├── Dockerfile
├── requirements.txt
├── docker-compose.yml (optional)
├── README.md
└── .gitignore
```

---

# Key Design Decisions

1. **Single embedding model** — reduces scope, focuses on shipping. Model choice is documented; comparison is future work.

2. **Feature engineering over model comparison** — spend the optimization budget on text preprocessing, normalization, and data augmentation rather than benchmarking multiple models. These are standard ML practices that demonstrate depth without scope creep.

3. **Data augmentation for embedding density** — 200–500 real postings is a sparse embedding space. Template-based synthetic postings fill the gaps, making retrieval more robust. Real vs synthetic is always tagged in metadata for transparency.

4. **No user accounts or persistence** — stateless API is simpler to deploy and maintain. Can be added later.

5. **Precomputed embeddings** — embeddings are generated once during setup, not at inference time. Faster queries, simpler deployment.

6. **Simple frontend** — demonstrates web integration without frontend complexity. Shows full-stack thinking without derailing the project.

7. **Focus on README** — clear documentation of decisions, architecture, and deployment is as important as the code. Signals maturity and professionalism.

8. **CI/CD from day one** — GitHub Actions is free and visible. A passing CI badge in the README demonstrates professional practices and gives recruiters confidence the code works.

---

# CI/CD Implementation Details

## GitHub Actions Workflow (Minimal)

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest
      - run: pytest tests/

  build-docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: docker/setup-buildx-action@v2
      - run: docker build -t semantic-career-map .
```

## README Status Badge

Add to the top of README.md:

```markdown
![CI Status](https://github.com/YOUR_USERNAME/semantic-career-map/actions/workflows/ci.yml/badge.svg)
```

Replace `YOUR_USERNAME` with your actual GitHub username. The badge turns green when tests pass, red when they fail. It's immediately visible to recruiters and demonstrates that code is actively tested.

## What This Signals

- **Professional practices** — CI/CD is standard in industry
- **Confidence in code** — passing tests signal the project is maintainable
- **Continuous improvement** — the workflow runs on every push, not just once
- **Visible quality** — the badge makes test status prominent in the README

---

# Concurrent Job Search

Build this project **while applying for jobs**, not instead of applying. Spend 2–3 hours/day on development and 2–3 hours/day on targeted applications.

This ensures:
- You maintain a pipeline of interviews even while coding
- You're not investing months in a project that becomes "almost done" forever
- By the time interviews are scheduled, the project is likely complete
- The portfolio artifact arrives at the right hiring moment

---

# Future Work Section (for README)

Include a "Future Improvements" section in the README that mentions:
- Clustering and job family discovery
- Multiple embedding model comparison
- GPU-accelerated inference
- Resume skill extraction and gap analysis
- Classification layer for job family prediction
- Interactive visualization of embedding space
- Recruiter-facing search interface

This demonstrates forward thinking and prevents the README from feeling like an incomplete project.

---

# Success Story

Feature engineering and data augmentation show real depth. A recruiter browsing the project sees:

1. A **green CI badge** at the top of the README — immediate signal of quality
2. "This person didn't just dump text into an API — they thought about preprocessing, weighting, and normalization"
3. "They identified a real problem (sparse data) and solved it with a practical technique (template-based augmentation)"
4. "They understood the project and shipped a working solution in a reasonable timeframe"
5. "They made deliberate trade-offs (e.g., single model, no GPU) to focus on shipping"
6. "The code is clean, the API works, and it's deployed"
7. "They documented their decisions clearly, including before/after metrics for their optimizations"
8. "Tests run automatically; the system is production-ready"


A project with thoughtful feature engineering + CI/CD + deployment is rare in portfolios and impossible to ignore.

That's hiring-signal-dense.
