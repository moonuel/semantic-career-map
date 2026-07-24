# Website Scaffold Plan — Semantic Career Mapping Platform

## Context

- **SSG:** Zensical (Material for MkDocs derivative), already configured in `zensical.toml` with `docs_dir = "site"`
- **Goal:** Replace existing Zensical nav with the proposed structure, scaffold all pages, auto-populate from existing docs
- **`site/` directory does not exist yet** — all files must be created from scratch
- **Content sources available:** `README.md`, `docs/experiment-notes.md`, `docs/implementation-plan.md`, `docs/research-reports/embedding-optimization-research.md`, `scripts/comp_embedding_variants.py`, `data/golden_cleaned.json`, `docs/mvp-project-idea.md`

## Phase 1: Directory Scaffold

Create the file tree:

```
site/
├── index.md                                # Home
├── project.md                              # Project Overview
├── demo.md                                 # Demo
├── architecture.md                         # Architecture
├── results.md                              # Results
├── technical/
│   ├── embedding-pipeline.md               # Technical → Embedding Pipeline
│   ├── data-processing.md                  # Technical → Data Processing
│   ├── evaluation.md                       # Technical → Evaluation
│   └── tech-choices.md                     # Technical → Technology Choices
├── experiments/
│   ├── index.md                            # Experiment Log Overview
│   ├── 001-baseline-embedding.md           # Experiment 001
│   ├── 002-boilerplate-removal.md          # Experiment 002
│   └── 003-llm-extraction.md              # Experiment 003
└── assets/
    └── images/
```

**Tasks:**
1. Create `site/` directory and all subdirectories (`technical/`, `experiments/`, `assets/images/`)
2. Create placeholder `.md` files for every page listed above (content populated in Phase 2)

## Phase 2: Auto-Populate Pages from Existing Content

### 2.1 Source → Target Mapping

| Source File | Section | Target Page | Content to extract |
|---|---|---|---|
| `README.md` | Lines 1–14 | `index.md` | Hero text, one-sentence description, what-it-does |
| `README.md` | Lines 22–33 | `index.md` | Current status table |
| `README.md` | Lines 80–94 | `architecture.md` | Architecture directory tree |
| `README.md` | Lines 113–122 | `tech-choices.md` | Key design decisions |
| `README.md` | Lines 126–147 | `results.md` | Future work section (for conclusions/next steps) |
| `docs/mvp-project-idea.md` | Lines 222–258 | `architecture.md` | Concrete architecture diagram + file tree |
| `docs/mvp-project-idea.md` | Lines 17–99 | `project.md` | Project finish line, deliverables, scope |
| `docs/mvp-project-idea.md` | Lines 265–279 | `tech-choices.md` | Key design decisions |
| `docs/implementation-plan.md` | Lines 7–13 | `technical/data-processing.md` | Phase 1 overview, pipeline flow |
| `docs/implementation-plan.md` | Lines 484–513 | `technical/data-processing.md` | Augmentation algorithm details |
| `docs/implementation-plan.md` | Lines 424–464 | `technical/evaluation.md` | pytrec_eval metrics, methodology |
| `docs/implementation-plan.md` | Lines 88–151 | `technical/evaluation.md` | Proxy metrics, self-retrieval, separation gap |
| `docs/implementation-plan.md` | Lines 158–191 | `technical/evaluation.md` | HDBSCAN cluster validation |
| `docs/implementation-plan.md` | Lines 657–683 | `technical/embedding-pipeline.md` | Production embedding module, similarity engine |
| `docs/research-reports/embedding-optimization-research.md` | Lines 24–35 | `technical/tech-choices.md` | 5-layer optimization framework |
| `docs/research-reports/embedding-optimization-research.md` | Lines 380–411 | `technical/tech-choices.md` | Technique comparison table, recommended stack |
| `docs/research-reports/embedding-optimization-research.md` | Lines 43–73 | `technical/embedding-pipeline.md` | Preprocessing steps, domain-specific techniques |
| `docs/research-reports/embedding-optimization-research.md` | Lines 78–95 | `technical/embedding-pipeline.md` | L2 normalization explanation |
| `docs/research-reports/embedding-optimization-research.md` | Lines 98–132 | `technical/embedding-pipeline.md` | Mean centering, PCA whitening |
| `docs/research-reports/embedding-optimization-research.md` | Lines 137–170 | `technical/embedding-pipeline.md` | Section weighting, concatenation strategies |
| `docs/experiment-notes.md` | Lines 3–92 | `experiments/001-baseline-embedding.md` | Baseline experiment: design, results, discussion |
| `docs/experiment-notes.md` | Lines 95–141 | `experiments/002-boilerplate-removal.md` | Boilerplate experiment: design, results, discussion |
| `docs/experiment-notes.md` | Lines 143–433 | `experiments/003-llm-extraction.md` | LLM extraction: design, results, per-posting analysis |
| `scripts/comp_embedding_variants.py` | Lines 57–88 | `technical/evaluation.md` | Self-retrieval and separation gap code snippets |
| `scripts/comp_embedding_variants.py` | Lines 91–147 | `technical/evaluation.md` | UMAP comparison plotting code |

### 2.2 Page Content Specifications

#### `index.md` — Home

**Sections to populate:**
- **Hero:** Project title, one-line description, badges (Python 3.12+, Sentence Transformers, FAISS, FastAPI, Docker), links (Demo, GitHub)
- **Project snapshot cards:** Problem (keyword search fails), Solution (semantic retrieval), System (embedding pipeline + vector index + API), Evaluation (quantitative benchmarks)
- **Architecture preview:** Simplified ASCII diagram (Documents → Preprocessing → Embedding Model → Vector DB → Query API → Results), link to architecture page
- **Current status:** Table from README.md lines 22–33

**Auto-extract from:** `README.md` hero section, status table, one-sentence description

#### `project.md` — Project Overview

**Sections to populate:**
- **Problem Statement:** Why semantic similarity matters for job matching
- **Goals:** 4 primary goals as checkboxes (from mvp-project-idea.md deliverables)
- **Non-goals:** What this project does NOT do
- **System capabilities cards:** Embedding Generation, Similarity Search, Evaluation, Deployment
- **Project evolution timeline:** Research prototype → Evaluation framework → Production pipeline → Cloud deployment

**Auto-extract from:** `docs/mvp-project-idea.md` deliverables section, design decisions section

#### `demo.md` — Demo

**Sections to populate:**
- **Live Demo placeholder:** Embedded application container (iframe or "coming soon")
- **Example queries:** Show sample queries and expected results
- **How the demo works:** 3-step flow diagram (User query → Embedding → Similarity search → Ranked results)

**Note:** This page is mostly placeholder since the live demo is not yet built. Keep it minimal — no auto-extraction needed.

#### `architecture.md` — Architecture

**Sections to populate:**
- **System overview diagram:** ASCII art from mvp-project-idea.md lines 222–258
- **Component table:** Each component and its responsibility
- **Data flow:** Step-by-step from document ingestion to query response
- **Design decisions:** Why cosine similarity, why this embedding model, why precomputed embeddings, why stateless API

**Auto-extract from:** `docs/mvp-project-idea.md` architecture section, design decisions; `README.md` architecture directory tree

#### `technical/embedding-pipeline.md`

**Sections to populate:**
- **Overview:** Raw text → Cleaning → Chunking → Embedding → Vector storage
- **Preprocessing:** Normalization, chunking strategy, metadata handling
- **Embedding model:** Model selection (all-MiniLM-L6-v2), dimensionality (384d), inference requirements
- **Post-processing:** L2 normalization, mean centering, PCA whitening
- **Implementation:** Code snippet from `comp_embedding_variants.py` `embed_texts()` function
- **Section weighting:** Weighted concatenation strategies from embedding-optimization-research.md

**Auto-extract from:** `docs/research-reports/embedding-optimization-research.md` sections 2–4; `docs/implementation-plan.md` Phase 3

#### `technical/data-processing.md`

**Sections to populate:**
- **Dataset:** Source (LinkedIn), size (27 postings), characteristics (role breakdown)
- **Processing pipeline:** Raw data → Validation → Transformation → Split
- **Data quality:** Duplicate handling, missing values, noise
- **Augmentation pipeline:** Template-based generation with code snippet from implementation-plan.md lines 484–513

**Auto-extract from:** `docs/implementation-plan.md` Phase 1, Phase 2; `README.md` data status section

#### `technical/evaluation.md`

**Sections to populate:**
- **Evaluation framework:** Question being answered ("Does the system retrieve relevant information?")
- **Metrics:** Recall@K, MRR, NDCG — definition, intuition, usage
- **Experimental methodology:** Dataset, baseline, embedding model, retrieval method, protocol
- **Results summary:** Tables from experiment-notes.md (self-retrieval scores, separation gaps)
- **Code:** Self-retrieval and separation_gap implementations from `comp_embedding_variants.py`

**Auto-extract from:** `docs/implementation-plan.md` Steps 1.3, 1.9; `docs/experiment-notes.md` metric results; `scripts/comp_embedding_variants.py` metric implementations

#### `technical/tech-choices.md`

**Sections to populate:**
- **Architecture decisions:** Each decision with rationale
- **Alternatives considered table:** What was selected vs what was rejected and why
- **Tradeoffs:** Accuracy vs latency vs cost vs complexity
- **Recommended stack:** Must-do, should-do, deferred — from embedding-optimization-research.md

**Auto-extract from:** `README.md` design decisions; `docs/mvp-project-idea.md` design decisions; `docs/research-reports/embedding-optimization-research.md` comparison table, recommended stack

#### `results.md`

**Sections to populate:**
- **Summary dashboard cards:** Retrieval accuracy, latency, dataset size, embedding dimension
- **Benchmark results table:** Raw vs boilerplate vs LLM-cleaned comparison from experiment-notes.md
- **Qualitative examples:** Per-posting LLM extraction analysis (BMO, Affirm, HelloFresh, Mastercard, Scribd)
- **Conclusions:** What worked, what failed, what changed

**Auto-extract from:** `docs/experiment-notes.md` results tables, per-posting analysis sections

#### `experiments/index.md`

**Sections to populate:**
- **Experiment timeline:** Chronological list of all 3 experiments with dates and one-line summaries
- Links to each experiment page

#### `experiments/001-baseline-embedding.md`

**Sections:** Question, Hypothesis, Setup, Results (with PCA/UMAP/t-SNE plots), Interpretation, Discussion

**Auto-extract from:** `docs/experiment-notes.md` lines 3–92

#### `experiments/002-boilerplate-removal.md`

**Sections:** Question, Hypothesis, Setup, Results (with PCA/UMAP/t-SNE plots), Interpretation, Discussion (brittle regex → plan for LLM extraction)

**Auto-extract from:** `docs/experiment-notes.md` lines 95–141

#### `experiments/003-llm-extraction.md`

**Sections:** Question, Hypothesis, Setup (3 scripts), Results (per-model table, per-posting analysis for BMO/Affirm/HelloFresh/Mastercard/Scribd), Interpretation, Discussion

**Auto-extract from:** `docs/experiment-notes.md` lines 143–433

### 2.3 Auto-Population Rules

**As markdown/writing guidelines for each page:**
1. Each page should follow Zensical's Material theme conventions: use admonitions (`!!! note`, `!!! info`, `!!! warning`), fenced code blocks with language tags, and Mermaid diagrams where applicable
2. Use Zensical card grids where layout proposals specify cards (e.g., index.md snapshot, project.md capabilities)
3. Extract verbatim where the source already reads well (experiment notes, metric tables)
4. Summarize and rephrase where the source is a planning document written for internal consumption
5. Replace "you" / "we" with neutral technical voice
6. For plots referenced in experiment notes (e.g., `../data/plots/raw_pca_baseline.png`), note them as `assets/images/` with a TODO to copy the actual image files later
7. Code snippets should reference the actual file and line numbers they come from
8. Status indicators: use `✅`, `🟡`, `🔴` as in README.md

## Phase 3: Update Zensical Config

### 3.1 Replace Navigation

Update `nav` in `zensical.toml` to match the proposed structure:

```toml
nav = [
  { Home = "index.md" },
  { "Project Overview" = "project.md" },
  { Demo = "demo.md" },
  { Architecture = "architecture.md" },
  { "Technical Details" = [
    { "Embedding Pipeline" = "technical/embedding-pipeline.md" },
    { "Data Processing" = "technical/data-processing.md" },
    { Evaluation = "technical/evaluation.md" },
    { "Technology Choices" = "technical/tech-choices.md" },
  ]},
  { Results = "results.md" },
  { "Experiment Log" = [
    { Overview = "experiments/index.md" },
    { "001 – Baseline Embedding" = "experiments/001-baseline-embedding.md" },
    { "002 – Boilerplate Removal" = "experiments/002-boilerplate-removal.md" },
    { "003 – LLM Extraction" = "experiments/003-llm-extraction.md" },
  ]},
]
```

### 3.2 Verification

Run `zensical build` after scaffold and verify all nav links resolve without errors.

## Phase 4: Copy Plot Images (Manual)

**Task:** Copy existing plot images from `data/plots/` (generated locally, git-ignored) into `site/assets/images/`:

| Source | Target |
|---|---|
| `data/plots/raw_pca_baseline.png` | `site/assets/images/raw_pca_baseline.png` |
| `data/plots/raw_umap_baseline.png` | `site/assets/images/raw_umap_baseline.png` |
| `data/plots/raw_tsne_baseline.png` | `site/assets/images/raw_tsne_baseline.png` |
| `data/plots/exp_boilerplate_pca.png` | `site/assets/images/exp_boilerplate_pca.png` |
| `data/plots/exp_boilerplate_umap.png` | `site/assets/images/exp_boilerplate_umap.png` |
| `data/plots/exp_boilerplate_tsne.png` | `site/assets/images/exp_boilerplate_tsne.png` |
| `data/plots/llm_cleaning_comparison.png` | `site/assets/images/llm_cleaning_comparison.png` |

Update image references in experiment pages from `../data/plots/` to `../assets/images/`.

## Phase 5: Run Locally and Verify

```bash
zensical build         # Build site, verify no broken links
zensical serve         # Preview at localhost
```

Verify:
- [ ] All nav links work
- [ ] All images render
- [ ] Code blocks are syntax-highlighted
- [ ] Admonitions render correctly
- [ ] Tables display properly
- [ ] No 404s for any page

## Risks and Edge Cases

1. **Plot images don't exist on disk**: `data/plots/` is git-ignored and may not contain the expected plots. If missing, pages should still build — image references will just be broken, not fatal.
2. **Experiment notes contain personal/chatty language**: The auto-extraction should rewrite conversational passages ("I think", "we should") into neutral technical prose. Flag any sections that need manual polishing.
3. **zensical.toml nav restructuring**: Ensure no dangling references to old nav entries remain. Delete the old `Methodology` and `Experiments` sections and the date-based experiment post filenames.
4. **`site/` directory in `.gitignore`**: Verify `site/` is NOT in `.gitignore` (presently only `.venv/`, `__pycache__/`, `*.pyc`, `*.npy`, and specific data files). The `site/` content should be committed.

## Validation Plan

After implementation, run:

```bash
zensical build 2>&1    # Must exit 0 with no errors or warnings
```

Then manually verify:
1. Open `http://localhost:8000` (or wherever `zensical serve` binds)
2. Navigate through every nav item
3. Confirm each page has content (not empty/TODO placeholders)
4. Confirm experiment pages have metric tables and plot references
5. Confirm architecture page has ASCII diagram and component table
6. Confirm tech-choices page has alternatives-considered table
