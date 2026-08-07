# Cross-Reference Index

> **Purpose:** Map every piece of information in the docs to all pages where it appears, so that when a research report is added, an experiment is completed, or status advances, the maintainer knows exactly which files to update.
>
> **Rule:** If you change one, update them all. Stale references are misleading.
>
> **Note — openspec/specs/ is the canonical capability tracker.** The 6 OpenSpec capability specs (data-ingestion, llm-text-extraction, embedding-pipeline, evaluation-framework, semantic-partitioning, taxonomy-label-generation) define *what the system SHALL do*. The docs site describes *what we did and why*. Capability status is now tracked by which specs exist; deliverable status (project.md) tracks implementation progress. They do not overlap.

---

## 1. Project Status & Phase Tracking

| What | Files That Reference It |
|---|---|
| Active work-in-progress items | `TODO.md` |
| Capability spec inventory (canonical) | `openspec/specs/` (6 specs: data-ingestion, llm-text-extraction, embedding-pipeline, evaluation-framework, semantic-partitioning, taxonomy-label-generation) |
| Current phase / deliverable status table | `project.md:81-93`, `index.md:88-106` (commented-out) |
| Experiment # and status (completed/planned) | `experiments/index.md:24-65`, `research-reports/index.md:36-44`, `project.md:81-93` |
| Dataset size (27 postings) | `project.md:85`, `technical/data-processing.md:11-12`, `technical/evaluation.md:140`, `technical/embedding-pipeline.md:38`, `results.md:113`, `results.md:10` (date), `experiments/001-baseline-embedding.md:11`, `architecture.md:33` |
| Embedding model name | `index.md:61` (badge), `architecture.md:77` (design rationale), `technical/embedding-pipeline.md:44-57`, `technical/tech-choices.md:19`, `index.md:81` (commented-out), `experiments/001-baseline-embedding.md:24`, `experiments/002-boilerplate-removal.md:18` |
| Tech stack badges (FastAPI/Docker "planned") | `index.md:63-64` |

**When status changes:** Update `project.md`, then uncomment/update `index.md:88-106`. Update `experiments/index.md` and `research-reports/index.md` if experiments are affected.

---

## 2. Experiment Results (Metric Values)

| Metric | Primary Definition | Summary Pages | Experiment Pages |
|---|---|---|---|
| Separation gap: +0.0060 (raw), +0.0203 (regex), +0.0493 (LLM) | `technical/evaluation.md:77-81` | `results.md:29-35` (commented-out), `experiments/index.md:48-54` (commented-out), `research-reports/index.md:28-34` (commented-out) | `experiments/003-llm-extraction.md:222-226` |
| Semantic partitioning gap: +0.0255 (golden job), +0.0588 (golden role), +0.0385 (LLM job), -0.0074 (LLM role) | `technical/evaluation.md:78-82` | `results.md:10-13` | `experiments/005-semantic-partitioning.md:251-259` |
| Self-retrieval: 100% (all) | `technical/evaluation.md:24-43` | `results.md:29-35` (commented-out) | `experiments/003-llm-extraction.md:222-226` |
| LLM extraction quality (Jaccard, boilerplate, hallucinations, over-deletion) | `technical/data-processing.md:79-87` | `results.md:39-49` (commented-out) | `experiments/003-llm-extraction.md:144-150` |
| LLM model tested (gpt-5.4-nano) | `technical/data-processing.md:68` | `results.md:41` (commented-out) | `experiments/003-llm-extraction.md:138` |

**When a new experiment produces new metrics:** Update or uncomment the summary table in `results.md`, `experiments/index.md`, and `research-reports/index.md`. Add a new experiment page and register it in `experiments/index.md` and `mkdocs.yml`.

---

## 3. Experiment Log Registry

Adding an experiment requires updating these files:

1. **`experiments/<NNN>-<slug>.md`** — the experiment page itself
2. **`experiments/index.md`** — add to the "Completed Experiments" grid (or "Planned" table)
3. **`results.md`** — update summary metrics and conclusions
4. **`project.md`** — update deliverable status table
5. **`mkdocs.yml`** — add to `nav.Experiment Log` list
6. **`technical/evaluation.md`** — update results table if proxy metrics change
7. **`technical/data-processing.md`** — update data quality section if preprocessing changes
8. **`technical/embedding-pipeline.md`** — update if embedding strategy changes

---

## 4. Research Report Registry

Adding a research report requires updating:

1. **`research-reports/<NNN>-<slug>.md`** — the report itself
2. **`research-reports/index.md`** — add to the grid cards and optionally "Planned Experiments"
3. **`mkdocs.yml`** — add to `nav.Research Reports` list

---

## 5. Architecture & Design Decisions

| Decision | Where Documented |
|---|---|
| Why L2 normalization | `architecture.md:72` (design decision), `technical/embedding-pipeline.md:69-82`, `research-reports/001-embedding-optimization-research.md:78-96` |
| Why all-MiniLM-L6-v2 | `architecture.md:76-79`, `technical/embedding-pipeline.md:57-65`, `technical/tech-choices.md:19` |
| Why precomputed embeddings | `architecture.md:83` |
| Why CPU-only | `architecture.md:91`, `technical/tech-choices.md:10` |
| Why stateless API | `architecture.md:87` |
| Why LLM-based boilerplate removal | `architecture.md` (commented-out), `technical/tech-choices.md:13`, `technical/data-processing.md:66-75` |
| Model comparison table (all-MiniLM-L6-v2 vs bge-small-en-1.5 vs mpnet-base-v2) | `technical/embedding-pipeline.md:59-63`, `technical/tech-choices.md:18-21` |
| Alternatives considered (FAISS, pgvector, Elasticsearch, ONNX, CUDA, hybrid BM25) | `technical/tech-choices.md:17-28` |
| Optimization layers table | `technical/tech-choices.md:32-38` |

---

## 6. Data Pipeline & Processing

| Information | Where Documented |
|---|---|
| Dataset properties (27 postings, LinkedIn, July 2026) | `technical/data-processing.md:5-13` |
| Role breakdown (8 DS, 5 MLE, 3 AIE, etc.) | `technical/data-processing.md:18-22`, `project.md:99` (commented-out) |
| Processing pipeline steps (parse → validate → LLM clean → embed → viz) | `technical/data-processing.md:28-38`, `architecture.md:30-37` (commented-out) |
| LLM cleaning details (system prompt, gpt-5.4-nano) | `technical/data-processing.md:66-75` |
| Quality validation metrics | `technical/data-processing.md:77-87` |
| Data augmentation plan (template-based, 27→350+, slot vocabularies) | `technical/data-processing.md:105-126` |
| Text normalization steps | `technical/embedding-pipeline.md:18-22` |
| Skill extraction plan (spaCy, 200-term vocabulary) | `technical/embedding-pipeline.md:33-41` |

---

## 7. Feature Engineering Plans

| Plan | Where Documented |
|---|---|
| Section weighting strategies (Variants A/B/C) | `technical/embedding-pipeline.md:106-138`, `research-reports/001-embedding-optimization-research.md:137-179` |
| Mean centering | `technical/embedding-pipeline.md:84-87`, `research-reports/001-embedding-optimization-research.md:98-105` |
| PCA whitening (deferred) | `technical/embedding-pipeline.md:89-90`, `research-reports/001-embedding-optimization-research.md:107-131` |
| HDBSCAN cluster validation | `technical/evaluation.md:123-135`, `research-reports/002-tutte-institute-tool-review.md:83-103` |
| DataMapPlot visualization upgrade | `research-reports/002-tutte-institute-tool-review.md:48-49, 78, 139-153` |
| Function-based taxonomy (13-category) | `research-reports/003-ml-ai-responsibility-taxonomy.md:86-436` |

---

## 8. Evaluation Framework

| Component | Where Documented |
|---|---|
| Proxy metrics definitions (self-retrieval, separation gap) | `technical/evaluation.md:16-81` |
| IR metrics plan (Precision@K, MRR, NDCG, pytrec_eval) | `technical/evaluation.md:85-121`, `research-reports/001-embedding-optimization-research.md:320-365` |
| Golden set plan | `technical/evaluation.md:121` |
| Current evaluation limitations (noisy labels, small dataset, no held-out set) | `technical/evaluation.md:138-142` |
| Visualization findings (PCA/UMAP/t-SNE) | `results.md:84-91` (commented-out), `experiments/001-baseline-embedding.md:29-74` |

---

## 9. Front-Matter / Navigation

| File | What to Update |
|---|---|
| `mkdocs.yml` `nav:` | Add new experiment/research report pages, uncomment technical sub-pages when ready |
| `index.md` | Update badges (FastAPI, Docker from "planned" when built), uncomment "Current Status" section |
| `project.md` | "Current Status" table — the canonical project progress tracker |
| `demo.md` | Update when demo is built (currently all placeholder) |
| `architecture.md` | Uncomment system overview, components table, data flow, repo structure when design settles |

---

## 10. Update Workflow Cheat Sheet

### Completing an experiment:
1. Write `experiments/<NNN>-<slug>.md`
2. Add to "Completed Experiments" in `experiments/index.md`
3. Move from "Planned" to "Completed" in `research-reports/index.md` if listed
4. Update results table in `technical/evaluation.md`
5. Update summary in `results.md` (uncomment table, update conclusions)
6. Update `project.md` deliverable status
7. Update `experiments/index.md` planned progression if scope changes
8. Register in `mkdocs.yml` nav

### Adding a research report:
1. Write `research-reports/<NNN>-<slug>.md`
2. Add to grid cards in `research-reports/index.md`
3. Register in `mkdocs.yml` nav

### Changing architecture/technical decisions:
1. Update `architecture.md` (uncomment sections, revise design decisions)
2. Update `technical/tech-choices.md` tables
3. Update `technical/embedding-pipeline.md` if embedding strategy changes
4. Update `technical/data-processing.md` if preprocessing changes
5. Update `index.md` badges if tech stack changes

### Advancing project phase:
1. Update `project.md` "Current Status" table
2. Uncomment relevant sections in `index.md`
3. Update `index.md` badges
4. Update `technical/overview.md` when done
5. Uncomment relevant sections in `architecture.md`

### Changing the dataset:
1. Update `technical/data-processing.md` dataset properties and role breakdown
2. Update `project.md` (commented-out data status section)
3. Update `technical/evaluation.md` limitations section
4. Update all experiment pages that reference dataset size (001, 002, 003)
5. Update `results.md` conclusions
