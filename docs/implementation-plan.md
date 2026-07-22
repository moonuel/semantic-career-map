# Implementation Plan — Semantic Career Mapping Platform

> Detailed phase-by-phase implementation plan. Each phase has concrete steps, expected outputs, and evaluation checkpoints.

---

## Phase 1: Preprocessing & Embedding Optimization Pipeline

**Duration:** 13–15 hours
**Goal:** Establish a rigorous preprocessing workflow with baseline metrics, visualization, experiment tracking, and a golden evaluation set.

> **Status:** Steps 1.0–1.2 complete (combined as `scripts/parse_and_embed_quickstart.py`). The combined script was a deliberate bootstrap for fast iteration; it will be refactored into the per-step scripts below once the experiment loop begins.

### Overview

```
Raw Markdown → Structured JSON → Baseline Embedding → Visualize (PCA/UMAP)
    → Proxy Metrics → Experiment Loop → Golden Set → Final Evaluation → Documentation
```

Each preprocessing change is evaluated against the baseline using visualization and metrics. Only changes that measurably improve separation are adopted.

---

### Step 1.0 — Structured Data Extraction ✅

**Input:** `data/selected-job-postings/*.md` (27 raw postings)
**Output:** `data/jobs.json`
**Script:** `scripts/parse_and_embed_quickstart.py` (parse portion)

Done. 27 postings parsed with title extraction (filename heuristics + explicit text patterns), company detection (known mapping + filename fallback), role category labels (hand-mapped, 6 categories), and basic section detection (23 patterns covering about_role, responsibilities, qualifications, nice_to_have, what_we_offer, about_team).

**Planned addition:** A `function_category` field will be added in a second labeling pass (Step 1.2c) — this captures what the job actually does (data-engineering, exploratory-analysis, model-development, model-production, research, applied-ai) rather than what it's called. See Step 1.2c for details.

### Step 1.1 — Baseline Embedding ✅

**Input:** `data/jobs.json`
**Output:** `data/raw_embeddings.npy`
**Script:** `scripts/parse_and_embed_quickstart.py` (embed portion)

Done. all-MiniLM-L6-v2 on CPU, L2-normalized to unit norm (verified). Shape (27, 384). Mean pairwise cosine sim: 0.40.

### Step 1.2 — Visualization ✅

**Input:** `data/raw_embeddings.npy`, `data/jobs.json`
**Output:** `data/plots/raw_pca.png`, `data/plots/raw_umap.png`, `data/plots/raw_tsne.png`
**Script:** `scripts/parse_and_embed_quickstart.py` (visualize portion)

Done. PCA (2-component + scree plot), UMAP (cosine metric, n_neighbors=5), and t-SNE (cosine metric, perplexity=5) generated with point labels colored by role category.

### Step 1.2b — Package Manager: uv

The project uses `uv` for package management via `pyproject.toml`. Dependencies are also mirrored in `requirements.txt` for compatibility. Commands:

```bash
uv sync                           # Install all dependencies
uv pip install -r requirements.txt  # Alternative: pip from requirements.txt
uv run spacy download en_core_web_sm  # One-time: download spaCy model
```

### Step 1.2c — Function-Based Role Labels (Planned)

Current `role_category` labels are derived from raw job titles (Data Scientist, ML Engineer, AI Engineer, etc.), but job titles are a noisy signal — a "Data Scientist" at Coca-Cola doing demand modeling and a "Data Scientist" at Scribd building NLP models are fundamentally different roles. For clustering evaluation, we need labels that reflect **what the job actually does**, not what it's called.

**Secondary labeling pass:** Each posting will be tagged with 1–2 function categories from a 12-category taxonomy derived from 2025–2026 job market research (see `docs/research-reports/ml-ai-responsibility-taxonomy.md` for the full report with sources):

| Label | Domain | Description |
|---|---|---|
| `agentic-ai` | LLM/AI | Autonomous agents, tool use, multi-agent orchestration |
| `llm-fine-tuning` | LLM/AI | LoRA/SFT/DPO model adaptation, distillation |
| `llm-information-retrieval` | LLM/AI | RAG pipelines, hybrid search, vector DBs, retrieval eval |
| `classical-ml` | Classical ML | Structured/tabular data, XGBoost, feature engineering |
| `mlops-production` | ML Ops | CI/CD, model serving, drift monitoring, on-call |
| `data-engineering` | Data | ETL/ELT, orchestration, data quality, warehousing |
| `analytics-storytelling` | Analytics | EDA, A/B tests, dashboards, stakeholder presentations |
| `computer-vision` | CV | Image/video, object detection, OCR, VLMs |
| `ml-platform` | ML Infra | Feature stores, model registries, shared infra for ML teams |
| `reinforcement-learning` | RL | PPO, reward engineering, environment simulation |
| `research` | Research | Papers, novel algorithms, prototyping frontier methods |
| `ai-safety-governance` | Safety | Bias auditing, explainability, red-teaming, compliance |

**Why this matters for clustering:** Same-title roles that do different work (e.g., ICBC "Data Science Analyst" doing SQL reporting vs Scribd "Data Scientist II" building NLP models) should not cluster together. If preprocessing improvements bring same-function postings closer, the embedding space is capturing real work similarity, not title artifacts. This is a more honest clustering target than raw title categories.

**Timing:** Applied before the experiment loop (Step 1.3), alongside the existing hand-labeled `role_category`. Both label sets will be stored in `jobs.json` and available to all evaluation scripts.

---

### Step 1.3 — Baseline Evaluation (Proxy Metrics)

**Input:** `data/raw_embeddings.npy`, `data/raw_metadata.json`
**Output:** Baseline metrics JSON, nearest-neighbor audit

Compute metrics without a labeled test set.

**Metric A: Self-Retrieval Check**

```python
for i, job in enumerate(jobs):
    # Query: first 200 chars of the role description
    query = job["sections"]["about_role"][:200]
    query_emb = model.encode(query, normalize_embeddings=True)
    similarities = cosine_similarity([query_emb], embeddings)[0]
    rank = np.argsort(similarities)[::-1]
    self_position = np.where(rank == i)[0][0]
    results[job["id"]] = self_position

# Expected: self_position == 0 for all postings
```

**Metric B: Cross-Role Separation Gap**

```python
from itertools import combinations

same_role_sims = []
diff_role_sims = []

for i, j in combinations(range(n), 2):
    sim = 1 - cosine(embeddings[i], embeddings[j])
    if jobs[i]["role_category"] == jobs[j]["role_category"]:
        same_role_sims.append(sim)
    else:
        diff_role_sims.append(sim)

gap = np.mean(same_role_sims) - np.mean(diff_role_sims)
print(f"Same-role mean sim: {np.mean(same_role_sims):.3f}")
print(f"Cross-role mean sim: {np.mean(diff_role_sims):.3f}")
print(f"Separation gap:     {gap:.3f}")
```
Compute separation gap for **both** `role_category` (title-based) and `function_category` (work-based, from Step 1.2c). The function-based gap is the more meaningful signal — same-title labels can be misleading, but same-function postings should genuinely cluster together.

**Metric C: Nearest-Neighbor Audit (Manual)**

For each posting, record top-2 nearest neighbors and a qualitative judgment:

| Posting | Nearest | 2nd Nearest | Sensible? |
|---|---|---|---|
| BMO Data Scientist | Intact Data Scientist | RBC Data Scientist | Yes |
| eBay Researcher | Huawei Researcher | Mastercard DS | Questionable |
| ... | ... | ... | ... |

**Output format:** `data/baseline_metrics.json`
```json
{
  "self_retrieval": {"bmo-associate-data-scientist": 0, "...": 0},
  "same_role_mean_sim": 0.72,
  "cross_role_mean_sim": 0.68,
  "separation_gap_title": 0.04,
  "separation_gap_function": 0.09,
  "nn_audit": {"bmo-associate-data-scientist": ["intact-data-scientist-2", "rbc-data-scientist"]}
}
```

**Script:** `scripts/evaluate_baseline.py`

---

### Step 1.3b — HDBSCAN Cluster Validation (Interleaved Metric)

**Input:** `data/raw_embeddings.npy`, `data/raw_metadata.json`
**Output:** Cluster-vs-category alignment notes

As a fourth sanity-check metric, run HDBSCAN clustering on the embeddings and compare against both `role_category` (title-based) and `function_category` (work-based, from Step 1.2c). The function alignment is the primary signal — if an unsupervised algorithm recovers work-based function labels, the embedding space captures what people actually do, not just what their title says.

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2,
    min_samples=1,
    metric="euclidean",
    cluster_selection_method="eom",
)
labels = clusterer.fit_predict(embeddings)

# Cross-tabulate HDBSCAN labels vs role_category and function_category
# Record: number of clusters found, noise points, alignment with hand labels
```

**What to record:**
- Number of clusters found by HDBSCAN
- Whether same-role_category postings fall in the same cluster (caveat: titles are noisy)
- Whether same-function_category postings fall in the same cluster (primary signal)
- Whether cross-function postings are merged into one cluster (indicates embedding doesn't separate those functions)
- Whether cross-category postings are merged into one cluster (indicates embedding doesn't separate those roles)
- Any postings assigned to the noise label (-1) — indicates an outlier in the embedding space

Run this after every experiment (Steps 1.4–1.7) to track whether cluster structure improves.

> See `docs/tutte-institute-tool-review.md` for background on HDBSCAN and the TIMC vector toolkit.

---

### Step 1.4 — Experiment 1: Boilerplate Removal

**Hypothesis:** Company descriptions, "About Us", EEO statements, salary disclosures, and recruiter notes are noise. Removing them increases the same-role vs cross-role separation gap.

**Implementation:**
- Construct `clean_text` by concatenating only: `sections.about_role` + `sections.responsibilities` + `sections.qualifications` + `sections.nice_to_have`
- Exclude: company description paragraphs, "About Us", "What We Offer", salary paragraphs, EEO boilerplate, recruiter notes
- Add `clean_text` field to each job in `jobs.json`

```python
clean_text = (
    job["sections"]["about_role"]
    + "\n\n"
    + job["sections"]["responsibilities"]
    + "\n\n"
    + job["sections"]["qualifications"]
    + "\n\n"
    + job["sections"].get("nice_to_have", "")
)
```

**Re-embed & evaluate:**
- Run same embedding pipeline as Step 1.1 but with `clean_text`
- Re-run metrics from Step 1.3 (self-retrieval, separation gap, NN audit, HDBSCAN cluster alignment)
- Re-visualize with UMAP (same params as Step 1.2, DataMapPlot)

**Record delta:**
```
Raw vs Boilerplate-removed:
  Separation gap: 0.04 → 0.07 (+0.03)
  Self-retrieval: all #1 ✓ (unchanged)
  Visual: company-level clustering reduced
```

**Script:** `scripts/exp_boilerplate.py`

---

### Step 1.5 — Experiment 2: Title Canonicalization

**Hypothesis:** Standardizing job title variations will improve role-level grouping when titles are weighted in the embedding.

**Implementation:**
- Extract `title_raw` from filename/section header
- Apply canonicalization rules:

```python
TITLE_MAP = {
    "Associate Data Scientist": "Data Scientist",
    "Data Scientist": "Data Scientist",
    "Data Scientist, AI Model Risk": "Data Scientist",
    "Applied Researcher 1": "Applied Researcher",
    "Applied Researcher": "Applied Researcher",
    "AI/ML Researcher": "Machine Learning Researcher",
    "Data Scientist 2": "Data Scientist",
}
```

- Add `title_canonical` field to each job
- **Do not re-embed yet** — this is a metadata step only

**Verification:** Print every job's `title_raw` → `title_canonical` mapping and confirm correctness.

**Script:** `scripts/exp_canonicalize_titles.py`

---

### Step 1.6 — Experiment 3: Skill Extraction

**Hypothesis:** Skills are the most discriminative signal for role matching. Extracting them into a structured field enables weighting in Experiment 4.

**Implementation:**
- Build a skill vocabulary covering ML/DS/AI domains (~200 terms):

```python
SKILL_VOCABULARY = [
    # Languages
    "Python", "R", "SQL", "Scala", "Java", "C++",
    # ML Frameworks
    "PyTorch", "TensorFlow", "Keras", "JAX", "scikit-learn", "XGBoost",
    "LightGBM", "Hugging Face", "Transformers", "spaCy", "NLTK",
    # ML Techniques
    "Deep Learning", "Reinforcement Learning", "NLP", "Computer Vision",
    "Time Series", "Generative AI", "LLM", "Statistical Modeling",
    "A/B Testing", "Causal Inference", "Feature Engineering",
    # Infrastructure
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "MLflow",
    "CI/CD", "Git", "Linux", "Spark", "Airflow", "Kafka",
    # Domain-specific
    "Bloomberg", "FactSet", "Algorithmic Trading", "Risk Modeling",
    # Soft/tool
    "Tableau", "Power BI", "Excel", "Jupyter",
]
```

- Use spaCy `PhraseMatcher` for multi-word terms:

```python
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp.make_doc(skill) for skill in SKILL_VOCABULARY]
matcher.add("SKILL", patterns)

doc = nlp(job["sections"]["qualifications"] + " " + job["sections"]["nice_to_have"])
matches = matcher(doc)
skills = list(set([doc[start:end].text for match_id, start, end in matches]))
```

- Separate required vs nice-to-have skills by which section they appear in
- Add `skills_required` and `skills_nice_to_have` arrays to each job

**Edge cases:**
- "Proficiency in statistical modeling" → tag `"Statistical Modeling"`
- "learning from data" → NOT a match for "Deep Learning" (false positive)
- "Familiarity with financial tools such as Bloomberg" → tag `"Bloomberg"` as domain skill
- Avoid tagging company names, section headers, or generic words

**Output:** Updated `data/jobs.json` with skill arrays.

**Script:** `scripts/exp_skill_extraction.py`

---

### Step 1.7 — Experiment 4: Weighted Concatenation

**Hypothesis:** Up-weighting title and skills sections in the embedding input text increases role-level separation and improves retrieval quality.

**Implementation — three variants:**

**Variant A: Concatenation with repetition**
```python
weighted_text_a = (
    f"[TITLE] {job['title_canonical']} " * 3
    + f"[SKILLS] {' '.join(job['skills_required'])} " * 2
    + f"[RESPONSIBILITIES] {job['sections']['responsibilities']} "
    + f"[QUALIFICATIONS] {job['sections']['qualifications']}"
)
```

**Variant B: 2× title, 2× skills**
```python
weighted_text_b = (
    f"[TITLE] {job['title_canonical']} " * 2
    + f"[SKILLS] {' '.join(job['skills_required'])} " * 2
    + f"[RESPONSIBILITIES] {job['sections']['responsibilities']} "
    + f"[QUALIFICATIONS] {job['sections']['qualifications']}"
)
```

**Variant C: Multi-field embedding + vector fusion**
```python
emb_title = model.encode(job["title_canonical"], normalize_embeddings=True)
emb_skills = model.encode(" ".join(job["skills_required"]), normalize_embeddings=True)
emb_body = model.encode(
    job["sections"]["responsibilities"] + " " + job["sections"]["qualifications"],
    normalize_embeddings=True,
)
# Weighted average, then re-normalize
emb = 0.4 * emb_title + 0.3 * emb_skills + 0.3 * emb_body
emb = emb / np.linalg.norm(emb)
```

**Evaluate:**
- Run all three variants through the same metrics pipeline (Step 1.3)
- Compare separation gaps side-by-side
- Run nearest-neighbor audit for the best variant
- Check token counts — weighted concat may exceed MiniLM's 256-token context window; if so, truncate responsibly

**Record deltas:**
```
Boilerplate-removed vs Weighted (Variant A):
  Separation gap: 0.07 → 0.19 (+0.12)
  Self-retrieval: all #1 ✓
  Token count: max 234 (safe)

Variant C (fusion) vs Variant A:
  Separation gap: 0.19 → 0.22 (+0.03)
  Decision: adpot Variant C if improvement is consistent
```

**Script:** `scripts/exp_weighted_concat.py`

---

### Step 1.8 — Golden Evaluation Set

**Input:** `data/jobs.json`
**Output:** `data/golden_set.json`

Build a small labeled evaluation set for proper IR metrics.

**Implementation:**
- Create 5–10 synthetic resumes by hand, covering diverse profiles:

```json
[
  {
    "resume_id": "r1_newgrad_ml",
    "profile": "New grad ML",
    "text": "Recent MSc in Computer Science. Coursework in Deep Learning, NLP. Python, PyTorch, Jupyter. Internship: built a text classifier at a startup.",
    "relevance": {
      "bmo-associate-data-scientist": 1,
      "ebay-applied-researcher-1": 0,
      "huawei-ai-ml-researcher": 0,
      "intact-data-scientist-2": 2,
      "mastercard-data-scientist-2": 1,
      "ontario-teachers-pension-plan-data-scientist": 1,
      "rbc-data-scientist-ai-model-risk": 0
    }
  },
  // ... 4–9 more resumes
]
```

**Relevance scoring:** 0 = irrelevant, 1 = somewhat relevant, 2 = highly relevant. Score based on skills match, seniority alignment, and domain appropriateness.

**Verification:** Have a second person review the labels, or review them yourself a day later to catch inconsistencies.

**Script:** Manual creation; store as `data/golden_set.json`.

---

### Step 1.9 — Final Evaluation with pytrec_eval

**Input:** `data/golden_set.json`, embeddings from best preprocessing config
**Output:** Precision@5, Recall@5, MRR, NDCG@5 — baseline vs optimized

```python
import pytrec_eval

qrels = {}   # ground truth: {query_id: {doc_id: relevance}}
run = {}     # system output: {query_id: {doc_id: score}}

for resume in golden_set:
    query_id = resume["resume_id"]
    query_emb = model.encode(resume["text"], normalize_embeddings=True)
    similarities = cosine_similarity([query_emb], job_embeddings)[0]

    # Top-k results
    ranked = np.argsort(similarities)[::-1]
    run[query_id] = {
        jobs[i]["id"]: float(similarities[i])
        for i in ranked[:10]
    }
    qrels[query_id] = {
        job_id: rel
        for job_id, rel in resume["relevance"].items()
    }

evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"P.5", "recall.5", "NDCG", "MRR"})
results = evaluator.evaluate(run)

# Aggregate
for metric in ["P_5", "recall_5", "NDCG", "MRR"]:
    scores = [r[metric] for r in results.values()]
    print(f"{metric}: {np.mean(scores):.3f}")
```

**Output comparison:**
```
              Raw     Boilerplate  Weighted(A)  Weighted(C)
Precision@5   0.52    0.58         0.76         0.82
Recall@5      0.45    0.51         0.71         0.78
MRR           0.61    0.67         0.84         0.88
NDCG@5        0.48    0.55         0.73         0.79
```

**Script:** `scripts/evaluate_final.py`

---

### Step 1.10 — Data Augmentation

**Input:** `data/jobs.json` (50+ real postings after data collection)
**Output:** `data/jobs_augmented.json` (350+ total postings)

Template-based synthetic posting generation.

**Implementation:**
- Define slot vocabularies:

```python
SENIORITY = ["Junior", "Mid-level", "Senior", "Staff", "Lead", "Principal"]
TECH_STACK = ["PyTorch", "TensorFlow", "JAX", "scikit-learn", "XGBoost"]
CLOUD = ["AWS", "GCP", "Azure"]
ROLE = ["ML Engineer", "Data Scientist", "MLOps Engineer", "Research Scientist", "NLP Engineer"]
VERBS = ["build", "deploy", "train", "evaluate", "optimize", "maintain"]
```

- For each seed posting, generate variants by swapping tokens:

```python
for seed in seed_postings:
    for seniority in SENIORITY:
        for tech in TECH_STACK:
            for cloud in CLOUD:
                # Generate 1 variant per combination (sample, don't exhaust)
                variant = seed.copy()
                variant["title_canonical"] = f"{seniority} {seed['role_category']}"
                variant["skills_required"] = [
                    tech if s in seed["skills_required"][:3] else s
                    for s in variant["skills_required"]
                ]
                variant["skills_required"].append(cloud)
                variant["is_synthetic"] = True
                variants.append(variant)
```

- Each variant must have semantically realistic skill-seniority combinations (e.g., no "Junior Engineer requiring 10 years of experience")
- Add `is_synthetic: true` to metadata for all generated postings
- Limit to ~50 variants per seed to avoid explosion

**Verification:** Generate 50 variants from BMO posting, manually spot-check 5 for realism.

**Script:** `scripts/augment_jobs.py`

---

### Step 1.11 — Augmented Data Visualization

**Input:** `data/jobs_augmented.json`, embeddings from best preprocessing config
**Output:** `data/plots/augmented_umap.png`

Full-scale UMAP visualization of the augmented embedding space.

**What to check:**
- Do synthetic variants form neighborhoods around their seed postings?
- Do synthetic variants bridge gaps between real postings?
- Are there synthetic postings that form isolated clusters (bad — unnatural generation)?
- Are real and synthetic postings spatially segregated? (If yes, the augmentation text is detectably different from real text)

**Visualization:**
- Color real postings black, synthetic postings by role category
- Use different marker shapes for real (circle) vs synthetic (cross)
- Annotate real posting labels

**Script:** `scripts/visualize_augmented.py`

---

### Step 1.12 — Document Results in README

Include a "Preprocessing Experiments" section:

```markdown
## Preprocessing Experiments

### Methodology
Each experiment isolates one variable. Metrics are measured against a raw-text
baseline to quantify improvement. The golden set consists of 10 synthetic resumes
with manually labeled relevance scores against 7+ real job postings.

### Results

| Experiment | Same-Role Gap | Precision@5 | MRR | Decision |
|---|---|---|---|---|
| Raw baseline | 0.04 | 0.52 | 0.61 | — |
| Boilerplate removal | 0.07 | 0.58 | 0.67 | Adopted |
| Weighted concat (3× title, 2× skills) | 0.19 | 0.76 | 0.84 | Adopted |
| Multi-field fusion (0.4/0.3/0.3) | 0.22 | 0.82 | 0.88 | Adopted (best) |
| PCA whitening (90% variance) | 0.20 | 0.79 | 0.86 | Deferred (marginal gain) |

### Production Pipeline

The final preprocessing pipeline:
1. Parse markdown into structured sections
2. Remove boilerplate (company description, EEO, salary, recruiter notes)
3. Extract skills with spaCy PhraseMatcher
4. Multi-field embedding: title (40%), skills (30%), body (30%)
5. L2 normalization after fusion
```

---

### Phase 1 File Manifest

```
scripts/
  parse_and_embed_quickstart.py  # Steps 1.0–1.2 combined (bootstrap, done)
  evaluate_baseline.py           # Step 1.3 — proxy metrics + NN audit + HDBSCAN cluster check
  exp_boilerplate.py            # Step 1.4 — boilerplate removal
  exp_canonicalize_titles.py    # Step 1.5 — title canonicalization
  exp_skill_extraction.py       # Step 1.6 — skill extraction
  exp_weighted_concat.py        # Step 1.7 — weighted concat + fusion
  augment_jobs.py               # Step 1.10 — template-based augmentation
  visualize_augmented.py        # Step 1.11 — full-scale UMAP (DataMapPlot)
  evaluate_final.py             # Step 1.9 — pytrec_eval metrics

data/
  jobs.json                     # Structured postings (Step 1.0)
  raw_embeddings.npy            # Raw baseline embeddings (Step 1.1)
  raw_metadata.json             # Baseline metadata
  baseline_metrics.json         # Proxy metrics (Step 1.3)
  golden_set.json               # Labeled evaluation set (Step 1.8)
  jobs_augmented.json           # Real + synthetic postings (Step 1.10)
  plots/
    raw_pca.png                 # Step 1.2
    raw_umap.png                # Step 1.2
    raw_tsne.png                # Step 1.2
    augmented_umap.png          # Step 1.11

backend/
  preprocessing.py              # Final production preprocessing class
  embeddings.py                 # Embedding pipeline (Phase 2)

tests/
  test_preprocessing.py         # Unit tests for preprocessing pipeline
```

---

### Phase 1 Success Criteria

- [x] All 27 real postings parsed into structured JSON with correct sections
- [x] Baseline embeddings stored and L2-normalized
- [x] PCA + UMAP + t-SNE visualizations generated and saved (matplotlib PNGs)
- [ ] Baseline metrics computed (self-retrieval, separation gap, NN audit)
- [ ] All 5 experiments run with before/after deltas recorded
- [ ] Golden set of 5–10 resumes created with relevance labels
- [ ] Final pytrec_eval metrics computed for baseline vs best config
- [ ] Best preprocessing config selected and documented
- [ ] Data augmentation runs without errors, produced postings pass manual spot-check
- [ ] Results documented in README with metric table and visuals

---

## Phase 2: Data Collection & Ingestion

**Duration:** 4–6 hours
**Goal:** Expand from 7 to 27+ real job postings, then later to 50+.

### Step 2.0 — Collect ML Job Postings (Complete)

- **Status:** 27 postings collected from LinkedIn
- **Sources:** LinkedIn
- **Role breakdown:** Data Scientist (8), ML Engineer (5), AI Engineer (3), Applied Researcher (2), AI Solutions Engineer (1), Performance Benchmarking (1), Data Engineering (1), DS/Analyst (1), Data Analytics Intern (1), AI Intern (1), Consultant Analyst (1), ML Recruitment (1), Decision Scientist (1)
- **Note:** LinkedIn no longer serves relevant ML roles at this volume. Collection stopped at 27.

### Step 2.0b — Future: Non-ML Postings for Contrast

- **Plan:** Collect 10–20 non-ML postings (Software Engineer, Product Manager, DevOps, Data Analyst, Project Manager) to add diversity
- **Why:** An all-ML dataset risks producing an overly homogeneous embedding space. Non-ML postings act as negative examples, making retrieval distinctions between ML sub-roles more meaningful and easier to evaluate.
- **Timing:** After Phase 1 preprocessing pipeline is validated on the 27 ML postings

### Step 2.1 — Re-run Phase 1 Pipeline on Expanded Dataset
- Parse all new postings through the `parse_postings.py` script
- Validate extraction quality on 10% random sample
- Re-run the best preprocessing config from Phase 1
- Regenerate golden set if needed (new roles may require new resume profiles)

### Step 2.2 — Generate Augmented Data
- Run `augment_jobs.py` on the collected dataset (27 seeds → ~200+ synthetic with non-ML contrast)
- Spot-check 10 synthetic postings for realism

---

## Phase 3: Embedding Pipeline & Similarity Engine

**Duration:** 4–6 hours

### Step 3.0 — Production Embedding Module
- Refactor `scripts/embed_baseline.py` and `exp_weighted_concat.py` into `backend/embeddings.py`
- Class-based API:
  ```python
  class EmbeddingPipeline:
      def __init__(self, model_name, preprocessing_config):
          ...
      def encode_job(self, job: dict) -> np.ndarray:
          ...
      def encode_resume(self, text: str) -> np.ndarray:
          ...
      def encode_batch(self, items: list[dict]) -> np.ndarray:
          ...
  ```

### Step 3.1 — Similarity Scoring Engine
- Implement `backend/retrieval.py`:
  ```python
  class SimilarityEngine:
      def __init__(self, embeddings: np.ndarray, metadata: list[dict]):
          ...
      def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[dict]:
          ...
      def precision_at_k(self, query_embedding, relevant_ids, k=5) -> float:
          ...
  ```

### Step 3.2 — Integration Tests
- Test end-to-end: resume text → embedding → ranked job list
- Verify ranking stability (same input → same output)
- Verify top-1 self-retrieval for all postings

---

## Phase 4: FastAPI Backend

**Duration:** 3–5 hours

### Step 4.0 — API Endpoints
```python
# backend/api.py
POST /upload-resume    # Accept text or PDF, return ranked jobs
POST /search           # Query by text, return top-k matches with scores
GET  /jobs             # List all jobs with metadata
GET  /jobs/{id}        # Single job details
```

### Step 4.1 — Resume Parsing
- Text: pass through directly
- PDF: use `pdfplumber` for text extraction
- Handle empty/invalid input with 400 errors

### Step 4.2 — Error Handling
- Input validation with Pydantic models
- Graceful error responses (4xx for client errors, 5xx for server errors)
- Logging with structlog or standard logging

---

## Phase 5: Docker Containerization

**Duration:** 2–3 hours

### Step 5.0 — Dockerfile
- Python 3.11 slim base
- Multi-stage build: separate build stage for heavy dependencies (torch, sentence-transformers)
- Expose port 8000
- Healthcheck endpoint

### Step 5.1 — Docker Compose
- Service definition for the API
- Volume mount for data directory (embeddings, jobs.json)
- Optional: add nginx reverse proxy for static file serving (frontend)

---

## Phase 6: Cloud Deployment

**Duration:** 3–5 hours

### Step 6.0 — Platform Selection
- Recommended: AWS Lightsail (simplest, fixed pricing)
- Alternative: GCP Cloud Run (serverless, scales to zero)
- Alternative: Azure Container Instances

### Step 6.1 — Deployment
- Push Docker image to container registry
- Configure environment variables (model path, data path, port)
- Set up HTTPS (Lightsail load balancer or Cloud Run managed cert)
- Verify public accessibility

---

## Phase 7: Frontend

**Duration:** 4–6 hours

### Step 7.0 — Single-Page Web Interface
- HTML/CSS/JS (no framework — keeps scope small)
- Resume upload form (drag-and-drop or file picker)
- Display ranked results as cards with similarity scores
- Show job title, company, match score, and a preview of the description
- Loading state while embedding is generated
- Error state for failed uploads/empty results

---

## Phase 8: CI/CD Pipeline

**Duration:** 2–3 hours

### Step 8.0 — GitHub Actions Workflow
- `.github/workflows/ci.yml`
- Jobs: test (pytest), build-docker (verify image builds)
- Trigger: push to main, pull requests
- Status badge in README

### Step 8.1 — Automated Deployment (Optional)
- If using Cloud Run or Lightsail with easy CI integration
- Deploy on successful build of main branch
- Otherwise: document manual deployment steps in README

---

## Phase 9: Documentation & Polish

**Duration:** 3–4 hours

### Step 9.0 — README
- Architecture diagram (ASCII art)
- Preprocessing experiment results table (from Phase 1)
- API documentation with curl examples
- Deployment instructions
- Known limitations
- Future work section

### Step 9.1 — Code Quality
- Type hints on all public functions
- Docstrings on all modules
- `pytest` tests for preprocessing, embedding, retrieval, API
- Run `ruff` or `black` for formatting

---

## Overall Timeline

| Phase | Description | Hours |
|---|---|---|
| 1 | Preprocessing & Embedding Optimization | 13–15 |
| 2 | Data Collection & Ingestion | 4–6 |
| 3 | Embedding Pipeline & Similarity Engine | 4–6 |
| 4 | FastAPI Backend | 3–5 |
| 5 | Docker Containerization | 2–3 |
| 6 | Cloud Deployment | 3–5 |
| 7 | Frontend | 4–6 |
| 8 | CI/CD Pipeline | 2–3 |
| 9 | Documentation & Polish | 3–4 |
| **Total** | | **38–53 hours** |

At 2–3 hours/day: **13–26 days** of focused work.
