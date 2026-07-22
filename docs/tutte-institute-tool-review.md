# Tutte Institute Tool Review — Relevance to Semantic Career Mapping

> Date: 2026-07-22
> Context: Systematic review of open-source tools published by the Tutte Institute for Mathematics and Computing (TIMC), assessing relevance to the Semantic Career Mapping Platform pipeline.

---

## Table of Contents

1. [Background: What Is the Tutte Institute?](#background-what-is-the-tutte-institute)
2. [Tool Catalog](#tool-catalog)
3. [Relevance Analysis by Project Phase](#relevance-analysis-by-project-phase)
4. [Recommended Integrations](#recommended-integrations)
5. [Licensing & Installation](#licensing--installation)

---

## 1. Background: What Is the Tutte Institute?

The [Tutte Institute for Mathematics and Computing](https://github.com/TutteInstitute) (TIMC) is a Canadian government research institute — part of the Communications Security Establishment (CSE), Canada's signals intelligence agency. Named after [W. T. Tutte](https://en.wikipedia.org/wiki/W._T._Tutte), a Bletchley Park codebreaker who later became a Canadian citizen, TIMC conducts fundamental research in cryptography and data science.

Their open-source tools have had outsized impact on the Python data science ecosystem. **UMAP** (8,177 GitHub stars) is now a standard dimensionality reduction technique used across biology, NLP, and visualization. **HDBSCAN** is the default density-based clustering algorithm in scikit-learn-contrib. These tools were originally developed for malware strain analysis at CSE before being released as open source.

The "TIMC vector toolkit" is their meta-package bundling these tools for exploratory data analysis of high-dimensional vector spaces — closely aligned with the goals of this project.

---

## 2. Tool Catalog

All tools are Python, BSD-3-Clause licensed, and available on PyPI. Star counts are as of July 2026.

### Core Embedding-Space Tools

| Tool | Stars | Description | Relevant? |
|---|---|---|---|
| **[UMAP](https://github.com/lmcinnes/umap)** | 8,177 | Uniform Manifold Approximation and Projection — dimensionality reduction | **Already planned** (Steps 1.2, 1.11) |
| **[HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan)** | — | Hierarchical density-based spatial clustering | **Yes** — cluster discovery, role taxonomy validation |
| **[EVōC](https://github.com/TutteInstitute/evoc)** | 297 | Embedding Vector Oriented Clustering — UMAP + HDBSCAN pipeline tuned for embedding vectors | **Future** — packaged job family discovery |
| **[fast_hdbscan](https://github.com/TutteInstitute/fast_hdbscan)** | 147 | Multi-core HDBSCAN optimized for low-dimensional Euclidean spaces | **No** — overkill at current dataset size (27 postings) |

### Preprocessing & Vectorization

| Tool | Stars | Description | Relevant? |
|---|---|---|---|
| **[vectorizers](https://github.com/TutteInstitute/vectorizers)** | 102 | scikit-learn-compatible vectorizers for unstructured sequence data (text, categorical, histogram, distribution) | **Maybe** — alternative/complement to manual spaCy skill extraction |

### Visualization & Labeling

| Tool | Stars | Description | Relevant? |
|---|---|---|---|
| **[DataMapPlot](https://github.com/TutteInstitute/datamapplot)** | 1,008 | Presentation-ready interactive data map plots on top of UMAP coordinates | **Yes** — polished visualization upgrade for Steps 1.2, 1.11, and deferred "Interactive UMAP" feature |
| **[ThisNotThat (TNT)](https://github.com/TutteInstitute/thisnotthat)** | 154 | Interactive Jupyter widget for labeling regions of a data map | **Future** — could assist golden set creation or cluster review |
| **[Glasbey](https://github.com/TutteInstitute/glasbey)** | — | Algorithmic categorical color palettes for data visualization | **No** — minor styling, not worth a dependency |

### Topic Modeling & Naming

| Tool | Stars | Description | Relevant? |
|---|---|---|---|
| **[Toponymy](https://github.com/TutteInstitute/toponymy)** | 97 | Automatic topic naming for clusterable data — generates human-readable names for clusters based on distinguishing terms | **Future** — auto-naming discovered job families |
| **[EnsTop](https://github.com/lmcinnes/enstop)** | — | Ensemble topic modeling with pLSA | **No** — different approach, not embedding-based |

### Infrastructure

| Tool | Stars | Description | Relevant? |
|---|---|---|---|
| **[install-tools](https://github.com/TutteInstitute/install-tools)** | — | Docker images, standalone installer generators for the TIMC vector toolkit | **Reference** — Docker image `tutteinstitute/vector-toolkit` available; notes on air-gapped deployment |
| **[tutorials](https://github.com/TutteInstitute/tutorials)** | — | Jupyter notebooks on bag-of-words, topic modeling with Tutte tools | **Reference** — methodology tutorials |

---

## 3. Relevance Analysis by Project Phase

### Phase 1: Preprocessing & Embedding Optimization

| Step | Current Approach | Tutte Tool | Enhancement |
|---|---|---|---|
| 1.2 (Visualization) | matplotlib PCA + UMAP scatter | **DataMapPlot** | Interactive, zoomable, labeled plots. Adds cluster-label overlays automatically. Saves as self-contained HTML. |
| 1.3 (Baseline Evaluation) | Self-retrieval + separation gap + manual NN audit | **HDBSCAN** | Add cluster structure analysis as a fourth metric: does HDBSCAN's cluster assignment match hand-labeled `role_category`? Quantifies "are the clusters real?" without needing a golden set. |
| 1.6 (Skill Extraction) | Manual skill vocabulary + spaCy PhraseMatcher | **vectorizers** | Distribution-based vectorization can complement keyword extraction. A spike experiment could compare: (a) manual keyword list → weighted concat vs (b) vectorizer-derived TF-IDF skill vectors → fused with Sentence Transformer embedding. Deferred — manual approach ships faster. |
| 1.7 (Weighted Concatenation) | Multi-field vector fusion (0.4/0.3/0.3) | — | No direct Tutte tool for this, but the TIMC vector toolkit philosophy (UMAP + HDBSCAN + DataMapPlot) provides the evaluation infrastructure to judge whether the fusion improves separation. |
| 1.11 (Augmented Visualization) | matplotlib UMAP with real/synthetic markers | **DataMapPlot** | Self-contained interactive HTML plots instead of static PNGs. Better for README — readers can hover to see labels. |

**HDBSCAN as an Interleaved Metric (Step 1.3b)**

After computing baseline embeddings, run:

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2,
    min_samples=1,
    metric="euclidean",
    cluster_selection_method="eom",
)
labels = clusterer.fit_predict(embeddings)

# Compare HDBSCAN labels to hand-labeled role_category
# Perfect alignment is not expected — but large disagreements
# indicate mislabeled postings or embedding artifacts.
```

This adds a data-driven sanity check that costs ~10 lines of code. It answers: "If an unsupervised algorithm recovers your hand-labeled categories, the categories are grounded in the data."

### Phase 2: Data Collection & Ingestion

No direct Tutte tool relevance. HDBSCAN could validate whether expanded dataset (27 postings) maintains coherent cluster structure.

### Phase 3: Embedding Pipeline & Similarity Engine

No direct Tutte tool relevance. The similarity engine uses cosine distance, which is independent of the visualization/clustering tools.

### Phase 7: Frontend

**DataMapPlot** could produce self-contained HTML data maps that embed directly in the frontend without a Python backend for the viz. This aligns with the deferred "Interactive UMAP visualization of embedding space" feature.

### Future Work (Deferred Features)

| Future Work Feature | Tutte Tool(s) | How |
|---|---|---|
| **Clustering and job family discovery** | **EVōC** | `evoc.fit(embeddings)` — drop-in pipeline that clusters and labels job families. Single import instead of chaining UMAP + HDBSCAN manually. |
| **Auto-naming discovered clusters** | **Toponymy** | Given a cluster of job postings, generates a label like "Senior ML Engineers — PyTorch, AWS, Kubernetes." Complements EVōC output. |
| **Interactive UMAP visualization** | **DataMapPlot** | Replaces matplotlib scatter with zoomable, hover-labeled HTML viz. Export as static HTML for embedding in frontend. |
| **Interactive labeling** | **ThisNotThat (TNT)** | Jupyter widget for manually labeling regions of the embedding space. Could accelerate golden set creation or cluster review workflows. |

---

## 4. Recommended Integrations

### Immediate (Phase 1 — Adds Value, Low Cost)

**HDBSCAN** — Add to `requirements.txt` and use as an interleaved metric in Step 1.3. Ten lines of code. Validates that your hand-labeled `role_category` taxonomy aligns with the natural cluster structure of the embedding space. No impact on the production pipeline — purely an evaluation/debugging tool.

```bash
pip install hdbscan
```

**DataMapPlot** — Add to `requirements.txt`. Upgrade Step 1.2 and Step 1.11 visualizations from static matplotlib PNGs to interactive HTML data maps. Better for the README and more impressive to reviewers. The API is a drop-in replacement for matplotlib scatter:

```python
import datamapplot

# After UMAP reduction to 2D
datamapplot.create_plot(
    coords_umap,
    labels=role_labels,       # colored by role_category
    title="Job Posting Embedding Space",
    sub_title="27 ML postings, all-MiniLM-L6-v2, cosine metric",
)
# Saves as interactive HTML by default
```

One-line replacement. Check that it doesn't pull in heavy additional dependencies beyond what UMAP already requires.

### Future (After MVP Ships)

**EVōC + Toponymy** — Together form a "discover and name job families" pipeline. One import each. Add when the deferred "Clustering and job family discovery" feature is activated.

**vectorizers** — Worth a spike experiment comparing distribution-based skill vectorization against the manual spaCy approach. Defer — the manual approach ships faster and is easier to explain in the README.

### Do NOT Add

- **fast_hdbscan** — Overkill at 27 postings. Standard HDBSCAN runs in milliseconds.
- **Glasbey** — DataMapPlot handles colors automatically. Extra dependency for no gain.
- **EnsTop** — pLSA topic modeling is a different paradigm from embedding-based retrieval. Adds complexity without clear benefit.
- **ThisNotThat** — Interactive labeling is useful for manual workflows but doesn't contribute to the automated pipeline. Revisit if golden set creation becomes a bottleneck at larger scale.

---

## 5. Licensing & Installation

All Tutte Institute tools are **BSD-3-Clause** licensed. No copyleft or viral license concerns. Fully compatible with this project's MIT license.

### Installation Options

**Individual packages (recommended — minimal dependencies):**

```bash
pip install hdbscan datamapplot
```

**Meta-package (all-in-one, heavier):**

```bash
pip install timc-vector-toolkit
# Installs: umap-learn, hdbscan, datamapplot, evoc, vectorizers, toponymy
```

**Docker image** (for deployment):

```bash
docker pull tutteinstitute/vector-toolkit
```

For the MVP, install individually to keep the Docker image small. The meta-package adds toponymy, evoc, and vectorizers — none of which are needed for Phase 1. Add them later when the corresponding features are activated.

---

## Decision Log

| Decision | Rationale | Date |
|---|---|---|
| Add HDBSCAN to Phase 1 evaluation pipeline | 10 lines of code validates role taxonomy alignment; zero production impact | 2026-07-22 |
| Upgrade visualizations to DataMapPlot | Static PNGs → interactive HTML; one-line API change; better README signal | 2026-07-22 |
| Defer EVōC + Toponymy to Future Work | Feature activation condition: after MVP ships and clustering is prioritized | 2026-07-22 |
| Defer vectorizers spike experiment | Manual skill extraction ships faster; vectorizers is an optimization, not a requirement | 2026-07-22 |
| Install individually, not via meta-package | Keeps Docker image smaller; unmet dependencies are future work, not current needs | 2026-07-22 |
