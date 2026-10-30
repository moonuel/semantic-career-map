# Semantic Clustering Methods and Quality Measures for Text Embeddings — Research Report

> Date: 2026-10-30
> Context: Semantic Career Mapping Platform — algorithm and evaluation survey for the job-posting clustering layer. The production pipeline embeds LLM-cleaned postings with L2-normalized `all-MiniLM-L6-v2` vectors (384-d) and is moving from cosine similarity toward job-family discovery ([data architecture §7](../architecture/data.md#7-embeddings), [Report 024](024-embedding-models-for-semantic-clustering.md)). Complements [Report 021](021-clustering-metrics-at-scale.md) (metric scaling and AWS placement) and [Report 012](012-evoc-embedding-clustering.md) (EVōC internals) by covering the **method landscape and the evaluation protocol** end to end.

---

## Table of Contents

1. [Summary](#1-summary)
2. [What Makes Semantic Clustering Different](#2-what-makes-semantic-clustering-different)
3. [Method Landscape](#3-method-landscape)
4. [Choosing a Method](#4-choosing-a-method)
5. [Quality Measures](#5-quality-measures)
6. [Recommended Evaluation Protocol](#6-recommended-evaluation-protocol)
7. [Scale and Engineering Notes](#7-scale-and-engineering-notes)
8. [Fit Against This Project](#8-fit-against-this-project)
9. [Risks and Limitations](#9-risks-and-limitations)
10. [Sources](#10-sources)

---

## 1. Summary

There is no universal "best" semantic clustering algorithm. The choice is governed by the **output contract** the downstream product needs, the **geometry** of the embedding space, and the **implementation constraints** (corpus size, CPU-only, dependencies). Benchmark studies across text-embedding datasets consistently find that no single algorithm wins everywhere; performance depends on the interaction between representation, algorithm, distance metric, and corpus structure [16][17].

Four conclusions drive this report:

1. **Pick the algorithm by the shape of the answer.** A fixed-`k` partition, a hierarchy, soft memberships, and a partition with explicit noise are different products. K-Means cannot answer "which postings are unlike anything else?" and HDBSCAN cannot guarantee a requested number of job families.
2. **Embedding geometry invalidates textbook defaults.** On L2-normalized vectors the operative space is the unit sphere with cosine distance. Centroid metrics assume convex, roughly isotropic clusters; high-dimensional embeddings violate that. Distance concentration and anisotropy compress separation, so internal indices are **comparative, not absolute**.
3. **Clustering quality is multi-dimensional.** Geometric coherence, agreement with labels, stability under perturbation, and downstream usefulness measure different things and routinely disagree. A defensible evaluation reports all four, or explicitly justifies dropping one.
4. **For this project the practical order is: fix encoder coverage, then compare a small clusterer grid on fixed vectors, then judge with external labels plus stability plus human audit.** The current MiniLM encoder truncates every golden-set posting (mean 539 tokens against a 256-token limit) [Report 024 §1](024-embedding-models-for-semantic-clustering.md#1-executive-summary), so any clustering comparison run before coverage is fixed is a comparison of truncated prefixes, not of full job postings.

| Default recommendation | Algorithm | Why |
|---|---|---|
| Fixed-`k` baseline | Spherical K-Means or MiniBatch K-Means | Fast, deterministic with a seed, easy to compare across encoders |
| Discovery (unknown `k`, noise expected) | HDBSCAN or EVōC | Variable density, explicit noise, no forced assignment of outliers |
| Interpretable hierarchy | Agglomerative clustering | Produces a dendrogram and multi-level cuts for taxonomy work |
| Graph-native / very large | Leiden or Louvain on a k-NN graph | Reuses the neighbor graph; scales without a dense distance matrix |
| Soft membership | von Mises–Fisher (vMF) mixture | Directional model aligned with cosine geometry; per-cluster confidence |

The canonical quality stack for this project should be: **ARI/AMI/V-measure against function tags** (when labels exist), **k-NN purity and persistence** for label-free checks, **bootstrap stability**, and **representative-document audit**. Silhouette, Calinski-Harabasz, and Davies-Bouldin remain useful only as same-encoder relative signals.

---

## 2. What Makes Semantic Clustering Different

Semantic clustering is not topic modeling with better vectors. Six properties of sentence embeddings change the algorithm and metric choices.

| Property | Consequence | Design rule |
|---|---|---|
| **Cosine geometry on the unit sphere** | Euclidean centroids and spherical-Gaussian assumptions are approximate at best | Use cosine distance, spherical K-Means, or normalize centroids every iteration; keep the distance metric identical between clusterer and metric |
| **High dimensionality and distance concentration** | Nearest and farthest neighbors become similar; Silhouette ceilings collapse toward zero as dimension grows (arXiv:2509.08625) | Treat internal scores as **relative rankings**, never as absolute quality labels; prefer graph or neighbor-based measures |
| **Anisotropy and hubness** | A few directions dominate; some points are neighbors of many unrelated documents | Measure neighbor purity and inspect hubs; consider centering/whitening as a separate, re-evaluated representation step |
| **Variable density** | Global `eps` in DBSCAN fails; job families have different tightness | Prefer HDBSCAN/OPTICS/EVōC, which adapt to local density |
| **Noise is first-class** | Real corpora contain duplicates, unique senior roles, and malformed postings | Use clusterers that emit `-1`, and metrics that score noise explicitly rather than silently dropping it |
| **Labels are not inherent** | Clusters are useful only if they correspond to a business or user concept | Define the target (role function, seniority, skill family) before choosing the algorithm; evaluate against labels or a documented rubric |

A second distinction matters for evaluation. There are three objects being measured:

1. **The embedding space** — are vectors semantically meaningful and well spread? (self-retrieval, separation gap, alignment/uniformity, downstream retrieval)
2. **The partition** — does the clustering recover structure in that space? (internal, external, stability)
3. **The labels** — are the resulting groups interpretable and actionable? (representative documents, cluster naming, human or LLM audit)

A good embedding with a bad partition is possible, and so is a geometrically coherent partition that groups the wrong concept. The protocol in [§6](#6-recommended-evaluation-protocol) keeps these separate.

---

## 3. Method Landscape

### 3.1 Pipeline patterns

Before comparing algorithms, choose the pipeline shape. These are not interchangeable.

| Pattern | Description | Strength | Risk |
|---|---|---|---|
| **Direct high-dimensional** | Cluster the 384-d vectors directly | Preserves all encoded signal; no projection loss | Distances concentrate; some algorithms degrade in high `d` |
| **Linear reduction, then cluster** | PCA to 50–256 dims, then cluster | Denoises and speeds up distance computation; PCA `n_dims=256` is already named for this project ([Report 009](../experiments/009-pca-informativeness.md)) | Components are variance-ranked, not semantics-ranked; re-tune clusterer |
| **Nonlinear reduction, then cluster** | UMAP/t-SNE to 5–15D, then HDBSCAN | Separates manifolds; common in BERTopic-style pipelines | UMAP distorts density and global distances; cluster results depend heavily on UMAP parameters |
| **Graph-first** | Build one k-NN graph, then run density or community detection on it | Reuses the expensive neighbor search; scales to large `n`; EVōC's architecture | Graph construction parameters (`k`, metric) become part of the model |
| **Cluster, then label** | Cluster first; derive labels with c-TF-IDF, keyword extraction, or an LLM | Keeps labels separable from geometry; supports human review | Labels are interpretations, not ground truth; do not evaluate the clusterer with labels it generated |

**Recommendation:** cluster on the full-dimensional normalized vectors, or on a PCA reduction whose explained variance is reported. Do **not** cluster on 2-D UMAP coordinates; 2-D is a visualization product, and density/cluster structure in 2-D is not the structure of the embedding space. If UMAP is used, use an intermediate 5–15-D embedding and record the parameters as part of the clustering model.

### 3.2 Algorithm families at a glance

| Family | Representative algorithms | Output | Requires `k`? | Noise? | Cluster shape | Typical cost | Best fit |
|---|---|---|---|---|---|---|---|
| **Centroid / partition** | K-Means, Spherical K-Means, MiniBatch K-Means, Bisecting K-Means | Hard partition | Yes | No | Convex / spherical | `O(n k d)` per iteration | Fast baseline, fixed taxonomy size, very large `n` |
| **Hierarchical** | Agglomerative (Ward, average, complete), HAC | Dendrogram + cuts | No (cut later) | No (unless filtered) | Linkage-dependent | `O(n²)` memory/time for naive; approximate variants exist | Small/medium corpora, multi-level job families, interpretability |
| **Density** | DBSCAN, OPTICS, HDBSCAN, EVōC | Partition + noise, sometimes hierarchy | No | Yes (`-1`) | Arbitrary | Roughly `O(n log n)` with a k-NN index; exact variants higher | Discovery, variable density, outlier retention |
| **Probabilistic** | Gaussian mixture, vMF mixture | Soft memberships | Yes | No (low-confidence points can be flagged) | Elliptical / directional | `O(n k d²)` for full covariance; less for diagonal/tied | Soft assignment, confidence scores, overlap |
| **Graph / community** | Spectral clustering, Leiden, Louvain, label propagation | Community labels | Resolution parameter | Sometimes | Graph-connected | Spectral `O(n³)` dense, sparse eigensolvers cheaper; Leiden near-linear per iteration | k-NN graph pipelines, very large corpora |
| **Neural / topic pipelines** | BERTopic, Top2Vec, deep embedded clustering | Pipeline output (often UMAP + HDBSCAN + labels) | Usually no | Usually yes | Inherited from base clusterer | Pipeline-dependent | Quick topic-like baselines; less control over geometry |

### 3.3 Centroid and partition methods

**K-Means** minimizes within-cluster squared Euclidean distance. For unit-length vectors, Euclidean distance is monotone in cosine similarity, so K-Means on normalized embeddings is effectively a cosine partitioner; **Spherical K-Means** makes this explicit by normalizing centroids after each update. scikit-learn has no dedicated `SphericalKMeans` estimator, so the practical recipe is `KMeans` or `MiniBatchKMeans` on L2-normalized inputs; if centroid normalization matters for interpretation, normalize the fitted `cluster_centers_` before using them for assignment or inspection.

- **Strengths:** fast, simple, deterministic with `random_state`, easy to compare across encoders, produces centroids that can be inspected and used for nearest-centroid assignment.
- **Weaknesses:** must choose `k`; assumes roughly convex, similar-sized clusters; every outlier is forced into a cluster; sensitive to initialization (use `n_init` > 1).
- **`k` selection:** elbow on inertia, Silhouette (comparative only), gap statistic, or stability across seeds/subsamples. For job families, domain constraints should bound `k`: a useful family must contain enough postings to be actionable.
- **Bisecting K-Means** gives a hierarchical sequence of binary splits and is a cheap alternative when a coarse-to-fine taxonomy is wanted.

**When to use:** a first baseline, a fixed number of job families, a production fallback with predictable cost, or a comparison point for density methods.

### 3.4 Hierarchical methods

Agglomerative clustering starts with one point per cluster and merges the closest pair until a stopping rule is met. Linkage determines the geometry:

| Linkage | Merge criterion | Behavior on embeddings |
|---|---|---|
| **Ward** | Minimum increase in within-cluster variance | Compact, roughly equal-sized clusters; Euclidean assumption; often strong on normalized vectors |
| **Average** | Minimum mean pairwise distance | More tolerant of elongated clusters; commonly used with cosine |
| **Complete** | Minimum maximum pairwise distance | Compact clusters; sensitive to outliers |
| **Single** | Minimum nearest-pair distance | Chains through density bridges; usually poor for semantic clusters |

- **Strengths:** no `k` required until cut time; dendrogram exposes multi-granularity structure; deterministic; natural for taxonomy construction and human review.
- **Weaknesses:** `O(n²)` distance storage and `O(n² log n)`-ish merge cost for naive implementations; sensitive to linkage and distance; a single cut may not represent all levels.
- **Quality signals:** **cophenetic correlation** (how faithfully the dendrogram preserves original pairwise distances) and stability of cuts across bootstrap samples.
- **Scaling:** use a sampled subset (e.g. 2–10k) for the dendrogram, then assign remaining points by nearest centroid or nearest neighbor; or use a graph-based community method instead.

**When to use:** the golden set and taxonomy exploration, where a human needs to see the merge order; multi-level job families; small corpora.

### 3.5 Density methods

Density methods find regions of the space that are denser than their surroundings and leave sparse points unassigned.

- **DBSCAN** uses a global radius `eps` and minimum points `min_samples`. It is a poor default for 384-d embeddings because a single global density threshold rarely fits every job family, and distance concentration makes `eps` hard to tune.
- **OPTICS** removes the need for one global `eps` by ordering points by reachability, but extracting clusters from the reachability plot adds its own parameters.
- **HDBSCAN** builds a mutual-reachability graph, computes a minimum spanning tree, condenses it into a cluster hierarchy, and extracts clusters that persist across density levels. Key parameters are `min_cluster_size` (smallest group worth calling a cluster) and `min_samples` (conservativeness toward noise). It emits `-1` for noise and `probabilities_`/membership strengths. `sklearn.cluster.HDBSCAN` is available in recent scikit-learn; verify that the installed version supports the chosen metric, or pass a precomputed distance matrix.
- **EVōC** is an embedding-specialized pipeline: one k-NN graph → low-dimensional node embedding → parallel Borůvka MST → condensed tree → PLSCAN persistence peaks. It returns multiple cluster layers, membership strengths, a cross-layer cluster tree, and duplicate pairs; `-1` is noise [Report 012](012-evoc-embedding-clustering.md). EVōC is attractive here because it avoids the UMAP→HDBSCAN duplication and exposes **persistence** as a built-in quality signal.

**Strengths:** arbitrary cluster shapes, variable density, explicit noise, no forced `k`. **Weaknesses:** parameter sensitivity (`min_cluster_size` especially), a large noise fraction on heterogeneous corpora, and no guarantee that the number of clusters matches a product requirement. A 20–40% noise fraction can be a legitimate result on a diverse job corpus, not necessarily a failure; report it and inspect it.

**When to use:** discovery of job families, outlier/duplicate investigation, any setting where "none of the above" is a valid answer.

### 3.6 Probabilistic methods

- **Gaussian mixture models (GMM)** assign soft responsibilities and can model elliptical clusters. In 384-d, full covariance is parameter-heavy (`O(d²)` per component) and unstable without regularization; use diagonal or tied covariance, or reduce dimensions first. GMM still requires `k` and assumes Gaussian components.
- **von Mises–Fisher (vMF) mixtures** are the directional analogue on the unit sphere and align better with cosine geometry. They provide soft memberships and a principled likelihood, but fitting is more involved and less standardized than K-Means.
- **Model selection:** BIC/AIC for GMM, but only within the same representation; likelihood is not comparable across different embedding models or preprocessing pipelines.

**When to use:** overlapping job families, a need for per-posting confidence, or a soft assignment layer on top of a hard partition.

### 3.7 Graph and community methods

Build a k-NN graph (or reuse the one the density clusterer already builds), then partition the graph:

- **Spectral clustering** uses the graph Laplacian eigenvectors; it can recover non-convex clusters but dense eigendecomposition is expensive at `n` in the thousands and above.
- **Leiden** improves on Louvain with guaranteed connected communities and a resolution parameter. It is fast, scales well, and is a strong choice when the k-NN graph is already the system of record. Modularity has a resolution limit: it may merge small but genuine job families, so sweep the resolution and inspect.
- **Label propagation** is cheap and useful as an initialization or for semi-supervised seeding, but can collapse to one giant community on noisy graphs.

**Quality signals:** modularity, conductance, and stability across graph perturbations. These judge **community structure**, not semantic correctness; pair them with labels or human audit.

### 3.8 Neural and topic-pipeline methods

**BERTopic** (embedding → UMAP → HDBSCAN → c-TF-IDF labels) and **Top2Vec** (joint document/word embedding → dimensionality reduction → density clustering) are convenient end-to-end baselines. They are pipelines, not single algorithms, and inherit the parameter sensitivity of UMAP and HDBSCAN. **Deep embedded clustering** jointly learns representations and cluster assignments; it requires training and is not a sensible default for a 9.6k-posting corpus with a CPU-first posture.

Use these as reference points when the question is "what would an off-the-shelf topic pipeline produce?", not as the production clusterer unless their added dependencies and opaque intermediate steps are acceptable.

---

## 4. Choosing a Method

### 4.1 Decision guide

| Requirement | First choice | Second choice | Avoid |
|---|---|---|---|
| Fixed, known number of families | Spherical K-Means / MiniBatch K-Means | GMM with diagonal covariance | DBSCAN |
| Unknown `k`, outliers should stay unassigned | HDBSCAN | EVōC | K-Means |
| Multi-level hierarchy / taxonomy | Agglomerative (average or Ward) | Bisecting K-Means | Single linkage |
| Soft membership / confidence | vMF mixture | GMM | Hard K-Means alone |
| Graph-native, very large corpus | Leiden on k-NN graph | Louvain | Dense spectral |
| Small corpus, human-in-the-loop | Agglomerative | HDBSCAN | Deep clustering |
| One-pass baseline across encoders | Spherical K-Means | MiniBatch K-Means | UMAP + HDBSCAN with per-encoder tuning |

### 4.2 Parameters that must be recorded

| Parameter | Affects | Practical starting point |
|---|---|---|
| `k` | Granularity of centroid/GMM methods | Domain-bounded range; select by stability, not only Silhouette |
| `min_cluster_size` | Smallest density cluster | Smallest domain-meaningful family; at 9.6k postings, roughly 30–100 is a reasonable starting range, not a rule |
| `min_samples` | Noise conservativeness in HDBSCAN | 1–10; higher ⇒ more noise |
| `resolution` | Leiden/Louvain granularity | Sweep; inspect number and size of communities |
| Distance metric | Every distance-based method and metric | Cosine on L2-normalized vectors; keep identical across clusterer and evaluation |
| PCA dimensions | Reduced representation | 256 is the project's named default in the centered frame; report explained variance |
| UMAP `n_neighbors` / `min_dist` | 2-D maps and any UMAP-based pipeline | Treat as a model choice; never tune only for a prettier map |
| `random_state` / seed | K-Means, GMM, approximate methods | Fix and record; run multiple seeds |
| Sample / filter | All methods | Fixed sampled IDs; document population, seed, and exclusions |

### 4.3 Tuning guardrails

- **Do not tune on the evaluation labels.** If ARI/AMI against function tags is the target, choose hyperparameters on a separate split, a stability criterion, or a domain prior; otherwise the score is optimistic.
- **Hold the representation fixed within a clusterer comparison.** Changing encoder, cleaning variant, PCA, and clusterer at once makes the result uninterpretable.
- **Record the full parameter set and artifact hash** so a cluster run can be reproduced after an encoder or cleaning change.
- **Prefer plateaus over peaks.** A parameter region with stable metrics and stable cluster counts is more trustworthy than a single sharp optimum.

---

## 5. Quality Measures

### 5.1 Four lenses

| Lens | Question | Representative measures | Needs labels? |
|---|---|---|---|
| **Internal / geometric** | Are clusters compact and separated in the embedding space? | Silhouette, Calinski-Harabasz, Davies-Bouldin, DBCV/DISCO, k-NN purity, modularity, persistence | No |
| **External / supervised** | Do clusters agree with a known concept? | ARI, AMI/NMI, homogeneity, completeness, V-measure, purity, FMI | Yes |
| **Stability** | Does the clustering reproduce under perturbation? | Bootstrap/subsample ARI or Jaccard, seed/parameter stability | No |
| **Task / human** | Are the clusters useful and interpretable? | Retrieval metrics, downstream classification, representative-document audit, label quality | Partly |

### 5.2 Internal measures

| Measure | Definition (intuition) | Range | Cost | Caveat |
|---|---|---|---|---|
| **Silhouette** | Per point `(b - a) / max(a, b)`, `a` = mean intra-cluster distance, `b` = mean distance to nearest other cluster | `[-1, 1]` | `O(n²)` | Collapses toward 0 with dimension; assumes convex clusters; comparative only |
| **Calinski-Harabasz** | Between-cluster scatter / within-cluster scatter, scaled by `(n-k)/(k-1)` | `[0, ∞)` | `O(n k d)` | Rewards spherical, equal-variance clusters; monotone-ish in `k` |
| **Davies-Bouldin** | Mean over clusters of worst `(σ_i + σ_j) / d(c_i, c_j)` | `[0, ∞)`, lower better | `O(n k d)` | Centroid-based; same convexity assumption |
| **DBCV** | Density sparseness/separation from mutual-reachability MSTs; noise-aware | `[-1, 1]` | High; k-NN + per-cluster MST | **Non-deterministic**: leaf points are excluded, and MSTs are not unique; do not compare single runs |
| **DISCO** | Deterministic density-based index with explicit noise scoring | `[-1, 1]` | `O(n²)` naive | 2026 successor to DBCV; check implementation maturity |
| **k-NN purity** | For each point, fraction of its `k` nearest neighbors sharing its label; average over non-noise points | `[0, 1]` | `O(n k d)` | Chance level depends on cluster sizes; compare to a permutation null |
| **Modularity / conductance** | Community quality on a graph | Modularity `[-0.5, 1]`; conductance lower better | Graph-dependent | Resolution limit; not comparable across graphs |
| **Persistence** | How long a cluster survives across density scales (HDBSCAN/EVōC) | `[0, 1]` typically | Built into the clusterer | Only available for density hierarchies; not a cross-algorithm score |
| **Cophenetic correlation** | Correlation between dendrogram merge distances and original pairwise distances | `[-1, 1]` | `O(n²)` | Hierarchical methods only; high value does not prove semantic correctness |

**Interpretation rule:** internal scores answer "which of these clusterings is more coherent **under this encoder and distance metric**?" They do not answer "is this a good job-family taxonomy?" and they are not comparable across encoders.

### 5.3 External measures

External measures require a target labeling. In this project the candidates are experiment 006 function tags (multi-category), experiment 005/008 golden role labels (small, hand-curated), or a future ESCO mapping ([Report 016](016-esco-eu-jobs-taxonomy.md)).

| Measure | Definition | Range | Notes |
|---|---|---|---|
| **ARI** | Pairwise agreement with labels, corrected for chance | `[-1, 1]`, 0 = chance | Symmetric; the default partition comparison; sensitive to cluster count |
| **AMI / NMI** | Mutual information between clustering and labels | `[0, 1]` | AMI is chance-corrected; NMI is not and favors fine partitions |
| **Homogeneity** | Each cluster contains only one class | `[0, 1]` | Diagnoses "too merged" |
| **Completeness** | All members of a class fall in one cluster | `[0, 1]` | Diagnoses "too split" |
| **V-measure** | Harmonic mean of homogeneity and completeness | `[0, 1]` | The pair is more diagnostic than the scalar |
| **Purity** | Fraction of points in the majority class of their cluster | `[0, 1]` | Trivially maximized by singleton clusters; never use alone |
| **FMI** | Geometric mean of pairwise precision and recall | `[0, 1]` | Largely redundant with ARI in practice |

**Noise policy.** For HDBSCAN/EVōC, keep `-1` as a distinct label when computing ARI/AMI if the reference labels have no "noise" class; document that choice. Dropping noise inflates scores and hides the clusterer's behavior. Report the noise fraction alongside every external score.

**Multi-label targets.** If function tags overlap (a posting can be both "research" and "infrastructure"), a single-label partition metric is the wrong tool. Use multi-label metrics such as example-F1, micro-F1, or Hamming loss against the tag matrix, or evaluate one binary tag at a time. ARI assumes a single-label partition; forcing multi-label data into it produces an arbitrary score.

**Small-label caution.** With a few dozen labeled postings, external scores have wide uncertainty. Report bootstrap confidence intervals and avoid ranking methods on differences smaller than that interval.

### 5.4 Stability measures

Stability asks whether the structure is real or an artifact of the sample, seed, or hyperparameters.

- **Bootstrap / subsample ARI.** Run the clusterer on `m` resamples (e.g. 80% of points), compare each pair of runs on their intersection using ARI, and average. High mean ARI indicates reproducible structure.
- **Per-cluster Jaccard.** For each cluster in run A, find its best-matching cluster in run B and compute Jaccard; report the distribution, not only the mean. A few stable families plus many unstable ones is a common and informative result.
- **Seed stability.** For stochastic methods (K-Means initialization, GMM, approximate k-NN), vary the seed and compare labels.
- **Parameter stability.** Sweep `min_cluster_size`, `k`, or Leiden resolution and plot metric and cluster count. A method that only works at one parameter value is fragile.
- **Graph stability.** For k-NN/Leiden methods, perturb `k` and the neighbor-search seed.

Stability is the most expensive lens per run because it multiplies the number of cluster fits. Reserve it for the final shortlist, not the inner loop.

### 5.5 Task and retrieval measures

Clustering is often a means to a retrieval or classification end; measure that end directly when possible.

- **Self-retrieval** — fraction of postings whose nearest neighbor is themselves. A degeneracy check, not a quality score; it should be 100% for non-collapsed embeddings and says nothing about cluster usefulness.
- **Separation gap** — `mean(within-role cosine) - mean(cross-role cosine)`. This project's existing proxy; label-driven and comparative only.
- **Precision@k / MRR / NDCG** — retrieval quality on a golden query set. These measure the embedding plus retrieval pipeline, not the partition.
- **Downstream classification / assignment** — train a simple classifier on cluster IDs or cluster centroids and evaluate on held-out labels. If cluster IDs predict the target better than a random partition, the clustering carries signal.
- **Cluster-label quality** — sample representative documents per cluster and ask a human (or a rubric-driven LLM judge) whether they share the intended concept. Report agreement and disagreements; do not treat an LLM's label as ground truth.

### 5.6 Measure selection by clusterer

| Clusterer | Primary internal | Secondary internal | External | Stability |
|---|---|---|---|---|
| K-Means / spherical K-Means | Silhouette, CH, DB | k-NN purity | ARI/AMI/V-measure | Seed/subsample ARI |
| Agglomerative | Cophenetic correlation | Silhouette at chosen cut | ARI/AMI/V-measure | Cut stability, bootstrap |
| HDBSCAN | DBCV or DISCO | k-NN purity, persistence | ARI/AMI/V-measure (noise kept) | Subsample ARI, parameter sweep |
| EVōC | Persistence, DBCV/DISCO | k-NN purity, layer diversity | ARI/AMI per layer | Layer stability, subsample ARI |
| GMM / vMF | BIC/AIC (within encoder) | Silhouette on hard labels | ARI/AMI | Seed/subsample ARI |
| Leiden / Louvain | Modularity, conductance | k-NN purity | ARI/AMI | Graph perturbation, resolution sweep |

### 5.7 Evaluation pitfalls

- **Absolute Silhouette worship.** On high-dimensional normalized embeddings, a Silhouette of 0.1 can be a good relative result and 0.5 can be an artifact of tiny clusters. Compare variants, not thresholds.
- **The purity trap.** Many small clusters maximize purity while destroying usability. Always report cluster-size distribution and noise fraction.
- **NMI without chance correction.** NMI rises with the number of clusters; prefer AMI for comparisons.
- **Dropping noise.** Silently removing `-1` inflates every external score. Keep it and report the fraction.
- **Clustering 2-D UMAP output.** The map is not the space. Evaluate on high-dimensional vectors or the k-NN graph.
- **Label leakage.** Tuning `k`, `min_cluster_size`, or UMAP parameters against the same labels used for the final ARI/AMI makes the score optimistic.
- **Comparing internal metrics across encoders.** Silhouette, CH, DB, DBCV, and DISCO are not on a common scale across embedding models. Use external/task metrics for cross-encoder decisions.
- **Treating correlated metrics as independent evidence.** Silhouette, CH, and DB often rank methods similarly; reporting all three is not three independent confirmations.
- **Ignoring multi-label structure.** Job postings can belong to several functions; a hard partition may be the wrong representation entirely.

---

## 6. Recommended Evaluation Protocol

This protocol is designed for the current corpus (about 9.6k cleaned postings, 384-d MiniLM vectors) and remains valid after an encoder change.

### 6.1 Steps

1. **Representation gate.** Confirm what text the encoder actually sees. Report token coverage and truncation rate. Do not compare clusterers until the representation is fixed for the comparison; otherwise the encoder defect dominates the result ([Report 024](024-embedding-models-for-semantic-clustering.md)).
2. **Fix the sample and labels.** Use a seeded sample with stable IDs. Hold out or freeze the label set used for final evaluation. Record population size, exclusions, and label version.
3. **Run a small clusterer grid.** Start with three: spherical K-Means (baseline, fixed `k` range), agglomerative (hierarchy on a sample), and HDBSCAN or EVōC (discovery). Add Leiden only if a k-NN graph is already the system of record.
4. **Compute the four lenses.** Internal, external, stability, and task/qualitative. Report the cluster-size distribution and noise fraction for every run.
5. **Audit representatives.** For each cluster, sample 5–10 postings (including boundary and noise points) and check whether they share the intended concept. Record disagreements; they are the most useful output of the evaluation.
6. **Report trade-offs, not a single winner.** A method with stronger ARI but 40% noise is not automatically better than a method with lower ARI and interpretable coverage. State the product preference explicitly.

### 6.2 Minimum reporting table

| Field | Example |
|---|---|
| Encoder / revision / normalization | MiniLM-L6-v2, pinned revision, L2 |
| Input variant | `cleaned_text`, `variant=llm`, `version=v1` |
| Coverage | mean tokens, truncation rate |
| Sample | `n`, seed, inclusion rule |
| Reduction | none / PCA `n_dims`, explained variance |
| Clusterer + parameters | HDBSCAN `min_cluster_size`, `min_samples`, metric |
| Output summary | clusters, noise %, size min/median/max |
| External labels | function tags / golden roles / ESCO, version |
| External scores | ARI, AMI, homogeneity, completeness, V-measure, bootstrap CI |
| Internal scores | Silhouette, CH, DB, DBCV/DISCO, k-NN purity, persistence |
| Stability | mean subsample ARI, per-cluster Jaccard distribution |
| Audit | sample size, agreement rate, notable disagreements |

### 6.3 Acceptance criteria

Use criteria that are defensible without inventing universal thresholds:

- External scores must beat a **label-permutation null** and a **random-partition baseline** on the same labels.
- The preferred method must remain preferred across a reasonable seed/parameter range, or the report must state that the ranking is unstable.
- Noise fraction must be reported and inspected; a high value is acceptable only if the noise set is semantically plausible (duplicates, unique roles, malformed text).
- Representative-document audit must agree with the intended concept at a rate stated in advance; disagreements must be listed.
- No decision may rest on a single internal metric.

---

## 7. Scale and Engineering Notes

### 7.1 Cost intuition

| Method | Dominant cost | Practical range on a single CPU node |
|---|---|---|
| K-Means / spherical K-Means | `O(n k d)` per iteration | Millions of vectors with MiniBatch |
| Agglomerative | `O(n²)` distances, `O(n²)` memory | Thousands to low tens of thousands; sample above that |
| HDBSCAN | k-NN graph + MST; roughly `O(n log n)` with an index | Tens of thousands comfortably; hundreds of thousands with care |
| EVōC | k-NN graph + node embedding + MST | Similar to HDBSCAN, designed for embeddings |
| GMM | `O(n k d²)` full covariance, less for diagonal | Thousands to low tens of thousands at 384-d |
| Spectral | Dense eigendecomposition `O(n³)` | Small samples only |
| Leiden / Louvain | Near-linear per iteration on a sparse graph | Millions of nodes with a sparse k-NN graph |

### 7.2 Reuse one k-NN graph

The highest-leverage optimization is to compute **one** k-NN graph and let the clusterer and the metric share it. EVōC is built this way; HDBSCAN and Leiden can consume the same neighbor structure; k-NN purity is a direct read. This avoids repeated `O(n²)` distance computations and makes the graph the unit of reproducibility. The graph's `k`, metric, and construction seed must be recorded with the cluster run.

### 7.3 Sample metrics, not clusters, when necessary

Silhouette, CH, DB, DBCV, and DISCO can be estimated on a stratified subsample of 10–50k points. The estimate is adequate for **ranking variants**, which is the only valid use of these scores at this scale. Do not sample for the final external metrics if the labels are cheap to compute; aggregate them exactly.

### 7.4 Output schema

Cluster labels are only useful if they can be joined, audited, and re-evaluated. The run should emit one row per posting with provenance:

```text
posting_id
source_hash
cluster_run_id
encoder_id / encoder_revision
input_variant / input_version
reduction (method, n_dims)
algorithm / algorithm_version
params_hash
distance_metric
seed
cluster_id            -- -1 for noise
membership_strength   -- null for hard methods
is_noise
cluster_label         -- optional, human/LLM-derived
run_at
```

With this schema, external metrics later become a join against a label table and a partitionable aggregation (Athena/Spark), rather than a re-run of the clusterer. [Report 021](021-clustering-metrics-at-scale.md) covers the partitionable external-metric path and the AWS placement in detail.

### 7.5 Dependency posture

`scikit-learn` already provides K-Means, MiniBatch K-Means, agglomerative clustering, GMM, spectral clustering, and (in recent versions) HDBSCAN. `umap-learn` is already a dependency. EVōC is **not** in `pyproject.toml`; adding it is a dependency decision for the user, and the project's rules forbid installing packages autonomously. For a first experiment, a scikit-learn-only grid is sufficient; treat EVōC as an optional challenger once the simpler baseline is established.

---

## 8. Fit Against This Project

### 8.1 Recommended mapping

| Project need | Method | Quality stack |
|---|---|---|
| Quick baseline across encoders / cleaning variants | Spherical K-Means on normalized vectors, fixed `k` range | ARI/AMI vs function tags; Silhouette only within encoder; seed stability |
| Discover job families without fixing `k` | HDBSCAN (`sklearn.cluster.HDBSCAN`) or EVōC | k-NN purity, DBCV/DISCO, persistence, ARI/AMI (noise kept), subsample stability |
| Build a multi-level taxonomy | Agglomerative (average or Ward) on a sample; or EVōC layers | Cophenetic correlation, cut stability, ARI/AMI per level, human audit |
| Graph-native scale path | Leiden on a shared k-NN graph | Modularity/conductance, ARI/AMI, resolution sweep |
| Soft assignment / confidence | vMF mixture (or diagonal GMM) | BIC within encoder, ARI/AMI on hard labels, audit of low-confidence points |
| Explain clusters to users | c-TF-IDF or LLM labels on top of any partition | Representative-document audit; label agreement; do not use labels as ground truth |

### 8.2 Concrete recommendation

1. **Fix encoder coverage first.** The current 256-token MiniLM encoding truncates all 39 golden-set postings; a clustering comparison now measures the truncation prefix. Resolve this with the chunk-pooling arm already proposed in [Report 024](024-embedding-models-for-semantic-clustering.md) before selecting a clusterer.
2. **Establish a spherical K-Means baseline** on normalized vectors over a domain-bounded `k` range. It is cheap, deterministic with a seed, and gives a fixed reference for every later method.
3. **Run HDBSCAN (or EVōC) as the discovery candidate.** Use `min_cluster_size` derived from the smallest actionable job family, keep `-1`, and report the noise fraction. If EVōC is available, use its persistence scores and multiple layers; if not, start with scikit-learn's HDBSCAN.
4. **Evaluate against function tags with ARI, AMI, homogeneity, completeness, and V-measure**, reporting bootstrap intervals. Keep golden role labels as a secondary, noisier check.
5. **Add k-NN purity and subsample stability** as label-free evidence. Prefer the method that stays coherent across seeds and parameters, not the one with the single best score.
6. **Audit representative postings per cluster** before calling the result a job-family taxonomy. The product question is whether the groups are useful, not whether a metric is high.
7. **Persist the run schema in [§7.4](#74-output-schema)** so the evaluation can be repeated after the encoder or cleaning pipeline changes.

### 8.3 Candidate Experiment 012

A follow-up experiment can reuse the fixed 1,000-posting sample from Experiment 011 and compare three clusterers on identical vectors:

- **Arms:** spherical K-Means (six values of `k`), agglomerative (average linkage, dendrogram cut at 2–4 levels), HDBSCAN (`min_cluster_size` in a small grid, `min_samples` in a small grid).
- **Constants:** same encoder, normalization, input variant, PCA setting, sample IDs, and distance metric.
- **Metrics:** ARI/AMI/V-measure against function tags; k-NN purity; DBCV or DISCO for density arms; Silhouette/CH/DB for centroid arms (comparative only); subsample stability; noise fraction; representative-document audit.
- **Decision rule:** reject any arm whose ranking does not survive seed/parameter perturbation; choose the simplest method that passes the external and audit criteria.

This experiment should not start until the encoder-coverage decision from Experiment 011 is settled, because a clusterer comparison on truncated vectors answers the wrong question.

---

## 9. Risks and Limitations

- **Small, noisy labels.** Dozens of hand-labeled postings cannot tightly rank methods; report uncertainty and prefer stability plus audit over small ARI differences.
- **Multi-label reality.** Job postings often span functions. A hard partition may be a lossy representation; consider overlapping clusters or per-tag evaluation.
- **Parameter sensitivity.** Density and graph methods can produce very different taxonomies at nearby parameter values. A single run is not evidence of structure.
- **Encoder dependence.** Every threshold, distance distribution, and internal score is model-specific. An encoder change invalidates the clusterer configuration and the metric baselines ([Report 024](024-embedding-models-for-semantic-clustering.md)).
- **Metric non-comparability.** Internal metrics are not comparable across encoders, and DBCV is not reproducible run-to-run. Use external/task metrics for cross-encoder decisions and DISCO for density validation when available.
- **Visualization bias.** A pleasing 2-D map can reflect UMAP parameters more than semantic structure. Never select a clusterer by its map alone.
- **Temporal drift.** Job postings and role vocabulary change; a clustering tuned on one scrape is a snapshot, not a permanent taxonomy.
- **No universal benchmark.** Published text-clustering comparisons disagree on the best algorithm and are sensitive to representation and distance metric; treat them as priors, not verdicts [16][17].

---

## 10. Sources

1. **MacQueen (1967)** — "Some methods for classification and analysis of multivariate observations." *Berkeley Symposium on Mathematical Statistics and Probability* (K-Means).
2. **Lloyd (1982)** — "Least squares quantization in PCM." *IEEE Transactions on Information Theory* (K-Means).
3. **Dhillon & Modha (2001)** — "Concept decompositions for large sparse text data using clustering." *Machine Learning* (spherical K-Means).
4. **Sculley (2010)** — "Web-scale k-means clustering." *WWW 2010* (MiniBatch K-Means).
5. **Ward (1963)** — "Hierarchical grouping to optimize an objective function." *JASA* (Ward linkage).
6. **Müllner (2011)** — "Modern hierarchical, agglomerative clustering algorithms." arXiv:1109.2378.
7. **Ester, Kriegel, Sander, Xu (1996)** — "A density-based algorithm for discovering clusters in large spatial databases with noise." *KDD 1996* (DBSCAN).
8. **Ankerst, Breunig, Kriegel, Sander (1999)** — "OPTICS: Ordering points to identify the clustering structure." *SIGMOD 1999*.
9. **Campello, Moulavi, Sander (2013)** — "Density-based clustering based on hierarchical density estimates." *PAKDD 2013* (HDBSCAN).
10. **McInnes, Healy, Astels (2017)** — "hdbscan: Hierarchical density based clustering." *Journal of Open Source Software*.
11. **Moulavi, Jaskowiak, Campello, Zimek, Sander (2014)** — "Density-based clustering validation." *SDM 2014* (DBCV).
12. **Bot, McInnes, Aerts (2026)** — "DISCO: Internal evaluation of density-based clusterings with noise." *ICLR 2026*, arXiv:2503.00127.
13. **Banerjee, Dhillon, Ghosh, Sra (2005)** — "Clustering on the unit hypersphere using von Mises-Fisher distributions." *JMLR*.
14. **von Luxburg (2007)** — "A tutorial on spectral clustering." *Statistics and Computing*.
15. **Traag, Waltman, van Eck (2019)** — "From Louvain to Leiden: guaranteeing well-connected communities." *Scientific Reports*.
16. **Mehta, Bawa, Singh (2024)** — "A comprehensive and analytical review of text clustering techniques." *International Journal of Data Science and Analytics*.
17. **"Experimental Study of Clustering Approaches for Text Embeddings" (2025)** — comparison of partition, hierarchical, and density methods on sentence embeddings; finds no universally superior combination and highlights distance-metric sensitivity. *IEEE Smart Industry Conference 2025*.
18. **Rousseeuw (1987)** — "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." *Journal of Computational and Applied Mathematics*.
19. **Caliński & Harabasz (1974)** — "A dendrite method for cluster analysis." *Communications in Statistics*.
20. **Davies & Bouldin (1979)** — "A cluster separation measure." *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
21. **Hubert & Arabie (1985)** — "Comparing partitions." *Journal of Classification* (adjusted Rand index).
22. **Vinh, Epps, Bailey (2010)** — "Information theoretic measures for clusterings comparison." *JMLR* (AMI/NMI).
23. **Rosenberg & Hirschberg (2007)** — "V-measure: A conditional entropy-based external cluster evaluation measure." *EMNLP-CoNLL*.
24. **Fowlkes & Mallows (1983)** — "A method for comparing two hierarchical clusterings." *JASA* (FMI).
25. **Sokal & Rohlf (1962)** — "The comparison of dendrograms by objective methods." *Taxon* (cophenetic correlation).
26. **McInnes, Healy, Melville (2018)** — "UMAP: Uniform Manifold Approximation and Projection for dimension reduction." arXiv:1802.03426.
27. **Grootendorst (2022)** — "BERTopic: Neural topic modeling with a class-based TF-IDF procedure." arXiv:2203.05794.
28. **Angelina, Lertvittayakumjorn, Nuttapong (2020)** — "Top2Vec: Distributed representations of topics." arXiv:2008.09470.
29. **Tutte Institute (2025–2026)** — `evoc` v0.3.1 source and documentation; PLSCAN (Bot, McInnes, Aerts, arXiv:2512.16558) as cited in [Report 012](012-evoc-embedding-clustering.md).
30. **arXiv:2509.08625 (2026)** — "An upper bound on the silhouette evaluation metric for clustering" — dimensionality collapse of average silhouette width.
31. **scikit-learn documentation** — `KMeans`, `MiniBatchKMeans`, `AgglomerativeClustering`, `HDBSCAN`, `GaussianMixture`, `SpectralClustering`, and `sklearn.metrics` clustering scores.

**Related project reports:** [001 Embedding Optimization](001-embedding-optimization-research.md) (normalization and cosine protocol), [009 PCA Informativeness](../experiments/009-pca-informativeness.md) (the `n_dims=256` reduction), [012 EVōC](012-evoc-embedding-clustering.md) (graph reuse, layers, persistence), [016 ESCO](016-esco-eu-jobs-taxonomy.md) (future external label target), [021 Clustering Metrics at Scale](021-clustering-metrics-at-scale.md) (partitionable metrics and AWS placement), [024 Embedding Models](024-embedding-models-for-semantic-clustering.md) (encoder selection and the truncation defect).
