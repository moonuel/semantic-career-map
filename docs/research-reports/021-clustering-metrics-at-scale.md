# Clustering Metrics for Vector Embeddings at Scale — Research Report

> Date: 2026-10-28
> Context: Semantic Career Mapping Platform — survey of clustering-quality metrics for high-dimensional embedding vectors, and where to compute them when the corpus outgrows a single process (targeting AWS). Extends the evaluation framework in [technical/evaluation.md](../technical/evaluation.md) (separation gap, self-retrieval) and the deferred "clustering / job-family discovery" feature reviewed in [Report 012](012-evoc-embedding-clustering.md). Complements the retrieval-side graduation ladder in [Report 014](014-semantic-search-tools-architecture.md) and the AWS lakehouse posture in [Report 019](019-s3-vectors.md).

---

## Table of Contents

1. [Summary](#1-summary)
2. [Why Clustering Metrics Are Hard for Embeddings](#2-why-clustering-metrics-are-hard-for-embeddings)
3. [Internal (Unsupervised) Metrics](#3-internal-unsupervised-metrics)
4. [External (Supervised) Metrics](#4-external-supervised-metrics)
5. [Stability and Resampling Metrics](#5-stability-and-resampling-metrics)
6. [The Pairwise O(n²) Wall and How to Get Around It](#6-the-pairwise-on2-wall-and-how-to-get-around-it)
7. [AWS: Where Clustering and Metrics Run at Scale](#7-aws-where-clustering-and-metrics-run-at-scale)
8. [Fit Against This Project](#8-fit-against-this-project)
9. [Decision and Recommendations](#9-decision-and-recommendations)
10. [Sources](#10-sources)

---

## 1. Summary

Clustering evaluation splits cleanly into **internal** metrics (no ground truth — judge a clustering on its own geometry) and **external** metrics (compare against held-out labels). The catch for this project is that both families were designed for low-dimensional tabular data and *misbehave on high-dimensional embeddings* in two distinct ways:

1. **Distance concentration.** In high dimensions, pairwise distances compress toward a common value, so centroid/partition indices (Silhouette, Calinski-Harabasz, Davies-Bouldin) drift toward degenerate scores that no longer separate "good" from "bad" clusterings. A 2026 upper-bound analysis of the average silhouette width formalizes this: the bound collapses toward zero as dimensionality rises (arXiv:2509.08625).
2. **The density metric of choice is broken.** DBCV — the standard internal index for density-based clusterings with noise (HDBSCAN/EVōC output) — is *non-deterministic*: it drops points at the leaf level of the cluster-wise MST, so the same clustering can receive different scores. The 2026 replacement is **DISCO** (ICLR 2026, arXiv:2503.00127), a deterministic density-based index that also scores noise assignment explicitly.

The practical resolution is three-fold: (a) prefer **external metrics** whenever *any* label signal exists — this project already has golden role labels and experiment 006's function tags; (b) for internal checks, use **graph/k-NN-based or embedding-native** measures (k-NN purity, persistence from EVōC's PLSCAN) rather than raw centroid indices; (c) at scale, replace the O(n²) pairwise distance matrix with a **shared k-NN graph** that both the clusterer and the metric read from — the exact optimization [Report 012](012-evoc-embedding-clustering.md) flagged as EVōC's headline win.

On AWS, clustering itself is a solved batch problem (SageMaker built-in K-Means, EMR + Spark MLlib), but *metric computation* has no managed service — it is a Spark/Athena/script-mode job, and the metrics must be chosen to be partitionable (external counts) rather than pairwise, because Athena cannot express the O(n²) geometry. The recommendation for this project is below in [§9](#9-decision-and-recommendations).

---

## 2. Why Clustering Metrics Are Hard for Embeddings

Three structural facts govern every choice in this report.

**Embeddings are not i.i.d. Euclidean points.** They live on a learned manifold, usually *already L2-normalized* (this project's protocol, per [Report 001](001-embedding-optimization-research.md) §3.1 and [design-decisions.md](../architecture/design-decisions.md)), so the operative geometry is *cosine on the unit sphere*, not Euclidean. Indices that assume spherical Gaussian clusters (CH, GMM likelihoods) are measuring the wrong shape.

**Distance concentration.** As dimension *d* grows, the contrast between nearest and farthest neighbor shrinks. The 2026 silhouette upper-bound paper shows the average-silhouette ceiling tends to zero with *d* (arXiv:2509.08625); the standard "0.5 = reasonable, 0.25 = weak" interpretation breaks down in 384-d embedding space. Silhouette is *not* wrong — it just becomes uninformative as an absolute score, and should only be used *comparatively* (variant A vs variant B on the same embedding model).

**Noise is first-class.** The clusterers this project is actually interested in (HDBSCAN per [technical/evaluation.md](../technical/evaluation.md) §"HDBSCAN Cluster Validation", EVōC per [Report 012](012-evoc-embedding-clustering.md)) emit `-1`/noise labels. Most internal metrics have no coherent definition for noise points — DBCV's non-determinism is precisely a symptom of this — so the metric must *explicitly* model noise to judge a density-based clustering fairly.

The consequence: the metric choice is downstream of the **clusterer choice**, not independent of it.

| Clusterer family | Output shape | Metric family that applies |
|---|---|---|
| K-Means / centroid | hard partition, no noise, convex | Silhouette, CH, DB (comparative only) |
| HDBSCAN / EVōC / density | partition + noise, arbitrary shape | DBCV→DISCO, k-NN purity, persistence |
| Hierarchical | dendrogram / layers | cophenetic correlation, stability, layer consistency |

---

## 3. Internal (Unsupervised) Metrics

### 3.1 Centroid / partition-based

| Metric | Definition (intuition) | High-dim caveat |
|---|---|---|
| **Silhouette coefficient** (Rousseeuw 1987) | per point: `(b − a) / max(a, b)`, *a* = mean intra-cluster dist, *b* = mean nearest-other-cluster dist | collapses toward 0 with *d* (arXiv:2509.08625); assumes convex clusters |
| **Calinski-Harabasz** (CH) | `trace(B_k) / trace(W_k) · (n−k)/(k−1)` — ratio of between/within-cluster scatter | assumes spherical Gaussian clusters; monotone-ish in *k*, rewards centroid models |
| **Davies-Bouldin** (DB) | mean over clusters of worst `(σ_i + σ_j)/d(c_i,c_j)` | penalizes centroid scatter; same convexity assumption |

These three are O(n·k·d) once centroids exist — the *cheapest* metrics here — and remain useful for a *relative* read (is variant A more compact than variant B?) but must not be read as absolute quality, especially on normalized 384-d vectors. The project's existing **separation gap** ([technical/evaluation.md](../technical/evaluation.md) §"Separation Gap") is a close cousin: `mean(within-role cos) − mean(cross-role cos)` is effectively a one-dimensional between-vs-within contrast computed on cosine, and it has the same "comparative-only, label-driven" character.

### 3.2 Density-based

- **DBCV** (Moulavi et al. 2014) — the *relative-validity* index built for HDBSCAN-style clusterings: computes a per-cluster *density sparseness/density separation* from a mutual-reachability MST over a k-NN graph, then aggregates into a [−1, 1] score that *does* account for noise. It is the go-to for this project's clusterer of record.
  - **Non-determinism defect.** DISCO (arXiv:2503.00127, ICLR 2026) documents that DBCV *excludes* leaf-level points of each cluster's MST, so re-running DBCV on the *same* clustering yields *different* scores — disqualifying it as a reproducible metric. Practically: fix a `random_state` / reproducible k-NN graph, or accept jitter and report a mean over seeds.
  - **Cost.** DBCV is the *least* scalable internal index: it needs a k-NN graph plus a per-cluster MST, and single-cell omics studies report it as the one metric that does **not** scale to large n (PMC12476229 / gkaf921).
- **DISCO** (Bot et al., ICLR 2026) — deterministic DBCV successor. Redefines silhouette on *density-connectivity* rather than raw distance, and scores *noise* by local object density, so it prefers correct noise assignment instead of ignoring it. This is the current state of the art for judging HDBSCAN/EVōC output at low-to-medium n.

### 3.3 Graph / k-NN-based

Because the clusterer already builds a k-NN graph (EVōC does; HDBSCAN does), graph metrics reuse that structure instead of the raw distance matrix:

- **k-NN purity** — for each point, the fraction of its k nearest neighbors sharing its cluster label. O(n·k), and directly readable per-cluster. Natural companion to the project's **self-retrieval** check.
- **Graph modularity / conductance** — clustering *community* quality on the symmetrized neighbor graph; works where clusters are graph-connected rather than centroid-shaped.
- **Local/neighborhood consistency** — variants of silhouette computed against the k-NN set rather than the full matrix, sidestepping distance concentration.

### 3.4 Embedding-native / persistence

EVōC (Report 012 §10) exposes a metric the partition families cannot: **persistence** of each cluster over the `min_cluster_size` scale. A cluster that survives across many resolutions is "real"; one that appears and vanishes is noise. `persistence_scores_` is effectively a *built-in* internal quality score that needs no ground truth and no second O(n²) pass — it is a by-product of the condensed tree. This is the most scale-honest internal signal available to this project, because it is computed inside the clusterer.

---

## 4. External (Supervised) Metrics

When labels exist (golden role labels, experiment 006 function tags, or a future ESCO mapping from Report 016), external metrics are both *more honest* and *more scalable* than internal ones — they are counts over a confusion matrix, not geometry.

| Metric | Definition | Notes |
|---|---|---|
| **ARI** (Adjusted Rand Index) | agreement of all point pairs, corrected for chance | the default; `-1`..`1`, 0 = chance. Handles `-1` noise as one extra "cluster" |
| **AMI / NMI** (Adjusted/Normalized Mutual Information) | information shared between label and clustering | AMI is chance-corrected; NMI is not and is biased toward fine partitions |
| **Homogeneity / Completeness / V-measure** | do clusters each contain one class / are all of a class in one cluster / harmonic mean | decompose the two failure modes of "too split" vs "too merged" |
| **FMI** (Fowlkes-Mallows) | geometric mean of pairwise precision/recall | redundant with ARI in practice |
| **Purity** | fraction of points in the majority class of their cluster | never use alone — trivially maximized by one point per cluster |

Two project-specific recommendations:

- **Noise handling.** HDBSCAN/EVōC emit `-1`. The correct treatment is to *keep* noise as a separate "label" for ARI/AMI (a point correctly called noise against a label set with no noise class is genuinely ambiguous — decide per experiment and document it), not to silently drop it, which inflates every score.
- **Report the interleaved metric.** This project's Step 1.3 "interleaved" harness (Report 012 §15) already compares clustering labels against hand-labeled `role_category`. ARI/AMI/V-measure are the canonical, `sklearn.metrics`-one-liner realization of that harness, and they do *not* require the O(n²) matrix.

These are also the only metrics in this report that are **trivially partitionable**: the pair counts underlying ARI/AMI/FMI are sums over point pairs, computable as a MapReduce/Spark/SQL aggregation without ever materializing the full pair set.

---

## 5. Stability and Resampling Metrics

A label-free way to score a clustering is to ask whether it *reproduces itself* under perturbation — stability is a proxy for "real structure vs. fitting noise":

- **Bootstrap/Jaccard stability.** Cluster on subsamples (e.g. 80% of points, *m* seeds), compute the mean Jaccard similarity of the induced clusterings. High stability ⇒ robust structure.
- **EVōC's layer diversity** is the persistence-side analogue: `select_diverse_peaks` already enforces Jaccard ≤ `min_similarity_threshold` between candidate layers (Report 012 §10c), i.e. it *measures* clustering similarity to keep layers distinct.

Stability metrics are O(m) cluster runs and thus the *most* expensive per-unit of signal — reserve them for final verification, not the inner experiment loop.

---

## 6. The Pairwise O(n²) Wall and How to Get Around It

The dominant cost across every internal metric is the pairwise distance matrix — n²/2 cosine similarities, n² memory, n²·d FLOPs. The remedies, in increasing order of engineering cost:

1. **Exact in-process (today).** NumPy matmul `E @ E.T` on L2-normalized vectors is the fastest possible cosine-all-pairs. Holds to ~10⁴–10⁵ points in a few GB of RAM ([Report 019](019-s3-vectors.md) §1's exact-retrieval reasoning applies identically to clustering).
2. **Sample the metric.** Silhouette/CH on a stratified subsample of 10–50k points is a near-exact estimate of the full-population score at a fraction of the cost — the standard trick, and sufficient for model comparison.
3. **k-NN graph reuse.** Compute *one* k-NN graph (NN-Descent / HNSW / faiss), then let the clusterer *and* the metric (k-NN purity, DBCV/DISCO, graph modularity) read it. This converts O(n²) into O(n·k·log n). This is EVōC's architecture (Report 012 §13) and the single highest-leverage scaling decision.
4. **Distributed / partitionable.** External metrics (ARI/AMI/FMI) reduce to pair-count aggregates over a Spark/Athena table — O(n) storage of counts, O(n) shuffle, no pair materialization. Internal geometry metrics do *not* decompose cleanly and should be avoided at that tier.

---

## 7. AWS: Where Clustering and Metrics Run at Scale

The project's migration posture is CPU-first, serverless-first, ~$1.50/mo steady state (aws-migration-roadmap.md §2, §4). Clustering at scale fits that posture as **batch jobs**, not services. There is **no managed "compute clustering metrics" service** — clustering is managed; metric evaluation is *your code* on one of these compute surfaces.

| Surface | Clustering | Metrics | Notes |
|---|---|---|---|
| **SageMaker built-in K-Means** | ✅ web-scale mini-batch K-Means (the *one* native unsupervised built-in) | ❌ | tabular input, `recordio`/CSV; 400-dim × 23M rows × k=500 in ~7 min (AWS ML blog). Overkill below ~10⁵ points |
| **EMR + Spark MLlib** | ✅ KMeans (kmeans\|\|), BisectingKMeans, GMM, PowerIterationClustering (graph), LDA | ✅ **external** (ARI/AMI as Spark SQL/aggregate); ⚠️ internal pairwise is awkward | `sagemaker-spark` pre-installed since EMR 5.11 to fan out SageMaker jobs from Spark |
| **Glue + Athena** | ❌ (no UDF surface for iterative clustering) | ✅ external metrics over the Parquet lake *if* labels live there | matches Phase 5 query layer; cannot express O(n²) geometry |
| **ECS Fargate / Lambda script-mode** | ⚠️ single-node clusterers (EVōC, HDBSCAN) | ✅ exact NumPy / sampled internal metrics | the natural first rung: same repo image, same IAM role as the scraper tiers |

Key guidance for the AWS case:

- **External metrics over the lake are the long-term win.** If cluster labels are written back to Parquet (`posting_id, source_hash, cluster_id, label`) and a golden/label table exists, then ARI/AMI/V-measure/FMI become Athena SQL counts — serverless, sub-cent at this corpus size, and consistent with the [Athena lakehouse proposal](../architecture/proposed-athena-layout.md) and Report 010's dimensional modeling.
- **Internal geometry metrics should stay single-node.** Silhouette/DBCV/DISCO on a sampled subset in a Fargate task (or even a Lambda under memory limits) is cheaper and simpler than forcing them through Spark. k-NN graph construction (faiss/HNSW) on CPU fits Fargate/Graviton well.
- **SageMaker K-Means only earns its keep at 10⁵+ vectors**, and only for *centroid* clustering — it cannot express the density-based (HDBSCAN/EVōC) clustering this project's job-family feature actually wants. Treat it as a scale fallback, not a default.
- **Cost guardrail.** Everything above is batch and idle-free; the risk is not cost but *scan* (Athena $5/TB) and *memory* (Fargate task size). Sample before you scale.

---

## 8. Fit Against This Project

| Project signal | Map to metric |
|---|---|
| **Separation gap** ([evaluation.md](../technical/evaluation.md)) | internal, centroid-family, comparative-only — keep as the fast proxy, but stop reading it as absolute |
| **Self-retrieval** ([evaluation.md](../technical/evaluation.md)) | degeneracy check — keep; k-NN purity is its natural clustering-level generalization |
| **HDBSCAN planned validation** ([evaluation.md](../technical/evaluation.md) §"HDBSCAN Cluster Validation") | pair with **DBCV/DISCO** (internal) + **ARI/AMI** against function tags (external) |
| **EVōC multi-layer output** ([Report 012](012-evoc-embedding-clustering.md)) | use **persistence_scores_** + layer diversity as built-in quality; ARI/AMI per layer against `role_category` |
| **Function tags** (experiment 006) | external ground truth for ARI/AMI/V-measure — the most honest clustering target available today |
| **Golden role labels** (experiment 005/008) | external; noisy, so prefer function tags when both exist |

The single most important gap to close is **write cluster labels back to Parquet** — without a `cluster_id` column, none of the scalable external-metric paths (Athena/Spark) can run, and the project is stuck in the single-process O(n²) regime.

---

## 9. Decision and Recommendations

1. **Adopt the external metric set now** (zero new dependencies): `sklearn.metrics.adjusted_rand_score`, `adjusted_mutual_info_score`, `homogeneity_completeness_v_measure` — computed against experiment 006 function tags. These are chance-corrected, label-honest, and partitionable.
2. **Keep separation gap and self-retrieval** as the cheap inner-loop proxies, but *re-label* them "comparative" in the docs; do not interpret absolute values on 384-d vectors.
3. **For the HDBSCAN/EVōC clusterer, report DBCV (with a fixed reproducible k-NN graph) and prefer DISCO** once it is packaged; rely on EVōC's `persistence_scores_` when available, since it costs nothing extra.
4. **Do not scale internal metrics to Spark.** Sample to ≤50k points and run single-node; reserve Spark/Athena for external pair-count metrics.
5. **Schema the output for the lake now:** ensure clustering emits `posting_id, source_hash, cluster_id, membership_strength, label` so ARI/AMI can later be an Athena query (Phase 5) rather than a re-run.
6. **Defer SageMaker K-Means and EMR** until the corpus reaches ~10⁵ vectors *and* a centroid-style family discovery is actually wanted; at current scale the EVōC/HDBSCAN single-node path (Report 012) is correct and the cost guardrails in aws-migration-roadmap.md §2 govern.

---

## 10. Sources

1. **Rousseeuw (1987)** — "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." *J. Comput. Appl. Math.* 20.
2. **Moulavi, Jaskowiak, Campello, Zimek, Sander (2014)** — "Density-Based Clustering Validation" (DBCV). *SDM 2014*.
3. **Bot, McInnes, Aerts (2026)** — "DISCO: Internal Evaluation of Density-Based Clusterings with Noise." *ICLR 2026*, arXiv:2503.00127.
4. **arXiv:2509.08625 (2026)** — "An upper bound on the silhouette evaluation metric for clustering" — dimensionality collapse of average silhouette width.
5. **Chowdhury & de Amorim (2019)** — "An efficient sanitization algorithm for balancing information privacy and knowledge discovery" (DBCV survey); and **PubMed 40989482 (2025)** — "The DBCV index is more informative than DCSI, CDbw, and VIASCKDE indices for unsupervised clustering internal assessment of concave-shaped and density-based clusters."
6. **PMC12476229 / NAR gkaf921 (2025)** — "On metrics for subpopulation detection in single-cell and spatial omics data" — embedding-based metric catalog (Silhouette, CDbw, DBCV) and DBCV's scalability ceiling.
7. **Hubert & Arabie (1985)** — "Comparing partitions" (adjusted Rand index).
8. **Vinh, Epps, Bailey (2010)** — "Information theoretic measures for clusterings comparison" (AMI/NMI).
9. **Rosenberg & Hirschberg (2007)** — "V-measure: A conditional entropy-based external cluster evaluation measure."
10. **Amazon SageMaker AI docs** — "K-Means Algorithm" (web-scale mini-batch k-means); AWS ML Blog (2018) — "K-means clustering with Amazon SageMaker" (400-dim × 23M × k=500 in ~7 min).
11. **Apache Spark MLlib docs (4.2.0)** — "Clustering" (KMeans/kmeans\|\|, BisectingKMeans, GaussianMixture, PowerIterationClustering, LDA).
12. **aws/sagemaker-spark (GitHub)** — Spark library for SageMaker, pre-installed on EMR ≥ 5.11.

**Related project reports:** [001 Embedding Optimization](001-embedding-optimization-research.md) (L2-normalization / cosine protocol), [010 Data Warehouse](010-data-warehouse-design-best-practices.md) & [proposed Athena layout](../architecture/proposed-athena-layout.md) (partitionable external metrics), [012 EVōC](012-evoc-embedding-clustering.md) (k-NN graph reuse, persistence, multi-layer output), [014 Semantic Search Tools](014-semantic-search-tools-architecture.md) (graduation ladder), [016 ESCO](016-esco-eu-jobs-taxonomy.md) (external label target), [019 S3 Vectors](019-s3-vectors.md) (exact-vs-ANN and serverless posture).
