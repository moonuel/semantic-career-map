# EVōC — Embedding Vector Oriented Clustering: Implementation Review

> Date: 2026-10-04
> Context: Technical research report on [`TutteInstitute/evoc`](https://github.com/TutteInstitute/evoc) (v0.3.1, BSD-2-Clause), the embedding-specialized clustering library flagged as future work in [002 — Tutte Institute Tools](002-tutte-institute-tool-review.md). Focus: how EVōC achieves clustering of embedding vector clusters, traced to source.

---

## Table of Contents

1. [Summary](#1-summary)
2. [What EVōC Is](#2-what-evoc-is)
3. [Repository Layout](#3-repository-layout)
4. [The End-to-End Pipeline](#4-the-end-to-end-pipeline)
5. [Stage 1 — Distance-Metric Dispatch & Quantized Inputs](#5-stage-1-distance-metric-dispatch-quantized-inputs)
6. [Stage 2 — k-NN Graph Construction (NN-Descent)](#6-stage-2-k-nn-graph-construction-nn-descent)
7. [Stage 3 — Node Embedding (UMAP-like, embedding-native)](#7-stage-3-node-embedding-umap-like-embedding-native)
8. [Stage 4 — Minimum Spanning Tree via Parallel Borůvka](#8-stage-4-minimum-spanning-tree-via-parallel-boruvka)
9. [Stage 5 — Condensed Tree (HDBSCAN-style)](#9-stage-5-condensed-tree-hdbscan-style)
10. [Stage 6 — PLSCAN: Persistence Peaks & Multi-Layer Extraction](#10-stage-6-plscan-persistence-peaks-multi-layer-extraction)
11. [Stage 7 — Duplicate Detection](#11-stage-7-duplicate-detection)
12. [How EVōC Differs from UMAP + HDBSCAN](#12-how-evoc-differs-from-umap-hdbscan)
13. [Performance Engineering](#13-performance-engineering)
14. [API Surface & Key Parameters](#14-api-surface-key-parameters)
15. [Relevance to Semantic Career Mapping](#15-relevance-to-semantic-career-mapping)
16. [Limitations & Risks](#16-limitations-risks)
17. [References](#17-references)

---

## 1. Summary

EVōC ("evoke") is a CPU-first clustering library specialized for high-dimensional
embedding vectors (CLIP, sentence-transformers, OpenAI/Cohere embeddings). Its
pitch is *"all the good parts of UMAP + HDBSCAN, with the time-consuming parts
removed and fewer hyper-parameters."*

It achieves this by replacing the two-stage UMAP→HDBSCAN workflow with a single
integrated pipeline:

| Conventional workflow | EVōC equivalent | Key difference |
|---|---|---|
| `umap.UMAP` → `hdbscan.HDBSCAN` | `evoc.EVoC().fit_predict(X)` | One estimator, one fit |
| UMAP reduces to 2–5D for visualization-grade embedding | Internal node embedding to `min(max(k/4,4),15)` dims | Keeps more dimensions; not visualization-oriented |
| HDBSCAN builds its own k-NN + MST | Reuses the *same* graph the embedding was built from | No duplicated neighbor search |
| Manual `min_cluster_size` tuning per dataset | Persistence-peak selection (PLSCAN) picks layers automatically | Automatic cluster-count / granularity selection |

The core technical claim is that because the input is *already* a meaningful
metric space (normalized embeddings with cosine geometry), EVōC can skip UMAP's
expensive global manifold optimization, learn a cheap low-dimensional
*node embedding of the k-NN graph* instead, and then run a fast, parallel,
embedding-specific density clustering on top.

The algorithm itself is unpublished; the cluster-extraction stage is based on the
**PLSCAN** paper (Bot, McInnes, Aerts, arXiv:2512.16558).

---

## 2. What EVōC Is

- **Author:** Leland McInnes (UMAP, HDBSCAN, DataMapPlot author) at the Tutte
  Institute for Mathematics and Computing (TIMC).
- **Version reviewed:** 0.3.1 (Beta), 162 commits.
- **License:** BSD-2-Clause (permissive; compatible with this project).
- **Dependencies (deliberately small):** `numpy`, `scikit-learn`, `numba`,
  `tqdm`. Note `tbb` is advertised as an optional runtime accelerator for Numba
  threading but is not in `pyproject.toml`.
- **Python:** >= 3.10.
- **API contract:** scikit-learn `BaseEstimator` + `ClusterMixin`, so
  `fit`, `fit_predict`, `labels_`, `-1` = noise — a drop-in for any sklearn
  clustering slot.
- **Everything is Numba-JIT `@numba.njit` with `parallel=True` and `cache=True`:**
  no PyTorch/TF, no GPU, CPU-only by design.

The unique outputs beyond `labels_` are:

- `cluster_layers_` — an ordered list of label vectors, finest grain first.
- `membership_strength_layers_` — per-point confidence, like HDBSCAN
  `probabilities_`.
- `cluster_tree_` — a dict mapping `(layer, cluster)` → child `(layer, cluster)`
  tuples, reconstructing a hierarchy *across* layers.
- `duplicates_` — pairs of near-duplicate points found for free from the k-NN graph.

---

## 3. Repository Layout

```
evoc/
├── evoc/
│   ├── __init__.py                # re-exports EVoC, evoc_clusters
│   ├── clustering.py              # EVoC estimator + evoc_clusters + build_cluster_layers
│   ├── knn_graph.py               # dtype dispatch, RP-forest + NN-Descent driver
│   ├── float_nndescent.py         # NN-Descent for float32 (cosine)
│   ├── int8_nndescent.py          # NN-Descent for int8 (quantized cosine)
│   ├── uint8_nndescent.py         # NN-Descent for uint8 (Hamming/Jaccard)
│   ├── common_nndescent.py        # shared heaps, RNG, candidate build
│   ├── graph_construction.py      # fuzzy k-NN graph + smooth_knn_dist (UMAP-style)
│   ├── node_embedding.py          # UMAP-like SGD node embedding
│   ├── label_propagation.py       # graph partition init (recursive)
│   ├── numba_kdtree.py            # Numba KD-tree build/query
│   ├── boruvka.py                 # parallel Borůvka MST
│   ├── cluster_trees.py           # linkage tree, condense_tree, label vectors
│   ├── clustering_utilities.py    # peaks, persistence, diversity, duplicates
│   └── disjoint_set.py            # union-find with rank
├── doc/source/                    # readthedocs: quickstart, user_guide, api
├── scripts/run_performance_tests.py
└── pyproject.toml
```

Everything that matters for clustering is in the nine modules listed above; the
three `*_nndescent.py` files are dtype-specialized copies of one algorithm.

---

## 4. The End-to-End Pipeline

The public entry point `evoc_clusters()` (`evoc/clustering.py:191`) executes this
pipeline. Every stage is where the "embedding-native" specialization lives.

```
 input X (n × d embeddings)
   │
   ▼
 [1] dtype dispatch ── float32/cosine │ int8/quantized-cosine │ uint8/Jaccard
   │
   ▼
 [2] knn_graph()  ── RP-forest seed ──► NN-Descent refinement ──► (n × k) indices + transformed distances
   │
   ▼
 [3] neighbor_graph_matrix()  ── smooth_knn_dist → fuzzy weighted sparse graph
   │
   ▼
 [3b] label_propagation_init()  ── recursive graph partition → initial coordinates
   │
   ▼
 [4] node_embedding()  ── UMAP-like SGD on the k-NN graph → n × d' coordinates (d' ≈ 4–15)
   │
   ▼
 [5] build_kdtree() + parallel_boruvka()  ── mutual-reachability MST (density-aware)
   │
   ▼
 [6] mst_to_linkage_tree() → condense_tree()  ── HDBSCAN-style hierarchy
   │
   ▼
 [7] min_cluster_size_barcode + find_peaks + select_diverse_peaks  ── PLSCAN persistence peaks
   │
   ▼
 [8] per-peak extract_clusters_by_id()  ── label vector + membership strength per layer
   │
   ▼
 cluster_layers_, membership_strengths_, persistence_scores_, nn_inds_, nn_dists_, duplicates_
```

Two structural details worth noting up front:

- **The MST (step 5) is built on the *node embedding*, not on the original
  high-dimensional vectors.** Density clustering happens in the low-dimensional
  learned space. This is the crucial divergence from stock HDBSCAN, which would
  cluster in the original metric space.
- **No stage is skipped for small data.** Even a 27-point dataset runs the whole
  pipeline (with `base_init` falling back to PCA/spectral below 64 points).

---

## 5. Stage 1 — Distance-Metric Dispatch & Quantized Inputs

`knn_graph()` (`evoc/knn_graph.py:188`) inspects `data.dtype` and selects a
distance semantics — this is the "embedding vector oriented" specialization:

| dtype | Geometry | Distance transform applied to raw NN-Descent output |
|---|---|---|
| `uint8` | **Binary embeddings** (bit-packed, `np.packbits`) | Bitwise Hamming → Jaccard; stored as `-log2(-d)` |
| `int8` | **Quantized embeddings** (scalar-quantized to 127) | Quantized cosine; stored as `1 / -d` |
| float | **Standard embeddings** | L2-normalize → cosine; stored as `max(-log2(-d), 0)` |

The dtype check happens before any graph work (`evoc/knn_graph.py:270-290`).
For float inputs the code L2-normalizes in place (single division by the row
norm) only if not already unit-norm, so cosine distance becomes squared
Euclidean. The three NN-Descent variants consume the raw dtype and emit a
*uniformly-scaled distance* via the `-log2(-d)` / `1/-d` transform in
`nn_descent()` (`evoc/knn_graph.py:131-183`). This makes the downstream
`make_epochs_per_sample` weighting behave consistently across formats.

This is exactly the claim from the README — "clustering of int8 or binary
quantized embedding vectors that works out-of-the-box."

---

## 6. Stage 2 — k-NN Graph Construction (NN-Descent)

EVōC does **not** use sklearn's `NearestNeighbors` or a GPU index. It ships its
own NN-Descent implementation, one file per dtype, all sharing `common_nndescent.py`.

The algorithm (`knn_graph` at `evoc/knn_graph.py:188`):

1. **Random projection forest seeding** (`make_forest`, `evoc/knn_graph.py:29`).
   Builds `n_trees = max(4, min(8, n_threads))` random-projection trees of
   `leaf_size = max(10, k)`. Each leaf yields a candidate-list block; trees are
   padded to a common width and stacked. This gives a cheap high-recall set of
   candidate neighbors.
2. **NN-Descent refinement** (`nn_descent`, `evoc/knn_graph.py:93`). Iteratively
   propagates each node's neighbor list to its neighbors' neighbors
   ("neighbor of my neighbor is a candidate"), converging to an approximate
   exact graph. Iterations default to `max(5, round(log2(n_samples)))`; the
   `delta`/`delta_improv` thresholds (default `0.001`) control early stopping.
   Two update strategies exist: heap-based and the default `sorted_updates`
   (merge-sort of the candidate heap), which is faster.
3. **Distance transform** per dtype as in [Stage 1](#5-stage-1-distance-metric-dispatch-quantized-inputs).

The returned `(n × k)` index/distance arrays are retained and returned to the
user (`nn_inds_`, `nn_dists_`) — they are *not* discarded, because they power
both the fuzzy graph and duplicate detection.

---

## 7. Stage 3 — Node Embedding (UMAP-like, embedding-native)

This is where EVōC's "improve on UMAP" claim is implemented, and it has three
sub-stages.

### 3a. Fuzzy graph construction — `neighbor_graph_matrix`

`evoc/graph_construction.py:124` converts `(indices, distances)` into a weighted
sparse graph using the same local-connectivity trick as UMAP:

- `smooth_knn_dist` (`evoc/graph_construction.py:24`) solves for a per-point
  bandwidth `sigma` such that the total membership of a point's neighbors equals
  a target `log2(k)`, with `rho` = distance to nearest non-zero neighbor
  (local-connectivity correction). This is a binary search capped at 64
  iterations.
- `compute_membership_strengths` (`evoc/graph_construction.py:90`) produces
  Gaussian-kernel edge weights `exp(-(d - rho)/sigma)`.
- Optionally symmetrized (default) via the fuzzy union
  `A + Aᵀ - A·Aᵀ` (`evoc/graph_construction.py:180`).

Note the effective `k` passed here is `neighbor_scale * n_neighbors`, letting
users trade local vs. global structure.

### 3b. Recursive label-propagation initialization — `label_propagation_init`

Rather than random init (UMAP's approach), EVōC computes a strong initial layout
by recursively partitioning the graph (`evoc/label_propagation.py:189`):

- For graphs `< 64` nodes: direct `PCA` / `SpectralEmbedding` / `MDS` / random.
- Otherwise: run weighted label propagation for `n_label_prop_iter` iterations to
  assign each point to ~`approx_n_parts` partitions (`np.clip(8*sqrt(n), 256, 16384)`),
  build a partition-reduction matrix, coarsen the graph and the data, and recurse.
- The recursion terminates against the base cases; results are up-scaled back to
  per-node coordinates (`upscaling='partition_expander'` by default).

This is the expensive part UMAP cannot skip for quality; EVōC makes it cheap by
operating on the *coarsened* graph.

### 3c. SGD optimization — `node_embedding`

`evoc/node_embedding.py:231` is a stripped, specialized UMAP-like optimizer:

- Objective is the standard attractive (edge) + repulsive (negative-sample)
  loss, but the attractive gradient uses an EVōC-specific coefficient
  `(-2·noise_level·d - 2) / (2d² - 0.5d + 1)` (`evoc/node_embedding.py:93`).
  `noise_level` (default 0.5) directly tunes how aggressively nearby-but-not-
  identical points repel — this is EVōC's single most important quality knob.
- **Reproducible path** (`node_embedding_epoch_repr`, `evoc/node_embedding.py:150`):
  instead of asynchronous per-edge updates, it processes nodes in blocks,
  accumulates gradients in an `updates` buffer, then applies them — making
  results deterministic at some speed cost. `reproducible_flag` is set
  automatically when `random_state` is provided (`clustering.py:630`).
- Output dimensionality defaults to `min(max(n_neighbors // 4, 4), 15)`
  (`clustering.py:338`) — *much* higher than UMAP's usual 2D. This is
  intentional: the embedding exists to be *clustered*, not plotted.

---

## 8. Stage 4 — Minimum Spanning Tree via Parallel Borůvka

EVōC computes the MST with its own parallel Borůvka (`parallel_boruvka`,
`evoc/boruvka.py:535`) over a Numba KD-tree (`numba_kdtree.py`) built on the
node embedding.

Key properties:

- **Mutual reachability distance.** With `min_samples > 1`, it queries the
  `min_samples+1` nearest neighbors to get per-point *core distances*, and edge
  weights become `max(core_dist[i], core_dist[j], d(i,j))`
  (`initialize_boruvka_from_knn`, `evoc/boruvka.py:485`). This is the same
  density correction HDBSCAN uses; it makes the MST robust to sparse regions.
- **Component-aware tree traversal.** `update_component_vectors`
  (`evoc/boruvka.py:87`) marks each KD-tree node with the union-find component of
  its points, so that the Borůvka candidate search can prune entire subtrees
  already in the same component — the core optimization that makes it fast.
- Parallel edge selection (`prange`) with a rank-based disjoint set
  (`disjoint_set.py`) to merge nearest inter-component edges each round until
  one component remains.
- `reproducible=True` switches to block-wise processing (`boruvka_tree_query_reproducible`)
  for deterministic parallel merges.
- Final edge weights are `sqrt`-ed (`boruvka.py:609`) to convert squared
  Euclidean distance back to Euclidean distance.

The result is a sorted edge list `(n-1, 3)` = the MST.

---

## 9. Stage 5 — Condensed Tree (HDBSCAN-style)

`mst_to_linkage_tree` (`cluster_trees.py:50`) turns the sorted MST into a
SciPy-style linkage tree via union-find, recording for each merge: children,
merge distance `delta`, and resulting cluster size.

`condense_tree` (`cluster_trees.py:138`) then does the HDBSCAN "condensation":
it walks the linkage tree top-down and

- records, for every point and every "true" cluster branch, the lambda value
  `λ = 1/distance` at which the point left the cluster (`lambda_val`),
- dissolves branches below `min_cluster_size`, re-attaching their points at the
  lambda where the branch disappeared.

The output is a `CondensedTree(parent, child, lambda_val, child_size)` namedtuple
— the same data structure HDBSCAN exposes. **λ is the resolution parameter**:
larger λ = finer clustering. This single structure is reused for every layer and
for the optional `approx_n_clusters` binary search.

`cluster_tree_from_condensed_tree` / `mask_condensed_tree` separate the
cluster-level nodes (`child_size > 1`) from the point-level rows; the cluster
tree is what persistence/peaks operate on.

---

## 10. Stage 6 — PLSCAN: Persistence Peaks & Multi-Layer Extraction

This is EVōC's novel departure from HDBSCAN's `eom`/`leaf` selection, and the
reason it returns *multiple layers* and *automatically selects the number of
clusters*. The report README points to PLSCAN (arXiv:2512.16558) for the theory;
here is the implementation.

### 6a. Barcode and total persistence — `min_cluster_size_barcode`

`clustering_utilities.py:165` builds, for each cluster, a `(birth_size,
death_size)` interval in terms of *cluster size*, plus the death λ. It then
`compute_total_persistence` (`clustering_utilities.py:194`) accumulates, for each
candidate birth size, the weighted sum of the interval lengths:

```
total_persistence[k] += (death - birth) * λ_death
```

This is essentially the "how long does a cluster survive as the resolution
scales" integral from persistent homology. A cluster that is stable across a wide
range of `min_cluster_size` accumulates high persistence.

### 6b. Peak detection — `find_peaks`

`clustering_utilities.py:33` is a Numba port of SciPy's `find_peaks` run over the
persistence curve. Local maxima of total persistence are candidate clusterings —
each peak corresponds to a "natural" number of clusters for the data.

### 6c. Diversity selection — `select_diverse_peaks`

`clustering_utilities.py:284`: peaks are sorted by persistence (best first) and
greedily accepted only if their resulting cluster set has Jaccard similarity
`<= min_similarity_threshold` (default 0.2) with every already-selected peak.
This prevents returning ten near-identical layers and yields the `max_layers`
(default 10) most distinct granularities. Similarity is estimated cheaply from the
barcode (`estimate_cluster_similarity`) rather than comparing full label vectors.

### 6d. Extraction — `extract_clusters_by_id`

For each selected peak, the clusters active at that birth size are pulled straight
from the condensed tree (`extract_clusters_by_id`, `clustering_utilities.py:228`)
into a `(labels, strengths)` pair:

- `get_cluster_label_vector` (`cluster_trees.py:502`) assigns labels via
  union-find over the condensed tree, honoring `cluster_selection_epsilon`; `-1`
  is noise.
- `get_point_membership_strength_vector` (`cluster_trees.py:550`) sets each
  point's strength to `λ_point / λ_death` of its cluster — a [0,1] confidence
  identical in spirit to HDBSCAN's member probabilities.

Layers are then sorted finest→coarsest by cluster count (`clustering.py:181`).

### 6e. Default `labels_` and the hierarchy

- `labels_` is the layer with the **highest total persistence** (`clustering.py:640`),
  i.e. the most stable clustering — not simply the finest.
- `base_n_clusters` switches the base layer to a `_binary_search_for_n_clusters`
  (`clustering_utilities.py:65`) that binary-searches `min_cluster_size` to hit a
  target count, then layers on top.
- `approx_n_clusters` short-circuits the whole layer machinery and returns a
  single vector (`clustering.py:372`).
- `cluster_tree_` (`build_cluster_tree`, `clustering_utilities.py:360`) maps each
  cluster in a coarse layer to the fine-layer clusters it contains, produced by
  a NumPy cross-layer label comparison (`_build_cluster_tree`).

---

## 11. Stage 7 — Duplicate Detection

`find_duplicates` (`clustering_utilities.py:373`) reuses the already-computed
k-NN graph: any pair `(i, j)` where `j` is within `max(nearest-neighbor distance)`
of `i` is flagged as a near-duplicate. This is essentially free — no extra pass
over the data — and is a nice fit for job-posting corpora that contain
reposted/near-identical listings.

---

## 12. How EVōC Differs from UMAP + HDBSCAN

| Aspect | UMAP + HDBSCAN | EVōC |
|---|---|---|
| Neighbor search | UMAP builds k-NN; HDBSCAN builds k-NN again | **Builds once**, reuses for graph, embedding, MST, and duplicates |
| Reduces dimensionality | Yes (usually 2D for plotting) | Yes, but to 4–15D for *clustering fidelity* |
| Global manifold optimization | Full UMAP SGD | Cheaper node embedding + recursive label-prop partition init |
| Cluster count selection | Requires `min_cluster_size` / `min_samples` tuning | **Persistence peaks** infer count + produce multiple layers |
| Multi-granularity | Separate `single_linkage_tree_` / manual `cluster_selection_epsilon` | Explicit `cluster_layers_` + cross-layer `cluster_tree_` |
| Quantized inputs | Manual metric plumbing | dtype-dispatched (float/int8/uint8) built in |
| Determinism | `random_state` | Explicit `reproducible_flag` block-synchronous SGD and Borůvka |
| Interface | Two libraries | One sklearn estimator |

The headline simplification is that **EVōC never needs to re-run neighbor search
between the "reduce" and "cluster" steps**, and **never asks the user for a
cluster count**.

---

## 13. Performance Engineering

The speed claims rest on four concrete choices:

1. **No Python loops in hot paths.** Every graph/embedding/tree operation is
   `@numba.njit(parallel=True, fastmath=True, cache=True)`. `cache=True` means
   first-call compile cost is amortized to disk.
2. **Reuse of the k-NN graph** across graph construction, embedding, MST, and
   duplicates ([Stage 2](#6-stage-2-k-nn-graph-construction-nn-descent)).
3. **Component-aware KD-tree Borůvka pruning** (`update_component_vectors`) —
   avoids recomputing distances for already-merged regions.
4. **Recursive/coarsened initialization** — label propagation runs on
   progressively smaller graphs, so the dominant O(n·k) work is done on a
   reduced problem.

Trade-off: with `reproducible_flag=True` the SGD uses block-synchronous updates
and the Borůvka search uses block-bounded queries, both of which are slower than
the async path. EVōC enables reproducibility automatically whenever
`random_state` is set (`clustering.py:630`), which is the right default for
auditable pipelines but should be benchmarked against the fast path.

---

## 14. API Surface & Key Parameters

```python
import evoc
clusterer = evoc.EVoC(
    n_neighbors=15,                # k-NN graph degree
    noise_level=0.5,               # attraction/repulsion balance in node embedding
    base_min_cluster_size=5,       # finest granularity
    base_n_clusters=None,          # target count for base layer (approx)
    approx_n_clusters=None,        # target total count; forces single layer
    min_samples=5,                 # core distance for mutual-reachability MST
    node_embedding_dim=None,       # defaults to min(max(k//4,4),15)
    neighbor_scale=1.0,            # denser/sparser graph multiplier
    min_similarity_threshold=0.2,  # layer diversity (Jaccard)
    max_layers=10,
    n_epochs=50,                   # SGD epochs
    node_embedding_init="label_prop",
    symmetrize_graph=True,
    random_state=None,             # setting this => reproducible path
)
labels = clusterer.fit_predict(embeddings)   # -1 = noise
layers = clusterer.cluster_layers_           # finest → coarsest
tree   = clusterer.cluster_tree_             # {(layer,cluster): [(layer,cluster)...]}
dupes  = clusterer.duplicates_               # {(i,j), ...}
```

This is a drop-in sklearn interface: `EVoC()` can replace `HDBSCAN()` in the
project's Step 1.3 interleaved-metric harness from [002](002-tutte-institute-tool-review.md)
with only a class swap.

---

## 15. Relevance to Semantic Career Mapping

EVōC maps cleanly onto the deferred "Clustering and job family discovery"
feature and onto the existing evaluation infrastructure:

- **Interleaved metric (Step 1.3).** `EVoC().fit_predict(embeddings)` gives
  cluster labels comparable to hand-labeled `role_category` — same use case as
  the planned HDBSCAN check, but with automatic cluster-count selection, which is
  valuable at small n where HDBSCAN's `min_cluster_size` is awkward.
- **Job-family discovery (deferred feature).** Multi-layer output is a natural
  fit for a *taxonomy*: coarse layers ≈ job families, fine layers ≈ specialisms.
  `cluster_layers_` + `cluster_tree_` can be fed straight into DataMapPlot
  (timc tool) for interactive labeled maps.
- **Duplicate detection.** `duplicates_` addresses the reposted/near-duplicate
  postings problem with no extra cost.
- **Quantized storage synergy.** The int8/uint8 paths mean embeddings stored in
  quantized form can be clustered without decompression — relevant if the object
  store holds quantized vectors.
- **CPU-first.** Matches the project's explicit no-GPU constraint (AGENTS.md).

**Caveat on scale:** at the current dataset size (tens of postings), EVōC's
machinery is overkill; the recursive partition init and multi-layer peak search
are designed for 10⁴–10⁶ points. Its value here is (a) as a validation tool and
(b) as future-proofing for larger corpora.

**Caveat on maturity:** 0.3.1 Beta, algorithm unpublished (cite PLSCAN), single
maintainer. It should be treated as an experiment dependency, not a load-bearing
production component, until the dataset justifies it.

---

## 16. Limitations & Risks

- **Beta software, narrow test surface.** Tests exist (`evoc/tests/`) but the
  README explicitly warns things "can and will break."
- **Unpublished algorithm.** The only citable reference is the PLSCAN paper;
  there is no peer-reviewed description of the full EVōC pipeline.
- **No incremental/streaming API.** `fit_predict` is batch-only; no partial fit,
  no `predict` on new points.
- **Numba tuning overhead.** First call triggers JIT compilation; cold-start
  latency can dominate on small data. `cache=True` mitigates but does not
  eliminate this.
- **Determinism costs speed.** `random_state` silently selects the slower
  reproducible paths.
- **tbb listed but not declared.** README lists `tbb` as a dependency;
  `pyproject.toml` does not. Threading performance may vary by environment.
- **Reproducibility is not bit-identical by default.** Without `random_state`,
  results vary run-to-run.

---

## 17. References

- Repository: <https://github.com/TutteInstitute/evoc>
- Docs: <https://evoc.readthedocs.io/en/latest/>
- PLSCAN paper (cluster extraction basis): D.M. Bot, L. McInnes, J. Aerts,
  *Persistent Multiscale Density-based Clustering*, arXiv:2512.16558,
  <https://arxiv.org/abs/2512.16558>
- Related project report: [002 — Tutte Institute Tool Review](002-tutte-institute-tool-review.md)
- Neighbor graph uses NN-Descent (Dong et al.) and UMAP-style fuzzy simplicial
  weights; MST uses parallel Borůvka; condensed tree / mutual-reachability
  follows HDBSCAN (Campello et al., McInnes et al.).
