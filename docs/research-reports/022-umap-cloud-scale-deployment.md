# Scaling UMAP to Cloud Hardware — Memory Maps, Memory-Optimized Instances, and GPU Out-of-Core

> Date: 2026-10-28
> Context: The job-map deliverable projects the full production corpus (`postings_embeddings`, 9,354 rows × 384 dims, float32) through UMAP. `scripts/map_umap_probe.py` already runs the whole corpus with `low_memory=True, n_jobs=1`, but that is the tail end of a much larger growth curve: the sampling manifests (`data/sample_manifests/per_source*_seed7_bands5000_15000.json`) are designed toward hundreds of thousands of postings. UMAP's memory demand grows super-linearly with corpus size and will outgrow this machine (23 GiB RAM, 12 cores, CPU-only) long before the pipeline is done. This report maps the memory model, evaluates memory maps and cloud instance choices, and sets a graduation ladder.

---

## Table of Contents

1. [Why UMAP Runs Out of Memory](#1-why-umap-runs-out-of-memory)
2. [UMAP's Memory Model](#2-umaps-memory-model)
3. [Memory Maps — What They Solve, What They Do Not](#3-memory-maps-what-they-solve-what-they-do-not)
4. [CPU-Side Mitigation Ladder](#4-cpu-side-mitigation-ladder)
5. [Cloud Hardware Options](#5-cloud-hardware-options)
6. [GPU Out-of-Core UMAP (2026 State of the Art)](#6-gpu-out-of-core-umap-2026-state-of-the-art)
7. [Recommendation & Graduation Ladder](#7-recommendation-graduation-ladder)
8. [Decision Log](#8-decision-log)

---

## 1. Why UMAP Runs Out of Memory

UMAP is not a single pass over the data. It is three phases, and the middle one is the killer:

1. **k-NN graph construction** — find the `n_neighbors` nearest neighbours of every point. This is the dominant cost in both time and memory.
2. **Fuzzy simplicial set** — turn the k-NN graph into a weighted sparse graph (roughly `n × n_neighbors` edges).
3. **Embedding optimization** — SGD over a low-dimensional layout against that sparse graph.

The input matrix is the cheap part. For this project's 9,354 × 384 float32 that is only ~14.4 MB. What blows up is phase 1, because the *all-neighbours* distance computation is O(n²) when done exactly, and even the approximate NN-descent path materializes large intermediate distance blocks. The MLSys 2026 paper on out-of-core UMAP ([§6](#6-gpu-out-of-core-umap-2026-state-of-the-art)) reports that a **16M subsample of the MIRACL corpus required ~1.4 TB of CPU memory** to run CPU UMAP — i.e. the CPU implementation is the binding constraint, not the input.

The current `map_umap_probe.py` docstring records the same failure mode locally: the default *exact* neighbour path "built dense distance matrices big enough to exhaust memory" — the reason the spike pins `low_memory=True, n_jobs=1`.

---

## 2. UMAP's Memory Model

Verified against the installed `umap-learn 0.5.12` / `pynndescent 0.6.0` source (in `.venv`).

### 2.1 Exact vs. approximate neighbour path

`umap.UMAP` chooses the neighbour search at `umap_.py:2550`:

```python
elif X[index].shape[0] < 4096 and not self.force_approximation_algorithm:
    self._small_data = True
    dmat = pairwise_distances(X[index], metric=_m, **self._metric_kwds)
```

- **`n < 4096` and not forced-approximate** → `sklearn.metrics.pairwise_distances`, a **dense `(n, n)` float64 matrix**, O(n²) memory. At n = 3,000 that is ~72 MB; at n = 50,000 it would be ~20 GB (but by then the threshold routes to the approximate path). This is the "default exact search" that OOM'd the probe.
- **`n ≥ 4096`** → `pynndescent.NNDescent`, an approximate NN-descent index.

### 2.2 `low_memory`

`umap.UMAP` defaults `low_memory=True` (`umap_.py:264, 293`) and threads it into `pynndescent.nn_descent(low_memory=...)` (`umap_.py:2642, 3458`). In pynndescent this flag means *"lower memory, but more computationally expensive approach to index construction"* — the NN-descent computes pairwise distances **in blocks** rather than materialising one full `(n, n)` distance matrix. It is a **compute-for-memory trade**, not a disk-backed memory map.

### 2.3 Where the bytes actually go

| Component | Growth | Note |
|---|---|---|
| Input matrix | `n × d × 4` (float32) | Cheap; ~14 MB at 9.4k, ~154 MB at 10⁵, ~1.5 GB at 10⁶ |
| Exact path distance matrix | `n² × 8` (float64) | The O(n²) OOM, only when `n < 4096` |
| NN-descent blocks (approximate) | `~n × k × block` | Bounded by `low_memory` blocking, but multiplies with `n_jobs` |
| Fuzzy simplicial set | `~n × n_neighbors` edges (sparse) | Linear in `n`, constant in `d` |
| Embedding arrays | `n × n_components` | Negligible |

The practical takeaways: keep `n` above 4096 to avoid the exact path (or set `force_approximation_algorithm=True`), keep `n_jobs` low enough that parallel distance blocks do not multiply peak RSS, and keep the input float32.

---

## 3. Memory Maps — What They Solve, What They Do Not

"Memory map" is doing two different jobs here, and conflating them leads to wrong expectations.

### 3.1 `numpy.memmap` — out-of-core *input*, not out-of-core *algorithm*

`numpy.memmap` (backed by the OS `mmap`) lets a file be accessed as a `numpy` array while the kernel pages it in and out on demand. It is the standard mechanism for *loading* arrays larger than RAM (NumPy, PyTorch dataloaders, and vLLM weight loading all use it — see the OSDI 2026 "Umap: Revisiting Memory-mapped I/O" paper). For UMAP specifically it buys two things:

- **Input staging.** Write the `(n, 384)` embeddings to an `.npy` and open it as `np.memmap(mode="r")`; UMAP's float32 input never needs to be fully materialised in a second copy.
- **Coordinate output.** Write `coords_2d.json`/`coords_3d.json` (and any `.npy` intermediates) to a memmap and flush lazily, so the spike never holds the whole output in RAM.

What it **does not** buy: UMAP's hot loops (NN-descent distance blocks, the fuzzy-simplicial-set sparse graph, the SGD embedding) perform *random access* over the data. If the working set exceeds available RAM, `mmap` degenerates into page-cache thrash — it trades an OOM for an unbounded slowdown. **`memmap` is not a substitute for RAM in UMAP's compute; it only removes redundant copies of the input and outputs.** Backed by fast local NVMe (EBS `io2`/`gp3`, or instance-store NVMe), it removes the *input-copy* overhead; it cannot make phase 1 out-of-core.

### 3.2 Naming collision — do not confuse the two "UMaps"

| Name | What it is | Relevance |
|---|---|---|
| `umap-learn` (McInnes) | The manifold-learning library this project uses | The thing being scaled |
| LLNL **UMap** (`github.com/LLNL/umap`) | A user-space **memory-mapping runtime** (user-level `mmap`) for out-of-core/NVRAM/disaggregated-memory data access | A *general* out-of-core primitive, not a UMAP projection accelerator |
| OSDI 2026 "Umap" (He et al.) | A DFS-aware mmap runtime for distributed-file-system matrix access | Same family as LLNL UMap; not the manifold algorithm |

They share a name and nothing else. The LLNL/OSDI "UMap" work is interesting as a *backing store* for `numpy.memmap`-style out-of-core matrix access, but it does not make `umap-learn` scale by itself.

---

## 4. CPU-Side Mitigation Ladder

Ordered cheapest-first; each step is independently useful and composes:

1. **Keep input `float32`.** Already done. Halves the input and the distance-block memory vs. float64.
2. **`low_memory=True` (default) + low `n_jobs`.** Already pinned in `map_umap_probe.py`. Keeps NN-descent distance blocks from multiplying across workers. 12 cores → `n_jobs=1` is deliberately conservative; raise only after measuring peak RSS.
3. **Stay on the approximate path.** Ensure `n ≥ 4096` at runtime, or set `force_approximation_algorithm=True`, so the O(n²) `pairwise_distances` branch is never taken.
4. **PCA / random-projection pre-reduction.** 384 → ~50 dims with PCA (or a random projection) before UMAP collapses the k-NN distance cost almost 8× and is near-lossless for embeddings whose intrinsic rank is far below 384. This is the single biggest CPU win.
5. **Fit on a subsample, `transform` the rest.** `umap.UMAP` supports `.fit(sample)` then `.transform(remainder)` (and cuML supports Dask-distributed `transform`). At constant map quality you fit on e.g. 10⁴–10⁵ anchors and project the tail in batches — turning a super-linear problem into a bounded one. This is the *de facto* standard for interactive maps at 10⁵–10⁶ points.
6. **Alternative ANN backends.** `force_approximation_algorithm=True` routes to `annoy` for the metrics it supports (euclidean, cosine, etc.); a precomputed k-NN graph (`precomputed_knn`, or passing a precomputed sparse graph) lets HNSW/FAISS do phase 1 outside UMAP.

Steps 4 + 5 together are the realistic CPU ceiling: they carry UMAP to ~10⁶ points on a well-provisioned node.

---

## 5. Cloud Hardware Options

The decisive question is **CPU-only vs. GPU**, because this project's posture is CPU-first (`AGENTS.md`: no GPU deps unless explicitly requested). At the current scale the answer is CPU-only; the GPU tier exists as a break-glass for the 10⁶+ rung.

### 5.1 CPU — memory-optimized EC2 (the default tier)

| Instance family | Max RAM | Fit |
|---|---|---|
| `r7i` / `r7iz` (Intel) | up to 1.5 TiB | Standard memory-optimized; 10⁵–10⁶ points |
| `r8g` (Graviton4) | up to 1.5 TiB | Cheaper per GiB; ARM — verify numba/`umap-learn` wheels build on ARM before committing |
| `x2idn` / `x2iedn` (Intel) | up to 4 TiB | The 10⁶–10⁷ CPU rung, when fit+transform headroom runs out |
| `u-*` high-memory | up to 24 TiB | Overkill; only for a pathological exact-path run |

Notes:

- **Attach fast NVMe for memmap backing.** `gp3`/`io2` EBS, or the instance's local NVMe instance store, keeps `memmap` input/output paging cheap (§3.1). EBS is durable across stop/start; instance store is ephemeral but faster and free. For a one-shot batch projection, instance store + S3 upload of the coordinate JSON is the cheap pattern — consistent with the project's S3-first migration (`architecture/aws-migration-roadmap.md`).
- **Spot + stop/start.** UMAP projection is a bounded batch job, not a service: run on spot instances and save the memmap'd input + coordinates to S3 so a preemption mid-run is re-runnable. Cost is order-of-magnitude below on-demand; treat exact prices as region/time-varying and pin a ceiling with a Spot block or a max-price cap.

### 5.2 GPU — when the CPU ladder is exhausted

| Instance | GPU | GPU mem (HBM) | System mem |
|---|---|---|---|
| `g6.4xlarge` (L4) | 1× L4 | 24 GB | 64 GiB |
| `p5.4xlarge` (H100) | 1× H100 | 80 GB HBM3 | 256 GiB |
| `p5.48xlarge` (H100) | 8× H100 | 640 GB HBM3 total | 2 TiB |
| `p5e.48xlarge` (H200) | 8× H200 | 1128 GB HBM3e total | 2 TiB |

Only enter this tier when `n` reaches the low millions and CPU `fit` on a subsample no longer holds quality. It pulls in a RAPIDS `cuML` dependency, which conflicts with the project's CPU-first rule and should be an explicit, approved exception (see §6).

---

## 6. GPU Out-of-Core UMAP (2026 State of the Art)

The relevant current work is NVIDIA/RAPIDS **cuML**, specifically its 2026 out-of-core all-neighbours k-NN construction:

- **MLSys 2026 paper — "Massive-Scale Out-Of-Core UMAP on the GPU"** (Park, Nolet, Raff, Oates, Naruse). Builds the all-neighbours k-NN graph in chunks that never require the full dataset in GPU memory, with optional multi-GPU support. Reports **22.7× speedup** on a single GPU at scales where CPU completes, and a **projected 74×** multi-GPU speedup at scales where CPU UMAP cannot finish at all (the MIRACL 16M case needing ~1.4 TB CPU RAM).
- **NVIDIA blog (2026-08-18) — "Run Massive-Scale UMAP in Minutes Using Multiple GPUs—Without Losing Accuracy."** The companion practical guide.

Key API knobs in `cuml.manifold.UMAP` (cuML 26.08+):

- **`knn_n_clusters`** — split the corpus into this many chunks for k-NN graph construction; raise it until each chunk's vectors + local graph fit comfortably in GPU memory. Higher = less GPU memory per step.
- **`knn_overlap_factor`** — controls how many neighbouring clusters a cluster's local graph overlaps with; raises accuracy at the cost of compute. Start at 2, increase for better recall.
- **`precomputed_knn`** — supply the k-NN graph yourself (e.g. HNSW on CPU/GPU) and skip cuML's construction entirely.
- **`cuml.dask.manifold.UMAP`** — Dask wrapper for **distributed inference only**: train `cuml.UMAP` on one GPU, then fan out `.transform` across GPUs/nodes. Distributed *training* is not supported.

Positioning: this is the *extreme-scale* rung (10⁶–10⁸ vectors). It is not the default for this project, but it is the documented ceiling if the corpus ever reaches millions of postings, and it is why the project should **not** try to "make CPU UMAP out-of-core by hand" — the work has already been done properly in cuML.

---

## 7. Recommendation & Graduation Ladder

**Adopt a CPU-first, memory-map-staged, cloud-batch projection path**, graduated by corpus size. The current 9,354-row projection already runs locally; the ladder is about not letting the next order of magnitude become an emergency.

| Rung | Corpus `n` | Input size | Strategy | Hardware |
|---|---|---|---|---|
| 0 — current | ~10⁴ | 14 MB | `low_memory=True, n_jobs=1` (as today) | local 23 GiB |
| 1 | ~10⁵ | 154 MB | + PCA→~50d, stay approximate, memmap input/output | local or `r7i`/`r8g` |
| 2 | ~10⁶ | 1.5 GB | + fit-on-subsample + `transform` tail | `r8g`/`x2iedn` + NVMe instance store, spot |
| 3 | ≥10⁷ | ≥15 GB | cuML out-of-core (`knn_n_clusters`, multi-GPU) — explicit exception to CPU-first | `p5`/`p5e` |

Concrete near-term actions (no new dependencies for rungs 0–2):

1. **Extract the projection into a reusable module** (`backend/`), per the immutable-scripts rule — `map_umap_probe.py` is a spike that gets deleted once `scripts/map.py` graduates. Move the pinned UMAP parameters and the memmap staging there.
2. **Memmap the input and the coordinate outputs** (`.npy` in, `.npy`/JSON out) so a cloud run never holds the full matrix twice, and so a preempted spot run resumes from S3.
3. **Add PCA pre-reduction as a flag** (384 → ~50), defaulted on for `n > 10⁵`. This is the highest-leverage CPU knob and touches no dependency.
4. **Wire `fit(sample)` / `transform(tail)`** for rung 2, with the sample size as a CLI arg, so the map remains a one-shot bounded batch job.
5. **Keep cuML off the dependency list** until a measured `n ≥ 10⁶` run actually fails the CPU ladder — then raise it as an explicit decision (§8), not a silent add.

Memory maps are the *staging* mechanism (input/output, cheap page-in on NVMe), **not** the scaling mechanism. Scaling comes from the CPU ladder (PCA + subsample-fit + approximate k-NN) and, only at the top, from cuML's GPU out-of-core k-NN.

---

## 8. Decision Log

| Decision | Rationale | Date |
|---|---|---|
| Memory maps = staging, not scaling | `mmap` removes redundant input/output copies but UMAP's random-access hot loops thrash if the working set exceeds RAM; it cannot make phase 1 out-of-core (§3.1) | 2026-10-28 |
| CPU-first, memory-optimized EC2 as the default tier | Matches the project's CPU-only posture; PCA + subsample-fit + approximate k-NN carry UMAP to ~10⁶ points without new deps (§5.1, §7) | 2026-10-28 |
| PCA pre-reduction (384→~50) as the primary CPU lever | Collapses k-NN distance cost ~8× with negligible loss for low-rank embeddings (§4.4) | 2026-10-28 |
| cuML GPU out-of-core reserved for the 10⁶+ rung, as an explicit exception | MLSys 2026 out-of-core UMAP is the proven extreme-scale solution (22.7×/74× speedups; CPU needs ~1.4 TB RAM at 16M rows); adopting it early would violate the CPU-first rule for no current benefit (§6, §7) | 2026-10-28 |
| Exclude LLNL/OSDI "UMap" as a scaling path | Same name, different artifact — a user-space mmap runtime, not a manifold scaler; at best a backing store for §3.1, not a substitute for the ladder (§3.2) | 2026-10-28 |

---

## References

- Park, Nolet, Raff, Oates, Naruse — *Massive-Scale Out-Of-Core UMAP on the GPU*, MLSys 8 (2026). https://proceedings.mlsys.org/paper_files/paper/2026/hash/0badcb4e95306df76a719409155e46e8-Abstract-Conference.html
- NVIDIA Technical Blog — *Run Massive-Scale UMAP in Minutes Using Multiple GPUs—Without Losing Accuracy* (2026-08-18). https://developer.nvidia.com/blog/run-massive-scale-umap-in-minutes-using-multiple-gpus-without-losing-accuracy
- NVIDIA cuML — `cuml.manifold.UMAP` API reference (26.08). https://docs.nvidia.com/cuml/26.08/api/generated/cuml.manifold.UMAP/
- AWS — Amazon EC2 P5 instances (H100/H200 specs). https://aws.amazon.com/ec2/instance-types/p5/
- LLNL — *UMap: A user-level memory mapping library*. https://github.com/LLNL/umap
- He et al. — *Umap: Revisiting Memory-mapped I/O on Distributed File Systems*, OSDI 2026.
- Nolet et al. — *Bringing UMAP Closer to the Speed of Light with GPU Acceleration*, AAAI 2021 (RAPIDS cuML origin).
