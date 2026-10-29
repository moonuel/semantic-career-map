# Experiment 010 — UMAP Subsample-Fit Stability

**Date:** October 29, 2026

**Status:** Approved 2026-10-29 — preregistration page; harness not yet built. Preregistered gate and hypotheses recorded before the run, per the 007/009 protocol.

## Motivation

Report 022's rung-2 strategy — fit UMAP on a subsample, `transform` the tail — carries the corpus past 10⁵ postings, where a full k-NN phase stops fitting the local 23 GiB box. Two numbers make that strategy deployable instead of folklore: the **sample size m\*** at which a subsample fit stops losing map quality, and the **transform penalty** — how much worse a `.transform()`-projected point scores than an in-fit point sitting at the same coordinates. Neither has been measured on this corpus. A third gap compounds them: 009/009A settled the PCA-frame question at the *neighborhood* level (Jaccard@15, k_d\* = 256), but never observed UMAP's own output — map quality in the centered frame vs the sphere frame is still unverified at the artifact level, and 010 is the cheapest place to close it.

This experiment has no competing design pressure: at n = 9,354 a full fit already runs in minutes, so 010's deliverable is purely a **scaling contract for `scripts/map.py`** — the m-ladder to consult once `n ≫ m\*` — plus the frame-decision's end-to-end confirmation.

## Assumptions

Recorded explicitly; each is stress-tested by the design rather than silently adopted.

- **A1 — Neighborhood gate ≈ map quality (the 009A bet).** In-frame 15-NN retention is treated as a sufficient proxy for what k-NN-driven embedders preserve. 010 checks it: if the 256-dim frame's full-fit layout underperforms the sphere-frame control beyond the tolerance below, A1 fails and rung-1's `n_dims=256` recommendation reopens for UMAP alone.
- **A2 — Spectrum stability under n.** The 256-dim cut's validity (tail ≈ dust) transfers to larger corpora. The ladder's trustworthiness-vs-m trend is the earliest warning available here; a true re-probe happens at the 10⁵ rung, not in 010.
- **A3 — Seed is the only stochasticity.** UMAP with `random_state` pinned, `low_memory=True`, `n_jobs=1` (parallel SGD ordering is the known nondeterminism — pinned, as in the probe) is deterministic given its input matrix; observed cross-run variation therefore measures input-set variation, not implementation noise.
- **A4 — Trustworthiness is comparable across arms.** Each layout is scored against its *own frame's* 15-NN (cosine on the same input space the layout was fit on), so the metric reads "how faithfully this layout preserves its input's neighborhoods" — frame-independent by construction, comparable across arms.
- **A5 — 2D is primary.** The deliverable's Stage-1 artifact is 2D (DataMapPlot). 3D repeats only if the 2D readout is ambiguous.
- **A6 — No label dependency.** All headline metrics are unsupervised; HDBSCAN cluster-count preservation is optional color, never a gate.

## Methodology (preregistered)

**Input spaces** (same cache as 009, no fetch): `S_ref` = centered top-256 basis coordinates, re-normalized — `rn(U_256 S_256)` from the exact SVD (009A arm-A pipeline, the deployment layout); `S_sphere` = the original unit vectors, re-normalized. Deterministic recomputation from cache everywhere.

**Pinned UMAP parameters** (probe-verbatim): `n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42, low_memory=True, n_jobs=1`.

**Arms.**

| Arm | Fit on | Projected | Purpose |
|---|---|---|---|
| R0 | all of `S_ref` (m = 9,354) | — | deployment reference; the map as actually shipped |
| F0 | all of `S_sphere` | — | frame control: what centered-256 costs the map *itself* (tests A1) |
| L(m, s) | subsample m of `S_ref` | fit points + transform of the tail (two record modes) | the m-ladder: m ∈ {500, 1000, 2000, 4000} × 5 seeds (seeds fix the subsample choice, rng(42) substream per (m, s)) |

20 subsample fits + 2 full fits ≈ 22 runs, plus transform passes (5.4k–8.9k tail points at m=4k/2k — bounded, batched).

**Metrics per layout.**

1. **Trustworthiness@15** (`sklearn.manifold.trustworthiness`, cosine) against the layout's own frame's exact 15-NN — absolute, frame-neutral quality (A4).
2. **Cross-seed stability** — within an m, over seed pairs: Spearman r between the 2D distance vectors of the seed-shared points (distance geometry, not coordinates — UMAP frames are not aligned across runs; upper-triangle sampled to ≤ 200k pairs per comparison, seeded).
3. **Subsample-vs-full agreement** — same Spearman, between L(m, s)'s shared-point distances and R0's.
4. **Transform degradation** — Δ = trustworthiness(transformed tail points) − trustworthiness(fit points), per (m, s); the rung-2 `.transform()` cost of record.

Cross-seed and vs-full reads exclude transformed points (they double-count drift); fit-point layouts only.

**Preregistered gates and hypotheses.**

- **m\*** = smallest m with (a) median-over-seeds trustworthiness@15 ≥ R0's own − 0.02 (absolute 2-point margin, no relative jiggering) and (b) mean cross-seed Spearman ≥ 0.90.
- H1: an m ≤ 4,000 clears both clauses (the ladder suffices for rung 2).
- H2: transform degradation |Δ| > 0 but < 0.02 absolute (real but bounded; if Δ ≥ 0.02 at every m, `.transform()` is disqualified and rung 2 must re-plan).
- H3 (frame control): R0 and F0 trustworthiness within 0.02 of each other — A1 confirmed at map level; a miss reopens the 009A decision for UMAP alone.
- Prediction for the record: m\* = 2000, Δ small, H3 holds.

**Invariants.** Frozen config header (`010_config.json`: pinned params, commit, m-ladder, seeds, gates — 007 pattern), raw JSONL append-per-fit interrupt-safe, single seeded rng per (m, s) substream, no `backend/` imports, no new dependencies (`umap`, `sklearn` already present), harness frozen after its first recorded run, `.venv` execution, ruff gates. Runtime envelope ≤ ~30 min; peak RSS dominated by one full fit (< probe's envelope).

## Artifacts

| Artifact | Path | Committed |
|---|---|---|
| Runner | `scripts/010_umap_subsample_fit/run_benchmark.py` | yes |
| Analyzer | `scripts/010_umap_subsample_fit/analyze_results.py` | yes |
| Frozen config | `scripts/010_umap_subsample_fit/data/010_config.json` | yes |
| Raw per-fit rows | `scripts/010_umap_subsample_fit/data/010_raw_results.jsonl` | yes |
| Canonical summary | `data/010_pca_stability.json` | yes |
| Plots | `docs/experiments/assets/010_quality_ladder.png`, `010_stability.png` — colocated, embedded at results time | yes |
| Report | this page | yes |

## Post-Run Wire Plan

Register in `mkdocs.yml` nav on creation; results ripple to `results.md` (October-29+ block), `experiments/index.md` (planned → completed), and `project.md` (m\* into the Phase-3 line) via docs-sync at completion. Rung-2 default lands as a named constant in the 022 §7 ladder status.

## References

- [Report 022](../research-reports/022-umap-cloud-scale-deployment.md) — rung-2 sample strategy (§4.5) and ladder contract (§7).
- [Experiment 009](009-pca-informativeness.md) + 009A amendment — frame decision (k_d\* = 256), A1's proxy basis, ground-truth machinery.
- [Experiment 007](007-llm-latency-benchmark.md) — protocol precedent (frozen config, JSONL raw rows, no inferential statistics).
- Venna & Kaski — trustworthiness metric; `sklearn.manifold.trustworthiness` implementation notes (metric parameter, per-`k` definition).
