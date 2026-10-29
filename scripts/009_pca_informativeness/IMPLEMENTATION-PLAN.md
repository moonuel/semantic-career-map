# Experiment 009 — Implementation Plan

> Alignment doc for future turns carrying `docs/experiments/009-pca-informativeness.md` from preregistration to completion. **Not wired into the docs site or nav** — lives beside the harness, not under `docs/`. Design, metrics, grids, thresholds, and decision rule are frozen in the experiment page; this file sequences the build. If the two ever disagree, the experiment page wins and this plan gets corrected.

## Canonical constants (mirror of the preregistration)

| Constant | Value |
|---|---|
| `CONFIGURATION_HASH` | `a29a35e37a0d7025a961b5241ff278904500ed43edf022ac7304dacdb97e0c98` |
| Corpus rows / dims | 9,354 / 384 (float32, L2-normalized as stored) |
| `K_GRID` | `{8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384}` + computed signal count |
| Ground-truth kNN | exact cosine, k=15 primary, k=50 secondary |
| Family-C pairs | 200,000, seed 42 |
| Permutation replicas | 10, column-shuffled, seed 42 |
| Decision rule | smallest `k_d` with mean Jaccard@15 ≥ 0.95 AND `k_d` ≥ signal count; nothing ≤ 96 clears → reject pre-reduction |
| Seeds / thresholds | everything else seed 42; all thresholds above are the *only* free numbers — no post-hoc tuning |

## Target layout

```
scripts/009_pca_informativeness/
├── IMPLEMENTATION-PLAN.md        # this file — mutable until M9 done
├── __init__.py                   # package marker (007 precedent)
├── run_benchmark.py              # fetch → cache → SVD → metrics A–D → raw+summary JSON
├── analyze_results.py            # offline: tables + plots from cache + raw JSON only
└── data/
    ├── 009_config.json           # frozen config header (written at run start)
    ├── vectors_cache.npy         # NOT committed (14.4 MB)
    ├── source_hashes.json        # NOT committed (row-identity sidecar)
    └── 009_raw_results.json      # committed
```

Outputs after run: `data/009_pca_benchmark.json` (canonical summary), `docs/experiments/assets/009_scree.png`, `docs/experiments/assets/009_retention_curve.png` — plots live inside the experiments folder (user decision 2026-10-29), referenced relatively by the page; no `data/plots` copy.

## Milestones

| # | Milestone | Acceptance criterion |
|---|---|---|
| M0 | Scaffold: dirs + plan + `__init__.py` | layout above exists |
| M1 | `run_benchmark.py` skeleton: CLI, `--self-check`, config freeze | `--self-check` passes with no corpus; `009_config.json` written with all canonical constants |
| M2 | Fetch + cache layer | `--refresh-cache` writes `vectors_cache.npy` + `source_hashes.json`; duplicate-hash validation; cached rerun touches no Athena |
| M3 | Family A — spectrum | EVR/cumEVR table, PR, scree knee, permutation signal count all emitted into raw rows |
| M4 | Family B — kNN retention (primary) | per-`k_d` Jaccard@15, 1-NN retention, rank displacement; secondary k=50 read |
| M5 | Family C — distance fidelity | Pearson/Spearman/MAE on 200k pairs + contrast ratio, per `k_d` |
| M6 | Family D — label-anchored (optional) | separation gap original vs best `k_d` on golden join; skipped cleanly if join < 20 points |
| M7 | Raw + canonical JSON writers | `009_raw_results.json` + `data/009_pca_benchmark.json` with decision block applying the pre-registered rule verbatim |
| M8 | `analyze_results.py` | console tables + both PNGs into `docs/experiments/assets/`, offline (cache + raw JSON only), no recompute |
| M9 | Full run + record | corpus-wide run ≤ ~5 min, peak RSS < ~1 GiB; results + decision sections appended to the experiment page; wire 009 into completed grid per `wire-experiment`/`docs-sync` |
| M10 | Closure | report 022 §7 action-3 status line updated (via docs-sync); `n_dims` graduation note recorded — implementing the flag in `scripts/map.py` is **out of scope** (spike graduation owns it) |

## Build decisions (fixed now, to stop relitigating)

1. **Self-contained harness, 007 protocol.** No imports from other experiment dirs. SQL constants and the `_run`/`_athena` client pattern are copied into `run_benchmark.py` (probe-style, thin); Athena access via `backend.pipeline.athena.aws_region` is the only `backend` import allowed, as in the probes.
2. **NumPy-only core, scipy only for correlations.** PCA = explicit `np.linalg.svd` on the centered matrix; exact k-NN per space = blockwise `X@X.T` (row blocks of 2,048, cap transient RSS) + `argpartition` top-50, per-point strict-greater counting for rank displacement. No sklearn in this experiment (010 will use it for trustworthiness; 009 does not need it). scipy confirmed in `.venv` (1.18.1) → `scipy.stats.pearsonr`/`spearmanr` for family C; matplotlib (3.11.2) for the plots.
3. **One SVD, slices for all `k_d`.** `X_k = U[:, :k] * S[:k] @ Vt[:k]`. No incremental/randomized PCA.
4. **`--self-check` = the test story.** Tiny synthetic fixture (known clustered geometry, e.g. three orthogonal Gaussian blobs, no real posting data): asserts Jaccard/retention/rank/EVR/PR return hand-computable values; runs in milliseconds; part of every run, refuses a corpus run when it fails. Experiment scripts stay immutable artifacts — no pytest suite, no `backend/` extraction before graduation.
5. **Raw rows are append-per-stage** (007 interrupt-safety shape): each stage writes its own array into `009_raw_results.json` (readable = present, partial = rerun-stage only); a `--stage` flag resumes after interruption.
6. **No post-run edits to the harness.** Bug found after M9 starts → report it in the experiment page, fix forward only if the run is invalid, never silently; a fixed run replaces the whole results section with the old one preserved as an inline note (scientific integrity).
7. **Metrics correctness over cleverness.** Blockwise kernels are the only performance concern; all else straightforward numpy. No `n_jobs`, no BLAS tuning, no pynndescent.
8. **Style gates before every milestone closes:** `ruff format` + `ruff check` on touched files, `.venv` active for any execution, no new dependency lines added to any requirements/pyproject file.

## Per-turn protocol (future turns)

1. Re-read `docs/experiments/009-pca-informativeness.md` + this file; re-state the active milestone at turn start.
2. Do the smallest milestone or milestone-slice that fits the turn; update the milestone table state (`☐`/`☑` in a scratch line below is fine — this file is mutable until M9).
3. Design drift is a stop-and-flag: if implementation reality contradicts the preregistration, surface it in the turn report *before* the experiment page or harness changes, then record the agreed resolution in both.
4. M4's Jaccard@15 and the decision block in M7 are the two load-bearing outputs — if time is short anywhere else, cut there, never there.

## State

- [x] M0 scaffold — dir + plan + `__init__.py`
- [x] M1 runner skeleton + self-check + config freeze (self-check green; header freeze/verify works; note: no `date_utc` in frozen header — timestamps live in the cache payload)
- [x] M2 fetch/cache (Athena fetch 12.6 s; cache persisted, contract checks green)
- [x] M3 family A spectrum (PR 57.6, signal count 71, knee chord method → 381 = no interior knee)
- [x] M4 family B retention (14 exact passes; control column k_d=384 = centering ceiling 0.5752)
- [x] M5 family C fidelity (pearson plateau 0.6256 at full rank; contrast expansion recorded)
- [x] M6 family D labels (skipped cleanly per preregistration — no slug→hash bridge in local data)
- [x] M7 raw + canonical JSON (`009_raw_results.json` + `data/009_pca_benchmark.json`, decision k_d*=None)
- [x] M8 analyzer + plots (`docs/experiments/assets/009_scree.png`, `009_retention_curve.png`)
- [x] M9 full run + record + rewire (2026-10-29 run; results section appended to experiment page; 009 card moved to completed grid; results.md note added; mkdocs nav already wired)
- [ ] M10 closure — **awaiting user decision on the two follow-up variants** (centered-frame ground truth; LSA-style uncentered truncation) listed in the experiment page's Follow-up proposals. Report 022 §7 action-3 "revisit notes" happen there; no `n_dims` default named by this experiment (preregistered consequence).
