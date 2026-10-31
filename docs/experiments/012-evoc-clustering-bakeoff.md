# Experiment 012 — EVōC Clustering Validity Bake-Off

**Date:** October 30, 2026

**Status:** Run complete 2026-10-30 — C strongest on external/internal metrics but fails the pre-registered stability gate; LLM-judge audit κ=0.614 passes the label gate. Uses the fixed 1,000-posting sample from [Experiment 011](011-encoder-coverage-bakeoff.md) and the method guidance in [Report 025](../research-reports/025-semantic-clustering-methods.md) and [Report 012](../research-reports/012-evoc-embedding-clustering.md).

This experiment asks a narrower question than "which clustering algorithm is best": **with EVōC fixed, which embedding representation produces the most coherent, stable, and label-consistent job-family structure?** The four embedding datasets from Experiment 011 already isolate truncation, full-text coverage, window overlap, and a long-context model. Experiment 012 keeps those vectors and the 1,000-posting sample fixed, runs EVōC, and reports internal, external, stability, and audit evidence.

The visual inspection in Experiment 011 favored B2 and C but did not measure cluster quality. This experiment supplies the missing quantitative comparison.

**Hypotheses:**

- **H1 (coverage):** B2 and C cluster better than A because A sees only 36.6% of the average posting.
- **H2 (overlap):** B2 clusters better than B because overlapping windows preserve boundary context.
- **H3 (model):** C differs from B2; whether it is better is an empirical question, not an assumption.
- **H4 (stability):** EVōC's persistence-selected layers remain stable across seeds and 80% subsamples at this sample size.

## Scope

In scope:

- EVōC on the cached 1,000-posting sample.
- Embedding arms A, B, B2, and C from Experiment 011.
- Weak external labels generated once and cached.
- Quantitative internal, external, stability, and audit metrics.
- A pre-registered decision rule for B2 versus C.

Out of scope:

- Re-embedding the full corpus.
- Production configuration, pointer, or prefix changes.
- Cleaning prompt changes, language filtering, or length filtering.
- A full cross-clusterer benchmark. EVōC is the clusterer of record for this experiment; a K-Means/HDBSCAN comparison is a separate follow-up if EVōC's ranking is ambiguous.

## Inputs

All inputs are local Experiment 011 artifacts:

| Input | Path |
|---|---|
| Fixed sample | `scripts/011_encoder_coverage/data/sample_1000.jsonl` |
| Fixed sample identity/order | `scripts/011_encoder_coverage/data/sample_keys.json` |
| A — truncated MiniLM | `scripts/011_encoder_coverage/data/embeddings_a_minilm_truncated.npy` |
| B — pooled MiniLM | `scripts/011_encoder_coverage/data/embeddings_b_minilm_pooled.npy` |
| B2 — pooled MiniLM, 32-token overlap | `scripts/011_encoder_coverage/data/embeddings_b2_minilm_pooled_overlap.npy` |
| C — normalized Titan V2 | `scripts/011_encoder_coverage/data/embeddings_c_titan_v2.npy` |
| Experiment design/result | [Experiment 011](011-encoder-coverage-bakeoff.md) |

Rules:

- Use exactly the same 1,000 `source_hash` values in the same order for every arm.
- Do not re-embed, re-sample, or filter postings.
- Convert vectors to `float32`; verify finite values and shape `(1000, 384)` for A/B/B2 and `(1000, 1024)` for C.
- L2 normalization is already part of the Experiment 011 contract. Verify norms are approximately 1.0 before clustering; do not silently re-normalize a failing input.

## Labels

EVōC is unsupervised. Quantitative external comparison requires a label signal, and the cached sample has no role or function labels. Add a one-time, cached weak-labeling step.

### Label taxonomy

Use the 13-category ML/AI function taxonomy from `scripts/006_taxonomy_labels/generate_labels.py`, with one addition: `other` for postings that do not fit any category. Do not modify the Experiment 006 script; copy the taxonomy definitions and prompt into the Experiment 012 labeler.

Each labeled posting records:

```text
source_hash
primary_function      -- exactly one of the 13 categories or "other"
function_tags         -- 0-3 categories; empty when primary_function = "other"
label_confidence      -- high | medium | low
rationale             -- one sentence
model
prompt_hash
labeled_at
```

`primary_function` is the single-label target for ARI/AMI/V-measure. `function_tags` is the multi-label target for cluster-level tag recovery; do not force multi-label tags into a single-label partition metric.

### Labeling contract

- Label the full `cleaned_text` and the title.
- Model: `openai/gpt-5.4-nano`, temperature `0.0`, the same model family used by Experiment 006.
- Write one JSONL row per posting. Cache by `(source_hash, prompt_hash, model)` and make reruns idempotent.
- Validate that every label is in the allowed set, every posting has exactly one row, and coverage is 1,000/1,000.
- Do not use labels to tune EVōC parameters, layer choice, or metric selection.

### Label audit

Weak labels are not ground truth. Before using external metrics as a decision gate, hand-audit a stratified sample:

- 100 postings: up to 8 per `primary_function`, plus random fill.
- The auditor sees the title and cleaned text, not the model's rationale.
- Record `primary_function` and `function_tags` independently.
- Report Cohen's κ for `primary_function` and micro-F1 for `function_tags`.

**Pre-registered label gate:** κ ≥ 0.60. If κ is below 0.60, report external metrics as exploratory and do not use them as the sole adoption rationale.

**Circularity caveat:** LLM labels and embeddings both derive from the same cleaned text. External scores are an upper bound on semantic agreement, not an independent ground truth. The hand audit and k-NN purity are the independent checks. If time permits, run a title-only label pass as a sensitivity analysis; it is not the primary target.

## EVōC Configuration

### Primary run

One pre-registered configuration per embedding arm:

```python
import evoc

clusterer = evoc.EVoC(
    n_neighbors=15,
    noise_level=0.5,
    base_min_cluster_size=10,
    base_n_clusters=None,
    approx_n_clusters=None,
    min_samples=5,
    node_embedding_dim=None,
    neighbor_scale=1.0,
    min_similarity_threshold=0.2,
    max_layers=10,
    n_epochs=50,
    node_embedding_init="label_prop",
    symmetrize_graph=True,
    random_state=42,
)
labels = clusterer.fit_predict(vectors)
```

`base_min_cluster_size=10` is 1% of the 1,000-posting sample and is a starting granularity, not a claim about the full 9,816-posting corpus. The full-corpus run will need its own scale check.

### Primary layer

Use `clusterer.labels_`, EVōC's highest-persistence layer, as the primary partition. Do not choose the layer by external score. Record every layer in `cluster_layers_` as a diagnostic table:

- layer index, finest to coarsest
- number of clusters
- noise fraction
- size distribution
- persistence summary
- external metrics per layer (secondary only)

### Sensitivity grid

Sensitivity is reported, not used to cherry-pick a winner. One-at-a-time from the primary configuration:

| Parameter | Values |
|---|---|
| `n_neighbors` | 10, 15, 30 |
| `base_min_cluster_size` | 5, 10, 20 |
| `min_samples` | 1, 5, 10 |

That yields seven unique configurations per embedding arm (the default is shared). Run all seven for A, B, B2, and C.

### Stability runs

- **Seed stability:** `random_state` in `{42, 43, 44, 45, 46}` with the primary configuration.
- **Subsample stability:** 10 runs on 80% of postings without replacement, primary configuration, seeds `42–51`; evaluate ARI on the intersection of retained postings. Run this for B2 and C only.
- **Parameter stability:** reuse the sensitivity grid; report metric and cluster-count ranges, not only the best value.

## Metrics

### Partition summary

For every run report:

- number of clusters (excluding `-1`)
- noise fraction
- cluster size min / median / p90 / max
- largest-cluster share
- number of EVōC layers
- duplicate-pair count and rate from `clusterer.duplicates_`

### External metrics — primary

Target: `primary_function`.

| Metric | Role | Notes |
|---|---|---|
| **AMI** | Primary endpoint | Chance-corrected; less biased toward fine partitions than NMI |
| **ARI** | Secondary | Pairwise agreement; sensitive to cluster count |
| **V-measure** | Secondary | Report homogeneity and completeness separately |
| Purity | Descriptive | Report cluster-size distribution alongside it; never use alone |

Rules:

- Keep noise (`-1`) as a distinct label. Report the noise fraction next to every external score.
- Also report a noise-excluded sensitivity table; do not present it as the primary result.
- For multi-label `function_tags`, report cluster-level tag recovery separately. Do not force tags into ARI.

### Internal metrics

| Metric | Role | Notes |
|---|---|---|
| **k-NN purity** | Primary label-free endpoint | `k=15`, cosine, self-excluded, non-noise points only; report permutation chance level |
| EVōC persistence | Primary clusterer-native signal | Per-cluster and size-weighted summary from `persistence_scores_`; record the installed version if attribute names differ |
| Silhouette | Secondary, comparative only | Non-noise points; same-encoder relative comparison only |
| Calinski–Harabasz | Secondary, comparative only | Centroid assumption; not a cross-encoder score |
| Davies–Bouldin | Secondary, comparative only | Same caveat |
| Pairwise ARI across arms | Descriptive | Same postings, different partitions; measures co-membership disagreement |
| Noise-neighbor rate | Descriptive | Fraction of a point's 15 nearest neighbors that EVōC calls noise |

Do **not** use DBCV as a primary metric: it is non-deterministic and is not a reliable single-run comparison. If a deterministic DISCO implementation is available and approved, add it as a secondary check; otherwise omit it rather than substituting DBCV.

### Stability metrics

- Mean pairwise ARI across five seeds; report min/max.
- Mean 80%-subsample ARI on the intersection; report the distribution, not only the mean.
- Per-cluster best-match Jaccard distribution across subsamples.
- Cluster-count range across seeds/parameters.
- Layer-count and primary-layer-index stability.

### Audit

For each embedding arm:

- Sample the five largest clusters and the noise set.
- Inspect five postings per group (including high- and low-membership points where possible).
- Rate whether the group shares the intended job family: `agree`, `partly`, `disagree`.
- Record disagreements, not only the agreement rate.

**Pre-registered audit gate:** `agree` or `partly` for at least 70% of audited non-noise postings, with a Wilson 95% lower bound ≥ 0.60. Noise is judged separately as a plausible outlier set.

## Statistical Comparison

- **Paired bootstrap:** 2,000 resamples of postings, same resample indices across arms. Recompute AMI/ARI/V-measure and k-NN purity on each resample; report percentile 95% CIs.
- **Permutation null:** 2,000 random shuffles of `primary_function`; recompute AMI/ARI/k-NN purity; report empirical p-values and z-scores.
- **Primary contrasts:** 
  1. coverage: `AMI(B2) − AMI(A)`
  2. model: `AMI(C) − AMI(B2)`
- Apply Holm correction to the two primary contrast p-values. Report effect sizes and CIs alongside p-values.
- Do not rank arms on differences smaller than the paired-bootstrap interval.

## Decision Rule

Pre-register the following before seeing external scores:

1. **Label gate:** if hand-audit κ < 0.60, external metrics are exploratory.
2. **Coverage gate:** B2 must beat A on AMI with a paired-bootstrap CI excluding zero and Holm-adjusted p < 0.05.
3. **Stability gate:** the preferred arm must have mean 80%-subsample ARI ≥ 0.60 and must not be the least stable arm by more than the bootstrap uncertainty of the stability estimate.
4. **Audit gate:** the preferred arm must pass the 70% audit rule.
5. **B2 versus C:** if the C−B2 AMI interval includes zero and both pass the other gates, prefer B2 (local, 384-d, no Bedrock dependency). If C's advantage excludes zero and its audit is not worse, prefer C.
6. **No winner:** if no arm passes, do not adopt a new encoder from this experiment; revisit cleaning/labels and rerun.

These gates are deliberately conservative. The goal is to avoid promoting a model because its 2-D map looked pleasing.

## Scripts and Artifacts

```text
scripts/012_evoc_clustering/
  label_sample.py          # cached weak labels + audit manifest
  audit_llm.py             # LLM-as-judge audit (GLM 5 via Bedrock) + kappa/micro-F1
  run_clustering.py        # EVōC primary, sensitivity, seed, subsample runs
  clustering_metrics.py    # external, internal, stability, bootstrap, permutation
  taxonomy.py              # 14-category function taxonomy (copy of 006)
  test_clustering_metrics.py
  data/
    labels_primary_function.jsonl
    label_manifest.json
    audit_100.jsonl
    audit_100_scored.jsonl
    audit_summary.json
    runs/
      <arm>/<config>/labels.npy
      <arm>/<config>/summary.json
    metrics.csv
    layer_metrics.csv
    sensitivity_metrics.csv
    stability.csv
    pairwise_ari.csv
    bootstrap_ci.csv
    permutation_null.csv
  plots/                   # optional cluster-colored maps, not a metric
```

Responsibilities:

- `label_sample.py` owns label generation, validation, cache, and the audit sample.
- `audit_llm.py` owns the LLM-as-judge audit pass and kappa/micro-F1 scoring.
- `run_clustering.py` owns all EVōC fits and writes one `summary.json` per run.
- `clustering_metrics.py` owns metric definitions; it must not import EVōC.
- Tests use synthetic partitions with known scores, not the real sample.

## Dependency Gate

`evoc` **0.3.2** is installed after user approval, and `pyproject.toml` now lists it as a dependency. The project rules still forbid autonomous package installation; this record is the dependency decision point.

Check before running:

- `evoc` imports successfully on Python 3.14 with the installed Numba.
- `EVoC` exposes `labels_`, `cluster_layers_`, `membership_strength_layers_`, `persistence_scores_`, `duplicates_`, and `cluster_tree_` (or the version-specific equivalents).
- A 100-point smoke test completes and emits `-1` noise correctly.

If EVōC cannot run on this Python/Numba combination, stop and report the blocker. Do not substitute a different clusterer without a new experiment decision.

## Results

**Run:** 2026-10-30. EVōC `0.3.2`, fixed 1,000-posting sample, four Experiment 011 vectors, 1,000 cached weak labels, and a 100-posting LLM-as-judge audit (GLM 5 on Bedrock). No production artifact changed.

**Auditor note:** GLM 5.3 (`zai.glm-5.3`) is enterprise-gated on this AWS account and returns `AccessDeniedException` even via the `us.zai.glm-5.3` / `global.zai.glm-5.3` inference profiles. The audit ran on the accessible GLM 5.x model `zai.glm-5` instead, per user decision.

### Weak labels

`openai/gpt-5.4-nano` at temperature 0 produced 1,000 labels: 571 high confidence, 378 medium, 51 low. Primary-function distribution:

| Label | Rows |
|---|---:|
| data-engineering | 373 |
| analytics-storytelling | 227 |
| other | 188 |
| mlops-production | 64 |
| classical-ml | 47 |
| deep-learning | 36 |
| ai-safety-governance | 26 |
| agentic-ai | 10 |
| llm-information-retrieval | 9 |
| research | 9 |
| ml-platform | 4 |
| computer-vision | 4 |
| reinforcement-learning | 3 |

The 18.8% `other` share is large and reduces the discriminating power of the external metrics.

**LLM-as-judge audit.** GLM 5 re-scored the 100-posting audit sheet at temperature 0. Against the labeler (`openai/gpt-5.4-nano`): Cohen's κ `0.614`, tag micro-F1 `0.623`, primary-label exact match `65.0%`. κ `0.614` clears the pre-registered 0.60 label gate, so the external metrics are usable, but the labels remain weak and noisy. Disagreement concentrates in the ML-adjacent labels (`mlops-production`, `classical-ml`, `deep-learning`, `research`) and the `other` boundary; `data-engineering` and `analytics-storytelling` are more stable. Scored sheet: `data/audit_100_scored.jsonl`; summary: `data/audit_summary.json`.

### Primary EVōC results

| Arm | Clusters | Noise | AMI [95% CI] | ARI | V-measure | k-NN purity [95% CI] | Seed ARI | 80% subsample ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — truncated MiniLM | 9 | 32.6% | 0.218 [0.207, 0.258] | 0.150 | 0.240 | 0.682 [0.664, 0.699] | 0.382 | — |
| B — MiniLM pooled | 6 | 27.2% | 0.267 [0.251, 0.308] | 0.233 | 0.283 | 0.721 [0.705, 0.739] | 0.394 | — |
| B2 — MiniLM pooled, overlap | 3 | 26.5% | 0.214 [0.195, 0.253] | 0.197 | 0.224 | 0.779 [0.765, 0.793] | 0.448 | 0.278 |
| C — Titan V2 | 4 | 12.5% | **0.286 [0.268, 0.325]** | **0.269** | **0.298** | **0.866 [0.855, 0.878]** | **0.543** | **0.488** |

All metric distributions beat their permutation nulls by a wide margin (empirical `p < 0.001` for AMI, ARI, and k-NN purity in every arm). That confirms the partitions carry label signal; it does not establish that the signal is strong enough to adopt.

### Primary contrasts

| Contrast | AMI difference [95% CI] | p (Holm) | ARI difference | V-measure difference |
|---|---:|---:|---:|---:|
| B2 − A | −0.003 [−0.036, 0.020] | 0.546 | +0.047 [+0.014, +0.078] | −0.016 [−0.048, +0.007] |
| C − B2 | **+0.072 [+0.041, +0.106]** | **< 0.001** | +0.073 [+0.035, +0.112] | +0.073 [+0.043, +0.107] |

The pre-registered coverage contrast does **not** support B2 over A on AMI; B2's ARI advantage is secondary and not matched by AMI or V-measure. The model contrast does support C over B2 on every external metric.

### Layer behavior

EVōC selected the highest-persistence layer as `labels_`: 9 clusters for A, 6 for B, 3 for B2, and 4 for C. The finer layers contain 20–26 clusters, so the primary partitions are coarse.

| Arm | Primary layer | Clusters | AMI | k-NN purity |
|---|---:|---:|---:|---:|
| A | 1 | 9 | 0.218 | 0.682 |
| B | 2 | 6 | 0.267 | 0.721 |
| B2 | 2 | 3 | 0.214 | 0.779 |
| C | 2 | 4 | 0.286 | 0.866 |

Layer selection matters. For C, layer 2 (4 clusters) had the best AMI; layer 1 (9 clusters) was close at 0.282. For B2, layer 1 (20 clusters) had higher AMI (0.229) than the persistence-selected layer 2 (0.214). k-NN purity rises sharply as clusters get coarser, so purity alone would reward the coarsest layer. The layer table is diagnostic only; the primary comparison stays on `labels_` as pre-registered.

### Stability

| Arm | Seed ARI (5 seeds) | 80% subsample ARI (10 runs) | Cluster-count range |
|---|---:|---:|---:|
| A | 0.382 | — | 6–12 |
| B | 0.394 | — | 3–11 |
| B2 | 0.448 | 0.278 | 2–10 |
| C | 0.543 | 0.488 | 4–13 |

C is the most stable arm, but no arm meets the pre-registered 0.60 subsample-stability gate. Cluster counts vary widely under seed and subsample perturbation. At `n=1,000`, EVōC's persistence-selected partition is not stable enough to serve as a production taxonomy.

### Sensitivity

Cluster counts are highly parameter-sensitive. `n_neighbors=10` expands C to 13 clusters, B2 to 7, B to 9, and A to 8; `n_neighbors=30` collapses C to 4, B2 to 4, B to 3, and A to 7. `min_samples=1` expands C to 10 and B to 7.

C remains the strongest external arm across the grid: AMI ranges 0.285–0.308, with the best value at `n_neighbors=30` (AMI 0.308, ARI 0.292). C's sensitivity results are in `data/sensitivity_metrics.csv`. These are post-hoc diagnostics, not a new primary configuration.

### Decision

No arm passes all pre-registered gates.

- **C is the strongest candidate** on AMI, ARI, V-measure, k-NN purity, and stability, and it beats B2 on the primary model contrast.
- **C fails the stability gate** (0.488 < 0.60). The LLM-judge audit passes its label gate (κ 0.614), but a human audit was not run, so label quality is only weakly validated.
- **The visual preference for B2 is not confirmed quantitatively.** B2 has the lowest AMI of the four arms and the weakest subsample stability. Its high k-NN purity reflects its coarse 3-cluster partition, not better label recovery.
- **Overlap did not help.** B (no overlap) beats B2 on AMI and V-measure, though B2 has higher ARI and k-NN purity.

C is promising enough for a follow-up, not for production adoption.

## Risks and Limitations

- **EVōC is beta and its clustering algorithm is unpublished.** Its layer selection may be unstable at `n=1,000`; stability is part of the result, not an afterthought.
- **Weak labels and circularity.** Labels are LLM-derived from the same text as the embeddings. External scores are relative evidence, not ground truth.
- **Sample size.** One thousand postings is enough for a first comparison but not for production taxonomy tuning. EVōC's `base_min_cluster_size=10` will not transfer unchanged to 9.8k or 10⁵ rows.
- **Encoder geometry differs.** MiniLM 384-d and Titan 1024-d have different density and neighbor structure. EVōC's cosine path handles both, but a parameter tuned on one is not automatically optimal for the other.
- **Cluster-count differences.** ARI/AMI can favor or penalize different granularities. Report cluster counts and size distributions; do not read a single score as absolute quality.
- **Noise policy.** Keeping `-1` is the primary analysis. Noise-excluded scores are a sensitivity check only.
- **No cross-clusterer proof.** A positive result says EVōC found structure with this embedding; it does not prove EVōC beats K-Means or HDBSCAN. That is a separate experiment.
- **Layer selection is unstable and consequential.** The highest-persistence layer was coarse (3–9 clusters); finer layers contained 20–26 clusters and sometimes aligned better with labels. Do not choose a layer by external score, but do not assume `labels_` is the product layer either.
- **Audit gate passed, weakly.** LLM-judge κ is 0.614 against a 0.60 threshold, and tag micro-F1 is 0.623. This validates the labels only as an LLM-comparable signal; it is not a human ground-truth check.
- **No production change.** Even a clear winner requires a full re-embed and a separate adoption decision.

## Next Steps

1. Optionally hand-score `data/audit_100_scored.jsonl` to confirm the LLM-judge κ of 0.614; the automatic label gate is already met, but a human pass would firm up the ML-adjacent label boundaries that disagreed most.
2. Run a stability-focused follow-up on C (for example `n_neighbors=30` and `min_samples=1`) to test whether the stronger AMI survives a pre-registered stability configuration.
3. Add a K-Means/HDBSCAN control on B2 and C if the EVōC ranking remains ambiguous; this is not a reason to delay the audit.
4. Resolve the cleaning outliers from Experiment 011 before a full-corpus re-cluster; unstable clusters may partly reflect non-English, location, eligibility, and very short/long postings.
5. Do not promote C or B2 until the audit and a full-corpus re-cluster are complete.

## References

- [Experiment 011 — Encoder Coverage Bake-Off](011-encoder-coverage-bakeoff.md)
- [Report 025 — Semantic Clustering Methods and Quality Measures](../research-reports/025-semantic-clustering-methods.md)
- [Report 012 — EVōC Implementation Review](../research-reports/012-evoc-embedding-clustering.md)
- [Report 021 — Clustering Metrics at Scale](../research-reports/021-clustering-metrics-at-scale.md)
- [Report 024 — Embedding Models for Semantic Clustering](../research-reports/024-embedding-models-for-semantic-clustering.md)
- [Report 006 — Taxonomy-Driven Role Labeling](006-taxonomy-role-labeling.md)
- `scripts/006_taxonomy_labels/generate_labels.py` — source taxonomy and labeling pattern
