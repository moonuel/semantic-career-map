# Experiment 011 — Encoder Coverage Bake-Off

**Date:** October 30, 2026

**Status:** Run complete 2026-10-30 — four maps rendered and awaiting inspection. Addressed to issue #36 and motivated by [Report 024](../research-reports/024-embedding-models-for-semantic-clustering.md) and [Report 023](../research-reports/023-context-engineering-for-embeddings.md).

This experiment asks a narrower question than issue #36 as written: **does the current MiniLM truncation defect dominate the map, and does full-text coverage or a stronger long-context model change what the semantic map shows?** The production encoder currently embeds only the first 256 MiniLM tokens of `cleaned_text`. With 93% of a random 100-posting sample exceeding that limit, the existing vector space is not a full-text representation and should not be treated as the baseline for model selection. Experiment 011 produces four comparable interactive maps over one fixed 1,000-posting sample. Selection is by direct inspection of those maps, not by an automatic leaderboard score.

**Hypothesis:** The truncated MiniLM map (A) will show a visibly poorer or more collapsed structure than full-content MiniLM mean-pooling (B/B2) and normalized Titan V2 (C), because A discards a large share of each posting. B isolates the effect of text coverage; B2 tests whether overlapping windows change the pooled representation; C adds long-context model quality on top of coverage.

## Experimental Design

### Sample

- **Population:** `semantic_career_map.artifacts_cleaned` rows with `variant='llm' AND version='v1'`.
- **Sample size:** 1,000 postings.
- **Selection:** fetch the full cleaned-row population, then draw a seeded random sample in Python (`seed=42`) without replacement. A seeded local sample is preferred over `ORDER BY rand()` because the sample must remain fixed across all four datasets and reruns.
- **Fixed identity:** all four datasets use the same 1,000 `source_hash` values in the same order.
- **Cache:** write the exact sample locally as JSONL with:
  - `source_hash`
  - `cleaned_text`
  - `title`, `employer`, `source` (joined from the raw lake where available)
  - MiniLM token count
  - sampling metadata: population size, seed, query text, timestamp, git commit

The cached sample is a local experiment artifact. The committed record is the sampling manifest and the sorted hash list, not the full text cache.

### Arms

| Arm | Name | Encoder | Input coverage | Dimensions | Normalization |
|---|---|---|---|---|---|
| A | Truncated MiniLM control | `sentence-transformers/all-MiniLM-L6-v2`, current pinned revision | first 256 tokens of `cleaned_text` | 384 | L2 |
| B | Mean-pooled MiniLM, no overlap | same MiniLM model and revision | full `cleaned_text` in 256-token windows | 384 | L2 per window, then pooled + L2 |
| B2 | Mean-pooled MiniLM, 32-token overlap | same MiniLM model and revision | full `cleaned_text` in overlapping 256-token windows | 384 | L2 per window, then pooled + L2 |
| C | Normalized Titan V2 | Bedrock `amazon.titan-embed-text-v2:0` | full `cleaned_text` | 1,024 | server-side `normalize=true`, verified client-side |

The comparison deliberately confounds model and coverage for C. B and B2 are the isolation arms: they keep the model fixed and change only coverage and chunk overlap. A is a diagnostic control, not a functional full-text baseline.

### Arm A — truncated MiniLM

Use the current production recipe:

```text
input text:     cleaned_text
max_length:     256
normalization:  L2
device:         CPU
batch size:     16
```

Re-embed the sample locally rather than reading production vectors. For hashes present in the current production prefix, compare the re-embedded vector against the stored production vector as a control; median cosine should be approximately 1.0 up to CPU nondeterminism.

Record per row:

- total MiniLM token count
- encoded token count (`min(total, 256)`)
- coverage ratio (`encoded / total`)
- truncation flag (`total > 256`)

### Arm B — mean-pooled MiniLM

Keep the same model, revision, tokenizer, and CPU settings. Cover the full cleaned text.

Primary pooling rule:

1. Tokenize the full `cleaned_text` with the MiniLM tokenizer.
2. Split the token sequence into non-overlapping content windows sized so each encoded window stays within the model's 256-token limit, including special tokens.
3. Decode each window back to text and encode it with the same MiniLM model.
4. Verify every window re-tokenizes to at most 256 tokens; fail the run if not.
5. L2-normalize each window vector.
6. Arithmetic-mean the window vectors.
7. L2-normalize the pooled vector.

Record per row:

- total token count
- number of windows
- encoded token count
- coverage ratio (should be 1.0 unless a window is rejected)
- pooled-vector norm before and after final normalization

No overlap is used for arm B. Arm B2 repeats the same pooling rule with a 32-token overlap between adjacent windows. Both variants use the same encoder, normalization, and mean-pooling; only the chunking signature differs.

### Arm C — normalized Titan V2

Use Amazon Bedrock `amazon.titan-embed-text-v2:0`.

Request contract:

```json
{
  "inputText": "<full cleaned_text>",
  "dimensions": 1024,
  "normalize": true
}
```

Record:

- Bedrock region
- request model ID
- response `inputTextTokenCount`
- output dimensions (must be 1,024)
- vector norm before and after any client-side normalization check
- request latency
- retry/failure count

Titan's tokenizer is not the MiniLM tokenizer. Use the response's `inputTextTokenCount` for Titan coverage reporting and verify no row exceeds Titan's 8,192-token context. If a row does exceed it, fail the run rather than silently truncating; at the measured corpus lengths this is not expected.

No task prefix is used. Titan V2 has no clustering input type; `normalize=true` is the model's native normalization contract.

### Controls and Constants

Held fixed across arms:

- the 1,000 sampled `source_hash` values and their order
- the same `cleaned_text` input
- L2-normalized output vectors
- no PCA pre-reduction
- the same UMAP parameters as `scripts/map.py`:
  - metric `cosine`
  - `n_neighbors = min(15, n - 1)`
  - `min_dist = 0.1`
  - `random_state = 42`
  - `low_memory = True`
  - `n_jobs = 1`
- the same map renderer, hover metadata, and click-detail text

### Diagnostics

The experiment is visual-first, but record these per arm so the maps can be interpreted:

| Diagnostic | Purpose |
|---|---|
| token coverage rate | confirms A is truncated and B/C cover the full text |
| truncation count | quantifies A's defect |
| vector norm distribution | verifies the normalization contract |
| pairwise cosine mean/quantiles | shows similarity-scale drift between models |
| nearest-neighbour overlap (A vs B vs C) | quantifies how much the maps disagree |
| encoder throughput and peak RSS | operational cost signal |
| Titan latency and failure count | Bedrock viability signal |

These are diagnostics, not adoption gates.

### Mapping

Four interactive 2D maps, one per dataset, using the same renderer and parameters as `scripts/map.py`.

`scripts/map.py` currently assumes production Athena vectors, a hardcoded configuration hash, 384 dimensions, and a MiniLM subtitle. The experiment uses a task-local copy (`scripts/011_encoder_coverage/map_local.py`) that:

- accepts a local `(hashes, vectors)` source
- accepts dynamic vector dimensions
- accepts a model label for the subtitle
- preserves the existing `project`, `write_coords`, hover, click-detail, and `render_interactive` behavior

Output layout:

```text
scripts/011_encoder_coverage/data/
  sample_1000.jsonl
  sample_manifest.json
  sample_keys.json
  embeddings_a_minilm_truncated.npy
  embeddings_b_minilm_pooled.npy
  embeddings_b2_minilm_pooled_overlap.npy
  embeddings_c_titan_v2.npy
  diagnostics_*.json
  titan_v2_cache.jsonl

scripts/011_encoder_coverage/plots/
  a_minilm_truncated/index.html
  b_minilm_pooled/index.html
  b2_minilm_pooled_overlap/index.html
  c_titan_v2/index.html
  manifest.json
```

Each map must contain all 1,000 points, resolve hover titles from the local sample cache, and include click-detail cleaned text.

### Planned Scripts

| Script | Responsibility |
|---|---|
| `scripts/011_encoder_coverage/run_experiment.py` | sample/cache, local MiniLM arms A/B/B2, Titan V2 arm C, diagnostics |
| `scripts/011_encoder_coverage/map_local.py` | local-vector copy of the production map renderer |
| `scripts/011_encoder_coverage/render_maps.py` | render all four interactive maps from cached vectors |

Reusable parts that graduate from this experiment belong in `backend/`; the numbered experiment scripts remain immutable records after the first run.

### Inspection

The user inspects all four maps and decides the next step. Possible outcomes:

- **B/B2 clearly improve on A:** full-text coverage is the main defect; model migration can be evaluated separately.
- **B2 improves on B:** overlapping windows retain useful boundary context; keep overlap in the next recipe.
- **C clearly improves on B/B2:** a long-context model adds value beyond coverage; plan a production model migration.
- **B/B2 and C all improve similarly:** prefer the cheaper local path if map quality is comparable.
- **No visible improvement:** the tail may not carry the signal the project hypothesized; reconsider the experiment or the map sample before any production change.
- **Maps are hard to compare:** add a follow-up diagnostic or a different map sample; do not force a model decision from an unreadable comparison.

No production configuration, embedding prefix, pointer, or map artifact is changed in this experiment.

## Results

**Run:** 2026-10-30. Population 9,816 cleaned rows; seeded sample 1,000 rows. All four embedding datasets and four interactive maps were produced. No production artifact was changed.

### Embedding diagnostics

| Arm | Rows | Dims | Input coverage | Windows | Wall time | Notes |
|---|---:|---:|---:|---:|---:|---|
| A — MiniLM truncated | 1,000 | 384 | 36.6% | 1 per posting | 68.5 s | 939/1,000 truncated (93.9%); median 631 tokens, p90 1,145, max 2,600 |
| B — MiniLM pooled | 1,000 | 384 | 100% | 3,308 | 238.2 s | 3.31 windows/posting mean, 15 max |
| B2 — MiniLM pooled, 32-token overlap | 1,000 | 384 | 100% | 3,706 | 252.4 s | 3.71 windows/posting mean, 18 max; 772,117 window-token incidences |
| C — Titan V2 normalized | 1,000 | 1,024 | 100% | 1 per posting | 837.0 s | Bedrock `us-east-2`; median 611.5 Titan tokens, p90 1,113, max 2,294; mean 0.56 s/request, p90 1.31 s |

The Titan wall time includes a resumed run: the first concurrent pass completed 125 rows before stalling; the sequential resume completed the remaining 875 rows. Per-request latency is the meaningful operational number.

### Maps

| Map | Path |
|---|---|
| A | `scripts/011_encoder_coverage/plots/a_minilm_truncated/index.html` |
| B | `scripts/011_encoder_coverage/plots/b_minilm_pooled/index.html` |
| B2 | `scripts/011_encoder_coverage/plots/b2_minilm_pooled_overlap/index.html` |
| C | `scripts/011_encoder_coverage/plots/c_titan_v2/index.html` |

Each map contains all 1,000 points, 246 detail buckets, and the patched interactive template (`tableFromIPC`, click handler, and detail loader present). Map manifest: `scripts/011_encoder_coverage/plots/manifest.json`.

### Inspection verdict

Pending user inspection. No automatic winner is declared.

## Discussion

Pending map inspection. The diagnostics already confirm the premise: truncation removes roughly two-thirds of the average posting (A coverage 36.6%), while B, B2, and C cover the full text.

## Limitations

- **Independent UMAP frames.** The four projections share parameters and seed but not a coordinate frame. Point positions are not directly comparable across maps; cluster structure, separation, and co-membership are the comparison targets.
- **Unlabeled sample.** The 1,000-posting sample has no role labels. The experiment cannot compute separation gap or k-NN label agreement on the full sample. It is a visual diagnostic.
- **Random sample variance.** One 1,000-row sample is not the corpus. The map may over- or under-represent some job families.
- **Mean pooling is one coverage fix.** It equal-weights windows regardless of length and may dilute short, high-signal sections. A different chunking or pooling rule could produce a different B.
- **Titan tokenizer and service drift.** Titan's tokenization differs from MiniLM's, and the Bedrock model is a closed service. Record the region, model ID, request date, and response metadata.
- **Similarity-scale drift.** MiniLM, pooled MiniLM, and Titan V2 can occupy different cosine ranges. Do not compare raw similarity values across maps as if they were calibrated.
- **Visual inspection is subjective.** This experiment is designed to decide whether a larger follow-up is worth it, not to establish a production-quality ranking.

## Next Steps

1. Inspect the four interactive maps.
2. Decide whether to keep MiniLM, promote coverage-only pooling, or test a long-context model as a production candidate.
3. If a candidate wins, design the new embedding configuration and full re-embed separately; this experiment does not change production.

## References

- [Report 024 — Embedding Models for Semantic Clustering](../research-reports/024-embedding-models-for-semantic-clustering.md)
- [Report 023 — Context Engineering for Embeddings](../research-reports/023-context-engineering-for-embeddings.md)
- [Experiment 009 — PCA Informativeness](009-pca-informativeness.md)
- [Experiment 010 — UMAP Subsample-Fit Stability](010-umap-subsample-fit.md) (deferred)
- `scripts/map.py` — pinned UMAP parameters and interactive map renderer
- `backend/pipeline/embedding/config.py`, `backend/pipeline/embedding/encoder.py`
- Issue #36 — replacement embedding model and re-embedding
