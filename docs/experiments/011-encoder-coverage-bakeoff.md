# Experiment 011 — Encoder Coverage Bake-Off

**Date:** October 30, 2026

**Status:** Proposed; not run. Addressed to issue #36 and motivated by [Report 024](../research-reports/024-embedding-models-for-semantic-clustering.md) and [Report 023](../research-reports/023-context-engineering-for-embeddings.md).

This experiment asks a narrower question than issue #36 as written: **does the current MiniLM truncation defect dominate the map, and does full-text coverage or a stronger long-context model change what the semantic map shows?** The production encoder currently embeds only the first 256 MiniLM tokens of `cleaned_text`. With 93% of a random 100-posting sample exceeding that limit, the existing vector space is not a full-text representation and should not be treated as the baseline for model selection. Experiment 011 produces three comparable interactive maps over one fixed 1,000-posting sample. Selection is by direct inspection of those maps, not by an automatic leaderboard score.

**Hypothesis:** The truncated MiniLM map (A) will show a visibly poorer or more collapsed structure than full-content MiniLM mean-pooling (B) and normalized Titan V2 (C), because A discards a large share of each posting. B isolates the effect of text coverage; C adds long-context model quality on top of coverage.

## Experimental Design

### Sample

- **Population:** `semantic_career_map.artifacts_cleaned` rows with `variant='llm' AND version='v1'`.
- **Sample size:** 1,000 postings.
- **Selection:** fetch the full cleaned-row population, then draw a seeded random sample in Python (`seed=42`) without replacement. A seeded local sample is preferred over `ORDER BY rand()` because the sample must remain fixed across all three arms and reruns.
- **Fixed identity:** all three arms use the same 1,000 `source_hash` values in the same order.
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
| B | Mean-pooled MiniLM | same MiniLM model and revision | full `cleaned_text` in 256-token windows | 384 | L2 per window, then pooled + L2 |
| C | Normalized Titan V2 | Bedrock `amazon.titan-embed-text-v2:0` | full `cleaned_text` | 1,024 | server-side `normalize=true`, verified client-side |

The comparison deliberately confounds model and coverage for C. B is the isolation arm: it keeps the model fixed and changes only coverage. A is a diagnostic control, not a functional full-text baseline.

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

No overlap is used in the primary arm. A small overlap (for example 32 tokens) is a documented sensitivity check, not part of the primary comparison.

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

Three interactive 2D maps, one per arm, using the same renderer and parameters as `scripts/map.py`.

`scripts/map.py` currently assumes production Athena vectors, a hardcoded configuration hash, 384 dimensions, and a MiniLM subtitle. The experiment needs a minimal local-vector path:

- accept a local `(hashes, vectors)` source
- accept dynamic vector dimensions
- accept a model label for the subtitle
- reuse the existing `project`, `write_coords`, `fetch_metadata`, `fetch_cleaned_text`, `write_details`, and `render_interactive` behavior

Output layout:

```text
scripts/011_encoder_coverage/data/
  sample_1000.jsonl
  sample_manifest.json
  embeddings_A_minilm_truncated.npy
  embeddings_B_minilm_pooled.npy
  embeddings_C_titan_v2.npy
  arm_manifest.json

data/plots/011/
  a_minilm_truncated/index.html
  b_minilm_pooled/index.html
  c_titan_v2/index.html
```

Each map must contain all 1,000 points, resolve hover titles from the raw lake, and include click-detail cleaned text.

### Planned Scripts

| Script | Responsibility |
|---|---|
| `scripts/011_encoder_coverage/sample_corpus.py` | fetch cleaned rows, draw the seeded 1,000-row sample, cache text + metadata + manifest |
| `scripts/011_encoder_coverage/embed_arms.py` | encode arms A and B locally; call Titan V2 for arm C; write vectors and per-arm diagnostics |
| `scripts/011_encoder_coverage/render_maps.py` | load cached vectors, project with the pinned UMAP settings, render one interactive map per arm |

Reusable parts that graduate from this experiment belong in `backend/`; the numbered experiment scripts remain immutable records after the first run.

### Inspection

The user inspects all three maps and decides the next step. Possible outcomes:

- **B clearly improves on A:** full-text coverage is the main defect; model migration can be evaluated separately.
- **C clearly improves on B:** a long-context model adds value beyond coverage; plan a production model migration.
- **B and C both improve similarly:** prefer the cheaper local path if map quality is comparable.
- **No visible improvement:** the tail may not carry the signal the project hypothesized; reconsider the experiment or the map sample before any production change.
- **Maps are hard to compare:** add a follow-up diagnostic or a different map sample; do not force a model decision from an unreadable comparison.

No production configuration, embedding prefix, pointer, or map artifact is changed in this experiment.

## Results

Not run. To be recorded after inspection.

## Discussion

Not run.

## Limitations

- **Independent UMAP frames.** The three projections share parameters and seed but not a coordinate frame. Point positions are not directly comparable across maps; cluster structure, separation, and co-membership are the comparison targets.
- **Unlabeled sample.** The 1,000-posting sample has no role labels. The experiment cannot compute separation gap or k-NN label agreement on the full sample. It is a visual diagnostic.
- **Random sample variance.** One 1,000-row sample is not the corpus. The map may over- or under-represent some job families.
- **Mean pooling is one coverage fix.** It equal-weights windows regardless of length and may dilute short, high-signal sections. A different chunking or pooling rule could produce a different B.
- **Titan tokenizer and service drift.** Titan's tokenization differs from MiniLM's, and the Bedrock model is a closed service. Record the region, model ID, request date, and response metadata.
- **Similarity-scale drift.** MiniLM, pooled MiniLM, and Titan V2 can occupy different cosine ranges. Do not compare raw similarity values across maps as if they were calibrated.
- **Visual inspection is subjective.** This experiment is designed to decide whether a larger follow-up is worth it, not to establish a production-quality ranking.

## Next Steps

1. Confirm Bedrock Titan V2 access in the configured AWS region.
2. Draw and freeze the 1,000-row sample.
3. Build the three-arm embedding script and reuse the map renderer with a local-vector path.
4. Inspect the three interactive maps.
5. Decide whether to keep MiniLM, promote coverage-only pooling, or test a long-context model as a production candidate.

## References

- [Report 024 — Embedding Models for Semantic Clustering](../research-reports/024-embedding-models-for-semantic-clustering.md)
- [Report 023 — Context Engineering for Embeddings](../research-reports/023-context-engineering-for-embeddings.md)
- [Experiment 009 — PCA Informativeness](009-pca-informativeness.md)
- [Experiment 010 — UMAP Subsample-Fit Stability](010-umap-subsample-fit.md) (deferred)
- `scripts/map.py` — pinned UMAP parameters and interactive map renderer
- `backend/pipeline/embedding/config.py`, `backend/pipeline/embedding/encoder.py`
- Issue #36 — replacement embedding model and re-embedding
