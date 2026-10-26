# Design — Embedding Pipeline

Architecture, decisions, and risk notes for the change. Spec deltas live in `specs/embedding-pipeline/spec.md`; this file records the reasoning behind them.

## Decisions (user-approved 2026-10-26)

1. **Cache granularity — SQL anti-join (row-level), mirroring `select_uncleaned`.** Selection = cleaned artifact rows for the pinned input version anti-joined against produced embedding rows on `(source_hash, configuration_hash)`: only postings with no embedding row under the run's configuration are embedded — the continuously-ingesting lake embeds deltas, never the corpus. The dataset-level short-circuit collapses into the degenerate case: an anti-join returning zero rows is the healthy no-op run (the cleaning tier's day-2 idempotence artifact). A configuration change still re-embeds deliberately — that lands under a new `version=` prefix as a new generation, not drift. Note on the withdrawn objection: the earlier design priced a Python-side row-level lookup ("full existing-parts scan for near-zero reuse value"); the SQL-side anti-join never performs that scan — its cost-bearing side is the tiny embeddings Parquet partition, server-side pruned, exactly as `artifacts_cleaned` is the tiny side for cleaning.
2. **`model_version` — config-pinned revision string, asserted at load.** The configuration carries `model_revision` (e.g. the HF revision hash of `sentence-transformers/all-MiniLM-L6-v2` the run intends); the encoder resolves the actually-loaded revision and fails fast on mismatch. Ambient resolution (whatever HF hands back) is banned from provenance — a silently-moved default revision would fork the vector space invisibly.
3. **SQL text storage — verbatim in `configs/<sha256>.json`.** The configuration object contains the full SQL text; its canonical serialization is hashed; the object is stored at `keys.config(hash)`. Manifests and rows carry only the hash — no keyspace pollution (data.md §5 config-object discipline), and any dataset resolves to the exact SQL text that selected it.
4. **One openspec change, four layers as task groups.** Spec deltas stay coherent as one capability name; layer boundaries become checkpoint gates (user review between layers).

## Layering rationale (feature-plan)

The risk is the integration seam — Athena runner lifecyle × model load × S3 write — not the pure functions. So Layer 1 is a throwaway spike (`scripts/embed_spike.py`) proving the whole seam with hardcoded inputs before any durable code exists. Layer 2 harvests only what the spike run used. Hardening (Layer 3) is bounded by the two reliable sources of evidence: what the spike exercised, and the `cleaned-artifacts` durability conventions this repo already adopted (checkpoint/flush discipline is repo policy, applied by override rather than spike evidence). Observability (Layer 4) adds manifest/accounting surfaces plus named log points, each answering a specific future question.

## Key shape (fixed by data.md §7)

```
lake/embeddings/model=<model>/variant=<variant>/version=<vN>/part-<run-id>-<seq>.parquet
```

- `model` partition: lowercased, slash-free segment of the model family (spike uses `all-minilm-l6-v2`); the full HF id rides in each row's `model` column, so partition name stays a prefixed grouping, not an identity claim.
- Rows: `source_hash, embedding, model, model_version, input_artifact, input_version, embedding_dimension, normalization, created_at` + `configuration_hash`, `run_id` (row-level provenance lives in the data too — not reconstructible only via manifests).
- One `<id>.npy` per posting is legacy (`live/`); the Parquet dataset is the authoritative representation; any index is rebuildable (data.md §7).

## Risk notes

- **Athena cost/scan:** the anti-join reads the tiny embeddings partition (server-side `model`/`variant`/`version`-pruned) against the artifacts side; both are Parquet-compact — the JSON raw side is deliberately not in this query. The spike caps rows with `LIMIT` and later drivers inherit the deployment-bounds pattern from the cleaning tier as needed.
- **Query-side registration dependency:** the anti-join can only see produced vectors through a Glue table registered over `lake/embeddings/` (partition columns `model`/`variant`/`version`). Provisioning rides the `provision_query_layer` pattern (additive, idempotent registration updates): no registration → no cache visibility → silent re-encodes. The embeddings table lands as part of Layer 2 with its own gate query.
- **Vector typing:** Parquet `list<float32>` for this change; a fixed-size-list migration (nice for downstream index builds) is additive later — versions add, they never get rewritten.
- **Writer reuse:** if the buffered flush/checkpoint writer duplicates `backend/pipeline/cleaning/writer.py` semantics, extract the shared writer rather than copy; the spike-to-extraction re-run will show which shape is honest.
- **Failure ledger:** cleaning's failure-ledger concept maps to embedding QA gate failures, not LLM failures — a posting with no usable text is a counted event, never a silent drop.
