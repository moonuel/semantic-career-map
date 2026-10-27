# Add Embedding Pipeline

## Why

The lake ends at cleaned text: the vector step still lives in scattered experiment scripts (`scripts/005_semantic_partitioning/compare_embeddings.py` and the legacy `live/embeddings/<one .npy per posting>` branch), so embeddings carry no dataset identity, no configuration provenance, and no dataset manifests. `openspec/specs/embedding-pipeline/spec.md` is a two-requirement stub, `backend/storage/keys.py` documents the `lake/embeddings` branch as "reserved but not yet pinned", and the dev-log (2026-10-25) names this the next build: "build embedding vector key builder and pipeline code".

The cleaned-artifacts pipeline (spec + landed code) already proved the pattern: artifacts as Parquet parts, configuration_hash-addressed datasets, flush checkpoints, run records. This change applies the pattern to vectors: an Athena SQL selection over `artifacts_cleaned`, CPU Sentence-Transformers encode, and lakeside Parquet parts with per-row + per-run provenance.

## What Changes

- **Configuration identity**: an embedding configuration object (model name, pinned `model_revision`, batch size, variant field, prompt name, input SQL text, output version) hashed into a `configuration_hash`; identical values in any process hash identically, any field change re-hashes. Verbatim SQL stored content-addressed in `configs/<sha256>.json` (repo decision 3).
- **Athena retrieval**: selected `(source_hash, cleaned_text)` rows streamed through the shared `backend.pipeline.athena` runner; variant fallback survives missing/empty fields; loud empty-input accounting; abort before write on a non-SUCCEEDED query.
- **Lake writes**: rows `source_hash, embedding, model, model_version, input_artifact, input_version, embedding_dimension, normalization, created_at` (+ `configuration_hash`, `run_id`) committed as Parquet parts under `lake/embeddings/model=<model>/variant=<variant>/version=<vN>/`, per `docs/architecture/data.md` §7. Key builders + `validate_key` shape pinner land in `backend/storage/keys.py`.
- **Delta selection (the cache)**: the run's selection is an Athena anti-join — cleaned artifact rows for the pinned input version against produced embedding rows on `(source_hash, configuration_hash)`, mirroring the cleaning tier's `select_uncleaned` — so the continuously-ingesting lake embeds only un-embedded postings; a zero-row selection is the healthy no-op record. A configuration change lands a new `version=` prefix rather than rewriting vector space (repo decision 1).
- **Query-side registration**: a Glue table over `lake/embeddings/` with `model`/`variant`/`version` partition columns (the `provision_query_layer` registration pattern, additive and idempotent) so the anti-join can see produced vectors; without it the cache is invisible and runs silently re-encode.
- **Durability**: buffered writer with flush triggers; part → flush-checkpoint commit order; orphan-part healing; idempotent replays.
- **Observability**: dataset manifests, run records (counters, latency aggregates, `data_scanned_in_bytes`, workgroup), a post-write QA gate (norms, dimension constancy, nulls, row-count reconcile) that refuses SUCCEEDED on violation, and a read-only `scripts/inspect_embeddings.py` resolver for full parameter provenance.
- **Determinism of model identity**: the pinned `model_revision` is asserted against the resolved HuggingFace revision at load (repo decision 2) — drift fails fast, no ambient resolution.

## Capabilities

### New Capabilities
- None — the capability already exists as the `embedding-pipeline` spec stub.

### Modified Capabilities
- **`embedding-pipeline`**: the two stub requirements MODIFIED (revision-pinned model load; fallback semantics made observable per-row), and ADDED requirements layered as: configuration hashing, lakehouse key shapes with validation, Athena retrieval with variant fallback, anti-join delta selection with embeddings-table registration, batch-order part commit, dataset manifests, flush checkpoints with healing and idempotent replay, run-level accounting, QA gate.

## Impact

- **Code:** new `backend/pipeline/embedding/` (`config.py`, `encoder.py`, `driver.py`); `backend/storage/keys.py` gains `embedding_part` + the `lake/embeddings` shape pinner; `scripts/embed.py` CLI; `scripts/inspect_embeddings.py` read-only. Existing `scripts/00N_*` untouched.
- **Deploy:** `deploy/task-definition-embed.json` (clone of the clean task's contract) + an ECR image with the model baked in + an EventBridge schedule mirroring the clean cadence — the cleaning tier's batch shape, validated at the Layer 3 gate on the *final* reliability surface; SageMaker stays out of scope (design.md decision 5).
- **Spike:** `scripts/embed_spike.py` is a tracked throwaway proving the seam; deleted in its own commit once Layer 2 extraction re-runs clean (feature-plan lifecycle).
- **Data:** new `lake/embeddings/` version prefixes; each version is additive — never overwritten (data.md §5).
- **Deps:** none added — sentence-transformers, numpy, pyarrow, pandas, boto3 already in `pyproject.toml`.
- **Docs:** README "Current Status" + docs-sync cross-references + dev-log at completion; `docs/architecture/data.md` §7 unchanged (the layout is already written — this change implements the reserved branch).
