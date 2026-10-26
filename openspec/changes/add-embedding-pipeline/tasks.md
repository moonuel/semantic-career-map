# Tasks — Add Embedding Pipeline

Layered per the feature-plan structure. Verification command listed per task; each layer ends at a user-review gate before the next begins. Spike script is tracked but throwaway — removed in its own commit at Layer 2 completion.

## 1. Layer 1 — Baseline spike (prove the seam)

- [ ] 1.1 Write `scripts/embed_spike.py` (hardcoded SQL over `artifacts_cleaned` with LIMIT, hardcoded model/batch/version v1, no manifests/retries/error handling): Athena select → CPU encode (all-MiniLM-L6-v2) → L2 normalize → one Parquet part under `lake/embeddings/version=v1/` → read back → assert norms/dimension/count. Verification: `python scripts/embed_spike.py` runs clean and prints the reconcile summary.
- [ ] 1.2 **Checkpoint (user gate):** spike output presented; go / kill / redirect.

## 2. Layer 2 — Extraction (harvest spike into modules)

- [ ] 2.1 `backend/pipeline/embedding/config.py` — config dataclass + `configuration_hash()` (canonical, field-order-fixed, no runtime-affine values). Tests: `tests/test_embedding_config.py`. Verification: scoped pytest green.
- [ ] 2.2 `backend/pipeline/embedding/encoder.py` — order-preserving `encode(texts) -> float32`, norm 1.0 ± 1e-5, `device="cpu"`, revision fail-fast, injectable encoder. Tests: `tests/test_embedding_encode.py` (stub encoder, no model download). Verification: scoped pytest green.
- [ ] 2.3 `backend/storage/keys.py` — `embedding_part(...)` builder + `_validate_lake_embeddings` shape pinner in `validate_key`; docstring builder index updated. Tests: `tests/test_embedding_keys.py`. Verification: scoped pytest green + `pytest tests/test_keys.py` regression.
- [ ] 2.4 `backend/pipeline/embedding/driver.py` — anti-join delta selection through the shared Athena runner (select only postings with no embedding row for the run's `configuration_hash`; zero-row delta → healthy no-op `completed` record, no store writes), variant fallback, loud empty accounting, pre-write abort → batch encode → per-batch Parquet part commits → one dataset manifest per run; `scripts/embed.py` CLI (config JSON, `--sql-file`, configurable output paths). Tests: `tests/test_embedding_pipeline_roundtrip.py` (fake Athena double + local Garage store). Verification: round-trip test green; full `pytest tests/ -v` green.
- [ ] 2.5 Delta-selection SQL (`select_unembedded` anti-join, adapting `backend/pipeline/selection.py`'s adapter pattern) + Glue/Athena registration of `lake/embeddings/` (`model`/`variant`/`version` partition columns, additive registration mirroring `scripts/provision_query_layer/`) + a store↔table row-count parity gate query. Verification: gate query passes; anti-join against a seeded part returns only un-embedded rows.
- [ ] 2.6 Mandatory re-run: spike logic reimplemented against the new modules, executed fresh; compare against the spine of the Layer 1 output. Verification: re-run reproduces spike behavior within documented (key/layout) parity; a second run re-selects zero rows (day-2 no-op).
- [ ] 2.7 Remove `scripts/embed_spike.py` in its own commit (idea trail preserved in git history).
- [ ] 2.8 **Checkpoint (user gate):** re-run evidence presented; go / change-order.

## 3. Layer 3 — Hardening (spec-mandated reliability)

- [ ] 3.1 Delta-selection idempotence end-to-end: seeded embeddings + new cleanings → anti-join selects only the new postings; rerun of a fully-selected-and-embedded delta selects zero rows and lands the no-op record. Tests: `tests/test_embedding_recovery.py`.
- [ ] 3.2 Buffered writer + flush triggers; reuse/extract the cleaned-artifacts writer rather than duplicating (decision recorded in design.md).
- [ ] 3.3 Flush checkpoints (`manifests/runs/.../flush=<seq>.json`) written strictly after their parts; orphan-part healing.
- [ ] 3.4 Interrupt replay semantics (only in-flight batch lost; idempotent rerun).
- [ ] 3.5 Fault-injection tests: crash between part and checkpoint; kill mid-run after N flushes; rerun reconciles to golden row multiset with zero re-encode and zero duplicates. Verification: `pytest tests/test_embedding_recovery.py -v` green (failing-then-passing per case); ruff + mypy clean.
- [ ] 3.6 **Checkpoint (user gate):** recovery evidence presented; go / change-order.

## 4. Layer 4 — Observability + provenance

- [ ] 4.1 Named log points: `select.zero_delta`, `flush.land`, `flush.stagger`, `qa.norm_violation`, `qa.dimension_mismatch`, `athena.scanned_bytes`. Verification: baseline-path rerun emits each evidence point.
- [ ] 4.2 Run records (`manifests/runs/run_id=<id>/run.json`): counters, latency aggregates, `data_scanned_in_bytes`, workgroup, status; partial failure retains counters and names the failing row.
- [ ] 4.3	Config objects: verbatim SQL at `configs/<sha256>.json`; manifests reference the hash.
- [ ] 4.4 QA gate: norms/dimension/nulls/row-count reconcile before SUCCEEDED; failure marks FAILED with named surface. Tests: `tests/test_embedding_accounting.py`. Verification: scoped pytest green; full `pytest tests/ -v` green.
- [ ] 4.5 `scripts/inspect_embeddings.py` (read-only): resolve a dataset id → manifest → configuration → model + revision + SQL + parameters.
- [ ] 4.6 Living documentation: README "Current Status", docs-sync cross-references, dev-log entry.
- [ ] 4.7 **Checkpoint (final):** end-to-end acceptance — synthetic-fixture dataset embedded via `scripts/embed.py`, manifests resolve full provenance, interrupted rerun reconciles, spike absent from tree.
