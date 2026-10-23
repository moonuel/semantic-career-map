## Why

The cleanest files today exist only as legacy experiment outputs (`data/jobs.json` fields, `data/.llm_clean_cache.json`) produced by the immutable `scripts/003_llm_extraction/` scripts over 27 legacy postings — outside the lakehouse. The raw layer now holds 7,812 lake postings across 6 sources whose `description_html` payloads have never been cleaned, and the artifacts/ and eval/ branches from `docs/architecture/data.md` don't exist yet. The next pipeline step (text cleaning) should be the change that populates both, reproducibly.

## What Changes

- Productionize LLM text cleaning as a composable backend stage: extract reusable logic from `scripts/003_llm_extraction/` and `scripts/008_boilerplate_only_cleaning/` (scripts stay untouched) into `backend/pipeline/cleaning/`, reading raw payloads through `backend/storage/` instead of `data/jobs.json`. Runtime is a bounded worker pool, not serial pacing, and per-call latency is recorded — exp 008 shipped with no latency measurement at all.
- Add the first lakehouse derived layer: cleaned-text rows written to `artifacts/type=cleaned_text/variant=<method>/version=v1/` as Parquet with dataset manifests under `manifests/datasets/`.
- Add a profiling + subset-sampling script over the raw lake: per-source HTML/noise inventory (exp 007 evidence) and a pinned, source-stratified sample manifest that selects which postings a cleaning run consumes.
- Move extraction-quality evaluation onto the lake: golden references migrate from `data/golden_cleaned.json` to the `eval/` branch (datasets there, results Parquet under `eval/results/`), computing Jaccard, boilerplate, hallucination, and over-deletion metrics from experiments 002/003.
- Config-aware caching: replace the flat `data/.llm_clean_cache.json` with cache identity keyed by `configuration_hash` + payload content hash (aligned with `data.md` §3).

## Capabilities

### New Capabilities
- `cleaned-artifacts`: The cleaned-text pipeline stage — raw payload in, Parquet artifact rows + dataset manifest out, configuration-hash caching, run-level accounting.
- `subset-sampling`: Profiling of the raw lake (per-source stats, noise inventory) and deterministic, stratified, manifest-pinned sample selection for cleaning runs.

### Modified Capabilities
- `llm-text-extraction`: Input contract changes from legacy `raw_full_text` field to lake raw payloads (`description_html`); output and cache locations change to the lake-backed `cleaned-artifacts` store; model/prompt configuration identified by `configuration_hash`.
- `evaluation-framework`: Extraction-quality requirements re-target from legacy JSON files to the lake — golden references read from `eval/datasets/`, cleaned comparisons read from `cleaned-artifacts` rows, results written to `eval/results/` Parquet.

## Impact

- Code: new `backend/pipeline/cleaning/` (module + CLI), `backend/pipeline/` eval module, `backend/storage/keys.py` gains `artifacts/` and `eval/` branch builders, manifest writers.
- Data: first `artifacts/` and `eval/results/` Parquet datasets in Garage; golden set copies to `eval/datasets/type=cleaning_golden/` (legacy JSON kept in place).
- Deps: none new — pyarrow, pandas, boto3 already required. LLM calls use the Kilo Gateway (`KILO_API_KEY`), default model `google/gemini-2.5-flash-lite` per exp 008 arm A3 — best fidelity on the 39-posting golden (Jaccard 0.894, 4.0% over-deletion, ~$0.00036/posting) and fastest arm (~2 s/call derived from cache timestamps). `gpt-5.4-nano` stays available as a control configuration, not the production default.
- Docs: README "Current Status", `docs/architecture/data.md` migration table, new exp 007 report after first profiled run.
