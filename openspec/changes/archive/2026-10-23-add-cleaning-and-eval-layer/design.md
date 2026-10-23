## Context

The lakehouse migration (docs/architecture/data.md) has landed only the raw branch: content-addressed payloads under `lake/raw/source=<source>/sha256=<hash>.json` in S3, across 6 sources. Re-baselined 2026-10-21 via `aws s3 ls` (the store has been updated since the original 7,812/6,224 figures below were written): dataroles 7,647; hiringcafe 325; greenhouse 766; lever 326; ashby 213; linkedin 39 — 9,316 objects total.

Cleaning and its evaluation exist as immutable experiment artifacts: `scripts/003_llm_extraction/` (extract → eval → compare) and `scripts/008_boilerplate_only_cleaning/` (hardened gateway client: 429/5xx retry, per-posting sleep pacing, single-writer flock, atomic cache writes) with proven behavior over 27 legacy postings (`raw_full_text` field), a 39-posting golden set (`data/golden_cleaned.json`, ported via 008's workpad flow), and flat caches. Known findings from exps 002–008: LLM cleaning beats regex boilerplate removal on separation gap; over-deletion strips organizational-role context (candidate for a separate artifact field); exp 008's 8-arm benchmark over the full golden selects `google/gemini-2.5-flash-lite` (Jaccard 0.894, 4.0% over-deletion, $0.00036/posting, ~zero-mean hallucinations) — gpt-5.4-nano is retained as the control configuration; `empty response` faults are posting-content-dependent and survive retries (exp 008, ling-3.0-flash); two concurrent cache writers corrupt flat caches, so single-writer + atomic replaces are mandatory conventions. Exp 008 recorded no latency fields; timestamp forensics over its cache later derived ~2 s/call for the chosen arm and a hidden 60 s median for the cheapest arm — a measurement gap this change closes for production (D9).

New raw payloads differ from legacy inputs in **three shapes, not one** (verified 2026-10-21 by fetching one live payload per source; sample keys/fields quoted in the resolved open question below):

1. **ScrapedJob 12-field shape** (ashby, dataroles, greenhouse, lever): `url, title, description_html, employer, location, country, employment_type, date_posted, salary_min, salary_max, salary_currency, source_id` — carries `description_html` (raw HTML) and a native `source_id` (`<ats>:<board>:<id>`; a dataroles posting carries its origin system's id verbatim, e.g. `workday:turo/wd12/turo_careers:R-102728`).
2. **hiringcafe MCP shape**: `url, title, description, employer, location, workplace_type, employment_types, date_posted, salary, source_id, extracted, extra` — plain-text `description`, **no `description_html`**; `source_id` is `hiringcafe:<native-id>`.
3. **linkedin markdown shape**: `content, format` only — **no `description_html`, no `source_id`, no `employer`, no `title`, no `date_posted`**; the posting text is the raw markdown `content` string.

Consequence for the input contract: the `llm-text-extraction` requirement's "normalize its `description_html`" applies to shape 1 only; shapes 2–3 enter the cleaning stage through their own text field (`description`, `content`) with missing identity fields (profiling lists them, per the `subset-sampling` malformed-payload requirement). Environment: Kilo Gateway API for LLM calls (`KILO_API_KEY`), pyarrow/pandas available, no new deps installable (AGENTS.md: no package installs). Storage environment note: the local file-mirror sink (`FileSink`, `--target files`) was removed — there is no `data/`-backed storage variant; every backend, including tests' in-memory double, speaks the object-store protocol, local development runs against Garage (itself being deprecated by the dev S3 bucket), and the sink is threaded through functions so tests inject a fake rather than choosing a storage flavor.

## Goals / Non-Goals

**Goals:**
- First regenerable derived layer: cleaned-text Parquet dataset with manifests, per §5–6 and §11 of data.md
- Reusable backend cleaning module (`backend/pipeline/cleaning/`) extracted from `scripts/003_llm_extraction/` — experiment scripts untouched
- Evidence-driven cleaning spec: profiling queries over the 6 sources through the cataloged layer (exp 009 evidence — first produced by a Python tool, re-derived and cross-checked via the D12 gate) before prompt/normalization tuning
- Deterministic subset selection for iterating on cleaning prompts without full-corpus cost
- Evaluation layer on the lake: golden references as authored datasets, quality metrics as result datasets, run lineage end to end (sample manifest → run → dataset → eval results)

**Non-Goals:**
- No embeddings, clustering, or visualizations in this change (they consume the cleaned layer; self-retrieval/separation-gap machinery stays in the existing evaluation-framework capability)
- No SQL engine executes transformation — the SQL layer (Glue Catalog + Athena, pulled forward from roadmap Phase 5 by D12) selects batches and serves interactive analysis; transformation stays code over the object store (D1 dependency direction unchanged)
- ~~No Glue/Athena/SQL engine~~ — replaced 2026-10-22: a Python profiling tool over the raw lake was implemented for exp 009, reviewed as a hand-rolled duplicate of the query layer, and rejected; the catalog with per-shape tables, the uncleaned anti-join, and the gate queries replace it (D12). Original stance part of the record: "Parquet rows are the interface; queries are ad hoc until an engine is warranted" — an engine is now warranted: the second aggregation tool (eval results, §4) is where ad hoc analysis accretes
- No semantic deduplication (data.md §4.1 stays deferred)
- No organizational-context field extraction yet (exp 003 finding) — record it as variant candidate in the report, not a new artifact type
- No scheduled/automated runs; the stage is a CLI-invoked batch transform

## Decisions

### D1 — Cleaning consumes lake rows, not a SQL layer
The cleaning stage iterates raw objects through `backend/storage/` (`ObjectStore`, `keys.py`) given a pinned sample manifest or a raw prefix. SQL/DuckDB was evaluated as the selection mechanism (see conversation history); it reduces to a sampler convenience — the transform itself is row-at-a-time with an LLM call, and its dependency direction must stay *raw → cleaning → artifacts*, with queries on top afterward. A DuckDB read view is out of scope; the sampler (D2) covers selection.

### D2 — Sampler: pandas/pass-through over the store, not an SQL dependency
Sampling reads payload metadata via the store, stratifies by source (quotas) and `description_html` length bands, seeded, and persists the chosen identity list. Rationale: deterministic and replayable without adding a query-engine dependency; recomputation via the same params reproduces the manifest, and the manifest itself is the pin. Alternative considered: DuckDB `read_json_auto` over the store's S3 API (httpfs) — cleaner interactive exploration but adds an uninstalled dep; revisit as a scratch tool, never a pipeline input. **Amended by D12 (2026-10-22)**: the sampler is no longer production selection — the `select_uncleaned` anti-join is; the sampler remains as the tuning-time pin (never a full-corpus substitute), and no query-engine dependency is added for it.

### D3 — Normalization is part of the cleaning stage, not a separate artifact layer
data.md's lineage sketches `raw → normalized → cleaned`. A standalone `normalized_posting` Parquet layer is deferred (YAGNI at this scale; it would double file count for a transform that is currently deterministic html→text). The stage normalizes `description_html` to plain text in-process, character by character, and the no-HTML invariant is tested. Revisit when a stage wants normalized-with-boilerplate text as a numbered input.

### D4 — Cache identity: (raw-text hash, configuration_hash) over artifact rows; no flat-file cache
The existing cache key (prompt hash, payload hash) survives in spirit as `configuration_hash` (input-identity) + `source_hash` (content identity), but persistence moves from `data/.llm_clean_cache.json` to lake rows: cache hit = artifact row already exists for that pair. Rationale: one storage mechanism (the dataset is the cache), no dual source of truth, and config change re-cleans incidentally re-creating old versions preserved per data.md §2.4. Legacy `.llm_clean_cache.json` is kept on disk (immutable artifact) but never read.

### D5 — Artifact row schema per data.md §5–6, frozen at dataset-definition time
Rows: `source_hash`, `variant`, `version`, `cleaned_text`, `configuration_hash`, `pipeline_version`, `created_at` (+ `employer`, `title`, `source` denormalized for eyeballing? — no; keep provenance joins via `source_hash`, viewer convenience is what the eval/query layer is for). `posting_id` was dropped from the row schema (see D11-rejected): `source_hash` is the posting's join handle. Version roots at `v1`; prompt iteration without semantic change stays within a version + new `configuration_hash` rows (versions don't overwrite, rows append).

### D6 — Failure policy: fail fast per posting, complete run accounting
Empty LLM response → RuntimeError for that posting (carried from the existing spec), run ends, counters (cleaned, cache-hit, failed) persisted in `manifests/runs/`. Alternative: skip-and-continue. Fail fast is kept because silent gaps in a first derived dataset are the classic un-governed-accumulation trap (report 009) — observable failure, cheap rerun (cache hits).

### D7 — Eval datasets/results split per data.md §8
Golden references land once, author-pinned (`eval/datasets/type=cleaning_golden/variant=author/version=v1/`). Results are regenerated per evaluation configuration (`eval/results/type=cleaning_quality/...`). Metric definitions (Jaccard, 7-pattern boilerplate, hallucination = word-set difference vs raw, over-deletion = golden−llm ratio with 5% flag) carry over verbatim from exp 003 for comparability across the legacy → lake migration. Golden rows migrate from `data/golden_cleaned.json` (39 entries after exp 008's workpad port; the first 5 are legacy); legacy JSON stays untouched as artifact history.

### D8 — Bounded worker pool, single process
The runner executes postings through a same-process thread pool (`--workers`, default 8; calls are IO-bound httpx requests), each worker pacing with jitter; the flock single-runner lock, single-writer cache discipline, and fail-fast policy (D6) survive unchanged — parallelism happens between API calls, never between writers. Alternatives rejected: process fan-out (cache write contention, defeats flock semantics) and provider batch tiers (−50% saves ~$1.40 at lake scale but adds ~24h turnaround; irrelevant when wall-clock is under an hour). Measured basis: A3 ~2 s/call ⇒ the 7,812-posting full lake is ~6.5h serial vs ~35–50 min at 8 workers; sample-manifest tuning runs scale down proportionally.

### D9 — Per-call latency recorded; aggregates in the run record
Every LLM call records `duration_s`. Run manifests under `manifests/runs/` carry call aggregates (n_calls, median, p90, max) covering LLM-cleaned postings only — cache-hit postings contribute no call duration. Artifact rows stay exactly per data.md §5–6 schema (no runtime column); latency is run telemetry, not dataset content. Rationale: exp 008 recorded no latency fields — arm latency was recoverable only via cache-timestamp forensics, which is how the hidden 60 s median on its cheapest-per-dollar arm surfaced after the fact. Future model/prompt swaps must be judged on workload-real latency, so the runner measures it by default (task 3.7 verification consumes the aggregates).

### D10 — Writer commit protocol: buffered flush watermark
The single writer thread buffers completed batch outcomes in memory (bounded ≤2 MiB) and commits on any-of trigger — 128 rows, 60 s dwell, input queue drained, stop event → **one Parquet part per flush** (`part-<run_id>-<seq>`) followed by **one flush checkpoint** `manifests/runs/run_id=<run_id>/flush=<seq>.json` (batch_range, source_hashes, row_count, counters, latency aggregates, usage, cost, part_key). Commit order is part → checkpoint: per-key PutObject atomicity plus S3 strong read-after-write carries the guarantee; no cross-key transaction is assumed (the documented S3 consistency model) — the checkpoint-document shape shown robust to mid-write kill in the 2026 commit-durability study (arXiv 2604.20081). The aggregated run record (D6 counters, D9 latency) is merged from flush checkpoints at run end.

Recovery semantics: an orphan part (checkpoint missing — crash between the two writes) is a journal hole the next run's aggregator back-fills from part contents (`recovered: true`); an interrupt triggers a final flush of the buffer, so the loss window is exactly the in-flight LLM calls; replay of unflushed postings is idempotent under the artifact-row cache (at-least-once redelivery — the set difference absorbs duplicates). Per-batch provenance lives as `batch_range` inside the flush checkpoint: a batch is a scheduling unit, the flush is the durability unit.

Part-count trajectory: ≈61 parts per full-lake configuration vs 488 for per-batch atomic parts — the flush watermark is the Kafka `batch.size`/`linger.ms` mechanism, chosen because "frequent micro-batch writes" is the named first cause of small-file degradation in AWS storage guidance (128 MB target file sizes; aspirationally distant at a ~10 MB corpus — parts stay below target by scale, revisited as compaction if tuning-run count grows the part count). Alternatives rejected: per-batch atomic parts (finest recovery granularity, but 8× the object accumulation across every prompt iteration for a durability premium priced in seconds of cheap rerun); whole-run aggregation (fewest objects, but a crash destroys all processed-not-yet-flushed work — precisely the client-buffering loss Kafka's design docs document — and offers zero recoverability before completion).

### D11 (rejected) — posting_id dropped; content hash is the posting identity

Task 1.2 resolved that no `posting_id` exists in any raw payload. A mint-once UUIDv7 identity carried by a posting-id registry (`lake/manifests/postings/`) was implemented and then removed, because its justifications did not survive scrutiny at this scale:

- **Ingestion-order UUIDv7s are moot**: object keys are content hashes (write-dedup was chosen over key-ordered objects, §2.1), and ingestion chronology is reconstructible from manifests alone — `observed_at` per row, `ingested_on` day partitions, encounter order within a run.
- **As a hash synonym, the registry was pure overhead**: an append-only mapping keyed on `source_hash` adds no semantics over the hash itself — its fast lookup and mint-once invariant decorate what the join key already says.
- **The one live argument — entity identity ("this job" vs "these bytes") — is deferred, not dead**: payload bytes are a scraper's rendering (any field change re-hashes the posting), eval golden rows and artifact joins dangle across scraper drift, and §4.1 semantic dedup is a posting-level concept. At 9.3k objects with reproducible derived layers, paying that now is not warranted; when the need materializes, identity should be minted *where it is used* (re-keying resolution on the ATS-native `source_id`, hash fallback) — the registry storage shape was deliberately agnostic to that pivot, so the pattern ports over cheaply.

Net: `source_hash` joins raw → artifacts → eval rows; no posting-id minting, registry branch, or backfill exists, and ingestion manifests stay schema v1.

### D12 — Query layer owns selection and profiling; code owns transformation

The original §2 (Python profiling tool; sampler as production selection) was implemented and then rejected as a **duplicated query mechanism**: per-source counts, length quantiles, and tag/boilerplate aggregations are one `GROUP BY` in any SQL engine, and a second ad hoc aggregating tool was exactly the accretion pattern to avoid (eval results, §4, would have become the next one). The Glue Data Catalog + Athena pull (roadmap Phase 5, per-shape tables — no crawler over the three-shape mush) now owns:

- **Profiling** (exp 009 evidence, re-derived through a second engine as the §2.3 gate — the gate *is* the independent cross-check of the store-count baseline).
- **Selection** for production cleaning runs: the `select_uncleaned` anti-join — raw keys with no artifact row for `(source_hash, configuration_hash)`, minus the failure ledger (D6 failures recorded in run manifests; a content-dependent `empty response` fault must not abort every future run). The anti-join and the D4 cache are the same mechanism: artifacts double as the done-index, and "cleaned postings fall out of selection" replaces bookkeeping entirely.

Unchanged: the transform stays a Python runner over the object store (D1 — LLM call loops are not SQL); the reader (`backend/storage/raw.py`) is its I/O path and survives; selected batches render to the *same* canonical manifest format the CLI consumes, so the runner never learns where its input came from. **Sampler demoted to tuning-time utility**: D2's cost-control argument applies per prompt iteration (a small pinned sample in minutes vs a full-corpus sweep per configuration at D5's append semantics); production selection is the anti-join. Warnings noted: configuration changes re-select the whole lake (accepted, priced in D4/D5); scheduling stays out of scope (runs are CLI-invoked; EventBridge is a later phase).

## Risks / Trade-offs

- [HTML normalization loss — tags may carry meaning (list bullets, headings)] → no-HTML invariant test plus exp 009 inventory (tag families per source, SQL-reproducible) precedes prompt tuning; transformations remain recoverable because raw is immutable
- [Over-deletion modulates quality; prompt tuning loops spend LLM calls] → seeded sample manifests keep tuning runs small and pinned; cache makes reruns of unchanged configs free
- [`configuration_hash` churn — trivial prompt edits split the cache] → accepted; versions preserve lineage and hash churn is observable in manifests rather than silent (revisit with a smarter canonicalization if churn bites)
- [Dataroles dominates the corpus 80%] → sampler quotas enforce per-source floors so tuning never fits to one source's formatting; profiling report makes the imbalance explicit
- [Garage listing latency over 7.8k objects] → prefix listing is one call per source partition; sample manifests and manifests/ keep run input enumerable; Parquet part-writes avoid per-posting objects going forward
- [Concurrent workers amplify gateway throttling or `empty response` faults (exp 008: content-dependent, retry-surviving)] → worker count CLI-bounded, jittered pacing, 429 backoff per worker, fail-fast run accounting (D6) records which posting faulted; reruns are cache-hit-cheap
- [Extra flush knobs (rows/seconds) surprise operators with unexpected part counts on small runs] → any-of triggers plus flush-on-drain bound the tiny-run case; defaults derived from measured row size; loss window documented in every run record
- [No dust-off of pyarrow table properties (e.g., dictionary encoding)] → behavior tested via round-trip read; storage tuning postponed until scale signals exist

## Migration Plan

1. Key builders for `artifacts/`, `eval/`, `manifests/` branches land first (pattern of the raw-branch migration: builders + tests, §2 of data.md).
2. Query layer lands (D12); gate queries re-derive exp 009's evidence; the uncleaned anti-join becomes selection; sampled manifests stay tuning-only.
3. Backend cleaning module formalized; run over pinned manifest → `v1` dataset manifests.
4. Golden set authored into `eval/datasets/`; eval module compares and writes results.
5. Rollback: every output lives only in new lake branches; deleting the new prefixes and manifests removes all effect. No legacy object is rewritten; the immutable scripts remain the fallback record (scientific integrity rule).

## Open Questions

- ~~Do `posting_id` values exist in current raw metadata (minted at lake ingestion per §2.2), or must the cleaning run mint them?~~ **Resolved 2026-10-21 (task 1.2): no `posting_id` field exists in any live raw payload — and none is minted or needed** (see D11-rejected): `source_hash` is the identity. Evidence: one payload per source fetched from the live store (`s3://semantic-career-map-data-dev-760852635093-us-east-2-an/lake/raw/source=<s>/`); `posting_id` is `ABSENT` in all six. Sample keys and fields:
  - ashby: `lake/raw/source=ashby/sha256=00e9dd37…5b399992.json` — fields `['country', 'date_posted', 'description_html', 'employer', 'employment_type', 'location', 'salary_currency', 'salary_max', 'salary_min', 'source_id', 'title', 'url']`; `source_id: 'ashby:supabase:142135ae-a15f-4ec7-90bb-cb4e7968bc09'`; `posting_id` ABSENT.
  - dataroles: `lake/raw/source=dataroles/sha256=0015ee3a…5d7da579.json` — same 12 fields; `source_id: 'workday:turo/wd12/turo_careers:R-102728'`; `posting_id` ABSENT.
  - greenhouse: `lake/raw/source=greenhouse/sha256=0002fa67…e48830b.json` — same 12 fields; `source_id: 'greenhouse:anthropic:5369181008'`; `posting_id` ABSENT.
  - lever: `lake/raw/source=lever/sha256=002a0075…945c70.json` — same 12 fields; `source_id: 'lever:palantir:3531977a-9b2f-40a0-a486-4eeb38e75fc1'`; `posting_id` ABSENT.
  - hiringcafe: `lake/raw/source=hiringcafe/sha256=0076c833…39a663ade.json` — fields `['date_posted', 'description', 'employer', 'employment_types', 'extra', 'extracted', 'location', 'salary', 'source_id', 'title', 'url', 'workplace_type']`; `source_id: 'hiringcafe:xl47beuoe8znuc9x'`; `posting_id` ABSENT; no `description_html` (plain-text `description`).
  - linkedin: `lake/raw/source=linkedin/sha256=056c4595…33efb78f9.json` — fields `['content', 'format']` only; `posting_id` ABSENT; no `source_id`/`employer`/`title`/`date_posted`/`description_html`.
  Consequence: the posting's identity in rows and manifests is its `source_hash` alone — nothing is minted, and raw payload objects stay byte-immutable (a posting-id scheme was considered for this gap and rejected; see D11-rejected for the reasoning and the §4.1 re-entry point).
- Should the profiling report live in `docs/experiments/` (exp 007 numbering since taken; landed as 009) or as a rerunnable tool in `backend/`? Superseded by D12 (2026-10-22): findings live in the exp report; the mechanism is the SQL layer. A Python profiling tool was built first, reviewed as a hand-rolled duplicate of the query layer, and removed.
