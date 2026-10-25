# cleaned-artifacts

## Purpose

TBD - Produce, cache, and account for cleaned-text artifact rows written to the lake, with dataset manifests and run-level accounting.

## Requirements

### Requirement: Clean lake raw payloads into cleaned-text artifact rows
The system SHALL accept raw posting payloads from the object store (`lake/raw/source=<source>/sha256=<hash>.json`), extract job-signal text from the payload, and produce one cleaned-text artifact row per posting under `artifacts/type=cleaned_text/variant=<method>/version=<vN>/` in Parquet format.

#### Scenario: Cleaning run over a sample manifest
- **WHEN** a cleaning run is given a pinned sample manifest of posting identities
- **THEN** each posting in the manifest produces exactly one artifact row with `source_hash`, `variant`, `version`, `cleaned_text`, `configuration_hash`, `pipeline_version`, and `created_at` populated

#### Scenario: No HTML markup in cleaned output
- **WHEN** the input payload's text carries HTML markup
- **THEN** the cleaned artifact row's text SHALL contain no HTML tags, and the run SHALL report normalization as part of the stage rather than leaking tags or stripping content by tag alone

#### Scenario: Empty LLM response handling
- **WHEN** the LLM returns an empty or whitespace-only response for a posting
- **THEN** the run SHALL raise a RuntimeError identifying the failing posting and record no artifact row for it

### Requirement: Write dataset manifests for cleaned-artifact versions
The system SHALL write one dataset manifest per cleaning run under `manifests/datasets/` recording dataset lineage.

#### Scenario: Manifest contents
- **WHEN** a cleaning run completes
- **THEN** the manifest SHALL record `dataset_id`, `input_dataset` (the input sample manifest or raw prefix), `model`, `configuration_hash`, `pipeline_version` (git sha), `record_count`, and `created_at`

### Requirement: Cache cleaned results by configuration and content
The system SHALL make repeated cleaning runs idempotent: a posting is sent to the LLM only when no artifact row already exists for its exact raw text and the run's `configuration_hash`.

#### Scenario: Cache hit on identical input and configuration
- **WHEN** a posting's raw text hash and the run's `configuration_hash` match an existing artifact row
- **THEN** the system SHALL resolve the posting from that row without an LLM call

#### Scenario: Configuration change re-cleans
- **WHEN** the system prompt, user template, or model changes between runs
- **THEN** the run SHALL resolve to a different `configuration_hash`, no cached row matches, and every posting is re-cleaned — previously generated versions remain preserved under their own version prefix

### Requirement: Flush commits carry their own checkpoint manifests
The system SHALL buffer completed posting results in the writer thread and commit them as one Parquet part per flush, where a flush is triggered by any of: a row threshold, a byte threshold, a dwell-time threshold, input-queue drainage, or a stop event. Each flushed part SHALL be followed by one flush checkpoint manifest under `manifests/runs/` covering exactly the rows in that part, written after the part exists.

#### Scenario: Flush trigger and commit ordering
- **WHEN** the writer buffer reaches any flush trigger (rows, bytes, dwell, drained queue, or stop)
- **THEN** the run SHALL append exactly one Parquet part to the dataset prefix and SHALL record one flush checkpoint manifest (batch range, source hashes, row count, counters, latency aggregates, part key) referencing that part, before the batches are acknowledged

#### Scenario: Orphan part healing
- **WHEN** a part exists without its checkpoint manifest (interruption between the two writes)
- **THEN** the next run's aggregation step SHALL back-fill the checkpoint from the part's contents and treat those rows as durable

#### Scenario: Interrupt loses only in-flight work
- **WHEN** a run is interrupted after some flushes are committed
- **THEN** completed flushes SHALL remain durable and their rows SHALL resolve as cache hits on rerun; only unflushed postings are re-cleaned, and that replay SHALL be idempotent under the artifact-row cache

### Requirement: Record run-level accounting for pipeline runs
The system SHALL record each cleaning run in `manifests/runs/` with `run_id`, `pipeline_version`, `started_at`, `completed_at`, the input sample reference, the output dataset id, `configuration_hash`, per-run counters (cleaned, cache-hit, failed), LLM-call latency aggregates (n_calls, median, p90, max `duration_s`), and final status.

#### Scenario: Partial failure accounting
- **WHEN** a posting fails mid-run
- **THEN** the run record SHALL retain the counters of cleaned and cache-hit postings before the failure and the run status SHALL reflect the failure with the failing posting identified

#### Scenario: Concurrency and latency recorded
- **WHEN** a run completes with more than one LLM call
- **THEN** the run record SHALL include its configured worker count and the call-latency aggregates, so model/prompt configurations are comparable on workload-real latency, not fidelity alone
