## MODIFIED Requirements

### Requirement: Embed text into L2-normalized vectors
The system SHALL encode posting text into dense vector representations suitable for semantic similarity search, producing one float32 row vector per input text with an L2 norm of 1.0 within a tolerance of 1e-5, loaded with `device="cpu"` and with the model's resolved revision asserted equal to the configuration's pinned `model_revision`.

#### Scenario: Dense embedding output
- **WHEN** a batch of N texts is embedded
- **THEN** the output SHALL be a float32 array with one row vector per input text, in input order

#### Scenario: L2 normalization
- **WHEN** embeddings are produced
- **THEN** every row vector SHALL have an L2 norm of 1.0 within a tolerance of 1e-5

#### Scenario: CPU-only inference
- **WHEN** the model is loaded
- **THEN** it SHALL be loaded with `device="cpu"`

#### Scenario: Revision drift fails fast
- **WHEN** the resolved model revision at load time does not equal the pinned `model_revision` in the configuration
- **THEN** the pipeline SHALL fail before any embedding or store write occurs

## ADDED Requirements

### Requirement: Configuration object with stable configuration_hash
The system SHALL accept an embedding configuration object (embedding model name, `model_revision`, batch size, normalization state, text variant field, prompt name, input SQL text, output version token) and SHALL derive a 64-char hex `configuration_hash` from a canonicalized serialization with fixed field order and no timestamps or runtime-affine values. The verbatim input SQL SHALL be stored as a content-addressed `configs/<sha256>.json` object reachable from the hash.

#### Scenario: Cross-process hash stability
- **WHEN** the same configuration values are constructed in two different processes
- **THEN** both `configuration_hash` derivations SHALL yield the identical 64-char hex digest

#### Scenario: Single-field change re-hashes
- **WHEN** any one configuration value changes
- **THEN** the `configuration_hash` SHALL change

#### Scenario: SQL text resolves from the hash
- **WHEN** a `configuration_hash` is recorded in a manifest or row
- **THEN** the exact SQL text and remaining configuration values SHALL be resolvable from the stored `configs/<sha256>.json` object

### Requirement: Lakehouse key builders pin the embeddings branch
The key layer SHALL provide builders that emit `lake/embeddings/model=<model>/variant=<variant>/version=<vN>/part-<run-id>-<seq>.parquet` keys, and `validate_key` SHALL pin the branch shape (partition name and order, `vN` version token, run-minted part leaf) so no code path can mint keys outside the layout.

#### Scenario: Built keys validate
- **WHEN** any key produced by the embeddings builder is checked with `validate_key`
- **THEN** it SHALL pass unchanged

#### Scenario: Hand-minted malformed key rejected
- **WHEN** a `lake/embeddings/...` key with a wrong partition order, missing `vN`, or non-run-minted leaf is validated
- **THEN** `validate_key` SHALL raise `ValueError`

### Requirement: Retrieve cleaned rows by Athena SQL
The system SHALL select `(source_hash, text)` rows via the shared Athena runner (submit → poll → paginate) using the configured SQL, SHALL apply the variant field with fallback for missing or empty fields, SHALL count postings with no usable text as empty rather than silently dropping them, and SHALL abort before any store write when the query ends in a non-SUCCEEDED state.

#### Scenario: Variant fallback applied per row
- **WHEN** a row's configured variant field is missing or empty
- **THEN** the fallback variant field's text SHALL be used, with no usable text in either field counted as an empty input

#### Scenario: Query failure aborts pre-write
- **WHEN** the Athena query reaches a non-SUCCEEDED terminal state
- **THEN** no embedding or Parquet part SHALL be written for that run

### Requirement: Commit embedding parts in batch order
The system SHALL encode selected rows in configured-size batch windows with order preservation, and SHALL commit one Parquet part per completed batch under the embeddings version prefix, with rows carrying `source_hash`, `embedding`, `model`, `model_version`, `input_artifact`, `input_version`, `embedding_dimension`, `normalization`, `created_at`, `configuration_hash`, and `run_id`. A produced row's `(source_hash, configuration_hash, input_version)` triple SHALL be exactly the anti-join's cache identity — a single join-key scheme shared between selection and written rows, so the cache correctness is verifiable by one query.

#### Scenario: Row count reconciles with selection
- **WHEN** a run writes its embedding parts
- **THEN** the total row count SHALL equal the selected usable rows, and every key SHALL pass `validate_key`

#### Scenario: Row-level provenance present
- **WHEN** an embedding row is read back from the store
- **THEN** it SHALL carry its `configuration_hash` and `run_id` in the data itself

### Requirement: Select un-embedded postings by anti-join
The system SHALL select its input as an Athena anti-join — cleaned artifact rows for the pinned input version joined to produced embedding rows on `(source_hash, configuration_hash)` — yielding exactly the postings with no embedding row under the run's configuration, mirroring the cleaning tier's `select_uncleaned` semantics. The run's model/variant/version generation scope comes from the registered table's pinned location, not the query.

#### Scenario: Delta-only selection
- **WHEN** a run selects its input
- **THEN** no posting that already has an embedding row under the run's `configuration_hash` and pinned input version SHALL be selected, and no such posting SHALL be encoded or written again by that run

#### Scenario: Zero-row delta is a healthy no-op
- **WHEN** the anti-join returns zero rows
- **THEN** the run SHALL still land a `completed` run record with `embedded = 0` (the day-2 idempotence artifact), with no store writes

#### Scenario: Configuration change re-embeds under a new version
- **WHEN** the configuration differs in any hashed field
- **THEN** the anti-join SHALL resolve the new `configuration_hash`, so every posting is selected and re-embedded under a new `version=` prefix, with prior version prefixes unchanged

### Requirement: Embeddings table registered for query-side cache visibility
The embeddings branch SHALL be registered as a Glue/Athena table following the additive, idempotent registration pattern of `scripts/provision_query_layer`, with its location pinned to one model/variant/version generation (the artifacts table's discipline — partition values live in the location, not the payload); a generation bump updates the registration. The anti-join joins payload columns `source_hash`/`configuration_hash` — without the registration it cannot see produced vectors and silently re-encodes them.

#### Scenario: Registration scopes the run's generation
- **WHEN** the anti-join runs
- **THEN** the embeddings side SHALL read only parts under the registered generation's location, joined on the payload cache-identity columns

#### Scenario: Registration verified like the artifacts gate
- **WHEN** the table is provisioned
- **THEN** a gate query SHALL verify row-count parity between the store's parts under the registered prefix and the table's count

### Requirement: Write dataset manifests for embedding versions
The system SHALL write one dataset manifest per embedding run under `manifests/datasets/` recording `dataset_id`, the input reference (input artifact version and Athena `QueryExecutionId`), `configuration_hash`, `pipeline_version`, `model_revision`, `record_count`, and `created_at`.

#### Scenario: Manifest lineage resolves
- **WHEN** a dataset manifest for an embedding version is read
- **THEN** it SHALL resolve the run that produced it, the Athena query execution that selected its rows, and the configuration hash that governs its parameters


### Requirement: Flush commits carry checkpoints written after their parts
The writer SHALL buffer completed batch results and commit one Parquet part per flush (triggered by row threshold, byte threshold, dwell time, drained queue, or stop), each part followed by one flush checkpoint manifest under `manifests/runs/run_id=<id>/flush=<seq>.json` covering exactly that part — written only after the part exists.

#### Scenario: Commit ordering
- **WHEN** a flush trigger fires
- **THEN** the part SHALL exist in the store before its checkpoint manifest is written

#### Scenario: Orphan part healing
- **WHEN** a part exists without its checkpoint manifest
- **THEN** the next run SHALL back-fill the checkpoint from the part's contents and treat those rows as durable

### Requirement: Interrupts lose only in-flight work
The system SHALL guarantee that interrupting a run leaves all committed flushes durable, loses only the unflushed batch, and permits idempotent replay.

#### Scenario: Interrupted rerun reconciles
- **WHEN** a run is killed after N flushes and rerun
- **THEN** committed rows SHALL remain, no rows re-encode, no duplicates SHALL appear, and the final row multiset SHALL equal the uninterrupted run's golden multiset

### Requirement: Record run-level accounting
The system SHALL record each embedding run under `manifests/runs/run_id=<id>/run.json` with `run_id`, `pipeline_version`, `started_at`/`completed_at`, the input reference, output dataset id, `configuration_hash`, counters (`selected`, `embedded`, `empty`, `failed`), encode latency aggregates (rows/s, p50/p90 per-batch ms), the Athena `data_scanned_in_bytes`, workgroup, and final status. Partial failure SHALL retain counters accumulated before the failure and identify the failing row.

#### Scenario: Partial failure accounting
- **WHEN** a row fails mid-run
- **THEN** the run record SHALL retain the counters and status SHALL name the failing `source_hash`

#### Scenario: Cost and throughput captured
- **WHEN** a run completes
- **THEN** the run record SHALL carry the encode latency aggregates and the Athena scanned-bytes so runs are comparable on workload-real cost/latency

### Requirement: QA gate blocks success on vector regressions
Before a run may finalize SUCCEEDED, the system SHALL verify: every written vector has a norm within 1e-5 of 1.0, the embedding dimension is constant across the version, no null vectors exist, and the row count reconciles with the selection. A violation SHALL mark the run FAILED with the violating surface named.

#### Scenario: Norm violation fails the run
- **WHEN** any written vector's norm is outside 1e-5 of 1.0
- **THEN** the run status SHALL be FAILED with `qa.norm_violation` named
