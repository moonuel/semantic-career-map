# subset-sampling

## Purpose

TBD - Profile the raw lake for cleaning evidence and select stratified pinned samples plus uncleaned anti-join posting sets for cleaning runs.

## Requirements

### Requirement: Profile the raw lake before cleaning
The system SHALL profile the raw layer by querying the cataloged raw layer through the SQL engine (Glue Data Catalog + Athena, design D12): per-source object counts, per-source text-length distributions over each shape's own text field, presence of HTML tag families via regex extraction, boilerplate-marker incidence, and malformed-payload detection (unparseable JSON via null-tolerant parsing, missing text field, or empty), to ground cleaning decisions in evidence.

#### Scenario: Per-source statistics summary
- **WHEN** profiling runs over the cataloged raw tables
- **THEN** queries SHALL return per-source counts, length quantiles (median, p90, max), and the most frequent HTML tag families for each source
- **AND** per-source counts SHALL be reported against a re-baselined lower bound, failing when a source drops below it (append-only lake: an ingestion may only raise a count; a drop means data loss)
- **AND** median and p90 SHALL remain within a fixed relative tolerance (10%) of the re-baselined values, a guard against a distribution shift rather than against ordinary sample growth, while max SHALL be a lower bound (a newer, longer posting legitimately raises an observed maximum)

#### Scenario: Malformed payload reporting
- **WHEN** a raw payload is unparseable, missing its text field, or empty
- **THEN** the profile queries SHALL list those objects by key and source rather than silently dropping them

### Requirement: Select stratified, pinned samples for cleaning tuning runs
The system SHALL select prompt-tuning input samples as a stratified subset of the raw layer — stratified by source and by text-length band — with a fixed seed, persisted as a sample manifest. Production cleaning-run selection is the uncleaned anti-join (see "Select uncleaned postings for a cleaning run"); the sampler is the tuning-time pin that keeps prompt-iteration loops small.

#### Scenario: Stratification by source
- **WHEN** a tuning sample is drawn with N posts per source requested
- **THEN** the sample SHALL contain approximately N postings from each source, drawing across distinct employers where available

#### Scenario: Sample manifest reproducibility
- **WHEN** the same sampling parameters (source quotas, length bands, seed) are re-run
- **THEN** the system SHALL produce the identical sample manifest, referencing postings by `source_hash` and `source_id`

#### Scenario: Query-layer candidates
- **WHEN** sample candidates are enumerated through the SQL layer's result set
- **THEN** the pinned-sample contract (stratification, seed, byte-identical re-run) SHALL be unchanged — the query layer replaces only the enumeration cost

#### Scenario: Sample manifest as run input
- **WHEN** a cleaning run is configured with a sample manifest
- **THEN** the run SHALL consume exactly the postings referenced by that manifest and record the manifest as `input_dataset` lineage

### Requirement: Select uncleaned postings for a cleaning run
The system SHALL select cleaning-run inputs through the SQL layer as the anti-join of the raw layer against the cleaned-artifact layer: raw keys with no artifact row for the run's `(source_hash, configuration_hash)` pair, minus postings recorded as failed in prior run manifests (the failure ledger), rendered into the canonical run-manifest format.

#### Scenario: Cleaned postings fall out of selection
- **WHEN** an artifact row exists for a posting under the run's configuration
- **THEN** that posting SHALL NOT be selected again for the same `configuration_hash`

#### Scenario: Failed postings are excluded until cleaned
- **WHEN** a prior run recorded a posting as failed
- **THEN** later selections SHALL omit it — content-dependent faults must not abort every future run

#### Scenario: Configuration change re-selects everything
- **WHEN** the run's `configuration_hash` differs from any existing artifact rows
- **THEN** the selection SHALL return all raw keys (the artifact join is empty)

#### Scenario: Selection is deterministic and pinned
- **WHEN** the selection query runs twice over an unchanged raw and artifact layer
- **THEN** the rendered manifest SHALL be byte-identical, and the run SHALL consume exactly the postings referenced by it, recording the selection as `input_dataset` lineage
