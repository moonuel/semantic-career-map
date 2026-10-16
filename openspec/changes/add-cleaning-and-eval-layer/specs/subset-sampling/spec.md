## ADDED Requirements

### Requirement: Profile the raw lake before cleaning
The system SHALL produce a profiling report over the raw layer: per-source object counts, per-source `description_html` length distributions, presence of HTML tag families (sections, lists, links, scripts), and boilerplate marker incidence, to ground cleaning decisions in evidence.

#### Scenario: Per-source statistics summary
- **WHEN** profiling runs over the raw prefix
- **THEN** the report SHALL summarize counts, length quantiles (median, p90, max), and the most frequent HTML tag families for each source

#### Scenario: Malformed payload reporting
- **WHEN** a raw payload is unparseable, missing `description_html`, or empty
- **THEN** the report SHALL list those objects by key and source rather than silently skipping them

### Requirement: Select stratified, pinned samples for cleaning runs
The system SHALL select cleaning-run input samples as a stratified subset of the raw layer — stratified by source and by text-length band — with a fixed seed, and SHALL persist the selection as a sample manifest.

#### Scenario: Stratification by source
- **WHEN** a sample is drawn with N posts per source requested
- **THEN** the sample SHALL contain approximately N postings from each source, drawing across distinct employers where available

#### Scenario: Sample manifest reproducibility
- **WHEN** the same sampling parameters (source quotas, length bands, seed) are re-run
- **THEN** the system SHALL produce the identical sample manifest, referencing postings by `source_hash` and `source_id`

#### Scenario: Sample manifest as run input
- **WHEN** a cleaning run is configured with a sample manifest
- **THEN** the run SHALL consume exactly the postings referenced by that manifest and record the manifest as `input_dataset` lineage
