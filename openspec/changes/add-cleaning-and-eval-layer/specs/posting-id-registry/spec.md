## ADDED Requirements

### Requirement: Mint posting ids at ingestion and record them in the registry
The system SHALL assign every raw posting a `posting_id` — a canonical UUIDv7 minted by `backend.storage.ids.new_id()` — at ingestion time, and SHALL record the assignment as a registry row `(source_hash, posting_id, source_id, minted_at, minting_run_id)` under `lake/manifests/postings/` (one Parquet part per minting run, grouped under `run_id=<run-id>/`). Raw payload objects SHALL NOT be rewritten: raw keys are content addresses (§2.1 of data.md) and `posting_id` is a join handle, never path content.

#### Scenario: New payload stored
- **WHEN** an ingestion run stores a payload whose content hash has no registry row
- **THEN** the system SHALL mint one `posting_id`, append exactly one registry row for that `source_hash`, and carry the `posting_id` on the ingestion event and the run's ingestion-manifest rows

#### Scenario: Duplicate payload resolves its existing id
- **WHEN** an ingestion run observes a payload whose content hash already has a registry row
- **THEN** the system SHALL reuse that `posting_id` without minting, and the manifest observation SHALL carry the existing id

#### Scenario: Legacy duplicate self-heals
- **WHEN** a duplicate (or stored) payload's content hash has no registry row because it predates the registry
- **THEN** the system SHALL mint a `posting_id`, append the registry row, and continue the run rather than failing

### Requirement: The registry is append-only with one id per hash
The registry SHALL enforce the uniqueness invariant: exactly one `posting_id` per `source_hash`, first mint wins; appends SHALL never re-mint an existing hash, and a replayed walk or re-ingestion SHALL be a no-op beyond observability.

#### Scenario: Same content never mints twice
- **WHEN** the same `source_hash` is minted against twice (re-ingestion, concurrent workers, or a repeated backfill)
- **THEN** both attempts SHALL resolve to the same single `posting_id` and the registry SHALL gain no duplicate row for that hash

### Requirement: Backfill assigns ids to pre-registry raw objects
The system SHALL provide a one-off backfill that walks `lake/raw/source=*/` in key order, mints one `posting_id` per unseen `source_hash`, and appends the registry rows, leaving every raw object byte-identical.

#### Scenario: Traversal order is mint order
- **WHEN** the backfill walks the raw prefixes
- **THEN** UUIDv7 mint order SHALL match traversal order, so id ordering is stable and reconstructible from the walk

#### Scenario: Idempotent re-walk
- **WHEN** the backfill runs again over a lake it already covered
- **THEN** it SHALL mint nothing and append nothing, and raw object counts and content hashes SHALL be unchanged
