## ADDED Requirements

### Requirement: Byte-exact span enumeration round-trip
The system SHALL enumerate every posting into an ordered set of spans such that rejoining the spans with their recorded delimiters reproduces the raw posting text byte-identically.

#### Scenario: Round-trip identity on enumeration
- **WHEN** a posting is enumerated into spans and the spans are rejoined without any deletion
- **THEN** the reassembled text SHALL equal the raw posting text byte for byte

#### Scenario: Fragment survival
- **WHEN** a posting contains list bullets, markdown headers, or unpunctuated fragments (e.g., "401(k) matching")
- **THEN** each fragment SHALL appear as a complete span with its original whitespace and structure preserved

### Requirement: Cleaning output is raw text minus deleted spans
The system SHALL produce cleaned text exclusively by rejoining the kept subset of enumerated spans; it SHALL NOT generate, reword, reorder, or reformat any text.

#### Scenario: Structural no-hallucination invariant
- **WHEN** a cleaned output is produced for any posting
- **THEN** the output SHALL be exactly the concatenation of the kept spans' original text, and this invariant SHALL be machine-checkable for every output

#### Scenario: Preserved content verbatim
- **WHEN** a span covering job duties, skills, qualifications, or team/role context is kept
- **THEN** its text SHALL appear in the output unchanged

### Requirement: Calibrated per-class span decisions
The system SHALL classify each span against every boilerplate class of the frozen v3 taxonomy, returning for each (span, class) pair a typed decision (delete / keep / unsure) with a calibrated confidence score.

#### Scenario: Confident class hit
- **WHEN** a span is clearly of a detected boilerplate class (e.g., a salary range)
- **THEN** the class decision SHALL be delete with confidence above the delete threshold

#### Scenario: Unsure band routing
- **WHEN** a span's class confidence falls between the keep and delete thresholds
- **THEN** the system SHALL route it to the unsure bucket and keep it by default, recording it in the per-span audit trail

#### Scenario: Position-bias-stable decisions
- **WHEN** the same span is classified with answer options presented in different orders
- **THEN** the resulting decision SHALL be stable under option rotation

### Requirement: Deterministic removal with cross-class header rule
The system SHALL apply removal decisions deterministically in code, including the rule that a section header SHALL be removed if and only if every span of its section is removed.

#### Scenario: Header follows a fully removed section
- **WHEN** every span under a header (up to the next header of equal or higher level) is decided delete
- **THEN** the header itself SHALL be removed from the output

#### Scenario: Header survives a partially kept section
- **WHEN** any span under a header is kept
- **THEN** the header SHALL be preserved in the output

### Requirement: Per-class error attribution
The system SHALL report, per arm and per boilerplate class, precision and recall against the hand-maintained golden slice, with the specific offending spans listed for every miss.

#### Scenario: Per-class quality reporting
- **WHEN** a benchmark run completes
- **THEN** the results SHALL include per-class precision/recall per arm, computed on the same hand-maintained golden slice as the monolith control arm

#### Scenario: Miss attribution
- **WHEN** a class shows false deletions or missed boilerplate
- **THEN** the affected spans SHALL be identifiable in the run's audit trail

### Requirement: Comparable multi-arm benchmark
The system SHALL benchmark the classifier arms against a monolith-prompt control arm on an identical golden slice, reporting Jaccard similarity, over-deletion, the output-subset invariant, per-class precision/recall, per-posting cost, and latency for every arm.

#### Scenario: Control arm comparability
- **WHEN** classifier arms are evaluated
- **THEN** the monolith control SHALL be evaluated under the same golden slice and metrics as Experiment 008's eval harness

#### Scenario: Adoption evidence gate
- **WHEN** adoption of the classifier path is considered
- **THEN** the decision SHALL be supported by arm-level results showing quality parity-or-better against the control at reported cost and latency
