## MODIFIED Requirements

### Requirement: Extract job-signal content via LLM
The system SHALL accept a lake raw payload (`lake/raw/source=<source>/sha256=<hash>.json`), normalize its `description_html` into plain text, and produce cleaned job-signal text containing only responsibilities, qualifications, skills, and education requirements — with all boilerplate content removed — using an LLM with a structured system prompt and a user message template via the Kilo Gateway API.

#### Scenario: Boilerplate removal
- **WHEN** the posting text contains salary ranges, benefits (health/dental/vision, vacation/PTO), EEO/diversity statements, company descriptions, team culture narratives, recruiter notes, office locations, or employee testimonials
- **THEN** the cleaned text SHALL exclude all such content

#### Scenario: Job content preservation
- **WHEN** the posting text contains job duties, responsibilities, technical skills, tools, frameworks, required qualifications, preferred qualifications, or education requirements
- **THEN** the cleaned text SHALL preserve all such content verbatim without summarization, paraphrasing, or invention

#### Scenario: Empty response handling
- **WHEN** the LLM returns an empty or whitespace-only response for a posting
- **THEN** the system SHALL raise a RuntimeError for that posting

#### Scenario: Model and prompt configuration identity
- **WHEN** a cleaning run is configured (model, system prompt, user message template)
- **THEN** the system SHALL compute a `configuration_hash` (SHA-256 over the concatenated configuration) and stamp it on every artifact row and manifest the run produces

### Requirement: Deterministic caching by content hash
The system SHALL cache LLM cleaning results in the lake: the existence of a cleaned-text artifact row keyed by the posting's raw-text hash and the run's `configuration_hash` constitutes the cache. Re-used rows MUST be read through the `cleaned-artifacts` capability; results SHALL NOT be persisted to any additional flat-file cache.

#### Scenario: Cache hit on identical input and configuration
- **WHEN** a posting's raw-text hash and the run's `configuration_hash` match an existing artifact row
- **THEN** the system SHALL return the stored cleaned text without making an API call

#### Scenario: Prompt or model change invalidates full re-clean
- **WHEN** the system prompt or user message template changes between runs (detected by a `configuration_hash` mismatch)
- **THEN** the system SHALL find no matching cached rows and SHALL re-clean every posting, writing the results as a versioned dataset without overwriting earlier versions

#### Scenario: Legacy flat-file cache not written
- **WHEN** a cleaning run persists cleaned results
- **THEN** the system SHALL write artifact rows and manifests to the lake and SHALL NOT write `data/.llm_clean_cache.json`
