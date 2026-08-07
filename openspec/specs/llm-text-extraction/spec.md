# llm-text-extraction

## Purpose

TBD - Extract job-signal content (responsibilities, qualifications) from raw posting text via LLM, removing boilerplate.

## Requirements

### Requirement: Extract job-signal content via LLM
The system SHALL accept a job posting's `raw_full_text` and produce an `llm_clean_text` field containing only responsibilities, qualifications, skills, and education requirements — with all boilerplate content removed — using an LLM with a structured system prompt via the Kilo Gateway API.

#### Scenario: Boilerplate removal
- **WHEN** raw_full_text contains salary ranges, benefits (health/dental/vision, vacation/PTO), EEO/diversity statements, company descriptions, team culture narratives, recruiter notes, office locations, or employee testimonials
- **THEN** llm_clean_text SHALL exclude all such content

#### Scenario: Job content preservation
- **WHEN** raw_full_text contains job duties, responsibilities, technical skills, tools, frameworks, required qualifications, preferred qualifications, or education requirements
- **THEN** llm_clean_text SHALL preserve all such content verbatim without summarization, paraphrasing, or invention

#### Scenario: Empty response handling
- **WHEN** the LLM returns an empty or whitespace-only response for a posting
- **THEN** the system SHALL raise a RuntimeError for that posting

### Requirement: Deterministic caching by content hash
The system SHALL cache LLM cleaning results keyed by SHA-256 hash of the concatenated system prompt and user message template, and by SHA-256 hash of each posting's raw_full_text content.

#### Scenario: Cache hit on identical input
- **WHEN** a posting has already been cleaned and its raw_full_text content hash matches the cached hash
- **THEN** the system SHALL return the cached result without making an API call

#### Scenario: Prompt hash mismatch invalidates entire cache
- **WHEN** the system prompt or user message template changes between runs (detected by SHA-256 mismatch of the prompt hash)
- **THEN** the system SHALL invalidate all cached entries and re-clean every posting

#### Scenario: Cache persistence
- **WHEN** the system writes cleaned results to a posting's llm_clean_text field
- **THEN** the system SHALL also persist the result to `data/.llm_clean_cache.json` for reuse across runs
