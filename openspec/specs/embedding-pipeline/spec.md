# embedding-pipeline

## Purpose

TBD - Convert cleaned job posting text into L2-normalized dense embedding vectors for semantic similarity search.

## Requirements

### Requirement: Embed text into L2-normalized vectors
The system SHALL encode job posting text into dense vector representations suitable for semantic similarity search, with L2 normalization applied to produce unit-norm vectors.

#### Scenario: Dense embedding output
- **WHEN** a batch of N texts is embedded
- **THEN** the output SHALL be a float32 array with one row vector per input text

#### Scenario: L2 normalization
- **WHEN** embeddings are produced
- **THEN** every row vector SHALL have an L2 norm of 1.0 within a tolerance of 1e-5

#### Scenario: CPU-only inference
- **WHEN** the model is loaded
- **THEN** it SHALL be loaded with `device="cpu"`

### Requirement: Support multiple text variants for comparison
The system SHALL support embedding different text variants (raw_full_text, llm_clean_text, etc.) from the same job posting for comparative evaluation.

#### Scenario: Variant selection
- **WHEN** a text variant field name is specified (e.g., "llm_clean_text")
- **THEN** the system SHALL use that field's text from each posting record, falling back to "raw_full_text" if the field is missing or empty
