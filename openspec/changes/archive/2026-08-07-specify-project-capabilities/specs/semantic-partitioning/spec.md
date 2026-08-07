## ADDED Requirements

### Requirement: Partition job postings into orthogonal semantic dimensions
The system SHALL accept undifferentiated job posting text and produce two independent extracts via separate LLM passes: job-context ("what is this job?") and role-context ("what purpose does this role serve?"), each produced by a distinct system prompt.

#### Scenario: Job-context extraction (Pass A)
- **WHEN** the job-context prompt is applied
- **THEN** the output SHALL contain job duties, responsibilities, required skills, qualifications, and experience — with company descriptions, team context, organizational purpose, salary, benefits, and boilerplate stripped

#### Scenario: Role-context extraction (Pass B)
- **WHEN** the role-context prompt is applied
- **THEN** the output SHALL contain team name and purpose, role scope within the organization, team structure and stakeholders, and organizational impact — with specific job duties, qualifications, skills, and technical requirements stripped

#### Scenario: Verbatin preservation
- **WHEN** either LLM pass extracts text
- **THEN** the system SHALL preserve original wording verbatim without summarization, paraphrasing, or invention

### Requirement: Validate partition quality against hand-curated golden fields
The system SHALL evaluate LLM partition output against golden-set reference partitions (job-context and role-context) present in `data/golden_cleaned.json`, computing Jaccard similarity, cross-contamination, partition overlap, partition coverage, and hallucination detection.

#### Scenario: Jaccard against golden fields
- **WHEN** evaluating partition quality
- **THEN** the system SHALL compute Jaccard similarity between LLM job-context and golden job-context, and between LLM role-context and golden role-context

#### Scenario: Cross-contamination measurement
- **WHEN** evaluating partition quality
- **THEN** the system SHALL compute Jaccard similarity between LLM job-context and golden role-context (and vice versa) to measure contaminiation between the two dimensions

#### Scenario: Partition integrity check
- **WHEN** evaluating partition quality
- **THEN** the system SHALL compute overlap (Jaccard between LLM job-context and LLM role-context) and coverage (fraction of undifferentiated source words present in the union of both LLM partitions)

#### Scenario: Hallucination detection per partition
- **WHEN** evaluating partition quality
- **THEN** the system SHALL report words in each LLM partition that do not appear in the undifferentiated source text

### Requirement: Embedding comparison across semantic dimensions
The system SHALL embed five text variants (undifferentiated, golden job-context, golden role-context, LLM job-context, LLM role-context) for the golden-set postings and compute self-retrieval, separation gap, and UMAP visualizations comparing the semantic dimensions.

#### Scenario: Five-variant embedding
- **WHEN** comparing semantic partitions
- **THEN** the system SHALL embed all five text variants for each golden-set posting and report per-variant self-retrieval and separation gap

#### Scenario: Dimension-aware UMAP
- **WHEN** comparing semantic partitions
- **THEN** the system SHALL generate a 1×5 UMAP comparison plot where each subplot shows a different text variant, colored by role_category, enabling visual comparison of how job-context and role-context distribute postings in the embedding space

## Status

> **Research artifact** — semantic partitioning is not yet integrated into the production pipeline. The system prompt and evaluation framework are stable, but the approach has not been run on all 27 postings and role-context extraction fidelity (Jaccard 0.7851 vs golden) is below the threshold for production use. This spec documents current research behavior, not production requirements.
