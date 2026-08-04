# taxonomy-label-generation

## Purpose

TBD - See experiment 006 design for context.

## Requirements

### Requirement: LLM assigns taxonomy function tags and generates functional titles for job postings
The system SHALL accept a job posting's `llm_clean_text` field and the 12-category ML/AI responsibility taxonomy definitions with a controlled title vocabulary, and produce a structured JSON output containing a list of applicable function tags, a generated functional title from the controlled vocabulary, and a rationale for the assignment.

#### Scenario: Input from llm_clean_text
- **WHEN** processing any job posting
- **THEN** the system uses the posting's `llm_clean_text` field as the uniform input for taxonomy classification

#### Scenario: Structured output format
- **WHEN** the LLM processes a posting
- **THEN** the output SHALL be a JSON object with fields `function_tags` (a list of 1-N taxonomy category names), `generated_title` (a functional title from the controlled vocabulary), and `rationale` (a 1-2 sentence explanation)

#### Scenario: Controlled title vocabulary
- **WHEN** the LLM generates a functional title
- **THEN** the title SHALL be selected from a predefined mapping of taxonomy categories to title descriptors (e.g., `classical-ml` → "Classical ML Engineer", `agentic-ai` → "Agentic AI Engineer"), ensuring the same function type always maps to the same title

#### Scenario: Multi-label assignment
- **WHEN** a posting's responsibilities span multiple taxonomy categories
- **THEN** the LLM SHALL include all applicable categories in `function_tags` without artificial primary/secondary distinction

#### Scenario: Cached results skip API calls
- **WHEN** a posting has already been labeled and cached
- **THEN** the system SHALL return the cached result without making a new API call

#### Scenario: Prompt hash mismatch invalidates cache
- **WHEN** the taxonomy definitions or prompt text change between runs
- **THEN** the system SHALL detect the hash mismatch and invalidate the cache, re-labeling all postings

### Requirement: Golden subset validation of LLM labeling accuracy
The system SHALL compare LLM-assigned function tags against a manually labeled golden subset of 5 postings and report agreement metrics.

#### Scenario: Golden subset comparison
- **WHEN** golden labels exist for a posting
- **THEN** the system SHALL compute whether the golden tag appears in the LLM-assigned `function_tags` list and report the overall recall rate (fraction of golden postings where at least one golden tag is found in `function_tags`)

#### Scenario: Golden subset precision reporting
- **WHEN** the LLM assigns function tags that differ from the golden labels
- **THEN** the system SHALL report the posting ID, golden label set, LLM-assigned tag set, and LLM rationale for manual audit

### Requirement: Tag-set-colored UMAP visualization for qualitative evaluation
The system SHALL generate a UMAP scatter plot where each posting is colored by its full function_tags combination, enabling visual inspection of whether taxonomy-based groupings align with embedding-space clusters.

#### Scenario: Tag-set coloring
- **WHEN** all postings have been assigned function_tags
- **THEN** the system SHALL produce a UMAP plot with one point per posting, colored by the sorted tuple of its function_tags (e.g., `["classical-ml", "data-engineering", "mlops-production"]` gets a unique color)

#### Scenario: Color scheme and legend
- **WHEN** generating the UMAP plot
- **THEN** the system SHALL use a qualitative color palette with one color per unique tag set combination, ordered in the legend by tag set frequency (most common combinations first)

#### Scenario: Tag co-occurrence reporting
- **WHEN** generating the UMAP plot
- **THEN** the system SHALL print tag pair and triplet co-occurrence counts alongside the plot to surface composite role patterns

#### Scenario: Plot annotations
- **WHEN** generating the UMAP plot
- **THEN** each point SHALL be annotated with a short label (first 10 characters of posting ID) consistent with existing experiment plotting conventions
