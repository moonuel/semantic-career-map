## ADDED Requirements

### Requirement: Embed text into L2-normalized vectors
The system SHALL encode job posting text into dense 384-dimensional vector representations using the `sentence-transformers/all-MiniLM-L6-v2` model on CPU, with L2 normalization applied to produce unit-norm vectors.

#### Scenario: Dimensionality
- **WHEN** a batch of N texts is embedded
- **THEN** the output SHALL be a float32 array of shape (N, 384)

#### Scenario: L2 normalization
- **WHEN** embeddings are produced
- **THEN** every row vector SHALL have an L2 norm of 1.0 within a tolerance of 1e-5

#### Scenario: CPU-only inference
- **WHEN** the model is loaded
- **THEN** it SHALL be loaded with `device="cpu"`

### Requirement: Support multiple text variants for comparison
The system SHALL support embedding different text variants (raw_full_text, clean_text, llm_clean_text) from the same job posting for comparative evaluation.

#### Scenario: Variant selection
- **WHEN** a text variant field name is specified (e.g., "llm_clean_text")
- **THEN** the system SHALL use that field's text from each posting record, falling back to "raw_full_text" if the field is missing or empty

### Requirement: Embedding comparison and metrics
The system SHALL compare embedding variants by computing self-retrieval accuracy and separation gap, and by generating side-by-side UMAP visualizations with role-category coloring.

#### Scenario: Self-retrieval computation
- **WHEN** self-retrieval is computed on an embedding matrix
- **THEN** each posting SHALL be scored as 1.0 if the diagonal is its own top-ranked neighbor, 0.5 if the diagonal is its second-ranked neighbor, and 0.0 otherwise, with the final score reported as (total score / N) × 100%

#### Scenario: Separation gap computation
- **WHEN** separation gap is computed on an embedding matrix with role_category labels
- **THEN** the gap SHALL be computed as mean(within-role cosine similarity) − mean(cross-role cosine similarity) across all pairs of postings

#### Scenario: UMAP comparison visualization
- **WHEN** a comparison is run across N text variants
- **THEN** the system SHALL generate a 1×N subplot figure with UMAP projections colored by role_category, using cosine metric with n_neighbors=min(5, N_postings-1), min_dist=0.15, and a fixed random seed of 42
