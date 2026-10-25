# evaluation-framework

## Purpose

TBD - Compute proxy metrics for fast iteration and evaluate extraction quality against golden set.

> **Metrics under research.** Self-retrieval and separation gap are current proxy metrics but the evaluation methodology is still under investigation. See `docs/research-reports/004-llm-evals-report.md`. This spec defines what the system computes today; metrics may change as evaluation research matures.

## Requirements

### Requirement: Compute proxy metrics for fast iteration without labeled data
The system SHALL compute self-retrieval accuracy and separation gap on any embedding matrix paired with role_category labels, enabling rapid comparison of preprocessing variants.

#### Scenario: Self-retrieval degeneracy check
- **WHEN** self-retrieval is computed
- **THEN** each posting receives 1.0 if its own embedding is the nearest neighbor, 0.5 if it is the second-nearest, and 0.0 otherwise; the aggregate score equals (sum of scores / N) × 100%

#### Scenario: Separation gap discriminability
- **WHEN** separation gap is computed
- **THEN** it SHALL equal the difference between mean pairwise cosine similarity of same-role postings and mean pairwise cosine similarity of different-role postings

#### Scenario: Summary table output
- **WHEN** multiple embedding variants are compared
- **THEN** the system SHALL print a formatted summary table with columns for variant name, self-retrieval percentage, separation gap, mean cosine similarity, and max cosine similarity

### Requirement: Evaluate LLM extraction quality against golden set
The system SHALL compare LLM-cleaned text from `cleaned-artifacts` rows against hand-cleaned golden references stored as authored rows under `eval/datasets/type=cleaning_golden/variant=author/version=<vN>/`, using Jaccard similarity, boilerplate detection, hallucination detection, and over-deletion measurement, and SHALL persist each evaluation run's results as rows under `eval/results/type=cleaning_quality/variant=<variant>/version=<vN>/`.

#### Scenario: Golden set storage
- **WHEN** the golden set is loaded for evaluation
- **THEN** reference rows SHALL identify the posting (source hash and source id) and carry the golden reference text, annotator, and creation date

#### Scenario: Jaccard similarity measurement
- **WHEN** comparing cleaned text to the golden reference
- **THEN** the system SHALL compute Jaccard similarity as the intersection of word sets divided by the union

#### Scenario: Boilerplate marker detection
- **WHEN** cleaned text is evaluated
- **THEN** the system SHALL scan for 7 regex patterns covering salary, benefits, EEO, testimonials, recruiter notes, company mission, and salary amounts, reporting any matches as under-deletion

#### Scenario: Hallucination detection
- **WHEN** cleaned text is evaluated
- **THEN** the system SHALL report any words present in the cleaned text that do not appear in the posting's raw text, with a sample of the first 10 hallucinated words

#### Scenario: Over-deletion measurement
- **WHEN** cleaned text is evaluated
- **THEN** the system SHALL report the percentage of golden-set words absent from the cleaned text, flagging cases where more than 5% of golden words are missing

#### Scenario: Results persistence with lineage
- **WHEN** an evaluation run completes
- **THEN** result rows SHALL record the posting identity, per-metric values, the evaluation `configuration_hash`, `evaluation_version`, and `created_at`, so the same dataset can be re-evaluated under different models or rubrics

### Requirement: Generate embedding-space visualizations
The system SHALL generate PCA (2-component + scree plot), UMAP (cosine metric), and t-SNE (cosine metric) scatter plots of the embedding space colored by role_category, with point labels for company name and title.

#### Scenario: PCA visualization
- **WHEN** a PCA plot is generated
- **THEN** the system SHALL produce a 2-panel figure with a 2D scatter plot (labeled by role category and company prefix, annotated with explained variance ratios) and a scree plot showing per-component and cumulative explained variance

#### Scenario: UMAP visualization
- **WHEN** a UMAP plot is generated
- **THEN** the system SHALL use cosine metric with n_neighbors=5, min_dist=0.15, and random_state=42, with points colored by role_category and labeled with company prefix and truncated title

#### Scenario: t-SNE visualization
- **WHEN** a t-SNE plot is generated
- **THEN** the system SHALL use cosine metric with perplexity=min(5, N-1) and random_state=42, with points colored by role_category and annotated similarly to UMAP
