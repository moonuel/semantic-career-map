## MODIFIED Requirements

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
