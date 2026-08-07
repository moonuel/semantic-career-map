## ADDED Requirements

### Requirement: Parse raw markdown postings into structured JSON
The system SHALL read LinkedIn job postings from individual Markdown files in `data/selected-job-postings/*.md` and produce a single structured JSON file at `data/jobs.json` containing one record per posting with extracted metadata and section text.

#### Scenario: Metadata extraction
- **WHEN** a Markdown posting file is parsed
- **THEN** the output record SHALL include fields `id` (derived from filename stem), `title_raw` (extracted from explicit patterns or filename heuristics), `company` (resolved from known mapping or filename fallback), `source_file` (original filename), and `role_category` (hand-mapped from 6 categories: Data Scientist, ML Engineer, AI Engineer, Applied/Research, Data Engineer, Other)

#### Scenario: Section detection
- **WHEN** a posting contains known section headers
- **THEN** the system SHALL split the text into named sections using 23 regex patterns covering about_role, responsibilities, qualifications, nice_to_have, what_we_offer, and about_team, with unrecognized content collected under a _preamble key

#### Scenario: Full text preservation
- **WHEN** a posting is parsed
- **THEN** the output record SHALL include a `raw_full_text` field containing the complete original Markdown content verbatim

### Requirement: All 27 postings produce complete records
The system SHALL parse every Markdown file in the postings directory without exceptions, producing exactly one record per file with all metadata fields populated.

#### Scenario: No missing postings
- **WHEN** the postings directory contains 27 .md files
- **THEN** the output SHALL contain 27 records, one per file, with no exceptions or partial records

#### Scenario: Non-empty sections
- **WHEN** a posting is parsed
- **THEN** the about_role, responsibilities, and qualifications sections SHALL each contain non-empty text for every posting
