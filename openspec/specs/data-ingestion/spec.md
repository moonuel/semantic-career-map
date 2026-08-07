# data-ingestion

## Purpose

TBD - Parse raw LinkedIn Markdown job postings into structured JSON records.

## Requirements

### Requirement: Parse raw markdown postings into structured JSON
The system SHALL read LinkedIn job postings from individual Markdown files in `data/selected-job-postings/*.md` and produce a single structured JSON file at `data/jobs.json` containing one record per posting with extracted metadata and the complete raw text.

#### Scenario: Metadata extraction
- **WHEN** a Markdown posting file is parsed
- **THEN** the output record SHALL include fields `id` (derived from filename stem), `title_raw` (extracted from the posting text), `company` (resolved from the filename), `source_file` (original filename), and `role_category` (hand-mapped from 6 categories: Data Scientist, ML Engineer, AI Engineer, Applied/Research, Data Engineer, Other)

#### Scenario: Full text preservation
- **WHEN** a posting is parsed
- **THEN** the output record SHALL include a `raw_full_text` field containing the complete original Markdown content verbatim

### Requirement: All 27 postings produce complete records
The system SHALL parse every Markdown file in the postings directory without exceptions, producing exactly one record per file with all metadata fields populated.

#### Scenario: No missing postings
- **WHEN** the postings directory contains 27 .md files
- **THEN** the output SHALL contain 27 records, one per file, with no exceptions or partial records
