## Why

The project currently has only 1 of its ~7 capability areas specified in OpenSpec (`taxonomy-label-generation`). The remaining capabilities — data ingestion, LLM text extraction, embedding pipeline, evaluation framework, and semantic partitioning — are described entirely in narrative mkdocs pages (`docs/technical/`, `docs/architecture.md`) with no structured requirements. This makes the system opaque to agents and brittle to change. The mirror is incomplete.

## What Changes

- Create new OpenSpec capability specs for 5 areas, extracting stable requirements from existing documentation and code
- Each spec captures what the system SHALL do — observable behavior, not implementation or metrics
- No code changes. No data changes. This is documentation of existing system behavior.

## Capabilities

### New Capabilities

- `data-ingestion`: Parse raw LinkedIn Markdown job postings into structured JSON with title, company, role category, and sections
- `llm-text-extraction`: Extract responsibilities and qualifications from raw posting text via LLM, removing boilerplate (salary, benefits, EEO, recruiter notes) while preserving job-relevant content verbatim
- `embedding-pipeline`: Convert cleaned job posting text into L2-normalized 384d embedding vectors using all-MiniLM-L6-v2, with deterministic content-hash caching and dot-product similarity
- `evaluation-framework`: Compute proxy metrics (self-retrieval, separation gap) for fast iteration; evaluate extraction quality against 5-posting golden set; support visualization of embedding space via PCA, UMAP, and t-SNE
- `semantic-partitioning`: Separate job postings into orthogonal job-context and role-context dimensions via LLM, enabling weighted embedding strategies (research artifact, not yet in production pipeline)

### Modified Capabilities

None — `taxonomy-label-generation` is unchanged and already specified.

## Impact

- Affected code: none (no code changes — specs derived from existing `scripts/001_baseline/bootstrap.py`, `scripts/003_llm_extraction/`, `scripts/005_semantic_partitioning/`, `scripts/006_taxonomy_labels/`, `backend/embeddings.py`, `backend/preprocessing.py`)
- Affected data: none (specs reference existing golden set and dataset, do not modify them)
- Affected docs: none initially. After all specs sync to main, `docs/CROSS-REFERENCE-INDEX.md` and related pages may be updated to point to specs rather than carrying duplicate authority.
