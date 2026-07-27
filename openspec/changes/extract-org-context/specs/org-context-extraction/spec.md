## ADDED Requirements

### Requirement: Golden set split into separate org-context and responsibilities references
The system SHALL use the already-split golden set in `data/golden_cleaned.json` where each of the 5 evaluation postings has two reference fields: `org-context` for organizational context and `req-context` for pure responsibilities.

**Boundary rule**: Organizational context is text that describes the team's function, domain, or mission within the company — what business problem the team exists to solve, narrated in domain language. Responsibilities are specific daily tasks, technical requirements, and qualifications, even when they reference the same domain. Mixed-zone sentences (blending domain AND duties) SHALL be classified as org context.

#### Scenario: Golden set validation
- **WHEN** the split golden set is loaded from `data/golden_cleaned.json`
- **THEN** each posting SHALL have non-empty `org-context` and `req-context` fields as separate evaluation targets

### Requirement: Extract pure responsibilities from job postings
The system SHALL extract pure job responsibilities from unstructured job posting text using an LLM with a tightened prompt that explicitly strips organizational context, producing an updated `llm_clean_text` field for each posting free of team function/domain descriptions.

#### Scenario: Responsibilities extraction excludes org context
- **WHEN** the LLM receives a job posting containing both team descriptions and specific responsibilities
- **THEN** the output SHALL contain only duties, qualifications, requirements, and technical skills — NOT team function, domain, or mission descriptions

#### Scenario: Mixed-zone sentence exclusion from responsibilities
- **WHEN** a sentence simultaneously describes the team's domain and specific duties (e.g., "On the Servicing ML team, you will build ML systems that automate customer operations such as disputes, returns, fraud, and chargebacks")
- **THEN** the sentence SHALL NOT appear in the `llm_clean_text` output

### Requirement: Evaluate responsibilities extraction quality
The system SHALL evaluate `llm_clean_text` extraction quality against the `req-context` field of the split golden set for 5 postings, using Jaccard similarity, hallucination detection, and over-deletion rate.

#### Scenario: Responsibilities evaluation
- **WHEN** the extraction is evaluated against the 5 golden-set postings
- **THEN** the system SHALL report Jaccard similarity, presence of hallucinated content, and over-deletion percentage using the `req-context` field as reference

### Requirement: Extract organizational context from job postings
The system SHALL extract organizational context from unstructured job posting text using a new LLM prompt, producing a separate `llm_org_context` field for each posting.

#### Scenario: Successful extraction from a posting with clear section boundaries
- **WHEN** the LLM receives a job posting containing distinct team description and responsibilities sections
- **THEN** the output SHALL contain the team's function, domain, mission, and role-purpose text but NOT include salary, benefits, EEO statements, or recruiter boilerplate

#### Scenario: Mixed-zone sentence handling
- **WHEN** a sentence simultaneously describes the team's domain and specific duties (e.g., "You will build ML systems that automate customer operations such as disputes, returns, fraud, and chargebacks")
- **THEN** the sentence SHALL be included in the org context output

#### Scenario: No organizational context present
- **WHEN** a job posting contains no team description, domain context, or role-purpose text
- **THEN** the system SHALL output an empty string for `llm_org_context`

### Requirement: Evaluate org context extraction quality
The system SHALL evaluate org context extraction quality against the `org-context` field of the split golden set for 5 postings, using Jaccard similarity, hallucination detection, and over-deletion rate.

#### Scenario: Org context evaluation
- **WHEN** the extraction is evaluated against the 5 golden-set postings
- **THEN** the system SHALL report Jaccard similarity, presence of hallucinated content, and over-deletion percentage using the `org-context` field as reference

### Requirement: Enable A/B comparison of embedding variants
The system SHALL support generating three embedding variants for each posting:
- Variant A: `llm_clean_text` only (re-extracted, pure responsibilities)
- Variant B: `llm_org_context` + `llm_clean_text` concatenated with section markers
- Variant C: `llm_org_context` only

#### Scenario: Variant generation
- **WHEN** the A/B comparison script is run
- **THEN** it SHALL produce embeddings for all three variants and report separation gap by function tag and nearest-neighbor audit results
