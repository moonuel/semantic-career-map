## 1. Validate spec deltas against implementation

- [x] 1.1 Walk through `bootstrap.py` and confirm data-ingestion spec captures parsing behavior (title extraction, section detection, company resolution, metadata fields)
- [x] 1.2 Walk through `extract_clean_text.py` and confirm llm-text-extraction spec captures cleaning behavior (system prompt, caching, prompt hash invalidation)
- [x] 1.3 Walk through `compare_variants.py` and confirm embedding-pipeline spec captures encode, self-retrieval, separation gap, and UMAP
- [x] 1.4 Walk through `eval_cleaning.py` and confirm evaluation-framework spec captures Jaccard, boilerplate detection, hallucination detection, over-deletion
- [x] 1.5 Walk through `extract_fields.py`, `eval_partition.py`, and `compare_embeddings.py` and confirm semantic-partitioning spec captures two-pass extraction, golden validation, partition integrity, and embedding comparison
- [x] 1.6 Run `ruff format` on all written files

## 2. Sync delta specs to main specs

- [x] 2.1 Run `/opsx:sync specify-project-capabilities` to merge all 5 delta specs into `openspec/specs/`
- [x] 2.2 Verify all 5 new capability specs exist under `openspec/specs/` (data-ingestion, llm-text-extraction, embedding-pipeline, evaluation-framework, semantic-partitioning)
- [x] 2.3 Verify `openspec list --specs` shows 6 total specs (5 new + taxonomy-label-generation)

## 3. Verify spec consistency

- [x] 3.1 Spot-check each spec for: at least one requirement, each requirement has at least one scenario using `#### Scenario:` format
- [x] 3.2 Confirm semantic-partitioning spec includes the research-artifact status annotation
- [x] 3.3 Run `openspec validate --specs` and fix any validation errors

## 4. Archive and document

- [x] 4.1 Run `/opsx:archive specify-project-capabilities` to close the migration change
- [x] 4.2 Update `docs/CROSS-REFERENCE-INDEX.md` section 1 (Project Status) to note that openspec/specs/ is now the canonical status tracker, superseding the mkdocs status table for capability tracking
- [x] 4.3 Optionally: update `docs/architecture.md` data pipeline diagram to reference spec names
