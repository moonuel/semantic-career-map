## 1. Setup

- [x] 1.1 Create `scripts/006_taxonomy_labels/` directory with `__init__.py`
- [x] 1.2 Embed the 12-category taxonomy definitions from the research report as a Python constant (12 entries with name, description, and core activities)
- [x] 1.3 Create the controlled title vocabulary: mapping from each taxonomy category to a canonical functional title (e.g., `classical-ml` -> "Classical ML Engineer")
- [x] 1.4 Create the LLM labeling prompt: system prompt with taxonomy definitions + controlled title vocabulary + user prompt template with posting llm_clean_text

## 2. Golden Subset Labeling

- [x] 2.1 Manually label 5 postings (bmo, affirm-ml-engineer-2, hellofresh, mastercard, scribd) with function tag sets (1-N taxonomy categories per posting)
- [x] 2.2 Store golden labels as a Python dict in `generate_labels.py` for validation

## 3. Label Generation Script

- [x] 3.1 Implement `generate_labels.py` with LLM call logic (Kilo Gateway, gpt-5.4-nano, temp 0.0, structured JSON response)
- [x] 3.2 Implement SHA-256 caching per posting (same pattern as exp 003/005: input hash -> cache entry)
- [x] 3.3 Implement prompt hash mismatch detection to invalidate cache when taxonomy changes
- [x] 3.4 Read `llm_clean_text` from each posting in `data/jobs.json` as uniform input
- [x] 3.5 Implement output parsing: extract `function_tags` (list), `generated_title` (string), `rationale` (string) from LLM JSON response
- [x] 3.6 Implement golden subset validation: compute recall (fraction of golden postings where >=1 golden tag appears in LLM `function_tags`) and report mismatches with rationale
- [x] 3.7 Output labeled results to `data/006_taxonomy_labels.json`
- [x] 3.8 Implement cost estimation (token counting, model pricing)

## 4. Tag-Set UMAP Visualization

- [x] 4.1 Load existing `llm_clean_text` embeddings (re-embed from `data/jobs.json` with same MiniLM-L6-v2) and the `data/006_taxonomy_labels.json` tag assignments
- [x] 4.2 Build tag-set labels: sorted tuple of `function_tags` per posting as the color class
- [x] 4.3 Generate UMAP scatter plot with one color per unique tag set combination, legend ordered by frequency
- [x] 4.4 Create qualitative color scheme for tag set combinations (one color per unique combination)
- [x] 4.5 Print tag pair and triplet co-occurrence counts alongside the plot
- [x] 4.6 Save UMAP plot to `data/plots/006_taxonomy_labels_umap.png`

## 5. Validation

- [x] 5.1 Run `generate_labels.py` end-to-end, verify all 27 postings receive tags and titles
- [x] 5.2 Verify golden subset recall rate is reported
- [x] 5.3 Verify tag-set UMAP plot is saved to `data/plots/006_taxonomy_labels_umap.png`
- [x] 5.4 Verify tag co-occurrence counts are printed
- [x] 5.5 Run `ruff check` and `ruff format` on all scripts
