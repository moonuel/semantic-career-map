## Why

Experiment 003 showed that LLM-cleaned text (stripping company boilerplate, team descriptions, and organizational context) improved separation gap by title from +0.0060 to +0.0493. But the stripping was aggressive — it removed text that describes the team's function and domain, which may carry meaningful semantic signal about what a role actually *is*.

The working assumption is that the embedding model clusters based on semantic similarity, not just text similarity. If true, then adding organizational context back should improve clustering by *function* (not just title). If false, the separation gap improvement was an artifact of removing unique text per posting, and adding context back will hurt.

To test this, we need organizational context stored as a separate, clean field alongside the existing `llm_clean_text` — enabling a proper A/B comparison.

However, the existing `llm_clean_text` field is **contaminated**: the current extraction prompt inconsistently preserves org context mixed in with responsibilities. This means the Experiment 003 evaluation metrics are muddied — the LLM was both penalized for correctly stripping org context and rewarded for inadvertently keeping it. We need to tighten the extraction prompt and re-evaluate against a clean golden set before a proper A/B test is possible.

## What Changes

### Data & Evaluation
- The golden set (`data/golden_cleaned.json`) has been manually split: each posting now has `org-context` (team function, domain, mission) and `req-context` (pure responsibilities, qualifications) fields
- Re-evaluate the existing `llm_clean_text` against the new `req-context` golden set field

### LLM Extraction — Tighten existing pass
- Update the `llm_clean_text` extraction prompt to explicitly strip org context (team function, domain, mission) even when interleaved with responsibilities
- Re-run extraction on all 27 postings with the tightened prompt
- Re-evaluate against the `req-context` golden set field

### LLM Extraction — New pass
- Add a new LLM extraction pass that extracts "organizational context" from raw job postings — text describing the team's function, domain, and mission within the company
- Store the extracted context as a new `llm_org_context` field in `data/jobs.json`
- Evaluate against the `org-context` golden set field

### A/B Comparison
- Run a three-way comparison of embedding variants:
  - Variant A: `llm_clean_text` only (re-extracted, pure responsibilities)
  - Variant B: `llm_clean_text` + `llm_org_context` concatenated
  - Variant C: `llm_org_context` only
- Evaluate using separation gap and nearest-neighbor audit, keyed by **function tag** from the responsibility taxonomy (not by title)

## Capabilities

### New Capabilities
- `org-context-extraction`: LLM-based extraction of organizational context (team function/domain) from unstructured job posting text, with golden-set evaluation

### Modified Capabilities
- None — no existing specs

## Impact

- `data/golden_cleaned.json`: `org-context` and `req-context` fields added (already done)
- `data/jobs.json`: `llm_clean_text` field regenerated, new `llm_org_context` field added
- `scripts/005_org_context/extract_clean_text.py` prompt (via 003 pipeline): updated to strip org context more aggressively
- `scripts/005_org_context/`: new extraction and evaluation scripts for org context pass
- `scripts/003_llm_extraction/eval_cleaning.py`: updated to evaluate against `req-context` field
- `scripts/005_org_context/compare_variants.py`: embedding comparison script handling three text variants
- No changes to backend or API code
