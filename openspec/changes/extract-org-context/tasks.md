## 1. Golden Set — Split and validate

- [x] 1.1 Verify `golden_cleaned.json` split: confirm all 5 postings have `org-context` and `req-context` fields with non-empty content
- [x] 1.2 Spot-check boundary rule: verify mixed-zone sentences are in `org-context`, specific duties and qualifications are in `req-context`
- [x] 1.3 Validate split consistency: run a word-level overlap check between `org-context` and `req-context` for each posting to confirm no significant duplication

## 2. Tighten llm_clean_text extraction prompt

- [x] 2.1 Update `extract_clean_text.py` prompt: add explicit instructions to strip team function, domain, and mission descriptions; add mixed-zone exclusion rule
- [x] 2.2 Re-run `extract_clean_text.py` on all 27 postings with tightened prompt
- [x] 2.3 Update `eval_cleaning.py` to use `req-context` field from split golden set as evaluation reference
- [x] 2.4 Run evaluation and confirm Jaccard similarity, hallucination, over-deletion metrics improve compared to Experiment 003 baseline

## 3. New org context extraction pass

- [x] 3.1 Create extraction script (`scripts/005_org_context/extract_org_context.py`) following the same LLM pipeline pattern as `extract_clean_text.py`
- [x] 3.2 Implement org context extraction prompt per design Decision 5, targeting only team function/domain/mission text
- [x] 3.3 Run extraction on all 27 postings, producing `llm_org_context` field in `jobs.json`
- [x] 3.4 Create evaluation script (`scripts/005_org_context/eval_org_context.py`) using `org-context` field from split golden set as reference
- [x] 3.5 Run evaluation and tune prompt if Jaccard, hallucination, or over-deletion metrics are unsatisfactory

## 4. A/B comparison of embedding variants

- [x] 4.1 Create `scripts/005_org_context/compare_variants.py` to handle three text variants: `llm_clean_text` (new), `[ORG] llm_org_context [RESP] llm_clean_text`, and `llm_org_context` alone
- [x] 4.2 Compute separation gap by function tag for all three variants
- [x] 4.3 Compute nearest-neighbor audit (top-5 neighbors share at least one function tag) for all three variants
- [x] 4.4 Generate comparison table and visualizations (UMAP/PCA color-coded by function tag)
- [x] 4.5 Document findings in experiment report, noting which variant performs best and whether the semantic similarity hypothesis is supported
