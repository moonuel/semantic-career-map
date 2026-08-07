## 1. Update Taxonomy Definitions

- [x] 1.1 Add `deep-learning` category definition (13th entry, numbered 13) to `TAXONOMY_DEFINITIONS` in `generate_labels.py` with prompt for "Neural Network Engineering" covering architecture design, training loops, PyTorch/TensorFlow/JAX, GPU optimization, transfer learning
- [x] 1.2 Add `"deep-learning": "Deep Learning Engineer"` to `TITLE_VOCABULARY` in `generate_labels.py`
- [x] 1.3 Update the taxonomy category count reference in the system prompt from "12 categories" to "13 categories"

## 2. Update Golden Subset Labels

- [x] 2.1 Update Scribd golden label from `["llm-fine-tuning", "llm-information-retrieval"]` to `["deep-learning", "llm-information-retrieval"]`
- [x] 2.2 Verify remaining 4 golden labels (bmo, affirm-ml-engineer-2, hellofresh, mastercard) do not need changes

## 3. Re-Run Labeling and Visualization

- [x] 3.1 Delete `data/.006_taxonomy_cache.json` to ensure fresh cache with new prompt hash
- [x] 3.2 Run `generate_labels.py` — verify all 27 postings receive tags with the 13-category taxonomy
- [x] 3.3 Run `compare_labels.py` — verify updated UMAP and co-occurrence output
- [x] 3.4 Verify golden subset recall is reported with updated labels

## 4. Update Documentation

- [x] 4.1 Update `docs/experiments/006-taxonomy-role-labeling.md` to reflect the 13-category taxonomy in the Experimental Design section, Results section, Discussion, and Appendix (per-posting label table)
- [x] 4.2 Run `ruff check` and `ruff format` on modified scripts
