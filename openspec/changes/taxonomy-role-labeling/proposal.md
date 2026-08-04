## Why

Current clustering evaluation uses title-based `role_category` labels (Data Scientist, ML Engineer, AI Engineer, etc.) as ground truth. These labels are noisy — a "Data Scientist" doing SQL analytics and a "Data Scientist" building NLP/LLM systems share a title but do fundamentally different work. The project acknowledged this limitation in the evaluation docs and commissioned a 12-category function taxonomy in `docs/research-reports/003-ml-ai-responsibility-taxonomy.md` specifically to address it. This experiment replaces title-based labels with taxonomy-based function tags and measures whether the new labels produce better cluster separation on the same embeddings.

## What Changes

- New experiment script that uses an LLM to assign taxonomy function tags (1-3 per posting) to all 27 postings based on their llm_clean_text
- Golden subset validation: 5 postings manually labeled with taxonomy tags to calibrate LLM accuracy
- UMAP scatter plot where each posting is colored by its full tag set combination, enabling qualitative inspection of whether taxonomy-based groupings align with embedding-space clusters
- Interactive tag toggle deferred to a follow-up experiment: selecting a tag highlights/greys-out postings in the plot for per-category semantic evaluation

## Capabilities

### New Capabilities

- `taxonomy-label-generation`: LLM-driven assignment of function category tags from the 12-category ML/AI responsibility taxonomy to job postings, with golden subset validation of label accuracy

### Modified Capabilities

<!-- No existing capabilities modified — this is a new evaluation experiment. -->

## Impact

- Affected code: New script `scripts/006_taxonomy_labels/generate_labels.py` (tag generation plus UMAP visualization)
- Affected data: New JSON output `data/006_taxonomy_labels.json`; new plot `data/plots/006_taxonomy_labels_umap.png`
- Dependencies: Uses existing `data/jobs.json`, existing taxonomy definitions from the research report, and the same `llm_clean_text` embeddings reused from experiment 003
- No changes to backend modules, pipeline, or existing experiments
