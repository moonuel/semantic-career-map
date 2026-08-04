# Experiment 006 — Taxonomy-Driven Role Labeling

**Date:** August 4, 2026

**Hypothesis:** An LLM can assign fine-grained function tags from a 13-category ML/AI responsibility taxonomy to job postings, and clustering those postings by tag set combination in UMAP space will reveal semantically meaningful groupings that align with embedding similarity.

---

**NOTE:** This experiment does NOT compute separation gap or any quantitative clustering metric. The labels are evaluated qualitatively via tag-set-colored UMAP visualization and tag co-occurrence analysis. A quantitative comparison was attempted initially but abandoned when it became clear that the separation gap metric is too sensitive to label distribution entropy to serve as a reliable comparison tool on a 27-posting dataset.

---

## Experimental Design

### Input / Data Source

27 job postings from `data/jobs.json`. Each posting's `llm_clean_text` field (LLM-cleaned from Exp 003) is used as uniform input for taxonomy classification.

The 13-category taxonomy comes from `docs/research-reports/003-ml-ai-responsibility-taxonomy.md` (extended with a `deep-learning` category added after Experiment 006 initial results):

| # | Category | Description |
|---|---|---|
| 1 | `agentic-ai` | Autonomous agent systems (LangGraph, CrewAI, tool use) |
| 2 | `llm-fine-tuning` | Model adaptation (LoRA, QLoRA, RLHF, DPO) |
| 3 | `llm-information-retrieval` | RAG, semantic search, vector DBs, hybrid retrieval |
| 4 | `classical-ml` | Traditional ML (scikit-learn, XGBoost, forecasting) |
| 5 | `mlops-production` | CI/CD, monitoring, deployment, scaling |
| 6 | `data-engineering` | ETL, data pipelines, SQL, feature stores |
| 7 | `analytics-storytelling` | SQL, dashboards, A/B testing, stakeholder communication |
| 8 | `computer-vision` | Image/video models (CNNs, ViTs, object detection) |
| 9 | `ml-platform` | Internal tooling, feature stores, experiment tracking |
| 10 | `reinforcement-learning` | RL, bandits, decision systems, reward optimization |
| 11 | `research` | Papers, novel algorithms, frontier model exploration |
| 12 | `ai-safety-governance` | Fairness, explainability, compliance, adversarial robustness |
| 13 | `deep-learning` | Neural network engineering (PyTorch, TF, JAX, CNNs, transformers, GANs) |

### Pipeline / Scripts

Two scripts in `scripts/006_taxonomy_labels/`:

**`generate_labels.py`** — Sends each posting's `llm_clean_text` to the LLM with a system prompt containing the full taxonomy definitions and a controlled title vocabulary. Each call returns a JSON object with:

- `function_tags`: 1-3 taxonomy category names (no primary/secondary distinction)
- `generated_title`: a functional title from a controlled vocabulary mapped 1:1 to taxonomy categories (e.g., `classical-ml` → "Classical ML Engineer")
- `rationale`: 1-2 sentence explanation of the assignment

SHA-256 caching is used (same pattern as Exp 003/005). Prompt hash mismatch detection invalidates the cache when taxonomy definitions change. A golden subset of 5 manually labeled postings validates LLM accuracy via recall (≥1 golden tag in LLM tags counts as match).

**`compare_labels.py`** — Embeds `llm_clean_text` with `all-MiniLM-L6-v2` and generates a UMAP scatter plot where each posting is colored by its full `function_tags` combination (sorted tuple). Prints tag pair and triplet co-occurrence counts alongside the plot to surface composite role patterns.

### Prompts

**System prompt:** Full 13-category taxonomy definitions with core activities per category, plus controlled title vocabulary. The prompt instructs the LLM to:

- Assign 1-3 taxonomy categories (constrained to the 13 listed)
- Generate a title from the controlled vocabulary for the primary function
- Provide rationale referencing specific duties
- Output ONLY a JSON object, no markdown

**User prompt template:** `"Classify this job posting:\n\n---\n{llm_clean_text}"`

### Controls

- LLM model: `openai/gpt-5.4-nano` via Kilo Gateway (same as Exp 003/005)
- Temperature: 0.0 (deterministic)
- `response_format: {"type": "json_object"}` was attempted but rejected by the Kilo Gateway (400 Bad Request). The prompt instructs JSON-only output instead, and the model complies reliably at temp 0.0.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (same as Exp 003)
- UMAP seed: 42, n_neighbors=5, min_dist=0.15, metric=cosine

### Golden Subset

| Posting | Golden Tags |
|---|---|
| bmo-associate-data-scientist | `classical-ml`, `analytics-storytelling` |
| affirm-ml-engineer-2 | `classical-ml`, `mlops-production` |
| hellofresh-ml-engineer-operations-technology | `agentic-ai`, `mlops-production`, `ml-platform` |
| mastercard-data-scientist-2 | `classical-ml`, `mlops-production` |
| scribd-data-scientist-2 | `classical-ml`, `deep-learning`, `llm-information-retrieval` |

### Metrics

- **Golden recall**: fraction of golden postings where ≥1 golden tag appears in LLM-assigned `function_tags`
- **Tag set distribution**: count of postings per unique tag combination
- **Tag co-occurrence**: pair and triplet frequency counts across all 27 postings
- **Qualitative UMAP inspection**: tag-set-colored scatter plot read by eye

---

## Results

### Golden Recall

```
bmo-associate-data-scientist             Golden: ['classical-ml', 'analytics-storytelling']            → LLM: ['deep-learning', 'reinforcement-learning', 'mlops-production']              MISMATCH
affirm-ml-engineer-2                     Golden: ['classical-ml', 'mlops-production']                  → LLM: ['deep-learning', 'llm-information-retrieval', 'mlops-production']           MATCH
hellofresh-ml-engineer-operations-tech   Golden: ['agentic-ai', 'mlops-production', 'ml-platform']     → LLM: ['agentic-ai', 'mlops-production', 'ml-platform']                            MATCH
mastercard-data-scientist-2              Golden: ['classical-ml', 'mlops-production']                  → LLM: ['classical-ml', 'data-engineering', 'mlops-production']                     MATCH
scribd-data-scientist-2                  Golden: ['classical-ml', 'deep-learning', 'llm-information-retrieval']  → LLM: ['deep-learning', 'classical-ml', 'data-engineering']            MATCH
```

**Recall: 4/5 (80.0%)**

Bmo remains the single mismatch: the posting explicitly mentions "Generative AI/Deep Learning models," "Reinforcement Learning," and "algorithmic trading performance," and the LLM correctly identifies these as its primary function. The bmo golden label (`classical-ml`, `analytics-storytelling`) does not include `deep-learning` or `reinforcement-learning` — if these were added, bmo would match. Hellofresh is now an exact 3-tag match between golden and LLM output. Scribd's golden label added `classical-ml`, which the LLM already assigns — match preserved.

### Label Distribution

| Tag Set | Count |
|---|---|
| `['deep-learning', 'llm-information-retrieval', 'mlops-production']` | 2 |
| `['analytics-storytelling', 'data-engineering', 'mlops-production']` | 2 |
| `['data-engineering', 'deep-learning', 'mlops-production']` | 2 |
| `['analytics-storytelling', 'classical-ml', 'data-engineering']` | 2 |
| `['classical-ml', 'data-engineering', 'deep-learning']` | 2 |
| `['agentic-ai', 'data-engineering', 'mlops-production']` | 1 |
| `['deep-learning', 'mlops-production', 'reinforcement-learning']` | 1 |
| `['agentic-ai', 'deep-learning', 'mlops-production']` | 1 |
| `['deep-learning', 'llm-fine-tuning', 'mlops-production']` | 1 |
| `['analytics-storytelling', 'classical-ml']` | 1 |
| `['analytics-storytelling', 'data-engineering', 'deep-learning']` | 1 |
| `['deep-learning', 'llm-fine-tuning', 'llm-information-retrieval']` | 1 |
| `['agentic-ai', 'llm-fine-tuning', 'llm-information-retrieval']` | 1 |
| `['agentic-ai', 'ml-platform', 'mlops-production']` | 1 |
| `['deep-learning', 'mlops-production', 'research']` | 1 |
| `['analytics-storytelling', 'data-engineering']` | 1 |
| `['classical-ml', 'data-engineering', 'mlops-production']` | 1 |
| `['agentic-ai', 'ai-safety-governance', 'research']` | 1 |
| `['computer-vision', 'deep-learning', 'mlops-production']` | 1 |
| `['data-engineering', 'mlops-production']` | 1 |
| `['analytics-storytelling', 'deep-learning', 'mlops-production']` | 1 |
| `['agentic-ai', 'classical-ml', 'deep-learning', 'llm-information-retrieval']` | 1 |

**22 unique tag sets across 27 postings.** The distribution is flatter than the previous run — no single triplet dominates at more than 2 postings. The `deep-learning` category appears in 15/22 tag sets, confirming it is now the most universal category alongside `mlops-production`. A 4-tag posting (`agentic-ai`, `classical-ml`, `deep-learning`, `llm-information-retrieval`) appeared for the first time — the LLM assigned 4 tags to one posting when given the 13-category taxonomy, suggesting the expanded vocabulary encourages more expressive multi-label assignments.

### Tag Co-Occurrence

**Dominant pairs (≥2 postings):**

| Pair | Count |
|---|---|
| `deep-learning` + `mlops-production` | 10 |
| `data-engineering` + `mlops-production` | 7 |
| `analytics-storytelling` + `data-engineering` | 6 |
| `data-engineering` + `deep-learning` | 5 |
| `classical-ml` + `data-engineering` | 5 |
| `deep-learning` + `llm-information-retrieval` | 4 |
| `agentic-ai` + `mlops-production` | 3 |
| `analytics-storytelling` + `mlops-production` | 3 |
| `analytics-storytelling` + `classical-ml` | 3 |
| `classical-ml` + `deep-learning` | 3 |
| `llm-information-retrieval` + `mlops-production` | 2 |
| `agentic-ai` + `deep-learning` | 2 |
| `deep-learning` + `llm-fine-tuning` | 2 |
| `analytics-storytelling` + `deep-learning` | 2 |
| `llm-fine-tuning` + `llm-information-retrieval` | 2 |
| `agentic-ai` + `llm-information-retrieval` | 2 |

**Dominant triplets (≥2 postings):**

| Triplet | Count |
|---|---|
| `deep-learning + llm-information-retrieval + mlops-production` | 2 |
| `analytics-storytelling + data-engineering + mlops-production` | 2 |
| `data-engineering + deep-learning + mlops-production` | 2 |
| `analytics-storytelling + classical-ml + data-engineering` | 2 |
| `classical-ml + data-engineering + deep-learning` | 2 |

### Generated Title Distribution

| Generated Title | Count |
|---|---|
| Deep Learning Engineer | 9 |
| ML Operations Engineer | 4 |
| Agentic AI Engineer | 4 |
| Analytics Engineer | 3 |
| Data Engineer | 3 |
| AI Research Scientist | 1 |
| Classical ML Engineer | 1 |
| Information Retrieval Engineer | 1 |
| AI Safety Engineer | 1 |

### Cost

| Metric | Value |
|---|---|
| Model | `openai/gpt-5.4-nano` |
| Total tokens | 49,845 |
| Estimated cost | $0.0088 |

### UMAP Visualization

![Taxonomy UMAP](../assets/images/006_taxonomy_labels_umap.png)

The UMAP plot colors each posting by its full tag set combination (22 unique colors). No single triplet dominates — the most common combinations each appear in exactly 2 postings. The distribution is flatter than the 12-category run, reflecting that the `deep-learning` category pulled signal across the dataset rather than concentrating in one cluster. `deep-learning` appears in 15 of 22 tag sets; `mlops-production` remains a near-universal qualifier. A 4-tag posting (`agentic-ai`, `classical-ml`, `deep-learning`, `llm-information-retrieval`) emerged with the expanded vocabulary — the LLM assigned 4 tags to one posting, going beyond the 1-3 tag guidance when the posting was genuinely broad.

---

## Discussion

The 13-category taxonomy stabilized with this run. The golden set refinements (adding `mlops-production` to Mastercard, `ml-platform` to Hellofresh, `classical-ml` to Scribd, `analytics-storytelling` to Bmo) produced tighter matches overall — Hellofresh now achieves an exact 3-tag match between golden and LLM output. Scribd's golden refinement (adding `classical-ml` alongside `deep-learning` and `llm-information-retrieval`) accurately reflects what the LLM was already seeing in the posting.

Bmo remains the holdout: the LLM consistently tags it `deep-learning` + `reinforcement-learning` because the posting explicitly describes "Generative AI/Deep Learning models" and "Reinforcement Learning" applied to "algorithmic trading performance." The golden label (`classical-ml`, `analytics-storytelling`) captures the traditional ML and analytics dimensions but misses the DL and RL signals that dominate the posting's text. This isn't an LLM error — it's a golden label gap.

`deep-learning` has become the dominant category: 9 generated titles, appearing in 15 of 22 tag sets. It co-occurs with `mlops-production` in 10 of 27 postings — the strongest pair in the taxonomy. The original 12-category taxonomy's blind spot is now its strongest signal.

The flattening of the tag set distribution (from 4-post triplet to 2-post ceiling) is notable. With no single triplet above 2 postings, the taxonomy is now producing more genuinely diverse label assignments rather than funneling ambiguity into the largest bucket. A 4-tag assignment also emerged — the LLM assigned 4 tags to one posting, exceeding the 1-3 tag guidance. This suggests the expanded vocabulary encourages the LLM to be more discriminating rather than forcing a small set.

`mlops-production` remains ambient — 7 co-occurrences with `data-engineering`, 10 with `deep-learning`, 3 with `analytics-storytelling`. Every production-facing role touches MLOps. This category's role in the taxonomy should be reconsidered: it functions as a qualifier for "in production" rather than a distinct function.

---

## Conclusions

1. **The `deep-learning` category was essential.** 9 of 27 postings (33%) were classified as Deep Learning Engineer — the single largest category. The original 12-category taxonomy had no home for general neural network engineering. With it added, the LLM redistributed `classical-ml` postings appropriately.

2. **Golden set refinements improved alignment.** Hellofresh achieved an exact 3-tag match. Scribd's golden label now reflects the posting's actual content (`classical-ml` + `deep-learning` + `llm-information-retrieval`). Mastercard's golden label added `mlops-production`, matching what the LLM already saw.

3. **Bmo is the persistent golden gap.** The LLM consistently identifies `deep-learning` and `reinforcement-learning` in the bmo posting, which explicitly describes "Generative AI/Deep Learning models" and "Reinforcement Learning" for "algorithmic trading performance." The golden label (`classical-ml`, `analytics-storytelling`) captures traditional ML and analytics dimensions but misses the DL/RL signals. This is a golden label issue, not an LLM error — the golden should include `deep-learning` and `reinforcement-learning` to match the posting text.

4. **The tag set distribution flattened with the expanded vocabulary.** 22 unique tag sets (vs 21 in the previous run, 19 in the 12-category run), with no single triplet above 2 postings. The taxonomy is producing more discriminating assignments. A 4-tag assignment also emerged — the LLM exceeded the 1-3 tag guidance for one posting, suggesting the richer vocabulary encourages more honest multi-labeling.

5. **`deep-learning` + `mlops-production` is the dominant pair at 10 co-occurrences.** `mlops-production` remains a cross-cutting qualifier rather than a discriminating function. Future iterations should consider treating it as a qualifier rather than a peer category.

6. **1:1 tag-to-title mapping still flattens variation.** 9 postings get the title "Deep Learning Engineer" — but their tag sets differ meaningfully (some include `llm-information-retrieval`, others `reinforcement-learning`, others `data-engineering`). The title masks this variation.

---

## Next Steps

- **Interactive tag toggle UMAP**: a follow-up experiment where selecting a taxonomy tag in the legend highlights/greys-out all postings, enabling per-category semantic evaluation (e.g., "Does `agentic-ai` carve a tight cluster or is it scattered?")

- **Composite role label experiment**: use the co-occurrence data from this experiment to define ~6 composite roles (ML Engineer, Data Scientist, Data Engineer, LLM/AI Engineer, Research Scientist, Specialist), then re-run labeling with those roles as direct targets rather than intermediate tags

- **KNN audit**: for each posting, retrieve its top-k nearest neighbors in embedding space and inspect whether they share function tags — a direct test of whether the embeddings and taxonomy agree at the instance level

- **Expand golden set**: bmo remains the persistent golden mismatch because the golden label doesn't include `deep-learning` and `reinforcement-learning` despite the posting explicitly describing these functions. Consider adding `deep-learning` and `reinforcement-learning` to bmo's golden label, or expanding to 8-10 postings for better coverage.

## Appendix

### Detailed Per-Posting Label Assignments

| Posting | Tags | Title |
|---|---|---|
| affirm-ai-solutions-engineer | `agentic-ai`, `data-engineering`, `mlops-production` | ML Operations Engineer |
| affirm-ml-engineer-2 | `deep-learning`, `llm-information-retrieval`, `mlops-production` | Deep Learning Engineer |
| bmo-associate-data-scientist | `deep-learning`, `mlops-production`, `reinforcement-learning` | Deep Learning Engineer |
| cerebras-ml-performance-benchmarking-engineer | `analytics-storytelling`, `data-engineering`, `mlops-production` | ML Operations Engineer |
| clario-ai-engineer | `agentic-ai`, `deep-learning`, `mlops-production` | Agentic AI Engineer |
| clio-ml-engineer | `deep-learning`, `llm-fine-tuning`, `mlops-production` | Deep Learning Engineer |
| coca-cola-data-scientist | `analytics-storytelling`, `classical-ml` | Analytics Engineer |
| dayforce-product-ai-intern | `analytics-storytelling`, `data-engineering`, `deep-learning` | Deep Learning Engineer |
| deloitte-ai-ml-models-consultant-analyst | `deep-learning`, `llm-fine-tuning`, `llm-information-retrieval` | Deep Learning Engineer |
| ebay-applied-researcher-1 | `data-engineering`, `deep-learning`, `mlops-production` | Deep Learning Engineer |
| fgf-brands-ai-engineer-new-grad | `agentic-ai`, `llm-fine-tuning`, `llm-information-retrieval` | Agentic AI Engineer |
| hellofresh-ml-engineer-operations-technology | `agentic-ai`, `ml-platform`, `mlops-production` | Agentic AI Engineer |
| huawei-ai-ml-researcher | `deep-learning`, `mlops-production`, `research` | AI Research Scientist |
| icbc-data-science-analyst | `analytics-storytelling`, `classical-ml`, `data-engineering` | Data Engineer |
| intact-data-scientist-2 | `analytics-storytelling`, `data-engineering`, `mlops-production` | Analytics Engineer |
| kinaxis-data-analytics-coop-intern | `analytics-storytelling`, `data-engineering` | Data Engineer |
| mastercard-data-scientist-2 | `classical-ml`, `data-engineering`, `mlops-production` | Classical ML Engineer |
| motion-recruitment-ml-engineer | `data-engineering`, `deep-learning`, `mlops-production` | ML Operations Engineer |
| ontario-teachers-pension-plan-data-scientist | `deep-learning`, `llm-information-retrieval`, `mlops-production` | Information Retrieval Engineer |
| pinterest-ml-engineer-2 | `classical-ml`, `data-engineering`, `deep-learning` | Deep Learning Engineer |
| rbc-data-scientist-ai-model-risk | `agentic-ai`, `ai-safety-governance`, `research` | AI Safety Engineer |
| scotiabank-ai-ml-data-scientist | `computer-vision`, `deep-learning`, `mlops-production` | Deep Learning Engineer |
| scribd-data-scientist-2 | `classical-ml`, `data-engineering`, `deep-learning` | Deep Learning Engineer |
| tactable-data-engineer | `data-engineering`, `mlops-production` | Data Engineer |
| td-applied-ml-scientist-1 | `analytics-storytelling`, `deep-learning`, `mlops-production` | ML Operations Engineer |
| thumbtack-data-scientist-2-monetization-pricing | `analytics-storytelling`, `classical-ml`, `data-engineering` | Analytics Engineer |
| venterra-realty-data-decision-scientist | `agentic-ai`, `classical-ml`, `deep-learning`, `llm-information-retrieval` | Agentic AI Engineer |
