# Results

!!! info "Results are also in progress!"
    This page will be used to summarize the performance results of the pipeline, once it is completed. 
    
    In the meantime, this page is used to document the strongest findings that will be solidified in the final work.

As of July 27, 2026:

- `MiniLM-L6-v2` is sufficient and capable of embedding text at this scale (job postings).
- UMAP appears the most effective (compared to PCA and t-SNE) for visualizing potential text embedding clusters, likely since the embedded vectors are normalized to the unit hypersphere.
- Using LLM-based (`gpt-5.4-nano`) text cleaning appears to significantly improve separation of embedded vectors, compared to raw and regex-based cleaning methods.
- Job postings can be semantically partitioned into "what is this job?" and "what purpose does this role serve?" dimensions with high fidelity for job-context (mean Jaccard 0.9720) and moderate fidelity for role-context (mean Jaccard 0.7851).
- LLM-extracted job-context improves separation gap by 5.6× over raw text, but LLM-extracted role-context shows a negative separation gap (-0.0074), suggesting this dimension is harder to extract cleanly.

<!-- 
## Summary Dashboard

<div class="grid cards" markdown>

-   **Current Dataset:** 27 job postings (target: "many more" with augmentation)
-   **Embedding Dimension:** 384 (all-MiniLM-L6-v2)
-   **Degeneracy Check:** 100.0% (no collapse over 27 embeddings)
-   **Improved Separation Gap:** +0.0493 (LLM-cleaned text, 8.2× over raw)

</div>

---

## Benchmark Results

### Embedding Variant Comparison

| Variant | Self-Retrieval | Separation Gap | Mean CosSim |
|---|---|---|---|
| Raw (`raw_full_text`) | 100.0% | +0.0060 | 0.4045 |
| Section-Cleaned (`clean_text`) | 100.0% | +0.0203 | 0.4298 |
| **LLM-Cleaned (`llm_clean_text`)** | **100.0%** | **+0.0493** | 0.5280 |

The separation gap measures the difference between mean within-role cosine similarity and mean cross-role cosine similarity. Higher values indicate better cluster separation between role categories. The LLM-cleaned variant achieves an **8.2× improvement** over raw text (+0.0060 → +0.0493) and a **2.4× improvement** over regex-based section cleaning (+0.0203 → +0.0493).

---

## LLM Extraction Quality

Primary model: gpt-5.4-nano, evaluated against 5 hand-cleaned golden set postings.

| Posting | Jaccard vs Golden | Boilerplate | Hallucinations | Over-deletion |
|---|---|---|---|---|
| BMO Data Scientist | 0.801 | CLEAN | CLEAN | 19.9% |
| **Affirm ML Engineer 2** | **1.000** | CLEAN | CLEAN | **0.0%** |
| HelloFresh ML Engineer | 0.775 | CLEAN | CLEAN | 22.5% |
| Mastercard Data Scientist 2 | 0.610 | CLEAN | CLEAN | 29.9% |
| Scribd Data Scientist 2 | 0.614 | CLEAN | CLEAN | 38.6% |

**Key observations:**

- Zero hallucinations across all 5 postings — the LLM never invents content
- Zero boilerplate detected — salary, benefits, EEO statements completely stripped
- Over-deletion increases with organizational context density (team descriptions, company mission)

---

## Qualitative Analysis

### What the LLM Consistently Preserves

- Job responsibilities and day-to-day tasks
- Required qualifications and experience levels
- Technical skills (Python, SQL, Spark, PyTorch, etc.)
- Education requirements
- Original wording of preserved sections

### What the LLM Consistently Removes

- Salary ranges and compensation details
- Benefits (health, dental, vacation, parental leave)
- EEO/diversity statements
- Recruiter notes and application instructions
- Company culture and "About Us" boilerplate

### What the LLM Inconsistently Removes

- Organizational context describing the team's role and the position's scope within the company
- This is the primary area for improvement — system prompt tuning to preserve team role context alongside job requirements

---

## Visualization Results

Baseline PCA/UMAP/t-SNE findings from Experiment 001:

- **PCA:** Weak clustering of ML Engineer roles. Data Scientist roles spread widely — title ambiguity where "Data Scientist" spans SQL analytics to production ML
- **UMAP:** AI Engineer roles form a distinct cluster (agentic work carries distinct semantic signal). Research roles are widely separated (search ranking vs multimodal AI)
- **t-SNE:** Weaker visualization overall but confirms semantic pairings (eBay search ranking adjacent to Pinterest ML Engineer — both involve search ranking)
- **Key insight:** Title-based role labels are noisy. Function-based labeling (what the job actually does) is needed for meaningful clustering evaluation

!!! info "Plot Availability"
    Visualization plots are in `assets/images/`:
    - `raw_pca_baseline.png`, `raw_umap_baseline.png`, `raw_tsne_baseline.png`
    - `exp_boilerplate_pca.png`, `exp_boilerplate_umap.png`, `exp_boilerplate_tsne.png`
    - `llm_cleaning_comparison.png`

---

## Conclusions

### What Worked

- **LLM-based boilerplate removal significantly improves embedding quality.** Separation gap of +0.0493 — 8.2× over raw text
- **100% self-retrieval confirms no embedding collapse.** All three text variants produce distinct, non-degenerate embeddings
- **Zero hallucinations across 5 gold-standard postings.** The LLM never invents content not present in the original text

### What Needs Improvement

- **LLM over-deletion of organizational context (19–39%).** Team role context carries semantic signal that helps distinguish similar-sounding roles
- **LLM role-context extraction shows moderate fidelity (mean Jaccard 0.7851) and negative separation gap (-0.0074).** Organizational mission extraction is harder than job-duty extraction; prompt refinement or different LLM models may improve this
- **Title-based role labels are noisy.** Function-based labeling (12-category taxonomy) is needed for honest clustering evaluation
- **Small dataset (27 postings).** Data augmentation (template-based synthetic postings) planned to densify the embedding space

### Next Steps

- Feature engineering on extracted partitions (job-context + role-context combined features)
- Expand evaluation set beyond 5 golden postings for statistical power
- Prompt refinement for role-context extraction to improve Jaccard and separation gap
- Experimentation with other LLM models for semantic partitioning
- Skill extraction with spaCy PhraseMatcher
- Weighted concatenation experiments (title weighting, multi-field fusion)
- Golden set construction for pytrec_eval (Precision@5, MRR, NDCG)
- Template-based data augmentation (27 seeds → ~350 postings) -->
