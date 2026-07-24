# Results

## Summary Dashboard

<div class="grid cards" markdown>

-   **Current Dataset:** 27 job postings (target: ~350 with augmentation)
-   **Embedding Dimension:** 384 (all-MiniLM-L6-v2)
-   **Self-Retrieval:** 100.0% (all variants — no embedding collapse)
-   **Best Separation Gap:** +0.0493 (LLM-cleaned text, 8.2× over raw)

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
- **Title-based role labels are noisy.** Function-based labeling (12-category taxonomy) is needed for honest clustering evaluation
- **Small dataset (27 postings).** Data augmentation (template-based synthetic postings) planned to densify the embedding space

### Next Steps

- Proxy metrics formalization (self-retrieval, separation gap, nearest-neighbor audit)
- Skill extraction with spaCy PhraseMatcher
- Weighted concatenation experiments (title weighting, multi-field fusion)
- Golden set construction for pytrec_eval (Precision@5, MRR, NDCG)
- Template-based data augmentation (27 seeds → ~350 postings)
