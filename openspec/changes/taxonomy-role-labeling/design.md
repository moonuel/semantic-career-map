## Context

The project has 27 job postings embedded with MiniLM-L6-v2. The `compare_variants.py` script (exp 003) evaluates clustering quality using `role_category` — a 6-label set derived from raw job titles (Data Scientist, ML Engineer, AI Engineer, Applied/Research, Data Engineer, Other). The evaluation doc explicitly calls out that these labels are noisy and a 12-category function taxonomy exists as a research artifact ready for use.

This experiment does not re-embed anything. It operates on the existing embedding matrix and compares two labeling schemes — same vectors, different ground truth.

## Goals / Non-Goals

**Goals:**
- Assign function tags (1-3 per posting) to all 27 postings via LLM
- Validate LLM labeling accuracy against a manually labeled 5-posting golden subset
- Generate a UMAP visualization where each posting is colored by its full tag set combination, for qualitative inspection of taxonomy-aligned clustering
- Surface tag co-occurrence patterns to inform future role-labeling experiments

**Non-Goals:**
- Re-embedding any text variant
- Computing separation gap or any quantitative clustering metric — evaluation is qualitative via tag-set-colored UMAP
- KNN audit or nearest-neighbor evaluation (deferred to future experiment)
- Training any model or modifying the embedding pipeline
- Expanding the golden set beyond 27 postings
- Interactive tag toggle in the UMAP (deferred to follow-up)

## Decisions

### D1: Input text for LLM labeling — llm_clean_text for all 27 postings

**Decision:** Use `llm_clean_text` as the sole input for taxonomy classification across all 27 postings. No per-posting logic, no fallback chain.

**Rationale:** The taxonomy classifies jobs by function — what the person actually does. `llm_clean_text` contains duties, responsibilities, required skills, and qualifications for all 27 postings. These are the signal the taxonomy definitions are built on. Team descriptions, company boilerplate, and benefits — which `llm_clean_text` removes — don't help distinguish between taxonomy categories. A "Data Cognition Team" description doesn't tell you whether a role is doing classical-ml or reinforcement-learning; the duties do.

**Alternatives considered:**
- `sections.responsibilities` with `llm_clean_text` fallback — rejected because `responsibilities` is only present in 19/27 postings, and the fallback introduces an input quality inconsistency. Using one uniform source removes a variable.
- `raw_full_text` — rejected because salary/benefits/boilerplate noise adds irrelevant signal that could confuse the LLM's function classification.
- Running Exp 005 `llm_job_context` extraction on all 27 as a pre-processing step — rejected because it adds 54 LLM calls (2 passes × 27 postings) without a clear benefit for classification. The taxonomy classification task is semantic reasoning, not verbatim text preservation.

### D2: LLM output format — structured JSON with function_tags, generated_title, and rationale

**Decision:** Each LLM call returns a JSON object with three fields:
- `function_tags`: A list of 1-3 taxonomy category names (e.g., `["classical-ml", "mlops-production"]`). No primary/secondary distinction — just whichever categories apply. The tag set is the primary output used for UMAP visualization.
- `generated_title`: A short functional title constrained to a controlled vocabulary derived from the taxonomy (e.g., "Classical ML Engineer", "Agentic AI Engineer"). Stored as metadata for qualitative reference and future search experiments — NOT used as a class label for any quantitative metric.
- `rationale`: 1-2 sentence explanation of the tag assignment.

**Rationale:** The `function_tags` list is the core deliverable — the tag set combination colors the UMAP plot, and tag co-occurrence patterns (e.g., `classical-ml + data-engineering + mlops-production` as the dominant triplet) inform intuition about how taxonomy categories compose into real-world roles. The `generated_title` remains useful as a human-readable label but is not leaned on for quantitative evaluation.

### D3: Qualitative evaluation via tag-set-colored UMAP — no separation gap

**Decision:** Evaluate taxonomy labeling quality by coloring a UMAP scatter plot with each posting's full tag set combination and inspecting visually whether same-colored postings cluster together. Do not compute separation gap or any quantitative clustering metric.

**Rationale:** The separation gap metric is sensitive to label distribution (entropy) and fragile with small datasets — a high-entropy label set can produce a higher gap simply by having more singleton categories filtered out, not because the labels are better. Tag-set coloring lets us read the structure directly: we can see whether `[classical-ml, data-engineering, mlops-production]` postings form a tight cluster, whether `mlops-production` is a noise tag spread everywhere, and which tag combinations produce natural groupings. Tag co-occurrence counts (printed alongside the plot) surface composite role patterns without imposing them programmatically.

### D4: LLM model and temperature — gpt-5.4-nano at temperature 0.0

**Decision:** Use the same model and temperature as experiments 003 and 005.

**Rationale:** Consistent with prior experiments, zero hallucination risk at deterministic temperature, same caching mechanism. Cost is negligible at 27 calls.

### D5: Golden subset size — 5 postings

**Decision:** Manually label 5 postings with taxonomy tags and compute LLM accuracy against them.

**Rationale:** Consistent with experiments 003 and 005 which used 5-posting golden sets. Picking diverse postings (across the 5 original golden set IDs: bmo, affirm-ml-engineer-2, hellofresh, mastercard, scribd) gives coverage across Data Scientist, ML Engineer, and AI Engineer original labels. 27 is small enough that 5 gives reasonable calibration.

### D6: Tag co-occurrence reporting

**Decision:** Print tag pair and triplet co-occurrence counts alongside the UMAP plot to surface composite role patterns.

**Rationale:** Co-occurrence counts reveal which taxonomy categories naturally compose into real-world roles without imposing programmatic rules. For example, `classical-ml + data-engineering + mlops-production` appearing as the dominant triplet signals that these three categories form the core of an "ML Engineer" profile. This information informs future experiments that may define composite role labels.

## Risks / Trade-offs

- **LLM misclassification risk:** The LLM may assign wrong taxonomy tags, especially for ambiguous postings. Mitigation: golden subset validation quantifies accuracy; rationale field enables manual audit.
- **Sparse categories risk:** With 12 taxonomy categories on 27 postings, several will appear in few postings, producing small or singleton tag sets on the UMAP. Mitigation: tag co-occurrence counts reveal which categories overlap; singleton tag sets are still visible in the plot and provide their own qualitative signal.
- **Tag-set coloring complexity:** With 17 unique tag combinations on 27 postings, the UMAP legend will be dense. Mitigation: color scheme uses a 17-color palette mapped to tag sets, with the legend ordered by tag set frequency. Future interactive toggle will improve legibility.
- **Golden set selection bias:** The 5 manually labeled postings may not represent the full diversity. Mitigation: pick the same 5 from experiments 003/005 golden set, which span Data Scientist, ML Engineer, and AI Engineer categories.
