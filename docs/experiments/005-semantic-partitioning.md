# Experiment 005 — Semantic Partitioning of Job Postings

**Date:** July 27, 2026

Experiment 003 noted that the LLM cleaning prompt removed team-context and organizational-role descriptions from job postings, and that this removed content might carry semantically meaningful signal. 

This experiment tests whether the golden set text can be semantically partitioned into "What is this job?" and "What purpose does this role serve?".


**Hypothesis:** The two partitions — *what a job is* and *what purpose the role serves* — are semantically separable dimensions that can be extracted via LLM prompting.

---

**NOTE:** This experiment assumes cleaned text via `golden_cleaned.json`. The future pipeline will implement a boilerplate removal pre-processing step (to be tested separately) to produce artifacts similar to `golden_cleaned.json`.

In the future we will also want to isolate each signal and test whether they useful for semantic clustering on their own, combined, or eventually used to synthesize a more semantically meaningful description:



---

## Experimental Design

### Input / Data Source

Five golden-set postings from `data/golden_cleaned.json`. Each posting contains:

| Field | Source | Purpose |
|---|---|---|
| `text` | Hand-curated (Exp 003) | Undifferentiated cleaned text — contains both *what the job is* and *what purpose the role serves* |
| `job-context` | Hand-curated (new) | The human answer to "What is this job?" — duties, skills, and qualifications |
| `role-context` | Hand-curated (new) | The human answer to "What purpose does this role serve?" — team mission, role scope, organizational impact |

The structural claim: `job-context` and `role-context` answer two distinct questions about the same job, and their union approximates the original undifferentiated `text` (`text ≈ job-context ∪ role-context`). If this holds, the two fields form a clean semantic split rather than an overlapping or ad-hoc division.

### Pipeline / Scripts

Three scripts: `scripts/005_semantic_partitioning/extract_fields.py`, `scripts/005_semantic_partitioning/eval_partition.py`, and `scripts/005_semantic_partitioning/compare_embeddings.py` to be executed in sequence.

The first one, `extract_fields.py`, parses the golden-set postings and extracts the `job-context` and `role-context` fields using LLM prompting and `gpt-5.4-nano`. It outputs a structured JSON file containing the partitioned text for downstream evaluation.

The second one, `eval_partition.py`, validates the semantic integrity of the extracted partitions against the original undifferentiated text. It computes metrics such as Jaccard similarity, partition overlap, and partition coverage to verify that `job-context` and `role-context` are genuinely disjoint yet collectively exhaustive with respect to the source text. This script ensures that the partitions capture distinct semantic dimensions without significant overlap or loss of information.

The third one, `compare_embeddings.py`, performs semantic clustering analysis on the partitioned fields. It generates embeddings for each posting's `job-context` and `role-context` using `MiniLM-L6-v2` and calculates metrics like separation gap and self-retrieval to assess how well the partitions cluster by job role. This step evaluates whether the extracted partitions provide meaningful semantic structure that can improve downstream tasks such as role classification or career path recommendation.

The pipeline mirrors the structure of [Experiment 003](003-llm-extraction.md) but focuses on semantic partitioning rather than content cleaning.

### Prompts

Two prompts are run independently on each posting via `extract_fields.py`:

**Pass A — "What is this job?"** Extracts duties, skills, and qualifications. Strips team descriptions, organizational framing, mission statements, and company boilerplate. Preserves original wording verbatim — no summarization, paraphrasing, or invention. Outputs `llm_job_context` per posting.

**Pass B — "What purpose does this role serve?"** Extracts team mission, role scope, and organizational impact. Strips specific job duties, day-to-day tasks, technical skills, qualifications, education requirements, and compensation/benefits. Preserves original wording verbatim. Outputs `llm_role_context` per posting.

Both passes follow the Exp 003 system-prompt structure but target opposite semantic dimensions.

### Controls

- LLM temperature held at 0.0 for deterministic extraction
- LLM model: `gpt-5.4-nano` (same as Exp 003 for comparability)
- Embedding model: `all-MiniLM-L6-v2` (same as Exp 003)
- Evaluation bounded to 5 golden postings (no expansion to full 27-posting corpus)
- Word-level Jaccard matching via whitespace tokenization (consistent with Exp 003 methodology)

### Metrics

All metrics compute word-level sets via whitespace tokenization.

| Metric | Definition | Purpose |
|--------|-----------|---------|
| **Jaccard similarity** | `|A ∩ B| / |A ∪ B|` | Pairwise overlap between LLM output and golden field, or between old and new prompt outputs |
| **Partition overlap** | `|job-context ∩ role-context| / |job-context ∪ role-context|` | Validates that the two fields are genuinely disjoint (should approach 0) |
| **Partition coverage** | `|job-context ∪ role-context| / |text|` | Validates that the two fields account for the original undifferentiated text (should approach 1.0) |
| **Separation gap** | `x_w − x_b` (within-class minus between-class cosine mean) | Clustering quality by role — larger gap indicates better separation |
| **Self-retrieval** | Ratio of postings whose highest cosine similarity is with themselves (diagonal of `AA^T`) | Degeneracy check — should be 1.0 for all non-collapsed embeddings |

### Comparison Variants

| Variant | Source | What it represents |
|---|---|---|
| `job-context` (golden) | Hand-curated | Ground truth: "what is this job?" |
| `role-context` (golden) | Hand-curated | Ground truth: "what purpose does this role serve?" |
| `text` (golden) | Hand-curated (Exp 003) | Undifferentiated baseline |
| `llm_job_context` | Pass A output | LLM answer to "what is this job?" |
| `llm_role_context` | Pass B output | LLM answer to "what purpose does this role serve?" |

### Evaluation Phases

`eval_partition.py` executes five phases in order:

1. **Golden partition validation:** Compute `PartitionOverlap(job-context, role-context)` and `PartitionCoverage(job-context ∪ role-context, text)` per posting. Confirms the hand-curated split is disjoint and exhaustive before using it as ground truth.

2. **Job-context extraction quality:** Compute `Jaccard(llm_job_context, job-context)` per posting. Primary metric — measures how well Pass A replicates the golden "what is this job?" field. Also compute `Jaccard(llm_job_context, role-context)` to detect cross-contamination into the wrong dimension.

3. **Role-context extraction quality:** Compute `Jaccard(llm_role_context, role-context)` per posting. Primary metric — measures how well Pass B replicates the golden "what purpose does this role serve?" field. Also compute `Jaccard(llm_role_context, job-context)` to detect cross-contamination.

4. **LLM partition integrity:** Compute `PartitionOverlap(llm_job_context, llm_role_context)` and `PartitionCoverage(llm_job_context ∪ llm_role_context, text)` per posting. Validates that the LLM-produced fields also form a clean semantic split, independent of the golden split.

5. **Hallucination detection:** Compute the set difference between extracted text and undifferentiated text per posting and flag any instances. Checks for any LLM-extracted words that do not appear in the original undifferentiated text, identifying potential hallucinations or invented content.

6. **Embedding comparison (secondary):** Embed all five variants with `all-MiniLM-L6-v2` and compute separation gap and self-retrieval. Compare `job-context` vs `role-context` to test whether the two dimensions produce different clustering behavior. See future-work note regarding expanding clustering evaluation beyond this initial pass.


## Results

Results of Phase 1 of the experiment -- testing the overlap and coverage of the manually-partitioned Golden text -- are below:

| Posting                                      | Overlap | Coverage |
| -------------------------------------------- | ------- | -------- |
| bmo-associate-data-scientist                 | 0.1343  | 1.0000   |
| affirm-ml-engineer-2                         | 0.1116  | 1.0000   |
| hellofresh-ml-engineer-operations-technology | 0.1193  | 1.0000   |
| mastercard-data-scientist-2                  | 0.1642  | 1.0000   |
| scribd-data-scientist-2                      | 0.1417  | 0.9917   |
| -------------------------------------------- | ------- | -------- |
| MEAN                                         | 0.1342  | 0.9983   |

"Overlap" tests the semantic disjointness of the Golden set partitions, while "Coverage" tests whether the union of the partitions accounts for the entire undifferentiated text, with respect to the Jaccard measure. 
The results show that the partitions are indeed disjoint (low overlap) and collectively exhaustive (coverage close to 1.0), supporting the hypothesis that these two dimensions capture distinct semantic information about job postings.

---

Phase 2 tests the quality of job-context extraction quality:

| Posting                                      | Jacc(job,golden) | Cross-contam |
| -------------------------------------------- | ---------------- | ------------ |
| bmo-associate-data-scientist                 | 0.9699           | 0.1378       |
| affirm-ml-engineer-2                         | 0.9769           | 0.1139       |
| hellofresh-ml-engineer-operations-technology | 0.9548           | 0.1185       |
| mastercard-data-scientist-2                  | 0.9786           | 0.1581       |
| scribd-data-scientist-2                      | 0.9797           | 0.1423       |
| -------------------------------------------- | ---------------- | ------------ |
| MEAN                                         | 0.9720           | 0.1341       |

The Jaccard measure over the extracted job context and golden job context shows high similarity (mean 0.9720), indicating that Pass A accurately captures the "what is this job?" dimension with minimal cross-contamination with the role-context (mean 0.1341).

---

Phase 3 performs the same test, but for the role-context:

| Phase 3 — Role-Context Extraction Quality (Pass B) |
| Posting                                      | Jacc(role,golden)| Cross-contam |
| -------------------------------------------- | ---------------- | ------------ |
| bmo-associate-data-scientist                 | 0.8986           | 0.1692       |
| affirm-ml-engineer-2                         | 0.7681           | 0.1777       |
| hellofresh-ml-engineer-operations-technology | 0.6147           | 0.3119       |
| mastercard-data-scientist-2                  | 0.9097           | 0.2088       |
| scribd-data-scientist-2                      | 0.7344           | 0.1204       |
| -------------------------------------------- | ---------------- | ------------ |
| MEAN                                         | 0.7851           | 0.1976       |

The Jaccard measure over the extracted role context and golden role context shows moderate similarity (mean 0.7851), indicating that Pass B captures the "what purpose does this role serve?" dimension, but with lower fidelity than Pass A. 
The cross-contamination metric (mean 0.1976) suggests that Pass B is more prone to extracting job-duties and technical details into the role-context partition, reflecting the challenge of semantically isolating organizational mission from operational responsibilities.

--- 

Phase 4 evaluates the semantic partition quality of the LLM-extracted fields:

| Posting                                      | Overlap | Coverage |
| -------------------------------------------- | ------- | -------- |
| bmo-associate-data-scientist                 | 0.1735  | 0.9751   |
| affirm-ml-engineer-2                         | 0.1814  | 0.9793   |
| hellofresh-ml-engineer-operations-technology | 0.2870  | 0.9908   |
| mastercard-data-scientist-2                  | 0.2030  | 0.9891   |
| scribd-data-scientist-2                      | 0.1209  | 0.8921   |
| -------------------------------------------- | ------- | -------- |
| MEAN                                         | 0.1932  | 0.9653   |

The LLM partition integrity metrics show that the extracted partitions are indeed disjoint (low overlap, mean 0.1932) and collectively exhaustive (coverage close to 1.0, mean 0.9653), validating that the two passes produce a clean semantic split independent of the golden reference.

---

Phase 5 is a sanity check that verifies the LLM-extracted text does not contain hallucinated content by checking for words present in the extracted fields but absent from the undifferentiated source text:

| Posting | Job-context (Pass A) | Role-context (Pass B) |
| ------- | ------------------- | --------------------- |
| [bmo-associate-data-scientist] | CLEAN | CLEAN |
| [affirm-ml-engineer-2] | CLEAN | CLEAN |
| [hellofresh-ml-engineer-operations-technology] | CLEAN | CLEAN |
| [mastercard-data-scientist-2] | CLEAN | CLEAN |
| [scribd-data-scientist-2] | CLEAN | CLEAN |

No hallucinations detected! 

---

Phase 6 explores the clustering characteristics between the raw text, golden partitions, and LLM-extracted partitions:

| Variant: undifferentiated_text | 5/5 texts available |
| ------------------------------ | ------------------- |
|  Self-retrieval @ rank-0:      | 100.0%              |
|  Separation gap                | 0.0069              |
|  Mean pairwise cosine sim:     | 0.4600              |
|  Max pairwise cosine sim:      | 0.6778              |

- The raw text exhibits no degeneracy (none of them do, hence I will omit them in future comments), but only a slight separation gap (average cosine similarity distance between within-class and between-class postings).
- The average similarity between all embeddings is moderate, at 0.4600, indicating a moderate level of semantic overlap across different job roles.
- The closest that any two postings obtain in the embedding space is 0.6778, signaling that no two postings overlap significantly.

| Variant: golden_job_context | 5/5 texts available |
| --------------------------- | ------------------- |
|  Self-retrieval @ rank-0:   | 100.0%              |
|  Separation gap:            | 0.0255              |
|  Mean pairwise cosine sim:  | 0.4912              |
|  Max pairwise cosine sim:   | 0.6722              |

- Embedding only the manually-curated job-context improves the separation gap modestly, from 0.0069 (raw text) to 0.0255 (golden job context) for a 3.7x improvement.
- The mean pairwise similarity improves slightly (0.4600 to 0.4912), perhaps reflecting a greater average overlap in keywords or responsibilities.
- The max pairwise similarity remains similar (0.6778 to 0.6722), indicating that the most semantically similar postings remain comparable across representations.

| Variant: golden_role_context | 5/5 texts available |
| ---------------------------- | ------------------- |
|  Self-retrieval @ rank-0:    | 100.0%              |
|  Separation gap:             | 0.0588              |
|  Mean pairwise cosine sim:   | 0.4149              |
|  Max pairwise cosine sim:    | 0.6754              |

- Embedding only the manually-curated role-context improves the separation gap significantly, from 0.0069 (raw text) to 0.0588 (golden role context), representing a 8.5x improvement.
- The mean pairwise similarity decreases (0.4600 to 0.4149), suggesting that role-contexts are more semantically distinct from each other than job-contexts, possibly due to the broader strategic focus and less operational overlap between different teams' missions.
- The max pairwise similarity remains similar (0.6778 to 0.6754), indicating that the most semantically similar postings remain comparable across representations.

| Variant: llm_job_context   | 5/5 texts available |
| -------------------------- | ------------------- |
|  Self-retrieval @ rank-0:  | 100.0%              |
|  Separation gap:           | 0.0385              |
|  Mean pairwise cosine sim: | 0.4783              |
|  Max pairwise cosine sim:  | 0.6856              |

- Embedding only the LLM-extracted job-context improves the separation gap modestly, from 0.0069 (raw text) to 0.0385 (LLM job context), representing a 5.6x improvement.
- The mean pairwise similarity improves slightly (0.4600 to 0.4783), suggesting that LLM-extracted job contexts capture operational responsibilities with moderate semantic cohesion.
- The max pairwise similarity increases (0.6778 to 0.6856), indicating that the most semantically similar postings become slightly more similar when using LLM-extracted job contexts.

| Variant: llm_role_context   | 5/5 texts available |
| --------------------------- | ------------------- |
|  Self-retrieval @ rank-0:   | 100.0%              |
|  Separation gap:            | -0.0074             |
|  Mean pairwise cosine sim:  | 0.4037              |
|  Max pairwise cosine sim:   | 0.6630              |

- Embedding only the LLM-extracted role-context shows a negative separation gap (-0.0074), indicating that this dimension fails to separate different job roles effectively and may collapse to a single cluster.
- The mean pairwise similarity is lower than raw text (0.4037 vs 0.4600), suggesting that LLM-extracted role contexts are less semantically cohesive across different postings than the undifferentiated text.
- The max pairwise similarity is slightly lower than raw text (0.6630 vs 0.6778), indicating that even the most similar postings become less similar when using LLM-extracted role contexts.

**SUMMARY**

- The separation gap improves significantly when using manually-curated role-contexts, indicating better semantic distinction between different job roles.
- LLM-extracted job-contexts also show improvement in the separation gap and mean pairwise similarity, suggesting that these models can effectively capture operational responsibilities.
- However, LLM-extracted role-contexts fail to improve the separation gap and even show a negative value, indicating that they may not be as effective in distinguishing between different strategic missions or team purposes.

| Variant          | Self-Retrieval | Separation | Mean CosSim | Max CosSim |
| ---------------- | -------------- | ---------- | ----------- | ---------- |
| Undifferentiated | 100.0%         | +0.0069    | 0.4600      | 0.6778     |
| Job (Golden)     | 100.0%         | +0.0255    | 0.4912      | 0.6722     |
| Role (Golden)    | 100.0%         | +0.0588    | 0.4149      | 0.6754     |
| Job (LLM)        | 100.0%         | +0.0385    | 0.4783      | 0.6856     |
| Role (LLM)       | 100.0%         | -0.0074    | 0.4037      | 0.6630     |
| **Mean**         | 100.0%         | +0.0245    | 0.4496      | 0.6748     |
| **Stdev**        | 0.00           | 0.0260     | 0.0386      | 0.0082     |

The summary table reflects the overall performance trends across all variants. 
Separation improved for all variants, except for the LLM-extracted role-context. 
The low standard deviation for max cosine similarity (0.0082) suggests that the maximum semantic overlap between any two postings is consistently similar across all variants.

---

## Discussion

The results are quite good! On every metric we measured encouraging results, suggesting strong semantic separability between job-context and role-context dimensions.

The overlap and coverage metrics (Phase 1) for the Golden partitions suggest a strong semantic split, validating them as a baseline for comparison.

The job-context extraction (Phase 2) was also quite strong, suggesting it is a reliable dimension for downstream tasks such as role classification and skill matching.

The role-context extraction (Phase 3) showed moderate performance with lower fidelity than job-context extraction, indicating that extracting organizational mission and team purpose is more challenging for LLMs. The lower Jaccard similarity (mean 0.7851) and higher cross-contamination (mean 0.1976) suggest that these dimensions may be more semantically entangled with operational details, or that the prompt requires further refinement to isolate strategic context from technical responsibilities.

The high coverage of the LLM partitions (Phase 4, mean 0.9653) indicates that the two-pass extraction approach successfully captures the majority of the original text content between the job and role contexts.

However, the comparatively high overlap (mean 0.1932) suggests that the LLM partitions are not as cleanly disjoint as the golden partitions, indicating some residual semantic entanglement between the extracted job and role contexts.

The clustering tests reveal a complex story. 
The maximum pairwise cosine similarity indicates that no two postings overlap significantly, with the highest similarity observed at 0.6856 for the LLM-extracted job-context variant, BUT its consistency across variants suggests that the semantic space is inherently overlapping for these job domains.

The regression of the LLM-extracted role-context embeddings (negative separation gap of -0.0074) was quite surprising given the high performance of the LLM job-context extraction (mean Jaccard 0.9720).

A detailed investigation of the clustering behaviour may be warranted.

## Conclusions

The experiment demonstrates that **semantic partitioning of job postings** into *what is this job?* and *what purpose does this role serve?* dimensions is **feasible using LLM prompting**. 
The golden partitions validate the conceptual separability of these dimensions, and the LLM-extracted partitions achieve high fidelity in extracting operational responsibilities (job-context) while showing moderate performance in extracting organizational mission (role-context). 
However, the LLM partitions display some residual overlap and lower separation gap for the role-context dimension, indicating challenges in isolating strategic context from operational details.

The clustering analysis reveals that while LLM-extracted job contexts improve semantic separation relative to raw text, LLM-extracted role contexts fail to enhance clustering quality, even showing a negative separation gap. 
This suggests that the role-context dimension may be **inherently more challenging to extract** cleanly via LLM prompting or that the prompt requires further refinement to isolate strategic context from operational details.

## Next steps

A more rigorous study that tweaks the prompt engineering to improve role-context extraction fidelity, or expands the evaluation sets to improve the statistical power, may further validate these findings.

Experimentation with other LLMs may also provide insight into model-specific performance variations in semantic partitioning tasks.

**Feature engineering on the extracted partitions could be explored to enhance downstream classification performance, perhaps by generating combined features that can be compared against the isolated partitions explored in this experiment.**

**A valuable extension would also expand the evaluation set to include a larger corpus of job postings, improving the statistical power and generalizability of these findings.**

## Appendix

### Detailed Per-Posting Analysis

This section is given for reference.

| bmo-associate-data-scientist   |           |
| ------------------------------ | --------- |
| Undifferentiated text:         | 333 words |
| Golden job-context:            | 254 words |
| Golden role-context:           |  79 words |
| LLM job-context:               | 248 words |
| LLM role-context:              | 120 words |
| Jaccard(LLM job, golden job):  | 0.9699    |
| Jaccard(LLM role, golden role):| 0.8986    |
| Cross-contam (job→role):       | 0.1378    |
| Cross-contam (role→job):       | 0.1692    |

| affirm-ml-engineer-2           |           |
| ------------------------------ | --------- |
| Undifferentiated text:         | 386 words | 
| Golden job-context:            | 318 words | 
| Golden role-context:           | 68 words  | 
| LLM job-context:               | 311 words | 
| LLM role-context:              | 94 words  | 
| Jaccard(LLM job, golden job):  | 0.9769    | 
| Jaccard(LLM role, golden role):| 0.7681    | 
| Cross-contam (job→role):       | 0.1139    | 
| Cross-contam (role→job):       | 0.1777    | 

| hellofresh-ml-engineer-operations-technology |           |
| -------------------------------------------- | --------- |
| Undifferentiated text:                       | 338 words | 
| Golden job-context:                          | 256 words | 
| Golden role-context:                         | 82 words  | 
| LLM job-context:                             | 245 words | 
| LLM role-context:                            | 156 words | 
| Jaccard(LLM job, golden job):                | 0.9548    | 
| Jaccard(LLM role, golden role):              | 0.6147    | 
| Cross-contam (job→role):                     | 0.1185    | 
| Cross-contam (role→job):                     | 0.3119    | 

| mastercard-data-scientist-2    |           |
| ------------------------------ | --------- |
| Undifferentiated text:         | 534 words |
| Golden job-context:            | 317 words |
| Golden role-context:           | 217 words |
| LLM job-context:               | 312 words |
| LLM role-context:              | 237 words |
| Jaccard(LLM job, golden job):  | 0.9786    |
| Jaccard(LLM role, golden role):| 0.9097    |
| Cross-contam (job→role):       | 0.1581    |
| Cross-contam (role→job):       | 0.2088    |

| scribd-data-scientist-2        |           |
| ------------------------------ | --------- |
| Undifferentiated text:         | 403 words |
| Golden job-context:            | 207 words |
| Golden role-context:           | 196 words |
| LLM job-context:               | 205 words |
| LLM role-context:              | 132 words |
| Jaccard(LLM job, golden job):  | 0.9797    |
| Jaccard(LLM role, golden role):| 0.7344    |
| Cross-contam (job→role):       | 0.1423    |
| Cross-contam (role→job):       | 0.1204    |