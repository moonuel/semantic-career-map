# Experiment 005 — Semantic Partitioning of Job Postings

**Date:** July 27, 2026

Experiment 003 noted that the LLM cleaning prompt removed team-context and organizational-role descriptions from job postings, and that this removed content might carry semantically meaningful signal. 

This experiment tests whether the golden set text can be semantically partitioned into "What is this job?" and "What purpose does this role serve?".


**Hypothesis:** The two partitions — *what a job is* and *what purpose the role serves* — are semantically separable dimensions that can be extracted via LLM prompting.

---

**NOTE:** This experiment assumes cleaned text via `golden_cleaned.json`. The future pipeline will implement a boilerplate removal pre-processing step (to be tested separately) to produce artifacts similar to `golden_cleaned.json`.

In the future we will also want to isolate each signal and test whether they useful for semantic clustering on their own, combined, or eventually used to synthesize a more semantically meaningful description.


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

Three scripts: `scripts/005_semantic_partitioning/extract_fields.py`, `scripts/005_semantic_partitioning/eval_partition.py`, and `scripts/005_semantic_partitioning/compare_embeddings.py`.