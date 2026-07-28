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

5. **Embedding comparison (secondary):** Embed all five variants with `all-MiniLM-L6-v2` and compute separation gap and self-retrieval. Compare `job-context` vs `role-context` to test whether the two dimensions produce different clustering behavior. See future-work note regarding expanding clustering evaluation beyond this initial pass.

## Results

