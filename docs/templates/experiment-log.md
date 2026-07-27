# Experiment NNN — Short Descriptive Title

**Date:** Month DD, YYYY

*One-paragraph contextual preamble: what this experiment tests, why, and what prior experiment motivated it (if any).*

**Hypothesis:** *One-sentence prediction of the expected outcome.* (Include when the experiment has a clear falsifiable claim; omit for baselines or benchmarks.)

## Experimental Design

- **Input / data source:** what text/prompts/corpus was used, how many samples, provenance.
- **Pipeline / scripts:** which scripts were run, in what order. Reference script paths inline: `` `scripts/NNN_slug/script_name.py` ``.
- **Model(s) tested:** embedding model, LLM model(s), temperature, system prompt if relevant.
- **Metrics:** define each metric with formulas or algorithm pseudocode when novel (separation gap, self-retrieval, Jaccard, boilerplate detection, hallucination detection, over-deletion rate, TTFT, tokens/s, etc.). Use code blocks for implementation notes.
- **Comparison variants:** list the variants being compared (raw, section-cleaned, LLM-cleaned, org-only, combined, etc.).
- **Controls / constants:** what was held fixed across variants (embedding model, normalization, post-processing).

*If multiple independent scripts were used, describe each in a subsection:*

### `scripts/NNN_slug/extract_clean_text.py`

*Description of what the script does, its system prompt (when LLM-based), and its output schema.*

### `scripts/NNN_slug/eval_cleaning.py`

*Description of evaluation methodology and Jaccard/boilerplate/hallucination/over-deletion detection logic.*

### `scripts/NNN_slug/compare_variants.py`

*Description of comparison logic, embedding, and metrics computation.*

## Results

*Lead with a short verdict paragraph (1–2 sentences).*

### Per-Model / Per-Variant Performance

| Metric / Variant | Variant A | Variant B | Variant C |
|---|---|---|---|
| Metric 1 | value | value | value |
| Metric 2 | value | value | value |

### Per-Sample Detail (when applicable)

| Posting | Jaccard vs Golden | Boilerplate | Hallucinations | Over-deletion |
|---|---|---|---|---|
| Posting A | 0.801 | CLEAN | CLEAN | 19.9% |
| Posting B | 1.000 | CLEAN | CLEAN | 0.0% |

### Per-Example Analysis

**Example Name**

The set intersection of the golden set and llm-cleaned text shows:

- "Preserved responsibility text 1"
- "Preserved requirement text 2"

The set difference `golden − llm` shows what the LLM removed that we decided to keep:

- "Removed contextual text 1"
- "Removed contextual text 2"

### Visualizations

```
![PCA comparison](../assets/images/exp_NNN_pca.png)
```

- Observation 1
- Observation 2

### Comparison to Previous Experiments

| Experiment | Variant | Metric |
|---|---|---|
| NNN-2 | Prior Variant | value |
| **NNN** | **New Variant** | **value** |

## Discussion

- **Does the evidence support/reject the hypothesis?** With nuance.
- **Explanation of observed effects** (e.g., why a metric increased or decreased, why a counterintuitive result occurred).
- **Contextualization against prior experiments** (how does this fit into the progressive refinement story?).
- **Limitations** (sample size, generalizability, metric limitations, confounds, lack of formal evaluation infrastructure).

## Next Steps

1. Concrete next action
2. Another next action
