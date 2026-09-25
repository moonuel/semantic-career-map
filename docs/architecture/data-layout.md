# Data Layout & Object Storage Naming

## 1. Buckets

Data is organized into three top-level containers -- two domain branches plus one shared
infrastructure bucket. The `eval`/`live` split separates the evaluation set from the live data. 

| Bucket | Contents | Mutability |
|---|---|---|
| `eval/` | Human ground truth — grades LLM output against author intent | Authored, versioned, never regenerated |
| `live/` | Pipeline inputs + derived artifacts | Regenerable |
| `configs/` | Config manifest — hash -> generation parameters, shared by eval and live | Append-only |

```
live/
  raw/<id>/
    source.md           # raw job descriptions
    metadata.json       # e.g. employer, date of addition, etc.
  derived/<id>/
    <variant>/v1.json   # e.g. cleaned data, engineered features, etc.
  embeddings/
    <model>/            # group embeddings by the same model
        <variant>/v1/<id>.npy

eval/
  postings/<id>/
    text/v1.json        # manually-cleaned
    <variant>/v1.json   # e.g. engineered features, labels, etc. 
  _current.json         # current eval set version

configs/
  <config-hash>.json    # e.g. model used, temperature, system prompt, etc.
```

## 2. Key Structure

### 2.1 The posting `id`

The posting `id` is the only globally unique entity key and the only cross-branch join. 
Refers to a single job posting across transformations.

- `raw/`, `derived/`, `eval/postings/` — entity-centric, `id` at depth 3.
- `embeddings/` — model-centric, `id` as the leaf; a vector is a row in a model's corpus space,
  not a posting's thing.

### 2.2 Version tokens

Data versioning across iterations is explicitly supported via ordinal version tokens (`v1`,
`v2`, etc.). 

- **`eval/`** — `vN` = the n-th human authoring of that field.
- **`live/derived/`** — `vN` = the n-th generation of that artifact; the producing config is
  recorded in the artifact's `generated_by` header, referencing `configs/<hash>.json`.

Deterministic artifacts (`sections`, `clean-text`) use the same `vN` token, so they can be
re-versioned when their logic changes.

### 2.3 Hashed configs

Model, temperature, and system prompt are deterministically converted into a hash for consistent
tracking across transformations, without polluting the object-key namespace. 

## 3. Schema principles

1. Posting `id` is globally unique across `eval/` and `live/`.
2. `eval` and `live` never cross; ground truth is authored, artifacts are generated.
3. `raw/` holds verbatim source plus its metadata; `derived/` holds anything computed from source.
4. Metadata explicitly held in accompanying objects; object keys retain what is necessary for efficient querying.
5. Iteration is additive: continuous scraping adds postings under `raw/`, and variants of derived artifacts are automatically 
separated by their object keys. 

## 4. Deferred (not this sprint)

Evaluation machinery from the [LLM-evals research](../research-reports/004-llm-evals-report.md), appended as siblings under `eval/` when needed:

```
eval/samples/        # evaluation inputs (real traces or synthetic)
eval/references/     # cached references, keyed by sample + ref-config
eval/judgments/      # judge scores, keyed by sample + output + judge + metric
eval/rubrics/        # judge rubrics, Git-versioned
```
