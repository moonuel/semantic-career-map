# Data Layout & Object Storage Naming

!!! info "Status — Proposed (2026-09-19)"
    This note defines the target object-storage layout for migrating the local `data/` corpus to S3 as the pipeline becomes MLflow-tracked. It is a convention proposal: the bucket name, layer prefixes, and key templates are the contract; implementation (IaC, migration scripts, CI validator) follows in a later step.

    **Scope decision:** a **single bucket** with one prefix per layer, chosen for simplicity. The alternative — one bucket per layer — is deferred; because the prefix taxonomy is 1:1 with the layers, splitting later is a mechanical re-keying, not a redesign. The trade-off is recorded in [§2.1](#21-single-bucket-trade-off).

## 1. Context

The pipeline currently stores everything as local files under `data/`: raw posting Markdown, a canonical `jobs.json`, LLM run caches, derived labels and partitions, benchmark results, and a NumPy embedding matrix. That layout served experimentation but does not support the production goals:

- **Immutable inputs** — raw postings must never be mutated in place; re-scrapes and re-enrichment need a new version, not an overwrite.
- **Reproducibility** — a cleaned text file is meaningless without the cleaning method and model that produced it, and an embedding vector is meaningless without its embedding model and input variant.
- **Lineage** — `raw → cleaned → embedded → labeled` must be traceable per posting.
- **MLflow tracking** — every pipeline stage becomes a tracked run whose artifacts live at a stable S3 URI.

This note adapts the general conventions surveyed in the [Object Storage Naming Conventions research report](../research-reports/005-object-storage-naming-conventions.md) to this project's actual data.

## 2. Data Taxonomy

Data is organized into six layers under a **single bucket**, `scm-data`, one prefix per layer. Logical separation is preserved by the prefix even though the physical container is shared; per-prefix lifecycle rules and per-prefix IAM policies still apply.

```mermaid
flowchart LR
    subgraph B["s3://scm-data"]
        raw["raw/"]
        clean["cleaned/"]
        art["artifacts/"]
        emb["embeddings/"]
        meta["meta/"]
        ev["eval/"]
    end
    raw --> clean
    clean --> emb
    clean --> art
    raw --> meta
    clean --> meta
    emb --> meta
    art --> meta
    meta --> mlflow["MLflow runs"]
```

| Layer prefix | Contents | Mutability |
|---|---|---|
| `raw/` | Original posting Markdown + immutable enriched envelope (parsed sections and metadata) | Immutable — never overwrite; corrections create a new version |
| `cleaned/` | Text variants (`raw`, `regex`, `llm`, `llm-org-context`); cleaning **method and model in the key** | Write-once per method + model |
| `artifacts/` | Taxonomy labels, generated titles, semantic partitions, benchmark results | Write-once per run; versioned by taxonomy/model |
| `embeddings/` | Vector files (per-posting `.npy` + matrix bundles); **embedding model in the key** | Write-once per model + input variant |
| `meta/` | Lineage manifests, model cards, cleaner cards, dataset registry, MLflow pointers | Mutable (small) |
| `eval/` | Golden set (manual), QA diff reports | Mostly immutable |

The four layers the pipeline requires at minimum — raw, cleaned, artifacts, embeddings — are covered directly. `meta/` is what makes cleaning methods and embedding models durable and discoverable; `eval/` isolates hand-maintained ground truth from generated data.

### 2.1 Single Bucket Trade-off

Putting every layer in one bucket means bucket-level settings apply to all layers simultaneously:

- **Versioning, encryption, and default retention are bucket-wide.** This is acceptable here — every layer benefits from versioning and SSE-KMS.
- **Object Lock cannot be scoped to a prefix.** If raw data ever requires regulatory WORM immutability, Object Lock would lock the entire bucket. If that requirement materializes, split `raw/` into its own bucket; until then, "immutable" is enforced by convention (write-once keys + versioning) rather than by lock.
- **Lifecycle rules remain prefix-scoped.** S3 lifecycle rules accept prefix filters, so `artifacts/benchmarks/` and `eval/diffs/` can still expire on their own schedules while `raw/` and `cleaned/` never expire.
- **Permissions remain prefix-scoped.** IAM policies and SCP conditions can constrain `s3:PutObject` and `s3:ListBucket` to a prefix, so least-privilege per pipeline stage is preserved.
- **One global name to secure.** A single bucket also means a single globally unique, publicly probeable name to protect and monitor.

The operational win is simplicity: one bucket to provision, one set of access policies to attach, one target for migration scripts, and no risk of writing a layer to the wrong bucket.

## 3. Bucket Naming

A single bucket follows the general template `<org>-<purpose>[-<region>][-<env>]`:

```
scm-data
```

Rules (enforced by the portability constraints in report §4.1 and provider docs):

- Lowercase letters, digits, and hyphens only; 3–63 characters; start and end alphanumeric.
- No periods (breaks HTTPS virtual-host addressing), no underscores, no IP-shaped names.
- `scm` is the project prefix; append region/account/env (`scm-data-useast1-<acct>-prod`) when the deployment becomes multi-region or multi-account.
- Bucket names are **globally unique and publicly probeable**: if `scm-data` is taken, qualify it (`scm-data-<acct>`) or use an account-regional namespace — do not add PII or sensitive business terms.
- The bucket name carries no layer information; the layer lives in the prefix.

## 4. Object Key Naming

Every key begins with its **layer prefix**, then reads **leftmost-general → rightmost-specific**. The dimensions that matter for querying — the data variant, then the method/model that produced it — follow the layer. Posting identity (the slug) is the stable entity. Date partitions use `key=value` pairs (`day=YYYY-MM-DD`, per report §3.2) and are reserved for genuinely time-ordered data.

### 4.1 Raw — `raw/`

```
scm-data/raw/postings/<posting_id>.md               # original scrape, immutable
scm-data/raw/postings/<posting_id>.json             # immutable envelope: sections + provenance metadata
```

The envelope JSON carries the fields that must not leak into key names: `source_url`, `scraped_at` (ISO-8601), `content_hash` (SHA-256 of the `.md`), `company`, `title_raw`, `role_category`. Both objects are write-once; a re-scrape or correction produces a new object (versioning records the history).

### 4.2 Cleaned — `cleaned/`

```
scm-data/cleaned/<method>/<model>/<posting_id>.json
```

| Method segment | Meaning |
|---|---|
| `raw` | Identity baseline — no cleaning (embedding input for the raw baseline) |
| `regex` | Regex boilerplate removal (Experiment 002); model segment is `<none>` |
| `llm` | LLM text extraction (Experiment 003) |
| `llm-org-context` | LLM organizational-context summary |

Examples:

```
scm-data/cleaned/raw/<none>/bmo-associate-data-scientist.json
scm-data/cleaned/regex/<none>/bmo-associate-data-scientist.json
scm-data/cleaned/llm/openai--gpt-5.4-nano/bmo-associate-data-scientist.json
scm-data/cleaned/llm-org-context/openai--gpt-5.4-nano/bmo-associate-data-scientist.json
```

Each JSON body records: `posting_id`, `method`, `model`, `prompt_hash`, `raw_hash`, `cleaned_at`, `usage` (token counts), and the `text`.

### 4.3 Artifacts — `artifacts/`

```
scm-data/artifacts/labels/<taxonomy-version>/<model>/<posting_id>.json
scm-data/artifacts/titles/<taxonomy-version>/<model>/<posting_id>.json
scm-data/artifacts/partitions/<method>/<model>/<posting_id>.json
scm-data/artifacts/benchmarks/<experiment>/<day=YYYY-MM-DD>/<model>/<run>.json
```

Embedding the taxonomy version in the key means re-labeling under a new taxonomy never clobbers existing labels:

```
scm-data/artifacts/labels/ml-ai-v13/openai--gpt-5.4-nano/affirm-ai-solutions-engineer.json
scm-data/artifacts/partitions/llm/openai--gpt-5.4-nano/bmo-associate-data-scientist.json
scm-data/artifacts/benchmarks/004-speed-bench/day=2026-07-27/openai--gpt-5.4-nano/run-0.json
```

### 4.4 Embeddings — `embeddings/`

```
scm-data/embeddings/<model>/<input-variant>/<posting_id>.npy
scm-data/embeddings/<model>/<input-variant>/matrix/<day=YYYY-MM-DD>/<dataset>-<n>-postings.npy
```

The `input-variant` mirrors the cleaned variant (`raw`, `regex`, `llm-openai--gpt-5.4-nano`), so the cleaning → embedding lineage is readable from the key alone. Per-posting vectors are the canonical incremental form; the matrix bundle is the convenience artifact logged to MLflow for a full pipeline run.

```
scm-data/embeddings/sentence-transformers--all-MiniLM-L6-v2/llm-openai--gpt-5.4-nano/bmo-associate-data-scientist.npy
scm-data/embeddings/sentence-transformers--all-MiniLM-L6-v2/raw/matrix/day=2026-07-22/raw-27-postings.npy
```

**Model-id normalization:** replace `/` and `:` with `--` so a model identifier stays a single valid path segment (`openai/gpt-5.4-nano` → `openai--gpt-5.4-nano`). The canonical, unmodified model string is stored in the object body and in the model card.

### 4.5 Metadata & lineage — `meta/`

```
scm-data/meta/models/<model-id>.json                 # embedding model card (dim, tokenizer, normalization)
scm-data/meta/methods/<method>-<model>.json          # cleaning method card (prompt hash, params)
scm-data/meta/datasets/<day=YYYY-MM-DD>/manifest.json # snapshot contents + content hashes
scm-data/meta/provenance/<posting_id>.json           # raw_hash → clean variant → embedding → labels
scm-data/meta/mlflow/<experiment>/<run_id>.json      # pointer between S3 URIs and MLflow runs
```

### 4.6 Evaluation — `eval/`

```
scm-data/eval/golden/partitions/<posting_id>.json    # hand-maintained golden set
scm-data/eval/diffs/<method>/<model>/<posting_id>.txt # raw-vs-clean QA reports
```

## 5. Documenting Cleaning Methods and Embedding Models

The requirement "the method of cleaning needs to be documented" and "the embedding model also needs to be documented" is satisfied in three redundant places, so provenance survives independently of any one record:

1. **In the key** — method/model are path segments, so a plain `aws s3 ls` reveals how an artifact was produced.
2. **In the object body and S3 object metadata** — `method`, `model`, `prompt_hash`, `raw_hash`, `cleaned_at`, `usage`.
3. **In `meta/` cards** — the cleaner card holds the full prompt and parameters; the model card holds dimensions, tokenizer, and normalization. Both are referenced by MLflow run parameters.

A content-addressed cache key (`raw_hash` + `prompt_hash` + `model`) makes re-runs idempotent and lets a manifest prove that a stored cleaned text corresponds to a specific raw input.

## 6. Migration Mapping

| Current local artifact | Target S3 key |
|---|---|
| `data/selected-job-postings/<file>.md` | `scm-data/raw/postings/<posting_id>.md` |
| `data/jobs.json` | Per-posting envelopes `scm-data/raw/postings/<posting_id>.json` + `scm-data/meta/datasets/<day>/manifest.json` |
| `data/.llm_clean_cache.json` / `_v2.json` | `scm-data/cleaned/llm/openai--gpt-5.4-nano/<posting_id>.json` (v1/v2 → distinct method cards) |
| `data/.llm_org_context_cache.json` | `scm-data/cleaned/llm-org-context/openai--gpt-5.4-nano/<posting_id>.json` |
| `data/005_semantic_partitions.json` | `scm-data/artifacts/partitions/llm/openai--gpt-5.4-nano/<posting_id>.json` |
| `data/006_taxonomy_labels.json` | `scm-data/artifacts/labels/ml-ai-v13/openai--gpt-5.4-nano/<posting_id>.json` |
| `data/golden_cleaned.json` | `scm-data/eval/golden/partitions/<posting_id>.json` |
| `data/004_benchmark_results.json` | `scm-data/artifacts/benchmarks/004-speed-bench/day=2026-07-27/<model>/run-<n>.json` |
| `data/raw_embeddings.npy` | `scm-data/embeddings/sentence-transformers--all-MiniLM-L6-v2/raw/matrix/day=2026-07-22/raw-27-postings.npy` |
| `data/.diff_clean.txt` / `.diff_raw.txt` | `scm-data/eval/diffs/<method>/<model>/<posting_id>.txt` |
| `data/plots/*.png` | Not migrated — experiment output; log as MLflow run artifacts |
| `data/.llm_*cache*.json`, `data/.005/006_*cache.json` | Not migrated — ephemeral API caches; keep local or in a scratch prefix |

!!! warning "Reconcile the corpus before migrating"
    Raw posting files and the canonical dataset can drift. At the time of writing, `data/selected-job-postings/` contains files not yet present in `data/jobs.json` (for example the `rbc-ai-engineer` and `scotiabank-ai-software-engineer` postings). Manifest generation must report this reconciliation explicitly so every raw posting is either ingested or consciously excluded — never silently dropped.

## 7. MLflow Alignment

Keep S3 key segments and MLflow run parameters **identical and greppable** so an artifact URI or a run ID each resolves to the other:

| MLflow parameter | S3 key segment |
|---|---|
| `cleaning_method` | `scm-data/cleaned/<method>/` |
| `cleaning_model` | `scm-data/cleaned/<method>/<model>/` |
| `embedding_model` | `scm-data/embeddings/<model>/` |
| `input_variant` | `scm-data/embeddings/<model>/<input-variant>/` |
| `taxonomy_version` | `scm-data/artifacts/labels/<taxonomy-version>/` |

Each MLflow run logs its produced objects as artifacts and writes a pointer to `scm-data/meta/mlflow/<experiment>/<run_id>.json`. This makes any embedding or label reconstructable from the tracking server plus the data lake, with no reliance on local files.

## 8. Lifecycle & Retention

Versioning and encryption are bucket-wide on `scm-data` (ON for every layer). Retention is applied with **prefix-scoped lifecycle rules**:

| Prefix | Retention / lifecycle |
|---|---|
| `raw/` | No expiration; transition to cold storage after several years |
| `cleaned/` | No expiration — reproducibility depends on it |
| `artifacts/` | Prefix transitions for `artifacts/benchmarks/` only |
| `embeddings/` | No expiration while the model is supported; archive when a model is retired |
| `meta/` | Retain cards, manifests, provenance permanently; expire scratch prefixes |
| `eval/` | Golden set immutable; `eval/diffs/` may expire by prefix |

Object Lock (WORM) is the one setting that cannot be scoped to a prefix; see [§2.1](#21-single-bucket-trade-off) for the split-if-required path.

## 9. Security

- **Names are public.** Keys contain only lowercase slugs (company + role) — no URLs, emails, or PII. Sensitive metadata lives in the envelope body and object metadata.
- **Block public access** on the bucket; encrypt at rest (SSE-KMS) and in transit.
- **Least privilege by prefix.** Grant each pipeline stage write access only to its layer prefix (ingestion → `raw/`, cleaning → `cleaned/`, embedding → `embeddings/`). Use `s3:prefix` conditions for `ListBucket` and prefix-scoped resource ARNs for object actions.
- **Enforce the standard at creation time.** A CI validator checks bucket/key strings against the templates in this note; in a multi-account setup, IAM/SCP conditions can constrain `s3:PutObject` to the permitted prefixes.

## 10. Scalability

At the current corpus size (tens of postings) no performance-specific key tricks are needed. A single bucket does not change S3 request-rate behavior — throughput scales per prefix, not per bucket — and it avoids the default per-account bucket quota. The forward-looking decisions already encoded here:

- Time-ordered, append-heavy data (`artifacts/benchmarks/`, `eval/diffs/`, caches) is naturally spread by its date prefix, matching report §5.2.
- If ingestion ever streams at high request rates to a single prefix, prepend a short entropy hash before the logical key — **except** in S3 directory buckets / hierarchical namespaces, where entropy prefixes are counterproductive (report §5.3).
- Per-posting objects keep writes independent and incremental, so adding one posting never rewrites a shared blob.

## 11. Validation Checklist

Before creating the bucket or writing a key:

- [ ] Bucket is lowercase, hyphenated, 3–63 chars, starts/ends alphanumeric, globally unique
- [ ] Key starts with a layer prefix (`raw/`, `cleaned/`, `artifacts/`, `embeddings/`, `meta/`, `eval/`)
- [ ] Key follows `<layer>/<domain>/<variant>/<model>/<posting_id>.<ext>` where applicable
- [ ] Method, model, and taxonomy version are present in the key where applicable
- [ ] Model ids normalized (`/`, `:` → `--`); canonical string stored in the body
- [ ] No PII, secrets, or URLs in bucket or key names
- [ ] Date partitions only on time-ordered data, in `key=value` form
- [ ] Object is immutably written; corrections create a new version
- [ ] `meta/` manifest updated and MLflow parameters match the key segments

## 12. References

1. [Object Storage Naming Conventions research report](../research-reports/005-object-storage-naming-conventions.md) — provider rules and general best practices this layout adapts
2. [Design Decisions](design-decisions.md) — pipeline architecture the layout serves
3. [Experiment Log](../experiments/index.md) — provenance of cleaned text, partitions, and taxonomy labels
