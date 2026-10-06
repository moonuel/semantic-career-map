# Data Layout & Object Storage Naming

!!! info "Data architecture under construction!"
  Currently migrating from the legacy object-per-posting store to the lakehouse layout described below. The [`keys.py`](../../backend/storage/keys.py) builders still emit legacy keys (§11) until per-branch builders land.


The lakehouse is the immutable, versioned source of truth for raw job-posting data, and the layer every derived dataset is reproducibly built from.

The design targets **S3 + Glue Data Catalog + Athena**: raw data is preserved verbatim as **JSON**, everything queryable becomes **Parquet**, and directories use **Hive-style partitioning** so Athena prunes scans instead of reading the whole lake. A single bucket is used for now (until scope justifies expansion into separate buckets), with the branches below as prefixes.

The core principle:

> Raw data is immutable; derived datasets are reproducible; Athena tables expose logical datasets independently of their physical storage layout.

The object keys stop being the query interface. They carry the physical layout; identity, provenance, and configuration move into row metadata and manifests; queries hit tables instead of prefixes.

Full rationale: [Athena Lakehouse Proposal](proposed-athena-layout.md). Storage-side and metadata-side research: reports [009](../research-reports/009-data-lake-design-best-practices.md), [010](../research-reports/010-data-warehouse-design-best-practices.md), and [011](../research-reports/011-object-store-metadata-design.md).

## 1. Branches

Data is organized into five top-level branches -- one immutable source layer, three regenerable data layers, and one append-only metadata layer.

| Branch | Contents | Format | Mutability |
|---|---|---|---|
| `raw/` | Verbatim payloads, content-addressed | JSON | Immutable |
| `artifacts/` | Derived per-posting artifacts (e.g. cleaned data, engineered features, labels) | Parquet | Regenerable |
| `embeddings/` | Vector datasets, grouped by the same model | Parquet | Rebuildable from source |
| `eval/` | Human ground truth -- grades LLM output against author intent, with datasets and results kept separate | Parquet | Authored, versioned, never regenerated |
| `manifests/` | Dataset provenance, ingestion events, pipeline runs | JSON | Append-only |

```
lake/                          # the single bucket, for now
  raw/
    source=<source>/
      ingested_on=<YYYY-MM-DD>/
        sha256=<content-hash>.json       # verbatim payload, content-addressed
  artifacts/
    type=<artifact-type>/
      variant=<variant>/
        version=<vN>/
          part-00000.parquet            # one artifact per row
  embeddings/
    model=<model>/
      variant=<variant>/
        version=<vN>/
          part-00000.parquet            # one posting per row
  eval/
    datasets/
      type=<type>/
        variant=<variant>/
          version=<vN>/
            part-00000.parquet          # authored ground truth
    results/
      type=<type>/
        variant=<variant>/
          version=<vN>/
            part-00000.parquet          # machine output over a dataset
  manifests/
    datasets/                           # one manifest per dataset generation
    ingestion/                          # one entry per ingestion event
    runs/                               # pipeline-run records
```

The former `eval`/`live` split survives as the `eval/` prefix -- authored ground truth never crosses with generated data. The former `configs/` branch is absorbed into `manifests/` (§3).

## 2. Object Keys

Keys carry what must be **listed and pruned by prefix** -- everything else is metadata. The discipline from the legacy layout carries over: a structured key builder ([`keys.py`](../../backend/storage/keys.py)) stays the single source of truth for well-formed keys, pure and I/O-free, and every other module derives keys through it rather than by string interpolation. Its `_SEGMENT_RE` already anticipated hive-style `key=value` pairs, so the new branch prefixes arrive the way the old ones did -- builder functions first, tests beside them.

### 2.1 Content identity

Raw objects are named by content, not by arrival: `sha256=<hash>.json` is the identity of the *received payload*, while the *ingestion event* gets its own identity (`ingestion_id`, recorded in raw metadata and `manifests/ingestion/`). The same content observed repeatedly is therefore not a new document version -- the object already exists, and only gains another observation.

Exact-byte hashing is sometimes too exact -- a trailing space should not mint a new identity. Once canonicalization rules exist, keep both a `raw_hash` (exact received bytes, the address) and a `canonical_content_hash` (semantically equivalent canonicalized content).

### 2.2 Posting ids

The posting `<id>` stays a time-ordered **UUIDv7** (RFC 9562 §5.7) minted once at ingestion -- lexicographic order equals chronological order, which keeps exports sortable for free. What changes is placement: `posting_id` stops being the load-bearing path segment and becomes a row column, the join handle tying one posting's artifact row, embedding row, and eval rows back together. Ids are still minted by exactly one generator in `keys.py`.

### 2.3 Partitions

Directory levels are Hive-style `<column>=<value>` pairs, which become Athena partition columns (e.g. `source=linkedin/`, `ingested_on=2026-10-05/`). Partition on low-to-moderate-cardinality dimensions that queries actually filter on -- `source`, `ingested_on`, `event_date`. Never partition on high-cardinality identifiers such as `posting_id` or `content_hash`: an address identifies an object, a partition prunes a scan. Partitioning exists to reduce what Athena reads, not to encode application identity into the directory tree.

### 2.4 Versions

`vN` identifies a transformation or dataset definition, not individual records. `responsibility_text/version=v2` may be a different extraction method over the same inputs; `embeddings/.../version=v2` may mean a different model or preprocessing configuration. Data science and ML engineering is an iterative process, and we should store artifacts from each iteration separately -- versions add, they don't overwrite, and a version never silently replaces an incompatible dataset.

## 3. Configuration & Manifests

MLflow for experiment tracking and versioning is still planned. In the meantime, generation parameters are recorded in `manifests/`, setting up the eventual migration -- the append-only `configs/<hash>.json` objects carry over in spirit: a configuration (the LLM used and at what temperature, the system prompt, the embedding model, clustering parameters) is deterministic enough to be identified by its 64-char hex SHA-256 `configuration_hash`, referenced from rows and manifests instead of polluting the key namespace.

A dataset manifest records what a dataset is and where it came from:

```
dataset_id            # e.g. embeddings-v17
input_dataset         # e.g. responsibility-text-v12
model, model_version
pipeline_version      # e.g. git sha
configuration_hash    # sha256:...
record_count
created_at
```

A pipeline-run manifest records the run itself:

```
run_id, pipeline_version
started_at, completed_at
input_snapshot, output_snapshot
configuration_hash
status
```

Between the two, any dataset is traceable to its exact inputs, code, and configuration -- reproducibility that doesn't fit in an S3 path.

## 4. Deduplication

Exact duplicates resolve at write time: identical payloads hash to the same raw object, so a re-ingestion is a no-op in storage and just one more record under `manifests/ingestion/`. Whether the same content means *the same document or a new version* remains a policy decision (§2.1), not a storage decision. Data engineering has tolerance for errors as long as they are observable, and re-observation is now cheap by construction.

### 4.1 Semantic deduplication (deferred)

A content hash still won't catch the same job posting collected twice but posted on different dates, or ingested with noise a naive hash can't normalize. The semantic-similarity filtering tool stays deferred -- for now, ingestion raises an identical-entry alarm on the employer and job title metadata, keeping accumulation observable rather than governed (the un-governed accumulation anti-pattern report 009 warns about).

## 5. Provenance & Lineage

Every derived artifact should identify its inputs and transformation. The chain is:

```
raw posting ──► normalized posting ──► responsibility text ──► embedding
```

Row-level metadata reconstructs each hop:

```
artifact_id, source_hash, input_artifact, input_version,
pipeline_version, configuration_hash, created_at
```

The per-posting `metadata.json` sidecar (still written by today's legacy ingestion) is what this grows out of -- provenance moves from one sidecar per posting into row columns and dataset manifests, which is the shape a relational catalog can actually query (§9).

## 6. Queryable Datasets (Parquet)

At tens of postings, one object per posting was fine -- good, even: simple, inspectable, greppable. It does not survive contact with Athena. Millions of tiny objects are the classic small-file anti-pattern (report 009), and every full-dataset query degrades into listing the whole store. So every derived layer is a Parquet dataset: an artifact is a **row**, the part files are implementation details, and Athena queries millions of artifacts without touching millions of files. Parquet buys compression, column projection, and predicate pushdown along the way, and schemas stay written down at dataset-definition time -- the schema-on-write habit, kept.

A row under `type=responsibility_text/variant=llm/version=v1`:

```
artifact_id, source_hash, posting_id, variant, version,
responsibility_text, pipeline_version, created_at
```

Raw data stays JSON -- verbatim source preservation beats columnar convenience.

## 7. Embeddings

Embeddings become datasets too, still grouped model-first -- a vector is a row in a model's corpus space:

```
model=<model>/variant=<variant>/version=<vN>/
```

with rows of:

```
posting_id, source_hash, embedding, model, model_version,
input_artifact, input_version, embedding_dimension, normalization, created_at
```

The dataset is the authoritative representation of the vectors. Any vector index (FAISS, or whatever scale eventually demands) is a rebuildable acceleration structure derived from it -- never the source of truth. One `<id>.npy` per posting exits with the legacy layout.

## 8. Evaluation Data

`eval/` separates what is authored from what is generated, the same discipline the old `eval/` branch enforced -- it just draws the line twice. `datasets/` holds human ground truth, versioned by authorship. `results/` holds machine output over those datasets, regenerated whenever a different model, judge, or prompt evaluates them.

A dataset row:

```
posting_id, query, expected_relevance, annotation, annotator
```

A results row:

```
posting_id, prediction, score, judge_model, prompt_version,
evaluation_version, created_at
```

The same evaluation set can therefore be evaluated repeatedly under different assumptions, which is the whole point. This absorbs the evaluation machinery deferred from the [LLM-evals research](../research-reports/004-llm-evals-report.md): samples and references land as datasets, judgments as results, and rubrics stay Git-versioned. The `_current.json` pointer gives way to explicit `version=vN` addressing and manifest-pinned snapshots (§10).

## 9. Catalog & Athena Logical Tables

The Glue Data Catalog provides the logical schema over the physical datasets, and Athena tables are the stable interface:

```
raw.job_postings
artifacts.responsibility_text
artifacts.skills
embeddings.job_posting_embeddings
eval.career_relevance
eval.career_relevance_results
```

A logical table maps onto S3 locations regardless of how they are partitioned underneath, so a physical repartition never breaks a query -- nothing downstream references the prefixes in §1. The directory structure is not the primary API; the tables are.

The object store remains the system of record; Athena is the query and serving layer over it (reports 009/010 draw this boundary in detail).

## 10. Serving Snapshots

The application should not consume the bucket directly -- it consumes a coherent, rebuildable **snapshot** of lakehouse datasets, pinned by a publication manifest:

```json
{
  "snapshot_id": "2026-10-05",
  "posting_dataset": "postings-v42",
  "embedding_dataset": "embeddings-v17",
  "cluster_dataset": "clusters-v8",
  "coordinate_dataset": "umap-v5",
  "published_at": "2026-10-05T18:00:00Z"
}
```

Indexes and serving objects are disposable, and are rebuilt from whichever datasets the manifest pins.

## 11. Legacy Layout

Everything this page described before the lakehouse rewrite -- and everything [`keys.py`](../../backend/storage/keys.py) still implements today -- is the legacy object-per-posting store:

```
live/
  raw/<id>/
    source.md           # verbatim source
    metadata.json       # e.g. employer, date of addition, etc.
  derived/<id>/
    <variant>/v1.json   # e.g. cleaned data, engineered features
  embeddings/
    <model>/<variant>/v1/<id>.npy
configs/
  <config-hash>.json

eval/
  postings/<id>/
    text/v1.json        # manually-cleaned
    <variant>/v1.json
  _current.json         # current eval set version
```

It is retrieval-first -- everything about one posting is one prefix away -- which was fine at the current dataset size. It fails two ways at lakehouse scale: per-posting JSON objects become the small-file anti-pattern (report 009), and queries across all postings degrade into recursive listing rather than a pruned scan.

| Concept | Legacy | Lakehouse |
|---|---|---|
| Raw address | `live/raw/<id>/source.md` | content-addressed JSON (§2.1) |
| Posting identity | `<id>` path segment | `posting_id` row column (§2.2) |
| Query interface | key prefix | Glue catalog + Athena tables (§9) |
| One artifact | one JSON object | one Parquet row (§6) |
| One embedding | one `<id>.npy` | vector column (§7) |
| Config record | `configs/<hash>.json` | `configuration_hash` + manifests (§3) |
| Eval-set pin | `eval/_current.json` | explicit version pins (§8, §10) |

Legacy objects aren't converted -- derived data is supposed to be regenerable, so only raw payloads are re-addressed by content hash and everything else re-generates as Parquet when its pipeline stage is formalized. Migration is branch-by-branch, key builders first. The old §A.1 hunch -- that the metadata sidecar may eventually expand into an Athena SQL database -- is answered by §9.
