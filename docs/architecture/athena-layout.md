# Athena Lakehouse Layout

> Status: **live.** The layout below is the implemented lakehouse and query layer. The document originated as a proposal; the live page keeps the design rationale and records the amendments made during implementation.

The lakehouse is an immutable, versioned source of truth for raw job-posting data and reproducible derived datasets.

The design is optimized for **Amazon S3 + AWS Glue Data Catalog + Amazon Athena**, using **Parquet for queryable datasets** and **Hive-style partitioning** for partition pruning.

The core principle is:

> Raw data is immutable; derived datasets are reproducible; Athena tables expose logical datasets independently of their physical storage layout.

## 1. Architecture

```text
                    ┌──────────────────────┐
                    │       Sources        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     S3 Raw Layer     │
                    │   Immutable JSON     │
                    └──────────┬───────────┘
                               │
                         transformation
                               │
                               ▼
                    ┌──────────────────────┐
                    │   S3 Data Layer      │
                    │   Parquet datasets   │
                    └──────────┬───────────┘
                               │
                         Glue Data Catalog
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Athena         │
                    └──────┬───────┬───────┘
                           │       │
                          SQL     analytics
                           │
                    ┌──────┴───────┐
                    ▼              ▼
                  Gold          Evaluation
                    │
                    ▼
              Serving / ML
```

## 2. S3 layout

!!! warning "Amendment — no `ingested_on` in raw keys"
  As adopted in [data.md §2.1](data.md#21-content-identity), the raw layer drops the `ingested_on=<YYYY-MM-DD>/` level shown below: a date in the path would give identical re-ingested payloads different keys and accumulate copies, defeating content-addressing within a source. Objects are never rewritten, so S3 `Last-Modified` is the first-ingestion time, and provenance dates for date-range queries live in manifests instead. The tree below is the original design tree, kept for migration history.

```text
lake/
├── raw/
│   └── source=<source>/
│       └── ingested_on=<YYYY-MM-DD>/
│           └── sha256=<raw-content-hash>.json
│
├── artifacts/
│   └── type=<artifact-type>/
│       └── variant=<variant>/
│           └── version=<version>/
│               ├── part-00000.parquet
│               ├── part-00001.parquet
│               └── ...
│
├── embeddings/
│   └── model=<model>/
│       └── variant=<variant>/
│           └── version=<version>/
│               ├── part-00000.parquet
│               └── ...
│
├── eval/
│   ├── datasets/
│   │   └── type=<type>/
│   │       └── variant=<variant>/
│   │           └── version=<version>/
│   │               ├── part-00000.parquet
│   │               └── ...
│   │
│   └── results/
│       └── type=<type>/
│           └── variant=<variant>/
│               └── version=<version>/
│                   ├── part-00000.parquet
│                   └── ...
│
└── manifests/
    ├── datasets/
    ├── ingestion/
    └── runs/
```

## 3. Raw data

Raw data preserves the exact payload received from external sources.

```text
raw/
└── source=linkedin/
    └── ingested_on=2026-10-05/
        └── sha256=abc123....json
```

Raw objects are immutable.

The SHA-256 hash provides content identity and allows duplicate payloads to be detected across ingestion runs.

Raw metadata should include:

```json
{
  "content_hash": "sha256:abc123...",
  "source": "linkedin",
  "source_document_id": "...",
  "ingestion_id": "...",
  "ingested_at": "2026-10-05T18:00:00Z",
  "observed_at": "2026-10-05T17:59:42Z"
}
```

### Content identity vs. ingestion identity

These represent different concepts:

```text
content_hash
    └── identity of the received content

ingestion_id
    └── identity of one ingestion event
```

The same content can therefore be observed repeatedly without being treated as a new document version.

Where appropriate, maintain both:

```text
raw_hash
canonical_content_hash
```

`raw_hash` identifies the exact received bytes, while `canonical_content_hash` identifies semantically equivalent canonicalized content.

## 4. Hive-style partitioning

Partition directories use the form:

```text
<column>=<value>
```

For example:

```text
source=linkedin/
ingested_on=2026-10-05/
```

These become Athena partition columns.

Partition primarily on attributes that are commonly used for filtering and have reasonable cardinality.

Good candidates:

```text
source
ingested_on
event_date
```

Avoid partitioning on high-cardinality identifiers such as:

```text
source_hash
uuid
content_hash
```

The hash identifies the object but should not be used as a partition.

The partition hierarchy should also avoid excessive partition counts. The purpose of partitioning is to reduce the amount of data Athena must scan, not to encode every attribute into the directory structure.

## 5. Queryable derived datasets

Derived artifacts should generally be stored as **Parquet datasets**, rather than one JSON file per artifact.

For example, the live cleaning artifact:

```text
artifacts/
└── type=cleaned_text/
    └── variant=llm/
        └── version=v1/
            ├── part-00000.parquet
            ├── part-00001.parquet
            └── ...
```

The individual artifact is represented as a row:

```text
source_hash
variant
version
cleaned_text
configuration_hash
pipeline_version
created_at
```

The physical files are implementation details.

This allows Athena to efficiently query millions of artifacts without requiring millions of individual files.

## 6. Artifact lineage

Every derived artifact should identify its inputs and transformation.

For example:

```text
raw posting
    │
    ▼
canonical payload
    │
    ▼
cleaned text
    │
    ▼
embedding
```

A derived dataset contains enough metadata to reconstruct this lineage:

```text
source_hash
input_artifact
input_version
pipeline_version
configuration_hash
created_at
```

The artifact path identifies the dataset class:

```text
type=cleaned_text/
variant=llm/
version=v1/
```

while row-level metadata identifies the specific source and transformation.

## 7. Embeddings

Embeddings are stored as Parquet datasets:

```text
embeddings/
└── model=amazon-titan-embed-text-v2/
    └── variant=titan-v2/
        └── version=v1/
            ├── part-00000.parquet
            └── ...
```

A record contains:

```text
source_hash
embedding
model
model_version
input_artifact
input_version
embedding_dimension
normalization
created_at
configuration_hash
run_id
```

The embedding dataset is the authoritative representation of the embeddings.

A vector index is a rebuildable acceleration structure derived from this dataset:

```text
embedding dataset
       │
       ├── Athena / analytics
       │
       └── vector index
```

Athena does not need to be responsible for nearest-neighbor search.

## 8. Evaluation datasets

Evaluation datasets and evaluation results are separate.

```text
eval/
├── datasets/
└── results/
```

For example, the live cleaning golden set:

```text
eval/datasets/
└── type=cleaning_golden/
    └── variant=author/
        └── version=v1/
            └── part-00000.parquet
```

The dataset contains:

```text
source
source_hash
source_id
golden_text
annotator
created_at
```

Results are stored separately:

```text
eval/results/
└── type=cleaning_quality/
    └── variant=exp003/
        └── version=v2/
            └── part-00000.parquet
```

with:

```text
source_hash
jaccard
boilerplate_hits
hallucination_count
over_deletion_pct
over_deletion_flagged
evaluation_configuration_hash
evaluation_version
created_at
```

A future relevance-judging dataset would use the same dataset/result split with `query`, `expected_relevance`, `score`, and `judge_model` columns. This allows the same evaluation set to be evaluated repeatedly with different models or evaluation procedures.

## 9. Glue Data Catalog

The Glue Data Catalog provides the logical schema over the physical S3 datasets.

Example Athena tables:

```text
raw_ashby, raw_dataroles, raw_greenhouse, raw_lever,
raw_hiringcafe, raw_linkedin
manifests_ingestion
artifacts_cleaned
postings_embeddings
```

The catalog maps these logical tables to S3 locations.

For example:

```text
artifacts_cleaned
        │
        ▼
s3://<bucket>/artifacts/type=cleaned_text/
        │
        ├── variant=llm/
        │     └── version=v1/
        │
        └── variant=heuristic/
              └── version=v2/
```

The physical directory structure should not be treated as the primary API. Athena tables provide the stable logical interface.

## 10. Parquet

JSON is retained for raw source preservation.

Parquet is preferred for queryable datasets because it provides:

* columnar storage
* compression
* predicate pushdown
* column projection
* efficient Athena scans
* efficient analytical queries

Therefore:

```text
raw source data       → JSON
derived document data → Parquet
tabular artifacts     → Parquet
embeddings            → Parquet
evaluation data       → Parquet
evaluation results    → Parquet
```

## 11. Dataset manifests

Dataset manifests provide dataset-level provenance.

Example:

```json
{
  "dataset_id": "embeddings-v17",
  "input_dataset": "cleaned-text-v1",
  "model": "amazon-titan-embed-text-v2",
  "model_version": "amazon.titan-embed-text-v2:0",
  "variant": "titan-v2",
  "version": "v1",
  "pipeline_version": "git-sha",
  "configuration_hash": "sha256:...",
  "record_count": 18234,
  "created_at": "2026-10-05T18:00:00Z"
}
```

Pipeline-run manifests should record:

```text
run_id
pipeline_version
started_at
completed_at
input_snapshot
output_snapshot
configuration_hash
status
```

These provide reproducibility beyond what can conveniently be encoded in S3 paths.

## 12. Dataset versioning

Versions identify the transformation or dataset definition, not individual records.

For example:

```text
cleaned_text/v1
cleaned_text/v2
```

may represent different cleaning prompts or models.

Likewise:

```text
embeddings/model=all-minilm-l6-v2/variant=minilm/version=v1
embeddings/model=amazon-titan-embed-text-v2/variant=titan-v2/version=v1
```

may represent different model or preprocessing configurations.

A version should never silently overwrite an incompatible dataset.

## 13. Serving layer

The application should consume a coherent, rebuildable snapshot of lakehouse datasets.

```text
                  Lakehouse
                     │
                     ▼
             published snapshot
                     │
            ┌────────┼────────┐
            ▼        ▼        ▼
          API      search    frontend
```

A publication manifest can identify the exact datasets:

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

Indexes and serving objects are disposable and should be rebuildable from the lakehouse datasets.

## 14. Data lineage

The complete lineage is:

```text
                    ┌──► normalized posting
                    │
raw posting ────────┼──► responsibility text
                    │
                    ├──► skills
                    │
                    └──► other artifacts
                              │
                              ▼
                         embeddings
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                clustering            search
                    │
                    ▼
                   UMAP
                    │
                    ▼
              visualization
```

Every derived dataset should be traceable to its source data, transformation code, configuration, and model versions.

## 15. Design principles

1. **Raw data is immutable.**
2. **SHA-256 content hashes provide raw-data identity and duplicate detection.**
3. **Ingestion events are distinct from content identity.**
4. **Raw data is preserved as JSON; queryable datasets are stored as Parquet.**
5. **Hive-style partitioning is used for low-to-moderate-cardinality query dimensions.**
6. **High-cardinality identifiers are not used as partitions.**
7. **Derived artifacts are versioned rather than overwritten.**
8. **Logical Athena tables are separated from physical S3 file organization.**
9. **Lineage is explicitly recorded in dataset metadata and manifests.**
10. **Evaluation datasets are distinct from evaluation results.**
11. **Model, prompt, pipeline, and configuration versions are independently tracked.**
12. **Embeddings and other ML artifacts are rebuildable datasets.**
13. **Indexes are acceleration structures, not sources of truth.**
14. **Serving data is a coherent, rebuildable snapshot of lakehouse datasets.**
15. **Dataset versions must be reproducible from immutable inputs.**
16. **Partitioning exists to reduce Athena scan volume, not to encode application identity.**
