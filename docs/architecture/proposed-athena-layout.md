# Athena Lakehouse Proposal

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
  As adopted in [data.md](data.md §2.1), the raw layer drops the `ingested_on=<YYYY-MM-DD>/` level shown below: a date in the path would give identical re-ingested payloads different keys and accumulate copies, defeating content-addressing within a source. Objects are never rewritten, so S3 `Last-Modified` is the first-ingestion time, and provenance dates for date-range queries live in manifests instead. The tree below is the original proposal text, kept verbatim.

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
posting_id
uuid
content_hash
```

The hash identifies the object but should not be used as a partition.

The partition hierarchy should also avoid excessive partition counts. The purpose of partitioning is to reduce the amount of data Athena must scan, not to encode every attribute into the directory structure.

## 5. Queryable derived datasets

Derived artifacts should generally be stored as **Parquet datasets**, rather than one JSON file per artifact.

For example:

```text
artifacts/
└── type=responsibility_text/
    └── variant=llm/
        └── version=v1/
            ├── part-00000.parquet
            ├── part-00001.parquet
            └── ...
```

The individual artifact is represented as a row:

```text
artifact_id
source_hash
posting_id
artifact_type
variant
version
responsibility_text
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
normalized posting
    │
    ▼
responsibility text
    │
    ▼
embedding
```

A derived dataset should contain enough metadata to reconstruct this lineage:

```text
artifact_id
source_hash
input_artifact
input_version
pipeline_version
configuration_hash
created_at
```

The artifact path identifies the dataset class:

```text
type=responsibility_text/
variant=llm/
version=v1/
```

while row-level metadata identifies the specific source and transformation.

## 7. Embeddings

Embeddings are stored as Parquet datasets:

```text
embeddings/
└── model=all-MiniLM-L6-v2/
    └── variant=responsibility_text/
        └── version=v3/
            ├── part-00000.parquet
            └── ...
```

A record might contain:

```text
posting_id
source_hash
embedding
model
model_version
input_artifact
input_version
embedding_dimension
normalization
created_at
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

For example:

```text
eval/datasets/
└── type=career_relevance/
    └── variant=author/
        └── version=v1/
            └── part-00000.parquet
```

An evaluation dataset might contain:

```text
posting_id
query
expected_relevance
annotation
annotator
```

Results might be stored as:

```text
eval/results/
└── type=career_relevance/
    └── variant=llm_judge/
        └── version=v3/
            └── part-00000.parquet
```

with:

```text
posting_id
prediction
score
judge_model
prompt_version
evaluation_version
created_at
```

This allows the same evaluation set to be evaluated repeatedly with different models or evaluation procedures.

## 9. Glue Data Catalog

The Glue Data Catalog provides the logical schema over the physical S3 datasets.

Example Athena tables:

```text
raw.job_postings
artifacts.responsibility_text
artifacts.skills
embeddings.job_posting_embeddings
eval.career_relevance
eval.career_relevance_results
```

The catalog maps these logical tables to S3 locations.

For example:

```text
artifacts.responsibility_text
        │
        ▼
s3://<bucket>/artifacts/type=responsibility_text/
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
  "input_dataset": "responsibility-text-v12",
  "model": "all-MiniLM-L6-v2",
  "model_version": "...",
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
responsibility_text/v1
responsibility_text/v2
```

may represent different extraction methods.

Likewise:

```text
embeddings/model=A/version=v1
embeddings/model=A/version=v2
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
