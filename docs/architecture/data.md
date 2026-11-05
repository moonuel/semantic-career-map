# Data Architecture

!!! info "Lakehouse is live"
  The lakehouse layout below is the production layout. Raw JSON, cleaned artifacts, Titan embeddings, eval datasets/results, ingestion manifests, run/dataset manifests, and embedding-version claims all live under their lakehouse prefixes. The legacy object-per-posting layout is retained in §11 as migration history only.

The lakehouse is the immutable, versioned source of truth for raw job-posting data, and the layer every derived dataset is reproducibly built from.

The design targets **S3 + Glue Data Catalog + Athena**: raw data is preserved as canonical JSON, everything queryable becomes **Parquet**, and directories use **Hive-style partitioning** so Athena prunes scans instead of reading the whole lake. A single lake bucket is used for now (until scope justifies expansion into separate buckets), with the branches below as prefixes. A separate `state/` bucket holds the mutable production pointer.

The core principle:

> Raw data is immutable; derived datasets are reproducible; Athena tables expose logical datasets independently of their physical storage layout.

The object keys stop being the query interface. They carry the physical layout; identity, provenance, and configuration move into row metadata and manifests; queries hit tables instead of prefixes.

Full rationale: [Athena Lakehouse Layout](athena-layout.md). Storage-side and metadata-side research: reports [009](../research-reports/009-data-lake-design-best-practices.md), [010](../research-reports/010-data-warehouse-design-best-practices.md), and [011](../research-reports/011-object-store-metadata-design.md).

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
lake/                          # the lake bucket
  raw/
    source=<source>/
      sha256=<content-hash>.json       # canonical payload, content-addressed, stored once
  artifacts/
    type=<artifact-type>/
      variant=<variant>/
        version=<vN>/
          part-00000.parquet            # one artifact per row
  embeddings/
    model=<model>/
      variant=<variant>/
        version=<version>/              # vN (stable) or x-... (ephemeral)
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
    ingestion/                          # one Parquet manifest per ingestion run
    runs/                               # one record per pipeline run
    versions/embeddings/                # version-ownership claims (§2.4)

state/                         # separate control-plane bucket
  production.json              # mutable pointer to the active generations
```

The former `eval`/`live` split survives as the `eval/` prefix -- authored ground truth never crosses with generated data. The former `configs/` branch is absorbed into `manifests/` (§3).

## 2. Object Keys

Keys carry what must be **listed and pruned by prefix** -- everything else is metadata. The discipline from the legacy layout carries over: a structured key builder ([`keys.py`](../../backend/storage/keys.py)) stays the single source of truth for well-formed keys, pure and I/O-free, and every other module derives keys through it rather than by string interpolation. Its `_SEGMENT_RE` already anticipated hive-style `key=value` pairs, so the new branch prefixes arrive the way the old ones did -- builder functions first, tests beside them.

### 2.1 Content identity

Raw objects are named by content, not by arrival: `sha256=<hash>.json` hashes the **canonical JSON payload**, while the *ingestion event* gets its own identity (`ingestion_id`, recorded in the ingestion manifest). The same content observed repeatedly is therefore not a new document version -- the object already exists, and only gains another observation.

The raw key therefore carries only `source=` plus the hash leaf. There is deliberately no `ingested_on=` partition: a date in the path would give identical re-ingested payloads different keys on different runs and accumulate copies, defeating content-addressing. Objects are never rewritten, so the object's S3 `Last-Modified` is its first-ingestion time, and provenance dates for Athena range queries live in `manifests/ingestion/` (§3) instead of the key namespace.

Hashing the canonical payload rather than the exact received bytes keeps insignificant serialization differences from minting a new identity. The ingestion layer strips source-specific wrapper fields, normalizes whitespace and Unicode, and drops null/empty values before hashing and storage; every ingest still emits a new manifest row with the action (`stored` or `duplicate`) and the run's ingestion id.

### 2.2 Posting identity

The lake has no minted posting id and needs none (D11, resolved 2026-10-21 by profiling the live raw layer: 6/6 sampled sources carry no posting id, and none is needed): **the `source_hash` -- the canonical payload bytes' SHA-256 -- is the posting's identity and its join handle**, tying one posting's artifact row and eval rows back to its raw object (whose key derives from the same hash, §2.1). A payload's hash is permanent, recomputable from the stored bytes, and constant across scraper re-ingestions of identical content; the mutable parts of a posting (the ATS it came from, its human-readable `source_id`) travel in row columns, not in identity. Provenance dates live in manifests, not in the address. Ids are still minted by exactly one generator in [`ids.py`](../../backend/storage/ids.py) -- for *run* and *batch* identities (UUIDv7, lexicographic order equals chronological order, which keeps exports sortable for free), never for postings. (Decision record: [dev log, October 23, 2026](../dev-log/2026.md#october-23-2026).)

### 2.3 Partitions

Directory levels are Hive-style `<column>=<value>` pairs, which become Athena partition columns (e.g. `source=linkedin/`). Partition on low-to-moderate-cardinality dimensions that queries actually filter on -- `source`, `event_date`. Never partition on high-cardinality identifiers such as `source_hash`: an address identifies an object, a partition prunes a scan. But a partition also has to earn its place: the raw layer carries no time partition not because dates are high-cardinality, but because identical re-ingested content must not mint new keys (§2.1) -- its time provenance lives in manifests, pruned there instead. Partitioning exists to reduce what Athena reads, not to encode application identity into the directory tree.

### 2.4 Versions

`vN` identifies a transformation or dataset definition, not individual records. `responsibility_text/version=v2` may be a different extraction method over the same inputs; `embeddings/.../version=v2` may mean a different model or preprocessing configuration. Data science and ML engineering is an iterative process, and we should store artifacts from each iteration separately -- versions add, they don't overwrite, and a version never silently replaces an incompatible dataset.

Two version vocabularies exist, and the difference is a policy the storage layer enforces, not a naming convention:

| Token | Class | Meaning |
| --- | --- | --- |
| `vN` | stable | A production generation address. One configuration owns it. |
| `x-...` | ephemeral | A throwaway prefix (`x-v1.1`, `x-scratch`). Never production, never query-layer-projected by default. |

Ownership is recorded, not assumed. Before a run writes its first part it claims its prefix with a conditional put (`ObjectStore.put_if_absent`) at `manifests/versions/embeddings/model=<model>/variant=<variant>/version=<token>/claim.json`; the claim names the owning `configuration_hash`. A second configuration aimed at a claimed prefix fails (`ClaimConflict`) rather than joining the dataset. Ephemeral tokens are claimed the same way -- reuse by the *same* configuration is allowed, reuse by a different one is not.

Legacy prefixes that pre-date the registry (parts written before §2.4 landed) are guarded against silent adoption: a run targeting a *stable* token whose prefix already has parts but no claim fails loudly instead of minting itself the owner, and the operator publishes the true owner with `scripts/claim_embedding_version.py` (dry run by default; `--apply` writes) before the next production run. Gate 5 in `scripts/provision_query_layer/gate.py` reports unclaimed prefixes with that exact command.

The rule exists because `configuration_hash` is a row column, never a path segment: a config change re-hashes but does not move the prefix, so without a claim a `LIMIT 100` smoke-test configuration wrote 400 rows into production `version=v1` ([issue #22](https://github.com/moonuel/semantic-career-map/issues/22)).

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
raw posting ──► canonical payload ──► cleaned text ──► embedding
```

Row-level metadata reconstructs each hop:

```
artifact_id, source_hash, input_artifact, input_version,
pipeline_version, configuration_hash, created_at
```

The per-posting `metadata.json` sidecar was the legacy ingestion's provenance record. Current ingestion writes provenance into row columns, the raw payload, and the ingestion/run/dataset manifests instead; legacy sidecars may remain for objects created before the migration but are not part of the live path.

## 6. Queryable Datasets (Parquet)

At tens of postings, one object per posting was fine -- good, even: simple, inspectable, greppable. It does not survive contact with Athena. Millions of tiny objects are the classic small-file anti-pattern (report 009), and every full-dataset query degrades into listing the whole store. So every derived layer is a Parquet dataset: an artifact is a **row**, the part files are implementation details, and Athena queries millions of artifacts without touching millions of files. Parquet buys compression, column projection, and predicate pushdown along the way, and schemas stay written down at dataset-definition time -- the schema-on-write habit, kept.

A row under `type=responsibility_text/variant=llm/version=v1`:

```
source_hash, variant, version,
responsibility_text, configuration_hash, pipeline_version, created_at
```

The landed shape is exactly this minus the subject field: the live
`type=cleaned_text/variant=llm/version=v1` dataset carries
`source_hash, variant, version, cleaned_text, configuration_hash,
pipeline_version, created_at` (plus the run-id naming the producing run)
-- the artifact hash *is* the join handle (§2.2); there is no `artifact_id`
and no minted posting id to collide with it.

Raw data stays JSON -- canonical payload preservation beats columnar convenience.

## 7. Embeddings

Embeddings become datasets too, still grouped model-first -- a vector is a row in a model's corpus space:

```
model=<model>/variant=<variant>/version=<vN>/
```

with rows of:

```
source_hash, embedding, model, model_version,
input_artifact, input_version, embedding_dimension, normalization,
created_at, configuration_hash, run_id
```

The live scheduled generation is `model=amazon-titan-embed-text-v2/variant=titan-v2/version=v1/` -- normalized Amazon Titan Text Embeddings V2 over Bedrock, `embedding_dimension=1024`, `normalization=l2` ([experiment 012](../experiments/012-evoc-clustering-bakeoff.md) chose it on 2026-10-31). The generation's identity still runs through the same hashed config: `model_name` carries the Bedrock model family id (`amazon.titan-embed-text-v2`), `model_revision` carries the pinned Bedrock model ID (`amazon.titan-embed-text-v2:0`) exactly as it earlier carried a HuggingFace commit, and the encoder is selected by family (`build_encoder`). Nothing in the row schema, the part keys, or the `configuration_hash` canonicalization changed -- only the values. The MiniLM generation (`model=all-minilm-l6-v2/variant=minilm/version=v1/`, 384-d) stays on disk and in the production pointer as the rollback target; versions add, they never overwrite.

The dataset is the authoritative representation of the vectors. Any vector index (FAISS, or whatever scale eventually demands) is a rebuildable acceleration structure derived from it -- never the source of truth. One `<id>.npy` per posting exits with the legacy layout. (One AWS-native candidate for that acceleration-structure slot -- S3 Vectors' vector buckets, GA 2025-12 -- is evaluated in [research report 019](../research-reports/019-s3-vectors.md): not adopted now; its decision log fixes the Parquet-to-index sync contract should it ever graduate.)

## 8. Evaluation Data

`eval/` separates what is authored from what is generated, the same discipline the old `eval/` branch enforced -- it just draws the line twice. `datasets/` holds human ground truth, versioned by authorship. `results/` holds machine output over those datasets, regenerated whenever a different model, judge, or prompt evaluates them.

A dataset row -- the landed cleaning-golden version shows the shape:

```
source, source_hash, source_id, golden_text, annotator, created_at
```

(a future relevance-judging dataset would add its own subject fields in the
same spirit -- identity first, reference text, annotation provenance)

A results row -- the landed `cleaning_quality` rubric shows the shape:

```
source, source_hash, source_id, jaccard, boilerplate_hits,
hallucination_count, hallucination_sample, over_deletion_pct,
over_deletion_flagged, over_deletion_sample, golden_words, clean_words,
evaluation_configuration_hash, evaluation_version, created_at
```

(generic relevance results would swap the metric columns for
`prediction, score, judge_model, prompt_version`)

The same evaluation set can therefore be evaluated repeatedly under different assumptions, which is the whole point. This absorbs the evaluation machinery deferred from the [LLM-evals research](../research-reports/004-llm-evals-report.md): samples and references land as datasets, judgments as results, and rubrics stay Git-versioned. The `_current.json` pointer gives way to explicit `version=vN` addressing and manifest-pinned snapshots (§10).

## 9. Catalog & Athena Logical Tables

The Glue Data Catalog provides the logical schema over the physical datasets, and Athena tables are the stable interface:

```
raw_ashby, raw_dataroles, raw_greenhouse, raw_lever,
raw_hiringcafe, raw_linkedin
manifests_ingestion
artifacts_cleaned
postings_embeddings
```

A logical table maps onto S3 locations regardless of how they are partitioned underneath, so a physical repartition never breaks a query -- nothing downstream references the prefixes in §1. The directory structure is not the primary API; the tables are. The `artifacts_cleaned` registration is currently pinned to `variant=llm/version=v1`, so a new cleaning version needs a table re-registration before it is queryable.

The object store remains the system of record; Athena is the query and serving layer over it (reports 009/010 draw this boundary in detail).

## 10. Serving Snapshots

The application should not consume the bucket directly -- it consumes a coherent, rebuildable **snapshot** of lakehouse datasets. The production control plane already pins the active cleaning and embedding generations in `state/production.json`; a broader publication manifest for the serving layer is still future work:

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

Indexes and serving objects are disposable, and are rebuilt from whichever datasets the manifest pins. Today the map sidesteps this by publishing its own `index.html`, coordinates, and detail buckets under the `job_map` artifact prefix.

## 11. Legacy Layout

Everything this page described before the lakehouse rewrite is retained here as migration history. [`keys.py`](../../backend/storage/keys.py) still carries the legacy builders and validators so old objects remain addressable, but the live pipeline writes the lakehouse prefixes above:

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
| Raw address | `live/raw/<id>/source.md` | content-addressed JSON (§2.1) — **live** |
| Posting identity | `<id>` path segment | `source_hash` = identity + join handle; no minted ids (§2.2, D11) |
| Query interface | key prefix | Glue catalog + Athena tables (§9) — **live for selection** (D12, Phase 5 pulled forward): production cleaning selects via the `select_uncleaned` anti-join |
| One artifact | one JSON object | one Parquet row (§6) — **live**: `type=cleaned_text/variant=llm/version=v1` (flush-watermark parts + per-run dataset manifests) |
| One embedding | one `<id>.npy` | vector column (§7) — **live**: `model=…/variant=…/version=…` partitions, Titan V2 production generation |
| Config record | `configs/<hash>.json` | `configuration_hash` + manifests (§3) — **live** (run/dataset manifests in `manifests/runs/`, `manifests/datasets/`) |
| Eval-set pin | `eval/_current.json` | explicit version pins (§8, §10) — **live**: golden `variant=author/version=v1` + quality results `variant=exp003/version=v1,v2` with per-run manifests |
| Production pointer | implicit "latest" | `state/production.json` immutable generation pointer — **live** |

Legacy objects aren't converted -- derived data is supposed to be regenerable, so only raw payloads are re-addressed by content hash and everything else re-generates as Parquet when its pipeline stage is formalized. Migration is branch-by-branch, key builders first. The old §A.1 hunch -- that the metadata sidecar may eventually expand into an Athena SQL database -- is answered by §9.
