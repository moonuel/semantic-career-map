---
id: "009"
title: "Data Lake Design & Best Practices"
date: "2026-10-04"
summary: "Survey of data lake and lakehouse design patterns — the medallion (bronze/silver/gold) layering, open table formats (Iceberg/Delta/Hudi), partitioning and the small-files problem, governance and lineage, and anti-patterns — mapped onto this project's object-storage layout."
---
# Data Lake Design & Best Practices — Research Report

**Date:** 2026-10-04
**Purpose:** Establish the design principles and operational best practices for the project's
data-lake-style object store, so the current `eval/` + `live/` + `configs/` layout can scale into
cloud deployment without rework.
**Sources:** Databricks lakehouse engineering docs and blog, Microsoft Learn (Azure Databricks /
Fabric medallion guidance), AWS Big Data & Storage blogs, Delta Lake and Iceberg docs, and
independent 2026 table-format comparisons (see §10).

---

## 1. Executive Summary

A **data lake** stores data in its native/raw format on cheap, scalable object storage (S3, GCS,
ADLS) with a flat key-value layout and schema-on-read. It was created to escape the cost and
rigidity of warehouses, but unmanaged lakes degrade into **"data swamps"** — hard to find, hard to
trust, expensive to query. The industry's answer is a set of *disciplines*, not a product:

1. **Layer by data quality** — the **medallion architecture** (bronze → silver → gold) makes the
   refinement path explicit and reproducible.
2. **Add a transactional table format** — **Delta Lake, Apache Iceberg, or Apache Hudi** bring
   ACID transactions, schema evolution, and time travel to object storage, turning a lake into a
   **lakehouse**.
3. **Design the physical layout** — partitioning, file sizing, and compaction control query cost;
   the **small-files problem** is the single most common performance failure.
4. **Govern from day one** — a catalog, lineage, access control, and data quality checks are
   first-class platform capabilities, not afterthoughts.
5. **Avoid the known anti-patterns** — raw-only lakes, over-partitioning, un-governed sprawl,
   duplicated data, and ad-hoc maintenance.

**Key recommendation for this project:** the existing `eval/` / `live/` / `configs/` split is
already a lifecycle-oriented design and maps cleanly onto the medallion concept. Continue
schema-on-write principles and provenance metadata now; adopt a lakehouse table format (Iceberg as
the default) and a catalog when relational querying (the planned Athena SQL layer) becomes real,
so the migration is additive rather than a rewrite.

---

## 2. What a Data Lake Is (and How It Fails)

A data lake is a central store holding large amounts of data in its **native, raw format** on object
storage with a **flat architecture**, using metadata tags and unique identifiers rather than a
directory hierarchy. It handles structured, semi-structured, and unstructured data, unlike a
warehouse (structured only), and stores all stages of refinement side-by-side at low cost.

The classic failure mode is the **data swamp**: raw data accumulates without metadata management,
quality controls, or access governance, so nobody can find, trust, or use it. Databricks names the
root causes bluntly — early lakes lacked *transaction support*, *quality/governance enforcement*,
and *performance optimization*. The lakehouse adds those back.

| | Data lake | Data lakehouse | Data warehouse |
|---|---|---|---|
| Data types | All (structured, semi, unstructured) | All | Structured only |
| Format | Open | Open | Closed/proprietary |
| Cost | Low | Low | High |
| Reliability | Low ("swamp" risk) | High | High |
| Performance | Poor | High | High |
| Primary users | Data scientists | Analysts + DS + MLE | Analysts |

---

## 3. Medallion Architecture (Bronze → Silver → Gold)

The dominant organizational pattern. Databricks frames it as **"incrementally and progressively
improving the structure and quality of data"** as it flows through layers, sometimes called a
"multi-hop" architecture. It is a *recommended best practice, not a requirement*.

### 3.1 Bronze (raw)

- Holds data **as-is** from source systems, in original format; append-only, grows over time.
- Serves as the **single source of truth**, preserving fidelity and enabling **reprocessing and
  audit** without re-reading the source.
- Minimal validation; store most fields as string/VARIANT/binary to survive unexpected schema
  changes; add provenance metadata columns (load time, source, process ID).
- Not intended for direct analyst consumption.

### 3.2 Silver (validated)

- **Cleaned, conformed, deduplicated, standardized**, type-cast data; an enterprise-wide view of
  key entities.
- Where schema enforcement, null handling, deduplication, late-arriving-data resolution, and data
  quality checks live.
- Typically third-normal-form-like models; **do not write directly from ingestion** — always stage
  through bronze first (writing straight to silver imports schema-change failures and corrupt
  records).

### 3.3 Gold (enriched/curated)

- Aggregated, business-level, **consumption-ready** data: dashboards, BI, ML features, apps.
- Denormalized / star-schema models optimized for read performance; fewer, highly-tuned datasets.
- Row/column-level security and pre-aggregation live here.

### 3.4 Landing zone (optional)

Larger estates add a **landing zone** ahead of bronze: raw file drops from external systems in
object storage, secured by volumes, with event notifications triggering ingestion. Design ingestion
to be **idempotent** so retries are safe.

---

## 4. Open Table Formats: Delta Lake, Iceberg, Hudi

A table format is a **file-layout specification** that tells engines how to read/write parquet data
files plus a metadata layer. It adds database-like guarantees to object storage:

- **ACID transactions** (snapshot isolation; readers never see partial writes),
- **Schema evolution** without rewriting data,
- **Time travel** to historical snapshots,
- **Efficient metadata pruning** (file/partition skipping before a scan).

| Capability | Apache Iceberg | Delta Lake | Apache Hudi |
|---|---|---|---|
| Origin | Netflix / multi-engine | Databricks / Spark | Uber / streaming |
| ACID | Yes (catalog-coordinated) | Yes (log + catalog commits) | Yes (timeline) |
| Schema evolution | Full | Full (column mapping) | Full |
| Partition evolution | Yes, no rewrite | Limited | Yes |
| Best fit | Multi-engine analytics | Spark-centric, multi-table txns | High-frequency upserts / CDC |

**2026 consensus:** *Iceberg is the default recommendation unless you have a specific reason
otherwise* — it has the broadest engine/vendor support, partition evolution, and the lowest lock-in
risk. Delta wins inside a committed Databricks estate; Hudi wins when record-level CDC upserts are
the dominant workload. All three store **Parquet** underneath and are converging via **Delta
UniForm** (read Delta as Iceberg) and **Apache XTable** (metadata translation).

> A table format is not a catalog. The **catalog** (Unity Catalog, Apache Polaris, AWS Glue, Nessie)
> is the system of record for which tables exist, where they live, and who may access them. The
> lock-in question in 2026 is increasingly *"whose catalog am I writing through?"* — prefer catalogs
> implementing the open **Iceberg REST spec** (Polaris, Glue) to preserve engine portability.

---

## 5. Physical Layout: Partitioning, File Sizing, and the Small-Files Problem

### 5.1 The small-files problem

Object-store queries pay fixed per-file overhead; many small files mean many tiny requests, slow
query planning from bloated metadata, and higher cost. AWS measured a **3.2× query speedup** after
compaction (1 MB → 512 MB files), with uncompacted 1 MB tables issuing **8.5× more read requests**.
Typical causes: over-partitioning, frequent incremental writes/streaming, and `repartition(n)`
misuse.

### 5.2 Mitigations

- **Target file size** of roughly **128 MB – 1 GB** (AWS S3 Tables default 512 MB; Databricks
  autotunes 256 MB → 1 GB by table size).
- **Compaction** (Delta `OPTIMIZE`, Iceberg `rewrite_data_files`) — bin-packs small files into
  right-sized ones; safe under ACID because it is a transactional rewrite (the `dataChange=false`
  flag distinguishes compacted files from genuinely new data).
- **Auto-compaction / optimized writes** on ingest plus scheduled maintenance for large tables.
- **Compact only the relevant subset** (predicate-scoped) in incremental pipelines.
- **Delete no compaction strategy for streaming writes** — always pair streaming ingest with
  maintenance or it silently accumulates until queries break.

### 5.3 Partitioning

- Partition on **low-cardinality columns that match query predicates** (commonly date) to enable
  **partition pruning**.
- **Over-partitioning** (high-cardinality keys) creates millions of tiny partitions and is an
  anti-pattern; modern formats increasingly favor **hidden partitioning** (Iceberg) or **liquid
  clustering / Z-order** (Delta) over Hive-style directory partitioning.
- Bronze partitioning is acceptable but discouraged; silver/gold favor clustering.

### 5.4 Maintenance ordering

Run maintenance **in the correct order**: compact → expire snapshots → delete orphan files →
rewrite manifests. Deleting orphans before snapshots expire can remove files still referenced by
active snapshots. Set retention windows (snapshot expiry, `VACUUM`/orphan-cleanup min age) based on
how far back jobs query, and codify maintenance as a **runbook**, not an ad-hoc task.

---

## 6. Governance, Metadata, and Lineage

Governance is a **first-class platform capability**, not tool configuration. The recurring
controls:

- **Catalog / metadata store** — one authoritative place for schemas, ownership, descriptions, and
  discovery; align catalog/schema structure to the org (domain-based, environment-based, or
  lifecycle-based) and mirror medallion layers as schemas.
- **Descriptions & tags** — annotate tables/columns with business context, sensitivity, and
  quality level at ingestion so data is discoverable and classifiable.
- **Access control** — role-based controls plus **fine-grained, view-based row filters and column
  masks**; review permissions periodically so they don't go stale.
- **Lineage** — capture data flow from source to insight (column-level where possible) for impact
  analysis, root-cause debugging, sensitive-data tracking, and cross-team dependency discovery.
  OpenLineage is the emerging interchange standard.
- **Data quality** — checks at each layer; enforcement tightens as data moves bronze → gold; monitor
  quality metrics and alert on violations. Define freshness SLAs.
- **PII handling** — strip/replace PII with a stable ID *before* landing raw data, preserving
  analytic linkage while satisfying GDPR/CCPA; this enables otherwise-throwaway data to be retained.

---

## 7. Best Practices Checklist

**Storage & layout**
- [ ] Raw data preserved as-is, append-only, native format, with provenance metadata.
- [ ] Refinement path explicit (landing → bronze → silver → gold).
- [ ] Partition on query-predicate, low-cardinality keys; avoid over-partitioning.
- [ ] Target 128 MB – 1 GB files; automate compaction; maintenance as a documented runbook.
- [ ] Open formats (Parquet + Iceberg/Delta) over proprietary; open catalog (Iceberg REST).

**Ingestion**
- [ ] Idempotent ingestion (safe retries); incremental over full reloads.
- [ ] Do not bypass bronze with direct-to-silver writes.
- [ ] Schema-on-write with explicit evolution; schema contracts for change management.

**Governance & trust**
- [ ] Catalog every dataset with owner, description, tags, sensitivity.
- [ ] Column-level lineage captured automatically where possible.
- [ ] Fine-grained access control; least privilege; periodic permission review.
- [ ] Data quality checks + monitoring + alerting + SLAs per layer.

**Cost & performance**
- [ ] Right-sized files; pruning-friendly layout; lifecycle tiers for cold data.
- [ ] Ingest at the *highest latency the business accepts* to minimize small files.

---

## 8. Anti-Patterns

1. **Raw-only lake** — dumping data with no refinement path; becomes a swamp.
2. **Skipping bronze** — losing the ability to reprocess/audit; corruption propagates downstream.
3. **Over-partitioning / high-cardinality partitions** — millions of tiny files, un-compactable.
4. **No compaction strategy for streaming/upserts** — silent small-file accumulation.
5. **Long snapshot retention** — defeats compaction by pinning old files in storage.
6. **Out-of-order maintenance** — orphan cleanup before snapshot expiry deletes live files.
7. **Duplicating data across silos** — defeats the single source of truth; classic lake-vs-warehouse
   duplication cost.
8. **Ungoverned data sprawl** — no catalog, no lineage, no access control; nothing is discoverable
   or trustworthy.
9. **Ad-hoc, manual maintenance** — error-prone and unscalable; encode it as code/schedule.
10. **Chasing a table format without a portable catalog** — the format is open, the governance
    layer around it is where new lock-in lives.

---

## 9. Relevance to This Project

The current storage design (`architecture/data-layout.md`, `backend/storage/keys.py`) already
reflects several core data-lake principles, and this research suggests where to reinforce it:

**Already aligned**
- **Lifecycle separation as the top-level axis.** `eval/` (authored, immutable ground truth) vs.
  `live/` (raw + derived, regenerable) vs. `configs/` (append-only provenance) is a
  *lifecycle-based* organization — the same principle Databricks recommends (raw / curated /
  analytics catalogs). `eval` and `live` never cross; §7's "preserve raw as source of truth" is met
  by `live/raw/<id>/source.md`.
- **Schema-on-write ingestion + provenance.** The structured key builder and per-object
  `metadata.json` are exactly the "annotate at ingestion" and "capture provenance" practices; the
  key encodes only what must be queryable, with richer metadata held alongside (§6 catalog/lineage).
- **Config hashing as provenance.** Hashing model/temperature/system-prompt into `configs/<hash>.json`
  is a lightweight, append-only lineage record for each derived artifact — the project's analogue of
  a catalog's transformation/lineage events.
- **Terminology already matches.** `raw/` ↔ bronze, `derived/` (clean-text, LLM clean, features)
  ↔ silver, and embeddings/visualizations ↔ gold-level consumption artifacts.

**Recommended reinforcements**
- **Deferred dedup is a known swamp risk.** The "observable for now" dedup stance (§4.1 of the
  data-layout note) is the classic *un-governed accumulation* anti-pattern. Since dedup is deferred,
  keep a registry/lookup path planned for the `metadata.json` fields (employer, title, date) so the
  eventual semantic-similarity filter has an index to build on.
- **Plan the catalog before the Athena layer.** The note already anticipates an Athena SQL layer over
  `metadata.json`. A **catalog** (Glue is the natural AWS match; Iceberg REST for portability) is
  the prerequisite for that layer, and defining it now prevents a later path-rewrite.
- **Choose an open table format before scale, not at scale.** When metadata becomes relational
  enough to want tables (rather than per-object JSON), adopt **Iceberg** as the default (broadest
  engine support, partition evolution) on a portable catalog (Glue/Polaris), with Delta UniForm as
  an interop bridge if a Databricks estate appears.
- **Watch file/object granularity.** Job postings are small objects; if derived artifacts or
  embeddings are stored one-per-object at thousands of IDs, consider Parquet/Iceberg tables for the
  queryable subset to avoid a small-files problem as the corpus grows toward the planned 350+.
- **Write maintainable ingestion now.** Keep ingestion **idempotent** and prefer incremental
  appends under `raw/` — the design already says "iteration is additive," which is the correct
  posture.

**Suggested next steps (not this sprint)**
1. Add a `schemas/` (or catalog-registration) step defining the `metadata.json` contract.
2. Spike an Iceberg table (local/Glue) over a small slice of `live/derived/` to validate the
   migration path and maintenance runbook.
3. Track file sizes/object counts as a metric so the small-files threshold is observable before it
   hurts.

---

## 10. Sources

1. **Databricks** — "What is the medallion lakehouse architecture?" <https://docs.databricks.com/aws/en/lakehouse/medallion>
2. **Microsoft Learn** — "What is the medallion lakehouse architecture?" (Azure Databricks). <https://learn.microsoft.com/en-us/azure/databricks/lakehouse/medallion>
3. **Databricks** — "Phase 6: Design Delta Lake architecture." <https://docs.databricks.com/aws/en/lakehouse-architecture/deployment-guide/delta-lake>
4. **Microsoft Learn** — "Implement Medallion Lakehouse Architecture in Fabric." <https://learn.microsoft.com/en-us/fabric/onelake/onelake-medallion-lakehouse-architecture>
5. **Databricks** — "What is Medallion Architecture?" (blog). <https://www.databricks.com/blog/what-is-medallion-architecture>
6. **Databricks** — "Data lake best practices." <https://www.databricks.com/discover/data-lakes/best-practices>
7. **Databricks** — "Introduction to Data Lakes." <https://www.databricks.com/discover/data-lakes>
8. **Databricks** — "Open Table Formats Explained: Iceberg vs. Delta vs. Hudi." <https://www.databricks.com/blog/open-table-formats>
9. **AWS Big Data Blog** — "Navigating architectural choices for a lakehouse using Amazon SageMaker." <https://aws.amazon.com/blogs/big-data/navigating-architectural-choices-for-a-lakehouse-using-amazon-sagemaker/>
10. **AWS Storage Blog** — "How Amazon S3 Tables use compaction to improve query performance by up to 3 times." <https://aws.amazon.com/blogs/storage/how-amazon-s3-tables-use-compaction-to-improve-query-performance-by-up-to-3-times/>
11. **AWS Big Data Blog** — "Apache Iceberg optimization: Solving the small files problem in Amazon EMR." <https://aws.amazon.com/blogs/big-data/apache-iceberg-optimization-solving-the-small-files-problem-in-amazon-emr/>
12. **AWS Storage Blog** — "Optimizing storage costs and query performance by compacting small objects." <https://aws.amazon.com/blogs/storage/optimizing-storage-costs-and-query-performance-by-compacting-small-objects/>
13. **Delta Lake** — "Small File Compaction with OPTIMIZE." <https://delta.io/blog/2023-01-25-delta-lake-small-file-compaction-optimize/>
14. **Delta Lake** — "Universal Format (UniForm)." <https://docs.delta.io/delta-uniform/>
15. **Databricks** — "Control data file size." <https://docs.databricks.com/aws/en/tables/tune-file-size>
16. **Microsoft Learn** — "Best practices for data and AI governance." <https://docs.databricks.com/aws/en/lakehouse-architecture/data-governance/best-practices>
17. **Google Cloud** — "About data lineage" / "Enable policy-compliant lakehouse access." <https://cloud.google.com/data-catalog/docs/concepts/about-data-lineage>
18. **Algoscale** — "Iceberg vs Delta vs Hudi in 2026." <https://algoscale.com/go/blog/iceberg-vs-delta-vs-hudi-2026/>
19. **Data Workers** — "Data Lakehouse Architecture 2026." <https://dataworkers.io/resources/data-lakehouse-architecture-2026/>
