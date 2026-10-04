---
id: "010"
title: "Data Warehouse Design & Best Practices"
date: "2026-10-04"
summary: "Survey of data warehouse design — the Kimball/Inmon/Data Vault modeling paradigms, dimensional modeling (grain-first stars, SCDs, conformed dimensions, surrogate keys), the three-tier and semantic layers, ETL vs ELT, physical design (partitioning, clustering, materialized views), governance, and the classic anti-patterns — mapped onto this project's planned Athena SQL layer over the object store."
---
# Data Warehouse Design & Best Practices — Research Report

**Date:** 2026-10-04
**Purpose:** Establish the design principles and modeling best practices for the relational
query layer this project anticipates (the `metadata.json` → Athena SQL step named in
`architecture/data-layout.md`), and to position that layer correctly relative to the
study's existing `eval/` + `live/` + `configs/` object store.
**Companion reading:** Report [009 — Data Lake Design & Best Practices](009-data-lake-design-best-practices.md)
covers the storage-side (lakehouse, table formats, small files). This report covers the
**relational/analytics modeling side** — the part that sits on top once data becomes tabular.
**Sources:** Kimball Group technique library, Microsoft Learn / Azure and Fabric warehouse
guidance, AWS analytics whitepaper and Well-Architected Analytics Lens, Snowflake fundamentals,
Databricks and Keboola design guides, Data Vault reference material, and a 2026 academic
comparative review (see §11).

---

## 1. Executive Summary

A **data warehouse** is a subject-oriented, integrated, time-variant, and non-volatile
store of structured data, shaped for analysis rather than transactions (OLAP, not OLTP).
Its purpose has not changed in thirty years; how it is built has.

1. **Modeling paradigm is a strategic choice.** **Kimball** (bottom-up dimensional marts),
   **Inmon** (top-down normalized EDW), and **Data Vault** (integration-layer hubs/links/
   satellites) answer *different questions*. Most real estates blend them: an integration
   layer that absorbs source change, and a dimensional presentation layer for consumption.
2. **Dimensional modeling is the default for consumption.** Facts (measured events at a
   declared grain) surrounded by denormalized dimensions, joined in a **star schema**, are
   what BI tools and analysts expect. **Grain is the first and most important decision** —
   everything else follows from it.
3. **Treat time as a first-class dimension.** Slowly changing dimensions (SCDs), surrogate
   keys, conformed dimensions, and late-arriving-data handling are the machinery that make
   historical analysis correct. Getting these wrong yields *plausible but wrong* numbers.
4. **Layer the platform.** Staging → integration (ODS/normalized) → presentation (marts) →
   semantic (metric contracts) → access. A **semantic layer** with single metric
   definitions is the contract that stops "four customer counts on four dashboards."
5. **Prefer ELT in the cloud.** Load raw first, transform in-warehouse; decouple storage and
   compute; lean on partitioning, clustering, and materialized views for performance.
6. **Avoid the classic anti-patterns** — undeclared grain, bypassing the semantic layer,
   ungoverned dimension copies, and ad-hoc (unrunbooked) maintenance.

**Key recommendation for this project:** the object store remains the system of record and the
lake/lakehouse layer. The warehouse is a *derived, regenerable* relational serving layer built
from `metadata.json` and derived artifacts — **not** a second source of truth. Model it as a
small star schema (posting fact + company/role/location/date dimensions), declare grain
explicitly, keep surrogate keys, and put metric definitions in one semantic layer. Start
simple; adopt Data Vault *only if* multiple volatile sources ever merge into the same entities.

---

## 2. What a Data Warehouse Is (and the OLAP/OLTP Split)

A data warehouse consolidates data from many sources into one analytical store. The canonical
four properties (Inmon):

| Property | Meaning |
|---|---|
| **Subject-oriented** | Organized around business subjects (customer, sale), not applications. |
| **Integrated** | Common encoding/naming across sources — one vocabulary. |
| **Time-variant** | Retains history; the time dimension is intrinsic, not incidental. |
| **Non-volatile** | Loaded and read; not updated transactionally once loaded. |

The core contrast is **OLTP vs OLAP**: transactional systems normalize heavily to keep writes
fast and avoid anomalies; analytical stores **denormalize deliberately** to keep reads fast,
because the dominant operation is filtering, grouping, and aggregating across history — not
single-row writes. A warehouse is not a bigger OLTP database; it is a differently *shaped* one.

A close cousin is the **operational data store (ODS)**: current-state, frequently refreshed,
low-history — used for near-real-time operational reporting. Warehouses hold *historical*,
periodically loaded data. Confusing the two produces freshness expectations the warehouse
was never designed to meet.

---

## 3. Modeling Paradigms: Kimball, Inmon, Data Vault

The three dominant methodologies are not competitors for one job; they optimize different
layers and constraints.

### 3.1 Kimball — bottom-up dimensional (default for analytics)

Build **data marts** (star schemas) per business process, tied together by **conformed
dimensions**; the warehouse *is* the union of conformed marts. Fast time-to-value, easy for
analysts to query, denormalized for performance. Cost: governance and conformance are
disciplines you must maintain, not properties the structure forces.

### 3.2 Inmon — top-down normalized EDW

Build a normalized (**3NF**) integrated enterprise model first, then derive dimensional marts
on top. Strong consistency, decoupled from consumers, low redundancy, clear grain and
semantics. Cost: heavy upfront modeling, many joins for ad-hoc analysis, slower first
deliverable.

### 3.3 Data Vault — integration layer (hubs, links, satellites)

Optimizes for **integration and auditability under source churn**, not consumption. Three
structures:

- **Hub** — one row per business key (e.g., customer number) plus load metadata. No
  descriptive attributes.
- **Link** — one row per relationship between hubs (the "physical many-to-many relationship").
- **Satellite** — descriptive attributes hanging off a hub or link; **insert-only**, so every
  change is a new row (SCD-Type-2-by-default), stamped with load date + record source.

Deterministic **hash keys** let independent loads generate the same key without coordinating,
enabling **parallel loading**. History and lineage are automatic. The trade-off is severe:
the raw vault is *not* human-queryable — a simple question can require 6–12 joins — so every
serious implementation adds **PIT (point-in-time) tables, bridge tables, and a dimensional
presentation layer on top**, typically +30–50% more models and sizable operational overhead.

### 3.4 Choosing between them

| | Kimball (Dimensional) | Inmon (Normalized EDW) | Data Vault |
|---|---|---|---|
| Layer modeled | Presentation | Enterprise integration + presentation | Integration / history |
| Sources | Few (~3–10) | Enterprise-wide | Many (10–100+), volatile |
| History | SCDs, per attribute | Time-variant normalized tables | Automatic (append-only satellites) |
| Schema change | Rebuild dims/marts | Structural change to core | Add satellites (additive) |
| Query complexity | Simple (star joins) | Medium (3NF joins) | High without PIT/bridges |
| Parallel loading | Dependent | Dependent | Independent |
| Best fit | Analytics-first, fast delivery, BI | Consistency-first, decoupled EDW | Many sources, hard audit, big teams |

**Decision rule (2026 consensus):** default to **Kimball dimensional modeling for the
presentation layer, always**. Add **Data Vault underneath only when you genuinely carry the
problem it solves** — many volatile/overlapping sources, a real audit requirement, and a large
enough team to amortize the discipline. **Inmon's normalized core** remains the right shape for
enterprise consistency and is effectively what a well-run "data vault-ish integration layer"
reproduces. The "versus" framing is mostly false: most mature estates use one paradigm for
integration and another for consumption.

---

## 4. Dimensional Modeling in Depth

Dimensional modeling separates **what happened** (facts) from **the context it happened in**
(dimensions), joined in a star optimized for reading.

### 4.1 Grain first — the decision everything depends on

The **grain** is the precise, formal definition of what one fact row represents — e.g.
*"one row per order line item at the time the shipment record is created."* Declare it
explicitly, in one sentence, in a reviewed design document. If two people would describe it
differently after reading it, it is not specific enough yet. Consequences of getting grain
wrong: **double counting** from fan-out joins, **lost detail** from premature aggregation, and
wrong measures. Build at the **lowest granularity that is meaningful** — you can always roll
up, but you can never drill into detail that was never stored.

### 4.2 Fact tables

- Hold **foreign keys to dimensions plus numeric, additive measures** (revenue, quantity).
- **Never** store text flags/indicators in a fact — move them to a dimension; that is where
  users want to filter and group, and a surrogate key is smaller than the text anyway.
- Beware **non-additive** measures (ratios, year-to-date totals) sneaking in: a YTD column in
  a fact row invites double counting whenever more than one date is queried.
- Three common types: **transaction facts** (one row per event), **periodic snapshots** (one
  row per entity per period, e.g. daily balances), **accumulating snapshots** (one row per
  lifecycle, with milestone-date columns updated as stages complete).

### 4.3 Dimension tables

- Hold **descriptive context**: names, categories, geographies, hierarchies.
- Prefer **wide and denormalized**; accept redundancy because the goal is readable, fast
  queries, not minimal storage.
- Collapse hierarchies into one dimension when possible; only normalize (snowflake) when a
  hierarchy changes frequently and would cascade edits across many rows, when an attribute set
  has high NULL density, or when a hierarchy (date, geography, currency) is shared across
  dimensions.
- Provide **`unknown` / `not applicable` members** so facts with an unresolved or truly
  unknown dimension can still join — this keeps row counts and totals stable across queries.

### 4.4 Keys

- **Surrogate keys** (system-generated, meaningless integers) are the primary keys of
  dimensions and the foreign keys in facts. They decouple the warehouse from source-system key
  churn, make SCD versioning possible (each version gets its own surrogate), and shrink join
  keys. Never expose them to business users.
- Business/natural keys live in the dimension as ordinary attributes and are the stable
  identity used for loading.

### 4.5 Slowly changing dimensions (SCD)

Handle *how each attribute changes over time*. **SCD type is an attribute-level decision, not
a dimension-level one** — one customer dimension may keep full history on loyalty tier, always
overwrite display name, and keep the prior value of region.

| Type | Behavior | History | Use for |
|---|---|---|---|
| **1** | Overwrite in place | None | Corrections; attributes where only current state matters |
| **2** | New row + effective/end dates + current flag, new surrogate key | Full | Attributes whose past value must be preserved |
| **3** | Prior + current in separate columns | Limited (one prior) | Occasional "was vs is" analysis |
| **6** | 1 + 2 + 3 hybrid: versioned rows carrying current attributes too | Full + current | Grouping history by *current* attributes without a self-join |

Type 2 is the workhorse; fact rows loaded before a change reference the **old** surrogate,
after a change the **new** one, so the warehouse can answer "what was the dimension state at
transaction time." Test surrogate-key resolution against a known historical output — a join
that resolves to the wrong version succeeds silently and is analytically wrong.

### 4.6 Conformed dimensions — the glue

A **conformed dimension** (date, customer, product) is defined once and referenced identically
by every fact table and mart. Conformance is what makes cross-process metrics comparable. When
teams fork a dimension into copies or variants, cross-process analysis silently breaks — classic
siloing. **Conformed dimensions are contracts**: shared keys, shared naming, shared semantics.

### 4.7 Late-arriving data

Facts can arrive before their dimension context. Two standard handles: create an **inferred /
placeholder dimension row** with the unresolved business key and unknown attributes, then
Type-1-update it when the real context arrives; or hold the fact and retry (a reconciliation
pattern) until the dimension lands. Either way, define the policy explicitly and test it — the
edge case is where silent errors hide.

---

## 5. Architecture: Layers and the Three Tiers

### 5.1 Three-tier structure (most common)

- **Bottom tier** — the warehouse database: integrated, historical, structured.
- **Middle tier** — an OLAP/transformation server: complex analysis, aggregation.
- **Top tier** — front-end: BI tools, dashboards, notebooks.

### 5.2 Logical layers (modern phrasing)

| Layer | Role |
|---|---|
| **Source / ingestion** | Capture raw data from operational systems, APIs, streams. |
| **Staging** | Raw landing zone, as-is, minimal transform; replayable, idempotent. |
| **Integration / ODS** | Cleansed, deduplicated, conformed; business keys resolved. Data Vault or 3NF lives here. |
| **Presentation / mart** | Dimensional star schemas, optimized for BI. |
| **Semantic** | Metric definitions, access policies, virtual views — the contract with consumers. |
| **Access** | Dashboards, BI, ML feature stores, reverse-ETL, AI agents. |

Not every warehouse implements every layer, but the separation is what lets ingestion churn,
integration logic, and presentation design evolve independently. Modern designs increasingly
follow the **medallion** zoning (bronze/silver/gold) inside a lakehouse, with the warehouse as
a *serving* layer on the curated (gold) zone rather than a standalone silo. This is the bridge
to the lakehouse pattern discussed in report 009.

### 5.3 ETL vs ELT

- **ETL** transforms before loading — less storage, but the transform engine is the bottleneck
  and raw data may not be retained.
- **ELT** loads raw first, then transforms *inside* the warehouse using its elastic compute —
  the modern cloud default. It preserves raw fidelity, decouples ingestion from transformation,
  and scales with the warehouse.
- For operational sources, **CDC (change data capture)** supplies deltas; for time-sensitive
  event streams, a streaming path feeds near-real-time tables.

### 5.4 Semantic layer

One authoritative definition per metric, owned and certified. Without it, analysts define the
same KPI differently across dashboards and reconcile forever. The semantic layer is the
contract between the warehouse and everything downstream — including AI/ML consumers. Most
"warehouse problems" turn out to be *definitional* problems the semantic layer is built to
prevent.

---

## 6. Physical Design & Performance

- **Partitioning** — split large tables by a low-cardinality, query-predicate column
  (usually date) to enable **partition pruning**. Avoid over-partitioning on
  high-cardinality keys.
- **Clustering / sort keys / Z-order** — co-locate rows frequently filtered together;
  complements partitioning on columnar engines.
- **Materialized views** — precompute expensive, recurring aggregates; refresh on schedule.
- **Columnar storage** (Parquet, ORC) — read only the columns and rows a query needs;
  dictionary/RLE encoding compresses repeated dimension values heavily, which is *why*
  denormalized stars are cheap on modern engines.
- **Narrow facts, lean hot paths** — keep fact tables narrow; if a dimension explodes in width,
  separate cold attributes into their own dimension.
- **Independent scaling** — cloud warehouses scale storage and compute separately, and isolate
  workloads (ingest, BI queries, transforms) so they don't contend.
- **Incrementality** — prefer incremental loads over full reloads to cap cost and latency;
  use watermarks/CDC to identify deltas.

---

## 7. Governance & Trust

- **Documented business definitions** before SQL. A metric without a written definition is a
  future argument.
- **Raw data immutable** — never edit raw; always transform forward. This preserves replay.
- **Catalog + lineage from day one** — one place for schemas, ownership, and column-level data
  flow; OpenLineage is the emerging interchange standard. Lineage powers impact analysis,
  debugging, and sensitive-data tracking.
- **Data quality tests in the pipeline** — uniqueness/not-null on keys, referential tests from
  facts to dimensions, freshness and row-count anomalies, schema-drift detection. Catch issues
  before they reach dashboards.
- **Access control** — role-based controls plus table/row/column-level policies (row filters,
  column masks); least privilege; periodic permission review so access doesn't go stale.
- **PII handling** — strip/replace identifiers before landing raw where regulation demands,
  preserving analytical linkage via a stable pseudonymous key.
- **CI/CD for data** — schema changes and mart definitions flow through version control and
  promotion through environments, like any other code.

---

## 8. Best Practices Checklist

**Modeling**
- [ ] Declare the grain of every fact in one formal sentence, in a reviewed doc.
- [ ] Build facts at the lowest meaningful granularity; keep them narrow and additive.
- [ ] Use surrogate keys for all dimension PKs / fact FKs; keep natural keys as attributes.
- [ ] Decide SCD type **per attribute**, not per dimension.
- [ ] Build and share **conformed dimensions** (date first); treat them as contracts.
- [ ] Provide `unknown` dimension members; define a late-arriving-data policy.
- [ ] Default to star; snowflake only specific dimensions/attribute groups with a concrete reason.

**Architecture**
- [ ] Separate staging → integration → presentation → semantic layers; keep raw immutable.
- [ ] Prefer ELT; use CDC/incremental loads; design ingestion to be idempotent/replayable.
- [ ] Implement a semantic layer with one certified definition per metric.

**Performance**
- [ ] Partition on query-predicate, low-cardinality keys; cluster where useful.
- [ ] Materialize recurring heavy aggregates; use columnar formats.
- [ ] Keep hot query paths lean; isolate workloads.

**Governance**
- [ ] Metric/business definitions written before code.
- [ ] Catalog, column-level lineage, and tags at ingestion.
- [ ] Automated quality tests + freshness/SLA monitoring + alerting.
- [ ] Column/row-level access control; PII handled at landing; CI/CD for models.

---

## 9. Anti-Patterns

1. **Undeclared / vague grain** — the root cause of double counting and fan-out joins.
2. **Bypassing the semantic layer** — every team recomputes metrics its own way; numbers
   stop reconciling.
3. **Copies of conformed dimensions per mart** — silo the warehouse; cross-process analysis
   breaks silently.
4. **Text, flags, and non-additive measures in fact tables** — bloats rows and invites double
   counting (YTD columns).
5. **SCD decided per dimension instead of per attribute** — rebuilds when the wrong strategy is
   discovered after loading.
6. **Snowflaking everything** — more joins, more complexity, little savings on columnar cloud
   warehouses where storage is cheap.
7. **No late-arriving-data / unknown-member policy** — facts silently drop or totals vary by
   query.
8. **Over-partitioning / no incremental strategy** — tiny partitions, full reloads, runaway cost.
9. **Data Vault without a problem to solve** — tripled table count, mandatory PIT/bridge/
   presentation layers, daily maintenance, no collected benefit.
10. **Ad-hoc, manual maintenance and untested pipelines** — errors ship to dashboards; encode
    checks and maintenance as code and schedule.
11. **Treating the warehouse as the system of record for raw data** — it should consume curated
    data, not duplicate the lake's source-of-truth role.

---

## 10. Relevance to This Project

The project's object store (`architecture/data-layout.md`, `backend/storage/keys.py`) is the
**system of record**; the warehouse is the anticipated **relational serving layer** built from
`metadata.json` and derived artifacts. This research clarifies how to keep that split clean.

**Already aligned**
- **Raw immutability and layered refinement.** `live/raw/` (verbatim source) preserved under
  `live/derived/` (clean-text, features, labels) mirrors the staging → integration →
  presentation progression. The `eval/` vs `live/` split keeps ground truth isolated, consistent
  with "never transform raw in place."
- **Schema-on-write + provenance.** Per-object `metadata.json`, hashed `configs/<hash>.json`, and
  UUIDv7 ids are already the "annotate at ingestion / capture lineage" discipline; a warehouse
  catalog would consume, not replace, this provenance.
- **Additive iteration.** The stated "iteration is additive" posture is exactly the
  idempotent, replayable ingestion the warehouse layer requires downstream.

**Where warehouse thinking should shape the next step**
- **`metadata.json` is a fact/dimension seed.** The anticipated Athena SQL layer is a
  *presentation-layer star*, not a mirror of the object keys. Model a **posting fact** at a
  declared grain (one row per posting per snapshot/version) and conformed **company**, **role/
  title**, **location**, and **date** dimensions. Declare the grain in writing before choosing
  columns — the object `id` is the degenerate/natural key, not the primary key.
- **Surrogate keys, not object ids, as FKs.** Use UUIDv7 as the business/natural key and add
  integer surrogates for the dimensions so SCD versioning works when a posting's company or
  title is corrected later. This keeps the warehouse insulated from object-store key changes.
- **Company/role dimensions will change — plan SCD now.** Employers get renamed/acquired and
  titles get renormalized; decide SCD type per attribute (e.g., Type 2 on canonical company and
  role family; Type 1 on display-only attributes). This pairs naturally with the deferred dedup
  problem (§4.1 of the data-layout note): the same posting re-ingested later is really a
  **new version / late-arriving** case, which the warehouse's own SCD and effective-dating
  machinery can model explicitly rather than treating as a content-hash duplicate.
- **One semantic layer for eval and live metrics.** "Separation gap", role distributions, and
  dataset-size claims already appear across `results.md`, `technical/evaluation.md`, and the
  experiment pages. As postings grow toward 350+, define these once in a semantic view so
  dashboards and reports cannot drift.
- **Start simple; escalate only on real pressure.** Given few sources (essentially one corpus)
  and a small team, **Kimball star schema is the correct default**. Data Vault is *not* indicated
  unless multiple independently-changing sources later need to merge into the same company/role
  entities — at which point an integration layer (vault-ish or otherwise) can be inserted
  underneath the existing stars without rewriting them.
- **Keep the warehouse derived and regenerable.** It should be rebuildable from the object
  store + configs at any time. That preserves the study's single-source-of-truth principle and
  avoids the classic lake-vs-warehouse duplication cost called out in report 009.

**Suggested next steps (not this sprint)**
1. Draft the presentation-layer star on paper for the posting domain: grain statement, fact
   measures, and conformed dimensions, with SCD types noted per attribute.
2. Define a `metadata.json` field contract so the eventual Athena/Glue table schema is a
   documented interface, not an ad-hoc projection.
3. Pick one metric (e.g., role distribution) and express its definition as the first
   semantic-layer entry, to prototype the "single definition" discipline early.
4. Only if/when a second volatile source appears, spike an integration layer (vault-ish or
   normalized) beneath the star and evaluate the added operational overhead honestly.

---

## 11. Sources

1. **Kimball Group** — "Dimensional Modeling Techniques." <https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/>
2. **Kimball Group** — "Late Arriving Dimension." <https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/late-arriving-dimension/>
3. **Kimball Group** — "Fistful of Flaws" (common dimensional design flaws). <https://www.kimballgroup.com/2003/10/fistful-of-flaws/>
4. **Ralph Kimball & Margy Ross** — *The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling*, 3rd ed., Wiley, 2013.
5. **Microsoft Learn** — "Dimensional modeling in Fabric Data Warehouse." <https://learn.microsoft.com/en-us/fabric/data-warehouse/dimensional-modeling-overview>
6. **Microsoft Learn** — "Exploring the Modern Data Warehouse" (data engineering playbook). <https://learn.microsoft.com/en-us/data-engineering/playbook/solutions/modern-data-warehouse/>
7. **Microsoft Learn** — "Implement Medallion Lakehouse Architecture in Fabric." <https://learn.microsoft.com/en-us/fabric/onelake/onelake-medallion-lakehouse-architecture>
8. **Databricks Blog** — "Practical Data Warehouse Design and Architecture Guide" (2026). <https://www.databricks.com/blog/data-warehouse-design>
9. **Databricks Blog** — "Data Warehouse Modernization: Roadmap, Architecture, and Services" (2026). <https://www.databricks.com/blog/data-warehouse-modernization>
10. **AWS Whitepaper** — "Modern Data Architecture: Rationales on AWS" (Kimball vs Inmon). <https://docs.aws.amazon.com/whitepapers/latest/modern-data-architecture-rationales-on-aws/modern-data-architecture-rationales-on-aws.html>
11. **AWS Well-Architected** — "Modern data architecture" (Analytics Lens). <https://docs.aws.amazon.com/wellarchitected/latest/analytics-lens/modern-data-architecture.html>
12. **Snowflake** — "Data Warehouse Architecture and Design: Best Practices." <https://www.snowflake.com/en/fundamentals/data-warehouse-architecture-and-design/>
13. **Keboola** — "Data Warehouse Design" (paradigms, layers, best practices, 2026). <https://www.keboola.com/blog/data-warehouse-design>
14. **DataCamp** — "Data Warehouse Architecture: Trends, Tools, and Best Practices." <https://www.datacamp.com/blog/data-warehouse-architecture>
15. **Fivetran** — "Modern Data Warehouses: Benefits And Migration Strategies." <https://www.fivetran.com/blog/modern-data-warehouse>
16. **Datawarehouseinfo.com** — "Dimensional modeling: a practitioner's guide" (2026). <https://datawarehouseinfo.com/foundations/dimensional-modeling/>
17. **Datawarehouseinfo.com** — "Star schema vs snowflake schema" (2026). <https://datawarehouseinfo.com/architecture/star-schema-vs-snowflake-schema/>
18. **Analytics Engineering** — "Data Warehouse Schema Design: Star Schema to Galaxy" (2026). <https://www.analyticsengineering.com/resources/data-warehouse-schema-design>
19. **dataarchitect.studio** — "Data Vault vs Dimensional Modeling: Which Belongs Where" (2026). <https://dataarchitect.studio/essays/data-vault-vs-dimensional-modeling/>
20. **ER/Studio Blog** — "What Is Data Vault Modeling? Complete Guide" (2026). <https://erstudio.com/blog/data-vault-modeling/>
21. **PuppyGraph** — "What Is Data Vault? Architecture & Modeling." <https://www.puppygraph.com/learn/what-is-data-vault>
22. **Levan Alibegashvili** — "Data Vault 2.0 in Practice: Lessons from Real Implementations" (2026). <https://levanalibegashvili.com/blog/data-vault-2-in-practice-lessons-from-real-implementations>
23. **arXiv 2606.29355** — "Enterprise Data Modelling Methodologies: A Comparative Analysis of Inmon, Kimball, and Data Vault" (2026). <https://arxiv.org/abs/2606.29355v1>
24. **Redgate** — "Five Common Dimensional Modeling Mistakes and How to Solve Them." <https://www.red-gate.com/blog/five-common-dimensional-modeling-mistakes-and-how-to-solve-them/>
25. **Databricks Blog** — "Handling Late Arriving Dimensions Using a Reconciliation Pattern." <https://www.databricks.com/blog/2020/12/15/handling-late-arriving-dimensions-using-a-reconciliation-pattern.html>
26. **martinsights** — "The Seven Deadly Sins of Data Warehouse Design." <https://www.martinsights.com/?p=1065>
