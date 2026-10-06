---
id: "011"
title: "Object Store Metadata Design & Best Practices"
date: "2026-10-04"
summary: "Survey of object-store metadata design — what metadata a cloud object store actually keeps (system vs custom, fixed-key vs user), how hyperscale metadata services scale (partitioning, sharding, tiering, distributed DB backends), key-namespace engineering (high-cardinality prefixes, time layout, delimiters, hot-spot avoidance), and the metadata-in-objects vs metadata-in-database trade-off — mapped onto this project's per-object metadata.json, UUIDv7 ids, and configs provenance branch."
---
# Object Store Metadata Design & Best Practices — Research Report

**Date:** 2026-10-04
**Purpose:** Establish the design principles for the metadata layer that sits beside — and increasingly
*behind* — the object keys this project already controls. The current design (`architecture/data-layout.md`,
`backend/storage/keys.py`) puts provenance in per-object `metadata.json` sidecars and queryable dimensions in
the key namespace. This report surveys how the hyperscalers and open-source object stores actually design
that metadata layer, so the project's `metadata.json` can scale into a relational/catalog layer (the planned
Athena SQL step, report 010) without rework.
**Companion reading:** Report [005 — Object Storage Naming Conventions](005-object-storage-naming-conventions.md)
covers the *key* (name) side. This report covers the *metadata* side — what to store alongside and behind the
keys, and how to scale and partition it. Reports [009](009-data-lake-design-best-practices.md) and
[010](010-data-warehouse-design-best-practices.md) cover the lake and relational layers that consume this
metadata. The anchor source is the System Design Handbook's "Object Storage System Design" guide (§2), which
frames metadata as the system's *control plane*.
**Sources:** System Design Handbook object-storage guide, AWS S3 / Google Cloud Storage / Azure Blob / Oracle /
Alibaba metadata documentation, Ceph / Apache Ozone / MinIO open-source implementations, and 2026 academic
papers on scalable metadata services (CFS, SoMeta, Ursa Minor, CephFS dynamic metadata management; see §10).

---

## 1. Executive Summary

An object's **key (name)** and its **metadata** are different things with different jobs. The key is a
flat, globally-addressable identifier; the metadata is everything the store (and your applications) must
know to locate, validate, protect, and manage the object. The recurring industry framing — repeated by the
System Design Handbook and every provider — is that **the metadata service is the hardest and most
performance-critical part of an object store**; scaling storage is comparatively easy, scaling *metadata*
is where systems fail. Seven principles organize the findings:

1. **Metadata is the control plane.** Every read/write consults it (find object, validate checksum, check
   policy, route to storage). It must be low-latency, high-throughput, and at least as durable as the data
   itself — corrupt metadata makes otherwise-present data unreachable.
2. **Separate the key from the metadata.** Put in the key only what must be queryable-by-prefix and
   lifecycle-selectable; push everything else (provenance, content hashes, custom attributes) into
   metadata. This is precisely the "keys encode what must be queried; metadata holds provenance" split this
   project already made in `data-layout.md` §2.
3. **Separate system metadata from user/custom metadata.** Providers maintain *system* metadata (size,
   timestamps, checksums, generation, storage class) automatically, and let users attach *custom*
   key-value metadata (S3 user-defined + tags, GCP custom, OCI custom) — each with hard size limits (S3
   2 KB user-defined; OCI 4 KB; OSS 8 KB). Design the boundary deliberately and budget for the limit.
4. **Make ids versionable.** A stable object **generation/version** (S3 `x-amz-version-id`, GCP
   generation + metageneration) is metadata, not part of the key. It enables safe overwrites, concurrency
   control, and recovery without renaming. This project's UUIDv7 id is the *identity*; versioning policy is
   a metadata concern to decide, not a key concern.
5. **Scale metadata by partitioning, not by a single store.** Hyperscale metadata services shard by hash,
   key-range, or dynamic subtree; increasingly they are backed by distributed databases (Spanner, ZippyDB,
   FoundationDB, TiKV) rather than hand-built servers. The design question is which partitioning matches
   your access pattern (locality vs. load-balance vs. transactional scope).
6. **Engineer the key namespace for partition spread and listing.** Put a high-cardinality dimension early
   in keys only when a hot prefix genuinely needs it; otherwise prefer Hive-style date partitions that a
   query engine (Athena) can prune. Avoid global hashing — it destroys prefix locality and breaks listing,
   lifecycle, and catalog tooling.
7. **Plan the catalog/table layer before scale.** Per-object JSON metadata becomes a small-files problem
   once the corpus grows. Move the queryable subset into an open table format (Iceberg, report 009) behind
   a catalog (Glue/Polaris) while *keeping the objects as the system of record* — exactly report 010's
   "warehouse is derived and regenerable" principle.

**Key recommendation for this project:** the existing split — UUIDv7 id in the key, provenance in
`metadata.json`, generation parameters in `configs/<hash>.json` — is already the correct control-plane
shape. The next step is to **declare the `metadata.json` field contract** (system vs. custom, which fields
are queryable, versioning semantics), add **an `x-amz-meta-*`-style custom-metadata side** for cheap
prefix-independent attributes, and **plan the migration to an Iceberg table + catalog** for the queryable
subset once relational querying is real — so the metadata layer scales by partitioning rather than by
growing one JSON-per-object store without an index.

---

## 2. What Object-Store Metadata Is (and Why It Is the Control Plane)

The System Design Handbook frames metadata as **the brain / control plane** of an object storage system: it
tracks where each object is stored, its attributes, and how it is accessed, and it sits on the read and
write paths of *every* request. Without efficient metadata, "even the most scalable storage layer becomes
unusable."

An object is a **self-contained package** with three parts: the **data** (opaque bytes), **metadata**
(attributes describing the data), and a **unique identifier (key)** used to address it. The metadata is the
part that makes the system operable — locate the object, verify integrity, enforce policy, apply lifecycle.

**The metadata table every provider keeps** (System Design Handbook; consolidated with GCP/AWS/Oracle):

| Metadata field | Purpose |
|---|---|
| Object ID / key | Unique address used to locate the object |
| Location | Storage node / shard reference |
| Size | Object size (for reads, billing, listing) |
| Timestamps | Creation / update / last-modified time |
| Checksum | Data-integrity validation (CRC32C, ETag, SHA) |
| Storage class | Durability/access tier (STANDARD, Glacier, Coldline, ...) |
| Generation / version | Distinguishes object versions; concurrency control |
| Access policy / ACL | Who may read / write / delete |

Two structural distinctions recur and are load-bearing for any design:

- **Flat namespace, no real hierarchy.** Keys are flat; folders are simulated with `/` as a delimiter and
  prefix-based listing. GCP objects are stored "directly under the bucket" with no real folders; S3's "folders"
  are zero-byte objects; Ozone's OBJECT_STORE layout stores full-path keys with *no intermediate directory
  entries*.
- **Metadata is consulted on essentially every request**, making it a high-throughput, low-latency component
  that must scale as object count grows — the "metadata explosion" problem of billions of small objects.

### 2.1 System vs. user-defined metadata

Every provider splits metadata into two categories with different ownership and limits:

- **System-defined metadata** — set and maintained by the service: content-type, size, storage class,
  timestamps, generation/metageneration (GCP), checksums, `x-amz-*` headers (S3). Not editable directly, or
  only with permission.
- **User-defined / custom metadata** — key-value pairs the user attaches at upload: S3 **user-defined**
  metadata (`x-amz-meta-*`, ≤ 2 KB) and **object tags** (≤ 10 per object), GCP **custom metadata**
  (`x-goog-meta-*`), OCI **custom metadata**, OSS **user metadata** (`x-oss-meta-*`, ≤ 8 KB), Azure metadata.
  Used for application context, cataloging, tags, and access control.

**Size budgets matter.** Providers cap custom metadata hard (S3 2 KB, OCI 4 KB total, OSS 8 KB) and *bill for
it* (GCP custom metadata incurs storage cost). This is a real design constraint: custom metadata is a
small, curated envelope — **not** a place to dump large or frequently-changing application state. The
project's `metadata.json` sidecar is an object in the store (essentially unbounded), so it can be richer;
but if it is ever mirrored into S3-style user-defined metadata, the 2 KB budget binds. Decide once, document
it (report §8).

### 2.2 Checksums and integrity metadata

Checksums are metadata calculated from object data and used to validate it isn't corrupted:

- **S3**: ETag (MD5-based), plus `x-amz-checksum-*` (CRC32, CRC32C, SHA-1, SHA-256) for integrity.
- **GCP**: every object carries a **CRC32C** hash; gcloud/SDK compute it.
- **OCI**: "actively monitors data integrity using checksums and automatically detects and repairs corrupt
  data," and monitors redundancy, auto-creating copies if a redundancy loss is detected.

This is the operational reason metadata must be *durable and replicated*: integrity metadata protects the
data, and corrupt/lost metadata defeats it.

### 2.3 Generation / versioning is metadata, not part of the key

- **GCP**: every object has a numeric **generation** (identifies the version) and **metageneration**
  (tracks metadata-only changes) — both metadata, not part of the name.
- **S3**: `x-amz-version-id` is a metadata value on each object version; deleting a versioned object is a
  soft delete until an expiration rule removes the old version.
- **System Design Handbook**: versioning lets the system keep multiple versions for recovery from
  accidental deletes/overwrites and to manage concurrent updates; updates in object storage are
  **replace-the-whole-object**, never in-place edits — so versioning is the primary safety mechanism.

**Relevance to this project:** the UUIDv7 posting id (`keys.py`) is stable *identity*; the question of
whether re-ingesting an identical posting is a **new version** or a **duplicate** is a metadata/policy
decision (SCD/late-arriving per report 010 §4.7, and the deferred-dedup problem in `data-layout.md` §4.1).
Design the `metadata.json` contract to carry a version/last-modified dimension from day one so that decision
isn't forced into the key namespace later.

> **Scope note — this claim is about *object generation*, not `vN` artifact versions.** "Versioning is
> metadata, not part of the key" refers to the object-store concept of *overwrite history* (S3
> `x-amz-version-id`, GCP `generation`): the same object re-written, kept as recoverable versions.
> This project's `keys.py` deliberately *does* put a different, orthogonal version token in the keys —
> `live/derived/<id>/<variant>/v<N>.json` and `live/embeddings/<model>/<variant>/v<N>/<id>.npy`, enforced
> by `_VERSION_RE` (`keys.py:71-72`, `data-layout.md` §2.4). That `vN` is the **iteration of a derived
> artifact variant** (e.g. `clean-text/v2`), an addressable dimension the team wants to list/query by
> prefix and tie to `configs/<hash>.json` — so it *belongs* in the key, per this report's own §5.2 rule
> ("keys = what must be prefix-queryable"). The two concepts are compatible: `vN` identifies which
> derived artifact to address; object generation records that a posting record was overwritten. Neither
> conflicts with keeping the posting `id` itself stable and versioning *policy* in `metadata.json`.

---

## 3. How Hyperscale Metadata Services Scale: Partitioning and Tiering

The System Design Handbook's core scalability claim: **a centralized metadata store is a hard limit at
scale.** "Scaling metadata is one of the biggest challenges in object storage design. As the number of
objects grows, the metadata service must handle increasing load while maintaining performance and
reliability." The remedy is partitioning + distribution + (modern) a distributed DB backend.

### 3.1 Partitioning strategies

The handbook's partitioning taxonomy for the data plane, which generalizes to the metadata plane:

| Strategy | Description | Trade-off |
|---|---|---|
| **Consistent hashing** | Distributes objects/metadata evenly across nodes | Requires rebalancing on node add/remove |
| **Key-based sharding** | Uses object-ID ranges | Risk of hotspots if keys aren't spread |
| **Bucket-based partitioning** | Groups objects logically | Less flexible; coupling to bucket layout |

The same tension recurs in the distributed-filesystem literature (which scales metadata *namespaces*):

- **Subtree partitioning** (CephFS/Lustre): assign directory subtrees to metadata servers → good locality,
  but workload growth *by depth* creates a single-server hotspot (a "near-root" hotspot for path resolution).
- **Dynamic subtree partitioning** (CephFS's modern answer): migrate load-intensive subtrees from busy to
  idle MDSs → relieves hotspots as they appear; supports vertical in addition to horizontal scaling.
- **Hash/DHT partitioning** (SoMeta, GIGA+): distribute metadata objects by hash for load balance; at the
  cost of locality — range/list scans become expensive. SoMeta uses a **two-level hash** (namespace →
  server, then server → hash-table slot) and partitions a *flat namespace* across servers via a DHT.
- **Access-content decoupling** (InfiniFS): separate *directory-access* metadata (goes with the parent, for
  locality of path resolution) from *content* metadata (goes with the children, hash-partitioned for load
  balance). The insight generalizes: **different kinds of metadata favor different partitioning.**
- **Tiered metadata** (CFS, 2023): keep **file attributes** close to their data on storage nodes, and use a
  **distributed database layer** for the namespace hierarchy. Attributes are hash-partitioned (balance, no
  hotspots); the namespace uses locality-preserving partitioning to avoid cross-shard transactions. This
  relieves the database layer and precludes hotspots — "two kinds of metadata are not treated equally."

**2026 trend (SNIA, 2025):** metadata services increasingly **stop hand-building servers and back onto a
distributed database** — Google Colossus (Spanner), Microsoft Cosmos (SQL Server), Facebook Tectonic
(ZippyDB), DeepSeek 3FS (FoundationDB), JuiceFS (TiKV/FoundationDB/Postgres), CERN EOS (QuarkDB). The
database supplies sharding, replication, and recovery; the storage system supplies the access pattern. This
is the architectural end-state this project's "metadata.json → Athena SQL" path is gesturing at.

### 3.2 Implications for design

- Choose partitioning by **access pattern**: prefix/list-heavy workloads want locality-preserving layouts;
  write-heavy/append workloads want hash spread to avoid hotspots.
- **Tier**: separate frequently-consulted, small attributes (store near/on the data) from the namespace
  hierarchy (distributed DB). Don't force one store to serve both.
- Expect **rebalancing** cost when adding nodes; design metadata keys so they can be redistributed without
  rewriting the data.

---

## 4. Open-Source Object-Store Metadata Layouts (Ceph, Ozone, MinIO)

Open implementations are concrete examples of the taxonomy above — worth studying because they separate
*metadata*, *bucket index*, and *data* explicitly.

### 4.1 Ceph Object Gateway

Ceph RG organizes everything into three conceptual kinds: **metadata, bucket index, data**.

- **Metadata** lives in three namespaces: `user`, `bucket` (name → bucket instance ID), and `bucket.instance`
  (instance info). Each object's "head" holds the manifest, ACL, content-type, ETag, and user-defined
  metadata in **xattrs**; the head may also inline up to 512 KB of object data for efficiency and atomicity.
- **Bucket index** is a key-value **OMAP** (object map) per bucket: key = object name, value = the listing
  metadata (what appears when you list the bucket), plus a header with bucket accounting (object count,
  total size).
- **Data** is the actual object bytes in RADOS pools; multiple data pools let different users' buckets
  land in different pools for scaling.
- Deletion uses **garbage collection (GC)**: the object is marked deleted, and a background process purges
  the space — soft-delete-then-reap, exactly the handbook's "deletion is not as simple as removing data."

**Takeaway:** the *bucket index* is a separate, listing-optimized structure from per-object metadata. A
"list my bucket" scan reads the OMAP, not a scan of every object — an important distinction this project
can mirror if it ever needs cheap bucket-level listings/aggregates.

### 4.2 Apache Ozone

Ozone supports three bucket layouts: **OBJECT_STORE (OBS)** (default for S3 Gateway, S3-compatible),
**FILE_SYSTEM_OPTIMIZED (FSO)**, and **LEGACY**. In OBS, keys are stored **with their full path names as
single entries in a flat namespace** — no intermediate directory entries; `dir1/dir2/file.txt` is just one
key. Listing is **prefix-based**; deleting one key never affects others that share a prefix. FSO creates
directory entries and normalizes paths, trading S3 compat for filesystem semantics.

**Takeaway:** even within one product the "flat key = single entry" vs "hierarchical with directory entries"
choice is a *layout decision* with real performance consequences for prefix listing and directory operations.
This project's flat `live/<category>/<id>/...` keys are the OBS-flat shape; fine at current scale, but a
prefix-list-heavy future may want FSO-style directory accounting.

### 4.3 MinIO and S3 semantics

MinIO inherits S3 semantics: bucket names become subdomains under virtual-host lookup (`MINIO_DOMAIN`),
object names can't contain `\` or `:` (on Linux), `/` creates folder structure, and lifecycle rules are
S3-compatible. The S3 metadata model (system headers + user `x-amz-meta-*` + tags) carries over directly,
reinforcing that a store's metadata contract is largely *the S3 contract*.

---

## 5. Key-Namespace Engineering: Prefixes, Time Layout, Hot Spots, and the 3,500/5,500 Limit

This is the metadata-via-key boundary: the key encodes the *addressable, prefix-queryable* subset of
metadata. Getting it wrong produces hot partitions, slow listing, and broken query pruning — even with a
perfect metadata DB behind it. (Deep coverage of naming rules lives in report 005; this section is the
*performance* side.)

### 5.1 The per-prefix throughput limit

- **S3**: at least **3,500 PUT/COPY/POST/DELETE and 5,500 GET/HEAD requests per second per partitioned
  prefix**, with **no limit on the number of prefixes**. Scaling happens gradually; during scale-up you may
  see transient HTTP **503 (Slow Down)**.
- **GCP**: sequential keys slow key-range-index auto-scaling and can cause elevated latency/errors.
- **Azure Blob**: partition key = `account + container + blob name`; append-only/prepend-only keys
  concentrate traffic on one partition server and can exceed scalability targets.

### 5.2 The modern key-design guidance (post-2018)

Older S3 guidance told you to *always* hash-prefix keys. **That advice was superseded** (S3 now
auto-partitions and can handle sequential date prefixes), and 2026 guidance is more surgical:

- **Goal:** give S3 enough **lexicographic spread** that auto-partitioning has somewhere to cut. Sequential
  keys (`00000001`, ...) share long common prefixes and give S3 nowhere to cut; diverse keys spread fine.
- **Prefer natural high-cardinality leading segments.** Put a high-cardinality dimension *early* — a tenant
  ID, region code, or (as a last resort) a hash prefix — **only when you genuinely need beyond-single-prefix
  throughput on a known hot prefix**. This is *not* a default for every bucket.
- **Sharding patterns, in increasing destructiveness:**
  - *A. Natural high-cardinality leading segment* (tenant/region first) — preserves locality.
  - *B. Digit reversal* — when the only natural key is sequential (DB row IDs, Kafka offsets), reversing the
    digits (`000000007654321` → `123456700000000`) spreads leading bytes across partitions while still
    grouping by ID; **only when range queries on the original ID are not on the access path** (they become
    full scans).
  - *C. Hex-prefix sharding* — prepend a fixed-length hash; max spread but destroys locality; queries
    against the hash prefix cannot prune (Athena scans all buckets).
- **Anti-shard pattern:** do **not** wrap your whole application in a layer that hashes every key globally —
  it destroys lexicographic locality for every prefix and breaks Athena/Glue pruning, lifecycle rules, and
  replication-rule design. **Sharding is a targeted tool, not a wholesale architectural choice.**

### 5.3 Time-series layout: Hive style vs project style

Most data lakes are time-partitioned, and the two ways to encode time are *not* equivalent:

- **Hive style** (`year=2026/month=05/day=06/`): `MSCK REPAIR TABLE` discovers partitions automatically;
  query engines prune on the key-value pairs. The default for catalog-driven analytics.
- **Project style** (`2026/05/06/`): requires explicit `ADD PARTITION` statements for non-Hive layouts.

**Rules of thumb:** never mix the two layouts in one bucket (a bucket with both `year=...` and `2026/...` is
unqueryable as a single table); keep total key length well under 256 bytes (leaving headroom for metadata
systems that cap keys more tightly than S3's 1,024); key length is **1,024 bytes, not characters** (multibyte
UTF-8 costs 2–4 bytes/char); keys are case-sensitive and **Unicode is not normalized** — pick one
normalization form (NFC or NFD) and enforce it at the producer, or the "same" name becomes two keys.

### 5.4 Hot objects and listing

- **Hot objects** (disproportionately popular) create bottlenecks even in balanced systems; handle via
  caching, replication of the hot object, and load balancing.
- **Small files** stress both storage and metadata (per-object overhead); handle via aggregation/batching and
  a separate table format for the queryable subset (reports 009/010).
- **Listing** in a flat bucket is O(keys in range); use the `/` delimiter in `ListObjectsV2`/`list` calls to
  render one level and roll up `CommonPrefixes`, and keep hierarchies shallow.

---

## 6. Durability, Consistency, and Security of the Metadata Layer

### 6.1 Metadata durability matches data durability

The handbook is explicit: **"metadata durability is just as important as data durability."** If metadata
becomes corrupted or inconsistent, objects may become inaccessible even though the data still exists.
Mitigations: replicate metadata; back it with a distributed/consensus DB; checksum it. Ceph FS stores
metadata as log-first (fast commits, quick recovery) then commits directory contents to an object pool;
OCI monitors metadata redundancy and auto-repairs.

### 6.2 Consistency: strong vs. eventual

- **Strong consistency** — once a write is acknowledged, all reads see it (S3, Oracle OCI, most modern
  object stores). Higher coordination/latency.
- **Eventual consistency** — replicas converge eventually; faster, but temporary inconsistency.

Most object stores ship strong consistency *per object* while allowing eventual semantics across
multi-object operations. Versioning and generation numbers are how they make concurrent updates predictable
(last-write-wins vs. version tracking). The handbook recommends defining concurrency policy explicitly.

### 6.3 Security & multi-tenant isolation

Metadata carries the security surface: **authentication** (tokens/API keys), **authorization** (bucket- and
object-level policies), **encryption at rest and in transit** with key management, and **multi-tenant
isolation** (strict boundaries between tenants across data, metadata, and access controls). GCP/AWS both warn
that *names and metadata* are more visible than data — avoid PII in keys (report 005 §6) and be deliberate
about what custom metadata is queryable by whom.

---

## 7. Best Practices Checklist

**Metadata content**
- [ ] Separate system vs. user/custom metadata; document each field's purpose.
- [ ] Budget custom metadata against provider limits (S3 2 KB / OCI 4 KB / OSS 8 KB) and bill-ability.
- [ ] Carry integrity metadata (checksum/hash) and validate it; replicate metadata like data.
- [ ] Make version/generation an explicit metadata field, not a key suffix.

**Key namespace**
- [ ] Put in the key only what must be prefix-queryable / lifecycle-selectable; push the rest to metadata.
- [ ] Prefer Hive-style date partitions (`year=/month=/day=`) for catalog-driven analytics.
- [ ] Add high-cardinality/hash prefix *only* for a known hot prefix; never hash every key globally.
- [ ] One normalization form (NFC/NFD) enforced at the producer; keys < 256 bytes; no mixed time layouts.
- [ ] Use `/` delimiter for hierarchical listing; keep hierarchies shallow.

**Scaling the metadata service**
- [ ] Partition metadata by access pattern; consider tiering attributes (near data) vs. namespace (DB).
- [ ] Back the metadata service with a distributed DB (Spanner/ZippyDB/FoundationDB/Postgres) as it grows.
- [ ] Keep a separate listing-optimized index (bucket-index/OMAP-style) if cheap bucket listings are needed.
- [ ] Plan the catalog/table layer (Iceberg + Glue/Polaris) for the queryable subset before the small-files
    problem sets in.

---

## 8. Anti-Patterns

1. **One giant per-object metadata store with no index.** JSON sidecars with no catalog/table layer
   become the small-files problem once the corpus grows past thousands.
2. **Encoding everything in the key.** Keys bloat, mixing many orthogonal dimensions that break prefix
   locality, push past the 1,024-byte/256-byte guidance, and entangle lifecycle/listing rules.
3. **Global key hashing.** Destroys prefix locality for every object, breaks Athena/Glue pruning, lifecycle,
   and replication-rule design — the 2026 anti-shard pattern.
4. **Sequential-key hot partitions.** Timestamp/auto-increment keys funnel writes to one partition (503s,
   GCP/Azure scale stalls).
5. **Centralized, single-node metadata service at scale.** A hard scaling ceiling and single point of
   failure for the whole read/write path.
6. **Not versioning.** Without a generation/version metadata field, overwrites are unrecoverable and
   concurrent updates are unpredictable.
7. **Treating metadata as optional/second-class.** No integrity checksum, no replication → corrupt metadata
   makes present data unreachable (the handbook's core warning).
8. **Leaking PII into keys or custom metadata.** Keys and metadata are more visible than data (URLs, logs,
   listings); sensitive values must move into secured fields, not names.
9. **Over-partitioning / mixing time layouts.** Millions of tiny partitions or mixed `year=/` and `2026/`
   layouts are un-compactable and unqueryable as a single table.

---

## 9. Relevance to This Project

The project's object store (`architecture/data-layout.md`, `backend/storage/keys.py`) already embodies the
key metadata-design principles this research surfaced. The alignment and the next step are both clear.

**Already aligned**
- **Key/metadata split is correct.** `keys.py` encodes only queryable dimensions (category, id, variant,
  version, model) and keeps provenance (content hash, source URL, employer, title, ingestion time) in the
  append-only `metadata.json` sidecar — exactly the "keys = addressable/queryable; metadata = provenance"
  separation the providers and the handbook recommend (§2, §5.2). The `vN` version token in the keys is
  consistent with this: it is the *addressable iteration* of a derived variant (§2.3 scope note), not the
  object's overwrite history.
- **Identity is stable and versioned-by-design.** UUIDv7 ids are minted at ingestion, not derived from
  content; the id is identity, and the "same posting re-ingested = new version or duplicate" question is a
  metadata/policy decision (report 010 §4.7, `data-layout.md` §4.1) — not a key question.
- **Config hashing is a metadata-by-provenance pattern.** `configs/<hash>.json` is the project's analogue of
  a catalog's lineage/transformation record — metadata held *as an object* rather than bloating keys.
- **A dedicated metadata namespace exists.** Top-level `configs/` and per-posting `metadata.json` give the
  project a control-plane structure that a catalog can later consume rather than replace.
- **The Athena path is already anticipated.** `data-layout.md` §9 names the relational layer over the
  metadata sidecars; reports 009/010 map it to the lake/warehouse side. This report covers the *metadata* side
  of that same path.

**Recommended reinforcements**
1. **Declare the `metadata.json` field contract now.** Write a `schemas/` (or `architecture/` note)
   defining which fields are *system* (always present: id, ingestion time, source URL, content hash) vs.
   *user/custom* (employer, title, tags), which are **queryable/partitioning** (the Athena predicates) vs.
   descriptive-only, and **versioning semantics** (per-report-010 SCD/late-arriving policy). This makes the
   future Glue/Iceberg table a documented interface, not an ad-hoc projection.
2. **Add a custom-metadata side for cheap attributes.** Mirror the `x-amz-meta-*` pattern: keep tiny,
   prefix-independent attributes (employer, title, date) as user-defined metadata so HEAD/catalog tooling can
   read them without opening the `metadata.json` object — while keeping the *rich* provenance in the sidecar.
   Watch the 2 KB budget; treat the sidecar as the authoritative store and any custom-metadata copy as derived.
3. **Plan the migration to Iceberg + catalog for the queryable subset** (not this sprint). When
   relational querying is real and the corpus grows toward 350+, move the queryable `metadata.json` subset
   into an **Iceberg table behind a catalog (Glue/Polaris)** — keeping the objects and the full sidecar as the
   system of record (report 010's "derived and regenerable"). This avoids the small-files / un-indexed-JSON
   anti-pattern (§8.1) by partitioning *metadata* instead of growing one store.
4. **Treat hot-spot/hot-object handling as future-proofing, not today's concern.** At this scale the flat
   UUIDv7-key layout is fine; when (if) one prefix (e.g. a single employer's `derived/`) becomes hot, apply
   the targeted high-cardinality-prefix fix — never global hashing.
5. **Pick one Unicode normalization for keys now.** `keys.py` accepts lowercase `[0-9a-f-]` ids, but derived
   variant names and future natural-language segments should enforce a single normalization form to avoid
   NFC/NFD key splits (§5.3).

**Suggested next steps (not this sprint)**
1. Draft the `metadata.json` field contract in a `schemas/` note: system vs. custom, queryable vs. derived,
   versioning policy, and which fields the Athena/Glue table will expose.
2. Add the custom-metadata read path (employer/title/date as user metadata) to `keys.py`/ingestion and a
   HEAD-style lookup, and unit-test it against the existing key builders.
3. Spike a local Iceberg table over a slice of `live/derived/` metadata to validate the partition/catalog
   path and maintenance runbook (links to report 009's next-step #2).

---

## 10. Sources

1. **System Design Handbook** — "Object Storage System Design: A Complete Guide To Building Durable Storage
   Systems" (2026). <https://www.systemdesignhandbook.com/guides/object-storage-system-design/>
2. **Google Cloud** — "Object metadata" (system vs. custom, generation/metageneration, checksums).
   <https://docs.cloud.google.com/storage/docs/metadata>
3. **Google Cloud** — "About Cloud Storage objects" (flat vs. hierarchical namespace, name limits, request-rate
   guidance). <https://docs.cloud.google.com/storage/docs/objects>
4. **Google Cloud** — "Request rate and access distribution guidelines."
   <https://docs.cloud.google.com/storage/docs/request-rate>
5. **AWS S3** — "Working with object metadata" (system-defined, user-defined, tags, annotations).
   <https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingMetadata.html>
6. **AWS S3** — "Optimizing performance with S3" (3,500/5,500 per-prefix limits; superseded hash-prefix advice).
   <https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html>
7. **AWS S3** — "Organizing objects using prefixes" (prefix + delimiter listing, CommonPrefixes).
   <https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-prefixes.html>
8. **AWS S3** — "Naming Amazon S3 objects" (key length in bytes, case sensitivity, sort order).
   <https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html>
9. **AWS Whitepaper** — "Best Practices Design Patterns: Optimizing Amazon S3 Performance."
   <https://d1.awsstatic.com/whitepapers/AmazonS3BestPractices.pdf>
10. **Hidekazu Konishi** — "Amazon S3 Object Key Design Best Practices — Performance and Partitioning" (2026;
    high-cardinality prefixes, Hive vs. project time layout, sharding patterns A/B/C, anti-shard).
    <https://hidekazu-konishi.com/entry/amazon_s3_object_key_design_best_practices.html>
11. **Oracle Cloud Infrastructure** — "Overview of Object Storage" (strong consistency, durability, custom
    metadata, 4 KB limit). <https://docs.oracle.com/en-us/iaas/Content/Object/Concepts/objectstorageoverview.htm>
12. **Alibaba Cloud OSS** — "Manage object metadata — HTTP headers and user metadata" (8 KB user metadata,
    `x-oss-meta-*`). <https://www.alibabacloud.com/help/en/oss/user-guide/manage-object-metadata-10>
13. **Azure Blob Storage** — "Optimize blob partitions" (partition key = account+container+blob; sequential
    names). <https://learn.microsoft.com/en-us/azure/storage/blobs/storage-performance-blob-partitions>
14. **Ceph** — "Ceph Object Gateway data layout" (metadata, bucket index/OMAP, data; xattrs; GC).
    <https://www.ibm.com/docs/en/storage-ceph/8.0.0?topic=gateway-ceph-object-data-layout>
15. **Apache Ozone** — "Object Store Buckets (OBS)" (flat full-path keys, no directory entries, prefix
    listing; FSO vs OBS). <https://ozone.apache.org/docs/next/core-concepts/namespace/buckets/layouts/object-store>
16. **MinIO** — "Object management" (S3 semantics, virtual-host bucket names, forbidden characters).
    <https://minio.community/community/minio-object-store/administration/object-management.html>
17. **CephFS** — "Dynamic Metadata Management" (dynamic subtree partitioning, hotspot migration).
    <https://docs.ceph.com/en/latest/cephfs/dynamic-metadata-management>
18. **Sage A. Weil et al.** — "Dynamic Metadata Management for Petabyte-scale File Systems" (metadata-server
    partitioning for object-based storage). <https://ceph.io/assets/pdfs/weil-mds-sc04.pdf>
19. **Tang/Byna et al.** — "SoMeta: Scalable Object-centric Metadata Management for High Performance
    Computing" (DHT/two-level-hash flat-namespace metadata). <https://sdm.lbl.gov/pdc/pubs/201709_SoMeta_Cluster_Tang_Byna.pdf>
20. **CMU PDL** — "Building a High-Performance Metadata Service by Reusing Partitioning..." (GIGA+/TableFS;
    log-structured metadata, hash partitioning). <https://www.pdl.cmu.edu/PDL-FTP/HECStorage/CMU-PDL-13-107.pdf>
21. **Sinnamohideen et al.** — "A Transparently-Scalable Metadata Service for the Ursa Minor Storage System"
    (CMU/USENIX ATC'10; metadata/data path separation). <https://www.usenix.org/legacy/events/atc10/tech/full_papers/Sinnamohideen.pdf>
22. **Lv et al.** — "InfiniFS: An Efficient Metadata Service for Large-Scale Distributed Filesystems"
    (USENIX FAST'22; access-content decoupled partitioning). <https://www.usenix.org/system/files/fast22_slides_lv.pdf>
23. **ACM** — "CFS: Scaling Metadata Service for Distributed File Systems" (2023; tiered metadata, attributes
    near data + distributed DB for namespace). <https://dl.acm.org/doi/pdf/10.1145/3552326.3587443>
24. **SNIA SDC 2025** — "Scalable Metadata in Distributed File Systems" (survey: CephFS/Lustre/HDFS +
    Spanner/ZippyDB/FoundationDB backends; consistent-hashing ring). <https://www.snia.org/sites/default/files/2025-09/SNIA-SDC25-Modrzyk-Scalable-Metadata-in-Distributed-File-Systems.pdf>
