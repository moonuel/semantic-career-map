# AWS S3 Vectors — Research Report

> Date: 2026-10-26
> Context: Semantic Career Mapping Platform — evaluation of Amazon S3 Vectors against this project's embedding data model ([Data Architecture §7](../architecture/data.md#7-embeddings)) and the [Baseline Embedding experiment](../experiments/001-baseline-embedding.md) lineage (`all-MiniLM-L6-v2`, 384-dim, L2-normalized). Complements the generic tool catalog in [Report 014](014-semantic-search-tools-architecture.md), which decided against hosted vector stores at MVP scale; this report evaluates the one AWS-native candidate that could eventually serve the rebuildable vector-index slot.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What S3 Vectors Is](#2-what-s3-vectors-is)
3. [Service Facts and Limits (GA)](#3-service-facts-and-limits-ga)
4. [Pricing and Cost at Relevant Scales](#4-pricing-and-cost-at-relevant-scales)
5. [Fit Against the Data Architecture](#5-fit-against-the-data-architecture)
6. [Relevance to the Baseline-Embedding Line of Work](#6-relevance-to-the-baseline-embedding-line-of-work)
7. [Position in the Retrieval Graduation Ladder](#7-position-in-the-retrieval-graduation-ladder)
8. [Risks and Open Questions](#8-risks-and-open-questions)
9. [Decision Log](#9-decision-log)
10. [Sources](#10-sources)

---

## 1. Executive Summary

**What it is.** S3 Vectors (GA 2025-12-02, 14 AWS Regions) adds a *vector bucket* type to S3 — indexes of `(key, float32[1–4096], metadata)` entries queried natively with approximate-nearest-neighbor search. Index geometry (dimension, distance: cosine/euclidean) is frozen at creation, consistency is strong, and IAM/ABAC/KMS carry over — but vectors are not S3 objects: no object APIs, no lifecycle rules, no Athena/Glue surface (only a custom federated connector), and no export path besides `ListVectors`. Billing is three-axis: $0.06/GB-month storage, $0.20/GB PUT, $2.50 per million queries plus tiered data-processed and data-returned charges.

**Relevance to this project.** At every plausible corpus size the cost is negligible (a 500-posting corpus serves for ~$0.03/month; a hypothetical 1M-posting corpus for ~$0.20/month), so the decision is purely architectural. S3 Vectors can *only* occupy the "rebuildable acceleration structure" slot that [data.md §7](../architecture/data.md#7-embeddings) already reserves next to the Parquet embeddings dataset — never the source of truth, never the raw/artifacts/eval branches. The fit is clean where it matters: `key = source_hash` preserves the §2.2 identity discipline; frozen geometry naturally means one index per model/variant/version (§2.4); the §10 snapshot manifest pins the serving index via `vector_index_arn`; and vector metadata should carry identity and serving filters only — not text.

**Verdict.** Not adopted now. Exact in-process NumPy matmul stays — microseconds vs. a ~100 ms latency floor, and S3 Vectors' ~90% ANN recall must stay out of the exact-retrieval paths the evaluation methodology depends on ([Report 014](014-semantic-search-tools-architecture.md)). It is registered as the **serverless rung** of the Report 014 graduation ladder: adopt when (a) retrieval serves from the Lambda/Fargate tier, (b) the corpus reaches ~10⁶ vectors and breaks in-process brute force, and (c) query volume stays low enough that per-query pricing beats an always-on cluster. Until all three hold, S3 Vectors buys nothing but a new API surface.

---

## 2. What S3 Vectors Is

Announced at preview July 16, 2025 (re:Inforce); **generally available December 2, 2025** in 14 AWS Regions. S3 Vectors is a new **bucket type** — the *vector bucket* — with a dedicated API surface and its own IAM namespace (`s3vectors`). Purpose: durable, elastic, pay-per-use vector storage and native approximate-nearest-neighbor (ANN) query for AI agents, RAG, inference, and semantic search, claimed at up to ~90% lower cost than purpose-built vector databases.

The hierarchy is `vector bucket → vector index → vector entries`, where an entry is `(key: string, data: float32 vector, metadata: JSON)`. Key properties:

- **Not S3 objects.** Vectors are not stored as general-purpose S3 objects and are not visible to `GetObject`/`ListObjects` — the vector bucket lives in a separate service namespace with its own endpoint (`s3vectors.<region>.amazonaws.com`). The only ways in or out are the `s3vectors` APIs (`PutVectors`, `GetVectors`, `ListVectors`, `DeleteVectors`, `QueryVectors`).
- **Immutable index geometry.** Dimension, data type (`float32`), and distance metric (cosine or euclidean) are fixed at index creation — changing any of them means a new index. Notably: no dot product; cosine covers our L2-normalized case.
- **Vector-keyed upserts.** `PutVectors` overwrites by key; `DeleteVectors` removes by key. There is no versioning, no object lock, no lifecycle policies, no storage classes — one flat, mutable, strongly consistent index per name.
- **Strong consistency.** After a write, reads (including `ListVectors`) reflect it immediately. Overwritten/deleted vectors leave query results immediately but remain billed in storage for up to a day.
- **Metadata filtering.** Every vector carries up to 50 metadata keys. `ENHANCED` index mode applies filters *before* the search (pre-filter — high recall even for selective filters); `CLASSIC` mode filters during search. Filterable metadata ≤ 2 KB/vector; additional non-filterable metadata can be bulk (≤ 40 KB total/vector).
- **Integrations.** Amazon Bedrock Knowledge Bases uses S3 Vectors as a first-class vector store (chunk → Titan embedding → vector index, text kept as non-filterable `AMAZON_BEDROCK_TEXT` metadata); OpenSearch Serverless can offload infrequently-accessed vectors to it as a warm tier; boto3/AWS SDKs and SageMaker AI notebooks speak the API directly.

The mental model that matters here: **S3 Vectors is a hosted ANN index with durable storage semantics and SQL-free, serverless billing — not a data lake**. The vector bucket does not appear in, and cannot be reconciled with, the Athena/Glue lakehouse surface described in [data.md](../architecture/data.md).

## 3. Service Facts and Limits (GA)

| Property | Value | Note for this project |
|---|---|---|
| Vectors per index | up to 2 billion (GA; 50M in preview) | 40× headroom beyond anything on the roadmap |
| Vector indexes per bucket | 10,000 | one index per model/variant/version trivially fits §7 |
| Vector buckets per region/account | 10,000 | one serving bucket suffices for every model/variant/version |
| Dimensions | 1–4,096 | 384 (`all-MiniLM-L6-v2`) and 768+ (bge/mpnet alternates) all fit |
| Distance metric | cosine or euclidean | cosine = our L2-normalized protocol |
| Metadata per vector | ≤ 40 KB total, ≤ 50 keys, ≤ 2 KB filterable, ≤ 10 non-filterable keys/index | far above our ~0.5 KB provenance payload |
| `PutVectors` batch | ≤ 500 vectors/call, ≤ 1,000 req/s or 2,500 vectors/s per index | one batched call per flushed corpus |
| Query | `topK` ≤ 10,000 (pagination ≤ 100), ≤ 100 filter constraints (`ENHANCED`), hundreds of QPS per index | `topK` far above our needs |
| Latency | infrequent queries < 1 s; warm queries ~100 ms or less | vs. microseconds for in-process exact matmul |
| ANN recall | "90%+ average recall for most datasets" (AWS) | below the ~0.99 HNSW tier of [Report 014 §3](014-semantic-search-tools-architecture.md#3-ann-index-algorithms); ANN noise must stay out of evaluation |
| Consistency | strong (query, get, list) | re-ingest → immediately queryable |
| Encryption | SSE-S3 default; SSE-KMS optional (bucket-default CMK, per-index CMK) | per-index keys enable isolation per dataset version |
| Export | `ListVectors`, ≤ 1,000/page, up to 16 parallel segments | the only path back from index to Parquet |
| Regions (GA) | 14: us-east-1/2, us-west-2, ca-central-1, eu-west-1/2/3, eu-north-1, eu-central-1, ap-south-1, ap-northeast-1/2, ap-southeast-1/2 | the migration roadmap's candidate regions covered |

## 4. Pricing and Cost at Relevant Scales

Three billable axes (us-east-1/eu-west-1):

| Axis | Price |
|---|---|
| Storage | $0.06/GB-month logical (vector data at 4 B/dim + key at 1 B/char + metadata) |
| PUT | $0.20/GB logical uploaded, minimum 128 KB per PUT (batch vectors to amortize) |
| QueryVectors | $2.50 per million calls + data-processed tiers by index size ($0.004/TB for the first 100K vectors, $0.002/TB 100K–10M, $0.0004/TB above 10M — reduced up to 80% in June 2026) + $0.01/GB returned (first 512 KB/query free, ≥ 256 B/result) |

Note the storage rate sits **~2.6× S3 Standard** ($0.023/GB) — vector buckets are cheaper than *vector databases*, not than object storage. Lifecycle tiering is unavailable on vector buckets.

Cost at the two scales that matter to us — vector ≈ 1.5 KB data + 0.07 KB key (`source_hash`, 64 hex chars) + ~0.4 KB filterable metadata ≈ **2 KB**:

| Scenario | Storage | One full reload (PUT) | 10K queries/month |
|---|---|---|---|
| Today's corpus, projected ~500 postings | 1 MB → **$0.00/mo** | $0.0004 | $0.025 + ~$0.00002 processed → **≈ $0.03/mo** |
| Hypothetical scraper-scale corpus, 1M postings | 2 GB → **$0.12/mo** | $0.40 | **≈ $0.07/mo** |

The conclusion is unusual: **cost is never the deciding variable at any plausible scale of this project.** A 1M-posting corpus served by a serverless vector index is a nickel-scale annual cost; even 10M queries/month against a 1M-vector index total under $150/mo. The real trade is architectural — latency floor, API surface, and the two-representation problem — and is analyzed in §5–§7.

## 5. Fit Against the Data Architecture

Mapping S3 Vectors onto [data.md](../architecture/data.md), branch by branch:

| Concern | Fit |
|---|---|
| **Source of truth (§7)** | ✅ unchanged — Parquet under `embeddings/model=<model>/variant=<variant>/version=<vN>/` remains authoritative. S3 Vectors can only ever play the *"rebuildable acceleration structure"* role that §7 already names ("Any vector index (FAISS, or whatever scale eventually demands)… never the source of truth"). It is one candidate for the `FAISS` slot, not a rival to the dataset. |
| **Identity (§2.2)** | ✅ clean — the vector key is a plain string; `key = source_hash` (64 hex chars) makes every vector addressable by posting identity, and `GetVectors`/`DeleteVectors`/`PutVectors` all speak source_hash with zero join-handle drift. Metadata caries `source`, `source_id`, `role_category`, `model`, `variant`, `version` (filterable) for `ENHANCED` pre-filtered queries. |
| **Versions (§2.4)** | ⚠️ via naming only — index geometry (dimension, distance) is immutable, so every `version=vN` becomes a *newly created index* (e.g. `mini-lm/cleaned-llm/v2`), matching "versions add, they don't overwrite". But there is no native version lineage; the index *name* is the only version carrier, so the sync job must derive index names from the same key-builder discipline as `keys.py`. |
| **Athena/Glue surface (§9)** | ❌ none — vector buckets are invisible to the catalog. Analytics over embeddings (Exp-001-style PCA/UMAP/t-SNE, [EVōC](012-evoc-embedding-clustering.md) clustering, golden-set self-retrieval metrics) requires the full vector matrix in memory, which the vector index never provides — only `ListVectors` export can bridge back, which is exactly the direction we refuse to need (keep Parquet authoritative). Athena can reach S3 Vectors only through a **custom federated-query connector** (awslabs `athena-s3vector-connector`, from the nx-neptune project) — non-native, non-GA-grade, and pointless for us since Parquet already sits next to Athena. |
| **Manifests (§3, §10)** | ✅ natural — the serving snapshot manifest gains one optional field (`"vector_index_arn": "arn:aws:s3vectors:…"`) pinning which rebuilt index the snapshot serves; the `configuration_hash` of the sync run records the upload. The app consumes the pinned index, never the prefix. |
| **Eval layer (§8)** | ✅ untouched — authored eval datasets and quality results stay Parquet; the ANN index must **never** enter the retrieval paths that proxy metrics depend on (same exactness-protects-attribution rule as [Report 014 §5.1](014-semantic-search-tools-architecture.md#5-reference-architectures)). |
| **Immutability (§1)** | ⚠️ different mutability model — our lake is immutable and content-addressed; vector buckets are mutable upsert-by-key. This is fine *because they're derivative*: idempotent sync (same keys → same vectors) makes re-runs no-ops logically, though a PUT of identical bytes is still billed; parity is enforced by comparing record counts + spot-check hashes rather than by content-addressing. |
| **Raw layer (§2.1)** | ❌ irrelevant — raw JSON payload preservation is orthogonal; S3 Vectors stores floats, not documents. |

**The sync contract** (if/when adopted): the embeddings Parquet dataset is the only upstream. One idempotent job per snapshot publication — batched `PutVectors` (≤ 500/batch) with `key = source_hash`, `DeleteVectors` for hash-diffs on corpus supersedes, count-verify → write `vector_index_arn` into the snapshot manifest. Corrupted/regressed index = rebuild from Parquet, exactly like the FAISS rebuild §7 already promises.

**A tempting anti-pattern:** AWS's Bedrock-KB pattern stores the *source text chunks* as non-filterable metadata so retrieval returns text without a second read. We should not copy this into the index wholesale — our text datasets (`cleaned_text`, golden, raw) are lakehouse rows under governance; duplicating cleaned text into a vector bucket creates a second mutable copy of governed data inside an *acceleration structure*. The discipline: **index metadata = identity + serving filters, not text.**

## 6. Relevance to the Baseline-Embedding Line of Work

The Baseline Embedding experiment ([001](../experiments/001-baseline-embedding.md)) embedded 27 postings raw; its follow-ups (feature engineering, per-section analysis, cleaning) feed experiments 002–008, and the embedding layer is where [data architecture §7](../architecture/data.md#7-embeddings) expects lakehouse Parquet datasets to land next.

- **Scale math unchanged:** 27 postings × 384-D float32 = 41 KB. Even the projected few-hundred-posting corpus is ≤ 2 MB. All of experiment 001's visualization duties (PCA, UMAP, t-SNE) and evaluation duties (self-retrieval, separation gaps) run from in-memory arrays — no vector-index service touches any current or near-term experiment stage.
- **The distance metric aligns by construction:** cosine over unit-normalized vectors is exactly what the baseline protocol prescribes, so should serving ever move to an S3 vector index, the geometry (384, cosine) carries over verbatim — the index shape *is* the dataset definition, just frozen. A `version=v2` with different preprocessing is a new index automatically.
- **What S3 Vectors does *not* help with:** anything in the experiment loop. It cannot produce embeddings (we run `sentence-transformers` locally/on Fargate), cannot cluster (EVōC/UMAP need the matrix), cannot evaluate (ANN noise corrupts attribution), and cannot be queried with SQL. Its value is exclusively in the **serving** end: similarity search for the app/API.
- **What it *would* help with eventually:** the career-map serving path as a cheap, zero-ops vector tier when the pipeline already lives on AWS (Phase-4.5 Lambda/Fargate tiers per [Cleaning Automation Report](../architecture/cleaning-automation-report.md); ingestion tier per [Report 013](013-job-board-api-scrapers.md)). A Lambda-backed API + `QueryVectors` avoids running any always-on index infrastructure; the frozen-index-per-version model even matches the "versions add" discipline that §2.4 prescribes.

## 7. Position in the Retrieval Graduation Ladder

[Report 014 §9](014-semantic-search-tools-architecture.md#9-recommendations-for-this-project) fixed the ladder: NumPy exact cosine now → embedded index (`sqlite-vec`/FAISS) → `Qdrant`/`pgvector` at server scale, each rung with numeric triggers. S3 Vectors slots in as a **serverless rung** — attractive on ops, unattractive on latency and recall ceiling:

| Property | In-process NumPy/FAISS (current/next) | **S3 Vectors** | Qdrant/pgvector |
|---|---|---|---|
| Query latency | µs (exact) | ~100 ms–1 s (ANN, 90%+ recall) | single-digit ms (ANN, ~0.99 recall) |
| Ops surface | none | none — serverless, IAM only | a service to operate |
| Index build | our job (script) | serverless, incremental upserts | our job (or built-in) |
| Cost at ≤ 1M vec, ≤ 10K queries | $0 | ≈ $0.10–0.20/mo | compute floor ($25–700/mo class) |
| Exactness | exact | ~90–99% recall ceiling | tunable near-exact |
| Analytics/Athena | n/a (local) | none (custom connector at best) | pgvector: full SQL side-by-side |
| Rig/lock-in | none | AWS API, separate namespace | OSS escape hatch |

**Graduation triggers for adopting S3 Vectors specifically** (all three should hold):

1. **Retrieval is served from Lambda/cloud and per-query-latency tolerance is ≥ ~100 ms** — i.e., the stateless-API MVP has moved to the same compute tier as the Phase-4.5 pipeline, and there is no in-process index.
2. **Corpus no longer comfortably brute-forces in memory** (≥ ~10⁶ vectors — 1M × 384-D float32 = 1.5 GB; matmul per query starts to hurt a Lambda/Fargate budget) — the same "150–500K" trigger line [Report 014](014-semantic-search-tools-architecture.md#9-recommendations-for-this-project) drew for graduating the *index*, with the S3 Vectors/Qdrant choice split by trigger condition (b) and (3).
3. **Query volume stays low** (tens/day–tens/min, e.g. interactive career-map queries) — per-query pricing beats an always-on cluster; conversely, a high-QPS path (< 100 ms p99, hybrid or reranked) in front of many concurrent map viewers re-points the ladder at Qdrant per the 014 evidence.

Before that triple trigger exists, S3 Vectors changes nothing about today's stack — and Report 014's "no hosted vector DB (Pinecone, Zilliz, Weaviate Cloud)" verdict stays true for the same reason: zero benefit at MVP scale, plain extra surface.

## 8. Risks and Open Questions

- **Two sources of truth is a discipline problem disguised as an infra one.** The failure mode is drift: experiments regenerate embeddings into Parquet, serving queries an index that nobody re-synced. The snapshot manifest (`vector_index_arn`) + count/hash verification is the answer, but it must be a *checked* invariant, not a convention.
- **ANN recall latency is a one-second cold-start tax** — infrequent queries < 1 s, warm ~100 ms. Fine for a career-map API (2 s budget), fatal if a future interactive clustering tool ever tried to iterate against the remote index.
- **IaC lag:** at time of writing CDK lacks first-class constructs for vector buckets/indexes ( boto3/provisioner route or CloudFormation custom resources needed — consistent with how the Phase-4.5 provisioner was hand-rolled).
- **No export-vs-lifecycle control:** no storage classes, no replication, no versioning on vector buckets; DR story is "rebuild from Parquet", which is acceptable precisely *because* Parquet is authoritative — but that dependency on the sync job being recoverable must be part of the run manifest invariant.
- **Pricing blind spot:** query charges scale with *index size* (data processed), not result count — fine at our scale, but a large index queried at high volume pays per-TB fees that scale linearly; verify against a concrete workload if ever relevant.
- **Open question (for a future experiment, not now):** does `ENHANCED` pre-filter behavior on selective filters (e.g. filterable `role_category = "data scientist"` on a 500-vector index) meaningfully degrade recall vs. exact pre-filtered matmul? Only measurable once golden-set precision@k exists ([evaluation.md](../technical/evaluation.md)); defer with the rerank-eval trigger.

## 9. Decision Log

| Decision | Rationale | Date |
|---|---|---|
| S3 Vectors does **not** enter the current stack | Zero benefit at ≤ 10³ vectors vs. exact in-process matmul; +100 ms floor and ANN noise would corrupt the evaluation methodology the whole pipeline leans on | 2026-10-26 |
| Registered as the **serverless rung** of the [Report 014](014-semantic-search-tools-architecture.md#9-recommendations-for-this-project) graduation ladder | Adopt only when (a) retrieval moved to Lambda/cloud, (b) corpus ≥ ~10⁶ vectors breaks in-process brute force, (c) query volume is low enough that per-query pricing beats a cluster | 2026-10-26 |
| If adopted: Parquet stays source of truth; `key = source_hash`; one index per model/variant/version; frozen geometry (cosine/384) aligns index shape with dataset definition | Preserves §2.2 identity discipline, §2.4 version discipline, §7 rebuildability, §10 snapshot pinning (`vector_index_arn`); Athena surface untouched | 2026-10-26 |
| Index metadata carries identity + serving filters **only** — no text | Avoids duplicating governed text data inside an acceleration structure (the Bedrock-KB chunk-text pattern is a convenience we don't need: text is in the lakehouse rows) | 2026-10-26 |
| No S3 Vectors in the raw/artifacts/eval branches | Vector index holds vectors; catalog/Athena over Parquet remains the lakehouse query interface — a custom federated connector would solve a problem we don't have | 2026-10-26 |

## 10. Sources

- Amazon S3 Vectors is now generally available (Dec 2, 2025) — <https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-s3-vectors-generally-available/>
- AWS News Blog: S3 Vectors GA with increased scale and performance (Dec 2, 2025) — <https://aws.amazon.com/blogs/aws/amazon-s3-vectors-now-generally-available-with-increased-scale-and-performance/>
- S3 User Guide: Working with S3 Vectors — <https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html>
- S3 User Guide: Limitations and restrictions — <https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-limitations.html>
- S3 User Guide: Querying vectors (ENHANCED vs. CLASSIC modes) — <https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-query.html>
- S3 User Guide: S3 Vectors best practices — <https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-best-practices.html>
- S3 Pricing (S3 Vectors section; PUT/storage/query examples) — <https://aws.amazon.com/s3/pricing/>
- AWS What's New: S3 Vectors reduces query charges up to 80% for large indexes (Jun 16, 2026) — <https://aws.amazon.com/about-aws/whats-new/2026/06/s3-vectors-reduces-query-charges-80-percent-large-indexes/>
- AWS Database Blog: Query billion-scale vectors with SQL — S3 Vectors + Aurora PostgreSQL (May 2026) — <https://aws.amazon.com/blogs/database/query-billion-scale-vectors-with-sql-integrating-amazon-s3-vectors-and-aurora-postgresql/>
- AWS Builder Center: Benchmarking S3 Vectors vs OpenSearch Serverless at Scale (Apr 2026; eu-west-1 unit prices) — <https://builder.aws.com/content/3BWqyF1PhA4Mc1fxUOl9n2B9yZo/benchmarking-amazon-s3-vectors-vs-opensearch-serverless-at-scale>
- awslabs/nx-neptune: Athena S3 Vector federated connector (the only Athena path today) — <https://awslabs.github.io/nx-neptune/data-lake/s3-vectors/>
- AWS Builder Center: S3 Vectors RAG walkthrough (S3-Vectors latency ~100–200 ms warm; CDK lag) — <https://builder.aws.com/content/3ATzmgNKgOPJi2fgBYZRVi69y3N/building-a-simple-rag-pipeline-with-strands-agents-and-amazon-s3-vectors>
- InfoQ: Amazon S3 Vectors Reaches GA, "Storage-First" Architecture for RAG — <https://www.infoq.com/news/2026/01/aws-s3-vectors-ga/>
