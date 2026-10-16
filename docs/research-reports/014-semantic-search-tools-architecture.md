# Semantic Search Tools and Architecture — Research Report

> Date: 2026-10-09
> Context: Semantic Career Mapping Platform — survey of the semantic search tool landscape and reference retriever architectures, informing `backend/retrieval.py` (currently NumPy cosine on precomputed arrays). Covers retriever families, ANN indexes, vector database engines, hybrid fusion, reranking, and the scale triggers that justify graduating from the current approach.

---

## Table of Contents

1. [Anatomy of a Semantic Search System](#1-anatomy-of-a-semantic-search-system)
2. [Retriever Families: Bi-Encoder, Sparse, Late-Interaction, Cross-Encoder](#2-retriever-families-bi-encoder-sparse-late-interaction-cross-encoder)
3. [ANN Index Algorithms](#3-ann-index-algorithms)
4. [Tool Catalog: Where Vectors Live](#4-tool-catalog-where-vectors-live)
5. [Reference Architectures](#5-reference-architectures)
6. [Hybrid Search and Fusion](#6-hybrid-search-and-fusion)
7. [Reranking](#7-reranking)
8. [The 2026 Vector-Database Evidence](#8-the-2026-vector-database-evidence)
9. [Recommendations for This Project](#9-recommendations-for-this-project)
10. [Decision Log](#10-decision-log)
11. [Sources](#11-sources)

---

## 1. Anatomy of a Semantic Search System

A production semantic search system is a stack of separable stages, each independently swappable and measurable:

```
 Query ──► Preprocessing ──► [ Stage 1: Recall ] ──► [ Stage 2: Precision ] ──► Presentation
                                  │                       │
                                  │  sparse (BM25/SPLADE) │  reranker scoring
                                  │  dense (bi-encoder)   │  (cross-encoder)
                                   \  → fusion (RRF)     /
                                      25–50 candidates      top 5–10 results
```

The two stages answer different questions:

- **Stage 1 (recall)** — "of the whole corpus, what could plausibly be relevant?" Runs over a pre-built index, O(N) avoided via ANN. Optimizes recall — it is allowed to return noise. Must be fast enough to touch the entire corpus.
- **Stage 2 (precision)** — "of these candidates, which are *actually* relevant?" Scores each (query, candidate) pair with a model that sees both texts jointly. Optimizes ordering quality — allowed to be slow because the pool is bounded (25–50 items, never the corpus).

Retrieval is a high-recall problem; reranking is a high-precision problem. They require different models, and conflating them is the most common architectural mistake. Other orthogonal services that attach to the stack:

- **Metadata filtering** (tenant, category, freshness) — must restrict candidates *before* reranking, not after: filtering post-rank leaks restricted content into ranker scores, caches, and timing side channels.
- **Query understanding** — normalization, expansion, synonyms.
- **Observability** — log the per-stage candidate lists (what each retriever returned, what fusion produced, what the reranker did). Without that log, debugging a "search missed the obvious result" complaint is guessing.

Every stage should earn its complexity by moving a defined metric on representative queries. A stage that doesn't move a metric is a latency and failure-surface tax.

---

## 2. Retriever Families: Bi-Encoder, Sparse, Late-Interaction, Cross-Encoder

| Family | How it works | Strength | Weakness | Used as |
|---|---|---|---|---|
| **Bi-encoder / dense** | Encodes query and document *independently* into vectors; scores by cosine/dot product | Document vectors precomputable; sub-ms lookup; handles paraphrase, synonyms, acronyms | Compresses everything into one vector ("information bottleneck") — fine-grained term interactions are lost; exact identifiers blur | Stage 1 retrieval — *what this project uses* |
| **Sparse lexical** | Inverted index over term statistics (BM25, TF-IDF) or learned term weights SPLADE | Zero representation drift — a token indexed Tuesday is retrieved perfectly Wednesday (new skill names, product codes, person names); interpretable | Misses paraphrase entirely; a relevant doc sharing no surface terms is *never retrieved*, and no downstream fusion can recover it | Stage 1 recall (hybrid partner) |
| **Late interaction (ColBERT-era)** | Per-token document vectors + MaxSim: sum of per-query-token maximum similarities; BGE-M3 unifies dense + sparse + late interaction in one 550M checkpoint | Near cross-encoder quality with precomputed document representations; phrase-level alignment preserved | Storage balloons 50–200× (per-token vectors); heavier index | Niche / high-end Stage 1 |
| **Cross-encoder / reranker** | Concatenates query + document, one full transformer forward pass per pair | Sees both texts simultaneously — detects negation, term overlap, fine-grained mismatch; dominates every other architecture choice in ablations | O(candidates) forward passes per query — never runs over the corpus | Stage 2 precision |

BEIR's zero-shot results (18 datasets) established the ordering: BM25 stays a robust baseline, but reranking and late interaction perform best — at higher cost. The lesson is not "replace lexical with dense"; it's that the families fail differently and a stack should combine them *when measurement says so*.

### Query–document asymmetry (our case)

Classic web search queries are short and documents are long. This project inverts part of that: the "query" is a **resume** (hundreds of lines) matched against **job postings** (also long). Consequences:

- Long queries hit the encoder's context window — boilerplate removal (Exp 003) and section segmentation act as query-side chunk selection, and the context-preamble idea from [Report 008](008-contextual-retrieval.md) applies *at embedding time* to restore chunk context.
- A cross-encoder reranker over (resume, posting) pairs pays the cost per *pair*, and both texts are long — context length and input cost matter more than in web search. Rerankers with ≥8K context (ModernBERT-based Ettin, Qwen3-Reranker) or careful resume condensation become prerequisites, not conveniences.

---

## 3. ANN Index Algorithms

When brute force over all vectors is too slow, approximate it. Approximation is a recall/latency/memory dial, not a binary.

| Index | Idea | Recall @ correct params | Cost driver |
|---|---|---|---|
| **Flat / brute force** | Compare query to *all* vectors; exact results by construction | 1.0 | O(N·d) per query — fine ≤ ~10⁵–10⁶ vectors for small `d` |
| **HNSW** (Hierarchical Navigable Small World) | Proximity graph, greedy search from a sparse top layer down | ~0.99 (M=16, ef=200) | Graph memory (~1–4 GB per 1M×768D); build time; graph walk is pre-filter-unfriendly |
| **IVF (+PQ/SQ)** | Cluster vectors into `nlist` cells, probe only nearest `nprobe` cells at query time; product/scalar quantization shrinks vectors 4–32× | ~0.90–0.95 | Builder-clustering step; reservoir of post-quantization reorder; pre-filtering works well |
| **DiskANN / StreamingDiskANN** | Vamana graph + PQ compressed vectors resident on SSD, full-precision rerank on candidates | ~0.98 | SSD I/O; designed for corpora far larger than RAM |
| **ScaNN / ScaNN-style reorder** | Anisotropic quantization + rescoring top candidates at full precision | ~0.99 | Fastest p99 in ANN-Benchmarks (<1 ms) but library, not a database |

ANN-Benchmarks `recall@10 vs QPS` Pareto curves are the honest evaluation frame, and the universal shape is: lifting recall from 0.95 → 0.99 costs ~2–5× latency and ~1.5–3× memory. Notably at 2026-05 GIST-960-1M: HNSW (M=16, ef=200) hits recall@10 0.99 at p99 1.2 ms per query — approximate search is no longer the bottleneck in most stacks; the reranker stage and the embedding call dominate query latency.

**Filtering interacts with index choice.** Post-filtering (ANN first, then filter) risks returning fewer than `k` results; pre-filtering with HNSW degenerates graph traversal; Qdrant builds payload indexes that integrate filters *into* graph traversal, and pgvector's `WHERE` clauses run before the index scan. If category/location/level filters are central to our retrieval, pre-filter capable engines (Qdrant, pgvector) beat HNSW-only libraries.

---

## 4. Tool Catalog: Where Vectors Live

Five deployment shapes, ordered by operational weight:

### 4.1 Embedded libraries and stores (no service)

| Tool | What it is | Sweet spot |
|---|---|---|
| **NumPy / BLAS matmul** | Vectors as a matrix; similarity = one matrix product. Exact. | ≤ ~100K vectors — *what we use today* |
| **FAISS** | The reference ANN library (C++, IVF/HNSW/PQ, optional GPU). No persistence, no filtering — app-level concerns | ~10⁵–10⁸ vectors, in-process, batch build |
| **LanceDB** | Embedded, Rust, Apache Arrow / Lance file format, runs straight off S3/GCS; IVF-PQ | Disk-resident, multimodal, batch — notably the fastest index *build* (11× vs Weaviate in the 2026 head-to-head, trading recall) |
| **sqlite-vec / DuckDB VSS** | Vector search inside an embedded DB engine | Single-process apps, ≤ ~1M vectors, zero-ops |
| **Chroma** | Developer-first embedded/HTTP store: HNSW (hnswlib) + SQLite. pip-install-and-go | Prototyping, notebooks, ≤ ~500K–1M vectors. No replication/auth/distributed — teams prototyping on it rewrite for production at scale — the cost is a **pipeline rewrite**, not a data migration |

### 4.2 Pure-play vector databases (a service to run)

| Engine | Identity | Best at |
|---|---|---|
| **Qdrant** | Rust, single binary; payload indexes integrated with HNSW traversal; TurboQuant quantization (1.18, from Google Research) | Lowest p50/p99 latency of the full DBs; filtered search 3–5× faster than peers on selective filters |
| **Weaviate** | Go; module system (vectorizers/rerankers in-DB); native BM25 + dense hybrid with built-in RRF; HNSW tuned for quality | Best out-of-the-box recall (>99%); object-centric hybrid search; 1.38 adds GA disk-based index (HFresh) + rotational quantization |
| **Milvus / Zilliz** | Distributed-first, storage-compute separated, K8s-native; broadest index menu (HNSW/IVF/DiskANN/GPU-CAGRA) | 1B+ vectors; highest-dim data; GPU indexing; heaviest ops footprint (etcd + object storage + node types) |
| **Pinecone** | Managed-only, serverless, namespaces | Zero-ops managed service; least tunable; predictable latency |

### 4.3 Relational extensions

**pgvector** (+ pgvectorscale): HNSW and IVFFlat inside PostgreSQL, halfvec (fp16), binary quantization; pgvectorscale adds StreamingDiskANN + statistical binary quantization for disk-resident corpora. The argument is *architectural*, not performance: embeddings stay beside the relational data they describe — one transaction boundary, one sync surface, JOINs between vector results and metadata for free, ACID which no pure-play vector DB offers. When Postgres *is* the system of record, pgvector removes an entire embed→store sync problem.
**ParadeDB** (`pg_search`) continues this with BM25 *inside* Postgres — hybrid search ends in one query.

### 4.4 Search engines with vector slots

**Elasticsearch** / **OpenSearch k-NN** / **Vespa**: BM25 + kNN + full ranking/freshness pipelines in one battle-tested cluster, ideal if vector retrieval is only *one* signal in a broader search-and-serving platform. Realistic only for teams already operating a search cluster (the platform is the point, not the vectors).

### 4.5 Hosted embedding platforms and curators

Marqo (self-hosted with built-in embedding), Turbopuffer (object-store-backed, cold-workload pricing), MongoDB Atlas / Redis / Couchbase vector slots ("finish it inside the DB you already have"). Useful as pattern references, not priorities for this project.

---

## 5. Reference Architectures

### 5.1 MVP monolith (this project, now)

```
FastAPI ──► load embeddings.npy + metadata.json
   query ──► SentenceTransformer (CPU) ──► L2-normalize
         ──► matmul over all job vectors (exact kNN)
         ──► top-k + scores
```

One process, one container, zero external state. Query embedding dominates latency (≈20 ms); search itself is microseconds. Exact search also *protects evaluation*: proxy metrics (self-retrieval, precision@k) to attribute preprocessing gains must not ship ANN approximation noise inside the retrieval path — one variable at a time.

### 5.2 The 2026 default production stack

1. BM25 over an inverted index → top 50–100
2. Dense retrieval over an HNSW index → top 50–100 (equal depth — asymmetric depth silently biases RRF toward whichever list is deeper)
3. RRF fusion (k=60) → 25–50 candidates
4. Cross-encoder rerank → top 5–10
5. (If RAG) pass the 5–10 as context — and optionally compress each chunk with a small LLM first

Reported gains for full two-stage vs naive single-stage: +15–30% answer accuracy, 40–60% input token reduction. It costs: two search services, fusion-weight ownership, rerank latency, and scheduled recalibration.

### 5.3 Serving platform (search engines / Vespa)

Vector retrieval as one signal inside a full search stack: everything in 5.2, but fusion, freshness, filters, dedup, and context assembly are cluster features rather than application code. Justified only when search is a core platform team's responsibility.

---

## 6. Hybrid Search and Fusion

Dense and sparse fail *differently*, which is why hybrid exists:

| Failure | Fix |
|---|---|
| Dense-only: "close but not the right doc" — exact identifiers (a company, a framework name, a regulation code) blur into near-neighbors | BM25 catches exact tokens |
| Sparse-only: relevant doc phrased differently ("applied scientist" for "ML engineer", "PyTorch" vs "Torch") — a recall ceiling no downstream ranker can fix | Dense catches paraphrase |
| Both: score-scale mismatch — BM25 emits unbounded positives (a typical hit: 12.4) while cosine is bounded [-1, 1] | Fuse on **ranks**, not scores |

**RRF** (Reciprocal Rank Fusion, Cormack et al., SIGIR 2009): `score(d) = Σ 1/(k + rank_i(d))`, `k=60` — the default in Elasticsearch, OpenSearch, Qdrant, Weaviate, and Azure AI Search for 17 years. Robust by construction to score drift (every bulk reload shifts BM25's score distribution but not its rank distribution).

**Convex combination** (`α·dense + (1−α)·sparse` after linear normalization) can beat RRF with ~40 labeled query-relevance pairs for tuning one α — but calibration erodes as the corpus changes: a legal-discovery platform recorded a **30% recall drop** after a 200K-file import silently invalidated pre-tuned fusion weights. RRF remains the safer default until a held-out evaluation job runs nightly.

**The anti-pattern evidence:** an 8-condition ablation (1,854 queries, 8,766 judgments) found hybrid RRF *worse* than vector-only on a corpus of dense, semantically uniform financial filings — vector-only 0.396 NDCG@10 vs hybrid 0.358. Every list RRF fuses contributes non-zero score, so a weak second retriever dilutes a strong one. Hybrid is a measured decision, not a default: adopt when exact-match recall is a measured recurring miss (expected: rare in role-similarity search — but real for queries naming skills/tools verbatim).

---

## 7. Reranking

The single highest-leverage quality stage. Typical gains: +5–15 NDCG@10 points over the best first-stage result; in the Superlinked ablation, adding a cross-encoder over the hybrid pool moved NDCG@10 from 0.358 → **0.600** — larger than any retrieval tweak.

Latency arithmetic (the reason Stage 1 exists at all): a 110M-param cross-encoder ≈ 45–65 ms on a T4 for 20 candidates, ~100–300 ms on CPU; reranking a *full* retrieved list measured 25–86 s per query (485–2051× worse) in one BEIR study. Rules: bound the pool at 25–50 candidates; if 200+ candidates are needed to hit the precision bar, Stage 1 is broken, and reranking amplifies the cost rather than fixing the cause.

### Model landscape (2026)

| Reranker | Params | MTEB NDCG@10 | Notes |
|---|---|---|---|
| Qwen3-Reranker-4B | 4.0B | 0.6367 | Decoder + yes/no LogitScore; strongest public; GPU-class at this size |
| mxbai-rerank-large-v2 | 1.5B | 0.6115 | |
| Ettin-reranker-1b / -400m (ModernBERT) | 1.0B / 0.4B | 0.6114 / 0.6091 | 8K context; SOTA at every size ≤1B; ships full recipe |
| Ettin-reranker-68m | 69M | 0.5915 | ≈ Qwen3-Reranker-0.6B quality at 1/9 the parameters |
| Ettin-reranker-17m | 18M | 0.5576 | Beats the 33M ms-marco-MiniLM-L12 historically-shaped baseline |
| ms-marco-MiniLM-L6-v2 | 23M | 0.5082 | The "old classic" reference |
| BAAI/bge-reranker-v2-m3 | 568M | 0.5526 | Multilingual |

Sentence Transformers v5.4 made all of these loadable as `CrossEncoder("...")` directly (including decoder-based Qwen3 rerankers via a `LogitScore` head) — so a future rerank stage adds one import and one call, no framework change.

**LLM listwise reranking** (send candidates to GPT-class model, ask for order): 0.78 vs 0.74 NDCG@10 over cross-encoders in ZeroEntropy's benchmark — a ~5% gain at 9× cost and 35× latency. Consume only for a final 5→1 cut on extremely high-value queries.

Late-interaction (ColBERT/BGE-M3) sits between the two stages: reusable doc representations, near-reranker quality, at 50–200× storage and a heavier index.

---

## 8. The 2026 Vector-Database Evidence

The first comprehensive, reproducible seven-system head-to-head (FAISS, Qdrant, Milvus, Weaviate, Chroma, pgvector, LanceDB — arXiv 2608.12812, 2026-08) on six datasets (SIFT/GIST/DEEP/MS MARCO/GloVe, 96–960D) found *no system dominates every dimension*:

| Priority | Pick | Evidence |
|---|---|---|
| Throughput / library role | FAISS | 866 QPS (4–16× other systems), sub-2 ms at all percentiles, in-process |
| Max recall out-of-box | Weaviate | >0.99 across four datasets; slowest responders of the group, i.e. deliberately quality-tuned HNSW |
| Latency balance (full DB) | Qdrant | 4.55 ms P50, 216 QPS, most consistent tail (P99/P50 1.85) |
| High-dimensionality | Milvus | Only system that held recall at 960D (GIST) |
| SQL/ACID integration | pgvector | Tightest tail ratio (1.63); 154 QPS |
| Fastest *build* | LanceDB | 42.8 s build (11× vs Weaviate) but recall collapses at high dim |

At 10M-vector production-scale benchmarks (1536D HNSW, single node): Qdrant ~4 ms p99 / ~3200 QPS / 0.992 Recall@10; Weaviate ~9 ms / ~1800 / 0.988; Milvus ~5 ms / ~2900 / 0.991; Pinecone ~7 ms / ~2400 / 0.989; pgvector ~18 ms / ~680 / 0.985. Clustered multi-node relieves most of pgvector's standing except memory.

Methodological honesty from the same sources: benchmark numbers vary by hardware, version, index params, and data — *the shape*, not the exact values, is the conclusion. Approaches themselves have moved on since the 2023 wave: every major engine shipped a 2026 major release, so re-benchmark on current versions before choosing. Evaluate with ANN-Benchmarks (algorithm-level recall-QPS Pareto) plus VectorDBBench (end-to-end with client overhead, filtering, concurrency), and with your own representative data.

Mature-decision heuristic that grew from all of this (*pick the architecture before the product*):

1. Is PostgreSQL already central? → **pgvector** first.
2. Retrieval belongs inside the app? → **LanceDB/Chroma**.
3. A dedicated vector service, metadata filters first-class? → **Qdrant**.
4. Object-centric hybrid + integrated rerank? → **Weaviate**.
5. Distributed scale / billions? → **Milvus**.
6. Vector retrieval one part of a search platform? → **Vespa/Elasticsearch**.

---

## 9. Recommendations for This Project

### Now (MVP): keep exact NumPy — it is not a placeholder, it is correct

- **Scale math:** 350–500 postings × 384D float32 = ≤0.8 MB; one query is a single matmul returning in microseconds. Even the projected 200–500 real postings are three orders of magnitude below the point where ANN pays.
- **Exactness is an evaluation asset.** The one-variable-at-a-time methodology in [tech-choices.md](../technical/tech-choices.md) requires retrieval to be deterministic and exact — ANN approximation inside the measurement path corrupts attribution of preprocessing gains.
- **Zero new dependencies.** NumPy retrieval has no server, no sync, no failure mode, no ops. CPU-only constraint respected; API budget (<2 s) satisfied with three spare orders of magnitude.
- **One requirement:** keep the search behind the `retrieval.py` boundary (function of `query_vec, doc_matrix, top_k`) so the index is a swappable stanza, not a rewrite. Unit-test that boundary against a brute-force reference.

### When scale warrants it: a graduation ladder, anchored by interface not product

| Trigger | Change |
|---|---|
| > ~500K vectors, still one process | **sqlite-vec / Chroma** for persistence + filtered search, or **FAISS IndexFlatIP → HNSW** only if query latency under load actually becomes the bottleneck |
| Cluster-scale ingestion pipeline or incremental updates with concurrency | **Qdrant** (operator budget matches ops footprint; pre-filter payload indexes fit role-category/post-validation filtering) |
| PostgreSQL adopted as the relational system of record (e.g., for live/ datasets, application metadata, or jobs/labels tables) | **pgvector** — one store for both; JOINs give ranked results + metadata in one query |
| Mixed corpus with structured signals needed as *filters* at query time (role taxonomy, seniority, location, remote-ness) | prefer pre-filter-capable engines; never filter after rerank |
| Hybrid search activation | only if precision@k analysis on the golden set shows exact-skill misses; simplest first cut is `sklearn TfidfVectorizer`-based sparse list fused with RRF inside the same process — no second service until proven necessary |

### Rerank stage: deferred, activation condition

Cheap to evaluate once the golden set and precision@k exist (both are already recommended by [evaluation.md](../technical/evaluation.md)): rerank top-25 with `cross-encoder/ms-marco-MiniLM-L6-v2` CPU (~200 ms per query at resume length; budget it), measure ΔNDCG@10 and Δprecision@5 via pytrec_eval. Promote in the stack only if the delta is material. If needed at higher quality later, the CPU-friendly ladder is Ettin-reranker-17m/32m → mxbai-rerank-base-v2; the ≥1B/Qwen3 tier is entirely GPU-class and remains future work (same activation as the existing "GPU when cross-encoder reranker added" note).

### What *not* to adopt now

- **Any hosted vector DB (Pinecone, Zilliz, Weaviate Cloud)** — operational surface + cost with zero gain at our scale; violates the stateless-API design decision.
- **Elasticsearch / OpenSearch / Vespa** — we are a single-container CPU service; a JVM cluster to serve 384D matmul results is the architecture moving backwards.
- **Milvus** — built for billions of vectors; nothing in the roadmap precedes the point where its ops burden pays.
- **ColBERT / SPLADE-style learned sparse / multi-vector storage in-process** — storage bloat and complexity both land now, quality benefit arrives only after hybrid and rerank have been measured.
- **Reranking behind the query path in the MVP** — latency budget and "no GPU" constraint both argue against unmeasured stages in the serving path.

### One-sentence summary of the decision space

Dense bi-encode + L2-normalized cosine over precomputed vectors is the *right* first-stage for scale ≤10⁴–10⁵; graduate the *index* (FAISS/sqlite-vec) before ever adding a second constant-cost process; defer hybrid and rerank to measured misses on the golden set; and if/when traditional relational storage arrives for jobs metadata, pgvector is the convergent end state that removes a whole sync class rather than adding one.

---

## 10. Decision Log

| Decision | Rationale | Date |
|---|---|---|
| Keep NumPy exact cosine as the MVP retriever | 3 orders of magnitude headroom; exactness protects metric attribution; zero ops | 2026-10-09 |
| Treat `retrieval.py` as the swappable-index interface | Index choice becomes a config stanza, not a rewrite; boundary is unit-testable against brute force | 2026-10-09 |
| Crystalize graduation tiers (embedded → Qdrant/pgvector) with numeric triggers | Avoids both premature migration and indefinite NumPy; peers' benchmarks justify *shape*, not exact numbers | 2026-10-09 |
| Hybrid BM25+dense stays deferred | Activation condition: measured exact-term precision@k miss on golden set; RRF in-process first | 2026-10-09 |
| Rerank stage stays deferred | Activation condition: golden set + precision@k present so ΔNDCG is measurable; ms-marco-MiniLM first (CPU), Ettin class only if needed | 2026-10-09 |
| No hosted vector DB, no search-engine clusters | Stateless-API + container-size constraints; zero benefit at ≤500 vectors | 2026-10-09 |

---

## 11. Sources

- Comprehensive Empirical Evaluation of Vector Database Systems for ANN (2026) — <https://arxiv.org/html/2608.12812v1>
- Vector Database Benchmarks: Production Selection Guide 2026 — <https://lushbinary.com/blog/vector-database-benchmarks-production-selection-guide-2026/>
- Vector Database Comparison 2026 (practitioner benchmarks, 1M×768D) — <https://www.kargin-utkin.com/vector-database-comparison-2026-benchmarks>
- Open-Source Vector Databases: Honest Comparison (2026) — <https://www.itechguides.com/an-honest-comparison-of-open-source-vector-databases/>
- Hybrid Search in Production: BM25 + Dense Embeddings (2026) — <https://tianpan.co/blog/2026/04/12/hybrid-search-production-bm25-dense-embeddings>
- Hybrid Retrieval Architecture for Production RAG Systems (2026) — <https://markaicode.com/architecture/hybrid-retrieval-architecture-best-practices-2026/>
- Retrieval Systems Beyond Basic RAG: Hybrid Search, Reranking, Evaluation (2026) — <https://blog.ecitis.org/retrieval-systems/>
- Reranker scale evidence: Qwen3-Reranker / mxbai-rerank v2 / Ettin families (2025–2026) — <https://huggingface.co/blog/ettin-reranker>, <https://huggingface.co/blog/multimodal-sentence-transformers>
- Sentence Transformers: CrossEncoder docs — <https://sbert.net/docs/package_reference/cross_encoder/model.html>
- ANN-Benchmarks, VectorDBBench — <https://ann-benchmarks.com/>, <https://github.com/zilliztech/VectorDBBench>
