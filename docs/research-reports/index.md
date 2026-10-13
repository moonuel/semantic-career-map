# Research Reports

Research log for exploratory reports that fuel experiments and the final design. Usually AI-generated and used to guide early prototypes and downstream agents.


<div class="grid cards" markdown>

-   [**001 — Embedding optimization**](001-embedding-optimization-research.md){ .md-button }

    **2026-07-22**

    Feeding raw data into a model is almost never optimal. Approximate techniques for data cleaning inputs are sufficient for initial validation. 

-   [**002 — Tutte Institute Tools**](002-tutte-institute-tool-review.md){ .md-button }

    **2026-07-22**

    I saw a cool talk by Dr. Valerie Poulin detailing the usage of LLMs to annotate branches in hierarchically-clustered, interactive UMAP embeddings. Would love to try it. 

-   [**003 — Taxonomy of ML/AI Responsibilities**](003-ml-ai-responsibility-taxonomy.md){ .md-button }

    **2026-07-22**

    I'm working with the hypothesis that jobs are more accurately grouped by responsibilities rather than title. An audit of ML/AI responsibilities was conducted and a taxonomy of roles constructed from that.

-   [**004 — LLM Evaluations**](004-llm-evals-report.md){ .md-button }

    **2026-08-06**

    Practical guide to LLM evals synthesized from Airbnb's Eval-Driven Development, Airbnb's From Weeks to a Day, and Hamel Husain's LLM Evals FAQ — error analysis, binary pass/fail evals, and LLM-as-judge.

-   [**005 — Object Storage Naming Conventions**](005-object-storage-naming-conventions.md){ .md-button }

    **2026-09-19**

    Bucket and object naming rules across AWS S3, Google Cloud Storage, Azure Blob Storage, and S3-compatible providers — best practices for consistency, discoverability, scalability, security, and lifecycle management.

-   [**006 — Proposed System Architecture**](006-proposed-system-architecture.md){ .md-button }

    **2026-09-25**

    Proposed end-to-end architecture — data ingestion, object storage, evals, cloud inference, and visualizations — synthesized from Chip Huyen's *Designing Machine Learning Systems* and *AI Engineering*.

-   [**007 — AnyJev & Jev: Calibrated LLM Classification**](007-anyjev-calibrated-classification.md){ .md-button }

    **2026-09-28**

    Nokia's AnyJev turns any open LLM into a typed, calibrated decision model (no training) — position-bias correction (L0), temperature scaling (L1), and closed-form hidden-state heads (L2). Relevant to taxonomy/role labeling and confidence-gated human review.

-   [**008 — Contextual Retrieval**](008-contextual-retrieval.md){ .md-button }

    **2026-09-28**

    Anthropic's Contextual Retrieval — prepending an LLM-generated, chunk-specific preamble before embedding and BM25 indexing to restore context lost by chunking. Cuts retrieval failure rate up to 67% with reranking; directly applicable to job-postings context loss.

-   [**009 — Data Lake Design & Best Practices**](009-data-lake-design-best-practices.md){ .md-button }

    **2026-10-04**

    Survey of data lake/lakehouse design — medallion (bronze/silver/gold) layering, open table formats (Iceberg/Delta/Hudi), partitioning and the small-files problem, governance and lineage, and anti-patterns — mapped onto the project's `eval/` + `live/` + `configs/` object-storage layout.

-   [**010 — Data Warehouse Design & Best Practices**](010-data-warehouse-design-best-practices.md){ .md-button }

    **2026-10-04**

    Survey of data warehouse design — the Kimball/Inmon/Data Vault paradigms, dimensional modeling (grain-first stars, SCDs, conformed dimensions, surrogate keys), the layered/semantic architecture, ETL vs ELT, and anti-patterns — mapped onto the project's planned Athena SQL layer over the object store.

-   [**011 — Object Store Metadata Design**](011-object-store-metadata-design.md){ .md-button }

    **2026-10-04**

    Survey of object-store metadata design — system vs. custom metadata, how hyperscale metadata services scale (partitioning, tiering, distributed-DB backends), key-namespace engineering (high-cardinality prefixes, Hive time layout, hot-spot avoidance), and durability/consistency/security of the metadata layer — mapped onto the project's metadata.json + UUIDv7 ids + configs provenance.

-   [**012 — EVōC: Embedding Vector Oriented Clustering**](012-evoc-embedding-clustering.md){ .md-button }

    **2026-10-04**

    Source-level review of TutteInstitute/evoc (v0.3.1) — how the UMAP+HDBSCAN replacement builds one k-NN graph, learns a 4–15D node embedding, runs parallel Borůvka + condensed tree, and uses PLSCAN persistence peaks to auto-select multi-granularity cluster layers.

-   [**013 — Job Board API Scrapers**](013-job-board-api-scrapers.md){ .md-button }

    **2026-10-07**

    Survey of ATS public job-board APIs (Greenhouse, Lever, Ashby, SmartRecruiters, Workable, Recruitee, Personio vs. Workday/Teamtailor and curators like dataroles.co) — per-source field coverage, pagination/sorting, change and deletion semantics, polling frequency, board-token discovery, dedup, and rate-limit/ToS posture. Mapped onto `backend/ingestion/scrapers.py`.

-   [**014 — Semantic Search Tools and Architecture**](014-semantic-search-tools-architecture.md){ .md-button }

    **2026-10-09**

    Survey of the semantic search stack — retriever families (bi-encoder, sparse, late-interaction, cross-encoder), ANN index algorithms, vector database engines (FAISS, Qdrant, Weaviate, Milvus, pgvector, Chroma, LanceDB), the 2026 two-stage default (hybrid RRF + reranker), and the graduation ladder from exact NumPy cosine. Mapped onto `backend/retrieval.py`.
</div>
