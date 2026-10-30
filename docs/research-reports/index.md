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

-   [**015 — HiringCafe MCP**](015-hiringcafe-mcp.md){ .md-button }

    **2026-10-13**

    Survey of the MCP surfaces over HiringCafe, the ATS-first AI job aggregator: its official no-auth ChatGPT-app MCP server (-first-party verified live), the Claude Connector, and community/Apify MCP wraps — tools, filter taxonomy, deliberate provenance de-scoping, robots/ToS posture, and the patterns reusable for ingestion and agent distribution.

-   [**016 — ESCO: The EU Jobs Taxonomy**](016-esco-eu-jobs-taxonomy.md){ .md-button }

    **2026-10-16**

    Survey of ESCO (European Skills, Competences, Qualifications and Occupations) — the EU's free multilingual taxonomy: ISCO-08-anchored occupations (3,039), the skills pillar (13,939 concepts with reuse levels, green/digital labels), v1.2.1 AI-driven maintenance and delta files, the 19-file CSV download + HAL web API, CC BY 4.0 licensing, O*NET/EURES crosswalks, and how NLP practitioners extract & map job skills onto it — with pin-the-snapshot recommendations for partitioning anchors and taxonomy label generation.

-   [**017 — Near-Duplicate Text Detection**](017-near-duplicate-text-detection.md){ .md-button }

    **2026-10-16**

    Tools and measures for detecting significant posting duplication in two shapes — containment/trim (one text a near-superset of another: RapidFuzz partial_ratio, suffix-array substring dedup, MinHash LSH Ensemble) and near-exact reposts differing by a date or a few words (RapidFuzz ratios, SimHash, char n-gram TF-IDF cosine) — plus volatile-field masking, scale math (pairwise vs LSH), and a zero-new-dependency layered pipeline for this project.

-   [**018 — spaCy & spaCy-LLM**](018-spacy-and-spacy-llm.md){ .md-button }

    **2026-10-22**

    Survey of spaCy 3.8.16 (already a dependency: PhraseMatcher/EntityRuler skill extraction, tokenizer-only CPU posture, Pydantic v2 + Python 3.14 story) and spacy-llm 0.7.4 (task/model architecture, CoT SpanCat, ESCO-ready EntityLinker KB) — with an honest maintenance verdict, a keep-our-Layer-LLM recommendation for the productionized cleaning stage, and an experiment sketch for span extraction.

-   [**019 — AWS S3 Vectors**](019-s3-vectors.md){ .md-button }

    **2026-10-26**

    Survey of S3 Vectors (GA 2025-12): vector buckets, indexes, immutable geometry, strong consistency, ENHANCED pre-filtering, and the three-axis pricing model — evaluated against the lakehouse embedding model (data.md §7) and the baseline-embedding lineage. Verdict: not adopted now; registered as the serverless rung of the retrieval graduation ladder with Parquet staying source of truth and `key = source_hash` sync.

-   [**020 — Job Map Visualization Tools**](020-job-map-visualization-tools.md){ .md-button }

    **2026-10-26**

    Survey of renderers for the GitHub Pages job-posting map — interactive embedding map with a per-point LLM summary panel (one JSON per embedded point). DataMapPlot's `extra_point_data`/`on_click`/DynamicTooltip API is load-bearing for the requirement (confirming #002's pick under a stricter lens); Plotly `scattergl` stands as fallback; Deepscatter (NC license), Nomic Atlas (hosted SaaS), and the archived TF Embedding Projector excluded.     Includes GitHub Pages limit math, a 3-stage packaging ladder (inline → split → per-point fetch), and the draft per-point summary JSON contract.

-   [**021 — Clustering Metrics at Scale**](021-clustering-metrics-at-scale.md){ .md-button }

    **2026-10-28**

    Survey of clustering-quality metrics for high-dimensional embeddings — internal (Silhouette/CH/DB, DBCV→DISCO, k-NN purity, EVōC persistence) vs external (ARI/AMI/V-measure) — why centroid indices and DBCV misbehave on embeddings, how to beat the O(n²) pairwise wall (sampling, k-NN graph reuse, partitionable pair-counts), and where clustering + metrics run on AWS (SageMaker K-Means, EMR/Spark MLlib, Athena external metrics, Fargate single-node). Mapped onto separation gap, the HDBSCAN/EVōC validation plans, and function-tag labels.

-   [**022 — Scaling UMAP to Cloud Hardware**](022-umap-cloud-scale-deployment.md){ .md-button }

    **2026-10-28**

    UMAP's memory grows super-linearly with corpus size (the k-NN graph, not the input, is the killer) and will outgrow this machine before the pipeline is done. Memory maps are staging, not scaling — `numpy.memmap` removes input/output copies but cannot make UMAP's hot loops out-of-core. Sets a CPU-first graduation ladder (float32 → low_memory → PCA pre-reduction → subsample-fit + transform) over memory-optimized EC2, with cuML's MLSys 2026 GPU out-of-core k-NN reserved as the 10⁶+ break-glass.

-   [**023 — Context Engineering for Embeddings**](023-context-engineering-for-embeddings.md){ .md-button }

    **2026-10-29**

    Does rewriting text before embedding help? Compares narrative stories, keyword sets, field-labelled prose, and JSON for sentence similarity. Verdict: short field-aware prose is the safest primary representation, keywords are a complementary view, narratives fit role-function clustering, and JSON syntax usually hurts. Includes a controlled experiment design for this pipeline.

-   [**024 — Embedding Models for Semantic Clustering**](024-embedding-models-for-semantic-clustering.md){ .md-button }

    **2026-10-30**

    Model-selection research for the clustering pipeline — Sentence Transformers families (MiniLM, BGE, BGE-M3, EmbeddingGemma, Qwen3, Harrier) and Amazon Bedrock (Titan V2, Cohere Embed v3/v4, Nova). Finds the current 256-token window truncates every golden-set posting (mean 539 tokens), recommends proving full-text coverage with MiniLM chunk-pooling before any model migration, and proposes a preregistered eight-arm encoder bake-off.

-   [**025 — Semantic Clustering Methods & Quality Measures**](025-semantic-clustering-methods.md){ .md-button }

    **2026-10-30**

    Algorithm and evaluation survey for the job-posting clustering layer — centroid (spherical K-Means), hierarchical, density (HDBSCAN/EVōC), probabilistic (vMF/GMM), graph (Leiden), and topic pipelines, plus the four-lens quality protocol (internal, external, stability, task/human). Recommends a spherical K-Means baseline, HDBSCAN/EVōC discovery, ARI/AMI/V-measure against function tags, k-NN purity, and bootstrap stability, with explicit noise and multi-label handling.
</div>
