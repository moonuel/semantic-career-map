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
</div>
