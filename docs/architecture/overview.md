# Architecture

This section documents the system design of the Semantic Career Map: how the data pipeline is structured, why each design decision was made, and how data is laid out as the project moves toward a production, MLflow-tracked pipeline.

<!-- <div class="grid cards" markdown>

-   [**Design Decisions**](design-decisions.md){ .md-button }

    The seven deliberate trade-offs behind the pipeline — composable stages, dense embeddings, L2 normalization, precomputation, a stateless API, LLM extraction, and immutable experiment scripts.

-   [**Data Layout & Object Storage Naming**](data-layout.md){ .md-button }

    Proposed S3 bucket and object-key convention for raw postings, cleaned variants, derived artifacts, and embeddings — including provenance tracking, lifecycle, and MLflow alignment.

</div> -->

---

## System Context

The Semantic Career Map is an information extraction and visualization system that transforms unstructured job postings into a structured semantic representation suitable for search, analysis, and career exploration. Job postings are processed through a multi-stage pipeline producing dense vector representations that capture semantic similarity between roles based on actual responsibilities rather than job titles.

```
┌──────────────┐      ┌──────────────────────┐     ┌───────────┐
│    Job       │────▶│ Semantic Career Map  │────▶│   Users   │
│  Postings    │      │                      │     │           │
└──────────────┘      │ ingestion → pipeline │     │ Search &  │
                      │ → embeddings → query │     │ Visualize │
                      └──────────────────────┘     └───────────┘
```

<!-- **Non-goals** — what this system deliberately does not do:

- Real-time inference on streaming job feeds
- Multi-language NLP beyond English
- User authentication, accounts, or persistence
- Production-grade rate limiting or auth middleware
- Mobile or web client (CLI/API only)
- Resume parsing or document scanning
- Automated job recommendations or alerting

These boundaries keep the architecture focused on the core problem: transforming free-form job descriptions into a semantically searchable space backed by measurable quality signals. -->

---

## Pipeline

The data pipeline is a chain of composable, independently testable stages — each stage changes one variable and is evaluated against the same golden set.

```
┌──────────────────────┐
│    Raw Job Posting   │  (Markdown from LinkedIn)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│    Data Ingestion    │  ✅ Built — Exp 001
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  LLM Text Extraction │  ✅ Built — Exp 003
│  (boilerplate removal│
│   + signal extraction)│
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Semantic Partitioning│  ⚠️ Research — Exp 005
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Taxonomy Labeling   │  ⚠️ Research — Exp 006
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Embedding Generation │  ✅ Built — Exp 001
│  (L2-Normalization)  │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   Vector Storage     │  ✅ Built — NumPy arrays
└──────────────────────┘
```

Each stage has a corresponding capability spec in `openspec/specs/` defining its requirements. Experiment numbers reference the validation artifacts — see [Experiment Log](../experiments/index.md).

As the pipeline is productionized, each stage's inputs and outputs move to versioned S3 object storage and are tracked as MLflow runs. The bucket and key convention for that migration is defined in [Data Layout & Object Storage Naming](data-layout.md).
