# Architecture

This section documents the system design of the Semantic Career Map: how the data pipeline is structured, why each design decision was made, and how data is laid out as the project moves toward a production, MLflow-tracked pipeline.

<!-- <div class="grid cards" markdown>

-   [**Design Decisions**](design-decisions.md){ .md-button }

    The seven deliberate trade-offs behind the pipeline — composable stages, dense embeddings, L2 normalization, precomputation, a stateless API, LLM extraction, and immutable experiment scripts.

-   [**Data Layout & Object Storage Naming**](data-layout.md){ .md-button }

    Lakehouse layout for raw postings, derived artifacts, and embeddings — content-addressed raw JSON, Parquet datasets, Hive-style partitions, Athena tables over the Glue catalog, and manifests for provenance.

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

At a glance, the pipeline turns job descriptions into a map you can look at:

![The pipeline in two steps: above, job posting text becomes a colourless 3D cloud of points; below, the same points are coloured by the role they belong to and flattened to a 2D map while every point keeps its colour.](pipeline.png)

*Every job posting starts as text. The model turns each one into a point in a 3D space where points that mean the same thing sit close together — at first the cloud has no labels. Then points are coloured by the cluster they fall into. The final step flattens the cloud onto a flat 2D map — the same clusters, the same colours, just easier to look at.*

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

As the pipeline is productionized, each stage's inputs and outputs move to versioned datasets in the S3 lakehouse and are tracked as MLflow runs. The lakehouse layout for that migration -- raw JSON layer, Parquet datasets, and Athena tables over the Glue catalog -- is defined in [Data Layout & Object Storage Naming](data.md).
