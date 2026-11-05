# Design Decisions

Each decision records a deliberate trade-off with measurable justification.

### Pipeline stages over end-to-end model

**Context:** An end-to-end model could theoretically map raw job text → embedding in one pass, eliminating the multi-stage pipeline.

**Decision:** Deliberately separate parsing, cleaning, extraction, and embedding into independently testable stages.

**Rationale:** Each stage changes one variable. When separation gap improves by 8.2×, you know exactly why — the LLM extraction stage changed, nothing else. End-to-end models make this impossible. Composability is also a practical concern: swapping the embedding model (e.g., MiniLM → E5) should not require retuning the extraction prompt.

### Dense embedding model for semantic similarity

**Context:** Modern embedding models span dense and sparse architectures with quality benchmarks heavily favoring larger models.

**Decision:** Use a dense embedding model as the embedding backbone. The production generation is normalized Amazon Titan Text Embeddings V2 (1,024-d) through Bedrock; the original all-MiniLM-L6-v2 (384-d) remains the rollback generation.

**Rationale:** Dense embeddings encode semantic structure — postings with similar responsibilities cluster together regardless of title differences. MiniLM was the initial choice for three practical constraints: CPU-only local inference (~20 ms per posting), a small 80 MB model, and proven generalization. Experiment 011 measured full-corpus encoder coverage, and Experiment 012 selected normalized Titan V2 for clustering quality while accepting its subsample-stability risk. Titan moves inference to a managed service, removes local model packaging from the embedding image, and uses an 8,192-token bound; MiniLM remains a reversible fallback because versions add rather than overwrite. The embedding model is a configuration choice, not an architectural commitment.

### L2 normalization with dot product over raw cosine similarity

**Context:** Two mathematically equivalent ways to compute cosine similarity: normalize vectors then dot product, or compute cosine directly on unnormalized vectors.

**Decision:** L2-normalize all embeddings at write time, use dot product at query time.

**Rationale:** L2 normalization eliminates vector magnitude bias — longer documents produce higher-magnitude embeddings, causing short but relevant documents to rank below verbose but weakly relevant ones. Normalizing both write-time (indexing) and query-time removes this artifact entirely. Dot product on normalized vectors is the fastest similarity operation on every vector engine — a single matrix multiply.

### Precomputed embeddings over on-the-fly inference

**Context:** Embeddings can be generated at query time (encode the query + every document on each request) or precomputed once and loaded at startup.

**Decision:** Precompute embeddings during the data pipeline, load the matrix into memory at startup.

**Rationale:** Job postings are static — re-embedding them at query time adds latency with no benefit. At the last snapshot the embedding matrix is roughly 10k postings × 1,024 dimensions (tens of MB as float32) — trivially loaded and queried. Precomputation also enables batch inference optimization on build and cheap dot-product retrieval at query time. At larger scale, approximate nearest neighbor indices would replace brute-force dot product, but precomputation remains the correct pattern — the index is built once, not per query.

### Stateless API over session state

**Context:** A query API could maintain user sessions, search history, and preference profiles.

**Decision:** The API is stateless — every request is self-contained.

**Rationale:** No user accounts, no persistence layer, no session management. This eliminates the most complex operational concern (state consistency) while gaining the simplest possible deployment model (scale horizontally with zero coordination). Search relevance depends on the embedding quality, not on user history — the core value proposition (mapping interests to job families by skills, not title) works without personalization. Persistence can be added later if needed, but removing it later is impossible.

### LLM extraction over regex cleaning

**Context:** Experiment 002 demonstrated regex-based boilerplate removal (3.4× separation gap improvement), but the regex approach was brittle — company-specific formatting, legal disclaimers embedded in qualification paragraphs, and varied heading structures all caused failures.

**Decision:** Replace regex cleaning with LLM-based text extraction.

**Rationale:** LLM extraction achieves an 8.2× separation gap improvement over raw text (2.4× over regex) with zero hallucinations across the golden set. Brittleness is eliminated — the LLM generalizes across posting formats without pattern maintenance. The cost (API calls per posting) is paid once during the pipeline, not at query time. Caching by content hash makes the cost effectively zero after the first run.

### Immutable experiment scripts over reusable notebooks

**Context:** Jupyter notebooks are the default for ML experimentation. They support iterative exploration but produce non-reproducible artifact chains.

**Decision:** Experiments are standalone Python scripts in `scripts/` with deterministic outputs and SHA-256 content caching. Scripts are immutable after the experiment is complete — reusable logic is extracted into `backend/`, not backported into experiment scripts.

**Rationale:** Scientists don't alter published papers. Experiment scripts are the same — immutable records of what was run and what it produced. Re-running an experiment 6 months later (with a different model or on a different dataset) should produce either identical results (via cache) or a meaningfully different comparison (on new data). Notebooks make both outcomes unreliable.

---

## Design Decision Summary

| Decision | Choice | Rationale |
|---|---|---|
| Pipeline architecture | Composable stages | Single-variable isolation; swappable components |
| Embedding model type | Dense model; Titan V2 production, MiniLM rollback | Semantic similarity encoding; configuration, not commitment |
| Similarity metric | L2 norm + dot product | Eliminates magnitude bias; fastest operation |
| Embedding strategy | Precomputed | Static data; zero query-time latency |
| API design | Stateless | Simplest deployment; no persistence needed |
| Text cleaning | LLM extraction | 8.2× improvement over raw; zero hallucinations |
| Experiment scripts | Immutable artifacts | Reproducibility; scientific integrity |
