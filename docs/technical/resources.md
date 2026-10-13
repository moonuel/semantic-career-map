# Favourite Resources

Just my favourite resources used in this project.

## Uncategorized/To read

https://www.anthropic.com/engineering/contextual-retrieval


## Engineering & System Design

| Resource | Author | Why it matters |
|---|---|---|
| [Object Storage System Design](https://www.systemdesignhandbook.com/guides/object-storage-system-design/) | Fahim Ul Haq | Grounds the object-store layout: buckets, keys, metadata, and durability trade-offs used in [Data Architecture](../architecture/data.md). |
| [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) | Chip Huyen | End-to-end ML system design: data engineering, feature stores, deployment, monitoring. |
| [AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) | Chip Huyen | Evaluation, RAG, agents, and the AI engineering stack. |

## LLM Evaluation

Synthesized in [Research Report 004 — LLM Evaluations](../research-reports/004-llm-evals-report.md). The three primary sources:

| Resource | Author | Why it matters |
|---|---|---|
| [Eval-Driven Development: Lessons from Evaluating GenAI at Scale](https://medium.com/airbnb-engineering/eval-driven-development-lessons-from-evaluating-genai-at-scale-e817e5ae5788) | Rohit Girme (Airbnb) | Layered evaluation stack; scalable judge calibration. |
| [From Weeks to a Day: How We Made LLM Evaluation Fast Enough to Iterate On](https://medium.com/airbnb-engineering/from-weeks-to-a-day-how-we-made-llm-evaluation-fast-enough-to-iterate-on-14e2d35198b4) | Baharak Saberidokht (Airbnb) | Deterministic evaluation via caching, micro-adapters, end-to-end validation at the seams. |
| [LLM Evals: Everything You Need to Know](https://hamel.dev/blog/posts/evals-faq/) | Hamel Husain & Shreya Shankar | Error analysis (open → axial coding), binary pass/fail, synthetic data generation, custom annotation tooling. |

## Semantic Search

Synthesized in [Research Report 014 — Semantic Search Tools and Architecture](../research-reports/014-semantic-search-tools-architecture.md). The three primary sources:

| Resource | Author | Why it matters |
|---|---|---|
| [A Comprehensive Empirical Evaluation of Vector Database Systems for ANN](https://arxiv.org/html/2608.12812v1) | (arXiv, 2026) | First reproducible 7-system head-to-head (FAISS, Qdrant, Milvus, Weaviate, Chroma, pgvector, LanceDB) — no system dominates; pick by bottleneck regime. |
| [Hybrid Search in Production: BM25 + Dense Embeddings](https://tianpan.co/blog/2026/04/12/hybrid-search-production-bm25-dense-embeddings) | Pan Tian | Fusion done right: RRF k=60 as robust default, convex-combination tuning, reranker placement, query-type diagnosis. |
| [Introducing the Ettin Reranker Family](https://huggingface.co/blog/ettin-reranker) | Tom Aarsen (HF) | SOTA cross-encoders at 17M–1B (ModernBERT, 8K ctx) — the CPU-friendly rerank ladder for a future Stage 2. |

## Key Takeaways Applied Here

- Manual error analysis before automated evaluators.
- My data engineering design instincts are good! 
- Scalable design for multi-label classification.
- Exact NumPy cosine is correct retrieval engineering at ≤500 vectors — not a placeholder.