# Technology Choices

## Architecture Decisions

| Decision | Rationale |
|---|---|
| Vector search over keyword matching | Captures semantic similarity — "ML engineer" matches "applied scientist" regardless of keyword overlap |
| Single embedding model (all-MiniLM-L6-v2) | Reduces scope. Model comparison is documented as future work |
| CPU-only inference | ~20 ms per posting, well below user-perceptible threshold |
| Precomputed embeddings | Generated once at setup. Faster queries, simpler deployment |
| Stateless API | No user accounts, no persistence. Reduces operational surface area |
| L2 normalization everywhere | Cosine similarity → dot product. Prevents magnitude imbalance |
| LLM-based boilerplate removal | More robust than regex. Handles varied posting formats |

## Alternatives Considered

| Choice | Selected | Reason |
|---|---|---|
| all-MiniLM-L6-v2 | ✓ | 80 MB, ~20 ms CPU, 384d — smallest viable model |
| bge-small-en-1.5 | | 133 MB, ~28 ms — slightly better scores but 67% larger |
| mpnet-base-v2 | | 438 MB, ~40 ms — 2× inference time |
| NumPy arrays (vector storage) | ✓ (current) | Zero dependencies, sufficient for ≤10K vectors |
| FAISS | ✓ (planned) | Fast ANN search needed when dataset exceeds a few thousand vectors |
| pgvector/Postgres | | Requires Postgres service — overkill for 27–350 vectors |
| Elasticsearch | | Separate service, operational complexity |
| ONNX Runtime | | 2–3× speedup, ~75% smaller Docker image — deferred |
| GPU (CUDA) | | Not justified at 27 postings; would matter with cross-encoder reranker |
| Hybrid BM25 + dense | | Higher quality but requires separate BM25 index — deferred |

## Optimization Layers

| Layer | Techniques Applied |
|---|---|
| Pre-embedding | Section segmentation, LLM-based boilerplate removal, planned skill extraction |
| Embedding model | all-MiniLM-L6-v2 selected; model comparison deferred |
| Post-embedding | L2 normalization (critical), planned mean centering, PCA whitening deferred |
| Retrieval | Cosine similarity on NumPy arrays; FAISS planned for scale |
| Evaluation | Self-retrieval + separation gap (proxy), pytrec_eval planned |

!!! info "One Variable at a Time"
    Measure baseline → Change one variable → Re-measure → Document delta. Changing multiple variables simultaneously makes it impossible to attribute gains or regressions.

## Tradeoffs

| Dimension | Current Choice | Cost |
|---|---|---|
| Accuracy | Dense-only retrieval | Hybrid BM25+dense would improve exact keyword matches (deferred) |
| Latency | ~20 ms CPU + dot product | Acceptable for single-user demo |
| Complexity | Single model, NumPy, stateless API | Limits flexibility but keeps scope manageable |
| Scale | 27 → 350 postings | FAISS/pgvector needed at 10K+ |

## Recommended Stack

### Must-Do (High Impact, Low Cost)

1. Text preprocessing: section segmentation + weighted concatenation before embedding
2. L2 normalization after every embedding call (index + query)
3. Golden test set: 10–20 resumes with hand-labeled relevance
4. Metric tracking: Precision@5, MRR via pytrec_eval

### Should-Do (Medium Impact, Low-Medium Cost)

5. Template-based data augmentation: 300+ synthetic postings from seeds
6. Section-weighted embedding A/B comparison with documented deltas

### Deferred (High Impact, High Cost)

7. Hybrid BM25 + dense retrieval
8. PCA whitening (requires larger dataset to avoid noise amplification)
9. LLM-based data augmentation
10. Fine-tuning Sentence Transformer on domain data

## Engineering Judgment Demonstrated

- **Scope prioritized over perfection:** Single model, no GPU, no multi-cloud
- **Every optimization measured:** Before/after deltas with proxy metrics
- **Alternatives named and rejected with reasons** in the table above
- **Future improvements have activation conditions:** FAISS when dataset exceeds 10K, GPU when cross-encoder reranker is added
- **Stack is reproducible:** All Python, pip-installable, CPU-compatible — runs on a laptop
