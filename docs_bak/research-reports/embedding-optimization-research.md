# Embedding Optimization & Feature Engineering for IR Systems — Research Report

> Date: 2026-07-22
> Context: Semantic Career Mapping Platform MVP — research into industry-standard techniques for optimizing embedding-based information retrieval pipelines.

---

## Table of Contents

1. [Categories of Optimization](#categories-of-optimization)
2. [Text Preprocessing Before Embedding](#text-preprocessing-before-embedding)
3. [Embedding Post-Processing](#embedding-post-processing)
4. [Feature Engineering for Embeddings](#feature-engineering-for-embeddings)
5. [Chunking & Pooling Strategies](#chunking--pooling-strategies)
6. [Data Augmentation](#data-augmentation)
7. [Hybrid Retrieval (BM25 + Dense)](#hybrid-retrieval-bm25--dense)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Comparison of Techniques](#comparison-of-techniques)
10. [Recommended Stack for MVP](#recommended-stack-for-mvp)
11. [Sources](#sources)

---

## 1. Categories of Optimization

Embedding optimization for IR systems falls into five layers (Shadecoder, 2025):

| Layer | What It Affects | Example Techniques |
|---|---|---|
| **Pre-embedding** | Text quality fed to the model | Section segmentation, skill extraction, stopword removal, normalization |
| **Embedding model** | The vector representation itself | Model selection, fine-tuning, pooling strategy |
| **Post-embedding** | Vector geometry | L2 normalization, mean centering, PCA whitening |
| **Retrieval** | Search quality and speed | ANN index config, hybrid BM25+dense, reranking |
| **Evaluation** | Measurement integrity | Recall@k, NDCG, precision@k, golden test sets |

The optimization loop (llmbestpractices.com, 2026):
1. Measure baseline → 2. Change **one** variable → 3. Re-measure → 4. Document delta

> Common mistake: "Changing too many variables at once makes it impossible to attribute gains or regressions" (Shadecoder, 2025).

---

## 2. Text Preprocessing Before Embedding

### 2.1 Is Preprocessing Still Worth It With Transformers?

Yes. A 2024 survey published in *Information Processing & Management* (Sciencedirect, 2024) found that:

- An educated choice on preprocessing **can improve accuracy by up to 25%** on classification tasks, even with Transformers like XLNet
- "Transformers and traditional models exhibit a higher impact of preprocessing on TC performance"
- With proper preprocessing, a Naïve Bayes classifier **outperformed the best Transformer by 2%**

Key takeaway: Preprocessing matters for Transformers, not just traditional models.

### 2.2 Core Preprocessing Steps (Standard Pipeline)

From Nishant Gupta (2025) and Michael Brenndoerfer (2025):

1. **Tokenization**: Split text into tokens (words, subwords, or characters)
2. **Normalization**: Lowercase, Unicode normalization (NFKC), whitespace collapsing
3. **Noise removal**: Strip HTML, URLs, special characters, repeated punctuation
4. **Stop word removal**: Domain-dependent — removing generic stop words is safe, but be cautious with domain-specific stop words (e.g., don't remove "python" or "ML" in an ML context)
5. **Stemming / Lemmatization**: Generally **skip for Transformer embeddings** — models are trained on raw or lightly normalized text and lemmatization can destroy semantic nuance

### 2.3 Domain-Specific Preprocessing for Job Postings

From the 2026 Sciencedirect paper on AI-driven job matching and the Agentic AI Job Copilot project:

- **Skill keyword extraction**: Use a curated skill vocabulary (NLP skills like "transformers", "attention", "RNN") and regex/PhraseMatcher to tag skill spans
- **Section segmentation**: Split each posting into `title`, `skills`, `description`, and `seniority` fields
- **Field-specific normalization**: Job titles benefit from canonicalization (e.g., "Sr. ML Eng" → "Senior Machine Learning Engineer"), while descriptions should preserve their original phrasing
- **Weighted term importance**: "Highly discriminative technical skills such as 'Python' and 'ETL' are boosted by assigning high weights. Generic words such as 'project' or 'management' are assigned much lower weights" (Sciencedirect, 2026)

---

## 3. Embedding Post-Processing

### 3.1 L2 Normalization (Critical, Non-Negotiable)

The single most impactful post-processing step. From llmbestpractices.com (2026):

> "Normalization is the most frequently skipped step in embedding pipelines, and the most reliably harmful to skip."

**Why it matters:**
- After L2 normalization (mapping every vector to the unit sphere), **cosine similarity reduces to dot product**, and dot product is the fastest similarity metric on every major vector engine
- Unnormalized vectors distort similarity by vector magnitude: "longer documents tend to produce higher-magnitude embeddings. A short, highly relevant document can rank below a verbose but weakly relevant one solely because of magnitude imbalance"
- **Library defaults vary**: `sklearn.preprocessing.normalize` normalizes by default, `sentence_transformers` normalizes only with `normalize_embeddings=True`, OpenAI and Voyage clients return raw (unnormalized) vectors
- **Rule**: Always check the output norm of your first batch — `np.linalg.norm(vecs[0])` should be ~1.0

**Implementation:**
```python
def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return np.where(norms > 0, vecs / norms, vecs)
```
Normalize **both** at write time (indexing) **and** at query time.

### 3.2 Mean Centering

From techbloat.com (2026):

- Subtract the training-set mean from all vectors
- **Useful when** the corpus has strong domain bias or repetitive wording (both true for job postings)
- **Critical**: compute the mean only on the training split, save it, and apply the same vector to validation/test/production data — recomputing on each split leaks distribution information
- Centering can reduce the dominance of "common boilerplate" directions in the embedding space

### 3.3 PCA Whitening

From EmergentMind (2025) and techbloat.com (2026):

**What it does:**
1. Fit PCA on training embeddings
2. Subtract training mean
3. Rotate into principal-component space
4. Divide by the square root of each component's variance

**When to use it:**
- When a small number of high-variance directions dominate retrieval or classification quality
- When embeddings exhibit high anisotropy (most modern embedding models do)

**When to avoid it:**
- Small training sets relative to embedding dimension — full whitening amplifies noise in low-variance components
- Discriminative tasks where high-variance directions carry useful semantic signal (techbloat.com, 2026: "while it benefits interpretability and noise reduction, it may degrade performance in discriminative tasks")

**Safer variants:**
- Keep only top-k principal components (80–95% variance explained)
- Use shrinkage: add a small constant to eigenvalues before division
- Then L2-normalize the result

**Production recipe** (techbloat.com, 2026):
> "Fit PCA on training embeddings, retain enough components to explain 90–99% of variance, whiten those components, then L2-normalize the result."

---

## 4. Feature Engineering for Embeddings

### 4.1 Section / Field Weighting

This is one of the most powerful and underused techniques for structured documents like job postings. Multiple industry sources confirm:

**The intuition:**
- A job posting is not a monolithic text blob — it's a composite of `title` (3–10 words), `skills` (10–30 words), and `description` (100–500 words)
- Embedding the entire text as one string gives the verbose description section disproportionate influence over the final vector
- Up-weighting the title and skills sections improves retrieval precision because these sections carry the most discriminative signal for matching

**Techniques:**
1. **Weighted concatenation**: Prepend/append title and skills text 2×–3× before embedding
2. **Separate embeddings + weighted average**: Embed each section independently, then compute `0.4 * emb(title) + 0.3 * emb(skills) + 0.3 * emb(description)`
3. **Attention-based fusion**: Use a learned weighting (overkill for MVP)

The OpenReview paper on query-paper retrieval (2024) confirms: "concatenating the question's 'question' and 'body' sections to form a comprehensive query, and using the paper's 'title' and 'abstract' to represent the document, we create rich text inputs that encapsulate the essence."

The Agentic AI Job Copilot project uses exactly this approach: "Weighted skill matching + role-aware prioritization + semantic embeddings similarity" with a final score of `0.7 * rule_based_score + 0.3 * semantic_similarity_score`.

### 4.2 Concatenation Strategies

**Simple concatenation**: `"[TITLE] Senior ML Engineer [SKILLS] Python, PyTorch, Transformers [DESC] We are seeking..."`

- Pros: Simple, no code changes to embedding pipeline
- Cons: Each section contributes equally by word count, so the description dominates

**Weighted concatenation**: Repeat title 3×, skills 2×, description 1×.

- Pros: Better signal from discriminative sections, simple to implement
- Cons: Increases token count (risk of exceeding model's max length)

**Multi-field embedding + fusion**: Embed each field separately, then combine.

- Pros: Most control over weighting, preserves field-level semantic integrity
- Cons: 3× inference cost (one embedding per field vs one for the whole doc), requires post-hoc vector arithmetic

### 4.3 Multi-Granularity Features

From techbloat.com (2026):

- Generate embeddings at **chunk level** (sections/paragraphs), **sentence level**, and **document level**
- For retrieval: compare query against chunk-level embeddings (fine-grained)
- For ranking: use document-level embedding as a secondary signal
- Combine: `score = 0.7 * max(chunk_similarity) + 0.3 * doc_similarity`

---

## 5. Chunking & Pooling Strategies

### 5.1 Chunking for Long Documents

From Pinecone (2025) and the 2026 chunking survey (alphaXiv, 2026):

**Fixed-size chunking** (naive):
- Split by character or token count
- Pro: simple, fast
- Con: breaks sentences mid-thought, poor retrieval quality

**Semantic / sentence-based chunking** (recommended):
- Split at sentence or paragraph boundaries
- 10–20% overlap between chunks to preserve boundary context
- Pro: 30–50% higher retrieval precision vs fixed-size (Pinecone, 2025)

**Sliding window**:
- Fixed-size windows with stride < window size
- Better retrieval coverage but more chunks to process
- Best when documents have no structural boundaries (legal text, transcripts)

For this project (job postings are ~200–500 tokens each):
- A single posting fits within most embedding models' context windows (512 tokens for MiniLM, 8192 for BGE)
- **Chunking is unnecessary** — each posting is its own retrieval unit

### 5.2 Pooling Strategies (Per-Document Embedding)

From the DICE paper (2026) and Zilliz (2025):

| Strategy | How It Works | Best For |
|---|---|---|
| **Mean pooling** | Average all token embeddings | General-purpose, balanced representation |
| **Max pooling** | Max per dimension across tokens | Emphasizing salient keywords |
| **CLS pooling** | Use the [CLS] token embedding | BERT-style models trained with CLS objective |
| **Top-k pooling** | Average top-k highest-norm token embeddings | Reducing noise from padding/irrelevant tokens |

> "Mean pooling preserves evidence from every chunk equally. Max pooling tests a more selective alternative by taking the elementwise maximum across chunks" (DICE paper, 2026).

For Sentence Transformers (`all-MiniLM-L6-v2`): the default is **mean pooling with attention mask** — this is appropriate and should be used as-is.

---

## 6. Data Augmentation

### 6.1 Why Augment?

From the comprehensive NLP DA survey (Sciencedirect, 2023) and the Bouthillier thesis (2026):

- Sparse embedding space from 200–500 real postings leads to poor retrieval for edge-case resumes
- Augmentation **densifies** the space by filling semantic gaps between natural clusters
- LLM-based augmentation is the state of the art, but template-based approaches are simpler and still effective

### 6.2 Template-Based Augmentation (Recommended for MVP)

This is the most practical approach for a small dataset:

```
Seed posting:
  "Senior ML Engineer at Company X — Python, PyTorch, AWS. Build and deploy ML models."

Generated variants (swap tokens in predefined slots):
  Variant 1: "Senior Data Scientist at Company X — Python, scikit-learn, GCP. Build and analyze ML models."
  Variant 2: "Junior ML Engineer at Company X — Python, TensorFlow, Azure. Train and evaluate ML models."
  Variant 3: "Staff MLOps Engineer at Company X — Python, Docker, Kubernetes. Deploy and monitor ML pipelines."
```

**Slot categories to vary:**
- Seniority: [Junior, Mid-level, Senior, Staff, Lead, Principal]
- Technology stack: [PyTorch, TensorFlow, JAX, scikit-learn, XGBoost]
- Cloud: [AWS, GCP, Azure, on-premise]
- Role: [ML Engineer, Data Scientist, MLOps, Research Scientist, NLP Engineer]
- Verbs: [build, deploy, train, evaluate, optimize, maintain]

With 5 seniority levels × 5 tech stacks × 3 cloud × 5 roles × 5 verbs = 1,875 possible combinations per seed. Starting with 50 seeds, you can generate 300+ diverse synthetic postings.

### 6.3 LLM-Based Augmentation (Future Enhancement)

From the InPars and Promptagator work, and the Bouthillier thesis (2026):

- GPT-4 can generate high-quality (query, document, negative) triplets for retrieval tasks
- "Few-shot LLM prompting recovers approximately 89% of the base model's nDCG@10 score using entirely synthetic data" (Bouthillier, 2026)
- Two-step prompting: (1) brainstorm tasks, (2) generate concrete examples
- Requires access to an LLM API and adds cost — deferred to future work

### 6.4 Best Practices for Augmentation

From the NLP DA survey (Sciencedirect, 2023):

1. **Tag synthetic data** in metadata — never blend real and synthetic without labels
2. **Evaluate augmentation impact** on retrieval metrics (Recall@k, NDCG) — augmentation can sometimes introduce noise
3. **Preserve semantic fidelity** — synthetic postings must reflect realistic skill-seniority combinations (don't generate "Junior Engineer requiring 10 years of Rust")

---

## 7. Hybrid Retrieval (BM25 + Dense)

### 7.1 Why Hybrid?

From multiple 2025–2026 sources:

- BM25 excels at **exact keyword matches** (technology names, acronyms like "AWS", "CI/CD")
- Dense retrieval excels at **semantic matches** (paraphrases, related concepts)
- They fail in **complementary** ways, so combining them outperforms either alone

Michael Brenndoerfer (2026): "A keyword system finds 'myocardial infarction' when you search for 'heart attack' only if someone mapped those synonyms in advance. A semantic system finds semantically related documents but can miss the one where a specific product code like 'AB-1042-X' appears verbatim."

For job matching: BM25 catches explicit skill mentions ("Kubernetes", "Rust"), dense catches semantic equivalents ("container orchestration" ≈ "Kubernetes", "systems programming" ≈ "Rust").

### 7.2 Fusion Strategies

**Weighted Sum** (recommended for MVP):
```
final_score = α × normalized_bm25_score + (1 - α) × normalized_dense_score
```
- α controls the BM25 vs dense weighting
- Requires score normalization (min-max scaling or z-score) since BM25 and cosine scores are on different scales

**Reciprocal Rank Fusion (RRF)** (simpler to implement):
```
RRF_score(doc) = Σ 1 / (k + rank_i(doc))
```
- k is typically 60
- No score normalization needed — operates on ranks, not scores
- Slightly less granular than weighted sum

### 7.3 Scope Decision

Hybrid search is **deferred to future work** for the MVP. Adding BM25 requires:
- A separate search index (or in-memory BM25 implementation)
- Score normalization logic
- Fusion ranking code
- Additional latency

The hiring signal from dense-only retrieval is sufficient. Hybrid is listed as a high-impact future improvement.

---

## 8. Evaluation Metrics

### 8.1 The Standard Metrics (All @K)

From Weaviate (2024), the 2026 retrieval evaluation guide (yaw.sh, 2026), and PromptZ2H:

| Metric | Formula | Use When | Best Primary For |
|---|---|---|---|
| **Recall@K** | relevant_in_top_k / total_relevant | Context retrieval (RAG, search) | Retrieval completeness |
| **Precision@K** | relevant_in_top_k / K | Fixed-size result UIs | Retrieval accuracy |
| **MRR** | 1 / rank_of_first_relevant | Single-answer QA | First correct result |
| **NDCG@K** | DCG / IDCG (position-weighted) | Graded relevance | Ranking quality |
| **MAP@K** | mean of precision@k across queries | Position-aware overall | Combined quantity + position |

### 8.2 What Matters for This Project

**Primary metric: Precision@K (K=5, 10)**

Reasoning: The ranked list UI shows top-10 results. Users care about "how many of the top results are relevant jobs?", not "did we find every possible relevant job in the entire corpus?" Precision directly measures what the user sees.

**Secondary metric: MRR**

Reasoning: For a user uploading a resume, the top result matters disproportionately. MRR penalizes systems that bury the best match at position 7–8.

**Supporting metric: Recall@K (K=10)**

Reasoning: For completeness — are we missing good matches? Low recall + high precision means the system is picky but some relevant postings are slipping through.

### 8.3 The Golden Set

From the 2026 retrieval evaluation guide (yaw.sh):

> "A golden set is a curated list of (query, relevant docs) pairs you score against. It is the single most load-bearing artifact in a retrieval system."

For this project, the golden set should be:
- **10–20 sample resumes** (real or synthetic)
- **Per-resume relevance labels**: for each resume, mark the top 10 most relevant job postings (manual labeling)
- **Stratify**: include resumes targeting different roles (ML Engineer, Data Scientist, NLP Engineer, MLOps)

Without a golden set, you cannot evaluate whether your optimizations are helping or hurting. Build this **before** starting optimization work.

### 8.4 Tooling

- **`pytrec_eval`**: Python interface to TREC's `trec_eval`, the standard IR evaluation toolkit. Use this — do not write custom metric implementations.
- **BEIR / MTEB**: Standardized benchmarks for comparing embedding models (not needed for MVP — these evaluate models, not retrieval pipelines)

---

## 9. Comparison of Techniques

| Technique | Impact | Implementation Cost | Risk | MVP? |
|---|---|---|---|---|
| L2 normalization | High | Trivial (1 line) | None | Yes |
| Section segmentation + weighted concat | Medium-High | Low (string manipulation) | Low | Yes |
| Skill keyword extraction | Medium | Medium (regex/vocab) | Low | Yes |
| Data augmentation (template) | Medium | Medium | Medium (noise) | Yes |
| Mean centering | Low-Medium | Low | Low (if fit on train only) | Optional |
| PCA whitening | Low-Medium | Medium | Medium (noise amplification) | Optional |
| Hybrid BM25+dense | High | High (separate index) | Medium (complexity) | Future |
| LLM-based augmentation | High | High (cost, API) | High (quality) | Future |
| Fine-tuning embedding model | High | Very high (training pipeline) | High (overfitting) | Future |

---

## 10. Recommended Stack for MVP

### Must-Do (High Impact, Low Cost)

1. **Text preprocessing before embedding:**
   - Split each job posting into `title`, `skills`, `description` fields
   - For embedding input: `"[TITLE] {title} [TITLE] [TITLE] [SKILLS] {skills} [SKILLS] [DESC] {description}"` (title repeated 3×, skills 2×)
   - Normalize whitespace, lowercase (Sentence Transformer models are typically case-insensitive), strip HTML

2. **L2 normalization** after every embedding call — both indexing and query time. Verify with `np.linalg.norm()`.

3. **Golden test set**: 10–20 resumes with hand-labeled relevance for top-10 job matchings per resume.

4. **Metric tracking**: Precision@5, MRR via `pytrec_eval`.

### Should-Do (Medium Impact, Low-Medium Cost)

5. **Template-based data augmentation:** 300+ synthetic postings from 50 real seeds, tagged in metadata.

6. **Section-weighted embedding comparison:** Run an A/B evaluation (with vs without weighting) and document the delta in the README.

### Deferred (High Impact, High Cost)

7. Hybrid BM25 + Dense retrieval
8. PCA whitening
9. LLM-based augmentation
10. Fine-tuning Sentence Transformer on domain data

---

## 11. Sources

1. **Sciencedirect (2024)**: "Is text preprocessing still worth the time? A comparative survey on the influence of popular preprocessing methods on Transformers and traditional classifiers." *Information Processing & Management*, 2024.

2. **llmbestpractices.com (2026)**: "Embeddings: Normalization." May 2026.

3. **Shadecoder (2025)**: "Embedding Optimization: A Comprehensive Guide for 2025."

4. **techbloat.com (2026)**: "7 Advanced Feature Engineering Tricks Using LLM Embeddings." May 2026.

5. **EmergentMind (2025)**: "PCA Whitening: Theory & Applications." November 2025.

6. **Adnan Masood, PhD (2025)**: "Optimizing Chunking, Embedding, and Vectorization for Retrieval-Augmented Generation." Medium, September 2025.

7. **Pinecone (2025)**: "Chunking Strategies for LLM Applications." June 2025.

8. **alphaXiv (2026)**: "A Systematic Investigation of Document Chunking Strategies and Embedding Sensitivity." March 2026.

9. **DICE paper (2026)**: "Lost in a Single Vector: Improving Long-Document Retrieval with Chunk Evidence Aggregation." arXiv.

10. **Sciencedirect (2023)**: "Data augmentation techniques in natural language processing." *Applied Soft Computing*, 2023.

11. **Bouthillier, M. (2026)**: "Synthetic Data Generation for Dense Retrieval: A Comparison of LLM Prompting and Agentic Approaches." Master's Thesis, University of Waterloo.

12. **InPars (2022)**: "Unsupervised Dataset Generation for Information Retrieval." SIGIR 2022.

13. **Sciencedirect (2026)**: "AI-driven semantic similarity-based job matching framework." *Information Sciences*, 2026.

14. **Weaviate (2024)**: "Evaluation Metrics for Search and Recommendation Systems." May 2024.

15. **yaw.sh (2026)**: "Retrieval Evaluation: A Practical Guide (2026)." May 2026.

16. **PromptZ2H (2025)**: "Measuring Retrieval: Recall, Precision, NDCG."

17. **Michael Brenndoerfer (2026)**: "Hybrid Search: BM25 and Dense Retrieval Combined." January 2026.

18. **Brenndoerfer (2025)**: "Text Preprocessing: Complete Guide to Tokenization, Normalization & Cleaning for NLP." August 2025.

19. **OpenReview (2024)**: "Query-Paper Retrieval via Concatenative Embedding and Cosine Ranking with Linq-Embed-Mistral."

20. **GitHub: golugoodboy** (2026): "Agentic AI Job Copilot — LangGraph + LLM + Semantic Search." Includes weighted skill scoring and hybrid final score.
