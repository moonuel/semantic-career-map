# Evaluation

## Evaluation Framework

The core question: **Does the system retrieve relevant job postings for a given resume or query?**

Two levels of evaluation:

| Level | Status | Requires Labels? | Purpose |
|---|---|---|---|
| Proxy metrics | Active | No | Fast iteration, compare preprocessing variants |
| IR metrics (pytrec_eval) | Planned | Yes (golden set) | Proper retrieval quality measurement |

---

## Proxy Metrics

Proxy metrics provide rapid feedback during the experiment loop without requiring labeled data. They are computed directly from the embedding matrix and existing role labels.

### Self-Retrieval Check

A degeneracy check verifying that every posting maps to a unique vector. If any two postings produce identical embeddings, the embedding space has collapsed.

**Implementation** (`scripts/003_llm_extraction/compare_variants.py:57–70`):

```python
def self_retrieval(embeddings: np.ndarray, labels: list[str]) -> float:
    sim = embeddings @ embeddings.T
    hits = 0
    total = len(labels)
    for i in range(total):
        rankings = np.argsort(-sim[i])
        top_idx = rankings[0]
        if top_idx == i:
            hits += 1
        else:
            second = rankings[1] if len(rankings) > 1 else None
            if second == i:
                hits += 0.5
    return hits / total * 100
```

Scoring: full credit (1.0) if diagonal is top result, half credit (0.5) if second-best, zero otherwise. Score = hits/total × 100%.

### Separation Gap

Measures how well embeddings separate by role category. Computed as `mean(within-role cosine similarity) − mean(cross-role cosine similarity)`.

**Implementation** (`scripts/003_llm_extraction/compare_variants.py:73–88`):

```python
def separation_gap(embeddings: np.ndarray, roles: list[str]) -> float:
    role_set = sorted(set(roles))
    if len(role_set) < 2:
        return 0.0
    sim = embeddings @ embeddings.T
    same_role_sims = []
    diff_role_sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if roles[i] == roles[j]:
                same_role_sims.append(sim[i, j])
            else:
                diff_role_sims.append(sim[i, j])
    if not same_role_sims or not diff_role_sims:
        return 0.0
    return np.mean(same_role_sims) - np.mean(diff_role_sims)
```

Higher values indicate better cluster separation. Analogous to Fisher's linear discriminant adapted for cosine similarity.

!!! warning "Limitation"
    Separation gap uses title-based role labels — these are noisy. A "Data Scientist" doing SQL analytics and a "Data Scientist" building NLP models share the same label but do fundamentally different work. Function-based labels will provide a more honest clustering target.

### Results

| Variant | Self-Retrieval | Separation Gap | Mean CosSim |
|---|---|---|---|
| Raw (`raw_full_text`) | 100.0% | +0.0060 | 0.4045 |
| Section-Cleaned (`clean_text`) | 100.0% | +0.0203 | 0.4298 |
| **LLM-Cleaned (`llm_clean_text`)** | **100.0%** | **+0.0493** | 0.5280 |
| Job-context (golden) | 100.0% | +0.0255 | 0.4912 |
| Role-context (golden) | 100.0% | +0.0588 | 0.4149 |
| Job-context (LLM) | 100.0% | +0.0385 | 0.4783 |
| Role-context (LLM) | 100.0% | -0.0074 | 0.4037 |

Experiment 005 demonstrated that semantic partitioning of job postings into job-context and role-context dimensions is feasible. Golden role-context achieves the highest separation gap (+0.0588, 8.5× raw), while LLM-extracted role-context shows a negative separation gap (-0.0074), indicating this dimension is harder to extract cleanly via LLM prompting.

---

## IR Metrics (Planned — Phase 1.9)

Proper information retrieval evaluation using `pytrec_eval`, the Python interface to TREC's `trec_eval`.

### Metrics

| Metric | Formula | Primary For |
|---|---|---|
| Precision@K | relevant_in_top_k / K | Fixed-size result UIs — **primary metric** |
| Recall@K | relevant_in_top_k / total_relevant | Retrieval completeness |
| MRR | 1 / rank_of_first_relevant | Single-best-match quality — **secondary** |
| NDCG@K | DCG / IDCG (position-weighted) | Graded relevance ranking quality |

### Planned Setup

```python
import pytrec_eval

qrels = {}   # ground truth: {query_id: {doc_id: relevance}}
run = {}     # system output: {query_id: {doc_id: score}}

for resume in golden_set:
    query_emb = model.encode(resume["text"], normalize_embeddings=True)
    similarities = cosine_similarity([query_emb], job_embeddings)[0]
    ranked = np.argsort(similarities)[::-1]
    run[query_id] = {
        jobs[i]["id"]: float(similarities[i])
        for i in ranked[:10]
    }

evaluator = pytrec_eval.RelevanceEvaluator(
    qrels, {"P.5", "recall.5", "NDCG", "MRR"}
)
results = evaluator.evaluate(run)
```

The golden set will consist of 5–10 synthetic resumes with hand-labeled relevance scores (0 = irrelevant, 1 = somewhat relevant, 2 = highly relevant) against the 27 real job postings.

## HDBSCAN Cluster Validation (Planned — Step 1.3b)

Unsupervised clustering as a sanity check on embedding quality. Run after every preprocessing experiment to track cluster structure improvement:

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2, min_samples=1,
    metric="euclidean", cluster_selection_method="eom",
)
labels = clusterer.fit_predict(embeddings)
```

## Evaluation Limitations

1. **Proxy metrics use noisy labels:** Title-based role categories conflate different types of work
2. **Small dataset:** 27 postings with 6 categories — O(n²) pairs but limited generalizability
3. **No held-out set:** All 27 postings used for both embedding and evaluation (acceptable for proxy metrics, not for final IR evaluation)
4. **Golden set not yet built:** Proper IR metrics await construction of synthetic resumes with hand-labeled relevance scores
