# GROWING THE SEMANTIC CAREER MAP INTO A GENUINE CAREER EXPLORATION TOOL

> A vision document, not a plan. It describes where this project could go once the current
> lakehouse, embedding, and clustering work matures — and what homework has to happen first.
> See `docs/project.md` for the canonical status tracker and `docs/implementation-plan.md`
> for phased execution.

---

## The Core Idea

Today the system answers: *"which job postings look like this text?"* The next step is to answer
the question a real person asks: *"what should I do with my life, given who I am?"*

That shift turns a retrieval demo into a **career exploration tool**:

1. **Evoke the user's interests.** A guided personality / interest quiz (built on top of, not
   bolted onto, the search criteria) elicits the signals a resume does not capture — what energizes
   someone, how they like to work, what they want to avoid.
2. **Translate those answers into query dimensions.** Real information-retrieval practice: the quiz
   answers map onto engineered features of the job space (role family, work modality, autonomy,
   research-to-production ratio, customer contact, tooling, scale, ambiguity tolerance, ...).
3. **Retrieve using genuine IR mechanisms.** Dense semantic retrieval, hybrid lexical+dense scoring,
   faceted filtering, and — critically — **rank fusion and re-ranking** to combine the structured
   quiz signal with the unstructured embedding similarity.
4. **Synthesize with RAG.** Retrieved postings and their profiles become the grounding context for a
   language model that explains *in plain language* why these careers fit, what a day looks like,
   what to learn next, and where the fit breaks down.

The result is not a search box with a chatbot stapled on. It is a dialogue about careers that is
grounded in an engineered, inspected, and visualized semantic space over real job data.

---

## Why Genuine IR Matters Here

Career advice is a retrieval problem disguised as a conversation. The value is not in fluent prose
(it is cheap to generate) — it is in **retrieving the right roles for the right reasons** and being
able to *show your work*:

- **Recall matters more than precision at this stage.** A user exploring careers is best served by a
  diverse candidate set covering adjacent role families, not ten near-identical postings.
- **Queries are multi-dimensional and sparse.** A quiz yields structured facets plus a few free-text
  self-descriptions. Classic IR says: do not collapse that into one vector — combine lexical, dense,
  and facet filters, then fuse ranks.
- **Evaluation is the product.** Offline judgments on a hand-maintained golden set (already planned at
  `data/golden_set.json`) plus retrieval metrics (recall@k, nDCG, MRR, facet coverage, diversity) are
  what separate this from a vibe-based recommender.
- **Grounding is non-negotiable.** Every sentence the RAG layer produces must trace to retrieved
  evidence. Unsupported claims are a trust bug.

---

## The Work That Has to Happen First

### 1. Deep Job Profiling

Postings are noisy and title-inflated. A profile per posting (or per posting cluster) should include:

- **Structured facets** — seniority, IC vs. management, domain, modality (remote/hybrid/onsite),
  company stage, team function.
- **Semantic facets** — responsibility mix (research / build / operate / mentor / communicate),
  ambiguity tolerance, customer proximity, data vs. systems focus.
- **Derived signals** — required tooling, "hidden" expectations, and the gap between the title and
  what the text actually asks for.

This is where deep-profiling research (existing reports 003, 006, 008, 012) becomes the foundation.

### 2. Feature Engineering for Query Dimensions

The quiz produces dimensions; the job space must have matching dimensions to retrieve against. That
means:

- Designing each quiz axis so it has a **defensible job-side counterpart**.
- Choosing per-axis representations: dense text embedding, extracted categorical field, or a small
  supervised/ranking model trained on labeled profiles.
- Documenting the mapping table (quiz dimension → job feature → retrieval mechanism).

### 3. Embedding Space Engineering

The latent space must be engineered, not inherited:

- Pick and justify the model (work already begun in reports 001, 023, 024).
- Define preprocessing that preserves career-relevant signal (report 008).
- **Visualize the space early and constantly** — projection plots, neighborhood diagnostics, per-
  cluster exemplars, drift checks between data slices. If a space cannot be read by a human, it cannot
  be trusted by a user.

### 4. Doing the Homework

Before writing the exploration feature, study and document:

- Foundational IR: term weighting, inverted indexes vs. ANN, hybrid retrieval, rank fusion (RRF /
  reciprocal rank fusion), re-ranking, evaluation metrics and judgment design.
- Semantic search optimization: query expansion, negative examples, hard negatives, calibration of
  similarity scores, diversity/MMR, faceted search.
- Embedding-space engineering: contrastive fine-tuning, normalization, anisotropy/whitening, PCA vs.
  UMAP trade-offs, cluster validity metrics, and deployment-scale considerations already covered in
  reports 009–012, 019, 021–025.
- RAG practice: grounded prompting, citation of evidence, refusal when retrieval is empty, and
  evaluation of grounded answers (report 004 provides a starting point).

---

## What This Shows

The completion of this vision puts on display the intersection of technical communication and data
visualization: an embedded, visual, human-readable explanation of how career interests map into an
engineered semantic space, backstopped by measurable retrieval quality. The artifact is as much a
demonstration of rigorous thinking as it is a working product.

---

## Open Questions

- Which quiz instrument is defensible and small enough to complete in one sitting?
- How do we avoid overfitting careers to personality while still making personalization meaningful?
- Do we retrieve postings or **profiles of role families**? (Probably both: families for exploration,
  postings for evidence.)
- What is the minimum evaluation harness that makes a RAG answer trustworthy?
