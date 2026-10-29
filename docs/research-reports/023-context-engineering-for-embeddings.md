# Context Engineering for Embeddings — Stories, Keyword Sets, and Structured Text

> Date: 2026-10-29
> Context: Semantic Career Mapping Platform — research into whether rewriting text before embedding (narrative prose, keyword sets, structured field output) improves sentence-embedding quality for semantic similarity, retrieval, and clustering. The production encoder is `sentence-transformers/all-MiniLM-L6-v2` with a 256-token input limit.
> Scope: pre-embedding text transformation only. Model fine-tuning, vector post-processing, and ANN index configuration are out of scope except where they confound the comparison.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Real Question: Format, Content, or Length?](#2-the-real-question-format-content-or-length)
3. [What the Encoder Expects](#3-what-the-encoder-expects)
4. [Evidence by Representation Family](#4-evidence-by-representation-family)
5. [Query-Side vs Document-Side Rewriting](#5-query-side-vs-document-side-rewriting)
6. [Synthesis: Which Format Wins?](#6-synthesis-which-format-wins)
7. [Proposed Experiment for This Project](#7-proposed-experiment-for-this-project)
8. [Implementation Notes](#8-implementation-notes)
9. [Threats to Validity](#9-threats-to-validity)
10. [Recommendation](#10-recommendation)
11. [Sources](#11-sources)

---

## 1. Executive Summary

There is no single best rewrite format. The literature does not contain a clean head-to-head
comparison of "story vs keyword list vs JSON" for sentence embeddings. What it does contain is a
consistent set of mechanisms that predict when each format helps:

1. **Embedding models are trained on natural language.** Inputs that look like prose are
   in-distribution. Keyword lists, JSON, and isolated words are out-of-distribution and can lose
   substantial accuracy even when they contain the same facts.
2. **Format cannot be separated from information loss and length.** A story that preserves every
   fact is different from a summary that drops facts. A keyword list that drops grammar is
   different from one that keeps relations. A rewrite can win because it removes boilerplate, not
   because its format is intrinsically better.
3. **Longer is not better.** Transformer embeddings exhibit length collapse: longer texts cluster
   toward the centre of the space and become more similar to everything. `all-MiniLM-L6-v2`
   additionally truncates at 256 word pieces, and was trained at a 128-token sequence length.
4. **Structure helps when it is semantic, not syntactic.** Field-labelled natural language
   ("Job title: ... Role: ... Skills: ...") can help. JSON serialization usually hurts because
   braces, quotes, and keys add tokens that carry no meaning to the embedder.
5. **The strongest pattern is multi-view retrieval, not one perfect rewrite.** Raw prose, a
   keyword view, and a summary view are complementary. Fuse them at the score or vector level
   rather than concatenating them into one long input.

For this project's job-posting similarity task, the evidence supports this ordering for the
**primary dense representation**:

| Rank | Representation | Verdict for `all-MiniLM-L6-v2` job similarity |
|---|---|---|
| 1 | Short, field-aware natural-language prose | Safest. In-distribution, keeps relations, controllable length. |
| 2 | Q&A / labelled bullet prose | Close second. Helps when queries mirror field intents. |
| 3 | Keyword set | Useful as an auxiliary view or sparse index; weak as the only dense input. |
| 4 | Narrative story | Only for narrative, role-function, or behavioural similarity; risky as a primary factual representation. |
| 5 | JSON / YAML serialization | Usually worst. Syntax tokens dilute the vector and the format is far from training data. |

The practical recommendation is not "pick a format". It is: **keep a clean prose baseline, add a
keyword view and a short role-function narrative as separate views, and evaluate late fusion
against the baseline.** For a 27-posting pilot, this is cheap enough to test and avoids betting the
pipeline on an unverified rewriting strategy.

---

## 2. The Real Question: Format, Content, or Length?

A rewrite changes at least four variables at once:

| Variable | Example | Why it confounds the comparison |
|---|---|---|
| **Content retention** | Summary drops salary and benefits; story invents a team size | A gain may be boilerplate removal or hallucination, not format |
| **Distribution match** | Prose vs JSON vs bare word list | The encoder was trained on prose, not JSON |
| **Length / token budget** | 80-word story vs 40-word keyword list | Longer inputs truncate or collapse toward the centroid |
| **Information structure** | "Python, PyTorch, AWS" vs "Built training pipelines in PyTorch on AWS" | Relations, negation, seniority, and scope are lost or preserved |

The clean experiment is therefore not "does a story beat a keyword list?" It is a factorial
comparison that holds facts and token budget roughly constant while varying representation. Most
published work does not do this, so its results must be read as evidence about a *pipeline*, not
about a format in isolation.

This matters directly for job postings. A narrative rewrite may improve role-function similarity
because it expresses what the person does, which is the project's working hypothesis. But if the
narrative invents a team size, a seniority level, or a technology stack, it can improve cluster
coherence while making retrieval less faithful. The experiment must score both.

---

## 3. What the Encoder Expects

### 3.1 Training distribution

`all-MiniLM-L6-v2` is a 6-layer MiniLM distilled from BERT, producing 384-dimensional vectors. It
was fine-tuned on more than 1 billion sentence pairs, with a training sequence length of 128
tokens; the released model truncates inputs at 256 word pieces [1][2]. Its training objective is
semantic similarity over natural-language pairs, not schema matching or keyword lookup.

That does not mean prose is always superior. It means a representation that looks like the
training distribution has an advantage before any semantic content is considered. JSON is the
clearest violation: `{"title": "ML Engineer", "skills": ["Python", "PyTorch"]}` contains syntax
tokens that the model never learned to treat as meaningful structure. A keyword list is a milder
violation: it is English, but it removes the syntactic relations that distinguish "Python required"
from "Python preferred" or "led a team using Python" from "used Python".

### 3.2 Prefixes and instructions are model-specific

Asymmetric embedding models are trained with different prefixes for queries and passages. E5 uses
`query:` and `passage:` prefixes; the E5 authors report that breaking the symmetry this way is
important for retrieval tasks where the target corpus contains paraphrases of the query [3]. The
Pinecone practitioner guide is explicit that omitting the prefix causes performance degradation
[4]. EmbeddingGemma goes further, with task-specific prefixes and a document form that includes
`title:` and `text:` fields [5].

Two lessons follow:

- **Use the input contract the model was trained with.** If the model card specifies prefixes,
  use them. Do not invent a JSON wrapper around them.
- **Prompt and instruction sensitivity is large.** A 2026 study evaluated 40 task-specific
  prompts across 11 datasets and 6 embedding models and found that the default prompt can sit in
  the tail of the score distribution, and that leaderboard rankings are not robust to prompt
  selection [6]. Format changes must be evaluated on the target data, not assumed from a model
  card.

### 3.3 Long inputs collapse and truncate

Two independent effects punish verbose rewrites:

- **Truncation.** `all-MiniLM-L6-v2` truncates at 256 word pieces by default [1]. A 300-word
  "day in the life" story loses its tail. A maintainer experiment on TREC-COVID found NDCG@10 of
  0.47246 at 256 tokens versus 0.46083 at 512, and argued that extending beyond the training
  length makes the model slower and worse [2].
- **Length collapse.** A 2024/2025 line of work shows that PLM-based embeddings of longer texts
  cluster toward the origin and have higher cosine similarity to each other; the proposed
  TempScale correction improves MTEB by 0.94% and LongEmbed by 1.10% [7]. In retrieval terms,
  long documents occupy a central, low-variance region, so they can appear spuriously similar to
  many queries.

This is the strongest argument against long narrative rewrites for a short-context model. A story
that is semantically richer but 2-4x longer may lose more from truncation and collapse than it
gains from added context.

### 3.4 Anisotropy and hubness

Sentence-BERT spaces exhibit hubness: a few texts are neighbours of many points while most are
neighbours of few [8]. Hubness reduction methods can cut error rate by about 9% on a tested model
[8]. Rewrites that make texts more generic (summaries, stories with common narrative scaffolding)
can increase hubness by pushing documents toward common directions. Rewrites that preserve
distinctive terms (skills, tools, domain nouns) keep documents separable.

The implication is not that generic text is always bad. It is that **a representation should be
judged by its discrimination, not only by how well it reads.** A beautiful narrative that makes
every job sound like "collaborative engineer who builds ML systems" is a hub.

---

## 4. Evidence by Representation Family

### 4.1 Natural-language prose and contextual prefixes

This is the best-supported family.

**Contextual Retrieval.** Anthropic prepends a 50-100 token, chunk-specific situating preamble to
each chunk before embedding and BM25 indexing. Reported top-20 retrieval failure rates: 5.7%
baseline, 3.7% with contextual embeddings (-35%), 2.9% with contextual embeddings plus contextual
BM25 (-49%), and 1.9% with reranking (-67%) [9]. The gain comes from restoring referents that
chunking destroyed, not from a particular prose style.

**Document context.** DAPR found that 53.5% of passage-retrieval errors were due to missing
document context, and that prepending the document title improved hard queries, though overall
performance remained modest [10]. This supports short structural prefixes (title, section) as a
cheap first step before generated context.

**Summaries are a noise filter, not a universal win.** A 2026 study on Portuguese consumer
complaints found Recall@1 of 0.527 for LLM-generated structured summaries versus 0.001 for raw,
noisy interactions [11]. That is an extreme case where raw text was almost unusable. Summary RAG
improved NDCG@100 by 16.5% on NFCorpus and 12.2% on TREC-COVID, but degraded SciFact by 7.9%
[12]. SciFact is fact-dense and claim-verification oriented: summaries dropped the exact details
that mattered.

**Lesson for job postings.** A concise prose rewrite that removes boilerplate and restores role,
seniority, and section context is well supported. A generic summary that drops specific tools,
quantifiers, and qualifications is not. The right prose rewrite is *additive and specific*, not
*short and abstract*.

### 4.2 Keyword sets and word lists

Keyword representations are attractive because they are compact, interpretable, and cheap. The
evidence says they are a good auxiliary signal and a poor sole dense representation.

**Multi-view indexing.** MC-indexing represents each document chunk in three views: raw text,
keywords, and summary, then retrieves from each view and aggregates [13]. Summary view performed
best overall, raw text second, and the keyword view underperformed overall because it has poor
semantic structure. Crucially, the keyword view still solved some tasks that the other views
missed. That is the pattern: keywords have low average quality but high complementary recall for
exact entities, tools, and rare terms.

**Bare words are out-of-distribution.** A 2025 study found that some text embedding models fail
completely on isolated words (Spearman correlation near zero on SimLex-999) but recover when
given a semantic prompt such as "meaning: dog" [14]. The result is model-dependent, but it
confirms that a token list is not a neutral representation. The embedder may not have learned to
encode a word without sentence context.

**Keyword-only loses relations.** "Python, AWS, leadership" does not distinguish "led a team that
used Python on AWS" from "used Python and AWS while reporting to a lead". For job similarity,
seniority and scope are often the discriminating signal. A keyword set is therefore better used
as a sparse/BM25 view or as one vector in a fusion than as the only input.

**Project evidence already points this way.** JobBERT learns job-title representations from
associated skills, showing that skills carry strong occupational signal [15]. But it uses skills as
a training signal for a title encoder; it does not claim that a bare skill list is a sufficient
document representation. A 2018 job recommender found the best results by combining job titles
with full-text descriptions [16]. The pattern is consistent: structured fields add signal, but
they do not replace prose.

### 4.3 Structured output formats

"Structured" means two very different things, and conflating them causes bad decisions:

1. **Structured content** — the rewrite organises information into named fields, e.g. title,
   responsibilities, required skills, seniority, domain.
2. **Structured serialization** — the rewrite is emitted as JSON, YAML, XML, or another syntax.

Structured content is often good. Structured serialization is usually bad for a bi-encoder.

**JSON syntax is noise.** A controlled study across 250 retrieval trials and five technical
domains, using `all-MiniLM-L6-v2`, compared standard prose, Q&A pairs, fact-dense lists, and
structured JSON-LD [17]. Chunked prose had the highest Top-1 win rate (28%), Q&A had the best MRR
(0.541), and structured JSON-LD had a 4% win rate and MRR 0.260. The authors attribute the JSON
failure to tokenization: braces, quotes, colons, and keys become tokens that dilute the semantic
content. This is a small, non-peer-reviewed study and should be treated as directional, but the
mechanism is credible and consistent with the training-distribution argument.

**JSON output can hurt even when the task is extraction.** An EMNLP 2026 industry study on
greenhouse-gas extraction found that a structured prompt with roles, definitions, extraction
rules, and JSON formatting underperformed a simpler question-style prompt; adding JSON output to
the question prompt modestly hurt it [18]. Again, the effect is task- and model-specific, but the
"more structure is better" assumption does not hold automatically.

**Schema-guided natural-language summaries can be excellent.** The Portuguese complaint study
used schema-guided structured summaries and achieved the large Recall@1 gain described above
[11]. Note what that pipeline did: the schema guided *what to include*, but the indexed text was
still natural language. That is the useful pattern.

**Recommendation.** If fields are needed, write them as short natural-language lines:

```text
Job title: Machine Learning Engineer
Role: Builds and deploys production ML systems.
Required skills: Python, PyTorch, AWS, Kubernetes.
Responsibilities: Owns model training, deployment, and monitoring.
Qualifications: 3+ years of ML engineering experience.
```

Do not write:

```json
{"title":"Machine Learning Engineer","role":"...","skills":["Python","PyTorch","AWS"]}
```

The first is still prose with useful labels. The second adds syntax the encoder was not trained to
interpret. If a downstream system needs JSON, generate JSON for storage but embed the prose
rendering.

### 4.4 Stories and narrative rewriting

The evidence here is the weakest and the most easily misread.

**Story embeddings are a task-specific result, not a general recipe.** The Story Embeddings paper
trains a contrastive model so that reformulations of the same narrative land close together; it is
evaluated on narrative retrieval and narrative understanding [19]. It does not show that rewriting
a job posting as a story improves a general sentence encoder. The model was trained for narrative
similarity; a general MiniLM was not.

**Narratives can capture function better than keyword lists.** A role-function narrative ("This
engineer spends most of their time building training pipelines, deploying models, and monitoring
drift") makes the *what someone does* explicit. That aligns with the project's hypothesis that
jobs cluster by responsibilities rather than titles. If the similarity target is role function,
narrative can be a better representation than a tool list.

**But narratives add length, hallucination risk, and genericness.** Three failure modes:

- **Length collapse and truncation** (§3.3): a story is the longest candidate representation.
- **Hallucinated specifics**: a model asked to write "a day in the life" will invent team size,
  cadence, and tools. Those inventions become part of the vector and are unrecoverable at query
  time.
- **Narrative convergence**: many jobs become "collaborate with stakeholders, build models, ship
  impact". The story may be more readable and less discriminative.

**The defensible use of narrative is query-side, not document-side.** HyDE generates a
hypothetical document from a query, embeds it, and retrieves real documents; the encoder's dense
bottleneck filters some hallucination because the generated document only needs to land in the
right neighbourhood [20]. Query2Doc does the same with pseudo-documents for query expansion [21].
On the document side, a fabricated narrative is indexed permanently and cannot be filtered by
retrieval. That asymmetry matters.

**Verdict.** Use narrative as an auxiliary view for role-function clustering or as a query-side
expansion. Do not make it the primary document embedding unless an experiment on this corpus shows
a gain after controlling for length and hallucination.

### 4.5 Document expansion and dual indexing

Document expansion is the closest well-studied analogue to "rewrite the document before
embedding".

- **Doc2query** predicts queries a document could answer and appends them; it improves BM25 and
  sparse retrieval [22].
- **Query2Doc** generates pseudo-documents from a query and expands the query; it improves BM25 by
  3-15% on ad-hoc IR and also helps dense retrievers [21].
- **Doc2Query++** notes that naively appending expansions degrades dense retrieval because the
  added queries inject semantic noise; it proposes separate text and query indices with dual-index
  fusion [23].
- **Anthropic's contextual retrieval** uses a single situating preamble, not a set of predicted
  queries, and reports gains across embedding models [9].

The consistent lesson: **expansion helps when the added text is relevant and when the retrieval
system can isolate it.** Concatenating everything into one long input is the naive version and is
where dense retrieval often loses.

---

## 5. Query-Side vs Document-Side Rewriting

Semantic search is asymmetric. The query is short; the document is long. A rewrite strategy that
helps one side can hurt the other.

| Side | Typical technique | Evidence | Risk |
|---|---|---|---|
| Query | HyDE, Query2Doc, query rewriting | Strong zero-shot gains; hypothetical docs land in the right neighbourhood [20][21] | Query drift; hallucinated constraints |
| Document | Contextual preamble, summary, keyword view, doc2query | Contextual retrieval reduces failures 35-67% [9]; summaries help noisy corpora but hurt fact-dense ones [12] | Permanent noise; hallucination cannot be filtered at query time |
| Both | Multi-view indexing + late fusion | Complementary views outperform any single view in MC-indexing [13] | More vectors, more storage, fusion tuning |

For job search, the query might be a job title, a candidate profile, or a free-text description.
A candidate profile is itself a document, which makes the task closer to symmetric similarity. In
that case, rewriting both sides into the same format is reasonable, but the format should match the
encoder's training distribution and the same length budget.

The important design choice is not "should we rewrite?" but **which side carries the rewrite and
whether the rewrite is stored or generated on the fly.** Query-side rewrites are cheap to iterate
and easy to A/B test. Document-side rewrites change the index and require re-embedding, so they
need stronger evidence before adoption.

---

## 6. Synthesis: Which Format Wins?

### 6.1 Decision matrix

| Representation | Semantic fidelity | Distribution match | Length control | Discrimination | Best use |
|---|---|---|---|---|---|
| Raw cleaned prose | High | High | Medium | High | Baseline |
| Field-labelled prose | High | High | Medium | High | Primary candidate |
| Q&A prose | High | High | Medium | Medium-high | Query-like fields |
| Contextual preamble + prose | High | High | Medium | High | Chunked long docs |
| Keyword list | Low-medium | Low | High | Low-medium | Auxiliary/sparse view |
| Summary | Medium | High | High | Medium | Noisy long docs |
| Narrative story | Medium | Medium-high | Low | Low-medium | Role-function view |
| JSON/YAML | Medium | Low | Medium | Low | Storage, not embedding |

"Semantic fidelity" means facts and relations survive. "Discrimination" means the representation
does not make all documents look alike. A keyword list has high length control but poor fidelity
and discrimination; a story has good readability but poor length control and can be generic.

### 6.2 Why the ranking is not universal

The ranking depends on the similarity target:

- **Factual similarity** ("same skills, same seniority, same domain") favours prose or field-labelled
  prose. Keywords lose relations; stories hallucinate.
- **Role-function similarity** ("same kind of work") favours narrative or responsibility-focused
  prose. This is the project's clustering target.
- **Lexical/entity matching** ("same tool, same certification") favours keywords plus BM25. Dense
  embeddings are weaker at exact rare-term matching.
- **Narrative similarity** ("same plot") needs a narrative-trained model, not a general MiniLM
  [19].

A single embedding cannot be optimal for all four. That is the argument for multi-view retrieval:
one view for prose semantics, one for exact terms, optionally one for role function, then fuse.

### 6.3 What "context engineering" should mean here

Context engineering is often described as prompt design for generation. For embeddings, it is
better understood as **input construction under a fixed encoder contract**:

1. Preserve the facts that define the similarity target.
2. Remove text that is irrelevant to that target (boilerplate, EEO, salary).
3. Match the encoder's training distribution (natural language, expected prefixes).
4. Stay inside the token budget, measured with the actual tokenizer.
5. Prefer multiple narrow views over one long concatenation.
6. Evaluate the representation, not the prose quality.

---

## 7. Proposed Experiment for This Project

### 7.1 Hypotheses

- **H1 (prose baseline).** Field-labelled prose beats raw cleaned text on separation gap and
  Recall@10, because it restores role and seniority context that flat text dilutes.
- **H2 (keywords).** A keyword-only embedding underperforms prose on semantic similarity but adds
  complementary recall for exact tools and rare terms in a fused index.
- **H3 (structure).** JSON serialization underperforms the equivalent field-labelled prose at
  equal information content, because syntax tokens dilute the vector.
- **H4 (narrative).** A grounded 80-120 word role-function narrative improves role-function
  clustering (ARI/AMI against function labels) but does not improve, and may hurt, fact-level
  retrieval.
- **H5 (multi-view).** Late fusion of prose + keyword (+ narrative) beats any single view on
  Recall@10 and separation gap.

### 7.2 Variants

Hold the source posting and the cleaning step constant. Generate all variants from the same
cleaned text with the same LLM and prompt version. Store the original and every derived text.

| ID | Variant | Sketch |
|---|---|---|
| A | Raw cleaned text | Current baseline |
| B | Field-labelled prose | Title, role, skills, responsibilities, qualifications as short natural-language lines |
| C | Q&A prose | "What does this role do? ... What skills are required? ..." |
| D | Keyword set | Canonical title + skill/technology terms only |
| E | Role-function narrative | 80-120 words, grounded, no invented facts; "A typical week in this role..." |
| F | JSON | Same fields as B, serialized as JSON |
| G | Multi-view | Separate embeddings for B, D, E; fuse by normalized average and by reciprocal rank fusion |

Variant F is included as a negative control, not because it is expected to win. Variant E must be
fact-checked against the source; any invented entity, tool, or seniority claim disqualifies the
generation and should be regenerated.

### 7.3 Controls

- **Same facts.** B, C, E, and F should preserve the same core fields. Measure field coverage with
  a checklist or an LLM judge against the source, and report omissions.
- **Same token budget.** Measure tokens with the MiniLM tokenizer. Report mean, median, and
  truncation rate per variant. If a variant exceeds 256 tokens, truncate deliberately at a section
  boundary and record it.
- **Same encoder.** One model, one pinned revision, one normalization setting. Do not change the
  model while testing format.
- **Same prompt family.** Use one generation prompt per variant, but run at least 3 prompt
  paraphrases to estimate prompt sensitivity, following the multi-prompt evaluation
  recommendation [6].
- **Same split.** If using the golden set, keep it fixed and hand-maintained. Do not auto-generate
  labels.

### 7.4 Metrics

Use the project's existing proxy metrics plus standard IR metrics:

| Metric | What it measures | Why |
|---|---|---|
| Self-retrieval@1 | Does each posting retrieve itself? | Sanity check; already used in the project |
| Separation gap | Distance between same-role and different-role pairs | Existing project metric; sensitive to role signal |
| Recall@10 / MRR / NDCG@10 | Ranking quality against golden relevance | Standard IR; catches gains that separation gap misses |
| ARI / AMI vs function labels | Cluster agreement | Tests the role-function hypothesis directly |
| Truncation rate / token count | Length confound | Mandatory control |
| Hallucination / omission rate | Faithfulness | Narrative and summary variants can look good for the wrong reason |

With 27 postings, these metrics are a pilot, not a powered study. Report confidence intervals via
paired bootstrap over postings (or over query-posting pairs) and treat small deltas as
inconclusive. The cleaned 100-posting sample may provide a better pilot if labels can be extended.

### 7.5 Expected outcomes and interpretation

- If B beats A, the gain may be context restoration or boilerplate removal. Inspect examples to
  attribute it.
- If D beats B on Recall@10 but loses on separation gap, keywords are adding lexical recall while
  hurting semantic structure. That supports a hybrid, not a replacement.
- If E improves ARI but hurts Recall@10, narrative is a clustering view, not a retrieval view.
- If F loses to B, do not conclude "structure is bad". Conclude "JSON syntax is bad"; test
  field-labelled prose separately.
- If G beats all single views, adopt multi-view indexing and defer the question of which single
  format is best.

---

## 8. Implementation Notes

### 8.1 Fit the existing configuration model

The embedding pipeline already exposes `text_variant` and `variant` in `EmbeddingConfig`
(`backend/pipeline/embedding/config.py`), and `scripts/embed.py` resolves a pinned model revision
and configuration hash. Derived text variants should be added as new `text_variant` values,
generated and stored upstream of embedding, not by mutating experiment scripts. The original
scripts in `scripts/` remain immutable artifacts.

A practical layout:

```text
data/derived_text/<variant>/<source_hash>.txt
```

Store the generator prompt version, model, and source hash alongside each derived text. The
embedding config's `configuration_hash` then covers the variant only if the variant name is part
of the config; if the derived text is an input artifact, its version belongs in
`input_artifact`/`input_version`.

### 8.2 Token-budget discipline

Before generating, measure the source token count. After generating, measure the variant token
count with the exact tokenizer used by the encoder. Reject or truncate variants that exceed the
budget. For MiniLM, 256 word pieces is a hard truncation, not a soft preference, and the model was
trained at 128 [1][2]. A 200-token field-labelled prose target is a reasonable default for this
project; a 400-token narrative is not.

### 8.3 Multi-view fusion

Prefer late fusion:

- **Vector average** of L2-normalized views, then re-normalize. Simple, but assumes views are
  equally reliable.
- **Weighted average** with weights tuned on a validation split.
- **Reciprocal rank fusion (RRF)** over per-view ranked lists. More robust when score scales
  differ, and it does not require the views to share a vector space.

Avoid concatenating all views into one long input. That recreates the length-collapse and dilution
problems the views were meant to avoid, and Doc2Query++ specifically reports that naive
concatenation degrades dense retrieval [23].

### 8.4 Caching and provenance

LLM rewriting is a one-time indexing cost, not a query-time cost. Cache by `(source_hash, prompt_hash,
model, variant)`. Store the original text separately from the derived text. Never display the
generated narrative as source evidence; it is a retrieval aid and may be wrong, the same principle
Anthropic applies to contextual preambles [9].

### 8.5 Do not change two things at once

The project's research rules already say to change one variable and re-measure [24]. Format,
model, and cleaning are separate variables. If the variant changes the cleaning step, it is a
cleaning experiment, not a format experiment.

---

## 9. Threats to Validity

- **No direct head-to-head evidence.** The literature compares pipelines, not isolated formats.
  The ranking in §6 is a synthesis of mechanisms and adjacent results, not a meta-analysis.
- **Model specificity.** Format effects are model-dependent. A model trained on JSON or on
  keyword lists could behave differently. The results here are most relevant to MiniLM-style
  bi-encoders trained on natural-language pairs.
- **Benchmark mismatch.** Most evidence comes from retrieval benchmarks (BEIR, MS MARCO, TREC)
  rather than semantic textual similarity or clustering. Retrieval rewards recall of relevant
  documents; clustering rewards separation. A format can help one and hurt the other.
- **LLM rewriting drift.** The generator can omit, add, or normalise facts. The experiment must
  measure faithfulness, not just downstream scores.
- **Small corpus.** 27 postings cannot support fine-grained conclusions. Use the cleaned 100
  sample or augmented data for a pilot with more power, and keep the golden set hand-maintained.
- **Label noise.** Function labels are themselves an interpretation. ARI/AMI gains against noisy
  labels are weak evidence.
- **Preprint quality.** Some cited studies are recent preprints or non-peer-reviewed venue papers.
  They are used for directional evidence, not as settled fact.

---

## 10. Recommendation

**Short answer.** For a MiniLM-style sentence encoder and job-posting similarity, the best default
is **short, field-labelled natural-language prose**, not a story, not a bare keyword set, and not
JSON. Add a **keyword view** for exact tools and rare terms, and an optional **short role-function
narrative** as a separate view for clustering. Fuse views late. Do not concatenate them into one
long input.

**Why.**

- Prose matches the encoder's training distribution and preserves relations.
- Field labels add context without JSON syntax noise.
- Keyword lists are compact and complementary but lose grammar and relations; bare words can be
  out-of-distribution.
- Stories are only clearly justified when the target similarity is narrative or role-function, and
  they add length, hallucination, and genericness risks.
- JSON is the worst fit for a bi-encoder trained on natural language, and the evidence points to a
  measurable penalty.

**For this project, in order:**

1. Build a field-labelled prose variant (B) and measure it against the current cleaned-text
   baseline (A) on separation gap, Recall@10, and MRR.
2. Add a keyword-only variant (D) and test late fusion with B. Do not replace B.
3. Add a grounded role-function narrative (E) and test it specifically on function-label
   clustering (ARI/AMI), not on fact-level retrieval.
4. Include a JSON variant (F) as a negative control to confirm the syntax penalty on this corpus.
5. Run at least 3 prompt paraphrases per variant and report the spread, not a single score.
6. Store original and derived text, prompt version, model revision, and token counts with every
   run.

The expected outcome is not "one format wins everywhere". It is a map of which representation
carries which kind of similarity, which is exactly what a multi-view semantic career map needs.

---

## 11. Sources

1. Sentence Transformers. "sentence-transformers/all-MiniLM-L6-v2" model card. <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
2. Sentence Transformers model discussion. "max_seq_length: 512 in sentence_bert_config.json" (maintainer experiment on TREC-COVID). <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/discussions/54>
3. Wang et al. (2022). "Text Embeddings by Weakly-Supervised Contrastive Pre-training" (E5). arXiv:2212.03533. <https://arxiv.org/html/2212.03533v2>
4. Pinecone. "The Practitioner's Guide to E5" (2024). <https://www.pinecone.io/learn/the-practitioners-guide-to-e5/>
5. Google AI for Developers. "Generating Embeddings with EmbeddingGemma" (2026). <https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers>
6. Kostiuk & Enevoldsen (2026). "One prompt is not enough: Instruction Sensitivity Undermines Embedding Model Evaluation." arXiv:2605.22544. <https://arxiv.org/html/2605.22544v2>
7. Zhou et al. (2024/2025). "Length-Induced Embedding Collapse in PLM-based Models." arXiv:2410.24200. <https://arxiv.org/html/2410.24200v2>
8. Nielsen et al. (2024). "Hubness Reduction Improves Sentence-BERT Semantic Spaces." PMLR v233. <https://proceedings.mlr.press/v233/nielsen24a.html>
9. Anthropic Engineering (2024). "Introducing Contextual Retrieval." <https://www.anthropic.com/engineering/contextual-retrieval>
10. DAPR (2024). "DAPR: A Benchmark on Document-Aware Passage Retrieval." ACL 2024. <https://aclanthology.org/2024.acl-long.236/>
11. "Structured Summaries for Retrieval-Augmented Generation in Portuguese-Language Consumer Complaints." PROPOR 2026. <https://aclanthology.org/2026.propor-1.84/>
12. "Summary RAG: A Multi-Format Document Retrieval System with Document-Level Summarization." Research and Science Today (2026). <https://www.rstjournal.com/article/21.2026>
13. "MC-indexing: Effective Long Document Retrieval via Multi-view Content-aware Indexing." EMNLP 2024 Findings. <https://doi.org/10.18653/v1/2024.findings-emnlp.150>
14. "One Word Is Not Enough: Simple Prompts Improve Word Embeddings." arXiv:2512.06744. <https://arxiv.org/html/2512.06744>
15. Decorte et al. (2021). "JobBERT: Understanding Job Titles through Skills." arXiv:2109.09605. <https://arxiv.org/pdf/2109.09605>
16. "Document-based Recommender System for Job Postings using Dense Representations." NAACL 2018. <https://doi.org/10.18653/v1/n18-3027>
17. "Impact of Document Structure on Vector Retrieval Performance" (prose vs Q&A vs fact-dense lists vs JSON-LD; 250 trials, all-MiniLM-L6-v2). IJSRSET. <https://ijsrset.com/home/article/download/IJSRSET261313/IJSRSET261313/1725>
18. "A Practitioner's Guide to Systematic Prompt Evaluation for Information Extraction." EMNLP 2026 Industry. <https://aclanthology.org/2026.emnlp-industry.59.pdf>
19. "Story Embeddings — Narrative-Focused Representations of Fictional Stories." EMNLP 2024. <https://aclanthology.org/2024.emnlp-main.339.pdf>
20. Gao et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE). arXiv:2212.10496. <https://aclanthology.org/2023.acl-long.99/>
21. Wang et al. (2023). "Query2doc: Query Expansion with Large Language Models." EMNLP 2023. <https://aclanthology.org/2023.emnlp-main.585/>
22. Nogueira et al. (2019). "Document Expansion by Query Prediction" (Doc2query). arXiv:1904.08375. <https://arxiv.org/html/1904.08375v2>
23. "Doc2Query++: Topic-Coverage based Document Expansion and its Application to Dense Retrieval via Dual-Index Fusion." arXiv:2510.09557. <https://arxiv.org/html/2510.09557v2>
24. This repository. `docs/research-reports/001-embedding-optimization-research.md`, §1 "Categories of Optimization" (measure baseline, change one variable, re-measure, document delta).
