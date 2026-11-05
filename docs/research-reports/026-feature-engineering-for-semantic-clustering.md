# Feature Engineering for Semantic Clustering — Enrichment, Compression, and View Fusion Before Embedding

> Date: 2026-11-02
> Context: Semantic Career Mapping Platform — research into how text should be transformed *before* it is embedded, so that the resulting vectors are better inputs to semantic clustering. The production pipeline embeds LLM-cleaned job postings with normalized **Amazon Titan Text Embeddings V2** (1,024-d, 8,192-token input bound) and clusters them with EVōC ([technical/embedding-pipeline.md](../technical/embedding-pipeline.md), [Experiment 012](../experiments/012-evoc-clustering-bakeoff.md)). Feature engineering is the next layer of deliverable 3 in [project.md](../project.md).
> Scope: pre-clustering feature construction — denoising, field extraction, keyphrase extraction, summarization, structured tagging, section weighting, expansion, capability profiles, decomposition, late chunking, and multi-view fusion. Model selection is covered by [Report 024](024-embedding-models-for-semantic-clustering.md), cluster algorithms and metrics by [Report 025](025-semantic-clustering-methods.md), and rewrite *format* by [Report 023](023-context-engineering-for-embeddings.md). This report covers the operators and the evidence for using them. A second independent research pass (2026-11-02) supplied the concept-enriched multi-view framing; its additions are integrated in [§2.4](#24-reduction-vs-enrichment-vs-decomposition) and [§5.4](#54-what-the-independent-research-pass-adds).

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What Counts as Feature Engineering Here](#2-what-counts-as-feature-engineering-here)
3. [Why Extraction Helps or Hurts](#3-why-extraction-helps-or-hurts)
4. [Operator Catalogue](#4-operator-catalogue)
5. [Evidence Review](#5-evidence-review)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Fit Against This Project](#7-fit-against-this-project)
8. [Proposed Experiment 016 — Feature-Engineering Ladder](#8-proposed-experiment-016-feature-engineering-ladder)
9. [Risks and Threats to Validity](#9-risks-and-threats-to-validity)
10. [Sources](#10-sources)

---

## 1. Executive Summary

Feature engineering for semantic clustering is not "make the text shorter". It is the deliberate choice of **which facts reach the encoder, how much each fact contributes, and in what linguistic form**. Those choices change the similarity relation itself, so they must be specified against the clustering target — role function, skill family, seniority, or domain — not against a generic notion of "cleaner text".

Six conclusions drive this report:

1. **The best-supported operator is targeted extraction, not blanket compression.** Removing boilerplate and keeping the discriminative fields is already the project's strongest measured intervention: LLM cleaning lifted the separation gap from `+0.0060` to `+0.0493` ([Experiment 003](../experiments/003-llm-extraction.md)). Summaries and keyword-only inputs discard information; extraction and field construction preserve it.
2. **Semantic enrichment is more promising than content reduction.** A grounded **capability profile** — role context, responsibilities, methods, technologies, domain, and seniority, rendered as natural-language prose — preserves the original text and adds a controlled, target-relevant view. This is the strongest recommendation of the independent research pass and the closest pre-embedding analogue to Anthropic's contextual retrieval [25]. Retrieval evidence does not yet prove a clustering gain, so the profile must be tested as an enrichment view, not assumed.
3. **Keyphrases are most useful as an *additional* view, not as the only input.** In a 2024 TACL study, LLM-generated keyphrases encoded and concatenated to each document's embedding improved cluster quality on every metric for all five datasets tested; the keyphrase-expansion arm was the strongest of the LLM interventions [1]. That is evidence for **augmentation**. Keyword-only representations remain weaker than prose because they lose relations and are out of the encoder's training distribution [2].
4. **Summarization is conditional and often loses the signal clustering needs.** A five-dataset study found that adding an abstractive-summary step before embedding did **not** consistently improve clustering; full texts generally scored higher, suggesting that details needed for the partition were lost [4]. Summaries help mainly when the source is noisy or redundant [5]. For fact-dense job postings, a summary is more defensible as a *cluster description* than as the primary vector.
5. **Structured tags are powerful and dangerous.** Canonical skills, role families, and seniority can make the representation more consistent, but injecting labels derived from the same text — then evaluating against those labels — inflates external metrics and can collapse clusters onto the injected schema. Tags must be treated as features only when their errors and independence from the evaluation labels are measured.
6. **The practical ladder for this project is: cleaned prose → role-aware cleaning → field-aware prose → grounded capability profile → auxiliary keyphrase/skill views → optional summary → multi-view fusion.** Do not start from keyword-only, profile-only, or summary-only. Keep the current cleaned-text baseline as the control, and require a pre-registered downstream improvement before adding a layer.

| Default recommendation | Operator | Why |
|---|---|---|
| Keep as baseline | LLM-cleaned prose + Titan V2 | Already measured; full-text coverage; cheapest defensible control |
| First cleaning change | Role-aware boilerplate removal: drop pay/location/legal from embedding text, keep role-relevant company/product/team context | Preserves discriminative context without shared company marketing |
| First feature layer | Field-aware natural-language prose (title, responsibilities, skills, seniority) | Restores structure without JSON syntax; preserves relations [2] |
| First enrichment | Grounded capability profile (role context, responsibilities, methods, technologies, domain, seniority) | Adds structured signal without replacing the source; strongest independent-pass recommendation |
| First augmentation | LLM keyphrases appended as a natural-language sentence | Strongest published pre-embedding augmentation result [1]; cheap at this corpus size |
| Auxiliary only | Keyword list or capability profile as a separate view | Complementary signal; weak as the sole dense input [2] |
| Conditional | Abstractive or schema-guided summary | Use for noisy sources or cluster naming; not a default for fact-dense postings [4][5] |
| Later | ESCO-canonical skill view | Consistency and multilingual labels, but mapping errors and taxonomy mismatch must be measured [6][7] |
| Evaluate, do not assume | Section weighting, decomposition, and multi-view fusion | Plausible and popular, but target-dependent and under-tested on job postings |

**The central warning:** feature engineering can improve a clustering metric while making the representation less faithful. A summary that drops a rare framework, a profile that invents a capability, or a tagger that assigns the majority role family can produce tidier clusters for the wrong reason. Every operator needs both a downstream clustering check and an upstream fidelity check.

---

## 2. What Counts as Feature Engineering Here

### 2.1 Three insertion points

Features can enter a clustering pipeline at three different places, and they are not interchangeable:

| Insertion point | What is produced | Examples | Risk |
|---|---|---|---|
| **Before embedding** | A new text or a structured representation that is encoded | denoised prose, field-labelled prose, keyphrase-augmented text, summary, canonical skill list | Permanent information loss or hallucination; every downstream vector changes |
| **During clustering** | Constraints, weights, or a second view combined in the distance | must-link/cannot-link, multi-view fusion, weighted sections | Harder to evaluate; clusterer-specific |
| **After clustering** | Labels, names, or descriptions of the discovered groups | c-TF-IDF topic words, KeyBERT cluster keywords, LLM cluster summaries | Does not change the partition; can create circular evaluation if reused as input |

Most confusion in the literature comes from mixing the first and third rows. **c-TF-IDF in BERTopic is a post-clustering representation technique; it does not alter the document embeddings that were clustered** [8]. KeyBERT ranks candidate phrases against a document embedding [9]; if those phrases are then embedded with the same encoder and used as the only document representation, the "feature" is largely a projection of the same geometry. Neither technique is automatically a pre-embedding feature-engineering win.

### 2.2 Two uses of keywords and summaries

| Use | Question it answers | Appropriate evidence |
|---|---|---|
| **Input feature** | Does adding/keeping this text change the partition? | Downstream ARI/AMI, k-NN purity, stability, audit |
| **Output description** | Can a human understand and name this cluster? | Keyword precision, summary faithfulness, human/LLM agreement |

A keyword extractor can be excellent at describing a cluster and useless as an input feature, or vice versa. The project should keep two separate scorecards. Report 025's four-lens protocol — internal, external, stability, task/human — applies to the partition; upstream extraction quality is a fifth, separate lens [10].

### 2.3 The similarity target is the specification

The same posting supports several valid clusterings:

| Target | Discriminative content | Feature implication |
|---|---|---|
| **Role function** (the project's working target) | responsibilities, scope, team context, methods | Weight responsibilities and role narrative; title is a weak proxy |
| **Skill family** | tools, frameworks, languages, certifications | Keyphrases and canonical skill tags matter more; prose can dilute them |
| **Seniority** | years, scope, leadership, ownership language | Preserve qualification and scope sentences; summaries often erase them |
| **Domain** | product area, industry, data modality | Preserve domain nouns and use cases; boilerplate removal is safe |

This is why a single "best" preprocessing recipe does not exist. Report 001's industry-derived recommendation to repeat the title 3× and skills 2× is aimed at **retrieval by title/skill**, not necessarily at the project's hypothesis that postings are better grouped by responsibilities [11]. If role function is the target, up-weighting the title can actively pull the clustering back toward title-based groups. Section weights must be derived from the target, then tested.

### 2.4 Reduction vs Enrichment vs Decomposition

The independent research pass frames the design space as five families. The framing is useful because it separates operators that are often conflated under "preprocessing":

| Family | What it does | Examples | Project verdict |
|---|---|---|---|
| **Content reduction** | Removes text or replaces it with a shorter version | boilerplate cleaning, extraction, summarization, keyword-only representation | Cleaning is proven; summary and keyword-only are conditional |
| **Semantic enrichment** | Keeps the source and adds structured context | capability profile, contextual preamble, metadata injection, entity/task/domain tags | **Most promising next layer**; test append vs separate view |
| **Content decomposition** | Splits a document into coherent units and embeds them separately or pools them | section-level embeddings, sentence-level views, late chunking | Useful for weighting; Titan V2 already covers full postings, so less urgent |
| **Multi-view representation** | Embeds different aspects independently and fuses them | original + profile + keyphrases + skills | **Strongest architecture**, but fusion weights need measurement |
| **Representation refinement** | Removes redundancy, weights features, or combines vectors | section weights, mean centering, vector fusion, PCA | Target-dependent; do after the views are understood |

The key distinction is **reduction vs enrichment**:

- Reduction assumes the source contains too much and deletes it. This is the cleaning and summarization family.
- Enrichment assumes the source is necessary but ambiguous, and adds a compact description that makes the target explicit. This is the capability-profile family.

Anthropic's Contextual Retrieval is the clearest precedent for enrichment: it prepends a short, chunk-specific explanation before embedding, rather than replacing the chunk. The reported gains came from **chunk-specific context**; simply prepending a generic document summary produced limited gains [25]. That is a direct warning for job postings: a generic summary of the whole posting is not the same as a profile that states the role context, responsibilities, methods, and domain.

Two further distinctions matter:

1. **Decomposition is not compression.** Splitting a posting into responsibilities, requirements, and skills preserves all the text; it changes the unit of representation. Late chunking goes further by pooling token representations from a long-context encoder, preserving surrounding context while producing chunk vectors [26]. Titan V2 is an API that returns one vector per input, so late chunking is not directly available in the current pipeline.
2. **Summary-based centroids are a cluster-algorithm change.** Using an LLM summary as the centroid of each cluster (`k-LLMmeans`) re-embeds cluster summaries during clustering [27]. That is a post-clustering representation loop, not a pre-embedding feature. It belongs in a separate experiment after the feature ladder.

The practical rule: **prefer enrichment over reduction when the source is fact-dense.** Job postings are fact-dense. Cleaning should remove non-job boilerplate, but the next layer should add a capability view rather than replace the posting with a summary.

---

## 3. Why Extraction Helps or Hurts

### 3.1 Mechanisms of benefit

| Mechanism | What it does | Example |
|---|---|---|
| **Denoising** | Removes text unrelated to the target | EEO statements, benefits, company history, application instructions |
| **Discriminative emphasis** | Keeps or repeats fields that separate groups | Responsibilities and must-have skills for role function |
| **Canonicalization** | Maps synonyms and spelling variants to one surface form | "PyTorch", "pytorch", "torch" → one skill label |
| **Prior injection** | Adds knowledge not explicit in the posting | ESCO skill labels, role-family tags, related-skill expansion |
| **Length control** | Prevents dilution and truncation | Compression to a token budget; chunking and pooling |
| **Distribution matching** | Makes the input look more like encoder training data | Field-labelled natural language instead of JSON or bare lists |
| **Complementary views** | Captures a relation the primary view misses | Dense prose + sparse keyword view |

The strongest mechanism for this project is denoising plus discriminative emphasis. Both are already partly implemented by the cleaning stage; the remaining opportunity is the **structure** of what remains.

### 3.2 Mechanisms of harm

| Mechanism | What it does | Evidence or example |
|---|---|---|
| **Information loss** | Drops the detail that separates near-duplicate roles | Summaries of job postings can omit a framework, seniority signal, or domain [4] |
| **Relation loss** | Removes grammar that encodes scope and negation | "Python required" vs "Python preferred"; "led a team using Python" vs "used Python" [2] |
| **Out-of-distribution format** | Produces text unlike the encoder's training data | Bare keyword lists and JSON are weaker than prose [2][3] |
| **Extractor bias** | Repeats the extractor's notion of importance | Frequency-based keyword methods favor common terms; embedding-based methods inherit hubness [9][12] |
| **Schema collapse** | Injected labels dominate the vector | A coarse role tag can make all postings in a family look identical |
| **Compounding errors** | Cleaning + extraction + tagging + summarization each lose or invent facts | Errors accumulate before the encoder sees the text |
| **Evaluation circularity** | Features derived from labels are scored against those labels | Inflated external metrics with no real improvement |

### 3.3 The information budget

Every pre-embedding operator spends part of a fixed budget:

```
raw posting
  → cleaned text            (removes boilerplate; loses some context)
  → extracted fields        (selects target-relevant facts; loses the rest)
  → keyphrases              (compresses further; loses relations)
  → summary                 (compresses further; loses details and may hallucinate)
  → embedding               (fixed-dimensional lossy projection)
```

Two rules follow:

- **Do not compress twice without measuring.** Cleaning plus summarization may remove more than either alone, and the second step can undo the first step's gains.
- **Match the budget to the target.** A role-function view can tolerate more compression than a skill-retrieval view. One representation cannot serve both optimally; that is the argument for multiple views, not for one aggressive compression.

Titan V2's 8,192-token bound removes the old MiniLM truncation defect, but it does not remove **length dilution**. Longer inputs still spread a fixed-dimensional vector across more content, and generic or repetitive text can dominate the direction of the embedding. Coverage and dilution are separate problems.

---

## 4. Operator Catalogue

### 4.1 Denoising and selection

**What it is:** Remove text that is irrelevant to the clustering target. In this project, the production cleaning stage already does this with an LLM prompt that strips EEO, benefits, company boilerplate, and application instructions ([technical/data-processing.md](../technical/data-processing.md), [Experiment 008](../experiments/008-boilerplate-only-cleaning.md)).

**Evidence:** The measured separation-gap improvement from cleaning is the largest in the project: `+0.0060` raw → `+0.0203` regex → `+0.0493` LLM [13]. The corrected boilerplate-only prompt also reduced over-deletion from 22.8% to 2.3–5.2% on one model [14].

**Verdict:** Adopt; already in production. The remaining question is not whether to denoise but whether the cleaning prompt preserves the fields the target needs. A cleaning prompt optimized for "remove boilerplate" can still delete responsibilities or qualifications that look like boilerplate.

**Role-aware boundary:** The current prompt preserves role/team context but strips generic company marketing. A role-aware variant should keep company **product, domain, scale, and business context** when it explains the role, and strip generic mission/values/awards/culture copy. Pay, location, and legal text should leave the embedding text but remain in structured metadata for filtering and analysis; the raw posting is already retained in the lake.

**Failure mode:** Over-deletion. The project has already seen that aggressive LLM extraction can remove target content; the boilerplate-only prompt was the fix. Any new feature layer must not reintroduce aggressive deletion under a different name. The opposite failure also exists: keeping company marketing re-introduces shared boilerplate and can make unrelated roles look similar. Experiment 008 showed that aggressive but consistent cleaning can raise the separation gap while losing role context, so fidelity and clustering must be read jointly [14].

### 4.2 Field construction and section weighting

**What it is:** Split the cleaned posting into semantically meaningful fields — title, role summary, responsibilities, must-have skills, nice-to-have skills, seniority, domain — and build a representation from them.

Three implementation patterns:

| Pattern | How | Strength | Weakness |
|---|---|---|---|
| **Field-labelled prose** | Concatenate short natural-language lines: "Title: ... Role: ... Responsibilities: ... Skills: ..." | In-distribution, preserves relations, controllable | Longer than a keyword list; field order matters |
| **Weighted concatenation** | Repeat high-value fields 2–3× before embedding | Simple; no extra inference | Repeats can exceed the token budget; weights are arbitrary without testing |
| **Separate field embeddings + weighted average** | Embed each field; combine with weights | Fine-grained control; can inspect field contributions | Multiple inference calls; vector arithmetic can blur semantics; weights need tuning |

**Evidence:** Report 023's analysis of format and structure concludes that short field-aware natural-language prose is the safest primary representation, ahead of keyword lists, narratives, and JSON [2]. Report 001 summarizes industry practice on section weighting for structured documents [11]. There is little peer-reviewed evidence specifically for job-posting clustering, so this is a **well-motivated hypothesis**, not a settled result.

**Project-specific twist:** The title is a strong but shallow signal. It is useful for role-family retrieval, but the project's hypothesis is that responsibilities group jobs better than titles (see [Report 003](003-ml-ai-responsibility-taxonomy.md)). A field-aware representation should therefore include the title but not let it dominate. Test at least two weightings: title-forward (retrieval-oriented) and responsibility-forward (role-function-oriented).

**Verdict:** Test first. It is the lowest-risk feature layer because it preserves prose and can be built deterministically from the existing cleaned text.

### 4.3 Keyword and keyphrase extraction

**What it is:** Select a small set of terms or phrases that represent the document. The methods differ in what "important" means:

| Family | Methods | Signal | Strength | Weakness |
|---|---|---|---|---|
| **Statistical** | TF-IDF, YAKE [15] | Frequency, position, co-occurrence, casing | Fast, unsupervised, corpus-independent (YAKE) | Favors frequent terms; misses semantic importance; no relations |
| **Graph** | TextRank [16], SingleRank | Co-occurrence graph centrality | Captures term associations | Computationally heavier; still lexical |
| **Embedding-based** | KeyBERT [9], EmbedRank [12] | Cosine similarity between candidate phrase and document embedding | Semantic relevance; no training | Inherits the embedding's biases; candidate generation matters |
| **LLM-based** | Prompted keyphrase generation | Instruction-following; task context | Can target the clustering question; produces phrases not just words | Cost, latency, hallucination, prompt sensitivity [3]; non-determinism unless temperature 0 and cached |

**Two modes of use:**

1. **Keyword-only representation.** Embed the keyphrase list. This is compact and interpretable, but it loses grammar, negation, seniority, and scope. Report 023 finds keyword-only weaker than prose for semantic similarity, with complementary recall for exact terms [2]. Use it as an auxiliary view or sparse index, not as the primary dense representation.
2. **Keyword-augmented representation.** Keep the prose and append or fuse a keyphrase view. In the TACL 2024 study, the authors generated keyphrases with an LLM, encoded them with the same encoder, and concatenated the keyphrase vector to the document vector; this improved cluster quality on every metric for all five datasets, and the keyphrase-expansion arm was the strongest in their ablation [1]. The study covers short texts (intents, entities, tweets) and a semi-supervised setting, so it is evidence for the mechanism, not a guarantee for long job postings.

**Why augmentation can work when keyword-only fails:** the prose preserves relations and distribution match; the keyphrase vector adds a compact, target-relevant signal. The two are complementary. Concatenation is not the only fusion option; weighted averaging and late fusion are alternatives, but they require their own evaluation.

**Post-clustering keyword extraction is different.** c-TF-IDF in BERTopic computes class-based term frequencies **after** clustering to describe each topic [8]. It is an excellent cluster-naming tool and a poor substitute for a pre-embedding feature. Similarly, running KeyBERT on each document after clustering does not change the partition.

**Verdict:** Test LLM keyphrase augmentation as an auxiliary view after the capability profile. Use a cheap statistical or embedding baseline (YAKE/KeyBERT) to separate "keyphrases help" from "LLM world knowledge helps". Do not replace prose with keywords.

### 4.4 Contextual enrichment and capability profiles

**What it is:** Keep the cleaned posting and add a short, document-specific semantic description that makes the target explicit. The independent pass calls this **metadata enrichment**; the practical form for this project is a **grounded capability profile**.

A profile should cover:

| Field | Example |
|---|---|
| **Role context** | "Senior ML engineer in a research-heavy computer-vision team." |
| **Core responsibilities** | "Builds training pipelines, optimizes inference, owns model quality." |
| **Methods** | "Distributed training, quantization, experiment tracking." |
| **Technologies** | "Python, PyTorch, CUDA, AWS." |
| **Domain** | "Infrared sensor data and motion analysis." |
| **Scope/seniority** | "Owns a workstream; mentors two engineers." |
| **Relationships** | "PyTorch used for training; CUDA used for inference optimization." |

Three rules make the profile different from a summary:

1. **Ground every line in the source.** Require evidence spans or a verbatim phrase for each extracted capability. If the profile cannot cite the posting, it is a hallucination risk.
2. **Preserve relationships.** "Python used for image processing" is more informative than "Python; image processing". Short sentences or labelled clauses keep the relation that a bare list loses.
3. **Do not replace the source.** The profile is an additional view. Keep the cleaned text as the primary representation until the enrichment view proves itself.

**Evidence:** Anthropic's Contextual Retrieval reports that chunk-specific contextual preambles improve retrieval, while generic document summaries produce limited gains [25]. That is evidence for **contextual, document-specific enrichment**, not for generic summarization. The TACL keyphrase-expansion result is a related mechanism: an LLM-generated compact view, encoded alongside the document, improved clustering on five datasets [1]. The independent pass also cites metadata-enriched retrieval work in the same direction, but retrieval gains do not automatically transfer to clustering; the clustering target and the noise structure are different.

**Job-posting application:** The profile is most useful when the posting is long, uses internal jargon, or describes responsibilities indirectly. It is least useful when the posting is already a clean, explicit list of duties and skills. Unlike chunk-level contextual retrieval, a posting is already a whole document for Titan V2, so the profile is not recovering context lost by chunking; it is **canonicalizing and foregrounding** the capabilities that the target cares about.

**Risks:** extraction errors, hallucinated capabilities, schema bias, and redundancy. If the profile merely repeats the posting, it adds no new information and only reweights the same concepts. If it is appended to the posting, it also increases length and can dilute the vector. Test append, separate-view, and profile-only forms.

**Verdict:** Highest-value next experiment. Build it as a grounded capability profile, not a free-form summary. Test it first as a separate view and as an appended block; do not make it the sole representation.

### 4.5 Summarization

**What it is:** Produce a shorter text that preserves the document's main points. The variants matter:

| Variant | Method | What survives | What is lost |
|---|---|---|---|
| **Extractive** | Select sentences (TextRank, LexRank, clustering-based) | Original wording and facts | Coverage; redundancy; may select generic sentences |
| **Abstractive** | Generate new sentences (LLM) | Compression; fluency; schema adherence | Exact terms, rare skills, negation; may hallucinate |
| **Schema-guided** | LLM fills a target schema, then renders prose | Target-relevant fields; controllable | Facts outside the schema; schema bias |
| **Hierarchical** | Recursively summarize chunks into a tree (e.g. RAPTOR) | Multi-level context; long-document coverage | Detail at lower levels; summary drift across levels |

**Do not confuse a summary with a capability profile.** A summary asks "what is the gist?" and compresses. A capability profile asks "which target-relevant capabilities are present, and how do they relate?" and selects. The profile can be longer than a summary while carrying more discriminative signal, because it is structured against the clustering target rather than optimized for fluency.

**Evidence:**

- The five-dataset study in [4] added summarization after preprocessing and before embedding. Clustering with summaries did **not** consistently improve over full texts; in most comparisons the full-text clustering scored higher, and the authors attribute this to loss of essential details. Smaller summarization models produced especially poor results.
- Report 023 finds summaries help when the source is noisy and redundant (the Portuguese consumer-complaint case) but hurt when the source is fact-dense (SciFact) [2][5]. The mechanism is consistent: summaries are denoisers, not feature extractors. If the cleaning stage has already removed noise, a summary has little left to remove and a lot of detail to lose.
- RAPTOR's recursive summary tree improves **retrieval** by exposing multiple abstraction levels [17]. That is not evidence that summary embeddings improve **clustering**; the tasks reward different things. Retrieval needs a relevant passage somewhere in the index; clustering needs every document's vector to land in the right neighbourhood.

**Where summaries are still useful:**

1. **Cluster naming and audit.** A short, grounded summary of a cluster's representative postings is easier for a human to review than raw text.
2. **Noisy sources.** If a posting contains long application instructions or legal text that cleaning missed, a schema-guided summary can remove it.
3. **Role-function view.** A short "what this person does" narrative may capture responsibility similarity better than a keyword list, but Report 023 treats this as unproven and warns about hallucination [2].

**Verdict:** Do not make summarization the primary representation by default. Test it as an optional view, with a factuality/omission audit. Prefer schema-guided summaries over free-form summaries because the schema makes omissions measurable.

### 4.6 Structured extraction and taxonomy tagging

**What it is:** Extract structured attributes — skills, seniority, role family, domain, work mode — and optionally map skills to a taxonomy such as ESCO [6]. The attributes can be used as:

- **Text features:** render them as field-labelled prose ("Seniority: senior. Skills: Python, PyTorch, AWS.").
- **Categorical features:** append one-hot or embedding features to the document vector.
- **Constraints or filters:** restrict clustering within a role family or seniority band.
- **Post-hoc labels:** describe and audit clusters without changing the partition.

**Evidence:** Skill extraction from job postings is an active field with established benchmarks such as SkillSpan and taxonomy mapping to ESCO [7]. ESCO provides canonical labels, hierarchical relations, and multilingual coverage, which can reduce synonym noise [6]. But mapping is error-prone: surface forms are ambiguous, granularity is coarse relative to postings, and taxonomy coverage changes over time. A tagger that maps "LLM" and "large language models" to different concepts adds noise rather than removing it.

**The leakage trap:** If the same tags are used as input features and as the external evaluation labels, ARI/AMI will rise even if the clustering has not discovered anything new. The features and the labels must be independent, or the evaluation must use a different source (hand audit, held-out labels, or a title-only sensitivity analysis). Experiment 012 already flags the related circularity between LLM labels and embeddings [18].

**Verdict:** Use tags first for **cluster audit and naming**, then test a canonical-skill view as an auxiliary representation. Do not inject role-family labels into the primary vector until the tagger's precision and the evaluation independence are established.

### 4.7 Expansion and canonicalization

**What it is:** Add terms or documents that increase lexical coverage:

- **Synonym/acronym expansion:** add "large language model" to "LLM", "PyTorch" to "torch".
- **Taxonomy expansion:** add ESCO preferred labels and related skills.
- **Document expansion:** generate likely queries or related passages and append them (Doc2query) [19].

**Evidence:** Document expansion improves sparse and dense retrieval when the added text is relevant and can be isolated; naively appending expansions to a dense input can add noise [2][19]. For clustering, expansion can help rare-skill recall but can also pull unrelated postings together through generic terms.

**Verdict:** Keep expansion as a separate view or a weighted component. Do not concatenate a large expansion block into the primary input without measuring its effect on cluster purity.

### 4.8 Decomposition, chunking, pooling, and fusion

**What it is:** Represent a long document as multiple chunks and combine them:

| Method | How | Best for |
|---|---|---|
| **Mean pooling** | Average chunk vectors, then L2-normalize | Simple full-text coverage; used in Experiment 011 arms B/B2 |
| **Weighted pooling** | Weight chunks by section, position, or salience | Emphasizing responsibilities or skills |
| **Max pooling** | Take the dimension-wise maximum | Preserving salient features; less stable |
| **Multi-view concatenation** | Concatenate normalized vectors from different representations | Combining prose + keyphrase + skill views |
| **Late fusion** | Cluster or retrieve per view, then combine scores/co-associations | Retrieval and ensemble clustering; harder for a single partition |

**Decomposition granularity:** A single vector forces every section of a posting into one representation. Alternatives range from coarse to fine:

| Granularity | Job-posting unit | Trade-off |
|---|---|---|
| **Document** | whole posting | Maximum context; maximum dilution |
| **Section** | responsibilities, requirements, skills, team context | Preserves local coherence; loses cross-section relations unless pooled |
| **Paragraph** | one project or responsibility block | More specific; more vectors and more fusion decisions |
| **Sentence** | one duty or requirement | High specificity; loses document-level context |
| **Concept** | skill, method, task | Matches keyphrase/profile views; loses relations |

For this project, **section decomposition is mainly a weighting device**, not a coverage fix. Titan V2 already sees the whole posting; splitting into sections lets the pipeline emphasize responsibilities over company copy or benefits residue. That is different from late chunking.

**Late chunking:** Günther et al. (2024) process the whole document through a long-context transformer, then pool token representations into chunk embeddings, so each chunk retains surrounding context [26]. This is attractive for long documents, but it requires access to token-level hidden states. Titan V2 is a managed embedding endpoint that returns one vector per input; late chunking is not available without a local long-context encoder. Defer it until the encoder stack changes.

**Project context:** Experiment 011 compared truncated MiniLM, full-text pooled MiniLM, overlapping-window pooled MiniLM, and full-text Titan V2. Titan V2 (arm C) covers the full posting and was selected for production; the coverage defect is fixed [18][20]. Chunk pooling is therefore no longer needed for **coverage** on this corpus, but it remains relevant for **weighting** and for any future encoder with a smaller context.

**Fusion caution:** Concatenating a 1,024-d prose vector with a 1,024-d keyphrase or profile vector doubles dimensionality and lets the block with larger variance dominate cosine distance. Normalize each block and choose weights deliberately, or evaluate late fusion. Do not assume that more views always produce better clusters. Weighted vector concatenation is the simplest fusion compatible with EVōC; pairwise late fusion needs a custom distance or a clusterer that accepts precomputed distances.

### 4.9 Summary-based centroids (post-clustering)

A 2026 ICLR paper proposes using summaries as cluster centroids: `k-NLPmeans` uses lightweight deterministic summaries, and `k-LLMmeans` uses LLM-generated summaries, re-embedding the summary to update the centroid instead of averaging document vectors [27]. The reported gains are over classical k-means baselines.

This is **not** a pre-embedding feature operator. It changes the clustering loop:

```
documents → embeddings → clustering → cluster summary → re-embedded centroid → repeat
```

It is interesting for interpretability and for cluster prototypes that read like job-family descriptions, but it changes the algorithm and the evaluation. It should be a separate experiment after the feature ladder, not a step in the 016 pipeline. The same distinction applies to post-clustering keyword extraction: describing clusters is not the same as improving their geometry.

---

## 5. Evidence Review

### 5.1 What is well supported

| Finding | Evidence | Implication |
|---|---|---|
| Cleaning/denoising improves separation | Project experiments 003/008 [13][14] | Keep the cleaning layer; do not undo it with aggressive compression |
| Contextual enrichment can improve retrieval | Anthropic Contextual Retrieval [25] | Chunk-specific context helps; generic summaries do not substitute |
| LLM keyphrase augmentation can improve clustering | TACL 2024, five datasets [1] | Test as an additional view, especially for short or noisy texts |
| Keyword-only loses relations and is out of distribution | Report 023 synthesis [2] | Auxiliary view, not primary |
| Summaries do not consistently improve clustering | Five-dataset study [4] | Do not default to summary-only |
| Summaries help noisy/redundant sources | Report 023 / PROPOR case [2][5] | Conditional use; audit factuality |
| Field-aware prose is safer than JSON or bare lists | Report 023 [2] | Use natural-language field labels |
| Post-clustering keywords describe but do not change clusters | BERTopic c-TF-IDF [8] | Separate naming from feature engineering |
| Taxonomy mapping is useful but error-prone | ESCO report and skill-extraction survey [6][7] | Measure mapping precision before using tags as features |
| Decomposition and late chunking can preserve local context | Late Chunking [26] | Useful for long documents; not directly available through the Titan API |
| Summary centroids can change the cluster prototype | ICLR 2026 summary-centroid paper [27] | Algorithm change, not pre-embedding; separate experiment |

### 5.2 What is not settled

- **Whether keyphrase augmentation transfers from short texts to long job postings.** The TACL study's documents are short; job postings are long and multi-section. The mechanism is plausible, but the effect size may be smaller or negative if keyphrases over-represent generic tech terms.
- **Whether a capability profile adds clustering signal beyond the cleaned text.** Contextual retrieval evidence is retrieval-based; metadata-enriched retrieval evidence is also retrieval-based. A profile can improve the neighbourhood structure, reweight concepts already present, or do nothing. Only a clustering comparison can tell.
- **Whether append or separate-view fusion is better.** Appending is cheap and keeps one vector; separate views preserve each representation's geometry but require normalization, weights, and a fusion rule.
- **Whether section weighting helps role-function clustering.** The industry evidence is retrieval-oriented. The project's target is responsibility similarity, which may favor different weights.
- **Whether summaries help role-function clustering.** A role-function narrative is a summary variant; it may capture "what the person does" better than raw text, but it can also invent scope and seniority. Report 023 treats this as an open hypothesis [2].
- **Whether ESCO tags add signal beyond the text.** Canonicalization may reduce synonym noise, but coarse taxonomy labels can erase distinctions the clustering needs.
- **Whether decomposition adds value when the encoder already sees the whole posting.** Section views may help weighting without improving coverage; the benefit may be small on Titan V2.
- **How operators interact.** Cleaning + profile, or fields + summary, can be complementary or redundant. Only a factorial comparison can tell.

### 5.3 Why extraction quality is not enough

A keyphrase extractor can score well on F1 against human keyphrases and still hurt clustering, because the clustering target may depend on different terms. Conversely, a "mediocre" extractor may surface exactly the rare skill that separates two job families. This is the same lesson as Report 024's warning that MTEB's clustering task is not the project's clustering task [21]. Evaluate the operator on the downstream partition, not only on its own benchmark.

### 5.4 What the independent research pass adds

The independent pass reaches a compatible overall verdict — **do not default to summarization; investigate a concept-enriched, multi-view pipeline** — but it sharpens four points:

1. **Name the families correctly.** Content reduction, semantic enrichment, content decomposition, multi-view representation, and representation refinement are different interventions. Calling all of them "feature engineering" hides the fact that reduction deletes information while enrichment adds context.
2. **Prefer enrichment over reduction for fact-dense documents.** A capability profile that states role context, responsibilities, methods, technologies, and domain is a better fit for job postings than a generic summary. This is the report's main change after integrating the independent pass.
3. **Use contextual retrieval as the precedent, not summarization.** Anthropic's gains came from chunk-specific context, while generic summaries gave limited gains [25]. For this project, the analogue is a posting-specific capability profile, not a corpus-level or document-level gist.
4. **Separate post-clustering representation from pre-embedding features.** Summary centroids [27] and BERTopic topic words [8] describe or update clusters; they do not belong in the same ablation as cleaning, profiles, and keyphrases.

The independent pass's recommended baseline is preserved here: keep the original document, embed the enriched representation independently, and experiment with weighted fusion. Its less directly applicable suggestions are filed as later work: late chunking requires token-level long-context access that Titan V2 does not expose [26], and summary-as-centroid changes the clustering algorithm rather than the input [27].

---

## 6. Evaluation Protocol

### 6.1 Upstream fidelity

Before asking whether clusters improved, measure what the operator did to the text:

| Metric | What it catches |
|---|---|
| **Field coverage** | Did extraction retain title, responsibilities, skills, seniority? |
| **Keyphrase precision@k** | Are the extracted phrases actually in the document and relevant? |
| **Profile claim precision** | Is every capability in the profile supported by the source? |
| **Profile omission rate** | Which target-relevant capabilities were left out? |
| **Summary factuality** | Does the summary contain only claims supported by the source? |
| **Omission rate** | Which target-relevant facts were dropped? |
| **Compression ratio / token count** | How much information was removed, and did it fit the encoder budget? |
| **Determinism** | Same input + same prompt + temperature 0 → same output? |
| **Cost/latency** | Is the operator affordable at corpus scale? |

### 6.2 Downstream clustering

Use the four-lens protocol from Report 025 [10]:

| Lens | Metrics | Role |
|---|---|---|
| **External** | ARI, AMI, V-measure against independent function tags | Primary, when labels are trustworthy and independent |
| **Internal** | k-NN purity, separation gap, EVōC persistence | Label-free checks; comparative only |
| **Stability** | Seed and subsample ARI; cluster-count stability | Detect brittle feature layers |
| **Task/human** | Representative-document audit; LLM-judge agreement; cluster naming | Detect coherent-but-wrong groups |

Add retrieval metrics when the representation is also used for search: Recall@10, MRR, NDCG on the golden set. A feature layer that improves clustering but destroys retrieval is not a universal win; the project needs both.

**Do not make Silhouette or Davies–Bouldin the primary endpoints.** The independent pass lists them first, which is reasonable for low-dimensional tabular clusters but weak for 1,024-d text embeddings. Centroid-based indices assume convex, roughly isotropic clusters; high-dimensional embeddings violate that, and distance concentration compresses the scores. Report 021 and Report 025 treat internal indices as **comparative, same-encoder signals**, not absolute quality. Use them as secondary diagnostics, never as the main evidence that a feature layer helped.

**Profile-specific check:** an appended profile can improve external agreement simply by making every posting look more like the evaluation schema. Compare the profile arm against a length-matched control and inspect whether the gain survives the independent audit and k-NN purity checks.

### 6.3 Circularity and leakage

Four circularity risks are specific to feature engineering:

1. **Tags as features and labels.** If the same role-family tags are used to build the input and to score the clusters, external metrics are inflated. Use independent labels, hand audit, or a title-only sensitivity analysis.
2. **Same model for extraction and labeling.** If the same LLM generates capability profiles, keyphrases, summaries, and evaluation labels, agreement may reflect shared model bias rather than semantic structure. Use a different model, a human sample, or a label-free metric as a check.
3. **Embedding-derived keywords.** KeyBERT/EmbedRank rank phrases using the same embedding geometry that will cluster the documents. This is not fatal, but it means the keyphrase view may add little independent signal. Test an LLM or statistical extractor as a contrast.
4. **Profile as a paraphrase of the label schema.** If the profile prompt uses the same role taxonomy that supplies the evaluation labels, the external metric can rise because the profile encodes the answer, not because the clustering discovered it. Keep the profile schema target-relevant but label-independent, or hold out the evaluation taxonomy.

### 6.4 Factorial design

Hold constant: source postings, cleaning version, encoder, clusterer, random seeds, and evaluation labels. Vary one operator at a time.

Because operators change length, also report token counts and consider a **length-matched control**: truncate or pad the baseline to the same token budget as the feature variant, so a gain is not merely "more text" or "less text". If a length-matched control is impractical, report the correlation between token count and cluster quality across arms.

For enrichment operators, compare **four forms** at the same source text: original, appended profile, separate profile view, and profile-only. This separates "the profile adds signal" from "the profile adds weight" from "the profile replaces the document". The same pattern applies to summaries and keyphrases.

Pre-register the primary endpoint and the success margin. With many arms, correct for multiple comparisons (Holm or Benjamini–Hochberg). Report effect sizes with paired bootstrap confidence intervals, not only p-values.

---

## 7. Fit Against This Project

### 7.1 Current constraints

| Constraint | State | Consequence for feature engineering |
|---|---|---|
| Encoder | Titan V2, 1,024-d, 8,192-token bound [22] | Coverage is no longer the binding constraint; dilution and content selection are |
| Cleaning | LLM boilerplate removal in production; current prompt preserves role/team context but strips generic company marketing | Test a role-aware variant that keeps company product/domain/scale context; move pay/location/legal to metadata rather than deleting from the lake |
| Corpus | ~9.6k postings; 1,000-posting clustering sample; 39-posting golden set | Enough for a 6–8 arm feature comparison on the sample; small for fine-grained taxonomy claims |
| Clustering | EVōC primary; K-Means/HDBSCAN comparison deferred | Feature results may be clusterer-specific; report the clusterer |
| Labels | LLM function tags from Experiment 012; hand audit | Independent enough for a first pass, but not ground truth; use audit + label-free checks |
| Architecture | Feature layer next in deliverable 3 | Store derived features as versioned artifacts, not as edits to `cleaned_text` |
| Cost | Bedrock embedding + LLM cleaning | Feature generation at 9.6k postings is cheap; re-embedding and re-tuning dominate |

### 7.2 Recommended feature ladder

Build in this order, stopping when a layer fails its gate:

| Rung | Representation | What it tests | Why this order |
|---|---|---|---|
| **0** | Cleaned text + Titan V2 | Control | Current production baseline |
| **1** | Role-aware cleaned text | Does keeping role-relevant company/product/team context help without reintroducing shared marketing? | Cleaning boundary is upstream of every feature; test it first |
| **2** | Field-aware natural-language prose (title, responsibilities, skills, seniority) | Does structure help without new facts? | Low risk; preserves prose; can be built deterministically |
| **3** | Grounded capability profile (role context, responsibilities, methods, technologies, domain, seniority) | Does semantic enrichment add signal beyond the cleaned text? | Strongest independent-pass recommendation; requires evidence spans |
| **4** | Capability profile as a separate view + weighted fusion with cleaned text | Does multi-view beat append and single-view? | Enrichment without replacing the source |
| **5** | Auxiliary keyphrase and ESCO-canonical skill views | Do lexical and taxonomy signals add complementary recall? | Published augmentation evidence [1]; needs mapping-quality check |
| **6** | Optional summary/narrative view | Does compression help a noisy or role-function subset? | Highest risk of information loss; conditional |
| **7** | Section decomposition / pooling | Does section weighting or decomposition help beyond full-text Titan? | Coverage is already solved; benefit may be small |

**What not to do:**

- Do not replace cleaned prose with a keyword-only, profile-only, or summary-only representation as the primary vector. It removes the relations that distinguish scope and seniority [2].
- Do not call a free-form summary a capability profile. If it compresses gist rather than selecting target-relevant capabilities with evidence, it is subject to the summary evidence [4].
- Do not append a profile without comparing it against a separate-view fusion and a length-matched control. Appending can win by weight rather than by signal.
- Do not use a summary as the default input for all postings. The evidence says it is conditional [4].
- Do not inject role-family labels into the embedding and then evaluate against the same labels. That is leakage, not feature engineering.
- Do not tune section weights on the same 1,000 postings used for the final comparison without a holdout or pre-registration.
- Do not attempt late chunking through the Titan API; it needs token-level hidden states [26].
- Do not modify the existing experiment scripts. Build the feature layer in `backend/` and a new numbered experiment directory, per `AGENTS.md`.

### 7.3 Architecture sketch

```
raw posting (lake, immutable)
        │
        ├── pay / location / legal ──► structured metadata (not embedded)
        │
        └── role-aware cleaning ──► cleaned_text (immutable, versioned)
                                      │
              ┌───────────────────────┼────────────────────────┐
              │                       │                        │
       field extractor        capability profiler        keyphrase extractor
       fields.json            profile.json (evidence)    keyphrases.json
              │                       │                        │
              │                skill mapper ──► esco_skills.json
              │                       │                        │
              └───────────────┬───────┴────────────┬───────────┘
                              │                    │
                      feature renderer             │
                              │                    │
              ┌───────────────┼────────────────────┐
              │               │                    │
       primary text view   profile view      auxiliary views
       (cleaned/field)     (capability)      (keyphrases, skills, summary)
              │               │                    │
           Titan V2        Titan V2             Titan V2
              └───────────────┼────────────────────┘
                              │
                  weighted fusion / concatenation
                              │
                            EVōC
```

Every artifact should carry `source_hash`, `feature_version`, `prompt_version`, `model`, `created_at`, `token_count`, `evidence_spans`, and `cost`. This makes the feature layer reproducible and lets a later experiment explain a cluster difference by pointing to a feature version rather than to a mutable script.

---

## 8. Proposed Experiment 016 — Feature-Engineering Ladder

> This is a proposal, not a completed experiment. It extends the variants sketched in [Report 023 §7](023-context-engineering-for-embeddings.md#7-proposed-experiment-for-this-project) with the operator evidence above. It should use the 1,000-posting sample and cached function labels from Experiment 012 where possible, and the 39-posting golden set for retrieval checks.

### 8.1 Hypotheses

- **H1 (role-aware cleaning).** A cleaning prompt that keeps role-relevant company/product/team context and removes only non-job boilerplate improves external clustering and audit fidelity without a stability loss, even if the separation gap falls slightly.
- **H2 (structure).** Field-aware prose beats the role-aware cleaned-text baseline on AMI/ARI against function tags and on k-NN purity, because it restores role and responsibility structure that flat text dilutes.
- **H3 (capability profile, appended).** A grounded capability profile appended to the role-aware cleaned text improves at least one primary clustering endpoint without increasing noise or reducing stability.
- **H4 (capability profile, multi-view).** A separate profile view fused with the cleaned text beats both the baseline and the appended profile, because the two views preserve complementary geometry.
- **H5 (profile-only).** A profile-only representation underperforms the prose baseline on clustering, because it loses the relations and context that the profile was meant to supplement.
- **H6 (keyphrase/skill augmentation).** Appending LLM keyphrases and canonical skills improves rare-skill retrieval and may improve skill-family clustering, but does not by itself beat the profile on role-function clustering.
- **H7 (summary).** A schema-guided summary appended to cleaned text does not beat the profile or multi-view arms on clustering; it may help a pre-defined noisy subset, but not the corpus average.
- **H8 (decomposition).** Section-level decomposition and weighted pooling do not materially improve over full-text Titan V2 on this corpus, because coverage is already solved; any gain is from weighting, not from additional context.
- **H9 (leakage check).** If role-family tags or the evaluation taxonomy are injected into the profile, the external metric rises more than the independent audit and label-free metrics; that is evidence of leakage, not improvement.

### 8.2 Arms

| ID | Representation | Purpose |
|---|---|---|
| A | Current cleaned text (production baseline) | Control |
| B | Role-aware cleaned text: keep company/product/team context, remove pay/location/legal from embedding text | Cleaning-boundary test |
| C | Field-aware natural-language prose from B (title, role, responsibilities, skills, seniority) | Structure without new facts |
| D | B + grounded capability profile appended as natural-language prose | Enrichment, appended |
| E | B and capability profile as separate views, normalized weighted fusion | Enrichment, multi-view |
| F | Capability profile only | Diagnostic: does the profile carry signal alone? |
| G | B + schema-guided abstractive summary (80–120 words) appended | Compression comparison |
| H | B + LLM keyphrases and ESCO-canonical skills appended as natural language | Auxiliary lexical/taxonomy signal |

Section decomposition and late chunking are follow-ups to 016, not primary arms. If D/E clear the gate, run a 016b decomposition check; late chunking waits for an encoder that exposes token-level hidden states [26].

### 8.3 Controls

- Generate all variants from the same cleaned text with the same LLM and prompt version where applicable; cache by `(source_hash, prompt_hash, model)`.
- Use Titan V2 for all embeddings; do not change the encoder between arms.
- Use the same EVōC configuration as Experiment 012; do not tune the clusterer per arm.
- Use cached Experiment 012 labels as the primary external target; do not tune features on the labels. Hold out a hand-audited subset for the final check.
- Build the capability profile with a fixed schema and require evidence spans for every claim; reject claims without source support.
- Generate the summary from B alone, not from B + profile, so the summary arm is comparable to the profile arm.
- Run at least two keyphrase extractors (e.g. YAKE and LLM) so a gain is not attributable to one method.
- Include a length-matched control: truncate B to the token count of D (or D to B) and compare. Report token counts, compression ratios, and cost per arm.
- Report all arms, not only the winner.

### 8.4 Metrics and gates

**Primary:** AMI against `primary_function`; k-NN purity. Separation gap is secondary because cleaning changes can raise it while reducing fidelity (Experiment 008); see also the proxy-metric definitions and limits in [23].

**Secondary:** ARI, V-measure, separation gap, EVōC noise fraction, seed/subsample stability, cluster audit agreement, Recall@10 on the golden set.

**Upstream:** profile claim precision and omission rate on 30 hand-checked postings; keyphrase precision@10; summary factuality/omission; ESCO mapping precision on 50 skills; determinism check; token count and cost.

**Success gate:** a variant must (a) improve a primary endpoint by the pre-registered margin with a paired-bootstrap 95% CI excluding zero, (b) not reduce stability below the baseline by more than the CI, and (c) not worsen the independent audit. For the profile arm, also require profile claim precision above the pre-registered threshold. Apply Holm correction across arms.

**Interpretation rules:**

- If B beats A on AMI/audit, keeping company context helped. If B lowers separation gap but improves audit, do not reject it on separation alone.
- If D beats B but E does not beat D, the profile helps by adding weight, not by adding a complementary view.
- If E beats D and B, multi-view is the right architecture.
- If F is close to B, the profile is carrying most of the signal and the source text is redundant — investigate before adopting.
- If G does not beat B, summarization is not earning its complexity. Keep it only for cluster naming.
- If H adds retrieval recall but not AMI, keep it as an auxiliary view, not the primary vector.

**Stop rule:** if no variant clears the gate, keep the cleaned-text baseline and invest in cleaning quality, encoder coverage, or clustering rather than adding a feature layer. A negative result is useful: it prevents a permanent complexity increase.

### 8.5 Implementation notes

- New module: `backend/pipeline/features/` with a `FeatureExtractor` protocol and one implementation per operator. Keep rendering (fields → text) separate from extraction so the experiment can vary format without re-running the LLM.
- New experiment directory: `scripts/016_feature_engineering/`; do not modify existing numbered experiment scripts (`AGENTS.md`).
- Store derived features as JSONL/Parquet under the data layer, not in git if large. Use `source_hash` as the join key.
- Use Pydantic models for the feature record and validate required fields. The capability profile record should carry `evidence_spans`, `claim_confidence`, and `schema_version`.
- Cache LLM outputs and record `prompt_version`, `model`, `temperature`, and `seed`.
- Tests: synthetic postings with known fields; profile schema validation with unsupported claims; keyphrase extractor mock; renderer snapshot tests; no real job data in fixtures.

---

## 9. Risks and Threats to Validity

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Small evaluation sample** | 1,000 postings and 39 golden postings limit statistical power; 1,000 is enough for large effects, not subtle ones | Pre-register primary endpoint; report CIs; treat non-significant differences as inconclusive, not proof of no effect |
| **Label circularity** | LLM function tags and LLM features share a model family | Hand audit, independent labeler, label-free metrics, title-only sensitivity |
| **Encoder-specific results** | Titan V2's long context and geometry differ from MiniLM and other encoders | Hold encoder fixed; do not generalize without re-running |
| **Clusterer-specific results** | EVōC may respond differently from K-Means or HDBSCAN | Report clusterer; rerun the winning arm on a second clusterer before production |
| **Prompt and model drift** | Keyphrase/summary quality changes with model version | Pin model and prompt; cache; re-baseline on model change |
| **Taxonomy mismatch** | ESCO granularity differs from posting language; mapping errors add noise | Measure mapping precision; use canonical skills as an auxiliary view first |
| **Company-context re-injection** | Keeping company descriptions can reintroduce shared marketing boilerplate and raise similarity across unrelated roles | Keep role-relevant product/domain/scale context; strip generic mission/values/awards; audit by company |
| **Profile hallucination or omission** | A profile can invent a capability or drop a rare one; the vector then looks cleaner for the wrong reason | Evidence spans; claim precision/omission audit; schema-guided prompt; keep source text |
| **Multi-view fusion weights** | A higher-variance view can dominate cosine distance; weights can be tuned to the evaluation labels | Normalize blocks; pre-register weights; compare against length-matched single views |
| **Late chunking unavailable** | Titan V2 returns one vector per input; token-level late pooling is not exposed | Defer until a local long-context encoder is in scope [26] |
| **Summary-centroid algorithm change** | `k-LLMmeans` changes the clustering loop, not the input features | Separate experiment; do not mix with feature-ladder results [27] |
| **Internal-metric misuse** | Silhouette/Davies–Bouldin can rank a geometrically tidy but semantically wrong representation | Use external labels, k-NN purity, stability, and audit as primary |
| **Hallucination/omission** | Summaries and keyphrases can invent or drop facts | Factuality/omission audit; schema-guided summaries; keep source text for traceability |
| **Metric gaming** | A feature can improve ARI while making the map less useful for job exploration | Combine external metrics with audit and downstream retrieval/visual inspection |
| **Cost creep** | Multiple LLM passes and re-embeddings add up at corpus scale | Cache; use the 1,000 sample for screening; estimate full-corpus cost before rollout |
| **Near-duplicate postings** | Duplicate or near-duplicate postings can inflate both cluster coherence and feature-extraction scores | Deduplicate before feature generation; use the near-duplicate pipeline in [24] |
| **Complexity without benefit** | A feature layer is permanent operational surface | Stop rule; adopt only with a measured, replicated gain |

A final validity note: the project's target is **role-function similarity**, not generic topic similarity. Most of the cited work evaluates topic/news/short-text clustering. The mechanisms transfer, but the effect sizes may not. That is why the recommended path is a controlled ladder rather than adopting a published recipe wholesale.

---

## 10. Sources

1. Viswanathan, V., Gashteovski, K., Lawrence, C., Wu, T., & Neubig, G. (2024). "Large Language Models Enable Few-Shot Clustering." *Transactions of the ACL*, 12:321–333. <https://aclanthology.org/2024.tacl-1.18/>
2. This repository. `docs/research-reports/023-context-engineering-for-embeddings.md` — format, keyword sets, summaries, and multi-view evidence.
3. Kostiuk & Enevoldsen (2026). "One prompt is not enough: Instruction Sensitivity Undermines Embedding Model Evaluation." arXiv:2605.22544. <https://arxiv.org/html/2605.22544v2>
4. Petukhova, A., Matos-Carvalho, J. P., & Fachada, N. (2024). "Text Clustering with Large Language Model Embeddings." *International Journal of Cognitive Computing in Engineering*. arXiv:2403.15112. <https://arxiv.org/abs/2403.15112>
5. "Structured Summaries for Retrieval-Augmented Generation in Portuguese-Language Consumer Complaints." PROPOR 2026. <https://aclanthology.org/2026.propor-1.84/>
6. This repository. `docs/research-reports/016-esco-eu-jobs-taxonomy.md` — ESCO structure, labels, licensing, and mapping caveats.
7. Senger, E., Zhang, M., van der Goot, R., & Plank, B. (2024). "Deep Learning-based Computational Job Market Analysis: A Survey on Skill Extraction and Classification from Job Postings." NLP4HR 2024. <https://aclanthology.org/2024.nlp4hr-1.1.pdf>
8. Grootendorst, M. (2022). "BERTopic: Neural topic modeling with a class-based TF-IDF procedure." arXiv:2203.05794. <https://arxiv.org/abs/2203.05794>; c-TF-IDF documentation: <https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html>
9. Grootendorst, M. (2020). "KeyBERT: Minimal keyword extraction with BERT." <https://github.com/MaartenGr/KeyBERT>
10. This repository. `docs/research-reports/025-semantic-clustering-methods.md` — four-lens clustering evaluation protocol.
11. This repository. `docs/research-reports/001-embedding-optimization-research.md` — section weighting, multi-granularity features, and optimization loop.
12. Bennani-Smires, K., Musat, C., Hossmann, A., Baeriswyl, M., & Jaggi, M. (2018). "Simple Unsupervised Keyphrase Extraction using Sentence Embeddings." CoNLL 2018. <https://aclanthology.org/K18-1022/>
13. This repository. `docs/experiments/003-llm-extraction.md` — separation gap `+0.0493` after LLM cleaning.
14. This repository. `docs/experiments/008-boilerplate-only-cleaning.md` — corrected boilerplate-only prompt and over-deletion reduction.
15. Campos, R., Mangaravite, V., Pasquali, A., Jorge, A., Nunes, C., & Jatowt, A. (2020). "YAKE! Keyword extraction from single documents using multiple local features." *Information Sciences*, 509:257–289. <https://doi.org/10.1016/j.ins.2019.09.013>
16. Mihalcea, R., & Tarau, P. (2004). "TextRank: Bringing Order into Texts." EMNLP 2004. <https://aclanthology.org/W04-3252/>
17. Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., & Manning, C. D. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval." arXiv:2401.18059. <https://arxiv.org/abs/2401.18059>
18. This repository. `docs/experiments/012-evoc-clustering-bakeoff.md` — Titan V2 selection, EVōC configuration, label circularity caveat, and stability gate.
19. Nogueira, R., Yang, W., Lin, J., & Cho, K. (2019). "Document Expansion by Query Prediction." arXiv:1904.08375. <https://arxiv.org/abs/1904.08375>
20. This repository. `docs/experiments/011-encoder-coverage-bakeoff.md` — truncation, full-text coverage, and Titan V2 arms.
21. This repository. `docs/research-reports/024-embedding-models-for-semantic-clustering.md` — why MTEB clustering is not the project's clustering task.
22. This repository. `docs/technical/embedding-pipeline.md` — Titan V2 contract, normalization, and production configuration.
23. This repository. `docs/technical/evaluation.md` — separation gap, self-retrieval, and planned IR metrics.
24. This repository. `docs/research-reports/017-near-duplicate-text-detection.md` — near-duplicate handling, which interacts with clustering and feature quality.
25. Anthropic Engineering (2024). "Introducing Contextual Retrieval." <https://www.anthropic.com/engineering/contextual-retrieval>
26. Günther, M., Mohr, I., Williams, D. J., & Wang, B. (2024). "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models." arXiv:2409.04701. <https://arxiv.org/abs/2409.04701>
27. "Summaries as Centroids for Interpretable and Scalable Text Clustering." ICLR 2026. <https://openreview.net/pdf?id=Uzku7RZXvI>
