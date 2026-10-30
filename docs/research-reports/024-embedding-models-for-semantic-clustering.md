# Embedding Models for Semantic Clustering — Sentence Transformers and Amazon Bedrock

> Date: 2026-10-30
> Context: Semantic Career Mapping Platform — model-selection research for the production encoder. The live pipeline embeds LLM-cleaned job postings with `sentence-transformers/all-MiniLM-L6-v2` (384-d, L2-normalized, `max_length=256`) into `postings_embeddings`, then clusters and projects the corpus ([data architecture §7](../architecture/data.md#7-embeddings), [Report 009](../experiments/009-pca-informativeness.md), [Report 012](012-evoc-embedding-clustering.md)).
> Scope: **text embedding models for semantic clustering**. Covers Sentence Transformers families and the text-capable Amazon Bedrock embedding models, plus the evaluation and migration consequences of changing encoder. Fine-tuning, rerankers, and ANN index selection are out of scope except where they change the model decision.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Clustering Is Not a Leaderboard Column](#2-clustering-is-not-a-leaderboard-column)
3. [What the Current System Actually Embeds](#3-what-the-current-system-actually-embeds)
4. [Sentence-Transformers Landscape](#4-sentence-transformers-landscape)
5. [Amazon Bedrock Landscape](#5-amazon-bedrock-landscape)
6. [What Public Benchmarks Do and Do Not Say](#6-what-public-benchmarks-do-and-do-not-say)
7. [Project-Specific Analysis](#7-project-specific-analysis)
8. [Cost, Latency, and Operational Fit](#8-cost-latency-and-operational-fit)
9. [Decision Framework and Recommended Path](#9-decision-framework-and-recommended-path)
10. [Proposed Experiment 011 — Encoder Bake-Off](#10-proposed-experiment-011-encoder-bake-off)
11. [Risks and Threats to Validity](#11-risks-and-threats-to-validity)
12. [Decision Log](#12-decision-log)
13. [Sources](#13-sources)

---

## 1. Executive Summary

**The most important finding is not a model.** The current encoder truncates every posting in the project's 39-posting golden cleaning set. With the MiniLM tokenizer, those cleaned postings have a mean length of **539 tokens**, a median of **516**, a 90th percentile of **807**, and a maximum of **1,004**; **39/39 exceed the 256-token limit**, 20/39 exceed 512, and 5/39 exceed 768. The current embedding therefore sees only the first portion of most postings. Until that is fixed, a model bake-off mostly measures how well each candidate handles the same truncated prefix.

**Recommendation in one sentence:** keep Sentence Transformers as the experiment substrate, fix full-text coverage first, then compare a small local shortlist against **Amazon Titan Text Embeddings V2** and **Cohere Embed v4** on this corpus before making any production switch.

| Decision | Recommendation | Why |
|---|---|---|
| Experiment encoder | Stay local with Sentence Transformers | Reproducible, no network/quota/cost per run, and the current pipeline already pins revisions and hashes encoder config |
| First no-regret change | Test MiniLM **chunk-pooling** against 256-token truncation | Uses the existing model, covers the full posting, and isolates the truncation defect before changing model family |
| Strong local challengers | `BAAI/bge-m3`, `google/embeddinggemma-300m`, `Qwen/Qwen3-Embedding-0.6B` | Long context (8k–32k), multilingual options, and clustering-aware prompts; `bge-m3` is the safest long-context upgrade, EmbeddingGemma the most efficient, Qwen3 the quality ceiling |
| Low-risk English challenger | `BAAI/bge-base-en-v1.5` | Well-understood 768-d/512-token upgrade, but it still truncates about half the sample |
| Bedrock default candidate | `amazon.titan-embed-text-v2:0` | 8k context, 256/512/1024-d, optional normalization, $0.02/M tokens, batch inference, AWS-native |
| Bedrock quality/context upgrade | `cohere.embed-v4:0` | 128k context, explicit `input_type="clustering"`, 256–1536-d, int8/binary output; ~6× Titan's token price |
| Bedrock multimodal option | `amazon.nova-2-multimodal-embeddings-v1:0` | Explicit `embeddingPurpose="CLUSTERING"`, 256/384/1024/3072-d, text/image/audio/video, 8k text context |
| Model change cost | Budget for re-embedding, re-tuning, and re-benchmarking | A new encoder invalidates the PCA `n_dims=256` default, UMAP fits, cluster thresholds, similarity distributions, and any S3 Vectors geometry |

**Do not select by MTEB overall score.** MTEB's clustering task is k-means with `k` equal to the number of labels, scored by V-measure. This project clusters with UMAP + HDBSCAN/EVōC and evaluates function tags and separation gap. Those are related but not the same question. Public scores are useful for shortlisting, not for adoption.

**Cost is not the deciding variable.** At the measured mean length, re-embedding the current 9,621-posting corpus costs roughly **$0.10 on Titan V2**, **$0.62 on Cohere Embed v4**, and **$0.70 on Nova 2**. At 1M postings the same rates are roughly **$11**, **$65**, and **$73**. The binding costs are engineering time, re-tuning, and the loss of local reproducibility — not the token bill.

---

## 2. Clustering Is Not a Leaderboard Column

A clustering encoder is judged by the geometry it induces, not by its average score across eight task families. Five properties matter more than a leaderboard rank:

### 2.1 Distance structure and thresholds are model-specific

HDBSCAN and EVōC operate on a density graph built from distances. Every encoder has a different similarity distribution: BGE v1.5 explicitly warns that its scores occupy roughly `[0.6, 1]`, so a raw cosine of 0.5 is not "similar" in the way it is for another model [10]. Cohere, Titan, and MiniLM each have different hubs and different typical nearest-neighbour distances. A distance threshold, `min_cluster_size`, or `min_samples` tuned on MiniLM is not transferable. A model swap is therefore **a re-clustering experiment**, not only a re-embedding job.

### 2.2 Anisotropy, hubness, and length collapse

Sentence embedding spaces are anisotropic: a few directions dominate, and some points become hubs that are neighbours of many unrelated points [14]. Hubness can be measured and partially corrected, but it also means that a model which looks better on average retrieval can produce worse density clusters. Long inputs add a second effect: embeddings of longer texts collapse toward the centre of the space and become spuriously similar to many documents [14]. This is directly relevant because the current corpus is long relative to MiniLM's 256-token window.

### 2.3 Context length is a representation decision

A 256-token encoder and an 8k-token encoder are not the same feature extractor applied to the same text. The first silently discards the tail; the second sees qualifications, seniority, and closing sections that may carry the discriminative signal for role function. The project has already established that LLM cleaning improves separation; the next question is whether the encoder actually reads the cleaned text.

### 2.4 Input contracts are part of the model

Several strong models require or recommend task-specific input formatting:

- **Cohere Embed v3/v4:** `input_type` must be `search_document`, `search_query`, `classification`, or `clustering`. The clustering option is not cosmetic; it conditions the embedding for the downstream task [6].
- **Nova Multimodal Embeddings:** `embeddingPurpose` accepts `GENERIC_INDEX`, `TEXT_RETRIEVAL`, `CLASSIFICATION`, `CLUSTERING`, etc. [8].
- **EmbeddingGemma:** clustering uses the symmetric prompt `task: clustering | query: {content}` on every input; retrieval uses asymmetric query/document prompts [4].
- **E5 / BGE / Qwen3:** query instructions or prefixes differ from document text; using the wrong side degrades retrieval and changes the geometry [1][11].

A fair bake-off must give each model its native input contract. Comparing a prompted Cohere model to an unprompted MiniLM is not a model comparison.

### 2.5 Dimensionality is a cost/latency lever, not a quality guarantee

384-d MiniLM, 768-d BGE/EmbeddingGemma, 1024-d Titan/bge-m3, 1536-d Cohere v4, and 3072-d Nova are different storage and distance-cost points. Matryoshka-trained models (EmbeddingGemma, Nova, Qwen3, Titan v2's dimension options) allow truncation, but truncated vectors must be re-normalized and re-evaluated. The project's PCA result — effective participation ratio 57.6, `n_dims=256` named only inside the MiniLM centered frame [9] — is **not transferable** to another encoder.

---

## 3. What the Current System Actually Embeds

The production configuration is explicit in `backend/pipeline/embedding/config.py` and `encoder.py`:

| Property | Current value |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Tokenizer truncation | `max_length=256` |
| Normalization | L2, asserted to `1e-5` tolerance |
| Device | CPU |
| Batch size | 16 |
| Revision | Pinned HuggingFace revision, asserted at load |
| Input column | `cleaned_text`, `variant=llm`, `version=v1` |
| Current corpus | 9,354 embedded rows in the 009 prefix; cleaned table has 9,621 rows |

MiniLM's model card confirms the 256-word-piece truncation and 384-d output [2]. It was trained at a 128-token sequence length, so raising `max_seq_length` to 512 is not a free upgrade; it runs the encoder beyond its training distribution and does not guarantee that the tail is used well. The project's prior context-engineering research reached the same conclusion: longer inputs can lose more from truncation and length collapse than they gain [Report 023](023-context-engineering-for-embeddings.md).

### 3.1 Measured length of the cleaned text

Using the MiniLM tokenizer on the 39-posting golden cleaning set (`data/golden_cleaned.json`, the hand-curated evaluation text):

| Statistic | Value |
|---|---|
| Mean tokens | 539.3 |
| Median | 516 |
| Standard deviation | 167.8 |
| 90th percentile | 807 |
| 95th percentile | 911 |
| Maximum | 1,004 |
| Postings over 256 tokens | 39/39 (100%) |
| Postings over 512 tokens | 20/39 (51%) |
| Postings over 768 tokens | 5/39 (13%) |

The sample is the project's curated evaluation set, not a random sample of all 9,621 postings, so the exact percentages should not be treated as corpus-wide estimates. The direction is unambiguous: the current encoder sees roughly the first half of a typical cleaned posting. If the tail contains qualifications, tool lists, or seniority language, those signals are absent from the vector.

This is also a **confound in the existing results**. The separation-gap improvements attributed to LLM cleaning (+0.0493, later +0.0657 on the expanded set) were measured on truncated embeddings. They are real relative to raw text under the same encoder, but they are not an upper bound on what the cleaned text can produce.

---

## 4. Sentence-Transformers Landscape

Sentence Transformers 6.1.0 is the installed library (`uv.lock`); it supports PyTorch, ONNX, and OpenVINO backends, prompts, `truncate_dim`, and quantization utilities [3]. The relevant model families for this project fall into four groups.

### 4.1 Shortlisted local models

| Model | Dims | Max tokens | Size/params | Languages | Input contract | Why it matters here |
|---|---:|---:|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | ~80–90 MB, 6-layer | English | None | Current baseline; fastest; truncates the corpus |
| `BAAI/bge-base-en-v1.5` | 768 | 512 | 109M | English | Query instruction optional; documents plain | Low-risk quality step, but still truncates ~half the sample |
| `BAAI/bge-small-en-v1.5` | 384 | 512 | 33M | English | Same as BGE | Cheap 512-token alternative; lower ceiling |
| `BAAI/bge-large-en-v1.5` | 1024 | 512 | 335M | English | Same as BGE | Higher quality, heavier CPU, still 512 tokens |
| `BAAI/bge-m3` | 1024 | 8,192 | ~568M class | 100+ | Dense/sparse/multi-vector; no required prompt for dense | Covers the whole posting and multilingual postings; heavier than MiniLM |
| `google/embeddinggemma-300m` | 768 (MRL: 512/256/128) | 2,048 | 300M | 100+ | Symmetric clustering prompt; separate retrieval prompts | Best efficiency/long-context fit; explicit clustering prompt; MRL |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 (MRL: 32–1024) | 32,768 | 0.6B | 100+ | Instruction-aware; last-token pooling | Quality ceiling among CPU-feasible models; slower and more complex |
| `microsoft/harrier-oss-v1-270m` | 640 | 32,768 | 270M | 94 | Instruction-tuned; last-token pooling | New long-context multilingual option; less independent evidence |

Table sources: model cards and Sentence Transformers documentation [1][2][4][5][11][12]. Sizes are approximate and exclude tokenizer/framework overhead.

### 4.2 The practical local shortlist

For this project, the local shortlist should be **four arms, not ten**:

1. **MiniLM + chunk-pooling.** No model migration. Split each cleaned posting into 256-token windows (with a small overlap), encode each window, L2-normalize, mean-pool, then re-normalize. This directly tests whether full-text coverage helps without confounding model quality. It is the cheapest and most interpretable next experiment.
2. **`bge-base-en-v1.5`.** The conventional upgrade: 768-d, 512-token context, strong English retrieval, widely deployed. It is the "did a standard base model beat MiniLM?" control. It still truncates 20/39 sample postings, so it is not the final answer if context is the defect.
3. **`bge-m3`.** The safest long-context local upgrade: 8,192 tokens, 1024-d, 100+ languages, dense retrieval plus optional sparse/multi-vector. It covers every sampled posting and handles bilingual postings if the scraper expands beyond English. The cost is CPU time and 2.7× the vector storage of MiniLM.
4. **`embeddinggemma-300m`.** The most interesting efficiency point: 2,048 tokens, 768-d with Matryoshka truncation to 512/256/128, 300M parameters, and a documented clustering prompt. It is a better fit than a 0.6B decoder for a CPU-first pipeline, while still covering the measured text.

`Qwen3-Embedding-0.6B` is the quality ceiling to keep in reserve. It has 32k context, 1024-d with MRL down to 32, and instruction-aware behavior, but a decoder-only 0.6B model is materially slower on CPU than MiniLM/BGE-base and adds prompt/pooling complexity. It is worth one arm if the local bake-off shows a clear quality gap that a 0.6B model might close; it is not the default.

`harrier-oss-v1-270m` is a credible newer alternative (640-d, 32k context, 94 languages, MTEB v2 66.5), but it has less independent production evidence than BGE or EmbeddingGemma. Treat it as a reserve candidate.

### 4.3 Model families not to prioritize

- **Static/edge embeddings (Model2Vec/Potion).** Very fast and small, but lower average MTEB quality; useful only if CPU throughput becomes the binding constraint. Not a clustering-quality upgrade [15].
- **Large decoder embedders (Qwen3-4B/8B, NV-Embed-v2, E5-Mistral-7B).** Strong public scores, but CPU inference is impractical and the project is explicitly CPU-first. Reserve for a GPU batch job if quality plateaus.
- **Multilingual-only models if the corpus stays English.** `multilingual-e5`, LaBSE, and `paraphrase-multilingual-*` add cost without a demonstrated English clustering gain. Revisit only when non-English postings enter the corpus.
- **Fine-tuning.** The project's scope is preprocessing and model selection, not training foundation models. Fine-tuning a retriever on 27–39 labels would overfit and is not justified.

---

## 5. Amazon Bedrock Landscape

Bedrock exposes several text-capable embedding models. The relevant set for job-posting clustering is Titan Text Embeddings V2, Cohere Embed English/Multilingual v3, Cohere Embed v4, and Nova Multimodal Embeddings. Titan v1, Titan Multimodal G1, and TwelveLabs Marengo are either legacy or media-first.

### 5.1 Facts that matter

| Model | Model ID | Context | Dimensions | Input contract | On-demand price | Notes |
|---|---|---|---:|---|---:|---|
| **Titan Text Embeddings V2** | `amazon.titan-embed-text-v2:0` | 8,192 tokens / 50k chars | 1024 default, 512, 256 | No task prefix; optional `normalize`; binary output | **$0.02 / 1M tokens** | Batch + provisioned throughput; 100+ languages (preview); optimized for retrieval but documented for clustering |
| Titan Embeddings G1 | `amazon.titan-embed-text-v1` | 8k class | 1536 | No task prefix | $0.10 / 1M tokens | Legacy; 25+ languages; larger vectors, no dimension choice |
| Cohere Embed English v3 | `cohere.embed-english-v3` | 512 | 1024 | `input_type` required | $0.10 / 1M tokens | Mature English model; binary output; provisioned throughput |
| Cohere Embed Multilingual v3 | `cohere.embed-multilingual-v3` | 512 | 1024 | `input_type` required | $0.10 / 1M tokens | 100+ languages; binary output |
| **Cohere Embed v4** | `cohere.embed-v4:0` | ~128k tokens | 1536 default, 1024, 512, 256 | `input_type` including **`clustering`**; float/int8/uint8/binary/ubinary | **$0.12 / 1M tokens** | Text + images; 96 texts/call; 20 MB payload; Marketplace billing |
| **Nova 2 Multimodal Embeddings** | `amazon.nova-2-multimodal-embeddings-v1:0` | 8,172 tokens | 3072 default, 1024, 384, 256 | `embeddingPurpose` including **`CLUSTERING`**; `embeddingDimension` | **$0.135 / 1M tokens** (batch ~$0.0675, third-party listing) | Text/image/audio/video; 200 languages; MRL; narrower regional availability |
| Titan Multimodal G1 | `amazon.titan-embed-image-v1` | Text/image | 1024 | No task prefix | $0.10 / 1M text tokens + image | Only relevant if job postings include images/screenshots |
| TwelveLabs Marengo 3.0 | `twelvelabs.marengo-embed-3-0-v1:0` | 500 text tokens | — | Video-first | Per media unit | Not suitable as a job-text encoder |

Table sources: AWS Titan docs/blog, Cohere v4 docs and Marketplace listing, Nova docs/blog/pricing, Bedrock model catalog [5][6][7][8][16][17][18][19].

### 5.2 Why Titan V2 is the Bedrock default

Titan V2 is the only Bedrock text embedding model that combines **8k context**, **three output dimensions**, **optional normalization**, **$0.02/M tokens**, **batch inference**, and a mature AWS lifecycle. Its 256-d option is particularly relevant to this project: AWS reports only about **3.24% accuracy loss** versus 1024-d on its internal evaluation, with a 4× storage reduction [16]. That is a credible storage/latency lever, though it must be measured on job postings.

Its weakness is task conditioning: Titan V2 has no `input_type` or `embeddingPurpose`. It is optimized for retrieval and documented as usable for clustering, but it cannot be told "this is a clustering workload" the way Cohere v4 or Nova can. For a pure clustering bake-off, that makes Cohere v4 and Nova the more task-aligned managed options.

### 5.3 Why Cohere Embed v4 is the quality/context upgrade

Cohere Embed v4 is the most complete managed option for this project's text:

- **128k context** removes truncation as a concern at any plausible posting length.
- **`input_type="clustering"`** directly matches the downstream task.
- **256–1536 dimensions** and **int8/binary output** give a storage/latency ladder without re-embedding at each dimension.
- **96 texts per call** and a 20 MB payload make backfills practical.
- The trade-offs are **closed weights**, **Marketplace billing**, **~6× Titan's token price**, and dependence on Cohere's model lifecycle.

Published third-party comparisons put Cohere v4 ahead of Titan v2 on retrieval quality, but often by only ~1–1.5 points on the tested workload [20]. The project should treat that as a hypothesis to test on job postings, not a settled ranking.

### 5.4 Why Nova is a conditional option

Nova 2 Multimodal Embeddings is the only Bedrock model in this set with an explicit **`embeddingPurpose="CLUSTERING"`** and a **384-d** output. The 384-d option is the same geometry as the current MiniLM corpus, which makes it the least disruptive managed migration on storage. It also covers text, images, audio, and video in one space, which matters only if the product expands beyond text postings.

The constraints are real: 8,172-token text context, default 3072-d (8× current storage), $0.135/M text tokens, and availability concentrated in `us-east-1`/`us-gov-west-1` at the time of writing [8][18]. It is a future-facing option, not the first managed candidate for a text-only clustering pipeline.

### 5.5 Operational facts

- **Throttling is RPM-based for embedding models**, not TPM-based; plan capacity around requests per minute, not only token volume [5].
- **Batch inference** is supported for Titan V2 and Nova, with JSONL input in S3 and asynchronous output. Verify Cohere v4 batch support before designing a backfill around it [21].
- **On-demand latency** includes network and service queueing; local MiniLM inference is ~20 ms/posting on this machine. A serial Bedrock backfill of 9,621 postings is minutes-to-tens-of-minutes, not milliseconds, and is subject to quota and retry behavior.
- **Model access** must be enabled per account/region; IAM, VPC endpoints, and Marketplace subscription (for Cohere v4) are part of the setup.
- **Portability:** Titan, Cohere, and Nova are closed weights. Leaving Bedrock means re-embedding the corpus. The project's model-first Parquet partitions and configuration hash make this tractable, but it is still a full re-embed and re-tune.

---

## 6. What Public Benchmarks Do and Do Not Say

Public scores are a shortlisting tool. The table below is deliberately heterogeneous: **do not compare rows across suites**.

| Model | Suite | Overall | Clustering | Reading |
|---|---|---:|---:|---|
| `all-MiniLM-L6-v2` | MTEB English v1 (legacy 56-task average) | 55.93 | 44.90 | Fast small baseline; clustering is its weakest family |
| `bge-base-en-v1.5` | MTEB v1 (56-task average) | 63.55 | 45.77 | Clear overall gain over MiniLM; clustering gain is modest |
| `bge-m3` | Third-party MTEB English | ~63.5 | — | Long-context/multilingual value, not a clustering score |
| Titan Text Embeddings V2 | AWS-reported MTEB weighted average | 60.37 | Not published as an average | Per-task clustering V-measure ranged ~30.9–41.8 on four tasks; vendor-reported |
| Cohere Embed v4 | Third-party MTEB English | 66.72 | 49.8 | Strong overall; third-party aggregation, verify |
| `embeddinggemma-300m` | MTEB(eng, v2) | 69.7 | 53.3 | Newer suite; strong for its size; not comparable to v1 rows |
| `Qwen3-Embedding-0.6B` | MTEB(eng, v2) | 70.5 | — | Official paper score; 0.6B decoder, CPU cost |
| `Qwen3-Embedding-8B` | MTEB(eng, v2) | 75.2 | — | Quality ceiling; not CPU-viable |
| `harrier-oss-v1-270m` | MTEB v2 (multilingual) | 66.5 | — | New long-context option; less independent evidence |

Sources: MiniLM/model2vec table [15], BGE official table [10], AWS Titan blog/service card [5][16], Cohere third-party aggregation [22], EmbeddingGemma and Qwen3 leaderboard/paper [4][11][23], Harrier model card [12], BGE-M3 third-party comparison [26].

Three traps:

1. **The average is not your task.** MTEB averages retrieval, classification, clustering, reranking, STS, and summarization. A model can win overall and lose on clustering.
2. **The clustering task is not your clustering.** MTEB uses k-means with `k` equal to the label count and scores V-measure. This project uses density clustering (HDBSCAN/EVōC) on a different label taxonomy, with a much smaller labelled subset. Rankings can differ.
3. **Benchmarks are public and trainable.** A two-point MTEB gain is not a guarantee of a two-point gain on job postings, especially when the model was released by a team optimizing for the benchmark.

The correct use of MTEB here is: shortlist 4–6 models, then run the project's own experiment with the project's text, labels, clustering algorithm, and cost constraints.

---

## 7. Project-Specific Analysis

### 7.1 The truncation defect dominates the model question

At `max_length=256`, the encoder cannot see most of a typical cleaned posting. A larger model with the same 256-token truncation would still miss the same tail; a 512-token model would fix only about half the sample. The first experiment should therefore separate **text coverage** from **model quality**:

- **Coverage fix without model change:** MiniLM chunk-pooling.
- **Coverage fix with model change:** `bge-m3` (8k), `embeddinggemma-300m` (2k), `Qwen3-0.6B` (32k), Titan V2 (8k), Cohere v4 (128k), Nova (8k).
- **Partial coverage:** `bge-base-en-v1.5` / 512-token models.

If chunk-pooling produces most of the gain, the model decision becomes about quality per unit cost, not about rescuing lost text. If chunk-pooling produces little gain, the tail is not carrying the signal the project hypothesized, and a long-context model is less urgent.

### 7.2 The existing calibration does not transfer

The project's PCA experiment named `n_dims=256` **inside the MiniLM centered frame**, after measuring participation ratio 57.6 and neighborhood retention [9]. That number is a property of the MiniLM embedding geometry and the centering operation, not a universal property of job postings. A new encoder requires:

- a fresh PCA/participation-ratio measurement;
- fresh UMAP `n_neighbors`/`min_dist` checks;
- fresh HDBSCAN/EVōC density thresholds;
- fresh similarity thresholds for any query-side filtering;
- a fresh storage geometry if S3 Vectors is used (dimension and cosine index are frozen at creation) [Report 019](019-s3-vectors.md).

The existing `EmbeddingConfig` already hashes model name, revision, max length, normalization, and text variant, so a new encoder lands as a new dataset version rather than overwriting the old one. That is the right migration mechanism. What it does **not** yet carry is a Bedrock provider/region/model-ID/input-type field; adding a managed encoder requires extending the config and the `EmbeddingEncoder` interface, not just swapping a model string.

### 7.3 Preprocessing has been the larger lever so far

The project's documented gains come mostly from text cleaning, not encoder choice:

- Raw → regex cleaning: separation gap +0.0060 → +0.0203.
- Regex → LLM cleaning: +0.0203 → +0.0493 (8.2× raw), later +0.0657 on the expanded golden set.
- LLM cleaning is now production, with mean Jaccard 0.958 and 0.9% over-deletion on the live eval intersection.

A new encoder might add a few points. Full-text coverage might add more. The report should not encourage a model migration that leaves the truncation defect in place.

### 7.4 Label scarcity constrains the evaluation

The function-taxonomy labels cover 27 postings; the golden cleaning set covers 39. That is enough for directional separation-gap and cluster-purity checks, not for a definitive MTEB-style clustering comparison. The bake-off should use:

- **separation gap** on the labelled subset (within-role vs cross-role cosine), with bootstrap confidence intervals;
- **k-NN label agreement / purity** on the labelled subset, which is less sensitive to clustering hyperparameters than ARI/AMI;
- **cluster stability** (ARI across seeds/subsamples) and **noise fraction** for HDBSCAN/EVōC;
- **qualitative map inspection** for the role-function clusters, because 27 labels cannot adjudicate subtle differences.

Report 021 already argues that internal indices such as silhouette and DBCV misbehave on high-dimensional embeddings and that k-NN purity and stability are better suited to this setting. The bake-off should follow that guidance rather than optimizing a single internal score.

### 7.5 Multilingual and multimodal triggers

- **Multilingual:** current postings are English. If the scraper expands to French/Canadian or other languages, MiniLM is the wrong encoder; `bge-m3`, EmbeddingGemma, Harrier, Cohere Multilingual/v4, or Nova become relevant. This is a product trigger, not a benchmark trigger.
- **Multimodal:** if postings include screenshots, PDFs, or images, Cohere v4 and Nova become relevant; Titan V2 and local text models do not. Do not pay the multimodal premium before there is multimodal data.

---

## 8. Cost, Latency, and Operational Fit

### 8.1 Token-cost math

Assume the measured mean of **539 tokens/posting** and the current cleaned corpus of **9,621 postings**, for **5.19M input tokens**. At 1M postings, the same mean gives **539M tokens**.

| Model | $/1M tokens | 9,621 postings | 1M postings | Relative to Titan |
|---|---:|---:|---:|---:|
| Local MiniLM (existing) | $0 marginal | $0 + compute | $0 + compute | — |
| Titan Text Embeddings V2 | $0.02 | **$0.10** | **$10.78** | 1× |
| Cohere Embed English v3 | $0.10 | $0.52 | $53.9 | 5× |
| Cohere Embed v4 | $0.12 | **$0.62** | **$64.7** | 6× |
| Nova 2 Multimodal | $0.135 | **$0.70** | **$72.8** | 6.75× |
| Nova 2 batch (third-party listing) | ~$0.0675 | ~$0.35 | ~$36.4 | ~3.4× |

These are order-of-magnitude estimates, not invoices. Bedrock tokenizers differ from MiniLM's, batch discounts vary, and region/Marketplace pricing changes. The point stands: **the API bill is trivial at this project's scale.** Engineering time to re-embed, re-tune, and validate dominates.

### 8.2 Storage math

At 9,621 rows, float32:

| Dimensions | Vector storage | Relative to 384-d |
|---:|---:|---:|
| 384 (MiniLM) | 14.8 MB | 1× |
| 512 | 19.7 MB | 1.33× |
| 768 (BGE-base, EmbeddingGemma) | 29.6 MB | 2× |
| 1024 (Titan, BGE-M3) | 39.4 MB | 2.67× |
| 1536 (Cohere v4) | 59.1 MB | 4× |
| 3072 (Nova default) | 118.2 MB | 8× |

At 1M postings, 3072-d float32 is ~12.3 GB. That is still inexpensive in S3, but it is 8× the distance cost and index size of the current corpus. Matryoshka truncation (Titan 256/512, Cohere 256/512/1024, Nova 384/1024, EmbeddingGemma 128/256/512, Qwen3 32–1024) is the right lever if storage or clustering time becomes binding.

### 8.3 Operational fit matrix

| Criterion | Local Sentence Transformers | Amazon Bedrock |
|---|---|---|
| Reproducibility | Strong: pinned HF revision, deterministic CPU, no network | Weaker: model ID + region + service behavior; closed weights; quotas |
| Experiment loop | Fast: no per-call cost/quota; can iterate freely | Slow: network, RPM, retries, cost accounting |
| Ops burden | Model download, CPU/RAM, dependency management | IAM, model access, VPC endpoints, quota planning, billing |
| Scale-up | Add CPU/instances; batch locally | Managed, batch inference, provisioned throughput |
| Data boundary | Data stays local | Data sent to AWS service; use VPC endpoint/KMS as needed |
| Portability | High (open weights) | Low (closed model, re-embed to leave) |
| Cost at 10⁴–10⁶ rows | Compute only | Cents to tens of dollars; negligible |
| Best role | Research loop, offline batch, fallback | Production backfill at scale, managed serving |

**Recommended split:** local Sentence Transformers for experiments and as a reproducibility fallback; Bedrock for a production backfill or a managed serving path once a model has won on this corpus.

---

## 9. Decision Framework and Recommended Path

### 9.1 Decision rules

1. **If the goal is clustering quality on this English corpus:** fix full-text coverage first. Test MiniLM chunk-pooling, then `bge-m3` and `embeddinggemma-300m`. Use `Qwen3-0.6B` only if the local shortlist plateaus.
2. **If the goal is a managed AWS production path:** start with **Titan V2** on cost and operational simplicity. Move to **Cohere Embed v4** only if its project-measured clustering gain justifies ~6× token cost and Marketplace/closed-model dependency.
3. **If multilingual postings become real:** prefer `bge-m3`, EmbeddingGemma, Harrier, Cohere Multilingual/v4, or Nova. Do not run a multilingual model on an English-only corpus without measuring the cost/quality trade.
4. **If multimodal postings become real:** consider Cohere v4 or Nova. Titan V2 and local text models are out.
5. **If storage or clustering time becomes binding:** evaluate MRL truncation (Titan 256/512, EmbeddingGemma 256/512, Nova 384/1024, Qwen3 256/512) before switching models. Re-normalize and re-measure.
6. **Never adopt on MTEB alone.** A model must win on this corpus's separation gap, k-NN label agreement, cluster stability, and cost.

### 9.2 Recommended sequence

| Stage | Action | Why |
|---|---|---|
| 0 | Re-embed a fixed 9,354-row prefix with the current MiniLM at 256 for the control | Apples-to-apples with the 009 prefix; no new model |
| 1 | Add MiniLM chunk-pooling (256-token windows, mean-pool, re-normalize) | Isolates full-text coverage from model quality; no migration |
| 2 | Add `bge-base-en-v1.5` (512) and `embeddinggemma-300m` (2,048, clustering prompt) | Tests standard base model vs efficient long-context model |
| 3 | Add `bge-m3` (8,192) and, if CPU budget allows, `Qwen3-0.6B` | Tests long-context/multilingual and quality ceiling |
| 4 | Add Bedrock Titan V2 (1,024) and Cohere v4 (1,024, `clustering`) | Tests managed cost/quality; one-off, not the experiment substrate |
| 5 | Re-tune UMAP/HDBSCAN/EVōC and re-run 009-style PCA on the winner | A new encoder invalidates old geometry and thresholds |
| 6 | Adopt as a new `model=`/`version=` dataset; keep MiniLM as fallback | Preserves lineage and rollback |

**Default if no experiment is run:** keep MiniLM for production, but treat the 256-token truncation as a known defect and do not claim the current clustering is full-text. Do not switch to Bedrock on public benchmarks alone.

---

## 10. Proposed Experiment 011 — Encoder Bake-Off

This report does not run the experiment. The following is a preregistered design for the next experiment.

### 10.1 Hypotheses

- **H1 (coverage):** MiniLM chunk-pooling improves separation gap and k-NN label agreement over MiniLM 256-token truncation, with no model change.
- **H2 (long context):** A long-context model (`bge-m3`, EmbeddingGemma, Titan V2, Cohere v4) beats MiniLM chunk-pooling on the same full text.
- **H3 (managed):** Bedrock Titan V2 or Cohere v4 beats the best local arm enough to justify the managed dependency; or it does not, and local stays.
- **H4 (geometry):** Any gain survives re-tuning of UMAP/HDBSCAN/EVōC; if it only survives with MiniLM-tuned thresholds, it is not a real gain.

### 10.2 Arms

Use the same cleaned text and the same fixed 9,354-row `source_hash` prefix as the 009 corpus. Normalize all outputs to unit length before comparison.

| Arm | Encoder | Context | Dims | Provider | Native contract | Approx. cost |
|---|---|---:|---:|---|---|---:|
| A0 | MiniLM-L6-v2 | 256 | 384 | Local | None | $0 |
| A1 | MiniLM-L6-v2, chunk-pool | 256/window | 384 | Local | None | $0 |
| A2 | bge-base-en-v1.5 | 512 | 768 | Local | Documents plain | $0 |
| A3 | embeddinggemma-300m | 2,048 | 768 (or 512/256) | Local | `task: clustering \| query:` | $0 |
| A4 | bge-m3 | 8,192 | 1,024 | Local | Dense mode, no prompt | $0 |
| A5 | Qwen3-Embedding-0.6B | 32,768 | 1,024 (or 512) | Local | Instruction-aware | $0 + CPU time |
| A6 | Titan Text Embeddings V2 | 8,192 | 1,024 | Bedrock | No prefix; normalize | ~$0.10 |
| A7 | Cohere Embed v4 | 128k | 1,024 | Bedrock | `input_type="clustering"` | ~$0.62 |

Optional: A8 Nova 2 at 384/1024 with `embeddingPurpose="CLUSTERING"` if multimodal or AWS-native 384-d is a live requirement.

### 10.3 Metrics

**Primary (decision):**

1. **Separation gap** on the 39-posting golden set: mean within-role cosine minus mean cross-role cosine, with 1,000-sample bootstrap 95% CI.
2. **k-NN label agreement** (k = 15) on the labelled subset: fraction of each posting's neighbours sharing its function tag, macro-averaged. This is less sensitive to cluster count than ARI/AMI.
3. **Cluster stability**: ARI between HDBSCAN/EVōC partitions across 5 seeds and 80% subsamples; report noise fraction and cluster count.

**Secondary (diagnostic):**

- self-retrieval (degeneracy check; should be 100%);
- cosine distribution (mean, quantiles, hubness proxy);
- UMAP trustworthiness / 15-NN preservation versus the full space;
- qualitative map inspection of role-function clusters;
- encoder throughput (postings/s), peak RSS, model size;
- vector storage and distance cost at 1M rows.

**Gates (preregistered):**

- A challenger is **adoptable** only if separation gap improves by at least **+0.005 absolute** with a 95% CI excluding zero, **and** k-NN label agreement does not decrease by more than 2 points, **and** noise fraction does not increase by more than 10 points.
- If no arm clears both quality gates, keep the current encoder and record the null result.
- If a Bedrock arm wins on quality but loses on reproducibility or cost, it becomes a **production candidate**, not the experiment encoder.

### 10.4 Re-tuning contract

For every arm:

- re-run the 009-style PCA/participation-ratio measurement (do not reuse `n_dims=256`);
- choose UMAP `n_neighbors`/`min_dist` from the same small grid, with a fixed random seed;
- tune HDBSCAN `min_cluster_size`/`min_samples` per arm on cluster stability, not on label peeking;
- report both the tuned and the default-threshold results, so the reader can see how much of the gain is tuning.

### 10.5 Expected outcome

The most likely outcome is **a modest but real gain from full-text coverage, and a smaller incremental gain from a larger model**. The project's largest documented wins came from cleaning; the encoder is the next lever, not the first. A null result on a 0.6B model would be valuable: it would show that the corpus is preprocessing-limited, not model-limited, and would justify stopping at a smaller, cheaper encoder.

---

## 11. Risks and Threats to Validity

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Truncation confound** | A model comparison on 256-token inputs measures prefix handling, not model quality | Run the chunk-pool arm; use full-text arms; report token coverage per arm |
| **MTEB mismatch** | Public clustering scores use k-means/V-measure on different labels | Treat MTEB as shortlist only; use project separation gap, k-NN agreement, stability |
| **Label scarcity** | 27–39 labelled postings cannot resolve small differences | Bootstrap CIs; prefer effect sizes over point rankings; qualitative map checks |
| **Threshold leakage** | Reusing MiniLM cluster parameters can make a new model look better or worse arbitrarily | Re-tune per arm; report tuned and default results |
| **Similarity-scale drift** | BGE, Cohere, Titan, and MiniLM have different cosine distributions | Re-derive thresholds; never compare raw cosine values across models |
| **Length collapse** | Longer postings can become generic and cluster toward the centre | Measure hubness and neighbour diversity; test MRL truncation |
| **Closed-model drift** | Bedrock model IDs can be updated or retired; no HF-style revision pin | Record model ID, region, request date, and API response metadata; keep a local fallback |
| **Quota/backfill surprise** | Bedrock throttles embedding models by RPM; batch support varies | Prototype the backfill; use batch where supported; implement exponential backoff and idempotent retries |
| **Marketplace billing** | Cohere v4 and other third-party models bill through Marketplace, not the Bedrock line item | Confirm billing/allowlist before the bake-off; set a budget alarm |
| **Licensing** | EmbeddingGemma is not Apache-2.0; some models have commercial terms | Check the model card license before production; keep license notes in the model registry |
| **Small-corpus overfitting** | A model that wins on 39 postings may not win at 10⁵–10⁶ | Re-evaluate at the next corpus milestone; do not hard-code a winner forever |

---

## 12. Decision Log

| Decision | Rationale | Date |
|---|---|---|
| Keep Sentence Transformers as the experiment encoder for now | Reproducible, zero per-run cost, no RPM/quota, existing pinned-revision and config-hash infrastructure | 2026-10-30 |
| Treat 256-token truncation as a known defect, not a neutral baseline | 39/39 golden cleaned postings exceed 256 tokens; mean 539; current vectors miss the tail | 2026-10-30 |
| Run MiniLM chunk-pooling before any model migration | Isolates full-text coverage from model quality; cheapest no-regret experiment | 2026-10-30 |
| Local long-context shortlist: `bge-m3`, `embeddinggemma-300m`, `Qwen3-0.6B` | 8k–32k context, multilingual options, clustering-aware prompts; CPU-feasible | 2026-10-30 |
| Bedrock shortlist: Titan V2 (default), Cohere v4 (quality/context), Nova (multimodal/384-d) | Titan is cost/ops default; Cohere has explicit clustering mode; Nova is the multimodal/MRL option | 2026-10-30 |
| Do not adopt on MTEB alone | MTEB clustering is k-means/V-measure, not this project's UMAP/HDBSCAN/EVōC + function tags | 2026-10-30 |
| Budget for full re-tuning after any model change | PCA `n_dims=256`, UMAP, HDBSCAN/EVōC thresholds, and S3 Vectors geometry are model-specific | 2026-10-30 |
| Do not switch production encoder before Experiment 011 | No project-specific evidence yet; public scores are shortlist signal only | 2026-10-30 |

---

## 13. Sources

1. Sentence Transformers, *Pretrained Models* — model selection guidance and MTEB caveats. <https://sbert.net/docs/sentence_transformer/pretrained_models.html>
2. Hugging Face, `sentence-transformers/all-MiniLM-L6-v2` model card — 384-d, 256-token truncation, training data. <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
3. Sentence Transformers 6.1.0 release notes — backends, benchmarks, quantization fixes. <https://github.com/huggingface/sentence-transformers/releases>
4. Google, *EmbeddingGemma model card* — 300M, 2,048 tokens, 768-d MRL, clustering prompt. <https://ai.google.dev/gemma/docs/embeddinggemma/model_card>
5. AWS, *Amazon Titan Text Embeddings models* — model IDs, 8,192-token context, 256/512/1024-d, normalize, binary output, RPM throttling. <https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html>
6. AWS, *Cohere Embed v4* — request schema, `input_type`, 128k context, 256–1536-d, quantization. <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html>
7. AWS, *Cohere Embed v4 model card* — lifecycle, endpoints, Marketplace billing. <https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-cohere-embed-v4.html>
8. AWS, *Using Amazon Nova embeddings* and *Complete embeddings request and response schema* — `embeddingPurpose`, `embeddingDimension`, 8k context. <https://docs.aws.amazon.com/nova/latest/userguide/embeddings.html> and <https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html>
9. Semantic Career Mapping Platform, *Experiment 009 — PCA Informativeness* — effective dimensionality and `n_dims=256` in the MiniLM centered frame. [../experiments/009-pca-informativeness.md](../experiments/009-pca-informativeness.md)
10. BAAI, `bge-base-en-v1.5` model card — 768-d, 512 tokens, MTEB table, similarity-distribution warning, query instruction. <https://huggingface.co/BAAI/bge-base-en-v1.5>
11. Qwen, `Qwen3-Embedding-0.6B` model card and paper — 0.6B, 32k context, 1024-d, MRL, 100+ languages. <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B> and <https://arxiv.org/html/2506.05176v3>
12. Microsoft, `harrier-oss-v1-270m` model card — 270M, 640-d, 32,768 tokens, 94 languages, MTEB v2 66.5. <https://huggingface.co/microsoft/harrier-oss-v1-270m>
13. BAAI, `bge-m3` model card — 8,192 tokens, 100+ languages, dense/sparse/multi-vector. <https://huggingface.co/BAAI/bge-m3>
14. Semantic Career Mapping Platform, *Report 023 — Context Engineering for Embeddings* — length collapse, anisotropy, hubness, and representation trade-offs. [023-context-engineering-for-embeddings.md](023-context-engineering-for-embeddings.md)
15. Minish, *Model2Vec MTEB results* — MiniLM MTEB English v1 average 55.93, clustering 44.90. <https://minish.ai/packages/model2vec/results/>
16. AWS Machine Learning Blog, *Get started with Amazon Titan Text Embeddings V2* — 256/512/1024-d, $0.02/M tokens, 3.24% loss at 256-d, MTEB 60.37. <https://aws.amazon.com/blogs/machine-learning/get-started-with-amazon-titan-text-embeddings-v2-a-new-state-of-the-art-embeddings-model-on-amazon-bedrock/>
17. AWS, *Amazon Bedrock Pricing* — embedding model on-demand and batch pricing. <https://aws.amazon.com/bedrock/pricing/>
18. AWS Machine Learning Blog, *Amazon Nova Multimodal Embeddings* — 8k text, 256/384/1024/3072-d, MRL, 200 languages. <https://aws.amazon.com/blogs/aws/amazon-nova-multimodal-embeddings-now-available-in-amazon-bedrock/>
19. AWS, *Models at a glance* — Bedrock embedding model catalog (Titan, Cohere, Nova, TwelveLabs). <https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.md>
20. Persol Cross Technology, *Titan vs Cohere Embed v4 on Japanese retrieval* — Cohere ahead by ~1–1.5 points on a MIRACL-ja subset. <https://note-it.persol-xtech.co.jp/n/n993bf692d19d>
21. AWS, *Supported Regions and models for batch inference* and *Process multiple prompts with batch inference* — batch support and JSONL contract. <https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-supported.html> and <https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html>
22. Ailog, *Embedding Models Leaderboard 2026* — third-party MTEB aggregation including Cohere v4. <https://app.ailog.fr/en/blog/news/embedding-models-leaderboard-2026>
23. MTEB leaderboard / embeddings-benchmark results — official task results and suite definitions. <https://github.com/embeddings-benchmark/results> and <https://huggingface.co/spaces/mteb/leaderboard>
24. Semantic Career Mapping Platform, *Report 019 — AWS S3 Vectors* — frozen index geometry and model/version implications. [019-s3-vectors.md](019-s3-vectors.md)
25. Semantic Career Mapping Platform, *Report 021 — Clustering Metrics at Scale* — why k-NN purity/stability are better suited than centroid indices on embeddings. [021-clustering-metrics-at-scale.md](021-clustering-metrics-at-scale.md)
26. Stochastic Sandbox, *Embeddings in Practice: Every Major Model Compared* (2026-03-31) — third-party MTEB English comparison including BGE-M3 ~63.5. <https://stochasticsandbox.com/posts/embeddings-in-practice-every-major-model-compared-2026-03-31/>
27. LunaPrompts, *Nova Multimodal Embeddings specs & pricing* — reported Nova batch text rate $0.0675/M tokens. <https://lunaprompts.com/models/amazon/nova-multimodal-embeddings>
