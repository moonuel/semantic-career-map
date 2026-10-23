# spaCy & spaCy-LLM — Rule-Based NLP and LLM Components in Structured Pipelines

> Date: 2026-10-22
> Context: Semantic Career Mapping Platform — research into spaCy (already a declared dependency in `pyproject.toml`, v3.8.16 in `.venv` but unused so far) and `spacy-llm` (v0.7.4, not installed). Two adoption questions for this project: (a) does spaCy add anything to the preprocessing/cleaning stage now being productionized in `backend/pipeline/cleaning/` (change `add-cleaning-and-eval-layer`) beyond the already-planned `PhraseMatcher` skill extraction (`docs/technical/embedding-pipeline.md:33-41`), and (b) should the hand-rolled LLM-calling layer (Kilo Gateway client, task 3.1) sit on top of spaCy-LLM instead. Research only — per repository rules, nothing is installed as part of this report.

---

## Table of Contents

1. [Executive Conclusion (First)](#1-executive-conclusion-first)
2. [spaCy: The 2026 State of the Library](#2-spacy-the-2026-state-of-the-library)
3. [The spaCy Toolbelt Relevant to This Project](#3-the-spacy-toolbelt-relevant-to-this-project)
4. [spaCy-LLM: Design, Tasks, Models, Operations](#4-spacy-llm-design-tasks-models-operations)
5. [Reality Check: Model Registry, Maintenance Posture, and Risk](#5-reality-check-model-registry-maintenance-posture-and-risk)
6. [Dependency and Stack Fit](#6-dependency-and-stack-fit)
7. [Relevance Map: Where Each Library Fits Here](#7-relevance-map-where-each-library-fits-here)
8. [Recommended Adoption Path](#8-recommended-adoption-path)
9. [Evaluation Notes for a Future Experiment](#9-evaluation-notes-for-a-future-experiment)
10. [Sources](#10-sources)

---

## 1. Executive Conclusion (First)

| Question | Verdict | Why |
|---|---|---|
| Adopt plain spaCy for the planned skill-gazetteer step? | **Yes — zero new cost** | `spacy==3.8.16` is already in `pyproject.toml` and installed; `PhraseMatcher` over the planned ~200-term vocabulary (technical/embedding-pipeline.md:33-41) needs no model package, no torch, no GPU — tokenizer-only operation |
| Adopt spaCy *rule-based* cleaning/normalization ahead of the LLM stage? | **No for boilerplate semantics; yes opportunistically for structure** | Experiment 002 already measured rule-based cleaning: separation gap +0.0203 vs LLM's +0.0493 (exp 003). Deterministic token/regex rules belong in `backend/pipeline/cleaning/normalization.py` (already landed, task 3.2); spaCy adds nothing decisive there |
| Adopt spaCy-LLM as the LLM framework for the productionized cleaning layer? | **No — YAGNI** | Capabilities we already shipped and tested (retry, cache, versioned prompts, sharding-adjacent) would be exchanged for an *experimental*, slow-moving dependency whose provider registry is frozen in 2023-era models (Claude ≤2.1, Google = PaLM `text-bison`) and does not speak our Kilo Gateway |
| Revisit spaCy-LLM later? | **Yes, for two future jobs** | (1) NER/SpanCat-style structured extraction (skills/requirements spans, ESCO-linked taxonomy — report 016) where its `CandidateSelector` KB machinery and CoT `SpanCat.v3` fit; (2) whenever prompt/pipeline versioning wants spaCy's config system. An explicit head-to-head experiment is sketched in §9 |

---

## 2. spaCy: The 2026 State of the Library

- **What it is:** Industrial-strength NLP in Python (Explosion, MIT license) — a tokenizer-first pipeline architecture (tokenize, then `Doc`-processing *pipes*), a shared `Vocab`/`StringStore` that interns strings for memory-efficiency, and an INI-style config system that makes every pipeline serializable and reproducible from a `.cfg` file. It remains the de-facto standard for practical CPU NLP: **~22M downloads/month**, first release 2015, 218 releases (`pypi.org/project/spacy`).
- **Current release: v3.8.16** (24 Aug 2026). The recent release history is a technical-health signal, and it has been genuinely bumpy:
  - **v3.8.7** (May 2025): Cython 3 compile for Python 3.13 support; factories and registry entries centralized (`spacy.pipeline.factories`, `spacy.registrations`) to remove import-time side effects.
  - **v3.8.8** (Nov 2025): dependency-hygiene pass (`typer-slim` to shrink the footprint), Python 3.9 dropped.
  - **v3.8.12** (Mar 2026): the headline event — **Pydantic v2 migration**. spaCy's config system (`confection`) had entangled Pydantic v1 behaviors for years; 3.8.12 ships confection 1.3 + Thinc 8.3.13 with custom validation in place of Pydantic, unlocking **full Python 3.14 support**. (3.8.13 re-pinned confection after the 3.8.12 dependency-dance broke model loading — a sign the Pydantic-v1 legacy surface was bigger than it looked.)
  - **v3.8.16** (Aug 2026): publishing-workflow housekeeping only.
- **Python support:** 3.8 line ships CPython wheels for 3.10–3.14 (macOS x86/ARM, manylinux/musllinux x86/ARM, Windows x86/ARM — no GPU wheels involved, this is a CPU library at its core). This project runs Python 3.14 — matching today.
- **Maturity signal:** spaCy 4.0 exists only as stale `4.0.0.dev` pre-releases (last touched Apr 2024, "nothing published"). The library is in steady **3.8 maintenance mode** — stable, but architecturally not moving fast.
- **Model packages:** `en_core_web_sm/md/lg/trf` (roughly 12 MB / 40 MB / 560 MB / 440 MB) bundle tokenizer + tagger, parser, NER, lemmatizer, attribute ruler etc. Because of the 3.8 packaging change (`--no-require-parent` on v3.8 models), model packages no longer hard-pin spaCy. Model downloads come via `spacy download` / `spacy.load` — model bytes are large, but only needed when the task wants statistical components. 3.8.14 specifically fixed model downloads in pip-less environments (relevant to containers using `uv`).

---

## 3. The spaCy Toolbelt Relevant to This Project

Everything below works on a `spacy.blank("en")` pipeline (tokenizer only, no model download), unless noted — CPU cost microseconds-to-low-milliseconds per posting.

| Component | What it does | Fit for this project |
|---|---|---|
| `PhraseMatcher` | Multi-term dictionary matching over token spans; stores reusable `Doc` vocab entries, so a 200-term vocabulary costs effectively nothing at runtime | **The already-planned skill extraction** (embedding-pipeline.md:33-41). Fits the section-weighting / augmentation plans (report 001 §137-179). Deterministic, auditable, cacheable — the *safest* possible extraction layer |
| `Matcher` | Token-attribute patterns (POS, regex-on-token, casing) — richer than phrase lists, brittle to rephrasing | Spot boilerplate *bookends* ("About X", "Why join us") as diagnostics only — exp 002 already lost the rules-vs-LLM fight for actual cleaning |
| `EntityRuler` / `SpanRuler` | Rule-based / pattern-driven NER producing `doc.ents` (`EntityRuler`) or `doc.spans` (`SpanRuler`) | Gazetteer skill extraction **with entity semantics** (ESCO-consistent labels, report 016) rather than bare term lists; the upgrade path if counts/IDs per skill are wanted |
| `Sentencizer` / `sbd` | Sentence boundaries without a model | Cheap section-ish splitting for chunked embedding — mostly superseded here by the heading-aware normalization in task 3.2 |
| `Lexeme`/`StringStore`-interned normalization | Custom attributes, cached vocab | Cheap continuous vocabulary for the duplicate-taxonomy ideas (report 017) — pattern at token level |
| `en_core_web_sm` (statistical) | Real NER (ORG/GPE/DATE/MONEY), morphology, dependency parse | **Metadata skimmer** over postings (employer mentions, years-of-experience DATE/MONEY spans) as *evaluation evidence*, not as pipeline stage — reversible, one model download (~12 MB) |
| Dependency-parsing / coref (trainable `CoreferenceResolver`, etc.) | Heavyweight structure | CPU cost per posting is real (sm parse ~ tens of ms; lg ~100-300 ms); not justified at 9,316-posting scale unless a specific downstream task needs it |

Two architectural notes for how this project should touch spaCy if it does:

1. **`nlp.pipe(docs, batch_size=…)`** is the natural batching harness; a tokenizer-only pipeline processes postings at tens-of-thousands-of-tokens/sec on one CPU core, i.e., the full 9,316-posting lake in seconds-to-a-minute — no service, no async.
2. spaCy's **config system** is the quiet star: a pipeline defined in a `.cfg` is byte-serializable — which is the same *shape* as this project's `configuration_hash` (SHA-256 over model + system prompt + user template, task 3.3), except spaCy does it with naming/versioning conventions instead of a raw digest.

---

## 4. spaCy-LLM: Design, Tasks, Models, Operations

`spacy-llm` (Explosion, MIT) integrates LLMs as ordinary spaCy pipeline components (current **v0.7.4**, 24 Mar 2026; requires `spacy>=3.5,<4.0`, `jinja2`, `confection>=1.3.3`, Python ≥3.10).

### 4.1 Architecture: task + model + wrapper

An `llm` component is an `LLMWrapper` with two pluggable halves (docs: spacy.io/api/large-language-models):

- **Task** — owns *prompting and parsing*: (1) load a Jinja prompt template and inject doc data (+ optional few-shot examples read via `spacy.FewShotReader.v1` from YAML/JSON/JSONL), (2) shard the prompt map-reduce-style if it exceeds the model's context window, (3) parse responses back into structured `Doc` annotations (`doc.ents`, `doc.spans`, `doc.cats`, custom `Doc._.` fields) with validation. Tasks are versioned and registered in `llm_tasks` (`spacy.NER.v3`, `spacy.TextCat.v3`, …).
- **Model** — owns *transport*: REST backends for hosted APIs, HuggingFace backends for self-hosted models, or a LangChain bridge (LangChain `>=1.0` since 0.7.4).
- **Wrapper** — a serializable spaCy pipe (`to_disk`/`from_bytes`), with `save_io=True` persisting prompt+response per doc into `Doc._.llm_io` — prompt/response provenance attached to the document itself.

Defaults are OpenAI GPT-3.5 unless configured otherwise.

### 4.2 Built-in tasks (v0.7.4)

| Task (registry name) | Mechanism / notes | Project relevance |
|---|---|---|
| `spacy.NER.v3` | Chain-of-thought NER from PromptNER (Ashok & Lipton 2023); Explosion reports up to **+15 F-points vs v1/v2**; positive+negative examples with `is_entity`/`reason`; `label_definitions` + freeform `description` injected into the prompt | Skill/requirement-span extraction, if we ever go span-shaped (report 016; exp 006's taxonomy labeling is currently LLM-classification, not spans) |
| `spacy.SpanCat.v3` | NER-v3 adapted for overlapping spans into `doc.spans` | Multi-skill overlaps, ESCO skill headings vs concepts |
| `spacy.TextCat.v3` | Classification with optional label definitions; `exclusive_classes`, `allow_none`, label normalizer | Role-labeling analog of exp 006 — but our taxonomy labeling already runs through its own client + eval |
| `spacy.REL.v1` | Relation extraction between entities from an upstream NER component | Emphasis: relations between skills/prereqs — speculative ("soon: Semantic role labeling" has been in the README since 2023) |
| `spacy.EntityLinker.v1` + `spacy.CandidateSelector.v1` | LLM picks the best candidate for each mention from a KB (`InMemoryLookupKB` from a spaCy pipeline or YAML; `top_n` candidates in prompt; entity descriptions via CSV/`desc_path`) | **The interesting one for us**: ESCO (13,939 skill concepts + descriptions, report 016) dropped in as a `CandidateSelector` KB — LLM-guarded entity linking for skill normalization |
| `spacy.Lemma.v1`, `spacy.Sentiment.v1`, `spacy.Summarization.v1`, `spacy.Translation.v1` | Utility tasks writing into `Doc._.` fields | Not needed — this project cleans and extracts, it doesn't generate |
| `spacy.Raw.v1` | No scaffold — model reply to doc content into `doc._.reply` | **Could host the boilerplate-only prompt** (tasks 3.1/008) — but then we'd still write our own parse+eval, i.e., the parts that are ours anyway |
| `spacy.NoOp.v1` | Test-only | — |

### 4.3 Operational knobs (the parts that overlap with task 3.1)

- **Retries/timeouts** (all REST models): `max_tries` (default 5), `interval` (1 s between tries), `max_request_time` (30 s), `strict` (raise vs. pass through malformed responses). This is coarser than `backend/pipeline/cleaning/`'s client (429/5xx-specific retry, empty-response RuntimeError) — and note: spaCy-LLM's REST models don't distinguish rate-limit vs server-error classes.
- **Caching**: a `Cache` object; the shipped file-backed implementation `BatchCache` takes `cache_dir` (None → NoOp — the component then just re-calls the API), `batch_size` (64 docs per persisted batch file), `max_batches_in_mem` (4). Keyed on the prompt+response pair per doc/batch. Compare our 3.5 cache: Parquet rows in the lake keyed `(source_hash, configuration_hash)` — the cache *is* the dataset, versioned like every artifact, and therefore survives prompts that change hash-when-changed. spaCy-LLM's cache is process-local and *unversioned*.
- **Sharding** (map-reduce, v0.7.0): for prompts longer than the context window, `ShardingLLMTask` tasks split the doc (token or sentence strategy) while keeping the prompt-templates intact, then fuse shards per task semantics ("prompt sharding introduces potential inaccuracies, as the LLM won't have access to the entire document at once"). For long postings (band 5,000–15,000 chars in the re-baselined lake, exp 009) this is the framework-y answer to keeping whole postings in context. Modern Gemini models make this ~moot for 15k chars (~4k tokens) — worth stating: we picked models whose window swallows whole postings, so we never needed sharding.
- **Custom endpoints** (0.7.0+) — documented `endpoint`/`base_url` override per model, explicitly "to support usage of fine-tuned models". Kilo Gateway is not a registered provider: reaching Gemini through spaCy-LLM means either an OpenAI-compatible base_url shim or a custom `@llm_models` registration — both are supported, neither is off-the-shelf.

### 4.4 Configuration-style example

```ini
[nlp]
lang = "en"
pipeline = ["llm"]

[components.llm]
factory = "llm"

[components.llm.task]
@llm_tasks = "spacy.TextCat.v3"
labels = ["DATA_SCIENCE", "ML_ENGINEER", "AI_ENGINEER"]

[components.llm.model]
@llm_models = "spacy.GPT-4.v3"   # provider registry is the catch — see §5
```

Assemble via `from spacy_llm.util import assemble; nlp = assemble("config.cfg")`. This is real value: prompts, few-shot files, retries and caches compose into one serializable artifact — with an explicit migration-guide discipline (0.3→0.4 model-centric refactor, 0.4→0.5 CoT-NER) that mirrors what `configuration_hash` does for us, minus the hash.

---

## 5. Reality Check: Model Registry, Maintenance Posture, and Risk

- **The model registry is a museum.** The REST table (live docs, 2026) offers `spacy.GPT-4.v1–v3`, `spacy.GPT-3-5.v1–v3`, a menagerie of retired `davinci`-family names, `spacy.Azure.v1`, `spacy.Command.v1` (Cohere), Anthropic names ending at **`spacy.Claude-2-1.v1`**, and Google represented by **`spacy.PaLM.v1` (`text-bison-001`/`chat-bison-001`)**. The v0.7.0 release note ("new tasks, map-reduce, custom endpoints-for-fine-tuned-models") was 19 Jan 2024; the v0.6.0-era work per provider is on 2023 timelines. Nothing in the registry speaks Gemini, Claude 3.x, o-series, or DeepSeek (exp 004's DeepSeek benchmark, exp 007's latency benchmark — outside spaCy-LLM's world entirely).
- **Cadence is the second red flag.** Release gaps: 0.6.4 (Nov 2023) → 0.7.0 (Jan 2024) → 0.7.3 (Jan 2025) → 0.7.4 (Mar 2026). Snyk's package-health score rates maintenance **"Inactive"** (health 62/100, latest 0.7.4). The 0.7.4 release is honest about why: its work is Pydantic v2 + Python 3.14 life-support across the tree — keeping `spacy-llm` compatible with modern spaCy, not reshaping the framework around the 2026 LLM ecosystem.

  (Accuracy note: spaCy's README has claimed since 2023 that "spacy-llm will be installed automatically in future spaCy versions" — it still hasn't happened by spaCy 3.8.16 / 2026.)

- **API status: still officially experimental.** The README warning survives every release: *"...it is possible that changes made to the interface will be breaking in minor version updates."* The library has broken configs 0.3→0.4→0.5 and has since settled into a patch-style maintenance cadence rather than feature development.
- **Where it would still win (be honest about this).** For a project *starting from zero* on NER/SpanCat/TextCat that wants config-driven, serializable, cached LLM components plus a mature KB entity-linking story, spaCy-LLM removes a meaningful amount of boilerplate. Explosion's own motivation section frames it exactly right: prototype with LLM prompts, then replace components with supervised models as production hardens. That graduation ladder (LLM → train a supervised component on accumulated LLM annotations) is the one capability we cannot cheaply rebuild — and it is *not yet relevant* here: our `backend/pipeline/cleaning/` module has already outgrown what the framework's prototype-grade caching/parsing layer offers, and has not yet reached the supervised-training end where the ladder would pay off.
- **What it can't do for us:** no structured span product beyond exp 003's cleaning exists yet (so the flagship tasks are unused), self-hosted HF models are out (CPU-first), and provider/model choice is centralized (Kilo Gateway, exp 007 latency/cost discipline). Every one of those pulls **away** from the framework, and replacing the 3.5 cache (dataset-as-cache, `(source_hash, configuration_hash)`-keyed, lineage-linked) with its own prompt-keyed `BatchCache` would be a step backwards.

---

## 6. Dependency and Stack Fit

- **`spacy` is already in** (`pyproject.toml`, installed v3.8.16) and currently *imported by nothing* in `backend/` — so the whole §3 toolbox is a free-pull and a two-line pipeline away.
- **`spacy-llm` adds on top of an existing spaCy:** `jinja2` + `confection` (+ negligible). Hard requirement `spacy>=3.5,<4.0` is already satisfied at 3.8.x. Python ≥3.10 ✔ (we run 3.14).
- **Hugging Face path avoidance:** running local models through spaCy-LLM requires the `transformers` extra (torch/transformers/einops/xformers pins) — rejected here as violating CPU-first. Other useful point: the project's `sentence-transformers` dependency already carrying torch/transformers debts means the *check-the-tree* answer would have been "no additional heavy packages", but self-hosted mini-LLMs (Dolly-3B-class, Mistral-7B-class) are CPU-infeasible for a cleaning batch job anyway — moot.
- `pydantic` interplay: spacy-llm 0.7.4 is Pydantic-v2-native; the project's FastAPI stack is Pydantic-v2 → no mixed-Pydantic warning noise (pre-0.7.0 code emitted the v1-deprecation warning on Python 3.14 — fixed in 0.7.4).
- **Compliance note:** nothing was installed for this report, per AGENTS.md — every claim above is about already-published wheels and released source, verified against their docs.

---

## 7. Relevance Map: Where Each Library Fits Here

| Project surface | Current / planned mechanism (canonical doc) | Role of spaCy / spaCy-LLM |
|---|---|---|
| HTML→text normalization (3.2, `backend/pipeline/cleaning/normalization.py`) | Deterministic 2-round unescape → tag strip; `<li>` → bullets; double-encoded entities resolved | Nothing. spaCy operates on linguistic tokens, not raw bytes — the wrong tool for tolerant-HTML repair |
| Cleaning stage (exp 003, 008; tasks 3.1–3.7) | LLM boilerplate-only prompt, Kilo Gateway, flush writers, cache-as-dataset, eval on `data/golden_cleaned.json` (Jaccard, boilerplate patterns, hallucination, over-deletion) | **Keep our layer.** §5 verdict. A spaCy-LLM re-host costs migration work and a prompt-cache/model-registry mismatch, and produces no new capability |
| Pre-embedding / section weighting (report 001; technical/embedding-pipeline.md) | Planned segmentation + weighted concatenation | spaCy tokenizer + sentence boundaries are usable *deterministic* input to a section splitter — lighter than heading heuristics, heavier than `re`-based ones; use only for splitting and never for semantic removal decisions |
| Skill extraction (planned; ~200-term vocab) | `PhraseMatcher` over curated vocabulary | **Primary adoption point** (§8). Deterministic list only, deterministic source, no LLM leakage |
| Span/rule skill extraction and ESCO mapping (report 016, exp 006) | LLM role labeling through project client; ESCO as download+CSV data | spaCy `EntityRuler` upgrades the gazetteer baseline to entity-semantic spans; spaCy-LLM `EntityLinker.v1` + `CandidateSelector.v1` over ESCO concepts is the best-modeled future ESCO-mapping option of any framework surveyed so far — revisit per §9 |
| Role labeling (exp 006) | LLM classification (hand-rolled, eval'd) | `TextCat.v3` adds nothing; role labeling stays in our own client + eval harness — not a reopen trigger |
| Dedup / near-duplicate detection (report 017) | Hash keys, TF-IDF, RapidFuzz layers (no NLP pipeline) | Not applicable. A side note: spaCy's token-interned vocab would make a *vocabulary-shape* audit of the lake (how much surface text is shared boilerplate) cheap if ever wanted — nice-to-have, not on the critical path |
| Latency/cost discipline (exp 007) | Hand-rolled client; per-flush usage/cost/latency aggregation, cost from price tables | spaCy-LLM doesn't surface usage/cost accounting anywhere near the run-record latencies in 3.4/3.6 — another confirm-keep |

---

## 8. Recommended Adoption Path

1. **Now (zero-cost):** use spaCy (blank pipeline + `PhraseMatcher`) for the planned skill-vocabulary extraction step. It is already a dependency; the 200-term vocabulary in `embedding-pipeline.md` slots directly into `PhraseMatcher(vocab, attr="LOWER")` with `doc.spans` output persisted to the lake as one processing layer. Because no model weight is required, CPU posture and Docker image are unchanged; and no install happens — the wheel is present.
2. **Don't:** do not re-host the productionized cleaning stage on spaCy-LLM, and do not add rule-based *semantic* cleaning via spaCy in front of the LLM (exp 002 already lost that comparison; the boilerplate boundary is a semantics question, not a pattern question).
3. **Keep an eye on:** spaCy-LLM 0.8.x and any spaCy 4.0 development. The two-minute check is its release cadence and whether any Gemini/Claude-3.x/Anthropic/gpt-4o-class entry ever appears in the `llm_models` registry — until the registry modernizes, the framework's provider story lags where our actual calls happen (1 Gateway).
4. **Trigger points to re-open** (any one suffices):
   - A change needs **structured span extraction with a KB** (skill→ESCO linking as a pipeline stage, §4.2) — spaCy-LLM's `EntityLinker`/`SpanCat.v3` is then a genuinely differentiated toolset, and DSL/kb machinery that would otherwise be hand-rolled.
   - Prompt **versioning beyond a hash** becomes a requirement (per-task few-shot registries, config-phase migrations) — spaCy's config system + `spacy-llm` configs are a *cheap* compliance story here.
   - The accumulation story arrives: we want to bootstrap a supervised skill/tagger model from LLM-annotated postings (the exact "prototype with prompts, graduate to trained pipes" ladder spaCy-LLM exists to walk).

---

## 9. Evaluation Notes for a Future Experiment

Sketch for **experiment 010** (not planned yet — only the shape, per the repo policy that research notes experiments):

- **Question:** does spaCy-LLM's `spacy.SpanCat.v3` (CoT prompting, `label_definitions`, few-shot with negative examples) beat our hand-rolled prompt at extracting *skill spans* on hand-maintained `data/golden_set.json`?
- **Set-up constraints from this research:** if attempted, run it against the *real* provider through a custom `@llm_models` registration; set `strict=False` so malformed responses become row-level eval failures, not crashes; set `cache_dir` only into a scratch dir — never treat spaCy-LLM's `BatchCache` as authoritative (its results are prompt-string-keyed, not `source_hash`×`configuration_hash`-keyed, and thus not lineage-linked to the lake).
- **Metrics:** reuse the LLM-extraction evaluation suite definitions (exp 003 / task 4.2: span-level Jaccard vs golden entities, hallucination word-set-difference vs raw text, over-deletion ratio flags) — spaCy-LLM's task `scorer` slots accept custom scorers, so our definitions can be handed in verbatim instead of rewritten.
- **Cheap first probe (no spaCy-LLM at all):** run `en_core_web_sm` over a sample of postings and print `doc.ents` at ORG/DATE/GPE/MONEY — a 10-minute glance at whether statistical NER span *labels* on postings align with the golden postings' structure. This sets the baseline any LLM-span candidate would have to beat for free.
- **Sizing:** the corpus is 9,316 postings and the labelled span surface is small; a spaCy-LLM trial would be an experiment, never a production write-path — keep it inside a new numbered `scripts/0NN_spacy_llm_probe/` directory, untouched after the run (per the immutable-artifact rules in AGENTS.md), with its report in `docs/experiments/`.

---

## 10. Sources

- spaCy usage + install documentation — spacy.io/usage (26 Aug 2026 state: "Version 3.8 out now!"; pip/conda as the standard installers).
- spaCy PyPI — pypi.org/project/spacy (v3.8.16, MIT, `>=3.9,<3.15`).
- spaCy release notes — github.com/explosion/spaCy/releases: v3.8.7 (Cython 3 / Py3.13, registry refactor), v3.8.8 (drop 3.9, `typer-slim`), v3.8.12 (confection 1.3 + Thinc 8.3.13, Pydantic deregistration story, Py3.14), v3.8.13 (confection re-pin), v3.8.14 (pip-less model download fix), v3.8.16 (latest, 2026-08-24; wheel matrix CPython 3.10–3.14).
- spaCy API docs — spacy.io/api/large-language-models (LLMWrapper, task protocol incl. ShardingLLMTask/NonShardingLLMTask; model REST args `max_tries`/`max_request_time`/`interval`/`strict`; `BatchCache` `cache_dir`/`batch_size`=64/`max_batches_in_mem`=4; NER v3 = PromptNER CoT, +15 F-points internal claim; `EntityLinker.v1` + `CandidateSelector.v1` + `InMemoryLookupKB`).
- spaCy-LLM PyPI — pypi.org/project/spacy-llm (v0.7.4: Python ≥3.10; core deps `spacy>=3.5,<4.0`, `jinja2`, `confection>=1.3.3`; extras `langchain>=1.0`, HF/transformers stack for `[transformers]`).
- spaCy-LLM releases — github.com/explosion/spacy-llm/releases: v0.7.0 (map-reduce doc sharding, EntityLinking/Translation/Raw tasks, custom endpoints for fine-tuned models), v0.7.4 (2026-03-24: Python 3.14, Pydantic v2, LangChain ≥1.0, Conda CI); maintenance-lead notes at rmitsch's org (explosion).
- Snyk PyPI health — security.snyk.io/package/pip/spacy-llm ("maintenance Inactive", health 62/100, latest 0.7.4; version history table: 0.7.3 → 2025-01-13, 0.7.4 → 2026-03-24).
- Ashok & Lipton — *PromptNER: A Zero/Few-Shot Named Entity Recognition* (arXiv:2305.15453, 2023) — Chain-of-Thought prompting basis for spacy-llm NER/SpanCat v3.
- Explosion — *Effortlessly integrate LLMs into spaCy pipelines* (explosion.ai/blog/spacy-llm, 2023) — prototyping-vs-supervised motivation, active LLM-component thesis.
- Project-local: `docs/technical/embedding-pipeline.md` (planned spaCy `PhraseMatcher` skill extraction), `docs/technical/tech-choices.md` (LLM-based boilerplate-removal rationale), `docs/experiments/002-boilerplate-removal.md` / `003-llm-extraction.md` / `006-taxonomy-role-labeling.md` / `007-llm-latency-benchmark.md`, `docs/research-reports/016-esco-eu-jobs-taxonomy.md` (KB candidate for entity linking), `docs/research-reports/017-near-duplicate-text-detection.md`, `openspec/changes/add-cleaning-and-eval-layer/tasks.md` (productionized 3.1–3.7 layer this report weighs spaCy-LLM against).
