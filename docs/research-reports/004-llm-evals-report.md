---
id: "004"
title: "LLM Evaluations"
date: "2026-08-06"
summary: "Synthesized from Airbnb's Eval-Driven Development, Airbnb's From Weeks to a Day, and Hamel Husain's LLM Evals FAQ. Practical guide to error analysis, binary pass/fail evals, and LLM-as-judge."
---
# Practical Guide to LLM Evaluations

> Synthesized from three sources: Airbnb's *Eval-Driven Development*, Airbnb's *From Weeks to a Day*, and Hamel Husain's *LLM Evals FAQ*.

---

## 1. Start with Manual Error Analysis (Do Not Skip This)

- **Review 20–50 real traces manually** before building any automated evaluators. Read outputs yourself. Build intuition.
- **Run 100 prototype examples** through your system (synthetic is fine at this stage), read every output and trace, categorize the mistakes.
- **Appoint a single "benevolent dictator"** (domain expert, PM, or SME) who makes the final call on what "good" means. One voice eliminates annotation deadlocks.
- **Use binary pass/fail**, not 1–5 Likert scales. Likert scales waste time debating 3 vs. 4, require larger sample sizes for statistical significance, and invite annotators to default to the middle. If you need granularity, decompose into separate binary checks (e.g., "fact 1 included," "fact 2 included").

## 2. Perform Structured Error Analysis (Open Coding → Axial Coding)

The most important activity in evals — adapted from qualitative research:

1. **Create a dataset** of representative traces (real user interactions or synthetic).
2. **Open coding**: Annotator(s) review traces and write open-ended notes about any/all failures. Focus on the *first* upstream failure per trace — downstream failures are often cascading.
3. **Axial coding** (the critical step): Categorize open notes into a failure taxonomy. Group similar failures. Count occurrences per category. An LLM *can* help with this grouping after you've done 30–50 manually.
4. **Iterate until theoretical saturation**: ~20 consecutive traces without new failure categories and you can stop (but review ≥100 to start).
- Use **tooling**: build a custom annotation interface (notebook or web app) that shows traces in domain-specific rendering, supports keyboard navigation, and tracks progress. Custom tools produce ~10x faster iteration than generic ones.
- **Re-run error analysis** every 2–4 weeks for mature systems, weekly for new systems, and after every model switch, prompt change, or incident.

## 3. Write Evaluators Only for Failures You've Actually Observed

- **Do not write evaluators for failures you imagine.** LLMs have infinite failure surface area. Build evaluators for the real failure modes that persist after fixing obvious prompt/config bugs.
- **Fix obvious issues first.** Many failures are caused by the model not receiving instructions you never wrote (formatting, length, reasoning style). Fix the prompt, then evaluate what remains.
- This contrasts with Eval-Driven Development (EDD), which advocates pre-defining goals/gates. In practice, do the EDD process for hard constraints you have concrete requirements for, and let error analysis drive everything else.

## 4. Use a Layered Evaluation Stack (Cheapest First)

```
Layer 1: Code-based checks (deterministic, fast, zero LLM cost)
         ├─ JSON schema validation, regex, length bounds
         ├─ Tool call verification
         └─ Binary assertions
         ↓
Layer 2: LLM-as-Judge (nuanced quality assessment)
         └─ Tone, coherence, faithfulness, relevance
         ↓
Layer 3: Human evaluation (ground truth, edge cases)
         └─ 20–100 rows by SMEs; scale only when rubric is rock-solid
```

- **Only use LLM-as-Judge for failures that persist after Layer 1 fixes.** Layer 2 requires 100+ labeled examples and ongoing maintenance — expensive.
- **One evaluator per dimension.** No "God evaluator" that tries to assess everything at once. 3–5 well-calibrated judges beat 20–30 noisy ones.
- **Do NOT use off-the-shelf eval metrics** (helpfulness, coherence, quality scores). They create false confidence and waste time. Measure what matters for *your* product.

## 5. Calibrate Every LLM Judge Against Human Labels

An uncalibrated virtual judge is **worse than no judge** — it gives false confidence.

1. Create a **golden set of 50–100 examples** that includes bad examples (you can't test discernment without them).
2. Run the judge against the golden set.
3. Measure agreement — target **high 80s–90s% agreement** (Cohen's kappa or Krippendorff's alpha). Perfect agreement is impossible; even humans disagree.
4. Analyze disagreements, refine the rubric, add few-shot examples, re-run the loop.
5. **Recalibrate periodically** as failure modes evolve. Judge drift is real (~1% per run).

## 6. Make Evaluation Deterministic

- **Cache aggressively.** Key references by sample ID + reference-generation config. Key judge scores by sample + model output + judge config + metric. This makes partial progress durable (failed runs resume from cache).
- **~75% of LLM-generated references differ across labeling runs** on identical inputs. Without caching, your measurement is noise.
- **>50% of model outputs across candidates are identical strings.** Most re-computation in naive eval loops is wasted.
- **Judge drift: ~1% across runs. Real signal: 1–3%.** If your measurement isn't deterministic, you cannot detect real improvements.

## 7. Fix One Variable at a Time

1. Fix model, vary prompt → narrow candidates with virtual judge
2. Fix prompt, vary model → narrow further
3. Fix both, vary serving config
- At each stage, judge results narrow the candidate pool. Then sharpen judges with samples from top candidates. Evaluators and candidates **sharpen each other until both stabilize.**

## 8. Evaluate the Full System, Not Just the Model

- **For agentic/multi-step systems**: evaluate trajectories (tool calls, reasoning paths, intermediate states), not just final output. A correct answer can mask a broken path.
- **The seams break**: component-level tests in isolation create false confidence. Run end-to-end tests through the full production path.
- **For RAG**: evaluate retrieval and generation separately.
  - Retrieval: traditional IR metrics (Recall@k, Precision@k, MRR).
  - Generation: error analysis → labels → LLM-as-Judge → validate against human labels.
- **For multi-turn conversations**: annotate the *first* upstream failure. Reproduce errors with the simplest possible test case before assuming it needs multi-turn analysis.

## 9. Mirror Evals in Production

- **Pre-production CI**: small (100+ examples), purpose-built datasets with assertion/regression checks. Runs on every change.
- **Production monitoring**: sample live traffic (5% daily), run programmatic checks + judges asynchronously, surface flagged outputs for human review.
- When production monitoring reveals a new failure pattern, add representative examples back to the CI dataset.
- **Guardrails ≠ Evaluators.** Guardrails are synchronous, deterministic, inline checks (regex, schema, keyword blocklists) with very low false-positive rates. Evaluators run asynchronously to measure quality. Do not conflate them.

## 10. Manage the Process as a Team Sport

- **PM + Engineer collaboration**: engineers catch retrieval/tool errors; PMs catch unmet expectations and confusing responses. Review together weekly.
- **Keep a running log**: errors caught, what you learned, the fix, impact avoided. Share monthly. Let results lead the conversation, not methods.
- **Version prompts in Git** alongside application code. Git web UI is accessible enough for non-engineer stakeholders (lawyers and accountants use it).
- **60–80% of development time** goes to error analysis and evaluation. This is normal — not a sign of broken process.

## 11. Be Wary of Common Traps

| Trap | Instead |
|---|---|
| Automating prompt optimization too early | Writing prompts forces you to externalize requirements. Optimizers hill-climb known failures but can't discover new ones. Use LLMs to suggest prompt improvements based on open coding notes, maintaining human review. |
| Outsourcing annotation to third parties | External teams lack product context and tacit knowledge. Build internal capability: benevolent dictator → rubric → measure inter-annotator agreement → iterate. Only outsource purely mechanical tasks. |
| Using generic eval libraries | "All you get is you don't know what they actually do and in the best case they waste your time and in the worst case they create an illusion of confidence." |
| Similarity metrics (ROUGE, BERTScore) for LLM outputs | Not useful for evaluating generated text. Exceptions: useful for debugging/sorting RAG retrieval relevance. |
| Same model for judge and main task | Usually fine — the judge does a different task. Only switch models if you can't achieve good alignment with human labels. |
| 100% eval pass rate | Means your evals aren't challenging enough. 70% pass rate may indicate more meaningful evaluation. |

## 12. Infra Pattern: Same-Day Fix Loops (Advanced)

When evaluation is deterministic and fast enough (Airbnb reduced eval turnaround from weeks to a day), you can implement a same-day fix loop:

1. **Micro adapters**: Small LoRA patches (rank < 50), trained in < 1 hour on 1 GPU, layered on an existing adapter. Fixes one scoped bug.
2. **Two validation gates**: (a) no regression on expert-reviewed domains, (b) high-uncertainty outputs flagged for human review.
3. **Canary deploy with auto-rollback**.
4. **Lifecycle**: fuse co-triggering patches, retrain when accumulated patches approach ~few hundred examples, unload unused patches automatically.

This is advanced infrastructure — build deterministic evaluation first.

---

# Appendix A: Full Source Summaries

## Source 1: Airbnb — Eval-Driven Development: Lessons from Evaluating GenAI at Scale

**Authors:** Rohit Girme | **Published:** July 28, 2026 | **Link:** [Medium](https://medium.com/airbnb-engineering/eval-driven-development-lessons-from-evaluating-genai-at-scale-e817e5ae5788)

### Why GenAI Evaluation is Different
- LLM outputs are non-deterministic; same input can produce different outputs
- "Correct" is subjective — no single right answer
- You often need AI to evaluate AI, introducing its own failure modes
- A single interaction can chain retrieval, reasoning, tool calls, and generation — each can fail independently
- Three failure patterns without a deliberate eval strategy: false confidence, undetected regressions, teams optimizing for the wrong thing

### Five Core Principles of EDD
1. **Define goals and gates upfront** — what are you optimizing for? what must be true before shipping?
2. **Let real errors guide your metrics** — co-develop metrics with cross-functional partners based on observed failures
3. **Keep your evaluator set small and sharp** — 3–5 well-calibrated judges beat 20–30 noisy ones; one evaluator per dimension
4. **Appoint a decision-maker** — final human arbiter on good vs. bad behavior
5. **Collaborate continuously** — product partners regularly answer "Is X better than Y?" and "What's actually wrong?"

### Three Evaluation Methods (Layered)
1. **Programmatic/Heuristic** — deterministic code-based checks (JSON schema, length bounds, regex), first filter before any LLM/human
2. **LLM-as-Judge** — stronger LLM evaluates output against rubric; assesses tone, coherence, faithfulness, relevance
3. **Human Evaluation** — gold standard for ground truth and high-stakes domains

### Judge Calibration
- Start with 50–100 golden examples including bad ones
- Virtual judge run against golden set
- Target high 80s–90s% agreement (Cohen's kappa, Krippendorff's alpha)
- Analyze disagreements, refine rubric, add few-shot examples, iterate
- Recalibrate periodically

### Agentic System Evaluation (Trajectories)
- Evaluate across three layers, not just final output
- Trace data contains: agent type, sub-agent invocations, I/O, tools, metadata
- Use DFS/tree traversal to reconstruct trace
- Verify subagents invoked at right time, correct tools called, scope evaluation to specific agents

### Four-Step EDD Workflow
1. **Explore** — manual testing, read 100 prototype traces, categorize mistakes
2. **Build** — programmatic checks + virtual judges (per dimension) + PM-labeled golden set of 60 including failures
3. **Calibrate & Iterate** — judge agrees with PM 78% → analyze → update rubric + few-shot → 88%. Fix one variable at a time (model, prompt, serving config)
4. **Scale & Monitor** — scale across 5,000 examples, sample 5% live traffic daily for monitoring, weekly PM review

### Infrastructure Architecture (Referenced from Companion Article)
- Layer 1: Diagnostic framing (epistemic vs. aleatoric uncertainty)
- Layer 2: Deterministic evaluation (per-sample caching)
- Layer 3: Micro-adapters for bounded model mutation (LoRA rank<50)
- Layer 4: End-to-end validation at the seams

---

## Source 2: Airbnb — From Weeks to a Day: How We Made LLM Evaluation Fast Enough to Iterate On

**Authors:** Baharak Saberidokht | **Published:** July 14, 2026 | **Link:** [Medium](https://medium.com/airbnb-engineering/from-weeks-to-a-day-how-we-made-llm-evaluation-fast-enough-to-iterate-on-14e2d35198b4)

### Core Thesis
Shipping production LLMs requires fast iteration on something non-deterministic. **Most friction comes from infrastructure challenges, not model quality.** The fixes come from classical software engineering. The seams are where things break — validate the full path, not components in isolation.

### The Four-Layer Dependency Stack
These are a **dependency stack, not a checklist**. Remove one layer and the others degrade.

#### Layer 1: Diagnostic Framing (Name the Noise)
- Separates **epistemic uncertainty** (model/judge lacks knowledge) from **aleatoric uncertainty** (task is ambiguous)
- Four terms: aleatoric, epistemic, rating indeterminacy, dual indeterminacy
- Conflating them produces wrong conclusions (e.g., misclassifying high-entropy responses as hallucinations)
- **~75%** of LLM-generated references differ across labeling runs on identical inputs
- Judge drifts **~1%** across runs; real signal is **1–3%**
- "Meaningful" means surviving perturbation: rotation of judges, versioned metrics, re-stratified samples

#### Layer 2: Deterministic Evaluation Foundation
- **Rejected approaches**: sampling + majority voting (converges to central tendency, not accuracy), Bayesian (same infra as caching without reproducibility)
- **Per-sample cache on two axes**: references (keyed by sample ID + reference config) and judge scores (keyed by sample + output + judge config + metric)
- Identical inputs → cached results. Evaluation becomes deterministic, efficient, comparable.
- >50% of model outputs across candidates are identical strings — most re-computation wasted
- Keyed at experiment level: partial progress durable (job failing at example 8,000 resumes from cache)
- **Reproducibility is a precondition for Layer 3.**

#### Layer 3: Micro Adapters (Bounded Model Mutation)
- Small LoRA patch: **rank < 50** (vs. few-hundred for full retraining)
- Trains in **< 1 hour on one GPU**
- Layered on existing adapter without modifying its weights
- Ships as software hotfix: scoped to one issue, validated behind two gates (no regression on expert domains, high-uncertainty outputs flagged for human review), canary-deployed with auto-rollback
- **Patch lifecycle**: fuse co-triggering patches (CACE principle), retrain on accumulation (~few hundred examples ceiling per Pletenev et al.), unload unused patches automatically
- Only works because Layer 2 makes same-day turnaround possible

#### Layer 4: End-to-End Validation at the Seams
- Small set of representative inputs through the **full production path** (language detection → preprocessing → modeling → serving)
- Traffic-weighted sampling + **deliberate over-representation of tail** (locales, input modalities, historically incident-prone patterns)
- Seeded with **regression cases from prior incidents**
- Stratified across highest-volume traffic segments
- Catches: language detection misclassification, preprocessing truncation, latency spikes from cache-warmth interactions
- Uses same eval framework as Layers 2/3 — deterministic measurement applies end-to-end

### Concrete Numbers
- ~75% of LLM-generated references differ across labeling runs
- Judge drift: ~1% across runs
- Real signal: 1–3%
- >50% model outputs identical across candidates
- Micro adapter rank: <50
- Micro adapter training: <1 hour/1 GPU
- Base model: 8B parameters
- Full retraining: takes days

### Key Insight
"The leverage is still in the boring, well-understood patterns of systems engineering, applied with judgment to where the new failure modes actually live."

---

## Source 3: Hamel Husain & Shreya Shankar — LLM Evals: Everything You Need to Know

**Authors:** Hamel Husain, Shreya Shankar | **Published:** May 28, 2025 (updated July 18, 2026) | **Link:** [hamel.dev](https://hamel.dev/blog/posts/evals-faq/)

### Getting Started
- LLM evals are product-specific — distinct from foundation model benchmarks
- Three-level framework: Unit Tests → Human & Model Eval → A/B Testing
- Eval systems unlock: fine-tuning, data synthesis & curation, debugging
- 60–80% of development time on error analysis and evaluation

### Minimum Viable Evaluation
- Start with error analysis, not infrastructure
- 30 minutes manually reviewing 20–50 LLM outputs after each significant change
- One domain expert as "benevolent dictator" for quality decisions
- Notebook for reviewing traces, or custom annotation interface built with AI assistant

### Error Analysis Process
**The most important activity in evals.** Four steps:
1. **Create dataset** — representative traces (real or synthetic)
2. **Open coding** — write open-ended notes about failures; focus on first upstream failure
3. **Axial coding** — categorize notes into failure taxonomy; **most important step**; count per category; LLM can help after first 30–50 manual
4. **Iterative refinement** — until theoretical saturation (~20 traces without new category; review ≥100)

### Synthetic Data
- Structured approach: define dimensions (e.g., Dietary Restriction × Cuisine Type × Query Complexity)
- Write 20 tuples manually first, then scale with two-step generation (tuples → natural language)
- Five failure scenarios: complex domain-specific content, low-resource languages, unvalidatable outputs, high-stakes domains, underrepresented groups
- Fix obvious problems first before generating synthetic data for them

### Evaluation Design
- **Binary pass/fail over Likert** — forces clearer thinking, more consistent
- **Do not use ready-to-use eval metrics** — create false confidence, waste time
- **Similarity metrics (ROUGE, BERTScore)** — not useful for evaluating LLM outputs; exception: RAG retrieval debugging
- **Same model for main task and eval** — usually fine (judge does different task); switch only if alignment issue
- **Reject eval-driven development** (as starting point) — start with error analysis, write evaluators for discovered failures
- **Build evaluators for persistent failures only** — fix obvious prompt gaps first

### Human Annotation
- Single benevolent dictator over committees for most teams
- PM + engineer collaboration: engineers catch retrieval/tool errors, PMs catch unmet expectations
- Outsourcing usually a big mistake — loses product intuition and tacit knowledge
- Exception: purely mechanical tasks, tasks without product context, hiring external SMEs (not outsourcing)

### Tooling
- **Custom annotation tools** are the single most impactful investment (~10x faster)
- Key features: domain-specific rendering, keyboard navigation, progress indicators, clustering/filtering/search, surface problematic traces
- Four gaps to fill: error analysis pattern discovery, AI-powered assistance, custom evaluators, APIs for custom apps
- **Version prompts in Git** over dedicated prompt management platforms

### Production & CI/CD
- **CI datasets**: small (100+), purpose-built, assertions preferred over LLM-as-judge for cost
- **Production eval**: sample live traces, run asynchronously, track confidence intervals
- **Complementary**: new production failure patterns → added to CI dataset
- **Guardrails ≠ Evaluators**: guardrails are synchronous, deterministic, inline; evaluators are async, measure quality
- Evaluators can auto-correct only if they meet guardrail criteria (fast/cheap, acceptable FP/FN trade-off)

### Domain-Specific Guidance
- **RAG is not dead** — naive vector search may be; use right retrieval strategy for your use case
- **RAG evaluation**: separate retrieval (IR metrics) from generation (LLM-as-judge per Jason Liu's 6 RAG Evals framework)
- **Chunk size**: fixed-output tasks → large chunks; expansive-output tasks → smaller chunks (map-reduce)
- **Multi-turn conversations**: annotate first upstream failure; reproduce with simplest case first
- **Human handoffs**: evaluate handoff quality and rate; failures at handoff boundaries common
- **Agentic workflows**: two-phase (end-to-end task success first, then step-level diagnostics); transition failure matrices
- **Uncertainty/abstention**: balanced set of answerable + unanswerable questions; pass = answer answerable AND refuse unanswerable

---

# Appendix B: Cross-Source Common Threads

## Agreements Across All Three Sources

| Theme | Source |
|---|---|
| **Manual review first, always** | EDD: "look at your data"; Fast Evals: diagnose noise before removing it; Hamel: "30 min reviewing 20–50 outputs" |
| **Cheap checks before expensive ones** | EDD: programmatic → judge → human layered; Hamel: assertions → regex → LLM-judge hierarchy |
| **One evaluator per dimension, no God evaluators** | EDD: "3–5 well-calibrated beats 20–30 noisy"; Hamel: binary pass/fail for each failure mode |
| **Calibrate judges against human labels** | EDD: 50–100 golden examples, 80–90% agreement target; Hamel: 100+ labeled examples |
| **Judges need bad examples to be calibrated** | EDD: golden set must include failures; Fast Evals: seed regression cases from prior incidents |
| **Evaluate the seams, not components in isolation** | EDD: evaluate agent trajectories; Fast Evals: "debt accumulates at the seams"; Hamel: multi-turn, handoffs, full traces |
| **Production eval is essential, not optional** | EDD: 5% live traffic daily; Fast Evals: traffic-weighted + tail sampling; Hamel: CI and prod complementary |
| **Do not use off-the-shelf generic metrics** | EDD: "build for your product's real failure modes"; Hamel: "generic metrics waste time and create false confidence" |
| **Evaluation is a team sport** | EDD: PM + engineer collaboration, weekly review; Hamel: benevolent dictator + engineer/PM partnership |
| **Classical engineering > AI magic** | Fast Evals: "leverage in boring patterns of systems engineering"; Hamel: evaluation is development, not separate |

## Key Tensions

| Issue | Airbnb (Both) | Hamel FAQ |
|---|---|---|
| **Eval-Driven Development** | Advocates EDD (pre-define goals/gates) | "Generally no" — start with error analysis; EDD only for specific constraints |
| **Scoring granularity** | Multi-level calibration (dimensional scoring) | Strictly **binary pass/fail**; rejects Likert scales |
| **Decision-maker** | Team-based with final arbiter | **Single benevolent dictator** for most teams |

## Complementary Coverage

- **Airbnb provides**: infrastructure architecture (4-layer stack: uncertainty → caching → micro-adapters → end-to-end), quantitative measurements (75% reference drift, 1% judge drift, 50%+ identical outputs), same-day fix loops
- **Hamel provides**: qualitative error analysis methodology (open coding → axial coding → saturation), synthetic data generation framework, custom annotation tooling guidance, domain-specific evaluation patterns (RAG, multi-turn, handoffs, agentic workflows, abstention)
