# Recursive Prompt Optimization for Cluster Separation — Internal Concept Note

> Date: 2026-11-02
> Audience: internal only. This is a working concept note, not a project deliverable. It is deliberately kept out of `docs/` and the MkDocs site — it is not published.
> Context: Semantic Career Mapping Platform — idea capture for automating the cleanup-prompt iteration loop that Experiments 002, 003, 008, and 013 currently run by hand. The production cleaning prompt lives in `backend/pipeline/cleaning/prompts.py` and is keyed by `configuration_hash` (`backend/pipeline/cleaning/configuration.py`).
> Status: Idea draft, not pre-registered. Nothing here is scheduled; it is a design sketch meant to be turned into an experiment document (likely 016) and a `scripts/NNN_*` artifact if approved.
> Scope: pre-embedding text transformation (the cleaning/extraction prompt) with coarse cluster separation as the objective. Encoder choice, clusterer choice, and fine-grained taxonomy clustering are out of scope.

---

## Table of Contents

1. [The Idea in One Paragraph](#1-the-idea-in-one-paragraph)
2. [Why This Fits the Project](#2-why-this-fits-the-project)
3. [Proposed Workflow](#3-proposed-workflow)
4. [The Evaluation Set](#4-the-evaluation-set)
5. [The Objective Function](#5-the-objective-function)
6. [The Proposer](#6-the-proposer)
7. [Search Control](#7-search-control)
8. [Constraint Preservation](#8-constraint-preservation)
9. [Experiment Tracking and Reporting](#9-experiment-tracking-and-reporting)
10. [How This Differs From Standard Prompt Optimizers](#10-how-this-differs-from-standard-prompt-optimizers)
11. [Risks and Threats to Validity](#11-risks-and-threats-to-validity)
12. [Open Questions](#12-open-questions)
13. [Minimal First Version](#13-minimal-first-version)
14. [Suggested Artifact Layout](#14-suggested-artifact-layout)
15. [References](#15-references)

---

## 1. The Idea in One Paragraph

The project's prompt work has always been a manual loop: change the cleaning prompt, re-clean, re-embed, look at the clusters, decide whether it got better. This idea automates that loop. A small, hand-selected set of job postings spanning deliberately distinct job areas (coarse split, for example technical vs. managerial) is cleaned by a candidate prompt, embedded, and clustered. Clustering metrics plus a sample of the embedded text are fed back to an LLM agent, which proposes a revised prompt. The experiment repeats for a bounded number of iterations, and every iteration is recorded and reported with its metrics, its diff, and its rationale. The purpose is to find a prompt that produces broad, well-separated clusters *before* attempting fine-grained clustering on top of it. It is recursive self-improvement of the preprocessing prompt, with the agent improving the prompt and the metrics improving the loop's target — not a model that rewrites itself unsupervised.

## 2. Why This Fits the Project

The manual loop already exists and already works:

| Experiment | Prompt change | Effect |
|---|---|---|
| 002 | regex boilerplate removal | separation gap +0.0203 (3.4×) |
| 003 | LLM extraction (aggressive) | separation gap +0.0493 (8.2×) but over-deletion at scale |
| 008 | boilerplate-only prompt v3 | over-deletion 22.8% → 2.3–5.2%, Jaccard 0.826 → 0.858 |
| 012 | (fixed prompt, encoder comparison) | coarse partitions of 3–4 clusters with AMI 0.21–0.29 |
| 013 (planned) | remove eligibility/location/schedule/travel/physical | not yet run |

Every row is a human editing prompt text and re-running the pipeline. Experiment 012 also showed that the coarse layer is the unstable, low-AMI layer — exactly the layer this loop targets. Automating the edit step is a natural extension of infrastructure that mostly exists: `configuration_hash` already versions prompts, run manifests already record lineage, and the cleaning/embedding stages are parameterized.

## 3. Proposed Workflow

```text
seed prompt P0 (current production boilerplate-only prompt)
    │
    ▼
  clean eval set ──► embed ──► cluster ──► measure ──► report iteration i
                                                 │
                                                 ▼
                                    sample embedded text + metrics
                                                 │
                                                 ▼
                          LLM proposer drafts prompt P(i+1) as a unified diff
                                                 │
                                                 ▼
                          constraint check (mechanical, not LLM)
                              ├─ reject/salvage ──► log rejected proposal, retry or stop
                              └─ pass ──► next iteration
```

Each iteration is one recorded experiment row with:

- `prompt_hash` (existing `configuration_hash` semantics)
- the prompt diff against the previous iteration
- cleaning stats: mean input/output length, over-deletion proxies, hallucination check
- embedding stats: self-retrieval, near-duplicate count, vector norm distribution
- clustering stats: cluster count, noise fraction, size distribution, separation/cohesion
- the proposer's stated rationale for the change
- the measured delta against the previous iteration and against the seed

Termination: fixed iteration cap *and* a metric plateau *and* a hard stop on any constraint violation streak. A bounded budget with an auditable trail is the point; open-ended self-optimization is not.

## 4. The Evaluation Set

The set is small, fixed, hand-selected, and frozen before iteration starts. It is the anchor that makes the loop comparable across prompts — if the set changes, the metric series is void.

Design rules:

- **Deliberately coarse separation.** Two strata to start (`technical` vs. `managerial`), selected so the intended separation is *obvious to a human from the title alone*. If a human cannot tell whether a posting belongs in stratum A or B, it is a bad eval item.
- **Stratum balance.** Equal counts per stratum so chance-adjusted metrics are not distorted by class frequency. Small classes inflate ARI/AMI variance.
- **Size.** Large enough for stable metrics, small enough to re-clean on every iteration cheaply. Experiment 012 used `n=1,000` and still saw high subsample instability (ARI 0.278–0.488), which means small sets will produce noisy iteration-to-iteration deltas. This is the single most important design tension and must be measured, not assumed.
- **Provenance.** Drawn from `data/selected-job-postings/` and the 011 sample, with `source_hash` keys so revisions are detectable and traceable.
- **Holdout.** A reserved slice that the loop never sees and that is scored once per iteration for reporting. Without it, the agent will overfit the prompt to the eval set — the feedback loop makes this very likely, not merely possible.
- **Frozen labels.** The stratum label for each item is hand-assigned and versioned, not LLM-derived. External metrics against a Weak LLM label would make the loop optimize label agreement with a labeler rather than real separation (the circularity problem 012 documents).

### What "good coarse separation" looks like

The first-stage target is not a taxonomy. It is: two clusters, each dominated by one stratum, with low noise and a tight size distribution, stable across seeds. Fine-grained structure is explicitly deferred until that holds.

## 5. The Objective Function

Do not optimize a single scalar. A single score invites the agent to game the visible number — for example by producing a coarse 2-cluster partition that trivially maximizes purity because every point lands in a huge cluster. Use a scored vector with hard constraints and a lexicographic or gate-based ranking instead.

**Hard constraints (any violation disqualifies the iteration):**

- Cleaning did not fail: no empty outputs, no truncated outputs beyond a length floor
- No-hallucination invariant: cleaned text is a subsequence/substructure of the input under the same check Experiment 008 used
- Language retained: output is not emptied of the job content vocabulary
- Output length within a band relative to the input (over-deletion proxy)

**Soft metrics (reported, ranked):**

| Metric | Role | Why |
|---|---|---|
| k-NN purity vs. hand strata (k=15, cosine) | Primary, label-free | Does not depend on clusterer; measures embedding-space separation directly |
| Additional reference: stratum-wise separation gap | Secondary | Already the project's historical proxy metric (002, 003); comparable to prior experiments |
| AMI/ARI vs. hand strata | Secondary | Same endpoints as 012; comparable across iterations |
| Noise fraction (`-1`) | Secondary | High noise likely means the prompt produced heterogeneous text |
| Cluster count and size distribution | Diagnostic | Prevents rewarding degenerate 1–2 cluster solutions |
| Seed stability (mean pairwise ARI over e.g. 3 seeds) | Secondary | A prompt can improve a single run and worsen stability |
| Prompt length / instruction count | Diagnostic, not a metric | Runaway prompt bloat is a known failure mode |

**Ranking rule (to be pre-registered):** prefer iterations that pass all hard constraints and improve primary k-NN purity by more than the bootstrap interval over the previous accepted iteration; require improvements not be reversed in seed stability. Do not pick winners on differences below the interval.

## 6. The Proposer

The proposer is an LLM agent with two inputs: **metrics** and **content**.

- **Metrics alone are not enough.** `AMI` going from 0.21 to 0.24 does not tell the agent *why* the clusters merged; naming a metric never tells an instruction what to change.
- **Content is the signal that matters.** Sample the most confusable items: pairs from opposite strata that are nearest neighbours in embedding space, plus representative sentences from the clusters that merged. The agent recommends an instruction edit (for example "strip a location section that carries the company-type signal", or "do not remove the job-level business context that separates managerial roles").
- **Output must be a diff, not a rewrite.** Require unified-diff output against the candidate prompt. Diffs are reviewable, bounded, and make it obvious when the agent is editing whole sections.
- **Key the cache on the prompt hash.** Re-proposing an identical prompt is a no-op; detect it and stop.

The proposer should be a *separate* model call from the cleaner, and must see only the *seed* text — never the holdout.

## 7. Search Control

This is the difference between a useful loop and a moot one.

- **Bounded iterations.** Fixed cap (for example 8–12). Beyond that, gains are almost certainly overfitting to the eval set.
- **Plateau stop.** Stop when the best primary metric does not improve across *k* consecutive iterations.
- **Repeated-proposal stop.** Stop when the diff is empty or the hash matches a prior iteration.
- **Murky-instruction detection.** Reject proposals that change more than a set number of lines, or that add contradicting rules, without a stated rationale tied to an observed failure.
- **Keep all iterations.** Never discard an intermediate prompt because a later one scored better; the trajectory is the finding, not only the winner.
- **Promotion is manual.** The agent proposes; a human accepts. This is consistent with the project's existing promotion discipline for cleaning variants.

## 8. Constraint Preservation

The loop optimizes separation. It will happily destroy other things it is not told to protect. Explicit constraints:

1. **No fabrication.** The cleaned text must remain a cleaned version of the original — not a summary, not invented responsibilities. This was the central finding of 008.
2. **No content deletion for separation.** The prompt must not delete category-bearing content just because it improves a number. This is the failure mode of 003 (over-deletion at scale).
3. **Do not optimize against an encoder artifact.** B2 vs. A in 012, and the length-collapse finding in Report 023, show that length and format can move metrics without semantic improvement. Include an output-length band as a hard constraint.
4. **Holdout integrity.** The proposer never sees holdout scores.
5. **Single variable per iteration.** Change the prompt, not the encoder and the clusterer at the same time.

## 9. Experiment Tracking and Reporting

Every iteration is a row, and the final report is a table plus a trajectory narrative:

```text
iteration | prompt_hash | Δdiff summary | proposed change rationale | clean stats | k-NN purity | sep gap | AMI | ARI | noise | clusters | seed ARI | holdout k-NN
```

Reporting requirements:

- **Per iteration:** a human-readable diff render and a "why" line from the proposer.
- **Trajectory view:** primary metric per iteration against the seed baseline and the holdout, plus the clean constraints as a separate panel.
- **Rejected proposals are part of the record** — a rejected edit explains a lot about what the agent believed separated the clusters.
- **Final comparison** against at least one manual prompt (the current production prompt, and 013's variant if it exists) so the loop has a real control. A self-optimizing loop that cannot beat an obvious manual prompt is a negative result worth publishing.

This maps cleanly onto the existing infrastructure: `data/selected-job-postings/`, run manifests with `pipeline_version` and `configuration_hash`, and the experiment-report format in `docs/templates/experiment-log.md`.

## 10. How This Differs From Standard Prompt Optimizers

This is not APO on a classifier task. The signal is not output accuracy; it is the geometry of the embedding space after a text transformation. That raises some non-obvious properties:

- The "model" being optimized is a *preprocessing step* whose output feeds a frozen encoder and clusterer; the objective lives two stages downstream.
- Metrics are unlabeled (k-NN purity, separation gap) or use hand labels only on first-stage strata.
- The proposer must reason over confusable pairs and cluster merges, not over per-example errors.
- The primary risk is not underfitting the prompt but **overfitting the prompt to a small eval set and gaming the metric through content deletion** — a kind of prompt-level reward hacking. That is why the constraints in §8 carry more weight than in a standard APO loop.

## 11. Risks and Threats to Validity

- **Metric gaming.** A coarse 2-cluster partition with a large share of points in one cluster maximizes purity. Must be checked explicitly.
- **Small-set noise.** 012's subsample stability (0.29–0.49) suggests small-set metric deltas will have noise comparable to real iteration-to-iteration gains. Bootstrap intervals are mandatory.
- **Eval-set overfitting.** With a fixed small set and an LLM reading per-iteration confusions, the loop will find the set's quirks. Holdout and manual acceptance are the guards.
- **Confounded variables.** The agent might attribute a metric change to its instruction edit when the real driver is length or format. Report length stats every iteration and consider a length-only control edit.
- **Weak-label-free external metrics.** Hand labels on a coarse split are fine; do not extend the loop to fine-grained taxonomy labels without human labels.
- **Prompt artifacts.** Each prompt iteration costs re-clean of the eval set; at scale this is nontrivial. The eval set must stay small and cacheable.
- **Reproducibility.** Prompt hashes, model versions, temperatures, and encoder/clusterer versions must all be in the iteration row.

## 12. Open Questions

1. **Which clusterer for the loop?** EVōC is production for the full corpus, but the finer/coarser choice changes the metric noise a lot. A cheap K-Means may be better for iteration speed; verify the winner with EVōC. Report both if feasible.
2. **Which encoder for the loop?** Arm C (Titan V2) is production, but a local MiniLM is cheaper. A prompt optimized on MiniLM may not transfer to Titan. Either run the loop on the production encoder or co-run both and report.
3. **How coarse is "coarse"?** Two strata, or three (technical / managerial / non-technical-analytical)? Three raises set-size requirements.
4. **How much of the "content of embedded text" should the proposer see?** k nearest neighbours per confusion, cluster medoids, or a stratified sample? Start with nearest-neighbour pairs across opposite strata; they are the most actionable signal.
5. **Does the loop have a real ceiling?** A negative result is still valuable if the loop cannot beat a manual prompt — record it either way.
6. **Does the loop produce transferable prompts?** A prompt that separates technical vs. managerial on the eval set may not generalize to the 9.8k corpus. Check holdout, and if the loop wins, re-validate on the production corpus before promotion.

## 13. Minimal First Version

Deliberately small, to answer the core uncertainty before building infrastructure:

1. Fix the eval set: ~200–400 postings, hand-labelled coarse (2 strata), frozen, with a held-out third for validation.
2. Fix encoder, clusterer, seeds, and metric definitions. Pre-register the ranking rule and the constraints.
3. Run the seed prompt (current production boilerplate-only) to establish the baseline.
4. Run 3–5 iterations with a single proposer, diff-only output, and a hard iteration cap of 5.
5. Report per-iteration metrics + diffs, holdout results, and a comparison against the seed.

If the loop cannot produce a plausible improvement in that envelope, the idea is not worth scaling, and that itself is the finding.

## 14. Suggested Artifact Layout

Follows the immutable-numbered-script convention from `AGENTS.md`; this does not modify any existing experiment script.

```text
scripts/016_prompt_opt_loop/
  build_eval_set.py        # strata selection, source_hash keys, frozen manifest
  clean_variant.py         # reuse cleaning contract; write to iteration-scoped dir
  embed_variant.py         # fixed encoder arm, reused across iterations
  cluster_variant.py       # fixed clusterer + seeds
  metrics.py               # k-NN purity, separation gap, AMI/ARI, stability
  propose_prompt.py        # LLM proposer with diff output + rationale
  check_constraints.py     # no-hallucination, length band, content retention
  run_loop.py              # orchestrator, iteration cap, plateau/repeat stops
  report.py                # per-iteration table + trajectory + diffs into docs/
  data/
    eval_set.json           # stratum labels + source_hash
    holdout.json
    iterations/<i>/{prompt.md,prompt.diff,clean.jsonl,metrics.json,proposal.json}
  tests/                    # synthetic partition fixtures, no real postings
```

`backend/` is not touched until the loop has a positive result; at that point the reusable pieces (constraint checks, metrics) are extracted as a module, per the experiment-immutability rule.

## 15. References

- [Report 023 — Context Engineering for Embeddings](docs/research-reports/023-context-engineering-for-embeddings.md) — length collapse, format effects on embeddings
- [Report 025 — Semantic Clustering Methods](docs/research-reports/025-semantic-clustering-methods.md) — cluster validity measures that can serve as objectives
- [Report 021 — Clustering Metrics at Scale](docs/research-reports/021-clustering-metrics-at-scale.md) — metric behavior at sample sizes
- [Report 004 — LLM Evaluations](docs/research-reports/004-llm-evals-report.md) — error analysis and judge design
- [Experiment 002 — Regex Boilerplate Removal](docs/experiments/002-boilerplate-removal.md)
- [Experiment 003 — LLM Extraction](docs/experiments/003-llm-extraction.md)
- [Experiment 008 — Boilerplate-Only LLM Cleaning](docs/experiments/008-boilerplate-only-cleaning.md)
- [Experiment 012 — EVōC Clustering Validity Bake-Off](docs/experiments/012-evoc-clustering-bakeoff.md) — coarse-layer instability and the circularity caveat
- `backend/pipeline/cleaning/prompts.py`, `backend/pipeline/cleaning/configuration.py` — prompt versioning via `configuration_hash`
- `docs/templates/experiment-log.md` — report format
