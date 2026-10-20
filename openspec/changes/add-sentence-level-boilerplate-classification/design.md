## Context

Experiment 008 benchmarked monolithic prompt cleaning across 7 arms (5 lightweight models + 2 gpt-5.4-nano controls) and produced: the v3 prompt whose remove-list encodes an 8-class boilerplate taxonomy built from golden-curation evidence, a composite-key append-only LLM cache, and an eval harness (Jaccard / over-deletion / hallucination). Its eval slice so far covers n=1 posting; the golden workpad already holds 39 hand-curated sections awaiting port. Error attribution is the structural gap: a weak arm either over-deletes (A4: 13.2%) or hallucinates (A4: 1) with zero signal about which of the 8 class instructions failed.

The DSML scalable multi-label architecture (N classifiers for N labels — [docs/TODO.md:29](../../docs/TODO.md)) and the AnyJev/Jev research ([research-reports/007](../../docs/research-reports/007-anyjev-calibrated-classification.md)) suggest the split: a **calibrated typed-decision layer detects** (per class, per span, no generation) and **deterministic code deletes**. This mirrors [docs/architecture/design-decisions.md](../../docs/architecture/design-decisions.md): independently testable stages, each changing one variable; and it upgrades the "LLM extraction over regex cleaning" decision one level — instead of a model doing detection *and* text surgery, the model only judges, surgery is code.

## Goals / Non-Goals

**Goals:**
- A sentence-level cleaning variant whose output is raw text minus deleted spans — byte-exact by construction, hallucination structurally impossible.
- Per-class, per-span error attribution (which class instruction fails, on which spans).
- Experiment 009 evidence sufficient for an adopt/reject decision vs the monolith: quality parity or better at lower cost/latency.
- Calibrated per-span decisions with an explicit unsure operator (thresholds chosen on data, not vibes).

**Non-Goals:**
- `backend/` integration into `cleaned-artifacts` — gated follow-up change.
- AnyJev L2 hidden-state heads — deferred (per-question/per-model head friction, per research report recommendation; revisit with stable labels).
- Lake-scale (7,812 postings) throughput engineering — benchmark runs on the golden slice only.
- Modifying Exp 003/008 scripts or caches (immutable artifacts; new dirs only).
- Embedding-space comparisons — quality of *cleaning* only.

## Decisions

### D1. Granularity: structure-first span enumeration with byte-exact round-trip
Split postings on structure first (blank lines, markdown headers, bullet/list markers), then sentence-split inside long blocks; bare fragments ("401(k) matching") become spans unchanged. Each posting yields `spans[1..M]` plus a delimiter map. **Invariant:** rejoining spans with the map reproduces raw input byte-identically; cleaning output is rejoining the kept subset — so `output ⊂ raw` is provable, and the 008 hallucination metric collapses to an invariant check.
*Alternatives:* pure sentence tokenizer (mangles list-heavy postings — fragments without terminal punctuation); token-level tagging (training-heavy, unnecessary at this scale).

### D2. Detection: AnyJev L0 typed decisions, N questions (one per class), local CPU model
Each boilerplate class from the v3 taxonomy becomes one `noul`-style binary question ("Is this span <class> boilerplate? y/n") answered over next-token logits with cyclic rotation readout (position-bias correction, zero labels). Model: a local Qwen3-class small model via `transformers` backend — CPU-first per project constraint; logit readout requires local inference, which the Kilo Gateway does not expose.
*Alternatives:* gateway LLM tagging (no logprobs → no calibration); regex/embedding detectors (Exp 002 lesson: brittle — retained only as a free pre-prune arm if useful); L2 linear heads (deferred, D-NM).

### D3. Prefill budget: posting-level gate before span-level questions
Ask "does this posting contain <class>?" first (1 prefill per class, full text as state); span-level questions run only for classes detected at posting level plus a margin class. Prefill count scales with *actual* boilerplate diversity, not N×M unconditionally.
*Alternative:* always run all N classes per span (blindness-proof but pays N×M always). The gate trades a small recall risk on undetected classes for a large latency win — measured as its own arm variant.

### D4. Removal: deterministic Python owns all cross-class logic
Span deletion, the **header-follows-removed-section rule** (delete a header iff every span until the next header-of-same-level is deleted), unsure routing, and reassembly live in code. This is the one piece of cross-class arbitration the N-classifier decomposition cannot distribute; centralizing it follows the Jev architectural takeaway (control flow and deterministic rules stay in code — research-reports/007 §4).
*Alternative:* per-class cascade passes each re-emitting text (rejected: N× output tokens, drift compounding, no joint header context).

### D5. Unsure policy: keep-by-default + optional rescue pass
Per-class calibrated probability below the delete threshold and above a keep threshold → span goes to an **unsure bucket**. Policy A keeps it (matches v3's "when unsure, keep"); Policy B routes it to a tiny gateway LLM (flash-lite-class) for a clause-level delete/keep call, cached under the 008 composite-key convention in a 009-local cache. Rescue only ever sees flagged spans — a small or empty batch, never the full posting.
*Alternatives:* hard threshold delete (silent false-positive deletion); monolith rescue of whole postings (defeats the cost argument).

### D6. Labels: bootstrap from aligned raw-vs-golden diffs, hand-audited
Golden curation already recorded what was stripped (the workpad *is* the taxonomy evidence). Auto-label KEEP spans by aligning golden text against raw spans difflib-style, DELETE spans by aligning golden-rejected residue; 008 cached arm outputs provide a second teacher for cross-checking (mismatch → review). Hand-correction on a stratified subsample; teacher-LLM agreement is known weak (~0.735, research-reports/007 §5) so golden stays the only eval ground truth — pseudo-labels serve threshold calibration (L1) alone.
*Alternative:* hand-label everything (unaffordable); trust teacher labels fully (circularity).

### D7. Benchmark: 4 arms, 008-style harness, cached controls
Arms: (1) C1 monolith v3 control — reuse 008 outputs where cache keys match, free; (2) classifier-only (keep-by-default unsure policy); (3) classifier + rescue; (4) classifier with posting-level gate off (D3 ablation). Metrics per arm: Jaccard vs golden, over-deletion %, structural hallucination check (kept-span rejoin invariant), per-class precision/recall, per-posting cost, latency. Eval slice = ported golden postings (pad_to_golden, hand-finalized); never auto-generated per AGENTS.md.

## Risks / Trade-offs

- [Span-level granularity can't split mixed sentences (salary clause inside a duties bullet)] → rescue pass handles flagged clauses; measured residual; known limitation of arm 2.
- [Classifier false positives delete silently — no prompt "conscience"] → precision-leaning thresholds, keep-by-default unsure band, per-class P/R surfacing every miss.
- [Posting-level gate (D3) misses an undetected class entirely] → gate-off ablation arm quantifies the blindness cost; adoption decision sees both.
- [AnyJev is weeks-old, Qwen-centric, headline numbers n=300] → L0 readout is training-free and model-portable, version pinned, benchmark is on *our* golden, not their numbers; spike-first framing per research report.
- [Prefill latency on CPU at N×M] → D3 gate + rotation budget (`adaptive_shifts`); golden-slice scale (≤39 postings) is tractable regardless; lake scale is the follow-up change's problem.
- [Golden padding is manual work] → workpad already curated 39 sections; port is mechanical via existing `pad_to_golden.py` (existing entries immutable, diffs flagged).
- [Pseudo-label teacher circularity inflates apparent P/R] → eval only against hand-maintained golden; teachers never generate golden.

## Migration Plan

None — additive experiment. Rollback = adopt nothing; the monolith path is untouched (`llm-text-extraction` spec unchanged, Exp 003/008 caches untouched).

## Open Questions

- L0 readout model choice: Qwen3-1.7B vs Qwen3-4B (report shows L0 gains holding across sizes; latency vs accuracy arm adds evidence).
- Thresholds: one global delete/keep band vs per-class bands selected on a held-out label slice.
- Rescue model: flash-lite-class via gateway vs nano — cost/quality arm.
- Whether the scoped-prompt monolith arm (detected classes deterministically composing the system prompt, from the earlier design discussion) belongs in 009 or its own experiment — depends on arm count vs harness effort.
- Class taxonomy freeze: reuse v3's 8 classes verbatim, or merge the two recruiting-adjacent classes first (affects label schema stability for L1 calibration).
