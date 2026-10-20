## Why

Monolithic LLM cleaning (Exp 003/008) makes every posting pay the full 8-class boilerplate taxonomy, and eval metrics can't attribute errors to a class — a weak model that over-deletes (A4: 13.2%) or hallucinates gives no signal about *which* instruction failed. The DSML scalable multi-label architecture — N classifiers for N labels ([docs/TODO.md:29](../../docs/TODO.md)) — applied at sentence granularity splits *detection* from *deletion*: calibrated typed decisions (AnyJev-style, [research-reports/007](../../docs/research-reports/007-anyjev-calibrated-classification.md)) mark spans, deterministic code removes them, making verbatim preservation and zero hallucination structural guarantees instead of prompt discipline.

## What Changes

- New structure-aware **span enumerator**: postings enumerate into M units (newlines/markdown first, sentences within blocks) with a byte-exact reassembly invariant — output is provably raw-text-with-deletions.
- **N per-class detectors**, one per boilerplate class of the Exp 008 v3 taxonomy, labeling each span keep/delete/unsure with calibrated per-class confidence (AnyJev L0 rotation readout on a local CPU model; L1 calibration once 100+ labels exist).
- **Deterministic removal step** in Python: span deletion, the header-follows-removed-section rule as code, and an unsure bucket routed to an optional tiny-LLM clause rescue pass (educated only on flagged spans, not the full posting).
- **Experiment 009 benchmark harness** (`scripts/009_sentence_classifier_cleaning/`): arms = monolith v3 prompt (C1 control), classifier-only, classifier+rescue. Metrics: Jaccard vs golden, over-deletion, hallucination (structural check), per-class precision/recall, cost, latency.
- **Golden-set padding prerequisite**: port the 39 curated workpad sections via `pad_to_golden.py` and hand-finalize the eval slice — benchmark conclusions at n=1 are noise.
- **Adoption is gated**: the classifier path replaces/augments the monolith in `cleaned-artifacts` only if it matches or beats C1 quality at lower cost/latency; that integration is a follow-up change.

## Capabilities

### New Capabilities
- `boilerplate-sentence-classification`: span enumeration with byte-exact round-trip, per-class calibrated span labeling (per-class audit trail), deterministic removal with header rule and unsure routing.

### Modified Capabilities
- None. `llm-text-extraction` monolith requirements stand unchanged; the classifier path is out-of-band until Experiment 009 evidence lands.

## Impact

- **Code:** new experiment directory `scripts/009_sentence_classifier_cleaning/` (immutable once run); no `backend/` changes this change.
- **Data:** `data/golden_cleaned.json` padded from the hand-curated 008 workpad (hand-maintained, never auto-generated); span pseudo-labels bootstrapped from cached 008 arm outputs + golden.
- **Deps:** `anyjev` (+ a local Qwen3-class model) added to `requirements.txt` for evaluation — user installs; no GPU deps.
- **Docs:** `docs/experiments/009-*` report after the run, TODO.md item payoff, README "Current Status" update on the adoption decision.
