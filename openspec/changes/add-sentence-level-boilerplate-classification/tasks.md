## 1. Golden slice prerequisite

- [ ] 1.1 Port the 39 curated workpad sections into `data/golden_cleaned.json`: run `scripts/008_boilerplate_only_cleaning/pad_to_golden.py` (`--dry` first, review skip/differs report, then `--write`). Verification: dry-run report shows expected skips only; golden entry count matches the pad.
- [ ] 1.2 Hand-finalize the Eval 009 golden slice (user diff pass over ported entries; existing signed-off entries stay immutable). Verification: user sign-off recorded on the pad/workpad; slice count ≥ 10 postings recorded in the run config.

## 2. Span enumerator (`scripts/009_sentence_classifier_cleaning/spanify.py`)

- [ ] 2.1 Implement structure-first enumeration (blank lines, headers, list markers → sentences inside long blocks) with delimiter map and `reassemble(spans) == raw` byte-exact round-trip. Verification: `pytest tests/` round-trip identity passes on synthetic fixtures and on every raw posting in the golden slice.
- [ ] 2.2 Fixture set for fragment edge cases (bullets without terminal punctuation, headers, mixed indent) under `tests/fixtures/` (synthetic, < 10KB). Verification: `pytest tests/test_spanify.py -v` green.

## 3. Label bootstrap

- [ ] 3.1 Auto-label spans by aligning golden text against raw spans (`difflib`): KEEP where golden contains the span, DELETE where golden omits it; write per-posting label drafts to `scripts/009.../data/009_label_drafts.json`. Verification: alignment covers ≥ 95% of spans per posting; label counts printed and sane.
- [ ] 3.2 Cross-check drafts against cached 008 arm outputs (reuse `.008_llm_cache.json` read-only); emit mismatch report. Verification: every draft label carries a 008-agreement flag; mismatches listed.
- [ ] 3.3 Hand-audit a stratified subsample of drafted labels (workpad style); correct in place. Verification: audit annotations recorded; corrected labels become the threshold-calibration slice per design D6.

## 4. Classifier layer (AnyJev L0, local CPU model)

- [ ] 4.1 Add `anyjev` and the chosen local Qwen3-class model to `requirements.txt` (user installs manually per AGENTS.md). Verification: import succeeds in `.venv`; one toy question returns a typed decision with level and confidence on 3 spans.
- [ ] 4.2 Define the frozen per-class question set (8 classes from the v3 taxonomy) plus posting-level gate questions, with cyclic rotation readout. Verification: same span re-asked under re-ordered options gives a stable decision (position-bias check) on a 10-span probe set.
- [ ] 4.3 Span-labeling driver: enumerate → posting-level gate → per-span per-class decisions → `009_span_decisions.json`. Verification: one full golden posting labeled end-to-end; schema includes span id, class, decision, confidence, level.

## 5. Deterministic removal

- [ ] 5.1 Removal module: header-follows-section rule, unsure bucket (keep-by-default), rejoin of kept spans; enforce `output == reassemble(kept_spans)`. Verification: unit tests for both header scenarios pass; output-subset invariant holds across all labeled golden postings.
- [ ] 5.2 Select keep/delete/unsure thresholds on a held-out labeled slice (per-class bands if global underperforms). Verification: per-class precision/recall at chosen thresholds recorded on the held-out slice; thresholds + selection data captured in run config.

## 6. Rescue pass

- [ ] 6.1 Port the 008 cache-client pattern into `scripts/009_sentence_classifier_cleaning/llm_client.py` (009-local cache, composite keys, empty-response retry). Verification: cache hit/miss and retry behave on a 2-span probe; cache isolation from 008 `.json` confirmed.
- [ ] 6.2 Rescue prompt + driver for flagged spans only (tiny gateway model, clause-level keep/delete). Verification: rescued set is a small fraction of total spans on the probe posting; cost logged per span.

## 7. Experiment 009 run

- [ ] 7.1 Build `compare_variants.py`: arms = (1) monolith v3 control (reuse 008 cache entries where prompt-hash matches, else fresh calls), (2) classifier-only, (3) classifier+rescue, (4) gate-off ablation; metrics = Jaccard, over-deletion, output-subset invariant, per-class P/R, cost, latency. Verification: dry-run on 1–2 postings produces a complete, well-formed results JSON.
- [ ] 7.2 Full run over the golden slice with latency capture. Verification: results JSON covers every arm × every slice posting; scripts treated immutable after this step (no post-run edits).
- [ ] 7.3 Analysis against the adoption gate (parity-or-better quality at reported cost/latency) with per-class attribution of every miss. Verification: all reported numbers traceable to the results JSON.

## 8. Documentation

- [ ] 8.1 Write `docs/experiments/009-*` per `docs/templates/experiment-log.md` (hypothesis, design, per-arm tables, per-class P/R, limitations). Verification: wire-experiment checklist complete (index, nav, cross-references).
- [ ] 8.2 Log the decision: TODO.md:29 payoff (DSML N-classifier idea), README "Current Status" update, dev-log entry. Verification: docs-sync validation passes; adopt/reject decision recorded with evidence pointer.
