## Context

We have 27 job postings in `data/jobs.json`, each with `raw_full_text` and `llm_clean_text` (the output of Experiment 003's LLM extraction pass). The existing extraction prompt strips all text that isn't a direct responsibility, qualification, or requirement — but inconsistently preserves org context mixed in with responsibilities.

The golden set (`data/golden_cleaned.json`) has been manually split: each of the 5 evaluation postings now has `org-context` (team function, domain, mission) and `req-context` (pure responsibilities, qualifications) fields. The original unsplit `text` field remains for backward reference.

Experiment 003 showed that LLM cleaning improved separation gap by title from +0.0060 to +0.0493. We need to test whether adding the stripped organizational context back improves clustering by function (the 12-category taxonomy). This requires clean, separated fields and golden sets.

## Goals / Non-Goals

**Goals:**
- Tighten the `llm_clean_text` extraction prompt to explicitly strip org context, re-run, and re-evaluate against the `req-context` golden set field
- Extract a clean `llm_org_context` field from every posting's raw text using a new LLM pass
- Define a clear boundary: org context = team function, domain, mission; responsibilities = specific daily tasks and requirements
- Handle the mixed zone by leaning toward org context (keeping responsibilities pure)
- Run a three-way A/B comparison with separation gap and nearest-neighbor audit

**Non-Goals:**
- Relying on the existing parsed `sections` fields — this must work on unstructured text
- Template-based or regex extraction — purely LLM-driven for generalizability

## Decisions

**1. Two LLM passes, both on raw_full_text, with independent prompts**
- Pass 1: tightened version of existing prompt → `llm_clean_text` (pure responsibilities)
- Pass 2: new prompt for org context → `llm_org_context`
- Rationale: each pass is optimized for a single extraction task. Two independent prompts can be validated against their own golden set. The raw_full_text is the same source for both, so no information is lost.

**2. Boundary rule for the mixed zone**
- Sentences that simultaneously describe team domain AND specific duties should be included in org context, NOT duplicated into `llm_clean_text`
- Example: "You will build ML systems that automate customer operations such as disputes, returns, and fraud" → belongs in org context (the "customer operations / fraud" domain signal is the valuable part)
- Rationale: lean toward keeping `llm_clean_text` pure responsibility text. The A/B test will reveal whether the domain signal helps or hurts.

**3. Golden set — already split**
- The 5 golden-set postings in `data/golden_cleaned.json` have been manually split with `org-context` and `req-context` fields added to each entry
- Same 5 postings from Experiment 003: BMO Data Scientist, Affirm ML Engineer 2, HelloFresh ML Engineer, Mastercard Data Scientist 2, Scribd Data Scientist 2
- Evaluation metrics: Jaccard similarity, hallucination detection, over-deletion rate (same approach as eval_cleaning.py), evaluated separately against each field

**4. Tightened prompt for Pass 1 (llm_clean_text)**
- Add explicit instruction to strip: "team descriptions, what the team does, team structure, team culture, and descriptions of the team's function or domain within the organization"
- Add explicit instruction: "If a sentence describes both the team's domain AND specific duties, remove the entire sentence"
- Keep the existing boilerplate categories (company description, benefits, salary, EEO, etc.)
- Keep the "preserve verbatim" rules for duties, skills, qualifications, requirements
- Rationale: the mixed zone rule means these sentences belong in org context, not responsibilities

**5. Prompt for Pass 2 (llm_org_context)**
```
Extract the "organizational context" from this job posting. This is text that
describes the team's function, domain, or mission within the company — what
business problem the team exists to solve.

Include:
- Team descriptions: what the team does, its mission, its domain
- The role's purpose within the organization
- Context about the team's stakeholders, customers, or users
- Sentences that blend domain context with specific duties
- The "About the Role" / "About the Team" narrative

Exclude:
- Company descriptions, "About Us", mission statements, values
- Salary ranges, benefits, EEO statements, recruiter boilerplate
- Specific job duties, day-to-day tasks (even if present in mixed sentences)
- Required qualifications, skills, experience levels
- Perks, "why you'll love working here", employee testimonials
- Office locations, hybrid/remote policy boilerplate

Output only the extracted organizational context. No headers, prefixes,
explanations, or formatting. If no organizational context is found, output
nothing.
```

**6. Evaluation metrics for the A/B test**
- Separation gap by function tag (not by title) — primary metric
- Nearest-neighbor audit: for each posting, check whether top-5 neighbors share at least one function tag from the 12-category taxonomy
- Rationale: title-based labels are noisy. The function taxonomy from research-report 003 is a better ground truth for whether organizational context improves semantic clustering.

**7. Text variant composition for embedding**
- Variant A: `llm_clean_text` only (re-extracted, pure responsibilities) — baseline
- Variant B: `"[ORG] {llm_org_context} [ORG] [RESP] {llm_clean_text}"` — concatenated with section markers, org context repeated 2x
- Variant C: `llm_org_context` only (probe: how much signal does context alone carry?)
- Rationale: weighted concatenation follows the research report's recommendation for section-weighted embeddings. The [ORG]/[RESP] markers provide clear section boundaries to the embedding model.

## Risks / Trade-offs

- **LLM cost**: ~54 API calls (27 for each pass) at ~300-500 tokens each, ~$0.04 total. Negligible.
- **Golden set subjectivity**: hand-labeling "org context" vs "responsibilities" involves judgment calls, especially in the mixed zone. Mitigation: boundary is encoded in the already-split golden set — evaluations will be consistent against those references.
- **Mixed zone ambiguity**: some sentences genuinely blend domain and duties. Mitigation: accept imperfect separation and measure impact via the A/B test itself — if Variant B outperforms, the extraction is good enough.
- **Re-extraction changes baseline**: tightening the prompt will change `llm_clean_text` values, so Experiment 003 results won't be directly comparable. Mitigation: document the change and the reason clearly. The new baseline is more honest.
- **Small N (27 postings)**: separation gap by function may be unreliable for rare categories. Mitigation: use nearest-neighbor audit as a secondary metric that doesn't require per-category aggregates.
