# LLM-Based Boilerplate Removal — Implementation Plan

> **Phase:** 1.4b — LLM cleaning via Kilo Gateway, replacing regex section filters
> **Date:** 2026-07-22

## Decision

**Feed raw full text directly to the LLM.** No section-based pre-filter. The regex section pipeline is inherently brittle — it depends on 23 handcrafted header patterns that fail silently on new posting sources. The cost of including `about_team` and `what_we_offer` in the LLM input is ~$0.002 total across all 27 postings. Not worth the code complexity or the fragility risk.

**Section segmentation stays for metadata.** The existing parse/embed script's title extraction, section detection, and role mapping are still needed for evaluation (function labels, skill extraction) and for Experiment 1.7 (multi-field fusion with separate title/skills/body fields). Section segmentation just stops being a step in the cleaning pipeline.

## Pipeline

```
Raw Markdown (from data/selected-job-postings/)
    │
    ▼
Section Segmentation (existing — parse_and_embed_quickstart.py)
    │  Produces: title_raw, sections, role_category
    │  Used for: metadata, eval labels, Experiment 1.7
    │
    ▼
LLM Cleaning (NEW — scripts/extract_clean_text.py)
    Input:  raw_full_text (entire posting, no section pre-filter)
    Output: llm_clean_text (job-signal content only)
    Cache:  data/.llm_clean_cache.json (per posting + prompt hash)
    │
    ▼
Write llm_clean_text to jobs.json (preserves clean_text from exp_boilerplate.py for comparison)
```

## LLM Integration

| Property | Value |
|---|---|
| API | Kilo Gateway `https://api.kilo.ai/api/gateway` |
| Endpoint | `POST /chat/completions` (OpenAI-compatible) |
| Auth | `Authorization: Bearer $KILO_API_KEY` from `.env` |
| HTTP client | `httpx` (sync) |
| Model | `google/gemini-2.5-flash` |
| Temperature | 0.0 |
| Max output tokens | 2048 (generous, postings won't exceed this) |

### Cost Estimate

```
27 postings × ~625 words avg + prompt overhead
≈ 27K input tokens × $0.15/M = ~$0.004 input
Output: ~16K tokens × $0.60/M = ~$0.01
Total: ~$0.015 per full run
```

With retries and prompt iteration, well under $0.50 total. A $5 Gateway credit covers 300+ runs.

## Prompt

### System Prompt

```
You are a job posting cleaner. Extract only the parts of a job posting that
describe the job itself: responsibilities, required qualifications, and
preferred/nice-to-have skills. Remove everything else.

Categories to strip completely:
- Company description, "About Us", mission statements, values
- Team descriptions (what the team does, team structure, team culture)
- Salary ranges, pay grades, equity, compensation details
- Benefits: health/dental/vision, vacation/PTO, parental leave, wellness
- EEO/diversity statements, "equal opportunity employer" boilerplate
- Recruiter notes, application instructions
- Office locations, hybrid/remote policy boilerplate
- Perks, "why you'll love working here", employee testimonials

Preserve verbatim (do not summarize, paraphrase, or invent):
- All job duties, responsibilities, and day-to-day tasks
- All technical skills, tools, frameworks, languages, platforms
- All required qualifications and experience levels
- All preferred/nice-to-have qualifications
- All education and certification requirements
- Original wording of preserved sections

Output only the cleaned text. No headers, prefixes, explanations, or formatting.
```

### User Message

```
Clean this job posting:

---
{raw_full_text}
```

## Output Format

Plain text. No JSON, no markdown headers, no commentary. The system prompt enforces this.

## Caching

File: `data/.llm_clean_cache.json`

```json
{
  "meta": {
    "prompt_hash": "sha256 of system prompt + user message template",
    "model": "google/gemini-2.5-flash"
  },
  "entries": {
    "posting_id": {
      "input_hash": "sha256 of raw_full_text",
      "output": "cleaned text...",
      "usage": {"prompt_tokens": 500, "completion_tokens": 300},
      "timestamp": "2026-07-22T17:49:56-04:00"
    }
  }
}
```

Cache behavior:
- Skip API call if `(posting_id, input_hash)` exists in cache and `prompt_hash` matches
- Write cache after each posting (crash-safe)
- Commit cache to git (cleaned text only, no secrets)
- Manual invalidation: delete `data/.llm_clean_cache.json`

## Eval Framework

### Golden Cleaned Set

`data/golden_cleaned.json` — 5 postings hand-cleaned by a human:

1. `bmo-associate-data-scientist` — heavy boilerplate (EEO, salary, "About Us")
2. `affirm-ml-engineer-2` — pay grade embedded in qualifications section
3. `hellofresh-ml-engineer-operations-technology` — benefits list in single-section posting
4. `mastercard-data-scientist-2` — EEO statement embedded
5. `scribd-data-scientist-2` — moderate boilerplate, clean structure (control)

Format:
```json
[
  {
    "posting_id": "bmo-associate-data-scientist",
    "source": "manual",
    "text": "We are seeking a talented and experienced Data Scientist..."
  }
]
```

### Eval Metrics

| Metric | Measure | Target |
|---|---|---|
| Hallucinations | Skills/duties in LLM output not present in raw text | 0 |
| Over-deletion | Skills/qualifications in raw text missing from LLM output | < 5% |
| Under-deletion | Boilerplate phrases in LLM output that should be removed | < 10% |
| Self-retrieval | Posting retrieves itself from `llm_clean_text` embedding | Must improve vs. section-based `clean_text` (currently 0/27) |

### Eval Script

`scripts/eval_llm_cleaning.py`:
1. Load `data/golden_cleaned.json` and `data/jobs.json` (with `llm_clean_text`)
2. For each golden posting, print:
   - Side-by-side: golden text (left) vs. LLM output (right)
   - Boolean checks: contains salary? EEO? benefits? company mission?
   - Word-level Jaccard similarity between LLM output and golden text
3. Flag discrepancies with `>>>` markers

## Task Order

### Step 1: Infrastructure

1. Add `KILO_API_KEY=<user-supplied>` to `.env` (manual, one-time)
2. Add `.env` to `.gitignore`
3. Add `"httpx>=0.28,<1.0"` and `"python-dotenv>=1.0,<2.0"` to `pyproject.toml`
4. Run `uv sync` to install

### Step 2: Extraction Script

`scripts/extract_clean_text.py`:

```
Class LLMCleaner
  __init__(api_key, model, base_url, temperature, cache_path)
  clean(posting_id, raw_text) -> str
    1. Hash input, check cache
    2. If miss: POST to /chat/completions via httpx
    3. Extract response content, validate non-empty
    4. Write entry to cache
    5. Return cleaned text
  clean_batch(jobs) -> None
    For each job: call clean(), save to cache after each, 1 req/sec rate limit
```

### Step 3: Golden Set

Hand-clean 5 postings → `data/golden_cleaned.json`. Review once, then re-read a day later to catch inconsistencies.

### Step 4: Eval Script

`scripts/eval_llm_cleaning.py` — loads golden set + jobs.json, prints comparison, flags issues.

### Step 5: Run + Iterate

1. `python scripts/extract_clean_text.py` — clean all 27 postings
2. `python scripts/eval_llm_cleaning.py` — eval against golden set
3. Iterate on prompt if under-deletion or over-deletion exceeds thresholds
4. Delete cache to force re-extraction after prompt changes

### Step 6: Integration

- `jobs.json` gets `llm_clean_text` field
- Run `exp_boilerplate.py` adapted to use `llm_clean_text` as embedding input:
  - Comparison: `raw_full_text` vs. `clean_text` (section-based) vs. `llm_clean_text`
  - Metrics: separation gap, self-retrieval, UMAP visualization

## Risks

| Risk | Mitigation |
|---|---|
| LLM invents skills not in original | `temperature: 0.0`, system prompt says "do not summarize, paraphrase, or invent", eval catches it |
| LLM deletes real job content | Golden set eval catches over-deletion. Prompt says "preserve verbatim." |
| API key in git | `.env` in `.gitignore`. Cache file stores only cleaned text, never the key. |
| Cache stale after prompt change | `prompt_hash` in cache meta detects prompt changes, forces re-extraction. |
| Self-retrieval doesn't improve | Title + skills fusion (Experiment 1.7) adds discriminative signal back. LLM cleaning removes noise; fusion adds signal. |

## File Manifest

```
scripts/
  extract_clean_text.py       # NEW — LLM cleaning via Kilo Gateway + caching
  eval_llm_cleaning.py        # NEW — golden set comparison

data/
  .llm_clean_cache.json       # NEW — LLM response cache (git committed)
  golden_cleaned.json         # NEW — 5 hand-cleaned postings

.env                          # NEW — KILO_API_KEY (gitignored)
.gitignore                    # MODIFIED — add .env

pyproject.toml                # MODIFIED — add httpx, python-dotenv
data/jobs.json                # MODIFIED — add llm_clean_text field
```

## What Stays

- `scripts/exp_boilerplate.py` — section-based cleaning stays as a comparison baseline
- `scripts/parse_and_embed_quickstart.py` — section segmentation stays for metadata
- All existing eval metrics — separation gap, HDBSCAN, NN audit — same pipeline, different input text
