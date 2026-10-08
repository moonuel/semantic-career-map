# Job Board API Scrapers — Coverage, Frequency, Sorting, and Operational Research Report

> Date: 2026-10-07
> Context: Semantic Career Mapping Platform — research into ATS-hosted public job-board APIs to inform the `backend/ingestion/scrapers.py` ingestion layer (currently Ashby, Greenhouse, Lever, plus the dataroles.co curator). Covers coverage (fields per source), pagination/sorting, change/frequency semantics, board discovery, and operational concerns.

---

## Table of Contents

1. [The Two-Tier Job-Data Landscape](#1-the-two-tier-job-data-landscape)
2. [Source-by-Source Reference](#2-source-by-source-reference)
3. [Coverage Comparison](#3-coverage-comparison)
4. [Pagination, Sorting, and Change Semantics](#4-pagination-sorting-and-change-semantics)
5. [Polling Frequency](#5-polling-frequency)
6. [Board Discovery (Finding Tokens at Scale)](#6-board-discovery-finding-tokens-at-scale)
7. [Deduplication and Deletion Detection](#7-deduplication-and-deletion-detection)
8. [Rate Limits, ToS, and Responsible Use](#8-rate-limits-tos-and-responsible-use)
9. [Recommendations for This Project](#9-recommendations-for-this-project)
10. [Sources](#10-sources)

---

## 1. The Two-Tier Job-Data Landscape

Two fundamentally different ingestion models exist:

| Model | What it is | Examples |
|---|---|---|
| **ATS board APIs (first-party)** | Each employer's own careers board, exposed by the ATS vendor as a public, key-free read endpoint. One call returns the whole board. | Greenhouse, Lever, Ashby, SmartRecruiters, Workable, Recruitee, Personio, Teamtailor |
| **Aggregators / curators (second-party)** | A third party that has already crawled many ATS boards and exposes a unified feed. | dataroles.co (curator over 16 ATS systems, sitemap-based read path), JobListings API, JobsPipe, Apify actors |

Key trade-offs (joblistingsapi.com, 2026):

- **ATS-first is free, current, and authoritative**, but requires *board discovery* — there is no master list of every Greenhouse token. You must find and maintain thousands of slugs as employers onboard and churn.
- **Aggregators solve discovery and polling**, but lag, can silently drop postings, and add an intermediary whose coverage and quality you cannot audit. Dedup and normalization still remain your problem.

The ATS-first tier is the right backbone for this project (already implemented); curators are a discovery aid, not a primary source.

---

## 2. Source-by-Source Reference

### 2.1 Greenhouse

- **Endpoint:** `GET https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true` (developers.greenhouse.io, Job Board API).
- **Auth:** none for any GET; only application submission (`POST .../jobs/{id}`) requires Basic Auth.
- **Coverage:** `id` (job *post* id — note `internal_job_id` is the job itself, so posts are the dedup unit), `title`, `absolute_url`, `location.name` (free text, possibly multi-site `"New York, New York, United States; Remote; ..."`), `first_published`, `updated_at`, `application_deadline`, `requisition_id`, `company_name`, `language`, `metadata` (opt-in custom fields). `content=true` adds the full HTML description plus embedded `departments` and `offices` arrays.
- **Caveats:**
  - Salary fields are **absent** from the public board API — pay ranges are usually rendered into the HTML content text, not structured data. Our scraper records `salary_min/max/currency = None`.
  - `content=true` on the list endpoint is the only way to get descriptions without one request per post.
  - There is also an older JSON feed (`boards.greenhouse.io/embed/job_board?for={token}&format=json`) still served; the `boards-api` host is canonical.
- **Sorting/pagination:** none — the whole board arrives in one array (`meta.total`); no `skip`/`limit` parameters are documented.

### 2.2 Lever

- **Endpoint:** `GET https://api.lever.co/v0/postings/{site}?mode=json` (lever/postings-api README). EU-hosted accounts use `postings.eu.lever.co` (careers sites on `jobs.eu.lever.co`).
- **Auth:** none. HTTPS only.
- **Coverage:** `id` (UUID), `text` (title), `description`, `lists[]` (requirements/responsibilities blocks), `additional`, `categories` (`location`, `commitment`, `team`, `department`, `allLocations[]`), `createdAt` (millis), `hostedUrl`, `salaryRange` (`min`, `max`, `currency`) when the employer publishes it, `workplaceType` (remote/hybrid/on-site on some accounts).
- **Caveats:**
  - The description is **split** across `description` + `lists` + `additional`; concatenation is required or the body looks truncated (implemented in `backend/ingestion/scrapers.py:_lever_description`).
  - `mode=json` (or `Accept: application/json`) is mandatory — without it the endpoint serves rendered HTML.
- **Sorting/pagination:** `skip` and `limit` query params exist; default ordering is alphabetical by title (the API's own example boards say "displays all jobs alphabetically"). Sorting by recency is not offered — sort client-side from `createdAt`.

### 2.3 Ashby

- **Endpoint:** `GET https://api.ashbyhq.com/posting-api/job-board/{slug}` (developers.ashbyhq.com, "Ashby Job Postings API"). Optional `includeCompensation=true`.
- **Auth:** none. This is distinct from Ashby's authenticated `jobPosting.list` admin API (`POST api.ashbyhq.com/jobPosting.list`, Basic Auth, requires `jobsRead`).
- **Coverage:** `id` (UUID), `title`, `descriptionHtml` **and** `descriptionPlain`, `department`, `team`, `employmentType` (`FullTime`, ...), `location` + `secondaryLocations[]` (structured `addressLocality`/`addressRegion`/`addressCountry`), `isRemote`, `workplaceType`, `isListed`, `publishedAt`, `jobUrl`, `applyUrl`, `address.postalAddress`.
- **Compensation:** the richest of the three — with `includeCompensation=true` a `compensation` object appears with `summaryComponents[]` (base Salary / EquityPercentage / EquityCashValue / Bonus / Commission tiers, each with `minValue`, `maxValue`, `currencyCode`, `interval`). Equity/bonus tiers carry no bounds; the base pay is the `compensationType == "Salary"` component (matching our `_salary_from_components` logic). Also present: `shouldDisplayCompensationOnJobPostings` — worth respecting.
- **Caveats:**
  - No pagination — the whole board in one response.
  - **`isListed == false` postings are included by default** — these are unlisted roles that should only be reachable by direct link. Third-party scrapers skip them (e.g. Apify's actor "Ashby postings marked as unlisted are skipped"); our scraper currently does **not** filter on `isListed` — a candidate improvement.
- **Sorting:** not documented; treat as unsorted, sort client-side from `publishedAt`.

### 2.4 SmartRecruiters

- **Endpoint:** `GET https://api.smartrecruiters.com/v1/companies/{identifier}/postings?limit=100&offset=0` (developers.smartrecruiters.com, Posting API).
- **Auth:** none.
- **Coverage:** paginated (`offset`, `limit`, `totalFound`) with strong structured metadata the "big three" lack: `releasedDate`, `location` (city/region/country/lat/lon/`remote`/`hybrid`), `industry`, `department`, `function`, `typeOfEmployment` (id + label), `experienceLevel` (id + label: associate, mid_senior_level, executive, ...), `customField[]`, `language`, `visibility` (`PUBLIC`).
- **Caveats:**
  - The list endpoint returns metadata only. The full description requires a second call per posting: `GET .../postings/{id}` (the "job ad" detail endpoint). So description coverage costs N+1 requests per company — a real difference from Greenhouse's one-call `content=true`.
  - Common among **large corporates** rather than startups (Babak-hasani/company-career-scraper), so it complements the big three geographically and vertically.

### 2.5 Workable

Two distinct APIs — do not confuse them:

- **Authenticated v3** (`https://{subdomain}.workable.com/spi/v3/jobs`, bearer token, `r_jobs` scope): full account API, rate limit **10 requests / 10 seconds** (429 beyond), with `since_id`/`max_id` scroll params, `created_after`/`updated_after` filters, and `include_fields=description,full_description,requirements,benefits`. Jobs sorted by insertion time ascending. Meant for HR integrations.
- **Public widget API** (no auth): `GET https://www.workable.com/api/accounts/{slug}?details=true` (redirects to `apply.workable.com/api/v1/widget/accounts/{slug}?details=true`). Returns published jobs for one account with `title`, `shortcode`, `url`/`shortlink`, `location` (structured + `telecommuting`), `published_on`, and — with `details=true` — `description`/`full_description`. Filter client-side to `state == "published"` / non-empty `published_on` (sevic.dev, 2026).

The public widget endpoint is the scraper-relevant one; it is undocumented-but-stable, powers Workable-hosted career sites and widgets.

### 2.6 Recruitee

- **Endpoint:** `GET https://{company}.recruitee.com/api/offers/` (docs.recruitee.com, Careers Site API).
- **Auth:** none (the authenticated ATS API is `api.recruitee.com/c/{company_id}/offers` with OAuth).
- **Coverage:** published offers with `id`, `guid`, `slug`, `title`, `department`, `locations`, `remote`/`hybrid`/`on_site` flags, `kind` (`job` vs `talent_pool` — filter to `job`), `status: "published"`, `careers_url`, `careers_apply_url`, `created_at`/`published_at`/`updated_at`, `tags[]`, plus `description`, `requirements` fields.
- **Sorting/filters:** server-side `department` and `tag` query filters; no pagination documented (whole set).
- Also offers an authenticated **XML offers feed** and webhooks.

### 2.7 Personio

- **Endpoint:** `GET https://{tenant}.jobs.personio.de/xml` (XML feed per tenant; Personio-hosted boards at `{tenant}.jobs.personio.de`).
- **Coverage:** XML feed of open positions (title, description, location, employment type, etc.). JSON-LD `JobPosting` blocks are also present on the rendered pages.
- **Note:** Personio is strong in the **German/DACH SMB market** — complementary to the US-centric big three. Documented by third parties as an XML feed, not a JSON API; parse as an extra source type.

### 2.8 Teamtailor

- **Public API requires a token** (`api.teamtailor.com/v1/jobs`, `Authorization: Token token=...`, JSON:API format, EU/NA stacks, dated `X-Api-Version`, rate limits documented per token scope).
- There is **no keyless public board API**; per-employer scraping must fall back to the career site's JSON-LD or a partner integration. Treat Teamtailor as out of scope for key-free ingestion.

### 2.9 Workday (and other enterprise suites: SAP SuccessFactors, Oracle, iCIMS)

- Workday dominates the enterprise segment but exposes **no simple per-company JSON endpoint** in a stable public shape: postings live inside the CXS/`site.ai`/`job` GraphQL-ish endpoints behind tenant-specific `wd3`/`wd5` hostnames, session-ish tokens, and frequent format changes. Community aggregators (e.g. `Feashliaa/job-board-aggregator` — 1M+ jobs across "Greenhouse, Lever, Ashby, Workday") report dedicating disproportionate engineering effort to Workday compared to the public-API ATSs.
- **Recommendation:** treat enterprise suites as out of scope for the MVP; the SMB/startup ATSs already carry the AI/tech job signal this project targets.

### 2.10 Curators (dataroles.co and peers)

- dataroles.co: a curator over **16 ATS systems**, documented read path via sitemap index → paged `jobs-data` sitemaps → per-job pages carrying a single schema.org `JobPosting` JSON-LD block whose `identifier.value` records the origin system (implemented in `list_dataroles_job_urls` / `fetch_dataroles_job`). Per-page fetching means the sitemap is a cheap URL discovery layer and the JSON-LD is rich (dates, salary, structured address).
- Commercial peers (JobListings API, JobsPipe, Apify actors) aggregate the same ATS feeds with `updated_since` cursors and per-source filters — useful *as references for field schema and dedup design* even if never adopted as dependencies.

---

## 3. Coverage Comparison

| Field | Greenhouse | Lever | Ashby | SmartRecruiters | Workable (public) | Recruitee |
|---|---|---|---|---|---|---|
| Full description (list call) | ✅ `content=true` | ✅ (split, must concat) | ✅ HTML + plain | ❌ (per-posting call) | ✅ `details=true` | ✅ |
| Structured salary | ❌ | ✅ `salaryRange` | ✅ `summaryComponents` | ❌ (rare customField) | ❌ | ❌ |
| Structured location | ⚠️ `location.name` + `offices[]` | ⚠️ text + `allLocations` | ✅ postalAddress + secondary | ✅ city/country/lat-lon | ✅ structured + telecommuting | ✅ + remote flags |
| Country code | ⚠️ via office name | ❌ (some accounts) | ✅ | ✅ | ✅ | ✅ |
| Posted date | ✅ `first_published` | ✅ `createdAt` (ms) | ✅ `publishedAt` | ✅ `releasedDate` | ✅ `published_on` | ✅ `published_at` |
| Updated date | ✅ `updated_at` | ❌ | ❌ | ❌ | ❌ | ✅ `updated_at` |
| Employment type | ⚠️ sparse | ✅ `commitment` | ✅ | ✅ typed | ⚠️ | ✅ |
| Department / team | ✅ (via content=true) | ✅ | ✅ | ✅ | ✅ departments | ✅ |
| Remote flag | ❌ (parse text) | ⚠️ `workplaceType` | ✅ `isRemote` | ✅ | ✅ | ✅ |
| Pagination | ❌ | `skip`/`limit` | ❌ | ✅ offset/limit | ❌ | ❌ |
| Auth | none | none | none | none | none | none |

Take-aways:

- **Ashby is the richest per posting** (structured salary + structured locations + remote flag); **SmartRecruiters has the best taxonomy metadata** (function, experience level) but pays for it with N+1 description fetching; **Greenhouse is the cheapest per byte** (one call, full HTML) but structurally weakest (salary and location parsing happen downstream).
- This matches the ecosystem consensus (Apify actor notes: "Ashby boards almost always [expose salary]"; "Greenhouse does not provide [description] in the list feed" without `content=true`).

---

## 4. Pagination, Sorting, and Change Semantics

- **Whole-board responses** (Greenhouse, Ashby, Recruitee, Workable widget): simplest possible ingest — one request per company per poll. Typical board sizes are tens-to-hundreds of postings (test boards in the wild: gitlab 218, leverdemo 384, notion 135), well within a single response. No cursor state to maintain.
- **Lever** supports `skip`/`limit`, but for board-sized data a single unpaginated call is the norm; pagination matters only if you avoid fetching descriptions.
- **SmartRecruiters** is the one offset/limit paginated source among the key-free set (`limit` up to ~100); walk pages until `offset >= totalFound`.
- **Sorting:** none of the public board APIs document server-side recency sorting. Lever's default is alphabetical by title. Always sort client-side from the source's timestamp (`createdAt`/`publishedAt`/`first_published`/`releasedDate`/`published_at`).
- **Change detection / deltas:** the public APIs expose no webhooks and no `updated_since`. The standard pattern (joblistingsapi.com; arianra/job-aggregator) is:
  1. Diff every poll against the previous snapshot.
  2. Present postings that appear = *new/reopened*; postings that vanish = *closed* (silent disappearance is the norm — no tombstone event is emitted).
  3. Use `updated_at` where offered (Greenhouse, Recruitee) to detect edits without refetching descriptions.
- **ID stability:** Greenhouse post `id` and Lever `id`/Ashby `id` are stable across runs (theprojectdesk actor: "job_id is the vendor's own and is stable across runs — the right key for diffing new and closed roles"). This is exactly the `<ats>:<board>:<id>` `source_id` scheme already in `ScrapedJob`.

---

## 5. Polling Frequency

- Job boards are low-churn data: a company posts a handful of new roles per week; descriptions change rarely. Community schedulers run **daily** (Feashliaa: "daily automated refreshes via GitHub Actions"; Matt-Kobzik: "writes new jobs to a Notion database daily"; davidbenittah: "keep scheduled run frequency reasonable").
- With whole-board responses, a daily or twice-daily poll of N boards costs N requests per cycle — trivial at hundreds of boards.
- Hourly polling buys little: posting freshness matters mostly for aggregator-style products, not for building a semantic corpus. Once-daily is sufficient for this project; use `updated_at`/diffing to avoid re-processing unchanged postings.
- Respect rate limits where documented (Workable authenticated API: 10 req/10 s; the public widget endpoint is undocumented — be conservative anyway).

---

## 6. Board Discovery (Finding Tokens at Scale)

There is **no vendor registry** of board tokens. Proven strategies:

1. **URL pattern extraction** — the dominant manual method. Careers-page or posting URL reveals ATS + token:
   - `boards.greenhouse.io/{token}`, `job-boards.greenhouse.io/{token}`, `job-boards.eu.greenhouse.io/{token}`
   - `jobs.lever.co/{token}`, `jobs.eu.lever.co/{token}`
   - `jobs.ashbyhq.com/{token}` (note: Ashby slugs are case-sensitive)
   - `careers.smartrecruiters.com/{token}`, `{tenant}.jobs.personio.de`, `{company}.recruitee.com`
   - If the careers page is a JS app without a visible board URL, open any posting — the address bar shows the real board (davidbenittah guide).
2. **Cross-source inference** — job aggregators (LinkedIn posts, etc.) expose the same companies; matching by company name reveals which ATS each uses. Community repos (Babak-hasani: 169 companies pre-tagged with `ats_type` + `board_token`) are reusable seeds.
3. **Curator sitemaps as a discovery index** — dataroles.co aggregates 16 ATS systems and embeds the origin system in each posting's `identifier.value`. Walking its sitemap yields (URL → ATS, company) pairs at scale — cheaper than guessing, and a natural complement to the existing `list_dataroles_job_urls`.
4. **HTML probing** — fetch the careers page and grep for `greenhouse.io/`, `lever.co/`, `ashbyhq.com/`, `smartrecruiters.com/` hostnames to auto-detect the ATS (chazee actor does exactly this "ATS auto-detection").

---

## 7. Deduplication and Deletion Detection

- **Within-source dedup:** native ids + our `<ats>:<board>:<id>` `source_id` scheme are sufficient. Greenhouse `internal_job_id` vs post `id` distinction matters — the same job can have several posts (locations); dedup on the post id, aggregate on the internal id if needed.
- **Cross-source dedup:** the same role is frequently cross-posted (e.g. on Greenhouse and Lever). Community approach (arianra/job-aggregator): fingerprint on `(company, normalized title, location)` and merge keeping the richest record. The chazee actor deduplicates only across *different boards*, keeping the fuller copy — conservative and sensible.
- **Curator overlap:** dataroles postings re-appear the origin ATS; `identifier.value` should map to the same `source_id` namespace so curator records join back to first-party fetches.
- **Deletion:** postings vanish silently. Only poll-diffing detects closure; retain last-seen timestamps per `source_id`.

---

## 8. Rate Limits, ToS, and Responsible Use

- The big-three board APIs are **public, documented, key-free, and explicitly intended** for building career pages on top of (Greenhouse: "Authentication … is not required for any GET endpoints"; Ashby's doc addresses companies hosting their own pages). Data returned is published job-advertising content, not personal data.
- **Anti-bot posture is minimal** (Apify ratio_tech: "Datacenter proxies are sufficient … residential is not required" for Ashby). A plain HTTP client like our `requests`-based scrapers is appropriate; no browser automation needed anywhere in the key-free set.
- Documented limits are rare outside authenticated APIs (Workable: 10 req/10 s). Undocumented endpoints (Workable widget) warrant conservative pacing and graceful backoff on 429/5xx.
- Always set a `User-Agent`, cache aggressively, and scale polling with board count rather than aggressiveness.

---

## 9. Recommendations for This Project

1. **Keep ATS-first as the backbone** (already implemented for Ashby/Greenhouse/Lever). Add **SmartRecruiters** next — structured locations, experience level, and function metadata are directly useful for the embedding/taxonomy work, and the N+1 description fetch is acceptable at our scale.
2. **Filter Ashby `isListed == false`** postings to match what the public board actually displays.
3. **Consider Recruitee and Personio** for geographic (EU/DACH) breadth; both are key-free (Recruitee JSON, Personio XML).
4. **Skip Workday/SuccessFactors/Teamtailor** for the MVP — either no key-free feed or unstable formats.
5. **Poll daily**, diff against previous snapshot, record `last_seen` per `source_id`, and treat vanishing postings as closed rather than deleted events.
6. **Use dataroles.co primarily as a discovery index** for (company → ATS, token) pairs, its per-page JSON-LD as a fallback parser already implemented.
7. **Do not rely on any source for structured salary except Ashby and Lever**; for Greenhouse, salary lives in description text and would have to be extracted (an LLM-extraction task, per experiment 003 lineage).

---

## 10. Sources

- Greenhouse Job Board API docs — developers.greenhouse.io/job-board.html; grnhse/greenhouse-api-docs (_boards.md, _jobs.md)
- Lever Postings API — github.com/lever/postings-api (README); hire.lever.co/developer/documentation
- Ashby Job Postings API — developers.ashbyhq.com/docs/public-job-posting-api; developers.ashbyhq.com/docs/dedicated-partner-job-feeds; ashby.slab.com (deprecated doc, field reference)
- SmartRecruiters Posting API — developers.smartrecruiters.com/docs/posting-api; developers.smartrecruiters.com/reference/v1listpostings
- Workable — workable.readme.io/reference/jobs (v3); help.workable.com (API FAQ, careers-page guide); sevic.dev (public widget endpoint integration, July 2026)
- Recruitee Careers Site API — docs.recruitee.com/reference/offers.md, /reference/intro-to-careers-site-api
- Personio boards — `{tenant}.jobs.personio.de/xml` feed (theprojectdesk ATS actor reference)
- Teamtailor API — docs.teamtailor.com (token-authenticated)
- Ecosystem / operational references — davidbenittah/greenhouse-lever-ashby-job-scraper-guide; Babak-hasani/company-career-scraper; arianra/job-aggregator; Feashliaa/job-board-aggregator; Matt-Kobzik/job-scraper; Apify actors (ratio_tech, chazee, theprojectdesk, vxc1104, cliqtomedia); joblistingsapi.com (Lever guide, June 2026); jobspipe.dev/sources/lever
- First-party verification — live API responses for boards-api.greenhouse.io (airtable, runpod, stabilityai), api.lever.co, api.ashbyhq.com (runway, posthog), api.smartrecruiters.com (freshworks, Axiado, BayOneSolutions4), and dataroles.co sitemap/JSON-LD read path (as implemented in `backend/ingestion/scrapers.py`)
