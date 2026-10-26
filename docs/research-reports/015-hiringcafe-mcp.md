# HiringCafe MCP — Agent Access to an ATS-First Aggregator — Research Report

> Date: 2026-10-13
> Context: Semantic Career Mapping Platform — research into HiringCafe (hiring.cafe / hiringcafe.com), the ATS-first, AI-extracted job aggregator, and the Model Context Protocol surfaces built around it: HiringCafe's **official no-auth MCP server** (verified live in this report), its official Claude Connector, and third-party community/commercial MCP wraps. Companion to [Report 013 — Job Board API Scrapers](013-job-board-api-scrapers.md); re-asks the same ingestion/access questions from the "agent door" of the strongest second-party aggregator.

---

## Table of Contents

1. [HiringCafe in One Page](#1-hiringcafe-in-one-page)
2. [Why MCP Matters Here](#2-why-mcp-matters-here)
3. [Official MCP Server (ChatGPT App) — Verified Live](#3-official-mcp-server-chatgpt-app-verified-live)
4. [The Filter Taxonomy (AI-Structured Schema)](#4-the-filter-taxonomy-ai-structured-schema)
5. [Official Claude Connector](#5-official-claude-connector)
6. [Third-Party MCP Surfaces](#6-third-party-mcp-surfaces)
7. [Field Coverage vs. Report 013 Sources](#7-field-coverage-vs-report-013-sources)
8. [Robots, ToS, and Rate Posture](#8-robots-tos-and-rate-posture)
9. [Recommendations for This Project](#9-recommendations-for-this-project)
10. [Sources](#10-sources)

---

## 1. HiringCafe in One Page

- **What it is:** a consumer job search engine that indexes postings **directly from employer career pages and ATSs** (Greenhouse, Lever, Ashby, Workday, BambooHR, PeopleAdmin, Zoho, …) instead of re-listing job-board ads. Founded 2023 by Ali Mir (ex-Meta) and Hamed Nilforoshan (Stanford PhD). Positioning is anti-ghost-job and anti-recruiter-spam: postings link straight to the employer's own application form (Business Insider, 2024; blog.hiring.cafe, 2025).
- **Pipeline (self-disclosed):** a bot crawls "tens of thousands of organization websites" about **twice a day**, then an **OpenAI LLM pass extracts and summarizes** structured fields per posting — salary, seniority, years of experience, industry, technical tools, requirements summary, workplace geography, company enrichment (Business Insider 2024; HiringCafe LinkedIn, 2023). Third-party profiling puts the AI-structured schema at **90+ fields per job and 20+ per company** (Apify haketa actor page).
- **Scale:** "...indexing **5 million+ jobs**" and running a 24/7 "Talent Agent" AI search, out of beta US-wide; the company raised **$6.8M led by Spark Capital** (with Nonfiction Capital, Silicon Gardens, angels from ZipRecruiter and Indeed) and claims 2M+ monthly users (blog.hiring.cafe, 2026-09-28). A single verified MCP search over "machine learning engineer" reported **8,722 matches** (first-party curl, 2026-10-13).
- **Stack (self-disclosed):** Next.js server-side rendering (→ `__NEXT_DATA__` JSON island in every page), Tailwind/Chakra UI, Google Cloud + Firestore, OpenAI API (HiringCafe LinkedIn, 2023).
- **Business model:** free for candidates; revenue from promoted listings and the Talent Network (FAQ v9); hired applicants apply on the **employer's own portal**, so the platform holds both the ATS provenance (`source`, `ats`, `boardToken`, direct `applyUrl`) and the AI-extracted layer in one record.

HiringCafe is essentially a deployed, scaled version of this project's thesis: **ATS-first ingestion + LLM-extracted structured metadata + semantic search**. Study its schema and agent interface even if its data is never ingested.

---

## 2. Why MCP Matters Here

- **There is no public REST API, and the terms do not invite scraping.** jobspipe.dev (2026): "Hiring.cafe does not offer a public API, and its terms do not invite scraping. It is a consumer product: the interface is the product, and its data pipeline is the moat behind it."
- **The industry answer to "no API" is MCP.** Consumer job-search properties are shipping official MCP servers instead of documenting REST APIs: Indeed (beta, Streamable HTTP, Claude-Connector-first — docs.indeed.com, 2026), startup.jobs (`https://api.startup.jobs/mcp`, 8 read-only tools), GigHQ (`https://mcp.gighq.ai/mcp`). HiringCafe does the same: two official agent docs pages (`documentation/hiringcafe-claude-connector`, `documentation/hiringcafe-chatgpt-app`, both lastmod 2026-10-07 in its own sitemap).
- **Access model taxonomy** for this source, worst to best:
  1. Adversarial scraping of the search API behind Cloudflare (Apify parsebird-style) — ToS-gray, brittle.
  2. Robots-respecting public-page scraping (allowed routes only) — legal-ish, shallow coverage.
  3. **Official MCP endpoints** — sanctioned, no-auth, read-only, but deliberately capped and **provenance-stripped**.

---

## 3. Official MCP Server (ChatGPT App) — Verified Live

HiringCafe ships a remote MCP server as its ChatGPT app backend. All facts below were **verified first-party on 2026-10-13** via `curl` against the endpoint (the code snippet is published in HiringCafe's own docs).

### 3.1 Server basics

| Property | Value |
|---|---|
| Endpoint | `https://hiringcafe.com/api/chatgpt/mcp` |
| Transport | Streamable HTTP, JSON responses; **POST-only** (GET returns "Method not allowed" — documented) |
| Auth | **None** (no account, no sign-in, no token) |
| Methods | `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `ping` |
| UI | MCP Apps UI resource: job-card carousel (ChatGPT in-app) |
| Posture | Read-only by design (no apply, save, alert, account ops) |

### 3.2 Tools

| Tool | Purpose | Caps / notes |
|---|---|---|
| `search_hiringcafe_jobs` | Keyword + structured-filter search over live postings | **1–25 results per call** (default 10); **one result per company** so one employer cannot flood the results; filters-only searches return newest-first |
| `get_hiringcafe_job_details` | Full posting: complete description, requirements, compensation, HiringCafe apply link | Accepts the job ID from search results or a hiring.cafe URL slug |
| `show_hiringcafe_job_cards` | Renders ≤10 selected results as a card carousel (MCP Apps UI resource) | Presentation only |

### 3.3 Verified response shape (`tools/call`, `search_hiringcafe_jobs`, n=1)

Structured content beside the markdown text (the **dual markdown + `structuredContent` response pattern** is worth copying):

```jsonc
{
  "jobs": [{
    "id": "plnatdilkz379d0c",               // 16-char slug token, stable in URL
    "title": "Senior Staff Machine Learning Systems Engineer, Ads ML Platform",
    "company": "Reddit",
    "location": "United States",
    "workplaceType": "Remote",
    "employmentTypes": ["Full Time"],
    "compensation": {"currency": "USD", "frequency": "Yearly",
                     "minimum": 292500, "maximum": 409500},
    "datePosted": "2026-08-25T18:40:31.000Z",
    "requirementsSummary": "Requires 8+ years in infrastructure, ...",   // LLM-written
    "seniorityLevel": "Senior Level",
    "minimumYearsExperience": 8,             // LLM-extracted
    "responsibilities": ["owning strategy", "defining architecture", "mentoring engineers"],
    "technologies": ["Apache Spark", "Apache Flink", "Apache Kafka", "Ray",
                     "Apache Airflow", "Apache Iceberg", "Kubernetes",
                     "Google BigQuery", "Snowflake", "Databricks"],
    "applyUrl": "https://hiringcafe.com/job/senior-staff-machine-learning-...\
                 ?utm_source=chatgpt&utm_medium=llm-app&utm_campaign=mcp"
  }],
  "count": 1,
  "total_matches": 8722
}
```

Observations:

- `applyUrl` is a **HiringCafe URL with attribution params**, not the employer's ATS link — matches the privacy note that tool responses "exclude employer apply URLs, ATS identifiers, board tokens, member signals". Provenance is **deliberately de-scoped at the MCP layer**.
- The `technologies`, `responsibilities`, and `minimumYearsExperience` fields are the output of the LLM extraction pipeline described in §1 — i.e., **a free, at-scale reference implementation of exactly what experiment 003 built** (section weighting aside, their field vocabulary is the target vocabulary for job-posting LLM extraction).
- `total_matches` (8,722 for "machine learning engineer") implies the live index is large and the MCP search is a thin ranked window over it, not a corpus.

### 3.4 Limits (their stated policy)

- Only active public postings; jobs may expire after return.
- Card carousel ≤ 10; search ≤ 25 per call; one company per result.
- No accounts, resumes or resume-aware matching through the app.
- **"Heavy automated use is not supported; the app is intended for normal conversational job search"** — this is the hard ceiling that disqualifies the official MCP as a corpus-ingestion path (§9).

---

## 4. The Filter Taxonomy (AI-Structured Schema)

`search_hiringcafe_jobs` exposes the richest structured filter vocabulary observed on any job source (verified against the live `tools/list` schema). This is the **state-of-the-art field dictionary** for LLM-extracted posting metadata; it is the benchmark our own extraction layer (experiment 003 lineage) should be compared against.

| Family | Fields (verified schema) |
|---|---|
| Query & place | `search`, `location` (city/state/country/`remote`), `radius_miles` (default 50; ignored for state/country) |
| Work arrangement | `workplace_types` (Remote/Hybrid/Onsite), `employment_types` (7: Full Time … Volunteer), `seniority_levels` (No Prior Experience Required/Entry/Mid/Senior), `role_type` (Individual Contributor / People Manager) |
| Pay & experience | `min_salary` + `salary_frequency` (Yearly/Monthly/Weekly/Daily/Hourly), **`posted_salary_only`** (only employer-published ranges, excluding HiringCafe estimates), `max_years_experience`, `posted_within_days` |
| Employer | `company_names` / `exclude_company_names`, `min/max_employees` (headcount), `organization_types` (public/private/non-profit/government/cooperative-mutual), `funding_stages` (23 values, Pre-Seed → Post-IPO Debt/ICO), `min_funding_amount`, `funded_since_year`, `investors` / `exclude_investors`, `industries` (32), `company_hq_countries`, `min/max_year_founded`, `company_keywords` / `exclude_company_keywords` |
| Role substance | `departments` (31, from job function), `benefits_and_perks` (visa sponsorship, four-day work week, 401(k) matching, parental leave, relocation, …), `encouraged_to_apply` (veterans, fair-chance), `security_clearances`, `application_form_ease` (Simple/Time Consuming) |
| Education & credentials | `associates/bachelors/masters/doctorate_degree_requirement` (Required/Preferred/Not Mentioned) + matching `_fields_of_study`, `licenses_and_certifications` (+ excludes; e.g. RN, CDL), `language_requirements` (+ `language_requirements_operator` AND/OR) |
| Physical demands | `air/land_travel_requirement`, `on_call_requirements`, `morning/evening/overnight_shift_work`, `weekend_availability`, `holiday_availability`, `overtime`, `physical_environments` (Office/Outdoor/Vehicle/Industrial/Customer-Facing), `physical_labor_intensity`, `physical_positions` (sitting/standing), `oral_communication_levels`, `computer_usage_levels`, `cognitive_demand_levels` (Low/Medium/High) |

Design details worth stealing:

- **`posted_salary_only` semantics** — the platform distinguishes employer-published pay from its own estimates; the datahamster actor codifies the same rule ("HiringCafe's internal estimate is never passed off as a real range"). Our schema's `salary_min/max` should carry the same provenance flag.
- **"Not Mentioned" tri-state** for degree requirements — the correct handling for LLM extraction of negative information (vs. our binary fields).
- **Agentic filter composition**: there is no `pre-IPO` value in the data; the docs show ChatGPT composing it from `organization_types=["private"]` + excluding both Post-IPO funding stages. Complex criteria are meant to be **composed by the LLM client**, not enumerated by the server.
- **One job per company** as a server-side de-noise rule, not a client-side afterthought.

---

## 5. Official Claude Connector

- `hiringcafe.com/documentation/hiringcafe-claude-connector` exists (sitemap `lastmod 2026-10-07`) — a companion connector for Claude, parallel to the ChatGPT app.
- The page returns **403 to datacenter fetches** (verified 2026-10-13 with and without a browser `User-Agent`; Cloudflare fronting) — details unverified from here. The pattern matches Indeed's beta MCP, which is likewise Claude-Connector-only (docs.indeed.com). Given the ChatGPT endpoint is already the same server with no auth, assume a Claude-native tool surface with equivalent caps.

---

## 6. Third-Party MCP Surfaces

### 6.1 Community: `maazghani/hiring-mcp.cafe` (verified live)

A minimal, **no-auth remote MCP** (Next.js + Vercel `mcp-handler`) that searches HiringCafe's **public HTML pages**. Status page live at `hiring-mcp.cafe` (verified 2026-10-13).

- **Read path:** fetches public routes over plain HTTP and parses the server-rendered `__NEXT_DATA__` JSON island with cheerio — "far more robust than scraping CSS class names". All navigation lives in one slug module (`lib/slug.ts`): `query + location → /jobs/{query-slug}-{location-slug}`, so requests go to **path-based routes that `robots.txt` allows** (`Allow: /jobs/`, `/job/` — verified §8), not the disallowed `?searchState=` search URLs.
- **Tools:** `search_jobs` (query/location → first result page), `get_job` (parse one `/job/...` page), plus the ChatGPT-compatible pair `search` (compact citation-friendly `{id,title,url}`) and `fetch` (full text + metadata). **Ids are base64url-encoded job URLs**, so `fetch` works even on a cold serverless instance with an empty cache.
- **Posture:** no auth, no DB, no browser automation, no background crawl, **first page only**, hard timeouts on outbound fetches, page content treated as untrusted text, in-memory cache per warm instance.
- **Risk:** zero stars/forks, single maintainer, private deployment behind one Vercel account — do not build on it; read it as a design specimen. Its whole utility is the pattern *wrap a server-rendered public page in 4 thin MCP tools*.

### 6.2 Commercial: Apify actors with MCP endpoints

Every Apify actor is exposed as an MCP tool via `https://mcp.apify.com/?tools=<actor>` (OAuth / bearer). Three distinct postures toward HiringCafe:

| Actor | Path & posture | Coverage & fields |
|---|---|---|
| `datahamster/hiringcafe-jobs` | **Robots-respecting**: reads only allowed routes — the `/recently-posted-jobs` feed (3 pages × 200 postings ≈ a few hundred/day) plus `/job/` detail pages | Per row: `id`, `url`, `title`, `company`, `location`, `isRemote`, `workplaceType`, `employmentType`, `seniority`, `jobCategory`, `salaryMin/Max/Currency` (**real ranges only**), `postedAt`, `applyUrl` (**employer's own link**), **`ats`, `boardToken`, `dedupId`**, `description`; monitor mode + webhook/Telegram alerts; pay-per-row |
| `parsebird/hiring-cafe-scraper` | **Adversarial**: stealth Chromium (Patchright) on US residential proxy, waits out the Cloudflare Managed Challenge (8–20 s, ≤6 retries), reuses challenge cookies to call HiringCafe's **internal search API**, paginates ~1,000/page | ≤1,000 records/run, 30+ fields (id, salary, seniority, commitment, `technicalTools`, `applyUrl`, `boardToken`, `source`, …) |
| `haketa/hiring-cafe-scraper` | **Rendered-page capture**: natural-language semantic queries or paste any filtered hiring.cafe URL (`?searchState=...`) and paginate it | **90+ AI-structured job fields + 20+ company fields** (funding type/year/amount/investors, direct-employer vs agency, degrees, licenses, `requirementsSummary`, `roleActivities`, `processedJobData`) |

Notes:

- The `datahamster` schema proves the **provenance pair exists publicly per job page** (`ats` + `boardToken` + a cross-source `dedupId`), completing the join key story from Report 013 §7: a hiring.cafe record can be joined back to its origin ATS posting — but only via page-level scraping, not via the official MCP.
- The `parsebird` posture is exactly what Report 013 §8 calls "anti-bot escalation"; expect breakage if the challenge rotates. Not a model to copy.

---

## 7. Field Coverage vs. Report 013 Sources

| Field class | ATS boards (013: Greenhouse/Lever/Ashby) | HiringCafe official MCP | HiringCafe `/job/` page (via scrapers) |
|---|---|---|---|
| Description | ✅ (Greenhouse `content=true`, Ashby HTML+plain) | ✅ full via `get_hiringcafe_job_details` | ✅ |
| Structured salary | ✅ Ashby/Lever only, sparse elsewhere | ✅ **always** (LLM-extracted; `posted_salary_only` filter to keep real ranges) | ✅ + period + real-vs-estimate flag |
| LLM-extracted extras (tools, responsibilities, years of experience, requirements summary) | ❌ | ✅ | ✅ (90+ fields on page data) |
| Company enrichment (headcount, funding, investors, HQ, direct-employer flag) | ❌ | ✅ as filters | ✅ (20+ fields) |
| **ATS provenance** (`ats`, `boardToken`, employer `applyUrl`) | ✅ (it *is* the ATS) | ❌ **deliberately stripped** | ✅ |
| Pagination | ❌ mostly whole-board; Lever skip/limit | ≤25/req, 1 company/req | feed pages of 200; search ≈1,000/page (API) |
| Auth | none | none | none |
| Poll-friendly whole-source read | ✅ one call per board | ❌ capped, windowed | ⚠️ only the daily feed; user-URL search disallowed by robots |

Take-away: HiringCafe is a **faucet with a limiter**. It is the only source that hands back pre-extracted structured metadata at 5M-posting scale, and the only one that equally *refuses* to hand back the provenance keys our dedup design needs (`<ats>:<board>:<id>` `source_id` scheme).

---

## 8. Robots, ToS, and Rate Posture

Verified `hiring.cafe/robots.txt` (2026-10-13), canonical domain `hiringcafe.com`:

- **Allowed:** `/`, `/about`, `/work-here(/)`, `/careers(/)`, `/jobs(/)`, `/job(/)`, `/recently-posted-jobs(/)`, `/insights(/)`, `/talent-network`, `/ai-search/$` (landing page only).
- **Disallowed:** `/*?searchState=*`, `/*?page=*`, `/*&page=*` (the real search + pagination), `/viewjob/`, `/org/`, `/company/`, `/b/`, plus non-canonical mirrors `/req/` (mirrors `/job/`) and `/board/` (mirrors `/b/`), admin/account/applicant areas, and the `/ai-search/` chats ("Only the AI Search landing page is public to crawlers; chats and demos are not").
- Hostile agents: `meta-externalagent` and `Applebot-Extended` disallowed from everything.
- **Five published sitemaps:** `sitemap.xml`, `jobs-sitemap.xml`, `job-search-sitemap.xml`, `job-posting-sitemap.xml`, `priority-jobs-sitemap.xml`, `vip-jobs-sitemap.xml` — an explicitly-crawler-blessed URL layer (root sitemap verified; the large job sitemaps were not walked in this report — follow-up item).
- **Cloudflare** fronting the DOM/documentation surface (verified 403s above); the search API behind the site sits behind the Cloudflare Managed Challenge per parsebird.
- **ToS:** no public API; scraping not invited (jobspipe, 2026). The official MCP's own limit section — "heavy automated use is not supported" — is the closest thing to a quantitative statement of what agent traffic is welcome. The friendly surface is: official MCP (conversational) + allowed path routes + published sitemaps. The unfriendly surface is: searchState API scraping of any depth.

---

## 9. Recommendations for This Project

1. **Use the official MCP as an *agent-access and eval* surface, never as ingestion.** The caps (25/req, one-per-company, no employer URLs, no ATS ids, "heavy automated use not supported") make corpus-building impossible by design. What it *is* good for: (a) live-lookup demos of a working job search pipe; (b) a golden-set curation assistant (pull top-N real postings per emerging taxonomy role during [golden-set maintainment](../technical/evaluation.md)); (c) borrowing the response contract (below).
2. **Treat their filter taxonomy as the field-dictionary benchmark** for our LLM extraction line (experiment 003 and successors). Their 90+ field schema — tri-state degree requirements, license/cert list, funding stage enum, physical/cognitive demand levels, `posted_salary_only` flag — is the most complete published target we're aware of; diff our extraction output against it field-by-field when scoping the next extraction experiment.
3. **For provenance discovery, stay with dataroles/ATS-first (Report 013).** HiringCafe-withholds-provenance at MCP level means it cannot seed our `<ats>:<board>:<id>` scheme; the `ats`/`boardToken` fields exist only on scrapable pages that robots' ToS posture makes a poor source for us (stage a small `robots`-compliant probe only if a specific missing-company lookup ever needs it).
4. **Steal three interface patterns** if/when the platform exposes an agent door: (a) no-auth remote endpoint with a read-only snowflake-free tool set; (b) `structuredContent` beside human-readable markdown in every tool result; (c) server-side noise rules ("one per company") plus LLM-composed compound filters instead of enum bloat.
5. **Note the recently-posted feed's composition** for corpus realism: our verified fetch of `/recently-posted-jobs` was dominated by US public-sector and education employers (school districts, municipalities, military civilian roles) — the AI/tech signal sits deeper in the index and is only reachable through the capped search surfaces. Corroborates Report 013's "curators are a discovery aid, not a primary source" conclusion.
6. **Do not adopt the community MCP** (`hiring-mcp.cafe`) as a dependency; read `lib/slug.ts` + its `__NEXT_DATA__` island parsing as a reusable technique for any future source that renders data server-side.

---

## 10. Sources

- Official HiringCafe MCP (first-party verified 2026-10-13): `POST https://hiringcafe.com/api/chatgpt/mcp` — `tools/list` (full input schema), `tools/call` (`search_hiringcafe_jobs`, n=1, Reddit/Reddit result quoted in §3.3); HiringCafe ChatGPT-app documentation — hiringcafe.com/documentation/hiringcafe-chatgpt-app (tools, filter families, privacy, limits); Claude Connector page — hiringcafe.com/documentation/hiringcafe-claude-connector (403 to automated fetch, sitemap lastmod 2026-10-07)
- HiringCafe platform: Business Insider — "Ex-Meta Staffer and Stanford Ph.D. Student Build Job Site Hiring Cafe" (2024-08-26); blog.hiring.cafe — "Scaling HiringCafe from 0 to 1M+ users" (2025-09-17) and "Talent Agent Is Out of Beta, and We Raised a Funding Round" (2026-09-28, $6.8M/Spark Capital, 5M+ jobs, 2M MAU); HiringCafe LinkedIn posts (2023, aggregation model + stack: NextJS, GCP, Firestore, OpenAI); simplify.jobs HiringCafe profile (2026 AI Talent Agent)
- Access policy: hiring.cafe/robots.txt (fetched 2026-10-13, quoted in §8); jobspipe.dev/blog/hiring-cafe-api ("no public API", 2026)
- Community MCP: github.com/maazghani/hiring-mcp.cafe (README, tools, slug/cache/safety design) + live status page at hiring-mcp.cafe (verified 2026-10-13)
- Apify actors: datahamster/hiringcafe-jobs (feed monitor, `ats`/`boardToken`/`dedupId` fields), parsebird/hiring-cafe-scraper (Cloudflare bypass description, field list), haketa/hiring-cafe-scraper (90+ AI-structured fields, company enrichment) — apify.com, 2026
- Ecosystem context: Indeed Partner Docs MCP (beta, docs.indeed.com, 2026); startup.jobs/mcp (official MCP, 8 read-only tools); gighq.ai/mcp-server (OAuth MCP); mcpservers.org listing for @servation/job-search-mcp (local stdio job-search assistant)
