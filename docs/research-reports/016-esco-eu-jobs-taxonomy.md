# ESCO — The EU Taxonomy of Occupations and Skills — Structure, Access, and Relevance Research Report

> Date: 2026-10-16
> Context: Semantic Career Mapping Platform — research into ESCO (European Skills, Competences, Qualifications and Occupations), the EU's free multilingual taxonomy of occupations, skills and competences, as a candidate external anchor for the project's semantic partitioning, taxonomy label generation, skill normalization (see [Specs](../specs/)), and evaluation. Covers structure, versions, download/API access paths, licensing, how NLP practitioners map jobs to it, and known limitations for tech-talent datasets.

---

## Table of Contents

1. [What ESCO Is](#1-what-esco-is)
2. [Classification Structure](#2-classification-structure)
3. [Versions and Maintenance](#3-versions-and-maintenance)
4. [Access Paths: Downloads, API, Crosswalks](#4-access-paths-downloads-api-crosswalks)
5. [Licensing and Provenance](#5-licensing-and-provenance)
6. [How Practitioners Use ESCO (Skill Extraction & Job Mapping)](#6-how-practitioners-use-esco-skill-extraction-job-mapping)
7. [Limitations and Operational Risks](#7-limitations-and-operational-risks)
8. [Alternatives and Complements](#8-alternatives-and-complements)
9. [Recommendations for This Project](#9-recommendations-for-this-project)
10. [Sources](#10-sources)

---

## 1. What ESCO Is

ESCO is the **European multilingual classification of Skills, Competences, Qualifications and Occupations**, developed by the European Commission's DG Employment, Social Affairs and Inclusion (DG EMPL) with Cedefop support and Member State stakeholders. It "works like a dictionary" describing and interlinking occupations and skills relevant to the EU labour market, free of charge and in an open IT format (DG EMPL / data.europa.eu).

Three use cases drive the design (ESCO portal, *Use ESCO*):

| Use case | Examples |
|---|---|
| Job matching / searching | EURES job mobility portal, vacancy-standardization, autocomplete |
| Career guidance & learning management | Europass CVs, LMS curricula, linked learning outcomes |
| Research & big data labour-market analysis | Skill extraction from job ads, skills intelligence, statistics |

Two structural ideas matter for us:

- **Occupations ≠ jobs.** An occupation groups jobs with similar tasks/skill sets ("project manager" is an occupation; "project manager for the ventilation system of a given aircraft model" is a job). ESCO contains occupations only — usable as vacancy titles.
- **Linked Open Data.** Every concept has a persistent URI (e.g. `http://data.europa.eu/esco/occupation/<uuid>`), and known concept schemes exist for the pillars (`.../esco/concept-scheme/occupations`, `.../skill`, `.../isco`). SKOS serves as the data model, so ESCO merges cleanly with graph/URI-keyed pipelines (marketplace: ESCOpedia, MSI Dublin Core directory).

**Scale (v1.2.1, portal snapshot 2026-10-16):** 3,039 occupations; 13,939 skill/knowledge concepts; 28 language packs (24 official EU languages + Icelandic, Norwegian, Ukrainian, Arabic; national sign languages included since v1.2). The first full release, v1, shipped in 2017 with 2,942 occupations (Engineers Europe background note).

---

## 2. Classification Structure

### 2.1 Occupations pillar

- Built **on ISCO-08**: ISCO provides the top 4 hierarchy levels (the 10 major groups "0 Armed forces" … "9 Elementary occupations"), ESCO occupations sit **at level 5 and below**.
- **Every ESCO occupation is mapped to exactly one ISCO-08 code** — ISCO is therefore a free, ready-made multi-granularity ancestor hierarchy for any ESCO occupation.
- Each occupation carries: preferred term, non-preferred (alternative) terms, hidden terms per language; description, scope note, definition; regulatory info; and an **occupational profile** listing related knowledge/skills/competences (essential vs optional relation types).

### 2.2 Skills pillar

- In v1.2.1 organized as a **mono-hierarchy of four sub-classifications**: Knowledge (K), Skills (S), Transversal skills (T), Language skills and knowledge (L). Skill groups occupy the top 3 levels; ESCO skills sit at level 4 and lower; each concept belongs to exactly one group.
- Concepts are typed `skill/competence` or `knowledge` (no distinction between "skill" and "competence" itself).
- Every concept has a **reusability level**: transversal / cross-sectoral / sector-specific / occupation-specific — a useful built-in granularity/importance axis (e.g. "utilise machine learning" is sector-specific).

### 2.3 Thematic labels & collections

Sub-collections and overlay labels ship as separate files (or portal filter facets):

| Collection | What it marks |
|---|---|
| `digitalSkillsCollection` | ESCO skills/knowledge labelled digital per DigComp-style definition (1,201 concepts at labelling time) |
| `digCompSkillsCollection` | Mapping of DigComp's 21 competences into the skills pillar |
| `greenSkillsCollection` + `greenShareOcc` | Green-transition skills; green-skill *share* per occupation profile |
| `researchSkillsCollection` / `researchOccupationsCollection` | Research occupations mapped to ResearchComp (the researcher competence framework) |

Thematic labels (DigComp / Digital / Green / Research / Gender-considerations) are first-class facets in the portal and CSVs — a cheap way to gate "digital/Green/Research" features.

---

## 3. Versions and Maintenance

| Version | Date | Nature |
|---|---|---|
| v1.0.x | 2017 | First full classification (2,942 occupations) |
| v1.1.x | 2022 | Minor: quality improvements, AI/data-science focus per Annual Report 2021 |
| v1.2 | May 2024 | Major: green/digital & emerging-tech focus; **+35 occupations, +42 skills, +196 knowledge concepts; 677 alt labels, 96 hidden terms; 12,000+ concepts updated** among 18 quality tasks |
| v1.2.1 | Dec 2025 | Minor: translation review, de-duplication, occupation–skill link quality (trimming over-rich occupations), green-label relationship revision, new alt labels |

Operationally relevant facts (ESCOpedia, news pages):

- **Minor ≠ additive**: minor releases change labels/relations, not concept counts — safer to pin a minor for stable downstream IDs.
- **Delta CSV files** document every change between consecutive versions (filterable by change type, language) — the mechanism to track migrates pinned versions.
- Since **v1.2 ESCO maintenance is itself AI-driven**: a data-driven pipeline combines human expertise with AI extraction of occupation/skill terms from job ads, courses, and publications, then links them to existing concepts via customized AI models. The Commission also runs an ESCO AI research track (Skills-Occupations Matrix tables, occupation–skill share stats per profile).
- The ESCO web-service API currently documents **v1.0.9 as its default version**, with newer versions selectable via the `selectedVersion` parameter — the API surface **lags the downloadable v1.2.1 dataset** (verified portal download selection vs API version notes, Oct 2026).

---

## 4. Access Paths: Downloads, API, Crosswalks

### 4.1 Bulk downloads (primary path)

Per-language packages (CSV or ODS; also RDF/TTL, JSON-LD, XML) from the [Download page](https://esco.ec.europa.eu/en/use-esco/download), versions v1.0.3 → v1.2.1. A package contains **19 files** (ESCO portal, *Structure of ESCO Downloadable Datasets*), in four groups:

| Group | Files |
|---|---|
| Occupations pillar | `ISCOGroups`, `occupations` (level 5+) |
| Skills pillar | `skills_<lang>`, `skillGroups`, `skillsHierarchy`, `languageSkillsCollection` |
| Relations (URI-keyed, language-independent) | `broaderRelationsOccPillar`, `broaderRelationsSkillPillar`, `occupationSkillRelations`, `skillSkillRelations` |
| Thematic | `digitalSkillsCollection`, `digCompSkillsCollection`, `greenSkillsCollection`, `greenShareOcc`, `researchSkillsCollection`, `researchOccupationsCollection` |
| Meta | `conceptSchemes`, `dictionary` |

Shape notes (structure page):

- The **URI is the merge key**; relation files hold URIs only — you must join them with the per-language files to render labels.
- To rebuild the full skills tree: join `skills_<lang>` + `skillsHierarchy_<lang>` + `broaderRelationsSkillPillar_<lang>`.
- Community mirrors exist for experimentation (e.g. GitHub `JakubJagielski/esco-graph-explorer` hosting the v1.2.1 en CSV pack; `skills_en.csv` alone is ~8.9 MB) — but pin official downloads for anything durable.

### 4.2 ESCO web-service API (secondary path)

- Base: `https://ec.europa.eu/esco/api`, REST/JSON responses in **HAL** format, hypermedia-driven (`_links` navigation; official docs: `esco_api_doc.html`, OpenAPI v3 definition, PDF).
- Main resource classes: `occupation`, `skill`, `concept` (generic), browsable concept schemes for the occupations / skills / ISCO hierarchies and supporting vocabularies (skill types, reuse levels, ICT + transversal groups).
- Notable operations: text **search** (`/search` with `type`, `language`), resource fetch by **URI**, hierarchy browse, and **version-conversion endpoints** (`conversionOccupationGet` / `conversionSkillGet`) that translate a concept URI to its state in another ESCO version (returns collections when a concept was split).
- `language` and `selectedVersion` parameters control the response language/version. **Gotcha: the default served version is v1.0.9** — always pin `selectedVersion`, and do not expect v1.2.1 parity via API.
- An **ESCO Local API** (downloadable service) exists for self-hosting/performance; the portal notes the current local-API build is reference-only pending replacement.

### 4.3 Crosswalks

- **O*NET ↔ ESCO crosswalk** — joint EC/US-DOL effort; downloadable crosswalk files plus a live `escocrosswalk` search in O*NET Web Services API 2.0 (v1.1.0 data). The bridge to the US SOC/O*NET world.
- **EURES Countries Mapping Tables** — per-country national-classification → ESCO occupation mapping tables, published on the ESCO portal.
- Qualifications live in **Europass/EQF**, not in ESCO proper; linking of qualification learning outcomes to ESCO skills is documented as a separate workflow.

---

## 5. Licensing and Provenance

- **ESCO itself**: free of charge for any purpose ("downloaded, used, reproduced and reused for any purpose") under the Commission Decision 2011/833/EU reuse rules — conditions: acknowledge the source, no implication of endorsement, disclose modifications. The Commission has since standardized **CC BY 4.0** for EU-owned content, and the portal's copyright notice states the ESCO content is licensed CC BY 4.0 unless otherwise indicated. No API key, no contract.
- **Inherited upstream content** (must credit through the copyright notice):
  - The ESCO skill hierarchy partially adapts **O*NET** (US DOL/ETA) — CC BY 4.0 — and the Canadian **Skills & Knowledge Checklist** (Government of Canada).
  - The transversal-skills hierarchy draws on OECD competency work and Bertelsmann Stiftung "skills of the future" cards (CC BY-SA 4.0).
- **Attribution string** expectation: "© European Union, [year], ESCO v1.2.1 … reuse authorised, provided the source is acknowledged."
- Practical consequence for this repo: attribution goes in the data manifest / configs provenance layer (aligning with the object-store metadata+manifest design, see [Research Report 011](011-object-store-metadata-design.md)).

---

## 6. How Practitioners Use ESCO (Skill Extraction & Job Mapping)

ESCO is the de-facto default skill/occupation vocabulary in European NLP-for-labour-market research. Directly reusable findings:

| Work | Method | Result of note |
|---|---|---|
| Senger, Zhang, van der Goot, Plank (2024), *Deep Learning-based Computational Job Market Analysis* | Survey of skill extraction & classification from postings | Canonical taxonomy of datasets/methods; standardizes hard/soft-skill terminology — the entry point for the field |
| Zhang et al. (2022), *Skill Extraction with Weak Supervision* (RecSysHR) | ESCO skills as weak-supervision spans (RoBERTa/JobBERT cosine similarity against ESCO n-grams) | Removes the need for manual span annotation — cheap labelled-data bootstrap |
| Decorte (Future Skills/JobBERT line), Clavié et al. | Extraction + ESCO mapping; zero-shot LLM mapping | LLMs map novel skill mentions onto ESCO acceptably for normalization |
| SkiLLens (EACL 2026 Industry) | 18M Eurostat Web-Intelligence-Hub ads, 27+1 countries; embedding-based extraction + LLM-assisted validation/ESCo mapping | The largest demonstrator of multilingual ESCO mapping at internet scale |
| SkiLLMo (WWW 2025) | BERT segment detection + LLM normalization onto ESCO | Long-ads pipeline; transformer-first segmentation then LLM mapping |
| ES-CoE "Making sense of skills" | NER extraction, mapping extracted entities to nearest ESCO/Lightcast skill | Only 6% of extracted entities inappropriate; **88% of extractions mapped well to ESCO**; unknown skills still catchable ("React" → "use scripting programming") |

Patterns that recur across all of them:

1. **Extract → normalize**: candidate mentions retrieved against the *alt-label/hyphenation-expanded* ESCO vocabulary (cosine of embeddings + n-gram match), then optionally LLM-adjudicated.
2. **Titles map to occupations semantically, not lexically**: the Commission's own AI deck illustrates `"C++ programmer" ⇒ "software developer"` — description-based semantic search over ESCO occupation profiles beats exact-match on preferred labels.
3. **Reuse levels prune noise**: transversal/cross-sectoral skills are generic; occupation-specific skills are discriminative — the same filtering logic our section-weighting experiments apply (see [Research Report 001](001-embedding-optimization-research.md)).
4. **Hierarchy enables level-collapsing**: mapping to a specific skill fails → fall back to its skill group / ISCO parent per the ES-CoE "match at different levels" strategy.

---

## 7. Limitations and Operational Risks

- **Tech-role granularity is thin.** ~3,039 occupations for the entire EU labour market means the AI/data/ML segment is a handful of ISCO-25 occupational leaf nodes; the novel "MLOps/Prompt-engineering" middle ground is unlikely to be an ESCO leaf. Mitigation: ESCO anchors *broad clusters*; fine distinctions need our custom taxonomy (see [Report 003](003-ml-ai-responsibility-taxonomy.md)).
- **Novel-skills lag.** Expert-derived vocabularies trail the market — ES-CoE had to map to a *nearest* skill ("React" absent); validation-cycle cadence (major ~2 years, yearly minor) is slow for AI jargon.
- **API version lag & default-version trap**: web-service default v1.0.9, newest downloadable v1.2.1 — mixing sources silently yields label drift. Pin the version in configs; prefer the snapshot over live API at runtime.
- **Alternative labels explode**: alt/hidden lists are multi-K tokens of noise/gender-variants; useful for recall in matching, dangerous as retrieval corpus entries without dedup (the exact v1.2.1 de-dup wave shows they know it too).
- **Language pack asymmetry**: not all 28 packs have equally complete alt-label baskets; cross-language label parity is not guaranteed (v1.2.1 specifically patched "no English alternative terms identified" cases).
- **Occupational profiles ≠ vocabulary**: relation files are URI-only; no ready-made per-language occupation↔skill text join — some offline assembly required (structure page explicitly says you craft it).
- **Regulated-professions metadata** matters only for ~EU regulated occupations; irrelevant for AI/data talent but beware when ingesting other segments.

---

## 8. Alternatives and Complements

| Taxonomy | Scope | Trade-off vs ESCO |
|---|---|---|
| **ISCO-08** (ILO) | Occupation classes, 4 levels, no skills | Coarser, universal; it *is* ESCO's spine — use as fallback ancestor labels |
| **O*NET-SOC** (US DOL) | ~1,000 detailed occupations + rich skill/ability/technology scales | Deeper numerically but US-flavored; ESCO↔O*NET crosswalk bridges both; our dataset is global/ATS-scraped, so both worlds appear |
| **Lightcast Open Skills** | Commercial, US-centric open-skills corpus, thousands of concepts | Finer-grained tech vocabulary; research use (per ES-CoE pairing with ESCO) is tolerated; licence less open |
| **National SOCs** (e.g. UK SOC 2020) | Country-specific | Fine locally, useless cross-border; EURES tables map them to ESCO |
| **Data-driven skill clusters** (ESCoE/Djumalieva–Sleeman 2018) | Build a taxonomy from co-occurrence embeddings | Fast refresh vs expert basis — conceptually our *evoc* cluster-labeling approach at skill level ([Report 012](012-evoc-embedding-clustering.md)) |
| **Custom functional taxonomy** (this project) | ML/AI responsibilities, 13 categories ([Report 003](003-ml-ai-responsibility-taxonomy.md)) | Fits our precision needs; zero external maintenance — keep as overlay on ESCO/ISCO anchors |

---

## 9. Recommendations for This Project

1. **Pin ESCO v1.2.1 (en) CSV snapshot as the external anchor.** Do not runtime-depend on the web API (lag v1.1.x/v1.0.9-default, latency, availability); instead fetch the official download once, hash it into the data manifest/provenance layer, and record attribution (Dec 2011/833/EU, CC BY 4.0) in configs. Skills ~9 MB + occupation/ISCO/relation files fit comfortably as a small, reproducible asset (fixtures stay synthetic per repo policy).
2. **Use ISCO levels 1–4 + ESCO level 5+ as a free multi-granularity label ladder** for [semantic-partitioning](../specs/) (evoc's multi-granularity cluster layers: [Report 012](012-evoc-embedding-clustering.md)): label each cluster at *multiple* cuts of the ESCO/ISCO hierarchy — coarse parent for navigation, leaf for precision, matching the skill-group-share statistics in occupation profiles.
3. **Mirror ESCO's ontology shape in our partition output** (URI-keyed nodes, preferredLabel + altLabels arrays, one ISCO parent each). This keeps our taxonomy swappable with ESCO, ISCO, or a future custom overlay and aligns with the [taxonomy-label-generation](../specs/) requirement of stable, addressable label concepts.
4. **Adopt SKOS-style normalization for employment queries**: match employment queries against URIs and alt/hidden-term baskets, not just preferred labels; join everything on URIs exactly as ESCO's own relation files do (mirrors the URI-keyed metadata design in [Report 011](011-object-store-metadata-design.md)).

5. **Leverage skills-side features cautiously.** `occupationSkillRelations` + `skills_en.csv` (typed skill/knowledge × reuse level × essential/optional) provide structured side-information for occupation profiling — beware concept drift across versions; use delta files to audit relationship stability between pinned upgrades.
6. **Golden-set enrichment only, not auto-generation.** ESCO occupation descriptions + ISCO ancestry are a legitimate human-audit aid for the golden set, but the repo forbids auto-generating golden labels — use ESCO to *propose* candidates for manual review.
7. **Bridge with O*NET** for any US-flavoured datapoints using the EC/USDOL crosswalk, so EU-anchored and US-anchored label spaces stay reconcilable.

---

## 10. Sources

Accessed 2026-10-16 unless noted.

1. ESCO Portal — [The ESCO Classification](https://esco.ec.europa.eu/en/classification), [Occupations](https://esco.ec.europa.eu/en/classification/occupation_main), [Skills & Competences](https://esco.ec.europa.eu/en/classification/skill_main)
2. ESCO Portal — [Structure of ESCO Downloadable Datasets](https://esco.ec.europa.eu/en/structure-esco-downloadable-datasets) (19-file list, URI-as-merge-key)
3. ESCO Portal — [Download](https://esco.ec.europa.eu/en/use-esco/download) (formats, versions, delta files), [Use ESCO](https://esco.ec.europa.eu/en/use-esco) (Linked Open Data, languages, use cases)
4. ESCO Portal — [ESCO v1.2 is live!](https://esco.ec.europa.eu/en/news/esco-v12-live) (May 2024), [ESCO v1.2.1](https://esco.ec.europa.eu/en/about-esco/escopedia/escopedia/esco-v121) (Dec 2025)
5. ESCO Portal — [ESCO API](https://esco.ec.europa.eu/en/about-esco/escopedia/escopedia/esco-api), [ESCO web-service API](https://esco.ec.europa.eu/en/use-esco/use-esco-services-api/esco-web-service-api); [REST API docs](https://ec.europa.eu/esco/api/doc/esco_api_doc.html) incl. PDF + OpenAPI v3
6. [Terms of Use / reuse decision 2011/833/EU FAQ](https://esco.ec.europa.eu/en/node/173); [Copyright notice — ESCO Skills & Competences](https://esco.ec.europa.eu/en/copyright-notice-esco-skills-competences) (O*NET/Canada provenance, CC BY 4.0); [Creative Commons: EC adopts CC BY/CC0](https://creativecommons.org/2019/04/02/european-commission-adopts-cc-by-and-cc0-for-sharing-information)
7. [O*NET Web Services — ESCO crosswalk search](https://services.onetcenter.org/reference/online/crosswalk/esco) (EC/USDOL joint crosswalk)
8. Senger, Zhang, van der Goot, Plank (2024) — [Deep Learning-based Computational Job Market Analysis survey](https://esco.ec.europa.eu/en/about-esco/publications/publication/deep-learning-based-computational-job-market-analysis-survey)
9. Zhang et al. (2022) — [Skill Extraction from Job Postings using Weak Supervision](https://ceur-ws.org/Vol-3218/RecSysHR2022-paper_10.pdf) (RecSysHR/CEUR-WS)
10. SkiLLens (2026) — [Recognising and Mapping Novel Skills from Millions of Job Ads](https://aclanthology.org/2026.eacl-industry.65.pdf) (EACL Industry Track, Eurostat WIH-OJA corpus); [SkiLLMo (2025)](https://dl.acm.org/doi/epdf/10.1145/3672608.3707960) (WWW companion)
11. ES-CoE — [Making sense of skills](https://www.escoe.ac.uk/projects/making-sense-of-skills) (NER + ESCO/Lightcast mapping, 88% mapping precision); Djumalieva & Sleeman (2018) — [An Open and Data-driven Taxonomy of Skills](https://www.escoe.ac.uk/publications/an-open-and-data-driven-taxonomy-of-skills-extracted-from-online-job-adverts)
12. European Commission ESCO team — [The future role of AI in ESCO](https://esco.ec.europa.eu/system/files/2021-07/77514867-b8e7-4ab1-a1dc-f9fb7956d934The_future_role_of_Artificial_Intelligence_in_ESCO_Joint_MSWG-MAI.pdf) (joint MSWG/MAI meeting deck, 2021); [ESCO Annual Report 2021](https://www.bollettinoadapt.it/wp-content/uploads/2022/09/KE-BW-22-001-EN-N.pdf) (v1.1 release, AI/data-science focus)
13. Engineers Europe — [Background note on ESCO](https://www.engineerseurope.com/sites/default/files/Background%20note_ESCO.pdf) (2019; v1 = 2,942 occupations)
