# Job Map Visualization — Interactive, Static-Hosted, Per-Point Information

> Date: 2026-10-26
> Context: Survey of visualization tools for the job-posting map deliverable — an interactive embedding map published on a static GitHub Pages site, where selecting any embedded point reveals project-chosen information about the posting — likely a LLM-generated "summary" layer sitting above the raw posting text, structured as one JSON object per embedded point. Extends [002 — Tutte Institute Tools](002-tutte-institute-tool-review.md) (which already recommended DataMapPlot for Step 1.2/1.11) through a sharper lens: per-point rich information panels and fully static hosting.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [GitHub Pages Constraints](#2-github-pages-constraints)
3. [Candidate Survey](#3-candidate-survey)
4. [Comparison Matrix](#4-comparison-matrix)
5. [Data Packaging Strategy](#5-data-packaging-strategy)
6. [Draft Data Contract: Per-Point Summary JSON](#6-draft-data-contract-per-point-summary-json)
7. [Recommendation](#7-recommendation)
8. [Licensing](#8-licensing)
9. [Decision Log](#9-decision-log)

---

## 1. Requirements

The target deliverable, derived from the MVP idea and the current dataset state (27 postings, `data/raw_embeddings.npy` at 27×384, all-MiniLM-L6-v2, UMAP-ready):

- **R1 — Interactive map.** Pan, zoom, and point picking over a 2D embedding data map, with cluster labels that behave sensibly at multiple zoom levels.
- **R2 — Per-point information panel.** Clicking (or tapping) any point opens content the project fully controls: title, company, extracted skills, the summary paragraph — whatever the summary layer contains. Not limited to what the charting library thinks a tooltip should be.
- **R3 — Static hosting.** The artifact must be plain files: HTML + JS + JSON + assets. No backend process, no user accounts, no external hosted SaaS holding the data. This is what GitHub Pages allows.
- **R4 — Python-generated artifact.** The map must be regenerable as the tail of the pipeline: scripts consume embeddings + summaries and emit the deployable artifact, so re-running the pipeline refreshes the map without touching application code.
- **R5 — Scale headroom.** 27 postings today; the sampling manifests (`data/sample_manifests/per_source*_seed7_bands5000_15000.json`) indicate the pipeline is designed toward thousands of postings. The tool must not collapse at 10³–10⁵ points, and the data payload must stay within GitHub Pages file/size limits.
- **R6 — Nice-to-haves.** Text search over postings, cluster/legend interactivity, mobile-friendly touch behavior.
- **R7 — Dependency posture.** CPU-first, minimal additional Python dependencies, license compatible with this project's MIT.

---

## 2. GitHub Pages Constraints

Hard numbers (GitHub's official Pages limits documentation, verified 2026-10):

| Limit | Value | Type | Consequence |
|---|---|---|---|
| Published site size | 1 GB | Hard | Effectively unlimited for JSON-based maps |
| Single file in repository | 100 MiB | Hard (blocked) | One-file inline data must stay under this |
| Git warning threshold | 50 MiB | Warning | Prefer data split well before this |
| Bandwidth | 100 GB/month | Soft | Generous for a report/deployment page |
| Builds | 10/hour | Soft | Not applicable when publishing via a custom GitHub Actions workflow |
| Deployment time | 10 minutes | Hard | Fast for our artifact size |

Implications:

- A single-file, all-data-inlined HTML map is viable up to roughly 10⁴–10⁵ postings (a ~5 MB JSON for 5k postings at ~1 KB each), comfortably under the 100 MiB hard block.
- Beyond that (or whenever cleaner separation is wanted), the data can live in separate JSON/binary files fetched at load time — the available fallbacks are covered in §5.

---

## 3. Candidate Survey

Tools are grouped by verdict. Titles marked ⭐ are the ones that came from the web review as genuinely fit for this shape of problem.

### Tier A — Fit

**⭐ DataMapPlot (v0.7+)** — *Primary candidate.* Tutte Institute library (same authors as UMAP/HDBSCAN). Python API (`create_interactive_plot`) produces a self-contained interactive HTML rendering via Deck.GL, a WebGL map engine, so pan/zoom and large point counts are handled by the underlying engine. Crucially, its interactivity model maps one-to-one onto the per-point-summary requirement:

- `hover_text` — one string per point for baseline tooltips.
- `extra_point_data` — a DataFrame of arbitrary extra columns, exactly one row per point. The project's per-point summary JSON drops in directly here.
- `hover_text_html_template` — an HTML tooltip template interpolating `{hover_text}` and any `extra_point_data` column by name (demonstrated in the official CORD-19 example with styled badges).
- `on_click` — a JavaScript action string per click, with access to `{hover_text}` and every `extra_point_data` column. Storing the summary fields in `extra_point_data` and rendering them into a custom panel div is the R2 mechanism.
- `dynamic_tooltip` with `DynamicTooltipManager` (`fetch_js`, `format_js`, `loading_js`, `error_js`) — the tooltip can fetch data live per point. This is the built-in path for "one JSON file per point, fetched when the user hovers/clicks" at large scale.
- `custom_html` / `custom_css` / `custom_js` — arbitrary extra page elements. The CORD-19 example builds a fully clickable custom legend whose JS reaches into `datamap.metaData.<column>` to drive selections — same plumbing a custom information panel would use.
- `selection_handlers` — built-in handlers (`DisplaySample`, `WordCloud`, `TagSelection`, …) for multi-select behaviour, subclassable for a "compare N selected postings" panel.
- Widget system (0.7): title, search, logo, minimap, histogram, layer toggles; corner or slide-out-drawer placement.
- `inline_data=False` + `offline_data_prefix` — splits point data into separate files instead of one HTML blob; plus a documented Offline Mode and system-font option, so the artifact can be made fully self-contained (no CDN calls) for GitHub Pages.
- Zoom-revealed multi-layer cluster labels, topic tree, cluster boundary polygons.

No tool surveyed matched this exhaustive list of bullet points against R1+R2+R4.

**⭐ Plotly.py → plotly.js** — *Fallback candidate.* `fig.write_html(...)` emits a static HTML file that works on GitHub Pages unchanged (proven pattern in many sites); `scattergl` (the WebGL trace type) handles ~10⁵–10⁶ points with pan/zoom/hover, using `customdata` for per-point fields. Click events (`plotly_click`) plus a few lines of custom JavaScript can fill a side panel div from `customdata` or an inlined JSON dictionary. Generic, extremely well documented, huge community. Costs over DataMapPlot: no built-in cluster label layers or zoom-revealed annotations, no search; information panels are DIY; aesthetics DIY; the resulting page is a chart, not a designed "data map".

**⭐ regl-scatterplot (+ Python wrappers jscatter / reglscatterpy)** — *Performance upper-end option.* A low-level WebGL scatter library rendering up to ~20M points with lasso selection, a pub/sub event API (`select`, `pointOver`), and explicit design for custom interaction code. `reglscatterpy.save_html(w, "map.html")` exports a self-contained interactive HTML snapshot (~0.5 MB bundle inlined). Chose over DataMapPlot only if: a) point counts reach millions (unlikely for a jobs map), or b) fine-grained custom lasso/encoding behaviour matters more than curated map aesthetics. No cluster labeling, no search; everything beyond points is DIY.

### Tier B — Partial fit (viable but weaker for R2/R4)

- **Bokeh** — standalone HTML with `CustomJS`/`TapTool` callbacks pointing click events at a div; mature and static-hostable. Weaker: heavier Python dependency tree, generic chart styling, per-point panel code entirely manual, no map/cluster aesthetics.
- **Vega-Lite / Altair** — declarative JSON spec embedded in a static page; built-in hover tooltips and selections for free; composes well into a docs site. Weaker: custom click-to-panel requires view-API JavaScript glue; rendering ceiling around 10⁴–10⁵ points before performance tuning.
- **Apache ECharts** — capable WebGL scatter with trivially simple click handlers and large-data mode; static hostable via CDN. Generic charting, same DIY-panel cost as Plotly, no data-map features.
- **Deck.GL directly** — the same engine DataMapPlot wraps. Maximum control, static-hostable, but demands an npm build pipeline and manual porting of label placement, tooltip math, and map aesthetics — i.e., re-implementing DataMapPlot. Viable at a later stage if the project outgrows the wrapper.
- **faerun** — generates standalone interactive HTML from Python (our scale, easy); but it is chemistry-infrastructure oriented, with 3D/Neighbourhood emphasis, hover-centric tooltips, and limited custom panel support.

### Tier C — Excluded, with reasons worth recording

- **Nomic Atlas** — polished hosted embedding-map SaaS (the pre- DataMapPlot answer), but data lands in Nomic's hosted infrastructure (with free-tier maps publicly exposed via Discovery Hub) and maps are embedded as iframes to their service. Violates R3 and R4 outright; also means posting text and summaries live outside the project's control.
- **TensorFlow Embedding Projector (standalone)** — classic tool; the standalone repo is now **ARCHIVED** (309 stars, no maintenance). Data model is checkpoints + TSV metadata only; per-point content is a label column, no custom HTML panel, no summary-JSON layer. Acceptable for exploration, not for this deliverable.
- **Nomic Deepscatter** — impressive (quadtree-tiled Arrow, zoomable to millions, runs statically on GitHub Pages); but its NC-CC-BY-SA license restricts to noncommercial use, and the tiling pipeline (`quadfeather`) plus JS build adds real weight. The wrong trade at this project's data size.
- **Kepler.gl** — geospatial (lat/lng) map-centric tool; embedding coordinates are abstract 2D space, not geography. Wrong abstraction.
- **fastplotlib** — GPU-accelerated Python plotting aimed at scientific workflows; primarily notebook/live infrastructure rather than standalone static HTML output.

---

## 4. Comparison Matrix

Ratings are relative to this project's R1–R7, at its current scale (27 postings → thousands).

| Criterion | DataMapPlot ⭐ | Plotly ⭐ | regl-scatterplot ⭐ | Bokeh | Vega-Lite/Altair | ECharts |
|---|---|---|---|---|---|---|
| R1 Pan/zoom/pick, cluster labels | ✅ best-in-class | ⚠️ chart only | ✅ no labels | ⚠️ | ⚠️ | ⚠️ |
| R2 Per-point custom panel | ✅ `extra_point_data` + `on_click` + templates + DynamicTooltip | ⚠️ DIY JS glue | ✅ DIY JS (full control) | ⚠️ CustomJS | ⚠️ view API glue | ⚠️ click handlers |
| R3 Static GitHub Pages artifact | ✅ self-contained HTML, offline mode | ✅ proven | ✅ save_html ~0.5 MB | ✅ | ✅ | ✅ |
| R4 Python-generated, rerunnable | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ JS-first |
| R5 Scale headroom | ✅ Deck.GL, ~10⁵–10⁶ | ✅ scattergl ~10⁶ | ✅ ~2×10⁷ | ⚠️ ~10⁵ | ⚠️ ~10⁵ | ✅ ~10⁶ |
| R6 Search/legend/mobile | ✅ built-ins | ❌ DIY | ❌ DIY | ❌ | ⚠️ | ⚠️ |
| R7 Deps/license | BSD-3; pandas+matplotlib (already in project's viz path) | MIT; one lib | MIT; small | BSD; larger stack | MIT; often via Altair | Apache-2 |

---

## 5. Data Packaging Strategy

Three-stage ladder, aligned to dataset growth and the §2 limits. DataMapPlot supports all three stages without changing the rendering layer:

1. **Inline (current, N ≤ ~10⁴).** `create_interactive_plot(..., inline_data=True)` — coordinates, cluster labels, and `extra_point_data` (the summary content) are embedded in one HTML file. One deployable artifact; zero fetch logic; trivially simple hosting. GitHub Pages file size: trivial at 27 postings, ~5 MB at 5k.
2. **Split (middle, ~10⁴–10⁵).** `inline_data=False, offline_data_prefix=...` — point data move to sequenced data files fetched by the page renderer on load. HTML stays small; data regenerate independently of layout; still fully static.
3. **Per-point fetch (large, ≥10⁵).** Summaries split to `points/<id>.json` (or an Arrow/binary layout) served statically; `dynamic_tooltip`/`DynamicTooltipManager(fetch_js=...)` fetches on hover or selection. Same ladder shape that Deck.GL-native (and Deepscatter-style quadtree tiling) uses, without leaving DataMapPlot's API.

Selection guidance: start at stage 1; promote on measured payload size, not in advance. All promotions change only the packaging step, not the deployed artifact's runtime shape.

---

## 6. Draft Data Contract: Per-Point Summary JSON

The summary layer is produced by the LLM pipeline (cleaned-text → structured summary) and must carry everything a click panel will ever want, in one bounded object per posting:

```json
{
  "id": "affirm-ai-solutions-engineer",
  "title": "AI Solutions Engineer",
  "company": "Affirm",
  "url": "https://careers.affirm.com/...",
  "role_category": "AI Engineer",
  "cluster_label": "AI/ML Platform & Infrastructure",
  "location": "Remote — US",
  "posted_date": "2026-10-20",
  "skills": ["python", "pytorch", "aws", "kubernetes", "llm-ops"],
  "seniority": "mid-senior",
  "summarized_text": "Owns agent and API build-out inside the People function: turns fragmented knowledge and manual processes into production AI tooling (agents, APIs, applications, infrastructure). Hands-on engineering with 3+ years of relevant experience."
}
```

Design notes:

- `summarized_text` is the summary-layer payload; raw posting text stays out of the map artifact (it can still be a link away via `url`).
- One row of this object maps one-to-one onto `extra_point_data`; the template columns become the panel content, interchangeably usable in `on_click`, `hover_text_html_template`, search (`search_field`), and selection handlers.
- Contract belongs to the deployment/rendering boundary, not the pipeline: incremental additions (e.g., salary, source board) are additive columns and require no rendering changes.

---

## 7. Recommendation

**Adopt DataMapPlot as the map renderer** (`create_interactive_plot`, packaged at its current state per §5 ladder), staying consistent with the existing decision in [002](002-tutte-institute-tool-review.md) but now justified for the specific per-point-summary + static-hosting requirement rather than aesthetics alone.

Minimal shape of the map-generation step (sketch, not a shipped script):

```python
import datamapplot
import pandas as pd

plot = datamapplot.create_interactive_plot(
    coords_umap,                        # (n, 2) from UMAP over raw_embeddings.npy
    cluster_labels_layer,               # colored by role_category / discovered clusters
    hover_text=[...titles...],
    extra_point_data=summaries_df,      # one row per posting, exactly the §6 contract
    hover_text_html_template=panel_template,   # title + company + skill badges
    on_click="document.getElementById('job-detail').innerHTML = `<div class=\"jd\">{summarized_text}</div>`",
    custom_html='<aside id="job-detail" class="detail-panel"></aside>',
    custom_css=panel_css,
    enable_search=True,
    inline_data=True,
)
plot.save("deploy/map/index.html")
```

Two edges to verify during implementation:

1. Brace escaping — `{summarized_text}` placeholders are interpolated server-side by DataMapPlot; any literal JavaScript braces inside `on_click` templates need `{{ }}` escaping (the JS template-literal interpolation above collides with this, so prefer non-template-literal JS or test carefully).
2. Heavy-tooltip feel — `summarized_text` in the hover template can be long; keep the hover template brief and reserve full summary rendering for the click panel.

Plotly `scattergl` stands as a fallback if DataMapPlot's curated aesthetic ever constrains layout needs; a Deck.GL reimplementation is the last resort, only after stage-3 packaging is in use.

---

## 8. Licensing

| Tool | License | Project-compatible? |
|---|---|---|
| DataMapPlot | BSD-3-Clause | ✅ |
| Plotly.py / plotly.js | MIT / MIT | ✅ |
| regl-scatterplot / jscatter / reglscatterpy | MIT / MIT / MIT | ✅ |
| Bokeh | BSD-3-Clause | ✅ |
| Vega-Lite / Altair | BSD-3 / MIT | ✅ |
| ECharts | Apache-2.0 | ✅ |
| Deepscatter | NC-CC-BY-SA | ❌ noncommercial-only |
| Nomic Atlas | SaaS, data leaves project | ❌ requirement violation |

---

## 9. Decision Log

| Decision | Rationale | Date |
|---|---|---|
| Renderer = DataMapPlot `create_interactive_plot` | Only surveyed tool with native per-point data (`extra_point_data`), click/tooltip customization (`on_click`, HTML templates, DynamicTooltip js hooks), static self-contained/offline HTML, and Deck.GL-backed scale — all four in one API | 2026-10-26 |
| Confirmed #002's earlier DataMapPlot pick, under the stricter R2 (per-point panel) + R3 (static Pages) lens | Original pick was aesthetics-driven; this survey's verdict is the same on load-bearing capability grounds (§3, §4) | 2026-10-26 |
| Plotly registered as fallback, not primary | Proven static-hosting path and WebGL scale, but panels/search/labels are DIY at exactly the points DataMapPlot already solves | 2026-10-26 |
| Packaging ladder: inline → split → per-point fetch | Stages map to dataset growth against GitHub Pages limits with no runtime/shape change (§5); promote on measured payload size | 2026-10-26 |
| Excluded Nomic Atlas | Hosted SaaS (public Discovery Hub exposure, iframe embedding) violates R3/R4; also means posting content leaves project custody | 2026-10-26 |
| Excluded Deepscatter | NC-CC-BY-SA (noncommercial) license blocks general reuse despite technical fit | 2026-10-26 |
| Excluded TF Embedding Projector | Standalone repo ARCHIVED; TSV-metadata data model cannot carry a rich per-point summary layer | 2026-10-26 |
| §6 summary JSON drafted as the per-point contract | Additive-columns design; one row = one `extra_point_data` row; raw posting text stays out of the artifact | 2026-10-26 |
