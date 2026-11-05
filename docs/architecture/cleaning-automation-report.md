# Automating Ingestion & Cleaning on AWS — Gap Report

> Status: **implemented and scheduled — code audit 2026-11-05.** The cleaning tier landed 2026-10-23/24 (PRs #13/#14) and now runs daily at 06:00 UTC. The embedding tier followed at 07:00 UTC, consuming the cleaned dataset. This page remains the gap analysis and design record; the work-item checkboxes track what landed.

---

## 1. Where things stand

| Layer | Current state | Evidence |
|---|---|---|
| Cleaning stage | `backend/pipeline/cleaning/` runs on Fargate from the `:clean` image; Athena anti-join selection, Gemini 2.5 Flash Lite, flush-watermark Parquet parts, artifact cache, orphan healing, fail-fast, run and dataset manifests | `deploy/Dockerfile.clean:12`, `backend/pipeline/cleaning/daily.py` |
| Selection | Production path is the `select_uncleaned` anti-join; scheduled runs are bounded to 3 days / 5,000 rows unless `--backfill` is explicit | `backend/pipeline/selection.py`, `backend/pipeline/cleaning/daily.py` |
| Evaluation | Manual gate over 39 golden references using the exp-003 rubric; results are versioned Parquet | `backend/pipeline/eval/`, `docs/results.md` |
| Query layer | Glue tables over six raw shapes, ingestion manifests, cleaned artifacts, and embeddings; Athena workgroup with a 256 MiB scan cap | `scripts/provision_query_layer/` |
| Scheduling | Four EventBridge schedules: ATS 04:00, curator 05:00, clean 06:00, embed 07:00 UTC | `scripts/provision_compute_stack/__init__.py:69-95` |
| Observability | DLQ alarm + non-zero ECS exit rule → SNS email; per-run records and logs | `scripts/provision_compute_stack/__init__.py` |

What the cleaning stage needs at runtime: S3 (raw + artifacts + manifests), one Athena query (selection), and one LLM API key (`KILO_API_KEY`, Kilo Gateway). All state lives in the lake — compute is stateless.

### Remaining operational gaps

- The eval-before-promote rule is still runbook prose, not an enforced gate ([#15](https://github.com/moonuel/semantic-career-map/issues/15)).
- The local single-runner flock is host-scoped; a manual and scheduled cloud task can overlap ([#19](https://github.com/moonuel/semantic-career-map/issues/19)).
- A hard `SIGKILL` still leaves a part plus checkpoint without a completed run record; the data is reconstructible, not fully recorded.
- The Glue `artifacts_cleaned` table is registered at `variant=llm/version=v1`; a new cleaning version requires re-registration.

## 2. Gap analysis

The roadmap covers storage (Phases 1–2), scraper compute (3a/3b), scheduling (4), and query (5 — done early). **It has no phase for cleaning automation.** Three specific gaps:

1. **No cleaning compute tier.** The cleaning CLI runs only on the dev box. Nothing matches "run cleaning over new raw postings" to a compute service.
2. **No secret-handling story for the LLM key.** `KILO_API_KEY` reads env; the roadmap's IAM-only principle covers everything *except* the LLM key, which no IAM role can replace.
3. **No chaining between scrape and clean.** Phase 4 schedules two scrape tiers and stops. Nothing triggers selection + cleaning after scrapes land raw data.

The good news — and it changes the architecture: **the pipeline's design already implies the automation.** Content-addressed raw inputs, the artifact-row cache (D4: the dataset *is* the cache), and the anti-join selection (D12: select only uncleaned rows) together make every cleaning run **idempotent and order-independent**. A wasted or duplicated run scrapes zero new LLM calls and writes zero duplicate rows. At-least-once semantics over a monotone dataset need no orchestrator.

## 3. Target architecture (the automated daily cycle)

```
EventBridge Scheduler — rate(1 day)
  ├─ 3a  Lambda .......... ATS scrape tier      → raw + ingestion manifest
  ├─ 3b  Fargate ......... dataroles curator    → raw + ingestion manifest
  └─ NEW Fargate ......... cleaning tier        → artifacts + run manifests
```

The cleaning tier is a third Fargate task (same pattern the roadmap established in Phase 3b), running one new entrypoint — the **daily driver** — that chains in-process:

1. **Select**: Athena `select_uncleaned.sql` (parameterized by `configuration_hash`) via boto3 — deployment-bounded by default (a `manifests_ingestion` recency window of `--window-days` and a deterministic `--max-postings` row cap), so a configuration change can never silently re-enroll the whole historical lake through the scheduled tier; `--backfill` disables both bounds for the explicit full-lake path.
2. **Render**: existing `backend/pipeline/selection.py` adapter turns query rows into the canonical pinned manifest (in-task temp file — its format is already the runner's input).
3. **Clean**: existing `python -m backend.pipeline.cleaning` runner consumes the manifest, writes Parquet parts + flush checkpoints + run record + dataset manifest to the lake.
4. **Exit 0** only on a `completed` run record; failures and `interrupted` status exit non-zero — feeding the Phase 4 alarm unchanged.

**No sequencing machinery.** If scraping is mid-run while cleaning selects, selection simply sees fewer rows — the next day's run drains the rest. If a task hard-kills, the orphan-checkpoint back-fill (`recovered: true`) and cache-hit replay already cover recovery. Step Functions and EventBridge-chain-on-STOPPED add moving parts for a property the data model already guarantees. (The roadmap rejected Step Functions for the curator for the same class of reason.)

Eval stays a **manual gate**: it runs when a cleaning configuration changes (`--evaluation-version`), or on a chosen cadence before the embeddings phase consumes `artifacts/`. Automated scoring of every run would produce mostly-redundant results over 39 golden rows.

## 4. Work items

### 4.1 Prerequisites (existing roadmap phases — landed)

- [x] Phase 0: account, billing alarm ($1), budget ($2/mo), region in `.env` comments.
- [x] Phase 1: S3 `lake` bucket live; `ObjectStore` is backend-neutral (`STORE_S3_*`). The local Garage store was retired 2026-10-28 — the stores had diverged, and the Garage-only payloads were retired with it rather than migrated.
- [x] Phase 2: pipeline execution role — `s3:GetObject/PutObject/ListBucket`, CloudWatch Logs. Verified in the IAM Policy Simulator.

### 4.2 IAM extension for the cleaning tier

- [x] Add to the Phase 2 role: `athena:StartQueryExecution/GetQueryResults/StopQueryExecution`, `glue:GetDatabase/GetTable/GetPartitions`, and S3 read/write on the Athena results prefix (defined in task 2.3's workgroup config).
- [x] Negative-test: Athena writes only to its workgroup results prefix, not artifact branches.

### 4.3 SSM Parameter Store for the LLM key

- [x] One `SecureString` (e.g. `/semantic-career-map/KILO_API_KEY`). **SSM, not Secrets Manager** — standard tier is free vs $0.40/secret/month.
- [x] Fargate task role grants `ssm:GetParameter*` on that one path. All non-secret config (bucket, dataset, model, workers, flush triggers) rides task-definition environment variables — the CLIs are already env/flag driven.
- [x] Roadmap §2 note: "no long-lived access keys" gains the one non-IAM secret and its handling.

### 4.4 Image + task definitions (extend Phase 3b's artifacts)

- [x] Three separate Fargate images on one ECR repo: `:scrape` (curator), `:clean` (cleaning), and `:embed` (Titan-only, no torch). The split keeps the embedding image small and avoids shipping unrelated dependencies.
- [x] `deploy/` gains the Dockerfiles, task-definition JSONs, and an SSM-parameter task role. Curator: 0.25 vCPU / 0.5 GB. Cleaning: 1 vCPU / 2 GB. Embedding: 2 vCPU / 4 GB. All use auto-assign public IP, default VPC, and **no NAT** (egress is public Kilo Gateway/Bedrock + S3/Athena — the guardrail-4 exception already cited for the scraper applies identically).

### 4.5 Daily driver (the only new code)

- [x] New `backend/pipeline/cleaning/daily.py` (module docstring + CLI): Athena query → manifest render → runner, then exit-code mapping (`completed` → 0; `interrupted`/`failed` → 1). Sizing: under ~150 lines; reuses `selection.py`, `runs.py`, runner CLI as libraries.
- [x] Add `--max-run-hours` guard: at the deadline the driver sets the existing stop event (flush-on-stop path), so an over-long run lands `interrupted` durably and the next run replays the remainder via cache hits. This is the Fargate substitute for a task timeout — wall-clock billing makes "must exit" (Phase 3b invariant) a first-class requirement.
- [x] Unit tests: driver over stubbed Athena client (fixture rows → manifest byte-identical to `selection.py`'s), stop-event fired at deadline path.
- [x] Backfill policy: run the 9,316-row full-lake clean **locally first** (~35–50 min at 8 workers, ~$3.40 LLM, D8's measured basis), then confirm the cloud tier idempotent (0 new LLM calls). Either host produces identical artifacts — rows are content-identical by design. A cloud full-lake run is the explicit `daily --backfill` invocation (both deployment bounds off).

### 4.6 Scheduling + observability (small extension of Phase 4)

- [x] Third and fourth EventBridge Scheduler targets: cleaning at 06:00 UTC and embedding at 07:00 UTC, after the 04:00 ATS and 05:00 curator scrape runs — offsets, not chains.
- [x] Alarms reuse Phase 4's: ECS task `STOPPED` with non-zero exit → SNS email. CloudWatch Logs groups per tier; the run record's latency/usage/cost aggregates (D9/D10) can be surfaced via a dashboard later.
- [ ] Optional watchers: `lens on run.json status` via a weekly Athena query over `manifests/runs` (cheap; only if operator trust in exit codes proves insufficient).

### 4.7 Roadmap + docs

- [x] Roadmap status and current-state table updated for the cleaning and embedding tiers; the detailed design remains here rather than inserted as a separate Phase 4.5 section.
- [x] `data.md` migration table gains the automation row; README status after landing.

## 5. Gate (verification trail for the new phase)

1. **Idempotence in cloud:** two consecutive scheduled days — day 2 with no new raw postings shows cleaning run `n_calls: 0`, zero new artifact parts, `completed` status (the Day-2-duplicates proof from Phase 4, mirrored).
2. **Delta pickup:** after a real scrape day, day 3's run cleans only the new rows (manifest size == anti-join count).
3. **Least privilege:** policy simulator + negative tests (no `athena` without role, no `ssm` beyond the one parameter path).
4. **Recovery kill test:** `docker stop` mid-run locally against the cloud store → `interrupted` status, durable flushes, next run replays remainder via cache hits.
5. **Lineage:** manifests alone reconstruct dataset → run → eval-results edges exactly as task 5.1 verified locally, now computing from cloud-hosted parts.

## 6. Cost delta (steady state, after the scrape/scheduler baseline)

| Service | Driver | Monthly |
|---|---|---|
| Fargate cleaning tier | ~1 task/day, minutes of work at delta scale | ~$0.15–0.30 |
| Athena selection query | 1/day, sub-cent scan (tens of MB raw JSON) | < $0.05 |
| SSM Parameter Store | 1 SecureString, standard tier | $0 |
| LLM API (Kilo Gateway) | delta postings/day at ~$0.00036/posting | ~$0.10 |
| **Total** | | **~$1.65–2.00/mo** |

One-off: full-lake backfill ~$3.40 LLM + ~$0.02 compute. Everything above stays inside the $200 signup credits unless a §2 guardrail is ignored.

## 7. Risks and open decisions

- **Kilo Gateway rate limits / availability** — the client already retries 429/5xx with backoff (D6 fail-fast preserved); a persistent outage fails the run visibly and replays free tomorrow. No fallback model is configured; decide whether a second model becomes a config-only control.
- **Hard-kill window** (documented in tasks §5.1): in-flight calls lost on `SIGKILL` are exactly the loss window; on-demand (non-Spot) Fargate reduces kill probability, cache bounds the cost. Do not enable Fargate Spot for the cleaning tier.
- **Fargate wall-clock creep** — the curator tier's crawl grows with the sitemap; the cleaner's runtime grows with uncleaned backlog. `--max-run-hours` bounds both; articulate the alarm threshold if backlog lags growth for a week (backlog growth is faster than daily cleanup ⇒ investigation, not retry).
- **Single-region, single-account** stands; nothing here adds cross-region state.
- **flock lock semantics** — the local single-runner flock is host-scoped; two cloud tasks could run concurrently by accident (manual + scheduled). Rare and harmless (both drain the same anti-join; both cache-complete), but document it rather than add distributed locking.

## 8. Recommended sequencing

1. Phases 0–1 (account + S3 migration) — gating everything.
2. Phase 2 + §4.2/§4.3 (role, Athena/SSM extensions) — one IAM edit instead of two.
3. Phases 3a–4 (scrape tiers + daily schedule + alarms) — the roadmap as written.
4. §4.4–4.6 (cleaning + embedding tiers: images, drivers, scheduled targets) — reuses 3b's proven pattern one tier later.
5. §4.7 docs wrap + new-phase gate run.

Principle carried through: the lakes's manifests are the observable record; cloud compute is disposable state riding on top.
