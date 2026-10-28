# AWS Migration Roadmap

> Status: **Phases 0-2 and 5 in use; Garage retired 2026-10-28.** The object store, the scraper tiers, and the query layer run on AWS. This document is kept as the migration record and the plan for the phases still open (3b, 4). Each phase names the primitives it teaches, the repo files it touches, a verification gate, and its cost.

**Companion documents:** [Data Architecture](data.md) (the lake layout this migration preserves), [Athena Lakehouse Proposal](proposed-athena-layout.md) (the Phase 5 query layer), Research Report [013](../research-reports/013-job-board-api-scrapers.md) (why polling is daily, whole-board, key-free).

---

## 1. Current State → Target State

| Component | Today | Target on AWS |
|---|---|---|
| Object store | ~~Garage v2.4.1 container~~ **retired 2026-10-28** | S3 bucket, same key namespace, zero file-format changes |
| Store client | `backend/storage/store.py` via `STORE_S3_*` env (`STORE_S3_ENDPOINT` empty/absent → AWS S3, default credential chain, static keys refused) | Same `ObjectStore`, IAM role instead of static keys in AWS contexts |
| Scraper run | `python scripts/scrape.py`, manually, `SOURCE_BOARDS` registry | Same script, **two compute tiers**: Lambda for ATS boards (seconds), Fargate task for the dataroles curator (> Lambda's 15-min ceiling; politeness-paced, wall-time grows with the sitemap) |
| Scheduling | none (manual) | EventBridge Scheduler, `rate(1 day)`, two targets (Lambda + ECS RunTask) — scraping stays **daily per Report 013 §5** |
| Failure visibility | run summary + manifest | SQS dead-letter queue + CloudWatch alarm → email |
| Query layer | none | Glue Data Catalog + Athena (Phase 5) |
| Local dev | AWS S3 (dev bucket) | Same dev bucket; no local store |

The migration principle: **S3 is a drop-in endpoint swap, not a redesign.** Keys are already content-addressed (`sha256=<hash>.json`), validated through `backend/storage/keys.py`, and the scrape run is a single process with the manifest as the observable record — that contract survives unchanged.

---

## 2. Cost Guardrails (apply before anything else)

1. **Account plan:** choose **Paid plan** at signup (Free plan accounts auto-close at 6 months / credit exhaustion). Both plans get the $100 signup credit; total earnable is $200. Credits cover this roadmap in full — steady-state monthly cost is ~$1.
2. **Billing alarm at $1** and an **AWS Budget at $2/month** — create these on day one, before any resources.
3. **Always-free caps** that this workload must stay inside (it does at current scale): Lambda 1M requests + 400,000 GB-seconds/month; EventBridge Scheduler free; Glue catalog free under 1M objects; CloudWatch Logs 10GB/month ingest.
4. **No NAT gateway, no VPC for Lambda.** The scrapers call public ATS APIs and S3; a VPC adds cost and breaks egress without benefit. This is the classic $30+/month surprise. The one VPC-touching exception: the Phase 3b Fargate task runs in the default VPC's public subnet with an auto-assigned public IP (egress via internet gateway, no NAT) and exits when the scrape completes.
5. **One region, one account.** Pick one (e.g. `us-east-1` or `eu-central-1`) and stay until the roadmap is done.
6. **No long-lived access keys for the pipeline.** One CLI profile for the human; the Lambda gets an execution role.

---

## 3. Phases

### Phase 0 — Account and billing spine

**Learns:** account structure, IAM basics, CloudWatch billing alarms.

**Tasks:**
- Create account (Paid plan), root MFA, lock root away.
- Create one IAM admin user for daily console/CLI use; store keys only in the local AWS CLI profile.
- Billing alarm ($1), Budget ($2/mo with email alert), region choice recorded in `.env` comments.

**Gate:** `aws sts get-caller-identity` works from CLI; alarm visible in Billing console.

---

### Phase 1 — S3 replaces the local store (done)

**Learns:** S3 bucket security defaults, boto3 against real AWS, one-off data migration with verification.

**Tasks:**
- Create bucket `lake`: Block Public Access on, SSE-S3 default encryption, versioning **off** (objects are content-addressed and never rewritten — versioning would earn nothing).
- Generalize `ObjectStore.from_env` (`backend/storage/store.py`): backend-neutral `STORE_S3_*` env vars, optional endpoint. boto3 already supports both.
- Point `.env` at S3; run `python scripts/scrape.py` — expect stored/duplicate observations to land in S3 and a manifest under `lake/manifests/ingestion/`.
- **Retirement (2026-10-28):** the Garage container, `docker-compose.yml`, `garage.toml`, and `data/objectstore/garage/` were removed. `STORE_S3_ENDPOINT` survives as a generic escape hatch for any S3-compatible server.

**Outcome:** the store runs on S3. The two stores had diverged rather than one being a superset — 1,577 raw payloads existed only in Garage — and they were **not** migrated; that data was retired with the container (a copy sits outside the repo at `/tmp/kilo/garage-retired-2026-10-28/`). Those postings are reproducible from the scrapers.

**Repo touchpoints:** `backend/storage/store.py`, `.env`, `README.md`.

---

### Phase 2 — IAM least privilege for the pipeline

**Learns:** IAM policy language, roles vs users, policy simulator.

**Tasks:**
- Create the pipeline execution **role** (trust policy: Lambda) with a policy scoped to `s3:GetObject/PutObject/ListBucket` on `lake/*`, plus CloudWatch Logs write.
- Test the policy in the IAM Policy Simulator against intended and *unintended* actions.

**Gate:** simulator shows lake read/write allowed and everything else denied. Role unused until Phase 3.

---

### Phase 3 — Scrape run as two compute tiers

The registry has two workload shapes (Report 013's two-tier landscape, now in compute): ATS boards are whole-board single GETs — seconds; the dataroles curator is a politeness-paced sitemap walk (`_PAGE_DELAY_S` in `backend/ingestion/scrapers.py`) that already exceeds Lambda's hard 15-minute ceiling and whose wall-time grows with the corpus. `scripts/scrape.py --source` flags already express the split; each tier is its own run, so each writes its own manifest — the existing observability contract, unchanged.

**Rejected for the curator tier:** Step Functions chunking (needs resumable slice semantics + per-chunk manifest aggregation, and moves the never-multiplex politeness invariant from code into infra config), shrinking page delays (breaks politeness), re-invoke chaining (hand-rolled state machine).

#### Phase 3a — ATS boards as a Lambda container image

**Learns:** ECR, Lambda from container image, execution roles, function env config, DLQ.

**Tasks:**
- `deploy/Dockerfile` (Lambda base image + handler) wrapping the repo (new `deploy/` directory — flag: introduces a top-level dir; decide vs `scripts/` at review).
- Push to ECR; create Lambda from the image (ARM64/Graviton is cheapest), memory 1024 MB, timeout suite: 300 s (ATS tier: 4 boards, seconds of work).
- Attach the Phase 2 role as execution role; store config via function env vars (non-secret); SQS dead-letter queue, retry 2.
- Manual invoke runs the registry's ATS tier: `script entry --source ashby --source greenhouse --source lever`.

**Gate:** one manual invocation completes; S3 shows new objects; DLQ empty; CloudWatch Logs show the run summary.

#### Phase 3b — Dataroles curator as an ECS Fargate task

**Learns:** ECS task definitions, Fargate launch type, task networking (public subnet, no NAT), ECR image with a plain CLI entrypoint.

**Tasks:**
- `deploy/Dockerfile.fargate` — same repo, plain `python:3.x-slim` base, `CMD ["python", "scripts/scrape.py", "--source", "dataroles"]`; push to a second ECR repo.
- Fargate task: 0.25 vCPU ARM / 0.5 GB (cheapest legal task size) in the default VPC public subnet, auto-assigned public IP, the Phase 2 role as task role, CloudWatch Logs group.
- Task must **exit when the scrape completes** — Fargate bills wall-clock; a finished task that lingers is the classic leak.
- Manual run: `aws ecs run-task` → full curator walk, object-level dedup makes duration the only cost.

**Gate:** one standalone task completes the corpus end-to-end, exits 0, writes objects + manifest; task transitions to STOPPED on its own.

**Repo touchpoints:** `deploy/Dockerfile`, `deploy/Dockerfile.fargate`, task-definition JSON in `deploy/`; possibly a `--max-workers` default note (Lambda has fewer cores than the dev box; politeness gates already bound per-host concurrency).

---

### Phase 4 — EventBridge Scheduler and observability

**Learns:** EventBridge Scheduler, CloudWatch metrics/alarms, SNS email.

**Tasks:**
- EventBridge Scheduler: `rate(1 day)` → two targets — the Lambda ARN (ATS tier) and an ECS RunTask target (curator tier, Phase 3b's task definition).
- CloudWatch alarm on DLQ depth (not empty → ATS tier failed) → SNS → email; an EventBridge rule on ECS task `STOPPED` events with non-zero exit code → same SNS topic (curator tier failed).
- Dashboard: invocations, errors, duration, approximate S3 object growth per day (`list_objects` is fine — or track manifest sizes).

**Gate:** two consecutive scheduled days produce both manifests (ATS run + curator run); day 2 records duplicates only (content-addressed dedup observable in the cloud) — this is the idempotence proof.

**Repo touchpoints:** `README.md` current-status section.

---

### Phase 5 — Glue Data Catalog + Athena (the query layer)

**Learns:** Glue crawler vs manual partition registration, Athena workgroups and scan caps.

**Tasks:**
- Glue database and table over `lake/raw/` with `source=` as a partition column (manual partition registration — content-addressed keys have no date partition, so provenance queries read `lake/manifests/ingestion/`, per [Data Architecture](data.md) §2-3).
- Athena workgroup with a per-query data-scan cap (remember $5/TB — at this corpus size, any single query is sub-cent).
- First queries: count per source, description-length distributions, dedup rate over manifests.

**Gate:** `SELECT source, count(*) ... GROUP BY` runs clean and matches `aws s3 ls` counts per prefix.

---

### Phase 6 — Deferred (explicit non-goals until the MVP finish line)

These exist on AWS but are **out of scope** until [implementation priorities](../project.md) say otherwise: MLflow server (Fargate task, ~$3-10/mo), FastAPI demo deployment (App Runner/ECS), CloudFront/frontend hosting, board-discovery scaling (SmartRecruiters/Recruitee/Personio per Report 013 §9).

---

## 4. Cost Model (steady state, after credits)

| Service | Driver | Monthly |
|---|---|---|
| S3 Standard | ~1 GB/yr growth, content-deduped | < $0.10 |
| S3 requests | ~7 boards/day + manifests | < $0.50 |
| Lambda | ~1 invocation/day, seconds, inside always-free | $0 |
| ECS Fargate (curator tier) | ~1 task/day, 0.25 vCPU ARM / 0.5 GB, wall-clock ≈ corpus walk (~$0.01/hr at that size) | ~$0.30 |
| ECR | 2 images, < 1 GB stored | ~$0.10 |
| EventBridge Scheduler | 2 rules | $0 |
| SQS | DLQ trickle | $0 |
| CloudWatch Logs | ~10 MB/mo | <$0.10 |
| Glue catalog + Athena | occasional queries | pennies |
| **Total** | | **≈ $1.50/mo** |

Stays inside the $200 free-tier credits from signup; nothing in phases 0-5 risks the credits unless a guardrail in §2 is ignored.

---

## 5. Verification Trail (what proves each layer survived the move)

1. **Storage identity:** `sha256=` keys re-hash identically in S3 — content-addressing intact.
2. **Dedup idempotence:** day-2 scheduled run writes a manifest of duplicates and zero new objects.
3. **Provenance:** each run's `lake/manifests/ingestion/` manifest is byte-stable for identical data (byte-identical manifests for identical inputs — already a scrape-run property).
4. **Least privilege:** policy simulator + a negative test (attempt an out-of-scope action, expect deny).
