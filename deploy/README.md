# deploy

Cloud-compute packaging for the two AWS tiers that run repo code
(`docs/architecture/cleaning-automation-report.md`, roadmap Phases 3a/3b/4.5).
Two images, one private ECR repo `pipeline-images`:

| Image | Dockerfile | Runs | Consumers |
|---|---|---|---|
| `scrape` | `Dockerfile.scrape` | `scripts/scrape.py` | Lambda `pipeline-ats-scrape` (ATS tier via image `CMD`), Fargate task definition (curator via command override) |
| `clean` | `Dockerfile.clean` | `python -m backend.pipeline.cleaning.daily` | Fargate task definition (daily driver: Athena select → manifest → runner) |

Provisioned by `python -m scripts.provision_compute_stack` (T1/T4 of the
plan); images are built and pushed here, then every consumer references the
same `:scrape` / `:clean` tags.

## Build + push (arm64 — Lambda and Fargate both run Graviton)

```bash
export PIPELINE_ECR=<account>.dkr.ecr.$$REGION$$.amazonaws.com
export AWS_REGION=$$REGION$$

aws ecr get-login-password --region $$AWS_REGION$$ \
  | docker login --username AWS --password-stdin $$PIPELINE_ECR$$

docker buildx create --use          # once, on an x86 host, for arm64 emulation

docker buildx build --platform linux/arm64 -f deploy/Dockerfile.scrape \
  -t $$PIPELINE_ECR$$/pipeline-images:scrape --push .
docker buildx build --platform linux/arm64 -f deploy/Dockerfile.clean \
  -t $$PIPELINE_ECR$$/pipeline-images:clean --push .
```

Pin the tags per-iteration (`:scrape-$(git rev-parse --short HEAD)`) when
promoting a new pipeline version; the moving tags stay for schedule targets.
`--create` re-apply is idempotent: the Lambda code is re-pointed only when
the target tag's ECR digest differs from the function's resolved image, so
plain re-applies cannot roll a pinned promotion back to a stale moving tag.

## Offline smoke (no network, no credentials needed)

```bash
docker build -f deploy/Dockerfile.scrape --platform linux/arm64 -t pipeline-scrape .
# the image's ENTRYPOINT is the Lambda runtime client — mirror the curator
# task def's entryPoint/command override for the argparse surface:
docker run --platform linux/arm64 --rm --entrypoint python pipeline-scrape scripts/scrape.py --help
docker build -f deploy/Dockerfile.clean --platform linux/arm64 -t pipeline-clean .
docker run --platform linux/arm64 --rm pipeline-clean python -m backend.pipeline.cleaning.daily --help
```

Both `--help` flags must exit 0 without any `KILO_API_KEY` or `STORE_S3_*`
set (`daily.py` fetches the secret and constructs the gateway client
inside `main()`; the scrape handler imports the batch CLI lazily).

## Redeploy after changes (the routine)

This stack is live: account `760852635093`, region `us-east-2`, repo
`pipeline-images`, schedules `daily-ats-scrape` / `daily-curator-scrape` /
`daily-clean` (04:00 / 05:00 / 06:00 UTC). After any code change:

```bash
# 0. never push red code
source .venv/bin/activate
ruff format --check . && ruff check . && pytest tests/ -q
git add -A && git commit -m "<change>"

# 1. rebuild + push both images (docker layer cache keeps unchanged tiers ~free)
REPO=760852635093.dkr.ecr.us-east-2.amazonaws.com/pipeline-images
aws ecr get-login-password --region us-east-2 \
  | docker login --username AWS --password-stdin $REPO
docker buildx build --platform linux/arm64 --load -f deploy/Dockerfile.scrape -t pipeline-scrape:test .
docker buildx build --platform linux/arm64 --load -f deploy/Dockerfile.clean  -t pipeline-clean:test .
# offline smokes from the previous section must pass before pushing:
docker tag pipeline-scrape:test $REPO:scrape && docker push $REPO:scrape
docker tag pipeline-clean:test  $REPO:clean  && docker push $REPO:clean

# 2. redeploy — one command does everything live:
python -m scripts.provision_compute_stack --create --reregister

# 3. verify + prove the new code actually runs
python -m scripts.provision_compute_stack --verify
python -m scripts.trigger_manual clean    # optional one-shot fire via the real scheduler
```

How `--create --reregister` lands the change: the Lambda's code is
re-pointed at the freshly pushed digest (`update_function_code`, waits for
`Successful`) so future invoke runs new code immediately; the task
definitions are registered as **new revisions**, and the schedules target
the **family** ARN — the next fire picks the new revision automatically.
In-flight tasks keep running the old revision: a redeploy never interrupts
a running tier.

## Per-change fast paths

| Change | What's enough |
|---|---|
| Only scraper code (`backend/ingestion/`, `scripts/scrape.py`) | rebuild+push `:scrape`; then `--create` (Lambda re-point) + `--reregister` (curator revision refresh) |
| Only cleaning code (`backend/pipeline/`, `backend/storage/`) | rebuild+push `:clean`; then `--create --reregister` |
| Prompts / model / `configuration.py` | **beware**: `configuration_hash` changes → the Athena anti-join re-selects the FULL raw lake → the next clean run is a full backfill (~9.3k rows, ~50 min, ~$3.40 LLM). Bump the run's evaluation version and read the eval gate before promoting |
| Dependencies changed in venv | update the pinned `pip install` versions inside the two Dockerfiles to `pip list` reality first, then the scrape-row flow above |

## Rollback

```bash
python -m scripts.provision_compute_stack --create --reregister \
  --image-tags scrape:<old-tag>,clean:<old-tag>
```

The schedules keep firing through all of it; failures page the confirmed
SNS email via the DLQ depth alarm and the ECS `STOPPED` exit-code rule.

## Runtime environment contract

Non-secret configuration is task-definition environment. The one secret,
`KILO_API_KEY`, is an SSM `SecureString`
(`/semantic-career-map/KILO_API_KEY`) fetched by `daily.py` into its own
process env — never baked into images or task definitions, never logged.

| Variable | `scrape` tier | `clean` tier | Notes |
|---|---|---|---|
| `STORE_S3_BUCKET` | required (task env) | required (task env) | the lake bucket, e.g. `lake` |
| `STORE_S3_REGION` | required (task env) | required (task env) | bucket region |
| `STORE_S3_ENDPOINT` | baked empty | baked empty | empty = real S3, default credential chain = the pipeline role |
| `SOURCE_*` CLI flags | `CMD`/command override | n/a | ATS: `--source ashby --source greenhouse --source lever`; curator: `--source dataroles` |
| `STORE_S3_ACCESS_KEY_ID` / `..._SECRET...` | **must not be set** | **must not be set** | `ObjectStore.from_env` refuses static keys on real S3 |
| `CONFIGURATION_HASH` | n/a | optional (default: computed from `CLEAN_MODEL` + shipped prompts) | re-run override only — binds the selection, the cache resolution, and the run record to one earlier generation; never pin a stale digest after a prompts/model change |
| `CLEAN_MODEL` | n/a | task def (default `google/gemini-2.5-flash-lite`, exp 008 A3) | |
| `CLEAN_WORKERS` | n/a | task def default 8 (D8) | |
| `MAX_RUN_HOURS` | n/a | task def default 2 | wall-clock guard over BOTH the selection query and the run; fires the runner's SIGINT/`interrupted` path |
| `PIPELINE_VERSION` | n/a | set by the provisioner from the image tag (`--image-tags`) | recorded in run manifests for lineage |
| `CLEAN_DATABASE`, `CLEAN_WORKGROUP` | n/a | optional | Glue/Athena names; defaults match `provision_query_layer` (`--database/--workgroup` flags win) |

## Dependencies

Docker-image deps (`pip install` layers above) are pinned to the versions
the operating venv resolves (`python:3.14-slim` — the repo's language);
`requirements.txt` is untouched on purpose — these are image-level deps,
not repository dependencies. The scrape path pulls `requests` + `boto3` +
`pyarrow` + `awslambdaric` (the Lambda Runtime Interface Client that makes
the image invokable); the clean path pulls `httpx` + `boto3` + `pyarrow`
(no pandas, no sentence-transformers — deliberately nowhere in either
image).

## Security posture note (accepted residual risk)

The tiers parse untrusted job-posting content while holding lake-write
IAM credentials (and, on the clean tier, the LLM key) in-process, with
default-VPC egress unrestricted. Egress control via NAT/gateway VPC
endpoints conflicts with the roadmap's explicit no-NAT/no-VPC guardrail
(`docs/architecture/aws-migration-roadmap.md` §2.4) and its cost budget
— so no SG egress allowlist is attached today. If the task-driving
compute ever widens, revisit: an S3 + gateway interface endpoint + egress
SG is the change, and the T1 policy-simulator gate covers IAM-side
blast radius.
