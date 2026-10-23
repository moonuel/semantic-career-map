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

## Offline smoke (no network, no credentials needed)

```bash
docker build -f deploy/Dockerfile.scrape --platform linux/arm64 -t pipeline-scrape .
docker run --platform linux/arm64 --rm pipeline-scrape python scripts/scrape.py --help
docker build -f deploy/Dockerfile.clean --platform linux/arm64 -t pipeline-clean .
docker run --platform linux/arm64 --rm pipeline-clean python -m backend.pipeline.cleaning.daily --help
```

Both `--help` flags must exit 0 without any `KILO_API_KEY` or `STORE_S3_*`
set (`daily.py` constructs the gateway client lazily inside `main()`).

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
| `CONFIGURATION_HASH` | n/a | required (task env) | 64-hex digest binding the Athena selection to the run |
| `CLEAN_MODEL` | n/a | required (task env) | default `google/gemini-2.5-flash-lite` (exp 008 A3) |
| `CLEAN_WORKERS` | n/a | optional (default 8, D8) | |
| `MAX_RUN_HOURS` | n/a | optional (default 2) | wall-clock guard; fires the runner's SIGINT/`interrupted` path |
| `PIPELINE_VERSION` | n/a | optional | overrides the `dev` git-sha fallback inside containers (no git on the image) |
| `DATABASE`, `WORKGROUP` | n/a | optional | Glue/Athena names, defaults match `provision_query_layer` |

## Dependencies

Docker-image deps (`pip install` layers above) are pinned to the versions
the operating venv resolves; `requirements.txt` is untouched on purpose —
these are image-level deps, not repository dependencies. The scrape path
pulls `requests` + `boto3` + `pyarrow`; the clean path pulls `httpx` +
`boto3` + `pyarrow` (no pandas, no sentence-transformers — deliberately
nowhere in either image).
