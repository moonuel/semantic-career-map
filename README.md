# Semantic Career Mapping Platform

> A production-oriented machine learning system that models the ML job market as a high-dimensional semantic space, finds structure, and eventually maps resumes into that space using modern embedding models, vector search, and clustering.
>
Currently in-progress. For more details, see the **documentation site:** [semantic-career-map](https://moonuel.github.io/semantic-career-map/) — live via GitHub Pages

---

## Object store access (S3)

`ObjectStore.from_env` (`backend/storage/store.py`) reads `STORE_S3_*` variables from `.env`. `STORE_S3_ENDPOINT` selects the target:

- **Empty/absent** → AWS S3; credentials must come from the default chain (`aws login` locally, IAM role in AWS) — static keys are refused.
- **Set** → any S3-compatible server; the `STORE_S3_ACCESS_KEY_ID` / `STORE_S3_SECRET_ACCESS_KEY` pair is required, since such servers have no credential chain of their own. Nothing in this repo ships one; the retired local Garage container no longer exists.

### Configure — real S3

```bash
aws login     # refresh the profile session; no static keys anywhere
aws sts get-caller-identity   # verify who boto3/the CLI will act as
```

### Query

```bash
# list buckets (expect: eval, live, configs, lake)
aws s3 ls

# list objects under a prefix
aws s3 ls s3://lake/raw/source=linkedin/

# everything, recursively
aws s3 ls s3://lake --recursive --human-readable --summarize

# print an object to stdout (no temp file)
aws s3 cp s3://live/raw/<id>/source.md -

# size/etag without downloading
aws s3api head-object --bucket lake --key "lake/raw/source=linkedin/sha256=<hash>.json"

# count objects in a branch
aws s3 ls s3://live/raw/ --recursive | wc -l
```

Notes:

- For `s3api` commands, wrap keys containing `=` in double quotes; unneeded in the `s3://` URI form.
- Endpoint servers need path-style addressing; against an IP endpoint the AWS CLI uses it automatically, and AWS keeps the SDK's default resolution (`backend/storage/store.py` only pins path-style when `STORE_S3_ENDPOINT` is set).
- Keep the exports shell-local; never commit credentials to any file.

---

## Current Status

Field-discipline lakehouse for job postings under construction (see `docs/project.md` — the canonical tracker):

| Deliverable | Status |
|---|---|
| Lake ingestion | 🟡 live — 9.2k content-addressed raw postings on the dev store, manifest branches live |
| Data selection (Athena catalog) | 🟡 live — `select_uncleaned` anti-join drives the production cleaning selection (D12) |
| LLM cleaning (boilerplate-only, exp 008) | 🟡 live — `cleaned_text/variant=llm/version=v1` (flush-watermark parts, artifact-cache, `python -m backend.pipeline.cleaning`) |
| Cleaning evaluation (exp 003 metrics) | 🟡 live — golden + `cleaning_quality` result datasets with run manifests (`python -m backend.pipeline.eval`) |
| Scrape + clean on AWS | 🟢 live — Lambda ATS tier + Fargate curator/clean tasks, daily EventBridge schedules (04:00/05:00/06:00 UTC), provisioner `python -m scripts.provision_compute_stack`; runbook in `deploy/README.md` |
| Embeddings → clustering → API → Docker | 🔴 next phases |

---

## License

MIT
