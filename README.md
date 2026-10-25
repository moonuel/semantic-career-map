# Semantic Career Mapping Platform

> A production-oriented machine learning system that models the ML job market as a high-dimensional semantic space, finds structure, and eventually maps resumes into that space using modern embedding models, vector search, and clustering.
>
Currently in-progress. For more details, see the **documentation site:** [semantic-career-map](https://moonuel.github.io/semantic-career-map/) — live via GitHub Pages

---

## Object store access (Garage or S3)

`ObjectStore.from_env` (`backend/storage/store.py`) reads `STORE_S3_*` variables from `.env`. `STORE_S3_ENDPOINT` selects the store:

- **Set** → local Garage (`127.0.0.1:3900`, host networking, see `garage.toml`, region `garage`); the `STORE_S3_ACCESS_KEY_ID` / `STORE_S3_SECRET_ACCESS_KEY` pair is required — Garage has no credential chain of its own.
- **Empty/absent** → real S3; credentials must come from the default chain (`aws login` locally, IAM role in AWS) — static keys are refused.

### Configure — Garage (one-time per shell)

```bash
source .env   # loads STORE_S3_* without printing them
export AWS_ACCESS_KEY_ID=$STORE_S3_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$STORE_S3_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=garage
export AWS_ENDPOINT_URL=http://127.0.0.1:3900
```

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
- Garage requires path-style addressing; against an IP endpoint the AWS CLI uses it automatically, and against real S3 the SDK's default resolution applies (`backend/storage/store.py` only pins path-style when `STORE_S3_ENDPOINT` is set).
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
| Embeddings → clustering → API → Docker → cloud | 🔴 next phases |

---

## License

MIT
