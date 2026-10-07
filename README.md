# Semantic Career Mapping Platform

> A production-oriented machine learning system that models the ML job market as a high-dimensional semantic space, finds structure, and eventually maps resumes into that space using modern embedding models, vector search, and clustering.
>
Currently in-progress. For more details, see the **documentation site:** [semantic-career-map](https://moonuel.github.io/semantic-career-map/) — live via GitHub Pages

---

## Querying the Garage object store

Garage serves the S3 API on `127.0.0.1:3900` (host networking, see `garage.toml`), region `garage`. The keys live in `.env` as `GARAGE_S3_*` — the same ones `ObjectStore.from_env` (`backend/storage/store.py`) reads. AWS CLI v2 respects the `AWS_ENDPOINT_URL` environment variable, so no `--endpoint-url` flag per command.

### Configure (one-time per shell)

```bash
source .env   # loads GARAGE_S3_* without printing them
export AWS_ACCESS_KEY_ID=$GARAGE_S3_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$GARAGE_S3_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=garage
export AWS_ENDPOINT_URL=http://127.0.0.1:3900
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
- `127.0.0.1` is an IP endpoint, so AWS CLI uses path-style addressing automatically — this matches Garage's requirement and the `Config(s3={"addressing_style": "path"})` in `backend/storage/store.py`.
- Keep the exports shell-local; never commit credentials to any file.

---

## License

MIT
