"""Embedding-pipeline Layer-1 spike: Athena select → CPU encode → lake part.

Feature-plan baseline spike for `add-embedding-pipeline`: proves the
integration seam (shared Athena runner × SentenceTransformers CPU encode ×
Garage/S3 ObjectStore write) with hardcoded inputs and no abstractions. This
script is intentionally throwaway — the Layer 2 extraction harvests its
learned shape into `backend/`, a re-run verifies parity, and this file is
removed in its own commit with the idea trail preserved in git history.
Repo experiment-script conventions (numbered immutable directories) do NOT
apply to feature spikes.

Run:

    python scripts/embed_spike.py

Baseline success: one run embeds the selected rows into one Parquet part
under ``lake/embeddings/.../version=v1/``, reads it back, and shows L2
norms 1.0 ± 1e-5 with row count == selected rows.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from io import BytesIO

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from backend.pipeline.athena import (
    aws_region,
    iter_query_results,
    poll_query_completion,
    results_location,
    submit_query,
)
from backend.storage import ids, keys
from backend.storage.store import ObjectStore

SQL = """
SELECT source_hash, cleaned_text
FROM artifacts_cleaned
WHERE version = 'v1' AND cleaned_text IS NOT NULL AND length(cleaned_text) > 0
LIMIT 25
"""

DATABASE = "semantic_career_map"
WORKGROUP = "semantic_career_map"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_SEGMENT = "all-minilm-l6-v2"
VARIANT = "minilm"
VERSION = "v1"
BATCH_SIZE = 16
INPUT_ARTIFACT = f"athena:{DATABASE}/artifacts_cleaned"


def main() -> None:
    load_dotenv()

    store = ObjectStore.from_env()
    athena = boto3.Session().client("athena", region_name=aws_region())

    execution_id = submit_query(
        athena,
        SQL.strip(),
        database=DATABASE,
        workgroup=WORKGROUP,
        output_location=results_location(),
    )
    poll_query_completion(athena, execution_id)

    source_hashes: list[str] = []
    texts: list[str] = []
    empty = 0
    for row in iter_query_results(athena, execution_id):
        # skip the CSV-parity header row the Athena pagination layer hands
        # back (same defensive shape used by the cleaning tier's select)
        if row[0] == "source_hash":
            continue
        source_hash, text = row[0], row[1]
        if not text or not text.strip():
            empty += 1
            continue
        source_hashes.append(source_hash)
        texts.append(text)

    print(
        f"athena: execution {execution_id}, rows selected: {len(texts)}, empty: {empty}"
    )
    if not texts:
        raise SystemExit(
            "no usable rows selected; nothing to embed (spike cannot prove the seam)"
        )

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    assert vectors.dtype == np.float32, vectors.dtype

    run_id = ids.new_id()
    now = datetime.now(UTC)
    embed_model_version = f"st-{__import__('sentence_transformers').__version__}"
    table = pa.table(
        {
            "source_hash": pa.array(source_hashes, type=pa.string()),
            "embedding": pa.array(vectors.tolist(), type=pa.list_(pa.float32())),
            "model": pa.array([MODEL_NAME] * len(texts), type=pa.string()),
            "model_version": pa.array(
                [embed_model_version] * len(texts), type=pa.string()
            ),
            "input_artifact": pa.array([INPUT_ARTIFACT] * len(texts), type=pa.string()),
            "input_version": pa.array([VERSION] * len(texts), type=pa.string()),
            "embedding_dimension": pa.array(
                [vectors.shape[1]] * len(texts), type=pa.int64()
            ),
            "normalization": pa.array(["l2"] * len(texts), type=pa.string()),
            "created_at": pa.array([now] * len(texts), type=pa.timestamp("us")),
        }
    )

    key = (
        f"lake/embeddings/model={MODEL_SEGMENT}/variant={VARIANT}/version={VERSION}/"
        f"part-{run_id}-00001.parquet"
    )
    keys.validate_key(key)
    sink = BytesIO()
    pq.write_table(table, sink)
    store.put(key, sink.getvalue())
    print(f"lake: wrote part {key}")

    # read back and verify the baseline success condition
    read_back = pq.read_table(BytesIO(store.get(key)))
    read_vectors = np.stack(read_back["embedding"].to_numpy(zero_copy_only=False))
    norms = np.linalg.norm(read_vectors, axis=1)
    assert len(read_back) == len(texts), (len(read_back), len(texts))
    assert np.all(np.abs(norms - 1.0) < 1e-5), (norms.min(), norms.max())
    dimensions = read_back["embedding_dimension"].to_pylist()
    assert set(dimensions) == {read_vectors.shape[1]}, set(dimensions)

    print(
        json.dumps(
            {
                "execution_id": execution_id,
                "rows": len(read_back),
                "empty_selected": empty,
                "dimension": int(read_vectors.shape[1]),
                "norm_min": float(norms.min()),
                "norm_max": float(norms.max()),
                "key": key,
                "status": "baseline-success",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
