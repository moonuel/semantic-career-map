"""Experiment 011 — encoder coverage bake-off: sample, local MiniLM, Titan V2.

This is the first script of Experiment 011. It builds one fixed 1,000-posting
sample from the production cleaned corpus, then produces four embedding
datasets over that same sample:

* A  — truncated MiniLM (`max_length=256`, current production recipe)
* B  — full-text MiniLM, non-overlapping 256-token windows, mean-pooled
* B2 — full-text MiniLM, overlapping 256-token windows, mean-pooled
* C  — full-text Titan Text Embeddings V2, normalized, 1024-d

All artifacts stay under ``scripts/011_encoder_coverage/data/``. The sample
cache is idempotent: if it exists, the sampling section is a no-op. Local
MiniLM and Titan vector caches are also resumable so a failed Titan run does
not repay for completed rows.

Usage::

    python scripts/011_encoder_coverage/run_experiment.py
    python scripts/011_encoder_coverage/run_experiment.py --stage sample
    python scripts/011_encoder_coverage/run_experiment.py --stage minilm
    python scripts/011_encoder_coverage/run_experiment.py --stage titan
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from backend.pipeline.athena import (
    aws_region,
    iter_query_results,
    poll_query_completion,
    results_location,
    submit_query,
)
from backend.storage.raw import parse_raw_key
from backend.storage.store import ObjectStore

EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"

DATABASE = "semantic_career_map"
WORKGROUP = "semantic_career_map"
SAMPLE_SIZE = 1_000
SAMPLE_SEED = 42

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_LENGTH = 256
OVERLAP_TOKENS = 32
BATCH_SIZE = 16

TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"
TITAN_DIMENSIONS = 1024
TITAN_MAX_ATTEMPTS = 6
TITAN_MAX_TOKENS = 8192

SAMPLE_PATH = DATA_DIR / "sample_1000.jsonl"
SAMPLE_MANIFEST_PATH = DATA_DIR / "sample_manifest.json"
SAMPLE_KEYS_PATH = DATA_DIR / "sample_keys.json"

ARM_FILES = {
    "a_minilm_truncated": DATA_DIR / "embeddings_a_minilm_truncated.npy",
    "b_minilm_pooled": DATA_DIR / "embeddings_b_minilm_pooled.npy",
    "b2_minilm_pooled_overlap": DATA_DIR / "embeddings_b2_minilm_pooled_overlap.npy",
    "c_titan_v2": DATA_DIR / "embeddings_c_titan_v2.npy",
}
ARM_DIAGNOSTICS = {name: DATA_DIR / f"diagnostics_{name}.json" for name in ARM_FILES}
TITAN_CACHE_PATH = DATA_DIR / "titan_v2_cache.jsonl"

SAMPLE_SQL = f"""
SELECT source_hash, cleaned_text
FROM {DATABASE}.artifacts_cleaned
WHERE variant = 'llm' AND version = 'v1'
ORDER BY source_hash
"""


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def git_commit() -> str:
    """Return the current git commit, or ``unknown`` when unavailable."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except OSError, subprocess.SubprocessError:
        return "unknown"


def _athena():
    """Build an Athena client pinned to the lake bucket's region."""
    import boto3
    from botocore.config import Config

    return boto3.Session().client(
        "athena",
        region_name=aws_region(),
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def _run_athena(sql: str) -> list[list[str]]:
    """Submit, poll, drain, and return an Athena query's rows."""
    athena = _athena()
    execution_id = submit_query(
        athena,
        sql,
        database=DATABASE,
        workgroup=WORKGROUP,
        output_location=results_location(),
    )
    poll_query_completion(athena, execution_id)
    return list(iter_query_results(athena, execution_id))


def _write_json(path: Path, payload: object) -> None:
    """Write JSON atomically so a killed run cannot leave a torn manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL, skipping a torn final line if one exists."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def fetch_metadata(store: ObjectStore, keys: list[str]) -> dict[str, dict[str, str]]:
    """Resolve sample hashes to raw-lake posting metadata."""
    from backend.storage.keys import RAW_SOURCES

    wanted = set(keys)
    found: dict[str, tuple[str, str]] = {}
    for source in sorted(RAW_SOURCES):
        prefix = f"lake/raw/source={source}/"
        for key in store.list_keys(prefix):
            try:
                _, source_hash = parse_raw_key(key)
            except ValueError:
                continue
            if source_hash in wanted:
                found[source_hash] = (source, key)

    meta: dict[str, dict[str, str]] = {}
    for source_hash, (source, key) in found.items():
        record: dict[str, str] = {"source": source}
        try:
            data = json.loads(store.get(key).decode("utf-8"))
        except OSError, ValueError, json.JSONDecodeError, AttributeError:
            meta[source_hash] = record
            continue
        for name in ("title", "employer", "url", "location", "country", "source_id"):
            value = data.get(name)
            record[name] = "" if value is None else str(value)
        meta[source_hash] = record
    return meta


def _token_count(tokenizer, text: str) -> int:
    """Count tokens with the model tokenizer, special tokens included."""
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def ensure_sample_cache() -> list[dict]:
    """Section 1: build the fixed 1,000-posting sample, or reuse its cache.

    Returns:
        Sample rows in fixed ``source_hash`` order.
    """
    if SAMPLE_PATH.exists():
        rows = _read_jsonl(SAMPLE_PATH)
        print(f"sample: cache hit {SAMPLE_PATH} ({len(rows)} rows) — no-op")
        return rows

    print("sample: fetching cleaned population")
    rows = _run_athena(SAMPLE_SQL)
    body = rows[1:]
    population = []
    for row in body:
        if len(row) < 2:
            continue
        source_hash, cleaned_text = row[0].strip(), row[1]
        if not source_hash or not cleaned_text:
            continue
        population.append({"source_hash": source_hash, "cleaned_text": cleaned_text})
    print(f"sample: population={len(population)}")

    if len(population) < SAMPLE_SIZE:
        raise SystemExit(
            f"population has {len(population)} rows, fewer than sample size {SAMPLE_SIZE}"
        )

    rng = random.Random(SAMPLE_SEED)
    chosen = sorted(rng.sample(range(len(population)), SAMPLE_SIZE))
    sampled = [population[i] for i in chosen]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    print("sample: resolving raw metadata")
    meta = fetch_metadata(
        ObjectStore.from_env(), [row["source_hash"] for row in sampled]
    )

    records: list[dict] = []
    for row in sampled:
        source_hash = row["source_hash"]
        record = dict(meta.get(source_hash, {}))
        record.update(
            {
                "source_hash": source_hash,
                "cleaned_text": row["cleaned_text"],
                "token_count": _token_count(tokenizer, row["cleaned_text"]),
            }
        )
        records.append(record)

    _write_jsonl(SAMPLE_PATH, records)
    _write_json(
        SAMPLE_KEYS_PATH,
        [record["source_hash"] for record in records],
    )
    _write_json(
        SAMPLE_MANIFEST_PATH,
        {
            "experiment": "011",
            "created_at": _utc_now(),
            "git_commit": git_commit(),
            "source_table": f"{DATABASE}.artifacts_cleaned",
            "source_filter": "variant='llm' AND version='v1'",
            "population_rows": len(population),
            "sample_rows": len(records),
            "sample_size": SAMPLE_SIZE,
            "seed": SAMPLE_SEED,
            "sampling": "random.Random(seed).sample over source_hash-sorted rows",
            "model": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "tokenizer_max_length": MAX_LENGTH,
            "query": SAMPLE_SQL.strip(),
        },
    )
    print(f"sample: wrote {SAMPLE_PATH} ({len(records)} rows)")
    return records


def load_minilm():
    """Load the pinned MiniLM model on CPU."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu", revision=MODEL_REVISION)
    model.max_seq_length = MAX_LENGTH
    return model


def _verify_unit_rows(vectors: np.ndarray, label: str) -> None:
    """Fail if any row is not unit length."""
    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms < 1e-12):
        raise ValueError(f"{label}: non-finite or zero vector norm")
    if not np.all(np.abs(norms - 1.0) < 1e-5):
        raise ValueError(
            f"{label}: normalization violated (min {norms.min():.8f}, "
            f"max {norms.max():.8f})"
        )


def _save_vectors(name: str, vectors: np.ndarray, diagnostics: dict) -> None:
    """Save one arm's vector matrix and diagnostics."""
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    np.save(ARM_FILES[name], vectors)
    _write_json(ARM_DIAGNOSTICS[name], diagnostics)
    print(f"{name}: saved {vectors.shape} -> {ARM_FILES[name]}")


def encode_truncated(model, sample: list[dict]) -> np.ndarray:
    """Section 2: encode arm A with the production 256-token truncation."""
    name = "a_minilm_truncated"
    if ARM_FILES[name].exists():
        print(f"{name}: cache hit — no-op")
        return np.load(ARM_FILES[name])

    texts = [row["cleaned_text"] for row in sample]
    started = time.monotonic()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    _verify_unit_rows(vectors, name)

    token_counts = np.asarray([row["token_count"] for row in sample], dtype=np.int64)
    encoded = np.minimum(token_counts, MAX_LENGTH)
    diagnostics = {
        "arm": name,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "max_length": MAX_LENGTH,
        "rows": len(sample),
        "dimensions": int(vectors.shape[1]),
        "seconds": round(time.monotonic() - started, 2),
        "tokens_total": int(token_counts.sum()),
        "tokens_encoded": int(encoded.sum()),
        "coverage_ratio": float(encoded.sum() / token_counts.sum()),
        "truncated_rows": int((token_counts > MAX_LENGTH).sum()),
        "truncated_rate": float((token_counts > MAX_LENGTH).mean()),
        "token_count_median": float(np.median(token_counts)),
        "token_count_p90": float(np.percentile(token_counts, 90)),
        "token_count_max": int(token_counts.max()),
    }
    _save_vectors(name, vectors, diagnostics)
    return vectors


def chunk_ids(ids: list[int], chunk_size: int, overlap: int = 0) -> list[list[int]]:
    """Split token ids into windows with a non-negative, bounded overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")
    if not ids:
        return [[]]
    stride = chunk_size - overlap
    return [ids[start : start + chunk_size] for start in range(0, len(ids), stride)]


def chunk_text(tokenizer, text: str, *, overlap: int = 0) -> list[str]:
    """Split text into decoded windows that fit MiniLM's 256-token limit.

    ``overlap=0`` is arm B; a positive overlap is arm B2. The window content
    size reserves room for the tokenizer's special tokens. A decoded window
    that re-tokenizes past the limit is split again, so the encoder never
    silently truncates a window.
    """
    specials = tokenizer.num_special_tokens_to_add(pair=False)
    content_size = MAX_LENGTH - specials
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    windows = chunk_ids(ids, content_size, overlap)

    def fit(window_ids: list[int]) -> list[str]:
        if not window_ids:
            return [""]
        decoded = tokenizer.decode(
            window_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if _token_count(tokenizer, decoded) <= MAX_LENGTH:
            return [decoded]
        middle = len(window_ids) // 2
        return fit(window_ids[:middle]) + fit(window_ids[middle:])

    fitted: list[str] = []
    for window in windows:
        fitted.extend(fit(window))
    return fitted


def mean_pool(
    model, tokenizer, sample: list[dict], *, overlap: int, name: str
) -> tuple[np.ndarray, dict]:
    """Embed full texts as windows, L2-normalize, mean-pool, re-normalize."""
    if ARM_FILES[name].exists():
        print(f"{name}: cache hit — no-op")
        diagnostics = {}
        if ARM_DIAGNOSTICS[name].exists():
            diagnostics = json.loads(ARM_DIAGNOSTICS[name].read_text(encoding="utf-8"))
        return np.load(ARM_FILES[name]), diagnostics

    started = time.monotonic()
    doc_windows: list[list[str]] = []
    window_counts: list[int] = []
    source_tokens: list[int] = []

    for row in sample:
        windows = chunk_text(tokenizer, row["cleaned_text"], overlap=overlap)
        doc_windows.append(windows)
        window_counts.append(len(windows))
        source_tokens.append(row["token_count"])

    flat_windows = [window for windows in doc_windows for window in windows]
    window_tokens = sum(_token_count(tokenizer, window) for window in flat_windows)
    print(f"{name}: encoding {len(flat_windows)} windows from {len(sample)} postings")
    window_vectors = model.encode(
        flat_windows,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    window_vectors = np.asarray(window_vectors, dtype=np.float32)

    vectors = np.empty((len(sample), window_vectors.shape[1]), dtype=np.float32)
    cursor = 0
    for i, windows in enumerate(doc_windows):
        count = len(windows)
        pooled = window_vectors[cursor : cursor + count].mean(axis=0)
        norm = float(np.linalg.norm(pooled))
        if norm < 1e-12:
            raise ValueError(f"{name}: zero-norm pooled vector at row {i}")
        vectors[i] = pooled / norm
        cursor += count

    _verify_unit_rows(vectors, name)
    diagnostics = {
        "arm": name,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "max_length": MAX_LENGTH,
        "overlap_tokens": overlap,
        "rows": len(sample),
        "dimensions": int(vectors.shape[1]),
        "windows_total": len(flat_windows),
        "windows_per_posting_mean": float(np.mean(window_counts)),
        "windows_per_posting_max": int(np.max(window_counts)),
        "tokens_total": int(sum(source_tokens)),
        "window_token_incidence": int(window_tokens),
        "coverage_ratio": 1.0,
        "seconds": round(time.monotonic() - started, 2),
    }
    _save_vectors(name, vectors, diagnostics)
    return vectors, diagnostics


def _titan_client():
    """Build a Bedrock Runtime client."""
    import boto3
    from botocore.config import Config

    return boto3.Session().client(
        "bedrock-runtime",
        region_name=aws_region(),
        config=Config(
            retries={"max_attempts": 5, "mode": "standard"},
            read_timeout=60,
            connect_timeout=10,
        ),
    )


def titan_embed_one(client, text: str) -> dict:
    """Embed one text with normalized Titan V2, retrying transient failures."""
    body = json.dumps(
        {
            "inputText": text,
            "dimensions": TITAN_DIMENSIONS,
            "normalize": True,
        }
    )
    last_error: Exception | None = None
    for attempt in range(TITAN_MAX_ATTEMPTS):
        started = time.monotonic()
        try:
            response = client.invoke_model(
                modelId=TITAN_MODEL_ID,
                accept="application/json",
                contentType="application/json",
                body=body,
            )
            payload = json.loads(response["body"].read())
            embedding = np.asarray(payload["embedding"], dtype=np.float32)
            if embedding.shape != (TITAN_DIMENSIONS,):
                raise ValueError(
                    f"Titan returned {embedding.shape}, expected ({TITAN_DIMENSIONS},)"
                )
            input_tokens = int(payload.get("inputTextTokenCount", 0))
            if input_tokens > TITAN_MAX_TOKENS:
                raise ValueError(
                    f"Titan input token count {input_tokens} exceeds {TITAN_MAX_TOKENS}"
                )
            norm = float(np.linalg.norm(embedding))
            if norm < 1e-12:
                raise ValueError("Titan returned a zero vector")
            if abs(norm - 1.0) > 1e-4:
                embedding = embedding / norm
                norm = float(np.linalg.norm(embedding))
            return {
                "embedding": embedding.tolist(),
                "input_tokens": input_tokens,
                "norm": norm,
                "latency_s": round(time.monotonic() - started, 3),
            }
        except Exception as exc:  # noqa: BLE001 — retry boundary for Bedrock
            last_error = exc
            if attempt + 1 == TITAN_MAX_ATTEMPTS:
                break
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(
        f"Titan embedding failed after {TITAN_MAX_ATTEMPTS} attempts: {last_error}"
    )


def _load_titan_cache() -> dict[str, dict]:
    """Read the resumable Titan cache keyed by source_hash."""
    if not TITAN_CACHE_PATH.exists():
        return {}
    records: dict[str, dict] = {}
    for row in _read_jsonl(TITAN_CACHE_PATH):
        if row.get("source_hash") and row.get("embedding"):
            records[row["source_hash"]] = row
    return records


def encode_titan(sample: list[dict]) -> np.ndarray:
    """Section 4: embed full cleaned text with normalized Titan V2 via Bedrock."""
    name = "c_titan_v2"
    if ARM_FILES[name].exists():
        print(f"{name}: cache hit — no-op")
        return np.load(ARM_FILES[name])

    keys = [row["source_hash"] for row in sample]
    cache = _load_titan_cache()
    missing = [row for row in sample if row["source_hash"] not in cache]
    print(
        f"{name}: cache={len(cache)} missing={len(missing)} "
        f"model={TITAN_MODEL_ID} region={aws_region()}"
    )

    started = time.monotonic()
    if missing:
        with TITAN_CACHE_PATH.open("a", encoding="utf-8") as handle:

            def write_record(record: dict) -> None:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

            client = _titan_client()
            for completed, row in enumerate(missing, start=1):
                result = titan_embed_one(client, row["cleaned_text"])
                record = {
                    "source_hash": row["source_hash"],
                    **result,
                }
                cache[row["source_hash"]] = record
                write_record(record)
                if completed % 25 == 0 or completed == len(missing):
                    print(
                        f"{name}: embedded {completed}/{len(missing)}",
                        flush=True,
                    )

    missing_keys = [key for key in keys if key not in cache]
    if missing_keys:
        raise RuntimeError(f"{name}: missing {len(missing_keys)} rows after run")

    vectors = np.asarray([cache[key]["embedding"] for key in keys], dtype=np.float32)
    _verify_unit_rows(vectors, name)

    input_tokens = np.asarray(
        [cache[key]["input_tokens"] for key in keys], dtype=np.int64
    )
    latencies = np.asarray([cache[key]["latency_s"] for key in keys], dtype=np.float64)
    diagnostics = {
        "arm": name,
        "provider": "amazon-bedrock",
        "model_id": TITAN_MODEL_ID,
        "region": aws_region(),
        "dimensions": TITAN_DIMENSIONS,
        "normalize": True,
        "rows": len(sample),
        "tokens_total": int(input_tokens.sum()),
        "tokens_median": float(np.median(input_tokens)),
        "tokens_p90": float(np.percentile(input_tokens, 90)),
        "tokens_max": int(input_tokens.max()),
        "coverage_ratio": 1.0,
        "latency_mean_s": round(float(latencies.mean()), 3),
        "latency_p90_s": round(float(np.percentile(latencies, 90)), 3),
        "seconds_wall": round(time.monotonic() - started, 2),
    }
    _save_vectors(name, vectors, diagnostics)
    return vectors


def main(argv: list[str] | None = None) -> int:
    """Run sampling and the four embedding datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "sample", "minilm", "titan"),
        default="all",
        help="run one stage or the full experiment",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sample = ensure_sample_cache()
    if args.stage == "sample":
        return 0

    if args.stage in ("all", "minilm"):
        model = load_minilm()
        encode_truncated(model, sample)
        mean_pool(model, model.tokenizer, sample, overlap=0, name="b_minilm_pooled")
        mean_pool(
            model,
            model.tokenizer,
            sample,
            overlap=OVERLAP_TOKENS,
            name="b2_minilm_pooled_overlap",
        )

    if args.stage in ("all", "titan"):
        encode_titan(sample)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
