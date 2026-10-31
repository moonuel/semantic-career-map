"""Layer 1a probe: verify the Athena vector query contract before any join or render.

Feature spike for the job-map deliverable (research report 020). This pass
does exactly one thing: assert that a SELECT against
``semantic_career_map.postings_embeddings`` returns the production vectors
correctly — one row per ``source_hash``, one dimension count per row,
64-hex keys — and that a bounded sample is retrievable and parseable
through the shared Athena runner.

Why the assertions exist: a non-keyed join in this engine does not raise, it
fans out (a probe joining on a title predicate returned 75,144,816 rows).
Query success alone proves nothing, so every check here is a count or shape
equality, never "it ran".

Spike, not production: no modules, no logging framework. It is deleted once
the feature graduates to ``scripts/map.py``. The generation is resolved from
the production pointer unless ``--configuration-hash`` names one.

Usage:
    python scripts/map_sql_probe.py
    python scripts/map_sql_probe.py --configuration-hash <hash> --dimensions 1024
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

from backend.pipeline.athena import (
    aws_region,
    iter_query_results,
    poll_query_completion,
    results_location,
    submit_query,
)

# The generation resolver is the graduated script's — one definition, so the
# probe and ``map.py`` cannot disagree about which generation a hash names.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from map import GenerationError, resolve_generation

DATABASE = "semantic_career_map"
WORKGROUP = "semantic_career_map"
SAMPLE_LIMIT = 100
MAX_SCAN_BYTES = 256 * 1024 * 1024
DEFAULT_OUT_DIR = Path("/tmp/kilo/map_probe")

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_SHAPE_SQL = """
SELECT COUNT(*) AS n,
       COUNT(DISTINCT source_hash) AS n_keys,
       MIN(CARDINALITY(embedding)) AS dim_min,
       MAX(CARDINALITY(embedding)) AS dim_max
FROM {database}.postings_embeddings
WHERE configuration_hash = '{hash}'
"""

_SAMPLE_SQL = """
SELECT source_hash, embedding
FROM {database}.postings_embeddings
WHERE configuration_hash = '{hash}'
LIMIT {limit}
"""


def _resolve_generation(configuration_hash: str | None, dimensions: int | None):
    """Resolve ``(hash, dims, model_label)`` with the probe's loud-fail policy."""
    try:
        return resolve_generation(configuration_hash, dimensions)
    except GenerationError as exc:
        _fail(str(exc))


def _athena():
    """An Athena client pinned to the lake bucket's region (repo rule)."""
    return boto3.Session().client(
        "athena",
        region_name=aws_region(),
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def _run(athena, sql: str) -> tuple[list[list[str]], int]:
    """Submit, poll, drain, and return ``(rows, bytes_scanned)``."""
    execution_id = submit_query(
        athena,
        sql,
        database=DATABASE,
        workgroup=WORKGROUP,
        output_location=results_location(),
    )
    poll_query_completion(athena, execution_id)
    rows = list(iter_query_results(athena, execution_id))
    stats = athena.get_query_execution(QueryExecutionId=execution_id)["QueryExecution"][
        "Statistics"
    ]
    return rows, int(stats.get("DataScannedInBytes", 0))


def _fail(message: str) -> None:
    """Print a contract violation and exit non-zero."""
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def _check_shape(rows: list[list[str]], expected_dims: int) -> int:
    """Assert the aggregate query reports a single, keyed generation shape."""
    if len(rows) != 2:
        _fail(f"shape query returned {len(rows)} rows, expected header + 1")
    header, data = rows
    if header != ["n", "n_keys", "dim_min", "dim_max"]:
        _fail(f"shape query header {header!r} does not match the projection")
    n, n_keys, dim_min, dim_max = (int(cell) for cell in data)
    if n == 0:
        _fail("no rows for this configuration_hash")
    if n != n_keys:
        _fail(f"rows={n} distinct_keys={n_keys}: duplicate keys for one config")
    if dim_min != expected_dims or dim_max != expected_dims:
        _fail(f"dims {dim_min}..{dim_max}, expected all {expected_dims}")
    return n


def _parse_sample(rows: list[list[str]], expected_dims: int) -> list[dict]:
    """Parse the bounded sample into ``[{source_hash, embedding}, ...]``."""
    if not rows:
        _fail("sample query returned no rows")
    body = rows[1:]
    if len(body) != SAMPLE_LIMIT:
        _fail(f"sample returned {len(body)} rows, expected exactly {SAMPLE_LIMIT}")

    parsed: list[dict] = []
    seen: set[str] = set()
    for row in body:
        if len(row) != 2:
            _fail(f"sample row has {len(row)} cells, expected 2")
        source_hash, cell = row
        if not _HASH_RE.match(source_hash):
            _fail(f"key {source_hash!r} is not 64 lowercase hex chars")
        if source_hash in seen:
            _fail(f"key {source_hash} appears twice in the sample")
        seen.add(source_hash)
        try:
            values = json.loads(cell)
        except json.JSONDecodeError as exc:
            _fail(f"embedding cell for {source_hash} is not JSON: {exc}")
        if not isinstance(values, list) or len(values) != expected_dims:
            got = len(values) if isinstance(values, list) else type(values).__name__
            _fail(
                f"embedding for {source_hash} has {got} values, expected {expected_dims}"
            )
        if not all(isinstance(v, int | float) and math.isfinite(v) for v in values):
            _fail(f"embedding for {source_hash} carries a non-finite value")
        parsed.append({"source_hash": source_hash, "embedding": values})
    return parsed


def _write_sample(sample: list[dict], out_dir: Path) -> Path:
    """Write the parsed sample as JSON and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "vectors_100.json"
    target.write_text(json.dumps(sample), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    """Run the shape and retrieval probes; print one summary line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--configuration-hash", default=None)
    parser.add_argument("--dimensions", type=int, default=None)
    args = parser.parse_args(argv)

    load_dotenv()
    configuration_hash, dimensions, model_label = _resolve_generation(
        args.configuration_hash, args.dimensions
    )
    print(f"generation={configuration_hash[:12]}… model={model_label}")
    shape_sql = _SHAPE_SQL.format(database=DATABASE, hash=configuration_hash)
    sample_sql = _SAMPLE_SQL.format(
        database=DATABASE, hash=configuration_hash, limit=SAMPLE_LIMIT
    )
    athena = _athena()

    shape_rows, shape_bytes = _run(athena, shape_sql)
    expected_rows = _check_shape(shape_rows, dimensions)

    sample_rows, sample_bytes = _run(athena, sample_sql)
    sample = _parse_sample(sample_rows, dimensions)

    for label, scanned in (("shape", shape_bytes), ("sample", sample_bytes)):
        if scanned > MAX_SCAN_BYTES:
            _fail(f"{label} query scanned {scanned / 2**20:.1f} MB, over the cap")

    target = _write_sample(sample, args.out_dir)
    scanned_mb = f"{max(shape_bytes, sample_bytes) / 2**20:.1f}"

    print(
        f"config={configuration_hash[:12]}… rows={expected_rows} "
        f"distinct_keys={expected_rows} dims={dimensions} "
        f"scanned_mb={scanned_mb}"
    )
    print(f"scanned_bytes shape={shape_bytes} sample={sample_bytes}")
    print(f"wrote {target} ({len(sample)} vectors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
