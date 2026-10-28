"""Gate queries over Athena: the §2.3/§2.4 verification of the query layer.

Runs the verification gates and prints their tables:

1. Per-source object counts — must match the `aws s3 ls` re-baseline
   (dataroles 7,647; hiringcafe 325; greenhouse 766; lever 326; ashby 213;
   linkedin 39; total 9,316).
2. Per-source text-length quantiles (median/p90/max) — must reproduce exp
   009's evidence through this second engine. Definitions legitimately
   differ: approx_percentile interpolates between adjacent order
   statistics, the report's Python numbers are single order statistics —
   the tolerance is ±100 chars (one inter-row gap at n≈326; lever's p90
   spans ~150), the max is exact.
3. The uncleaned-selection dry run over a no-artifact configuration: runs
   the raw-only variant of `select_uncleaned`, renders the rows through
   `backend.pipeline.selection`, and asserts 9,316 pin-able postings.
4. (optional, ``--embeddings``) Embeddings store-vs-table row-count
   parity — the query-side registration the anti-join's cache depends on.
5. (optional, ``--embeddings``) Every embeddings version prefix on disk has
   an ownership claim; a missing one is the issue-#22 condition and comes
   with the exact backfill command to run.

Usage:

    python -m scripts.provision_query_layer.gate [--skip-dry-run] [--embeddings]

Read-only against S3/Athena. Costs: a full-source JSON scan ≈ 100 MB →
about $0.0005 per query at the $5/TB rate; the workgroup cap enforces a
ceiling. Failure-ledger semantics are unit-tested (tests/test_selection.py),
not gate-tested — no run records exist until task 3.6 lands.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import sys

import boto3
from dotenv import load_dotenv

from backend.pipeline.athena import (
    iter_query_results,
    poll_query_completion,
    results_location,
    submit_query,
)
from backend.pipeline.selection import (
    build_selection_manifest,
    parse_query_results,
    select_uncleaned_sql,
)
from backend.storage import keys
from scripts.provision_query_layer import (
    DATABASE_DEFAULT,
    INGESTION_TABLE,
    WORKGROUP_DEFAULT,
    aws_region,
)

# The re-baseline (design.md Context, re-baselined 2026-10-21).
#
# These are LOWER BOUNDS, not equalities: the lake is append-only, so a
# passing ingestion run only ever raises a per-source count. Equality turned
# every successful ingest into a spurious gate failure (2026-10-27: ashby
# 213 expected vs 263 live) and taught everyone that a red gate is normal —
# which is how a genuinely broken table stayed red-unnoticed. A count BELOW
# the bound still means data loss and still fails.
EXPECTED_COUNTS: dict[str, int] = {
    "ashby": 213,
    "dataroles": 7647,
    "greenhouse": 766,
    "hiringcafe": 325,
    "lever": 326,
    "linkedin": 39,
}
TOTAL = sum(EXPECTED_COUNTS.values())

# exp 009's tables (docs/experiments/009-raw-lake-profiling.md), over each
# shape's own text field. These are SHAPE checks, not frozen pins: the raw
# layer grows, so median and p90 drift as postings are appended (lever's p90
# moved 10301 -> 10474 after 17 new postings on 2026-10-27, which tripped the
# old +/-100 band). The band is a stability guard — a large shift means
# extraction or the schema changed, which is what this gate is for.
EXPECTED_QUANTILES: dict[str, dict[str, int]] = {
    "ashby": {"median": 7440, "p90": 9109, "max": 16365},
    "dataroles": {"median": 6510, "p90": 10207, "max": 24691},
    "greenhouse": {"median": 10736, "p90": 13316, "max": 23207},
    "hiringcafe": {"median": 5789, "p90": 8732, "max": 12000},
    "lever": {"median": 8742, "p90": 10301, "max": 16317},
    "linkedin": {"median": 4836, "p90": 6998, "max": 9065},
}

# Tolerance for median/p90 as a FRACTION of the expected value, not a fixed
# character count: 10% of 6-24k chars absorbs organic sample drift while
# still catching a real distribution shift (boilerplate removal, a new
# markup shape). The previous flat +/-100 chars was ~1% and failed on
# ordinary growth.
QUANTILE_TOLERANCE = 0.10

# Per-source text field (the shape's own column; length() rides it).
TEXT_COLUMN: dict[str, str] = {
    "ashby": "description_html",
    "dataroles": "description_html",
    "greenhouse": "description_html",
    "lever": "description_html",
    "hiringcafe": "description",
    "linkedin": "content",
}


def _csv_line(cells) -> str:
    """One strict-CSV row row (quotes comma-bearing cells like ``source_id``).

    Gate 1/2 duplicate-splitting stays safe (their fields never contain
    commas) and gate 3's ``parse_query_results`` reads the quoting — the
    bare ``",".join`` this replaced silently corrupted such cells.

    Args:
        cells: The row's cell values (any order).

    Returns:
        The serialized line (no newline).
    """
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow(cells)
    return buffer.getvalue()


def run_query(athena, database: str, sql: str) -> list[str]:
    """Submit one query, poll to completion, return CSV rows (with header).

    The lifecycle (submit/poll/paginate) is the shared runner — the module
    docstring's workgroup-cap claim holds because the query is bound to the
    provisioned workgroup here, where the fixed dry-run digest binding and
    the row serialization stay this gate's own.

    The selection-style queries bind their ``:config`` placeholder to a
    fixed dry-run digest; over an artifacts-less layer the join is empty, so
    the binding only matters after task 3.4 lands.

    Args:
        athena: The boto3 Athena client.
        database: The Glue database to run against.
        sql: The query text (may contain ``:config``).

    Returns:
        Result rows as CSV strings, header row included.

    Raises:
        RuntimeError: when the query ends in a non-SUCCEEDED state.
    """
    config = hashlib.sha256(b"query-layer-dry-run").hexdigest()
    sql = sql.replace(":config", f"'{config}'")
    execution_id = submit_query(
        athena,
        sql,
        database=database,
        workgroup=WORKGROUP_DEFAULT,
        output_location=results_location(),
    )
    poll_query_completion(athena, execution_id)
    return [_csv_line(cells) for cells in iter_query_results(athena, execution_id)]


def counts_query() -> str:
    """Per-source counts union."""
    parts = [
        f"SELECT '{source}' AS source, count(*) AS n FROM raw_{source}"
        for source in sorted(EXPECTED_COUNTS)
    ]
    return "\nUNION ALL\n".join(parts) + "\nORDER BY source"


def quantiles_query() -> str:
    """Per-source length quantiles union (median/p90/max)."""
    parts = []
    for source, column in sorted(TEXT_COLUMN.items()):
        parts.append(
            f"SELECT '{source}' AS source, approx_percentile(length({column}), 0.5) AS median, "
            f"approx_percentile(length({column}), 0.9) AS p90, "
            f"max(length({column})) AS max FROM raw_{source}"
        )
    return "\nUNION ALL\n".join(parts) + "\nORDER BY source"


def uncleaned_dry_query() -> str:
    """The raw-only variant of select_uncleaned over a no-artifact config.

    Drift guard: this copy's hash extraction must stay verbatim-identical
    to the shipped selection SQL — a raw-leaf format change applied only to
    ``select_uncleaned.sql`` must fail this gate loudly, not silently
    verify parity against a stale expression.
    """
    hash_expr = "regexp_extract(r.\"$path\", 'sha256=([0-9a-f]{64})\\.json', 1)"
    if hash_expr not in select_uncleaned_sql():
        raise ValueError(
            "uncleaned_dry_query hash extraction drifted from "
            "backend/pipeline/sql/select_uncleaned.sql; update both together"
        )
    parts = [
        f"SELECT '{source}' AS source, {hash_expr} AS source_hash, "
        f"r.source_id FROM raw_{source} r"
        for source in sorted(EXPECTED_COUNTS)
    ]
    return "\nUNION ALL\n".join(parts) + "\nORDER BY source, source_hash"


def store_embedding_row_count(
    store, prefix: str, part_keys: list[str] | None = None
) -> int:
    """Count embedding rows across the store's parts under ``prefix``.
    Metadata-only: this reads each part's Parquet footer metadata — a
    filter-listing count would miss multi-row parts.
    """
    import pyarrow.parquet as pq

    count = 0
    for key in part_keys if part_keys is not None else store.list_keys(prefix):
        if not key.endswith(".parquet"):
            continue
        count += pq.read_metadata(io.BytesIO(store.get(key))).num_rows
    return count


def check_embeddings_parity(
    store,
    athena,
    database: str,
    part_keys: list[str] | None = None,
) -> tuple[bool, int, int]:
    """Gate 4: store-vs-table row-count parity for the embeddings table.

    The query-side registration makes the cache visible to the anti-join;
    a drifted registration (or a part outside the registered location)
    would silently re-encode rows. Parity by row count is the cheap
    cross-check the §2.3 gates established for the other tables.

    Args:
        store: The object store.
        athena: The boto3 Athena client.
        database: The Glue database the table is registered under.
        part_keys: Pre-enumerated embedding part keys; ``None`` lists the
            registered location itself.

    Returns:
        ``(match, store_rows, table_rows)``.
    """
    from scripts.provision_query_layer import EMBEDDINGS_TABLE, embeddings_location

    bucket = os.environ["STORE_S3_BUCKET"]
    location = embeddings_location(bucket)
    store_rows = store_embedding_row_count(
        store, location.removeprefix(f"s3://{bucket}/"), part_keys
    )
    rows = run_query(athena, database, f"SELECT count(*) FROM {EMBEDDINGS_TABLE}")
    table_rows = int(rows[1].strip())
    return store_rows == table_rows, store_rows, table_rows


CLAIM_ROOT = "lake/manifests/versions/embeddings/"


def check_embedding_version_claims(
    store, part_keys: list[str] | None = None
) -> tuple[bool, list[str]]:
    """Gate 5: every embedding version prefix on disk has an owner claim.

    Claims are written by the run before its first part, so an unclaimed
    prefix means either a legacy prefix that predates the registry or a run
    that bypassed the driver. Both are the condition that let 400 foreign rows
    into production ``version=v1`` (issue #22), so the drift is reported.
    The pipeline also refuses to mint a claim for such a prefix (first
    writer must not silently adopt a production address), so clearing this
    gate means publishing the owner via ``scripts/claim_embedding_version.py``.

    Args:
        store: The object store.
        part_keys: Pre-enumerated embedding part keys; ``None`` lists
            ``lake/embeddings/`` itself (tests and standalone callers).

    Returns:
        ``(all_claimed, missing)`` — the unclaimed version prefixes, sorted.
        A prefix carrying a malformed version token is reported as unclaimed
        rather than raised: a bad token on disk is exactly the drift this
        gate exists to surface, not a reason to lose the whole report.
    """
    prefixes: set[tuple[str, str, str]] = set()
    for key in (
        part_keys if part_keys is not None else store.list_keys("lake/embeddings/")
    ):
        parts = key.split("/")
        if len(parts) < 5:
            continue
        model = parts[2].removeprefix("model=")
        variant = parts[3].removeprefix("variant=")
        version = parts[4].removeprefix("version=")
        prefixes.add((model, variant, version))

    claimed = set(store.list_keys(CLAIM_ROOT))
    missing = []
    for model, variant, version in sorted(prefixes):
        try:
            claim_key = keys.embedding_version_claim(model, variant, version)
        except ValueError:
            missing.append(f"{model}/{variant}/version={version}")
            continue
        if claim_key not in claimed:
            missing.append(f"{model}/{variant}/version={version}")
    return not missing, missing


def _backfill_command(entry: str) -> str | None:
    """The exact operator command that publishes the owner of ``entry``.

    Args:
        entry: A missing-prefix string from
            :func:`check_embedding_version_claims` (``model/variant/version=<token>``).

    Returns:
        The backfill command with the prefix's coordinates filled in, or
        ``None`` when the token is not a valid version token — there is no
        command that can adopt a prefix whose name the key layer rejects.
    """
    model, _, rest = entry.partition("/")
    variant, _, version = rest.partition("/version=")
    try:
        keys.version_kind(version)
    except ValueError:
        return None
    return (
        "python scripts/claim_embedding_version.py "
        f"--model {model} --variant {variant} --version {version} "
        "--configuration-hash <64-char hex> --apply"
    )


def main() -> None:
    """Run the gates and print pass/fail per check."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the query-layer gate queries")
    parser.add_argument(
        "--skip-dry-run",
        action="store_true",
        help="skip gate 3 (the adapter dry run)",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="also run gate 4 (parity) and gate 5 (version claims)",
    )
    args = parser.parse_args()

    athena = boto3.Session().client("athena", region_name=aws_region())
    ok = True

    print("== gate 1: per-source counts vs aws s3 ls re-baseline (lower bound)")
    rows = run_query(athena, DATABASE_DEFAULT, counts_query())[1:]
    for line in rows:
        source, n = line.split(",")[0], int(line.split(",")[1])
        match = n >= EXPECTED_COUNTS[source]
        ok &= match
        print(
            f"  {source:<11} {n:>6} >= {EXPECTED_COUNTS[source]:>6} "
            f"{'OK' if match else 'BELOW BASELINE'}"
        )
    total = sum(int(line.split(",")[1]) for line in rows)
    print(
        f"  {'total':<11} {total:>6} >= {TOTAL:>6} "
        f"{'OK' if total >= TOTAL else 'BELOW BASELINE'}"
    )

    print(
        "== gate 2: length quantiles vs exp 009 tables "
        f"(median/p90 within {QUANTILE_TOLERANCE:.0%}, max a lower bound)"
    )
    rows = run_query(athena, DATABASE_DEFAULT, quantiles_query())[1:]
    for line in rows:
        source, median, p90, max_text = line.split(",")
        expected = EXPECTED_QUANTILES[source]

        def close(got: str, want: int) -> bool:
            """Within the relative tolerance of the expected value."""
            return abs(int(float(got)) - want) <= want * QUANTILE_TOLERANCE

        # median and p90 describe the typical posting and may drift with the
        # sample, but not shapedly; max is a single observed outlier that a
        # newer, longer posting legitimately raises, so it is a floor.
        match = (
            close(median, expected["median"])
            and close(p90, expected["p90"])
            and int(float(max_text)) >= expected["max"]
        )
        ok &= match
        print(
            f"  {source:<11} median {median:>7} (exp {expected['median']:>6})  "
            f"p90 {p90:>7} (exp {expected['p90']:>6})  max {max_text:>7} "
            f"(>= {expected['max']:>6})  {'OK' if match else 'CHECK'}"
        )

    if args.skip_dry_run:
        print("== gate 3 skipped (--skip-dry-run)")
    else:
        print("== gate 3: uncleaned dry run over a no-artifact configuration")
        # Route the query output through the production adapter's CSV
        # parser (validation is the point of the gate) — same feed shape
        # the selection flow uses: Athena CSV text, header included.
        csv_text = "\n".join(run_query(athena, DATABASE_DEFAULT, uncleaned_dry_query()))
        rows = parse_query_results(csv_text)
        manifest = build_selection_manifest(rows, configuration_hash="0" * 64)
        n_postings = manifest.count(b'"key"')
        match = n_postings >= TOTAL
        ok &= match
        print(
            f"  manifest postings {n_postings:>6} >= {TOTAL:>6} "
            f"{'OK' if match else 'BELOW BASELINE'}"
        )

    if args.embeddings:
        print("== gate 4: embeddings store-vs-table row-count parity")
        from backend.storage.store import ObjectStore
        from scripts.provision_query_layer import embeddings_location

        # One enumeration of the embeddings prefix feeds both gate 4 (row
        # counts) and gate 5 (claim discovery) — the lake is append-only, so
        # re-listing every part key per gate grows without bound.
        embeddings_store = ObjectStore.from_env()
        bucket = os.environ["STORE_S3_BUCKET"]
        prefix = embeddings_location(bucket).removeprefix(f"s3://{bucket}/")
        part_keys = [
            key
            for key in embeddings_store.list_keys(prefix)
            if key.endswith(".parquet")
        ]

        match, store_rows, table_rows = check_embeddings_parity(
            embeddings_store, athena, DATABASE_DEFAULT, part_keys
        )
        ok &= match
        print(
            f"  store parts rows {store_rows:>6} table rows {table_rows:>6} "
            f"{'OK' if match else 'MISMATCH'}"
        )

        print("== gate 5: every embeddings version prefix has an owner claim")
        claimed, missing = check_embedding_version_claims(embeddings_store, part_keys)
        ok &= claimed
        if claimed:
            print("  all version prefixes claimed OK")
        for entry in missing:
            print(f"  UNCLAIMED {entry}")
            command = _backfill_command(entry)
            if command is None:
                print(
                    "  fix: the version token is not a valid vN / x-... token — "
                    "rename or remove that prefix; no claim can adopt it"
                )
            else:
                print(f"  fix: {command}")
    else:
        print("== gate 4 skipped (pass --embeddings)")

    print("== gate 6: manifests_ingestion is readable (partition key is virtual)")
    # A registration that declares a partition column twice makes Athena
    # reject the whole table, so every consumer fails — including the
    # cleaning tier's recency window. One count(*) is enough to catch it,
    # and it is what would have caught the four-day outage.
    try:
        rows = run_query(
            athena, DATABASE_DEFAULT, f"SELECT count(*) FROM {INGESTION_TABLE}"
        )
        manifest_rows = int(rows[1].strip())
        match = manifest_rows > 0
        print(f"  rows {manifest_rows:>6} {'OK' if match else 'EMPTY'}")
    except RuntimeError as error:
        match = False
        print(f"  UNREADABLE: {error}")
    ok &= match

    print("GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
