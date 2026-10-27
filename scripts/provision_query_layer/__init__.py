"""Glue Catalog + Athena provision and gate scripts for the lake (D12).

See ``docs/experiments/009`` for the evidence and ``tasks.md`` §2.2–2.4 for
the workflow this package automates: per-shape raw tables (no crawler),
ingestion-manifest table with day-partition projection, workgroup with a
bytes-scanned cap, and the gate queries that cross-check store counts and
exp 009's tables.
"""

from __future__ import annotations

import json
import os

import boto3

from backend.pipeline.athena import aws_region
from backend.pipeline.embedding.selection import EMBEDDINGS_TABLE

SCRAPED_JOB_COLUMNS: list[tuple[str, str]] = [
    ("url", "string"),
    ("title", "string"),
    ("description_html", "string"),
    ("employer", "string"),
    ("location", "string"),
    ("country", "string"),
    ("employment_type", "string"),
    ("date_posted", "string"),
    ("salary_min", "double"),
    ("salary_max", "double"),
    ("salary_currency", "string"),
    ("source_id", "string"),
]

# hiringcafe MCP shape: plain-text description; no description_html.
HIRINGCAFE_COLUMNS: list[tuple[str, str]] = [
    ("url", "string"),
    ("title", "string"),
    ("description", "string"),
    ("employer", "string"),
    ("location", "string"),
    ("workplace_type", "string"),
    ("employment_types", "string"),  # array serialized by the MCP export
    ("date_posted", "string"),
    ("salary", "string"),
    ("source_id", "string"),
    ("extracted", "boolean"),
    ("extra", "string"),  # JSON object as string; query via json_extract
]

# linkedin markdown shape: content only; typed columns NULL for this shape.
LINKEDIN_COLUMNS: list[tuple[str, str]] = [
    ("content", "string"),
    ("format", "string"),
    ("source_id", "string"),  # always NULL (no native id in this shape)
]

# Table name -> (source partition, payload columns).
SHAPE_TABLES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "raw_ashby": ("ashby", SCRAPED_JOB_COLUMNS),
    "raw_dataroles": ("dataroles", SCRAPED_JOB_COLUMNS),
    "raw_greenhouse": ("greenhouse", SCRAPED_JOB_COLUMNS),
    "raw_lever": ("lever", SCRAPED_JOB_COLUMNS),
    "raw_hiringcafe": ("hiringcafe", HIRINGCAFE_COLUMNS),
    "raw_linkedin": ("linkedin", LINKEDIN_COLUMNS),
}

INGESTION_TABLE = "manifests_ingestion"

ARTIFACTS_TABLE = "artifacts_cleaned"

# The cleaned-artifact row schema (writer.py ROW_COLUMNS, §5–6): Parquet
# parts under lake/artifacts/type=cleaned_text/variant=llm/version=v1/.
# select_uncleaned.sql anti-joins this table on (source_hash,
# configuration_hash); the location is pinned to the v1 llm generation —
# a version bump updates this registration (additive, idempotent).
ARTIFACTS_COLUMNS: list[tuple[str, str]] = [
    ("source_hash", "string"),
    ("variant", "string"),
    ("version", "string"),
    ("cleaned_text", "string"),
    ("configuration_hash", "string"),
    ("pipeline_version", "string"),
    ("created_at", "timestamp"),
]

# The registered embeddings table name (EMBEDDINGS_TABLE, imported above) is
# the selection layer's single source of truth
# (backend.pipeline.embedding.selection.EMBEDDINGS_TABLE) — a rename there
# must drive both the anti-join SQL and this registration.

# The embedding row schema (driver.py _part_bytes, data.md §7 + provenance):
# Parquet parts under lake/embeddings/<model-segment>/<variant>/<version>/.
# select_unembedded.sql anti-joins this table on (source_hash,
# configuration_hash); the location is pinned to the v1 minilm generation —
# a model/variant/version bump updates this registration (additive,
# idempotent, same discipline as the artifacts table). The column list is
# the writer's single source of truth (writer.PART_COLUMNS), so a schema
# change in part_bytes propagates here instead of drifting.
def _embeddings_columns() -> list[tuple[str, str]]:
    from backend.pipeline.embedding.writer import PART_COLUMNS

    return PART_COLUMNS


EMBEDDINGS_COLUMNS: list[tuple[str, str]] = _embeddings_columns()

# Full ingestion-manifest row schema (header + observation, manifest.py).
INGESTION_COLUMNS: list[tuple[str, str]] = [
    ("source", "string"),
    ("action", "string"),
    ("content_sha256", "string"),
    ("source_id", "string"),
    ("board", "string"),
    ("employer", "string"),
    ("observed_at", "timestamp"),
    ("schema_version", "string"),
    ("ingestion_id", "string"),
    ("ingested_on", "date"),
    ("pipeline_version", "string"),
    ("configuration_hash", "string"),
    ("run_started_at", "timestamp"),
    ("completed_at", "timestamp"),
    ("status", "string"),
    ("board_status", "string"),  # map<string,string> surfaced as JSON string
]

DATABASE_DEFAULT = "semantic_career_map"
WORKGROUP_DEFAULT = "semantic_career_map"
MAX_SCAN_BYTES_DEFAULT = (
    256 * 1024 * 1024
)  # full-lake JSON scan ≈ 100 MB; 256 MB ≈ $0.001

# The single ``aws_region`` implementation is the shared Athena runner's
# (backend/pipeline/athena.py): one region rule for provisioners, gate, and
# the daily driver. Re-exported here to keep every existing importer stable.


def _raw_columns(columns: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Column list for a raw table: shape columns + the key-derived hash."""
    return [{"Name": name, "Type": ctype} for name, ctype in columns]


def _create_database(glue, name: str, bucket: str) -> None:
    try:
        glue.get_database(Name=name)
        print(f"database exists: {name}")
    except glue.exceptions.EntityNotFoundException:
        glue.create_database(
            DatabaseInput={
                "Name": name,
                "Description": "Raw lake + manifests for the semantic-career-map lakehouse",
                "LocationUri": f"s3://{bucket}/",
            }
        )
        print(f"created database: {name}")


def _create_raw_table(
    glue,
    database: str,
    bucket: str,
    table: str,
    source: str,
    columns: list[tuple[str, str]],
) -> None:
    """Create one per-shape raw table (create-then-catch idempotence)."""
    table_input = {
        "Name": table,
        "TableType": "EXTERNAL_TABLE",
        "StorageDescriptor": {
            "Location": f"s3://{bucket}/lake/raw/source={source}/",
            "InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
            "SerdeInfo": {
                "Name": f"{table}_serde",
                "SerializationLibrary": "org.openx.data.jsonserde.JsonSerDe",
                "Parameters": {
                    # absent payload fields -> NULL, unknown -> ignored
                    "ignore.malformed.json": "true",
                },
            },
            "Columns": _raw_columns(columns),
        },
    }
    try:
        glue.get_table(DatabaseName=database, Name=table)
        print(f"table exists: {table}")
    except glue.exceptions.EntityNotFoundException:
        glue.create_table(DatabaseName=database, TableInput=table_input)
        print(f"created table: {table} (source={source})")


def _storage_descriptor_matches(stored: dict, target: dict[str, dict]) -> bool:
    """True when the catalog's stored table already matches the managed
    definition.

    Managed: location, input/output formats, serde, and the provisioned
    columns — external additive evolution (extra stored columns) survives;
    missing, retyped, or relocated ones trigger the update so the
    registration stays authoritative. Skipping the no-op update keeps
    externally evolved definitions (a later version-bump location
    registration, manually extended columns) from being silently reverted
    on every provision run.

    Args:
        stored: The ``StorageDescriptor`` of the existing catalog table.
        target: The table_input's ``StorageDescriptor`` this provisioner
            would apply.

    Returns:
        True when an ``update_table`` call would be a no-op-or-regression.
    """

    def shape(sd: dict) -> tuple:
        """The format/location fingerprint of one descriptor."""
        return (
            sd.get("Location"),
            sd.get("InputFormat"),
            sd.get("OutputFormat"),
            (sd.get("SerdeInfo") or {}).get("SerializationLibrary"),
        )

    if shape(stored) != shape(target):
        return False
    stored_columns = {(c["Name"], c["Type"]) for c in stored.get("Columns") or []}
    return all(
        (name, ctype) in stored_columns
        for name, ctype in ((c["Name"], c["Type"]) for c in target["Columns"])
    )


def _apply_managed_table(glue, database: str, table: str, table_input: dict) -> None:
    """Create the table, or update it only when its managed def drifted.

    Args:
        glue: The boto3 Glue client (or a testing double).
        database: The Glue database.
        table: The table name (``table_input["Name"]``).
        table_input: The full ``create-table``/``update-table`` input.
    """
    try:
        stored = glue.get_table(DatabaseName=database, Name=table)["Table"][
            "StorageDescriptor"
        ]
    except glue.exceptions.EntityNotFoundException:
        glue.create_table(DatabaseName=database, TableInput=table_input)
        print(f"created table: {table}")
        return
    if _storage_descriptor_matches(stored, table_input["StorageDescriptor"]):
        print(f"table up to date: {table}")
        return
    glue.update_table(DatabaseName=database, TableInput=table_input)
    print(f"table updated: {table}")


def _create_ingestion_manifest_table(glue, database: str, bucket: str) -> None:
    table_input = {
        "Name": INGESTION_TABLE,
        "TableType": "EXTERNAL_TABLE",
        "StorageDescriptor": {
            "Location": f"s3://{bucket}/lake/manifests/ingestion/",
            "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
            "SerdeInfo": {
                "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
            },
            "Columns": [{"Name": c, "Type": t} for c, t in INGESTION_COLUMNS],
        },
        "PartitionKeys": [{"Name": "ingested_on", "Type": "date"}],
        "Parameters": {
            "projection.enabled": "true",
            "projection.ingested_on.type": "date",
            "projection.ingested_on.format": "yyyy-MM-dd",
            "projection.ingested_on.interval": "1",
            "projection.ingested_on.interval.unit": "DAYS",
            "projection.ingested_on.range": "2025-01-01,NOW+31DAYS",
            "storage.location.template": f"s3://{bucket}/lake/manifests/ingestion/ingested_on=${{ingested_on}}",
        },
    }
    _apply_managed_table(glue, database, INGESTION_TABLE, table_input)


def _create_artifacts_table(glue, database: str, bucket: str) -> None:
    """Register ``artifacts_cleaned`` over the v1 llm cleaned-text parts."""
    table_input = {
        "Name": ARTIFACTS_TABLE,
        "TableType": "EXTERNAL_TABLE",
        "StorageDescriptor": {
            "Location": (
                f"s3://{bucket}/lake/artifacts/type=cleaned_text/variant=llm/version=v1/"
            ),
            "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
            "SerdeInfo": {
                "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
            },
            "Columns": [{"Name": c, "Type": t} for c, t in ARTIFACTS_COLUMNS],
        },
    }
    _apply_managed_table(glue, database, ARTIFACTS_TABLE, table_input)


def embeddings_location(bucket: str) -> str:
    """The embeddings table's pinned location (the v1 minilm generation).

    Args:
        bucket: The lake bucket name.

    Returns:
        The ``s3://`` location string.
    """
    return (
        f"s3://{bucket}/lake/embeddings/"
        "model=all-minilm-l6-v2/variant=minilm/version=v1/"
    )


def _create_embeddings_table(glue, database: str, bucket: str) -> None:
    """Register ``postings_embeddings`` over the v1 minilm vector parts."""
    table_input = {
        "Name": EMBEDDINGS_TABLE,
        "TableType": "EXTERNAL_TABLE",
        "StorageDescriptor": {
            "Location": embeddings_location(bucket),
            "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
            "SerdeInfo": {
                "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
            },
            "Columns": [{"Name": c, "Type": t} for c, t in EMBEDDINGS_COLUMNS],
        },
    }
    _apply_managed_table(glue, database, EMBEDDINGS_TABLE, table_input)


def _create_workgroup(athena, name: str, bucket: str, max_bytes: int) -> None:
    try:
        athena.get_work_group(WorkGroup=name)
        print(
            f"workgroup exists: {name} (enforced cap: {max_bytes / 1024 / 1024:.0f} MB/query)"
        )
    except (
        athena.exceptions.ResourceNotFoundException,
        athena.exceptions.InvalidRequestException,
    ):
        athena.create_work_group(
            Name=name,
            Configuration={
                "ResultConfiguration": {
                    "OutputLocation": f"s3://{bucket}/athena-results/"
                },
                "BytesScannedCutoffPerQuery": max_bytes,
                "EnforceWorkGroupConfiguration": True,
                "PublishCloudWatchMetricsEnabled": True,
            },
            Description="Per-query scan cap; results pinned in-bucket",
        )
        print(f"created workgroup: {name} (cap {max_bytes} bytes/query)")


def provision(
    database: str, workgroup: str, max_scan_bytes: int, *, dry_run: bool
) -> dict:
    """Plan and apply the provision set; return the applied plan.

    Args:
        database: Glue database name.
        workgroup: Athena workgroup name.
        max_scan_bytes: Per-query bytes-scanned cutoff.
        dry_run: Print the plan without calling AWS.

    Returns:
        The plan dict (the blueprint of what exists after the call).
    """
    session = boto3.Session(region_name=aws_region())
    # The bucket name is a constant across the codebase (keys.py docs,
    # runbooks) — read the same env source as ObjectStore.from_env rather
    # than re-deriving from account/region.
    bucket = os.environ["STORE_S3_BUCKET"]
    print(f"bucket: {bucket}")

    plan = {
        "glue_database": {"name": database, "location": f"s3://{bucket}/"},
        "raw_tables": [
            {
                "name": table,
                "location": f"s3://{bucket}/lake/raw/source={source}/",
                "columns": [c["Name"] for c in _raw_columns(columns)],
            }
            for table, (source, columns) in SHAPE_TABLES.items()
        ],
        "manifests_ingestion": {
            "location": f"s3://{bucket}/lake/manifests/ingestion/",
            "columns": [c for c, _ in INGESTION_COLUMNS],
        },
        "artifacts_cleaned": {
            "location": (
                f"s3://{bucket}/lake/artifacts/type=cleaned_text/"
                "variant=llm/version=v1/"
            ),
            "columns": [c for c, _ in ARTIFACTS_COLUMNS],
        },
        "postings_embeddings": {
            "location": embeddings_location(bucket),
            "columns": [c for c, _ in EMBEDDINGS_COLUMNS],
        },
        "athena": {
            "workgroup": workgroup,
            "output_location": f"s3://{bucket}/athena-results/",
            "bytes_scanned_cutoff_per_query": max_scan_bytes,
        },
    }
    print(json.dumps(plan, indent=2))
    if dry_run:
        print("(dry run — no changes made)")
        return plan

    glue = session.client("glue")
    athena = session.client("athena")
    _create_database(glue, database, bucket)
    for table, (source, columns) in SHAPE_TABLES.items():
        _create_raw_table(glue, database, bucket, table, source, columns)
    _create_ingestion_manifest_table(glue, database, bucket)
    _create_artifacts_table(glue, database, bucket)
    _create_embeddings_table(glue, database, bucket)
    _create_workgroup(athena, workgroup, bucket, max_scan_bytes)
    print("provision complete")
    return plan
