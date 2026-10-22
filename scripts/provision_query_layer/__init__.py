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
import re

import boto3

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


def aws_region() -> str:
    """The lake bucket's region (Athena/Glue endpoints are region-local).

    ``STORE_S3_REGION`` is authoritative when set; else parse the bucket
    name's region suffix (``…-<region>-an``). AWS CLI defaults never
    override — an ambient us-east-1 default against a us-east-2 bucket was
    observed live: Athena rejects out-of-region results prefixes.
    """
    env = os.environ.get("STORE_S3_REGION")
    if env:
        return env
    match = re.search(r"-(us-[a-z]+-\d)-an$", os.environ["STORE_S3_BUCKET"])
    if match is None:
        raise ValueError(
            "STORE_S3_REGION unset and bucket name carries no <region> suffix"
        )
    return match.group(1)


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


def _create_ingestion_manifest_table(glue, database: str, bucket: str) -> None:
    table_input = {
        "Name": INGESTION_TABLE,
        "TableType": "EXTERNAL_TABLE",
        "StorageDescriptor": {
            "Location": f"s3://{bucket}/lake/manifests/ingestion/",
            "InputFormat": "org.apache.hadoop.hive.ql.io.MapredParquetInputFormat",
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
    try:
        glue.get_table(DatabaseName=database, Name=INGESTION_TABLE)
        print(f"table exists: {INGESTION_TABLE}")
    except glue.exceptions.EntityNotFoundException:
        glue.create_table(DatabaseName=database, TableInput=table_input)
        print(f"created table: {INGESTION_TABLE}")


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
    _create_workgroup(athena, workgroup, bucket, max_scan_bytes)
    print("provision complete")
    return plan
