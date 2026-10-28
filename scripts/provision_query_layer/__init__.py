"""Glue Catalog + Athena provision and gate scripts for the lake (D12).

See ``docs/experiments/009`` for the evidence and ``tasks.md`` §2.2–2.4 for
the workflow this package automates: per-shape raw tables (no crawler),
ingestion-manifest table with day-partition projection, workgroup with a
bytes-scanned cap, and the gate queries that cross-check store counts and
exp 009's tables.

The embeddings table's projection enums are derived from the production
pointer (:func:`embeddings_projection_values`) — the pointer is the only
authority over which generations are production, so the catalog cannot
drift from it by construction, and the pointer gate catches the case
where an operator forgot to re-provision after a flip.
"""

from __future__ import annotations

import json
import os

import boto3

from backend.pipeline.athena import aws_region
from backend.pipeline.embedding.selection import EMBEDDINGS_TABLE
from backend.storage.pointer import read_pointer

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
# configuration_hash). Those path levels are declared as partition keys
# (EMBEDDINGS_PARTITION_KEYS), so a model/variant/version bump adds a
# projection value rather than relocating the table. The column list is
# the writer's single source of truth (writer.PART_COLUMNS), so a schema
# change in part_bytes propagates here instead of drifting.
def _embeddings_columns() -> list[tuple[str, str]]:
    from backend.pipeline.embedding.writer import PART_COLUMNS

    return PART_COLUMNS


EMBEDDINGS_COLUMNS: list[tuple[str, str]] = _embeddings_columns()

# The embedding table's partition keys, in path order: the
# ``model=/variant=/version=`` levels written by keys.embedding_part. They are
# STRING per Athena's partition-key guidance.
#
# ``model_segment`` and not ``model``: PART_COLUMNS already carries a ``model``
# column holding the full model id, and a partition key sharing a column name
# makes the table unqueryable ("Table descriptor contains duplicate columns")
# — the defect that once made manifests_ingestion unreadable.
EMBEDDINGS_PARTITION_KEYS: list[tuple[str, str]] = [
    ("model_segment", "string"),
    ("variant", "string"),
    ("version", "string"),
]

PRODUCTION_POINTER_KEY = "state/production.json"

# Full ingestion-manifest row schema (header + observation, manifest.py).
#
# ``ingested_on`` is deliberately ABSENT: it is the table's partition key
# (``_create_ingestion_manifest_table``), and Hive partition columns are
# virtual — they name the directory level, they are not fields of the row.
# Declaring it here as well as in ``PartitionKeys`` makes Athena reject the
# table outright ("Table descriptor contains duplicate columns") and every
# query fails, including the cleaning tier's recency window. The Parquet
# files do carry the column (manifest.py ``_SCHEMA``); the partition value
# takes precedence.
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
    ("pipeline_version", "string"),
    ("configuration_hash", "string"),
    ("run_started_at", "timestamp"),
    ("completed_at", "timestamp"),
    ("status", "string"),
    ("board_status", "string"),  # map<string,string> surfaced as JSON string
]

# The manifest table's partition key (the ``ingested_on=<date>/`` directory
# level written by keys.manifest_ingestion). Single source of truth for both
# the registration and the column list's exclusion of it.
INGESTION_PARTITION_KEYS: list[tuple[str, str]] = [("ingested_on", "date")]

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


def _storage_descriptor_matches(
    stored: dict, target: dict[str, dict], *, partition_names: set[str] | None = None
) -> bool:
    """True when the catalog's stored STORAGE DESCRIPTOR matches the managed
    definition.

    One half of :func:`_table_matches`, which is what the provisioner calls:
    the descriptor covers location, formats, serde and columns only. The
    table-level keys a descriptor does not own (``PartitionKeys``,
    ``Parameters``) are compared there.

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
        partition_names: The table's partition key names. A stored column
            sharing one of them is illegal to Hive (duplicate columns) and
            is reported as drift despite the additive-column rule.

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
    if not all(
        (name, ctype) in stored_columns
        for name, ctype in ((c["Name"], c["Type"]) for c in target["Columns"])
    ):
        return False
    # Extra stored columns are tolerated (external additive evolution), with
    # one exception: a column whose name is a partition key is never legal.
    # Hive rejects such a table outright ("Table descriptor contains
    # duplicate columns"), so it must count as drift even though the
    # additive rule would otherwise keep it. This is the exact defect that
    # made manifests_ingestion unqueryable.
    return not any(name in (partition_names or set()) for name, _ in stored_columns)


def _table_matches(stored: dict, target: dict) -> bool:
    """True when the catalog's stored table already matches the managed def.

    Compares the WHOLE managed definition, not just the storage descriptor:
    a partition-key or projection-parameter drift is as much a drift as a
    moved location, and ignoring it meant a wrong registration could never
    be repaired by the tool that wrote it.

    - ``StorageDescriptor``: location and formats exactly; provisioned
      columns by presence (external additive columns survive).
    - ``PartitionKeys``: exactly, in order — an extra or reordered key is
      a real drift, unlike an extra column.
    - ``Parameters``: the managed keys, as a subset (Glue synthesises
      further keys of its own, which this provisioner must not revert).

    Args:
        stored: The catalog's ``Table`` (not its nested descriptor).
        target: The table_input this provisioner would apply.

    Returns:
        True when an ``update_table`` call would be a no-op-or-regression.
    """
    if not _storage_descriptor_matches(
        stored.get("StorageDescriptor") or {},
        target["StorageDescriptor"],
        partition_names={key["Name"] for key in (target.get("PartitionKeys") or [])},
    ):
        return False
    if (stored.get("PartitionKeys") or []) != (target.get("PartitionKeys") or []):
        return False
    stored_params = stored.get("Parameters") or {}
    return all(
        stored_params.get(name) == value
        for name, value in (target.get("Parameters") or {}).items()
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
        stored = glue.get_table(DatabaseName=database, Name=table)["Table"]
    except glue.exceptions.EntityNotFoundException:
        glue.create_table(DatabaseName=database, TableInput=table_input)
        print(f"created table: {table}")
        return
    if _table_matches(stored, table_input):
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
        "PartitionKeys": [
            {"Name": name, "Type": ctype} for name, ctype in INGESTION_PARTITION_KEYS
        ],
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
    """The embeddings table's root location, above the partition levels.

    The generation segments (``model=/variant=/version=``) are Hive partition
    keys on the registered table, not part of the location: pinning them here
    was what made every generation after ``v1`` invisible to the catalog and
    left no way to write a partition predicate. The rows still carry a
    ``model`` column (the full model id), which is why the partition key is
    named ``model_segment`` — a partition key sharing a column name makes the
    table unqueryable ("duplicate columns").

    Args:
        bucket: The lake bucket name.

    Returns:
        The ``s3://`` location string.
    """
    return f"s3://{bucket}/lake/embeddings/"


def embeddings_projection_values(pointer_doc: dict) -> dict[str, str]:
    """Derive the embeddings table's projection enum values from the pointer.

    The catalog authority question ("which model/variant/version partitions
    are registered?") has exactly one source: the production pointer. Its
    embeddings entries' segments become the three projection enums — comma-
    joined, first-appearance order, deduplicated. Stable tokens only: an
    ephemeral ``x-...`` prefix is never projected (a pointer parse refuses
    one outright; this re-check keeps the enum derivation safe if called
    with a document that bypassed :func:`parse_document`).

    Args:
        pointer_doc: The production pointer document (parsed).

    Returns:
        ``{"model_segment": "…", "variant": "…", "version": "…"}`` — the
        comma-joined enum value strings for the partition parameters.

    Raises:
        ValueError: when a pointer entry names an ephemeral version token.
    """
    from backend.storage import keys as storage_keys

    stacks: dict[str, list[str]] = {
        "model_segment": [],
        "variant": [],
        "version": [],
    }
    for entry in pointer_doc["tiers"]["embeddings"]["entries"]:
        if storage_keys.version_kind(entry["version"]) != "stable":
            raise ValueError(
                f"pointer entry version {entry['version']!r} is ephemeral: "
                "x-... tokens are never projected into the query layer"
            )
        for name, value in (
            ("model_segment", entry["model_segment"]),
            ("variant", entry["variant"]),
            ("version", entry["version"]),
        ):
            if value not in stacks[name]:
                stacks[name].append(value)
    return {name: ",".join(values) for name, values in stacks.items()}


def _create_embeddings_table(
    glue,
    database: str,
    bucket: str,
    *,
    projection_values: dict[str, str],
) -> None:
    """Register ``postings_embeddings`` over the partitioned vector lake.

    The projection enums come from the pointer (``projection_values`` —
    the output of :func:`embeddings_projection_values`); this provisioner
    is the only catalog writer, and the pointer is the only enum source,
    so a re-provision after a flip is what re-registers a generation.
    """
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
        "PartitionKeys": [
            {"Name": name, "Type": ctype} for name, ctype in EMBEDDINGS_PARTITION_KEYS
        ],
        "Parameters": {
            # Projection rather than registered partitions: the segments are
            # pointer-derived (embeddings_projection_values), so their values
            # are known at registration time and no BatchCreatePartition call
            # is needed.
            "projection.enabled": "true",
            "projection.model_segment.type": "enum",
            "projection.model_segment.values": projection_values["model_segment"],
            "projection.variant.type": "enum",
            "projection.variant.values": projection_values["variant"],
            "projection.version.type": "enum",
            "projection.version.values": projection_values["version"],
            "storage.location.template": (
                f"s3://{bucket}/lake/embeddings/model=${{model_segment}}"
                "/variant=${variant}/version=${version}"
            ),
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

    Requires the production pointer: the embeddings table's projection
    enums are derived from it (the pointer is the catalog's one authority
    over which generations are production), so provisioning after a flip
    is what re-registers a generation — and a flip without a re-provision
    is caught by the pointer gate.

    Args:
        database: Glue database name.
        workgroup: Athena workgroup name.
        max_scan_bytes: Per-query bytes-scanned cutoff.
        dry_run: Print the plan without calling AWS.

    Returns:
        The plan dict (the blueprint of what exists after the call).

    Raises:
        RuntimeError: when the production pointer is absent (bootstrap
            first — the pointer describes reality before anything derives
            from it).
    """
    from backend.storage.store import ObjectStore

    store = ObjectStore.from_env()
    try:
        pointer_doc = read_pointer(store)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "provision requires the production pointer (the catalog derives "
            "its projection enums from it); bootstrap with: "
            "python scripts/promote_production.py --bootstrap --apply"
        ) from exc
    projection_values = embeddings_projection_values(pointer_doc)

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
            "partition_keys": [c for c, _ in INGESTION_PARTITION_KEYS],
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
            "partition_keys": [c for c, _ in EMBEDDINGS_PARTITION_KEYS],
            "projection_enums": projection_values,
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
    _create_embeddings_table(
        glue, database, bucket, projection_values=projection_values
    )
    _create_workgroup(athena, workgroup, bucket, max_scan_bytes)
    print("provision complete")
    return plan
