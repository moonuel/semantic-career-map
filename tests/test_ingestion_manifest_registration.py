"""Ingestion-manifest table registration: the partition key must be virtual.

Regression guard for the defect that made ``manifests_ingestion`` unqueryable
in Athena (``HIVE_INVALID_METADATA: Table descriptor contains duplicate
columns``), which broke the cleaning tier's recency window for four days.

Two failures had to line up:

1. ``ingested_on`` was declared both as a storage column and as the partition
   key. Hive partition columns are virtual — naming it twice makes Athena
   reject every query against the table.
2. The drift check compared only ``StorageDescriptor``, so the provisioner
   could not see the drift and would never repair it.

These tests pin both.
"""

from __future__ import annotations

from scripts.provision_query_layer import (
    INGESTION_COLUMNS,
    INGESTION_PARTITION_KEYS,
    INGESTION_TABLE,
    _apply_managed_table,
    _create_ingestion_manifest_table,
    _storage_descriptor_matches,
    _table_matches,
)
from tests.test_embedding_registration import GlueDouble


def _target_table_input() -> dict:
    """The table_input the provisioner would apply, via the real code path."""
    glue = GlueDouble()
    _create_ingestion_manifest_table(glue, "semantic_career_map", "bucket")
    return glue.created[0]


def test_no_partition_key_is_also_a_storage_column() -> None:
    # D1: the exact shape Athena rejects.
    partition_names = {name for name, _ in INGESTION_PARTITION_KEYS}
    storage_names = {name for name, _ in INGESTION_COLUMNS}
    assert partition_names == {"ingested_on"}
    assert not partition_names & storage_names


def test_registration_declares_the_partition_once() -> None:
    table_input = _target_table_input()
    column_names = [c["Name"] for c in table_input["StorageDescriptor"]["Columns"]]
    assert "ingested_on" not in column_names
    assert table_input["PartitionKeys"] == [{"Name": "ingested_on", "Type": "date"}]
    assert table_input["Name"] == INGESTION_TABLE


def test_registration_keeps_day_projection_and_location_template() -> None:
    parameters = _target_table_input()["Parameters"]
    assert parameters["projection.enabled"] == "true"
    assert parameters["projection.ingested_on.type"] == "date"
    assert parameters["storage.location.template"].endswith(
        "/lake/manifests/ingestion/ingested_on=${ingested_on}"
    )


def test_matches_when_stored_equals_target() -> None:
    table_input = _target_table_input()
    stored = {
        **table_input,
        "StorageDescriptor": dict(table_input["StorageDescriptor"]),
    }
    assert _table_matches(stored, table_input)


def test_partition_drift_is_detected() -> None:
    """The live broken state: stored carries ingested_on as a column AND as
    the partition key; the corrected target carries it only as the key."""
    table_input = _target_table_input()
    stored = {
        **table_input,
        "StorageDescriptor": {
            **table_input["StorageDescriptor"],
            "Columns": [
                *table_input["StorageDescriptor"]["Columns"],
                {"Name": "ingested_on", "Type": "date"},
            ],
        },
    }
    # same location/formats/serde, same partition key, same parameters —
    # the ONLY difference is the duplicated column, and it must be visible
    # to the provisioner (it was not, which is why the table stayed broken).
    assert not _storage_descriptor_matches(
        stored["StorageDescriptor"],
        table_input["StorageDescriptor"],
        partition_names={"ingested_on"},
    )
    assert not _table_matches(stored, table_input)


def test_partition_key_change_alone_is_detected() -> None:
    table_input = _target_table_input()
    stored = {**table_input, "PartitionKeys": []}
    stored["StorageDescriptor"] = dict(table_input["StorageDescriptor"])
    assert not _table_matches(stored, table_input)


def test_projection_parameter_drift_is_detected() -> None:
    table_input = _target_table_input()
    stored = {**table_input, "Parameters": {"projection.enabled": "false"}}
    stored["StorageDescriptor"] = dict(table_input["StorageDescriptor"])
    assert not _table_matches(stored, table_input)


def test_extra_catalog_parameters_do_not_trigger_an_update() -> None:
    # Glue synthesises keys of its own; reverting them would be a regression.
    table_input = _target_table_input()
    stored = {
        **table_input,
        "Parameters": {**table_input["Parameters"], "EXTERNAL": "TRUE"},
        "StorageDescriptor": dict(table_input["StorageDescriptor"]),
    }
    assert _table_matches(stored, table_input)


def test_extra_stored_column_does_not_trigger_an_update() -> None:
    table_input = _target_table_input()
    stored = {
        **table_input,
        "StorageDescriptor": {
            **table_input["StorageDescriptor"],
            "Columns": [
                *table_input["StorageDescriptor"]["Columns"],
                {"Name": "externally_added", "Type": "string"},
            ],
        },
    }
    assert _table_matches(stored, table_input)


def test_broken_registration_gets_repaired() -> None:
    """The end-to-end guard: a stored table with the duplicate column is
    updated rather than reported as current."""
    table_input = _target_table_input()
    broken = {
        **table_input,
        "StorageDescriptor": {
            **table_input["StorageDescriptor"],
            "Columns": [
                *table_input["StorageDescriptor"]["Columns"],
                {"Name": "ingested_on", "Type": "date"},
            ],
        },
    }
    glue = GlueDouble(table=broken)
    _apply_managed_table(glue, "semantic_career_map", INGESTION_TABLE, table_input)
    assert len(glue.updated) == 1
    applied_columns = [
        c["Name"] for c in glue.updated[0]["StorageDescriptor"]["Columns"]
    ]
    assert "ingested_on" not in applied_columns
    assert glue.updated[0]["PartitionKeys"] == [{"Name": "ingested_on", "Type": "date"}]
