"""Selection SQL + query-layer registration tests: the embeddings table
definition, idempotent registration updates, and the parity gate path."""

from __future__ import annotations

from backend.pipeline.embedding import selection
from scripts.provision_query_layer import (
    EMBEDDINGS_COLUMNS,
    EMBEDDINGS_PARTITION_KEYS,
    EMBEDDINGS_TABLE,
    _create_embeddings_table,
    embeddings_location,
)


class GlueExceptions:
    class EntityNotFoundException(Exception):
        pass


class GlueDouble:
    """Records create/update; `table` plays the catalog's stored table.

    The stored table must be the WHOLE ``Table`` (descriptor plus
    ``PartitionKeys``/``Parameters``): handing back only the descriptor is
    the shape that let the manifests_ingestion partition drift go unseen.
    """

    def __init__(
        self, descriptor: dict | None = None, table: dict | None = None
    ) -> None:
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self._table = (
            table
            if table is not None
            else ({"StorageDescriptor": descriptor} if descriptor is not None else None)
        )

    def get_table(self, *, DatabaseName: str, Name: str):
        if self._table is None:
            raise GlueExceptions.EntityNotFoundException()
        return {"Table": self._table}

    def create_table(self, *, DatabaseName: str, TableInput: dict) -> None:
        self.created.append(TableInput)
        self._table = TableInput

    def update_table(self, *, DatabaseName: str, TableInput: dict) -> None:
        self.updated.append(TableInput)
        self._table = TableInput

    @property
    def exceptions(self):
        return GlueExceptions()


def test_embeddings_table_definition_shape() -> None:
    types = dict(EMBEDDINGS_COLUMNS)
    assert types["source_hash"] == "string"
    assert types["configuration_hash"] == "string"
    assert types["embedding"] == "array<float>"
    assert types["embedding_dimension"] == "bigint"
    assert types["run_id"] == "string"
    assert types["created_at"] == "timestamp"


def test_create_embeddings_table_pins_location_and_columns() -> None:
    glue = GlueDouble()
    _create_embeddings_table(glue, "semantic_career_map", "bucket")
    assert len(glue.created) == 1
    table_input = glue.created[0]
    assert table_input["Name"] == EMBEDDINGS_TABLE
    descriptor = table_input["StorageDescriptor"]
    assert (
        descriptor["Location"]
        == embeddings_location("bucket")
        == "s3://bucket/lake/embeddings/"
    )
    assert descriptor["Columns"] == [
        {"Name": c, "Type": t} for c, t in EMBEDDINGS_COLUMNS
    ]


def test_embeddings_partition_keys_cover_path_segments() -> None:
    glue = GlueDouble()
    _create_embeddings_table(glue, "semantic_career_map", "bucket")
    keys = glue.created[0]["PartitionKeys"]
    assert keys == [
        {"Name": name, "Type": ctype} for name, ctype in EMBEDDINGS_PARTITION_KEYS
    ]
    assert [k["Name"] for k in keys] == ["model_segment", "variant", "version"]
    assert all(k["Type"] == "string" for k in keys)


def test_embeddings_partition_keys_never_collide_with_columns() -> None:
    # A partition key sharing a column name makes the table unqueryable
    # ("Table descriptor contains duplicate columns"). The row owns a
    # ``model`` column, so the key is ``model_segment``.
    column_names = {name for name, _ in EMBEDDINGS_COLUMNS}
    partition_names = {name for name, _ in EMBEDDINGS_PARTITION_KEYS}
    assert not (column_names & partition_names)
    assert "model" in column_names
    assert "model" not in partition_names


def test_embeddings_projection_templates_the_partition_path() -> None:
    glue = GlueDouble()
    _create_embeddings_table(glue, "semantic_career_map", "bucket")
    params = glue.created[0]["Parameters"]
    assert params["projection.enabled"] == "true"
    assert params["projection.model_segment.values"] == "all-minilm-l6-v2"
    assert params["projection.variant.values"] == "minilm"
    assert params["projection.version.values"] == "v1"
    assert params["storage.location.template"] == (
        "s3://bucket/lake/embeddings/model=${model_segment}"
        "/variant=${variant}/version=${version}"
    )


def test_registration_idempotent_when_current() -> None:
    fresh = GlueDouble()
    _create_embeddings_table(fresh, "semantic_career_map", "bucket")
    current_table = fresh.created[0]

    current = GlueDouble(table=current_table)
    _create_embeddings_table(current, "semantic_career_map", "bucket")
    assert not current.created and not current.updated


def test_unpartitioned_registration_is_repaired() -> None:
    # The pre-partition registration: pinned location, no PartitionKeys.
    stale = GlueDouble(
        table={
            "StorageDescriptor": {
                "Location": (
                    "s3://bucket/lake/embeddings/"
                    "model=all-minilm-l6-v2/variant=minilm/version=v1/"
                ),
                "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
                "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
                "SerdeInfo": {
                    "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
                },
                "Columns": [{"Name": c, "Type": t} for c, t in EMBEDDINGS_COLUMNS],
            }
        }
    )
    _create_embeddings_table(stale, "semantic_career_map", "bucket")
    assert len(stale.updated) == 1
    assert stale.updated[0]["PartitionKeys"] == [
        {"Name": name, "Type": ctype} for name, ctype in EMBEDDINGS_PARTITION_KEYS
    ]
    assert stale.updated[0]["StorageDescriptor"]["Location"] == (
        "s3://bucket/lake/embeddings/"
    )


def test_stale_registration_updates_and_never_downgrades() -> None:
    stale = GlueDouble(
        descriptor={
            "Location": embeddings_location("bucket"),
            "InputFormat": "x",
            "OutputFormat": "y",
            "SerdeInfo": {"SerializationLibrary": "old"},
            "Columns": [{"Name": "source_hash", "Type": "string"}],
        }
    )
    _create_embeddings_table(stale, "semantic_career_map", "bucket")
    assert len(stale.updated) == 1
    applied = stale.updated[0]["StorageDescriptor"]
    assert applied["Columns"] == [{"Name": c, "Type": t} for c, t in EMBEDDINGS_COLUMNS]
    assert applied["Location"] == embeddings_location("bucket")


def test_selection_sql_names_the_registered_table() -> None:
    # a rename must touch both sides together
    assert EMBEDDINGS_TABLE in selection.select_unembedded_sql()
