"""Selection SQL + query-layer registration tests: the embeddings table
definition, idempotent registration updates, and the parity gate path."""

from __future__ import annotations

from backend.pipeline.embedding import selection
from scripts.provision_query_layer import (
    EMBEDDINGS_COLUMNS,
    EMBEDDINGS_TABLE,
    _create_embeddings_table,
    embeddings_location,
)


class GlueExceptions:
    class EntityNotFoundException(Exception):
        pass


class GlueDouble:
    """Records create/update; `descriptor` plays the catalog's stored table."""

    def __init__(self, descriptor: dict | None = None) -> None:
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self._descriptor = descriptor

    def get_table(self, *, DatabaseName: str, Name: str):
        if self._descriptor is None:
            raise GlueExceptions.EntityNotFoundException()
        return {"Table": {"StorageDescriptor": self._descriptor}}

    def create_table(self, *, DatabaseName: str, TableInput: dict) -> None:
        self.created.append(TableInput)

    def update_table(self, *, DatabaseName: str, TableInput: dict) -> None:
        self.updated.append(TableInput)
        self._descriptor = TableInput["StorageDescriptor"]

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
        == (
            "s3://bucket/lake/embeddings/model=all-minilm-l6-v2/variant=minilm/version=v1/"
        )
    )
    assert descriptor["Columns"] == [
        {"Name": c, "Type": t} for c, t in EMBEDDINGS_COLUMNS
    ]


def test_registration_idempotent_when_current() -> None:
    fresh = GlueDouble()
    _create_embeddings_table(fresh, "semantic_career_map", "bucket")
    current_descriptor = fresh.created[0]["StorageDescriptor"]

    current = GlueDouble(descriptor=current_descriptor)
    _create_embeddings_table(current, "semantic_career_map", "bucket")
    assert not current.created and not current.updated


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
