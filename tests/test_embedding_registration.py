"""Selection SQL + query-layer registration tests: the embeddings table
definition, idempotent registration updates, the parity gate path, and the
pointer-derived projection enums (the pointer is the catalog's one enum
authority — provision re-registers a generation after a flip)."""

from __future__ import annotations

import pytest

from backend.pipeline.embedding import selection
from scripts.provision_query_layer import (
    EMBEDDINGS_COLUMNS,
    EMBEDDINGS_PARTITION_KEYS,
    EMBEDDINGS_TABLE,
    _create_embeddings_table,
    embeddings_location,
    embeddings_projection_values,
)

# The pointer's tiers → projection enums (the pointer_doc() fixture shape
# matches the live bootstrap document: one all-minilm entry + cleaning v1).
POINTER_DOC = {
    "pointer_version": 1,
    "updated_by": "olimoon",
    "updated_at": "2026-10-28T17:00:00+00:00",
    "reason": "bootstrap: describes reality",
    "previous": None,
    "tiers": {
        "cleaning": {
            "artifact_type": "cleaned_text",
            "variant": "llm",
            "version": "v1",
        },
        "embeddings": {
            "entries": [
                {
                    "model_segment": "all-minilm-l6-v2",
                    "variant": "minilm",
                    "version": "v1",
                }
            ]
        },
    },
}

PROJECTION_VALUES = {
    "model_segment": "all-minilm-l6-v2",
    "variant": "minilm",
    "version": "v1",
}


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


def _register(glue: GlueDouble) -> dict | None:
    """Register with the pointer-derived enums; the created input if it created."""
    _create_embeddings_table(
        glue, "semantic_career_map", "bucket", projection_values=PROJECTION_VALUES
    )
    return glue.created[0] if glue.created else None


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
    _register(glue)
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
    _register(glue)
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
    _register(glue)
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
    _register(fresh)
    current_table = fresh.created[0]

    current = GlueDouble(table=current_table)
    _register(current)
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
    _register(stale)
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
    _register(stale)
    assert len(stale.updated) == 1
    applied = stale.updated[0]["StorageDescriptor"]
    assert applied["Columns"] == [{"Name": c, "Type": t} for c, t in EMBEDDINGS_COLUMNS]
    assert applied["Location"] == embeddings_location("bucket")


def test_selection_sql_names_the_registered_table() -> None:
    # a rename must touch both sides together
    assert EMBEDDINGS_TABLE in selection.select_unembedded_sql()


# --- Pointer-derived projection enums ----------------------------------------


class TestEmbeddingsProjectionValues:
    """The catalog's single enum source: the production pointer."""

    def test_derives_the_live_enums(self) -> None:
        assert embeddings_projection_values(POINTER_DOC) == PROJECTION_VALUES

    def test_multi_entry_enums_are_column_joined(self) -> None:
        doc = {
            **POINTER_DOC,
            "tiers": {
                "cleaning": POINTER_DOC["tiers"]["cleaning"],
                "embeddings": {
                    "entries": [
                        POINTER_DOC["tiers"]["embeddings"]["entries"][0],
                        {
                            "model_segment": "bge-small",
                            "variant": "bge",
                            "version": "v2",
                        },
                    ]
                },
            },
        }
        values = embeddings_projection_values(doc)
        assert values == {
            "model_segment": "all-minilm-l6-v2,bge-small",
            "variant": "minilm,bge",
            "version": "v1,v2",
        }

    def test_ephemeral_entry_is_never_projected(self) -> None:
        doc = {
            **POINTER_DOC,
            "tiers": {
                "cleaning": POINTER_DOC["tiers"]["cleaning"],
                "embeddings": {
                    "entries": [
                        {
                            "model_segment": "all-minilm-l6-v2",
                            "variant": "minilm",
                            "version": "x-scratch",
                        }
                    ]
                },
            },
        }
        with pytest.raises(ValueError, match="never projected"):
            embeddings_projection_values(doc)

    def test_registration_carries_pointer_enums_after_flip(self) -> None:
        """A flip's re-provision re-registers the new generation's enum."""
        flipped = PROJECTION_VALUES | {"version": "v2"}
        glue = GlueDouble()
        _create_embeddings_table(
            glue, "semantic_career_map", "bucket", projection_values=flipped
        )
        params = glue.created[0]["Parameters"]
        assert params["projection.version.values"] == "v2"
        assert params["projection.model_segment.values"] == "all-minilm-l6-v2"
