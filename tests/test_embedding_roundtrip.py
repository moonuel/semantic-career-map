"""Round-trip driver tests: fake Athena + MemorySink — part commits, schema
columns, manifest lineage, zero-row no-op, config-object accounting."""

from __future__ import annotations

import io
import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from backend.pipeline.embedding import driver
from backend.pipeline.embedding.config import (
    EmbeddingConfig,
    configuration_hash,
    dataset_id,
)
from backend.pipeline.embedding.encoder import StubEncoder
from backend.storage import keys
from tests.conftest import JournalSink, MemorySink


@pytest.fixture(autouse=True)
def lake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The store-rooted env pieces results_location/athena helpers read."""
    monkeypatch.setenv("STORE_S3_BUCKET", "lake")
    monkeypatch.setenv("PIPELINE_VERSION", "emb-test-1")


H1 = "0" * 64
H2 = "1" * 64
H3 = "2" * 64
HQ = "f" * 64


class FakeAthena:
    """Athena stand-in (mirrors the cleaning tier's double)."""

    def __init__(self, page: list[list[str]]) -> None:
        self._page = page
        self.executed: list[str] = []
        self.failure_state: str | None = None

    def start_query_execution(self, *, QueryString: str, **kw) -> dict:
        self.executed.append(QueryString)
        return {"QueryExecutionId": "q-emb-1"}

    def get_query_execution(self, *, QueryExecutionId: str) -> dict:
        if self.failure_state:
            return {
                "QueryExecution": {
                    "State": "FAILED",
                    "Status": {
                        "State": "FAILED",
                        "StateChangeReason": self.failure_state,
                    },
                }
            }
        return {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}

    def get_query_results(self, *, QueryExecutionId: str, NextToken=None) -> dict:
        return {
            "ResultSet": {
                "Rows": [
                    {"Data": [{"VarCharValue": c} for c in row]} for row in self._page
                ]
            }
        }


def _config(**kw) -> EmbeddingConfig:
    defaults = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_revision": "rev-1",
        "model_segment": "all-minilm-l6-v2",
        "variant": "minilm",
        "text_variant": "variant_text",
        "batch_size": 2,
        "version": "v1",
        "sql_text": "SELECT a.source_hash, a.cleaned_text AS variant_text "
        "FROM artifacts_cleaned a LEFT JOIN postings_embeddings e "
        "ON e.source_hash = a.source_hash AND e.configuration_hash = :config "
        "WHERE e.source_hash IS NULL",
    }
    defaults.update(kw)
    return EmbeddingConfig(**defaults)


def _page(*texts: str) -> list[list[str]]:
    hashes = [f"{i}" * 64 for i in range(len(texts))]
    return [["source_hash", "variant_text"]] + [[h, t] for h, t in zip(hashes, texts)]


def _read_part(store, key: str):
    return pq.read_table(io.BytesIO(store.get(key)))


def test_round_trip_commits_parts_and_manifest(memory_sink: MemorySink) -> None:
    athena = FakeAthena(_page("text one", "text two", "text three"))
    encoder = StubEncoder(dimension=8)
    config = _config()
    summary = driver.run_embedding(
        memory_sink, athena, encoder, config, pipeline_version="dev"
    )

    assert summary.selected == 3 and summary.embedded == 3 and summary.empty == 0
    assert not summary.zero_delta
    assert len(summary.part_keys) == 2  # batch_size 2 -> parts seq 1,2
    for seq, key in enumerate(summary.part_keys, start=1):
        assert key == keys.embedding_part(
            "all-minilm-l6-v2", "minilm", "v1", summary.run_id, seq
        )
    # parts are validated keys and readable parquet with the full schema
    table = _read_part(memory_sink, summary.part_keys[0])
    expected_cols = {
        "source_hash",
        "embedding",
        "model",
        "model_version",
        "input_artifact",
        "input_version",
        "embedding_dimension",
        "normalization",
        "created_at",
        "configuration_hash",
        "run_id",
    }
    assert set(table.column_names) == expected_cols
    vectors = np.stack(table["embedding"].to_pylist())
    assert vectors.shape[0] == 2
    assert table["embedding_dimension"].to_pylist() == [8, 8]
    assert table["configuration_hash"].to_pylist() == [configuration_hash(config)] * 2
    assert table["normalization"].to_pylist() == ["l2", "l2"]
    # order preserved across batch windows
    assert table["source_hash"].to_pylist() == [H1, H2]
    table2 = _read_part(memory_sink, summary.part_keys[1])
    assert table2["source_hash"].to_pylist() == [H3]

    manifest = json.loads(memory_sink.get(summary.dataset_manifest_key))
    assert manifest["dataset_id"] == dataset_id(config)
    assert manifest["record_count"] == 3
    assert manifest["parts"] == summary.part_keys
    assert manifest["query_execution_id"] == "q-emb-1"
    assert manifest["model_revision"] == "rev-1"
    assert manifest["pipeline_version"] == "dev"

    config_key = keys.config(configuration_hash(config))
    stored = json.loads(memory_sink.get(config_key))
    assert stored["config"]["sql_text"] == config.sql_text


def test_zero_delta_is_healthy_noop(memory_sink: MemorySink) -> None:
    athena = FakeAthena([["source_hash", "variant_text"]])
    summary = driver.run_embedding(memory_sink, athena, StubEncoder(8), _config())
    assert summary.zero_delta and summary.embedded == 0 and summary.selected == 0
    assert summary.part_keys == [] and summary.dataset_manifest_key is None
    assert list(memory_sink.objects) == [keys.config(configuration_hash(_config()))]


def test_empty_rows_counted_not_dropped(memory_sink: MemorySink) -> None:
    page = [
        ["source_hash", "variant_text"],
        [H1, "   "],
        [H2, "kept"],
    ]
    athena = FakeAthena(page)
    summary = driver.run_embedding(memory_sink, athena, StubEncoder(8), _config())
    assert summary.empty == 1
    assert summary.embedded == 1 and summary.selected == 2


def test_failed_query_state_writes_nothing(memory_sink: MemorySink) -> None:
    athena = FakeAthena(_page("a"))
    athena.failure_state = "FAILED"
    with pytest.raises(RuntimeError):
        driver.run_embedding(memory_sink, athena, StubEncoder(8), _config())
    keys_set = [keys.config(configuration_hash(_config()))]
    assert list(memory_sink.objects) == keys_set


def test_encoder_row_mismatch_raises(memory_sink: MemorySink) -> None:
    class DerpyEncoder(StubEncoder):
        def encode(self, texts):
            return np.empty((0, 8), dtype=np.float32)

    athena = FakeAthena(_page("a"))
    with pytest.raises(RuntimeError, match="encoder returned"):
        driver.run_embedding(memory_sink, athena, DerpyEncoder(), _config())


def test_empty_sql_text_rejected(memory_sink: MemorySink) -> None:
    with pytest.raises(ValueError, match="sql_text is empty"):
        driver.run_embedding(
            memory_sink, FakeAthena([]), StubEncoder(8), _config(sql_text="")
        )


@pytest.fixture
def journal() -> JournalSink:
    """A fresh put-order-recording sink per test."""
    return JournalSink()


def test_config_object_write_is_idempotent(journal: JournalSink) -> None:

    config = _config()
    first = driver.ensure_config_object(journal, config)
    second = driver.ensure_config_object(journal, config)
    assert first == second == configuration_hash(config)
    assert journal.journal.count(keys.config(first)) == 1
