"""Unit tests for the embedding configuration: canonical hashing, dataset
ids, and the config document."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from backend.pipeline.embedding.config import (
    EmbeddingConfig,
    config_document,
    configuration_hash,
    dataset_id,
    default_model_segment,
)


def _config(**kw) -> EmbeddingConfig:
    defaults = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_revision": "a" * 40,
        "model_segment": "all-minilm-l6-v2",
        "variant": "minilm",
        "text_variant": "variant_text",
        "version": "v1",
        "sql_text": "SELECT 1",
    }
    defaults.update(kw)
    return EmbeddingConfig(**defaults)


HASH_SCRIPT = (
    "from backend.pipeline.embedding.config import EmbeddingConfig, configuration_hash;"
    "print(configuration_hash(EmbeddingConfig("
    "model_name='m', model_revision='r', model_segment='m2', variant='v',"
    "text_variant='t', version='v1', sql_text='SELECT 1')))"
)


def test_cross_process_hash_stability() -> None:
    """Same values in two separate processes hash identically."""
    first = subprocess.run(
        [sys.executable, "-c", HASH_SCRIPT], capture_output=True, text=True, check=True
    ).stdout.strip()
    second = subprocess.run(
        [sys.executable, "-c", HASH_SCRIPT], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert first == second
    assert len(first) == 64 and all(c in "0123456789abcdef" for c in first)


@pytest.mark.parametrize(
    "field", ["model_name", "model_revision", "variant", "sql_text"]
)
def test_single_field_change_re_hashes(field: str) -> None:
    base = configuration_hash(_config())
    changed = configuration_hash(_config(**{field: "changed-value"}))
    assert base != changed


def test_hash_is_timestamp_and_device_free() -> None:
    """No runtime-affine value can enter the hash (none is a field)."""
    field_names = [f.name for f in EmbeddingConfig.__dataclass_fields__.values()]
    assert "device" not in field_names
    assert "created_at" not in field_names


def test_config_document_carries_hash_and_sql() -> None:
    config = _config()
    document = json.loads(config_document(config))
    assert document["configuration_hash"] == configuration_hash(config)
    assert document["config"]["sql_text"] == "SELECT 1"
    # byte-stable for idempotent re-writes
    assert config_document(config) == config_document(_config())


def test_dataset_id_plain_and_unique_per_hash() -> None:
    config = _config()
    identifier = dataset_id(config)
    assert identifier.startswith("embeddings-all-minilm-l6-v2-minilm-v1-")
    assert "=" not in identifier
    other = dataset_id(_config(batch_size=8))
    assert other != identifier


def test_default_model_segment() -> None:
    assert (
        default_model_segment("sentence-transformers/all-MiniLM-L6-v2")
        == "all-minilm-l6-v2"
    )
    with pytest.raises(ValueError):
        default_model_segment("  /  ")
