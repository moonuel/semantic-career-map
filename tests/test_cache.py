"""Tests for the artifact-row cache index (task 3.5)."""

from __future__ import annotations

import hashlib

import pytest

from backend.pipeline.cleaning.cache import (
    dataset_prefix,
    list_dataset_parts,
    load_cache_index,
    parse_part_key,
)
from backend.pipeline.cleaning.writer import FlushWriter
from backend.storage import ids, keys
from tests.conftest import JournalSink


def _hash(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def _row(tag: str, config: str) -> dict[str, str]:
    return {
        "source_hash": _hash(tag),
        "variant": "llm",
        "version": "v1",
        "cleaned_text": f"cleaned {tag}",
        "configuration_hash": config,
        "pipeline_version": "dev",
    }


def test_index_pairs_over_parts() -> None:
    store = JournalSink()
    w = FlushWriter(store, run_id=ids.new_id(), pipeline_version="dev", flush_rows=2)
    w.add(_row("a", _hash("c1")), batch_id="b1")
    w.add(_row("b", _hash("c1")), batch_id="b2")
    index = load_cache_index(store)
    assert index.row_count == 2
    assert (_hash("a"), _hash("c1")) in index
    assert (_hash("b"), _hash("c1")) in index
    assert (_hash("a"), _hash("c2")) not in index  # config change re-cleans
    parts = list_dataset_parts(store)
    assert len(parts) == 1
    assert index.row_count == 2


def test_index_not_shares_state_across_versions() -> None:
    store = JournalSink()
    w = FlushWriter(store, run_id=ids.new_id(), pipeline_version="dev", flush_rows=1)
    w.add(_row("a", _hash("c1")), batch_id="b1")
    empty = load_cache_index(store, version="v2")
    assert len(empty) == 0 and empty.row_count == 0


def test_parse_part_key_round_trip() -> None:
    run_id = ids.new_id()
    part = keys.artifact_part("cleaned_text", "llm", "v1", run_id, 7)
    assert parse_part_key(part) == (run_id, 7)


def test_dataset_prefix_validates_segments() -> None:
    assert (
        dataset_prefix() == "lake/artifacts/type=cleaned_text/variant=llm/version=v1/"
    )
    with pytest.raises(ValueError):
        dataset_prefix("unknown_type")


def test_part_leaf_rejects_non_part_leaves() -> None:
    with pytest.raises(ValueError):
        parse_part_key("lake/artifacts/blob.json")
