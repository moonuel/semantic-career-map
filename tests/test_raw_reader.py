"""Tests for the raw-lake reader helpers."""

from __future__ import annotations

import hashlib
import json

import pytest

from backend.ingestion.raw import canonical_json
from backend.storage import keys
from backend.storage.raw import (
    fetch_all,
    list_raw_keys,
    parse_raw_key,
    read_raw_payload,
)
from tests.conftest import MemorySink


def _digest(title: str) -> str:
    """Deterministic 64-char hex digest for a fixture title."""
    return hashlib.sha256(title.encode()).hexdigest()


def _payload(title: str) -> dict:
    """A minimal ScrapedJob-shape payload."""
    return {
        "url": f"https://example.com/{title}",
        "title": title,
        "description_html": f"<p>{title} description</p>",
        "employer": "Example Corp",
        "location": "Remote",
        "country": "US",
        "employment_type": "full_time",
        "date_posted": "2026-10-01",
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "source_id": "greenhouse:example:1",
    }


def _store_with(memory_sink: MemorySink) -> MemorySink:
    """Populate the sink with one payload per source partition."""
    for source, title in [
        ("greenhouse", "alpha"),
        ("ashby", "beta"),
        ("ashby", "gamma"),
    ]:
        payload = _payload(title)
        payload["source_id"] = f"{source}:example:1"
        memory_sink.put(
            keys.raw_payload(source, _digest(title)), canonical_json(payload)
        )
    return memory_sink


def test_parse_raw_key_round_trip(memory_sink: MemorySink) -> None:
    key = keys.raw_payload("greenhouse", _digest("alpha"))
    source, source_hash = parse_raw_key(key)
    assert source == "greenhouse"
    assert source_hash == _digest("alpha")


def test_parse_raw_key_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="not a raw payload key"):
        parse_raw_key("lake/raw/source=greenhouse/sha256=short.json")


def test_list_raw_keys_all_sources(memory_sink: MemorySink) -> None:
    _store_with(memory_sink)
    listed = list_raw_keys(memory_sink)
    assert len(listed) == 3
    assert listed == sorted(listed)
    assert all(key.startswith("lake/raw/source=") for key in listed)


def test_list_raw_keys_single_source(memory_sink: MemorySink) -> None:
    _store_with(memory_sink)
    listed = list_raw_keys(memory_sink, source="ashby")
    assert len(listed) == 2
    assert all(parse_raw_key(key)[0] == "ashby" for key in listed)


def test_read_raw_payload_returns_parsed_with_hash(memory_sink: MemorySink) -> None:
    _store_with(memory_sink)
    key = keys.raw_payload("greenhouse", _digest("alpha"))
    payload = read_raw_payload(memory_sink, key)
    assert payload.source == "greenhouse"
    assert payload.source_hash == _digest("alpha")
    assert payload.key == key
    assert payload.data["title"] == "alpha"


def test_read_raw_payload_missing_object_raises(memory_sink: MemorySink) -> None:
    key = keys.raw_payload("greenhouse", _digest("alpha"))
    with pytest.raises(FileNotFoundError):
        read_raw_payload(memory_sink, key)


def test_read_raw_payload_unparseable_json_raises(memory_sink: MemorySink) -> None:
    key = keys.raw_payload("greenhouse", _digest("alpha"))
    memory_sink.objects[key] = b"{not json"
    with pytest.raises(ValueError, match="not parseable JSON"):
        read_raw_payload(memory_sink, key)


def test_read_raw_payload_non_object_json_raises(memory_sink: MemorySink) -> None:
    key = keys.raw_payload("greenhouse", _digest("alpha"))
    memory_sink.objects[key] = json.dumps(["list"]).encode()
    with pytest.raises(TypeError, match="not a JSON object"):
        read_raw_payload(memory_sink, key)


def test_fetch_all_tolerates_malformed_in_place(memory_sink: MemorySink) -> None:
    _store_with(memory_sink)
    bad_key = keys.raw_payload("greenhouse", _digest("bad"))
    memory_sink.objects[bad_key] = b"{oops"
    fetched = fetch_all(memory_sink, "greenhouse")
    assert len(fetched) == 2
    assert [(p is None) for p in fetched] == [
        parse_raw_key(k)[1] == _digest("bad")
        for k in sorted(memory_sink.objects)
        if parse_raw_key(k)[0] == "greenhouse"
    ]
    explicit = fetch_all(
        memory_sink,
        "greenhouse",
        object_keys=[bad_key, keys.raw_payload("greenhouse", _digest("alpha"))],
    )
    assert explicit[0] is None
    assert explicit[1] is not None
