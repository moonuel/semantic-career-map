"""Tests for the seeded, source-stratified tuning sampler."""

from __future__ import annotations

import hashlib
import json

import pytest

from backend.ingestion.raw import canonical_json
from backend.pipeline import sampling
from backend.storage import keys
from tests.conftest import MemorySink


def _digest(title: str) -> str:
    """Deterministic 64-char hex digest for a fixture title."""
    return hashlib.sha256(title.encode()).hexdigest()


def _put(memory_sink: MemorySink, source: str, title: str, data: dict) -> None:
    """Place one payload at its content-addressed key."""
    memory_sink.put(keys.raw_payload(source, _digest(title)), canonical_json(data))


def _scraped_job(title: str, text: str, employer: str, n: int) -> dict:
    """A minimal ScrapedJob-shape payload with a controlled text length."""
    return {
        "url": f"https://example.com/{title}",
        "title": title,
        "description_html": text,
        "employer": employer,
        "location": "Remote",
        "country": "US",
        "employment_type": "full_time",
        "date_posted": "2026-10-01",
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "source_id": f"greenhouse:example:{n}",
    }


def _filler(name: str, length: int) -> str:
    """A filler body of exactly ``length`` chars without boilerplate words."""
    return (name + " ") * (length // (len(name) + 1))


@pytest.fixture
def lake(memory_sink: MemorySink) -> MemorySink:
    """A raw lake: three sources, varying lengths and employers.

    greenhouse: five postings — two short (employer "A"), two medium
    ("C", "D"), one long ("E"). hiringcafe: one medium plain-text posting.
    linkedin: one short markdown-only posting. One unparseable greenhouse
    object is planted directly into the object map.
    """
    _put(
        memory_sink,
        "greenhouse",
        "s1",
        _scraped_job("s1", _filler("aa", 1000), "EmployerA", 1),
    )
    _put(
        memory_sink,
        "greenhouse",
        "s2",
        _scraped_job("s2", _filler("bb", 1000), "EmployerA", 2),
    )
    _put(
        memory_sink,
        "greenhouse",
        "m1",
        _scraped_job("m1", _filler("cc", 6000), "EmployerC", 3),
    )
    _put(
        memory_sink,
        "greenhouse",
        "m2",
        _scraped_job("m2", _filler("dd", 6000), "EmployerD", 4),
    )
    _put(
        memory_sink,
        "greenhouse",
        "l1",
        _scraped_job("l1", _filler("ee", 16000), "EmployerE", 5),
    )
    _put(
        memory_sink,
        "hiringcafe",
        "h1",
        {
            "url": "https://example.com/h1",
            "title": "h1",
            "description": _filler("ff", 6000),
            "employer": "EmployerF",
            "location": "Remote",
            "workplace_type": "remote",
            "employment_types": ["full_time"],
            "date_posted": "2026-10-01",
            "salary": None,
            "source_id": "hiringcafe:abc",
            "extracted": False,
            "extra": {},
        },
    )
    _put(
        memory_sink,
        "linkedin",
        "k1",
        {"content": _filler("gg", 1000), "format": "markdown"},
    )
    memory_sink.objects[keys.raw_payload("greenhouse", _digest("bad"))] = b"{oops"
    return memory_sink


def test_deterministic_byte_identical_manifests(lake: MemorySink) -> None:
    first = sampling.sample_bytes(lake, per_source=3, bands=(5000, 15000), seed=7)
    second = sampling.sample_bytes(lake, per_source=3, bands=(5000, 15000), seed=7)
    assert first == second
    other = json.loads(
        sampling.sample_bytes(lake, per_source=3, bands=(5000, 15000), seed=8)
    )
    hashes = [(p["source"], p["source_hash"]) for p in other["postings"]]
    assert hashes == sorted(hashes)


def test_quota_and_bands_respected(lake: MemorySink) -> None:
    manifest = json.loads(
        sampling.sample_bytes(lake, per_source=3, bands=(5000, 15000), seed=7)
    )
    assert manifest["params"] == {"per_source": 3, "bands": [5000, 15000], "seed": 7}
    gh = [p for p in manifest["postings"] if p["source"] == "greenhouse"]
    assert len(gh) == 3
    assert len({p["band"] for p in gh}) == 3  # spread across all three bands


def test_employer_diversity_within_band(lake: MemorySink) -> None:
    manifest = json.loads(
        sampling.sample_bytes(lake, per_source=4, bands=(5000, 15000), seed=3)
    )
    gh4 = [p for p in manifest["postings"] if p["source"] == "greenhouse"]
    assert len(gh4) == 4
    # allocation 2+1+1: band 0 (both EmployerA) unaffected by the draw cap
    assert [p["employer"] for p in gh4].count("EmployerA") == 2


def test_quota_exceeding_availability_takes_all(lake: MemorySink) -> None:
    manifest = json.loads(
        sampling.sample_bytes(lake, per_source=50, bands=(5000, 15000), seed=1)
    )
    assert len(manifest["postings"]) == 7


def test_shape_fields_in_manifest(lake: MemorySink) -> None:
    manifest = json.loads(
        sampling.sample_bytes(lake, per_source=50, bands=(5000, 15000), seed=1)
    )
    by_source = {p["source"]: p for p in manifest["postings"]}
    assert by_source["hiringcafe"]["source_id"] == "hiringcafe:abc"
    assert by_source["greenhouse"]["employer"] == "EmployerA"
    # linkedin carries neither source_id nor employer (shape 3)
    assert by_source["linkedin"]["source_id"] == ""
    assert by_source["linkedin"]["employer"] == ""
    assert all(
        len(p["source_hash"]) == 64 and p["length_chars"] > 0
        for p in manifest["postings"]
    )


def test_malformed_payloads_excluded(lake: MemorySink) -> None:
    manifest = json.loads(
        sampling.sample_bytes(lake, per_source=50, bands=(5000, 15000), seed=1)
    )
    bad_key = keys.raw_payload("greenhouse", _digest("bad"))
    assert all(p["key"] != bad_key for p in manifest["postings"])


def test_band_of_boundaries() -> None:
    bands = (5000, 15000)
    assert sampling.band_of(0, bands) == 0
    assert sampling.band_of(4999, bands) == 0
    assert sampling.band_of(5000, bands) == 1
    assert sampling.band_of(14999, bands) == 1
    assert sampling.band_of(15000, bands) == 2


def test_allocate_spreads_and_caps() -> None:
    # 5-quota, bands of sizes (2, 2, 1): even spread until quota runs out.
    assert sampling._allocate(5, [2, 2, 1]) == [2, 2, 1]
    # Quota scarcity over denser bands: densest-first, capped at availability.
    assert sampling._allocate(2, [10, 10, 10]) == [1, 1, 0]
    assert sampling._allocate(0, [1, 1]) == [0, 0]
    # Cannot exceed availability.
    assert sampling._allocate(10, [2, 1, 0]) == [2, 1, 0]


def test_allocate_remainder_prefers_denser() -> None:
    # Quota 2, bands (10, 1): both bands get scarce units first (min-1 pass).
    assert sampling._allocate(2, [10, 1]) == [1, 1]
