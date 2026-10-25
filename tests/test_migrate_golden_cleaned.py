"""Tests for the golden-set migration script (task 4.1).

The matching core (word-set coverage, unique winner) is exercised against
synthetic payloads; the file verify (row count match + untouched sha) runs
against a tiny synthetic golden JSON in ``tmp_path`` — never against the
real 39-reference file. Unmatched or ambiguous references abort the
migration with nothing written (the honest failure the verify demands).
"""

import hashlib
import io
import json

import pyarrow.parquet as pq

from backend.pipeline.eval import run as eval_run
from scripts import migrate_golden_cleaned as mig
from tests.conftest import MemorySink

CREATED_AT = "2026-10-22T00:00:00Z"


def _canonical(data: dict) -> bytes:
    """Serialize payload data exactly like ingestion's canonical JSON."""
    return json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _canon_put(store: MemorySink, source: str, payload_bytes: bytes) -> str:
    """Content-addressed put of one raw payload; returns its ``source_hash``."""
    source_hash = hashlib.sha256(payload_bytes).hexdigest()
    store.put(f"lake/raw/source={source}/sha256={source_hash}.json", payload_bytes)
    return source_hash


def _seed_payload(
    store: MemorySink,
    source: str,
    field: str,
    text: str,
    source_id: str | None = None,
) -> tuple[str, dict]:
    """Put one raw payload object; return ``(source_hash, data)``.

    The hash is the canonical bytes' sha256 — the lake's content-addressed
    identity — and the key parses as a real raw key.
    """
    data: dict = {field: text}
    if source_id is not None:
        data["source_id"] = source_id
    return _canon_put(store, source, _canonical(data)), data


def _rp(source: str, source_hash: str, data: dict) -> mig.RawPayload:
    """A raw payload shaped like the store reader's product."""
    return mig.RawPayload(
        source=source,
        source_hash=source_hash,
        key=f"lake/raw/source={source}/sha256={source_hash}.json",
        data=data,
    )


def test_payload_text_normalizes_shape_field() -> None:
    payload = _rp("ashby", "0" * 64, {"description_html": "<b>Hi</b> there"})
    assert mig.payload_text(payload) == "Hi there"


def test_scan_candidates_streams_and_skips_malformed() -> None:
    """The scan reduces payloads to identity-only records, tolerating junk.

    Malformed objects are skipped in place (read-or-none semantics) and
    the candidate records carry no payload reference — the whole parsed
    lake is never resident.
    """
    store = MemorySink()
    _seed_payload(
        store, "ashby", "description_html", "alpha beta", source_id="ashby:b:1"
    )
    store.put(f"lake/raw/source=ashby/sha256={'a' * 64}.json", b"{ not json")
    cands = mig.scan_candidates(store, workers=2)
    assert len(cands) == 1  # malformed payload skipped, one candidate kept
    assert cands[0].source == "ashby"
    assert cands[0].source_id == "ashby:b:1"
    assert "alpha" in cands[0].words
    assert all(
        not isinstance(c, mig.RawPayload) for c in cands
    )  # identity-only records


def test_match_golden_unique_winner() -> None:
    payloads = [_rp("ashby", "0" * 64, {"description_html": "alpha beta gamma delta"})]
    match, coverage, runners = mig.match_golden("alpha beta", mig.candidates(payloads))
    assert match is not None
    assert coverage == 1.0
    assert runners == 0


def test_match_golden_rejects_no_match() -> None:
    payloads = [_rp("ashby", "0" * 64, {"description_html": "alpha beta gamma"})]
    match, coverage, runners = mig.match_golden(
        "alpha omega epsilon", mig.candidates(payloads)
    )
    assert match is None
    assert coverage < mig.MIN_COVERAGE
    assert runners == 0


def test_match_golden_rejects_ambiguity() -> None:
    text = {"description_html": "alpha beta gamma delta"}
    payloads = [_rp("ashby", "0" * 64, text), _rp("lever", "f" * 64, text)]
    match, coverage, runners = mig.match_golden(
        "alpha beta gamma", mig.candidates(payloads)
    )
    assert match is None
    assert runners == 1
    assert coverage == 1.0


def test_match_golden_tiebreak_prefers_tightest_superset() -> None:
    """Scraper-overlap: full-coverage ties resolve by fewest extra words."""
    tight = {"description_html": "alpha beta gamma delta resume"}
    loose = {"content": "alpha beta gamma delta resume plus extra fluff words"}
    payloads = [
        _rp("linkedin", "1" * 64, loose),
        _rp("dataroles", "2" * 64, tight),
    ]
    match, coverage, runners = mig.match_golden("alpha beta", mig.candidates(payloads))
    assert match is not None
    assert match.source == "dataroles"  # tightest superset wins
    assert coverage == 1.0
    assert runners == 0


def test_match_golden_tiebreak_keeps_close_dupes_ambiguous() -> None:
    """Two supersets within the extra-words margin stay ambiguous."""
    a = {"content": "alpha beta gamma delta one two"}
    b = {"description_html": "alpha beta gamma delta one two three"}
    payloads = [_rp("linkedin", "1" * 64, a), _rp("dataroles", "2" * 64, b)]
    match, _, runners = mig.match_golden("alpha beta", mig.candidates(payloads))
    assert match is None
    assert runners == 1


def test_migrate_dry_then_write_round_trip(tmp_path) -> None:
    """Dry run writes nothing; --write writes one part, legacy file untouched."""
    legacy = [
        {"posting_id": "ashby-golden", "source": "manual", "text": "alpha beta"},
        {"posting_id": "lever-golden", "source": "manual", "text": "one two"},
    ]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    legacy_bytes = golden_path.read_bytes()

    store = MemorySink()
    ashby_hash, ashby_data = _seed_payload(
        store,
        "ashby",
        "description_html",
        "alpha beta gamma delta resume",
        source_id="ashby:board:42",
    )
    lever_hash, lever_data = _seed_payload(
        store, "lever", "description_html", "one two three words"
    )
    payloads = [
        _rp("ashby", ashby_hash, ashby_data),
        _rp("lever", lever_hash, lever_data),
    ]
    sha_before = hashlib.sha256(legacy_bytes).hexdigest()

    rc = mig.migrate(
        store,
        legacy,
        mig.candidates(payloads),
        sha_before=sha_before,
        write=False,
        variant="author",
        version="v1",
        created_at=CREATED_AT,
        golden_path=golden_path,
    )
    assert rc == 0
    assert not store.list_keys(eval_run.golden_prefix())  # dry: no part

    rc = mig.migrate(
        store,
        legacy,
        mig.candidates(payloads),
        sha_before=sha_before,
        write=True,
        variant="author",
        version="v1",
        created_at=CREATED_AT,
        golden_path=golden_path,
    )
    assert rc == 0

    parts = store.list_keys(eval_run.golden_prefix())
    assert len(parts) == 1
    table = pq.read_table(io.BytesIO(store.get(parts[0])))
    assert table.num_rows == len(legacy)
    rows = table.to_pylist()
    ashby_row = next(r for r in rows if r["source"] == "ashby")
    lever_row = next(r for r in rows if r["source"] == "lever")
    assert ashby_row["source_hash"] == ashby_hash
    assert ashby_row["source_id"] == "ashby:board:42"
    assert ashby_row["golden_text"] == "alpha beta"
    assert ashby_row["annotator"] == "manual"
    assert ashby_row["created_at"] == CREATED_AT
    assert lever_row["source_hash"] == lever_hash
    assert lever_row["source_id"] == ""  # absent in the payload

    assert golden_path.read_bytes() == legacy_bytes  # the legacy file is untouched

    manifest_key = "lake/manifests/datasets/cleaning_golden-author-v1.json"
    manifest = json.loads(store.get(manifest_key))
    assert manifest["kind"] == "authored_eval_dataset"
    assert manifest["input_sha256"] == hashlib.sha256(legacy_bytes).hexdigest()
    assert manifest["record_count"] == 2


def test_migrate_manifest_only(tmp_path) -> None:
    """--manifest-only skips part writing and reuses the landed part."""
    legacy = [
        {"posting_id": "solo", "source": "manual", "text": "alpha beta"},
    ]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

    store = MemorySink()
    source_hash, data = _seed_payload(
        store, "ashby", "description_html", "alpha beta gamma delta", source_id="x:1"
    )
    payloads = [_rp("ashby", source_hash, data)]
    sha_before = hashlib.sha256(golden_path.read_bytes()).hexdigest()

    assert (
        mig.migrate(
            store,
            legacy,
            mig.candidates(payloads),
            sha_before=sha_before,
            write=True,
            variant="author",
            version="v1",
            created_at=CREATED_AT,
            golden_path=golden_path,
        )
        == 0
    )
    parts_before = store.list_keys(eval_run.golden_prefix())
    manifest_before = store.get(
        "lake/manifests/datasets/cleaning_golden-author-v1.json"
    )

    rc = mig.migrate(
        store,
        legacy,
        mig.candidates(payloads),
        sha_before=sha_before,
        write=True,
        variant="author",
        version="v1",
        created_at=CREATED_AT,
        golden_path=golden_path,
        part=False,
    )
    assert rc == 0
    assert store.list_keys(eval_run.golden_prefix()) == parts_before  # no new part
    assert (
        store.get("lake/manifests/datasets/cleaning_golden-author-v1.json")
        == manifest_before
    )


def test_migrate_aborts_on_ambiguity(tmp_path) -> None:
    """Two equal-coverage candidates abort the migration; nothing written."""
    legacy = [
        {"posting_id": "ambiguous", "source": "manual", "text": "alpha beta gamma"},
    ]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

    store = MemorySink()
    payloads = []
    for source in ("ashby", "lever"):
        source_hash, data = _seed_payload(
            store, source, "description_html", "alpha beta gamma delta"
        )
        payloads.append(_rp(source, source_hash, data))

    rc = mig.migrate(
        store,
        legacy,
        mig.candidates(payloads),
        sha_before=hashlib.sha256(golden_path.read_bytes()).hexdigest(),
        write=True,
        variant="author",
        version="v1",
        created_at=CREATED_AT,
        golden_path=golden_path,
    )
    assert rc == 2
    assert not store.list_keys(eval_run.golden_prefix())


def test_migrate_aborts_on_no_match(tmp_path) -> None:
    """A golden reference matching nothing aborts; the store keeps no part."""
    legacy = [
        {"posting_id": "ghost", "source": "manual", "text": "absent words here"},
    ]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

    store = MemorySink()
    source_hash, data = _seed_payload(
        store, "ashby", "description_html", "completely different words", source_id="x"
    )

    rc = mig.migrate(
        store,
        legacy,
        mig.candidates([_rp("ashby", source_hash, data)]),
        sha_before=hashlib.sha256(golden_path.read_bytes()).hexdigest(),
        write=True,
        variant="author",
        version="v1",
        created_at=CREATED_AT,
        golden_path=golden_path,
    )
    assert rc == 2
    assert not store.list_keys(eval_run.golden_prefix())
