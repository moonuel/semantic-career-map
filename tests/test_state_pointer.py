"""Unit tests for the production pointer module (Layer 3 harvest).

Covers the document contract (canonicalization, hashing), the parse
validator (re-mint through key builders, stable tokens only, exhaustive
tier map), the two content-addressed mutations (``create``, ``replace``)
over the MemorySink double, and the S3 ``If-Match`` translation on a
fake client (``put_if_match`` + ``get_with_etag`` on the real ObjectStore
shell).
"""

from __future__ import annotations

import io

import pytest
from botocore.exceptions import ClientError

from backend.storage import keys, pointer
from backend.storage.store import ObjectStore
from tests.conftest import JournalSink, MemorySink

# --- Fixtures ----------------------------------------------------------------

UTC_STAMP = "2026-10-28T17:00:00+00:00"
HASH = "a" * 64
OTHER_HASH = "b" * 64


def valid_doc() -> dict:
    """A minimal schema-valid pointer document (spike reality)."""
    return {
        "pointer_version": 1,
        "updated_by": "olimoon",
        "updated_at": UTC_STAMP,
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


def seed_pointer(sink: MemorySink, doc: dict) -> None:
    """Write ``doc`` as the sink's live pointer bytes."""
    sink.objects[keys.state_pointer()] = pointer.canonical_bytes(doc)


# --- 1. Canonical content hashing -------------------------------------------


class TestContentHash:
    """The document's identity: canonical bytes, one digest."""

    def test_deterministic_and_independent_of_dict_order(self) -> None:
        shuffled = valid_doc()
        shuffled["tiers"] = dict(reversed(list(shuffled["tiers"].items())))
        assert pointer.content_hash(valid_doc()) == pointer.content_hash(shuffled)

    def test_distinct_documents_hash_distinctly(self) -> None:
        doc = valid_doc()
        doc["reason"] = "promoted"
        assert pointer.content_hash(valid_doc()) != pointer.content_hash(doc)

    def test_hash_is_64_hex(self) -> None:
        assert keys.is_sha256_hex(pointer.content_hash(valid_doc()))


# --- 2. parse_document -------------------------------------------------------


class TestParseDocument:
    """Pointer shape: version gate, exhaustive tiers, builder re-mints."""

    def test_valid_document_round_trips(self) -> None:
        doc = valid_doc()
        assert pointer.parse_document(doc) is doc

    @pytest.mark.parametrize(
        "mutate, fragment",
        [
            (lambda d: d.update(pointer_version=2), "other majors"),
            (lambda d: d.update(updated_by=""), "updated_by"),
            (lambda d: d.update(reason=None), "reason"),
            (lambda d: d.update(previous="nothex"), "previous must be null"),
            (lambda d: d["tiers"].pop("cleaning"), "must tier exactly"),
            (lambda d: d["tiers"].update(eval={}), "unknown tier"),
            (
                lambda d: d["tiers"]["cleaning"].update(artifact_type="unheard_of"),
                "unknown artifact type",
            ),
            (
                lambda d: d["tiers"]["cleaning"].update(version="x-v1"),
                "artifact version",
            ),
            (
                lambda d: d["tiers"]["embeddings"]["entries"][0].update(version="x-v1"),
                "never production",
            ),
            (
                lambda d: d["tiers"]["embeddings"]["entries"][0].update(
                    model_segment="Bad_Segment"
                ),
                "does not re-mint",
            ),
            (
                lambda d: d["tiers"]["embeddings"].update(entries=[]),
                "non-empty list of entries",
            ),
            (lambda d: d.update(tiers={}), "must tier exactly"),
        ],
        ids=[
            "other-major",
            "empty-updated-by",
            "missing-reason",
            "bad-previous",
            "missing-tier",
            "unknown-tier",
            "unknown-artifact-type",
            "ephemeral-cleaning",
            "ephemeral-embedding",
            "malformed-model",
            "empty-entries",
            "tiers-shape",
        ],
    )
    def test_refuses(self, mutate, fragment: str) -> None:
        doc = valid_doc()
        mutate(doc)
        with pytest.raises(ValueError, match=fragment):
            pointer.parse_document(doc)


# --- 3. create (bootstrap) ---------------------------------------------------


class TestCreate:
    """The conditional first write; a race loser proceeds, never overwrites."""

    def test_creates_canonical_bytes(self, memory_sink: MemorySink) -> None:
        doc = valid_doc()
        assert pointer.create(memory_sink, doc) is True
        stored = memory_sink.objects[keys.state_pointer()]
        assert stored == pointer.canonical_bytes(doc)

    def test_race_loser_returns_false(self, memory_sink: MemorySink) -> None:
        seed_pointer(memory_sink, valid_doc())
        loser_doc = valid_doc()
        loser_doc["reason"] = "second writer"
        assert pointer.create(memory_sink, loser_doc) is False
        assert memory_sink.objects[keys.state_pointer()] == pointer.canonical_bytes(
            valid_doc()
        )

    def test_invalid_document_never_lands(self, memory_sink: MemorySink) -> None:
        doc = valid_doc()
        doc["tiers"].pop("cleaning")
        with pytest.raises(ValueError):
            pointer.create(memory_sink, doc)
        assert keys.state_pointer() not in memory_sink.objects


# --- 4. replace (promotion) --------------------------------------------------


def promoted_doc() -> dict:
    """The replacement candidate: cleaning v2 (``previous`` set by replace)."""
    doc = valid_doc()
    doc["tiers"]["cleaning"]["version"] = "v2"
    doc["reason"] = "promote: cleaning v2"
    return doc


class TestReplace:
    """Content-addressed CAS: chain recording, stale refusal, never-create."""

    def test_applies_and_records_previous(self, journal: JournalSink) -> None:
        seed_pointer(journal, valid_doc())
        current_hash = pointer.content_hash(valid_doc())
        applied = pointer.replace(journal, promoted_doc(), expected_hash=current_hash)
        assert applied["previous"] == current_hash
        assert journal.objects[keys.state_pointer()] == pointer.canonical_bytes(applied)
        assert applied["tiers"]["cleaning"]["version"] == "v2"

    def test_stale_expectation_conflicts(self, memory_sink: MemorySink) -> None:
        seed_pointer(memory_sink, valid_doc())
        with pytest.raises(pointer.PointerConflict, match="pointer moved"):
            pointer.replace(
                memory_sink,
                promoted_doc(),
                expected_hash=pointer.content_hash(
                    {**valid_doc(), "reason": "someone else promoted"}
                ),
            )

    def test_absent_pointer_never_creates(self, memory_sink: MemorySink) -> None:
        with pytest.raises(pointer.PointerConflict, match="never creates"):
            pointer.replace(
                memory_sink,
                promoted_doc(),
                expected_hash=pointer.content_hash(valid_doc()),
            )
        assert keys.state_pointer() not in memory_sink.objects

    def test_records_predecessor_even_when_caller_omits(
        self, memory_sink: MemorySink
    ) -> None:
        seed_pointer(memory_sink, valid_doc())
        orphan = promoted_doc()
        orphan["previous"] = None
        applied = pointer.replace(
            memory_sink,
            orphan,
            expected_hash=pointer.content_hash(valid_doc()),
        )
        assert applied["previous"] == pointer.content_hash(valid_doc())

    def test_lost_store_race_conflicts(self, memory_sink: MemorySink) -> None:
        """A write landing between the read and the put_if_match: conflict."""
        seed_pointer(memory_sink, valid_doc())
        current_hash = pointer.content_hash(valid_doc())
        # The interleaving: another promoter overwrites after our read.
        winner = valid_doc()
        winner["reason"] = "the concurrent promote"
        memory_sink.put(keys.state_pointer(), pointer.canonical_bytes(winner))
        with pytest.raises(pointer.PointerConflict):
            pointer.replace(memory_sink, promoted_doc(), expected_hash=current_hash)


# --- 5. read_pointer ---------------------------------------------------------


class TestReadPointer:
    """The single read path: decode + validate; malformed JSON is loud."""

    def test_round_trip(self, memory_sink: MemorySink) -> None:
        seed_pointer(memory_sink, valid_doc())
        assert pointer.read_pointer(memory_sink) == valid_doc()

    def test_absent_pointer_raises_file_not_found(
        self, memory_sink: MemorySink
    ) -> None:
        with pytest.raises(FileNotFoundError):
            pointer.read_pointer(memory_sink)

    def test_malformed_json_raises_value_error(self, memory_sink: MemorySink) -> None:
        memory_sink.objects[keys.state_pointer()] = b"{not json"
        with pytest.raises(ValueError, match="not valid JSON"):
            pointer.read_pointer(memory_sink)


# --- 6. Real-store IfMatch translation (fake client) -------------------------


class FakeBody:
    """``get_object``'s streaming body."""

    def __init__(self, payload: bytes) -> None:
        self._buffer = io.BytesIO(payload)

    def read(self) -> bytes:
        return self._buffer.read()


class FakeS3Client:
    """boto3 S3 client double for precondition + 404 translation tests."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.etag = '"etag-1"'
        self.fail_if_match = False

    def get_object(self, Bucket: str, Key: str) -> dict:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": FakeBody(self.objects[Key]), "ETag": self.etag}

    def put_object(self, Bucket: str, Key: str, Body: bytes, **kwargs: object) -> None:
        if "IfMatch" in kwargs and self.fail_if_match:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body


@pytest.fixture
def fake_store() -> ObjectStore:
    """A real ObjectStore shell over a fake S3 client (translation only)."""
    store = ObjectStore.__new__(ObjectStore)
    store._bucket = "bucket"
    store._client = FakeS3Client()
    return store


class TestRealStoreConditionalWrites:
    """The S3 error-code translation the pointer CAS leans on."""

    def test_put_if_match_true_and_false(self, fake_store: ObjectStore) -> None:
        fake_store._client.objects["state/production.json"] = b"{}"
        assert (
            fake_store.put_if_match(
                "state/production.json", b"x", expected_etag='"etag-1"'
            )
            is True
        )
        fake_store._client.fail_if_match = True
        assert (
            fake_store.put_if_match(
                "state/production.json", b"y", expected_etag='"etag-1"'
            )
            is False
        )

    def test_get_with_etag_gives_version_token(self, fake_store: ObjectStore) -> None:
        fake_store._client.objects["state/production.json"] = b"{}"
        data, etag = fake_store.get_with_etag("state/production.json")
        assert data == b"{}"
        assert etag == '"etag-1"'

    def test_get_with_etag_absent_is_file_not_found(
        self, fake_store: ObjectStore
    ) -> None:
        with pytest.raises(FileNotFoundError):
            fake_store.get_with_etag("state/production.json")
