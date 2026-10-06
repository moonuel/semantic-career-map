"""Contract tests for :mod:`backend.storage.ids` (identity minting and hashing).

``ids.py`` owns the two identity concerns of the store: the UUIDv7 posting id
(:func:`backend.storage.ids.new_id`) and the SHA-256 content hash
(:func:`backend.storage.ids.content_sha256`). These are pure and I/O-free, so
they are tested here without a store.

Scope: id and hash shape and behaviour only. Key-shape rules live in
``test_keys.py``.
"""

from __future__ import annotations

import sys
import uuid

import pytest

from backend.storage import ids

requires_py314 = pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="uuid.uuid7() is new in Python 3.14 (RFC 9562); new_id() needs it",
)


class TestContentHash:
    """``content_sha256`` mints a deterministic 64-char lowercase hex SHA-256."""

    def test_is_64_char_hex(self) -> None:
        assert ids._HASH_RE.fullmatch(ids.content_sha256(b"hello")) is not None

    def test_is_deterministic(self) -> None:
        assert ids.content_sha256(b"hello") == ids.content_sha256(b"hello")

    def test_differs_on_content(self) -> None:
        assert ids.content_sha256(b"a") != ids.content_sha256(b"b")

    def test_known_digest(self) -> None:
        # SHA-256 of "hello" is a fixed, well-known digest.
        assert ids.content_sha256(b"hello") == (
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )


@requires_py314
class TestNewId:
    """``new_id`` mints canonical, time-ordered UUIDv7 ids."""

    def test_is_parseable_uuid(self) -> None:
        parsed = uuid.UUID(ids.new_id())
        assert parsed.version == 7
        assert parsed.variant == uuid.RFC_4122

    def test_matches_id_regex(self) -> None:
        assert ids._ID_RE.fullmatch(ids.new_id()) is not None

    def test_is_canonical_lowercase_form(self) -> None:
        minted = ids.new_id()
        assert len(minted) == 36
        assert minted == minted.lower()
        assert minted == str(uuid.UUID(minted))

    def test_ids_are_monotonic_in_a_tight_loop(self) -> None:
        # UUIDv7 sorts lexicographically by generation time; the intra-ms
        # counter guarantees non-decreasing order within the same millisecond.
        ids_ = [ids.new_id() for _ in range(1000)]
        assert ids_ == sorted(ids_)

    def test_ids_are_unique(self) -> None:
        ids_ = {ids.new_id() for _ in range(1000)}
        assert len(ids_) == 1000
