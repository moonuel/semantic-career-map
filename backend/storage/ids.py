"""Pure identity helpers for the object store.

``ids.py`` owns identity *minting and hashing*: the UUIDv7 posting id and the
SHA-256 content hash. It is deliberately separate from :mod:`backend.storage.keys`,
which owns key *shape*. Both are pure (no I/O) so they are unit-testable without
a store.

Posting ids are time-ordered **UUIDv7** values (RFC 9562 §5.7), rendered in the
canonical 36-char lowercase hyphenated form. Identity is *assigned at ingestion*
by :func:`new_id`, not derived from content. Because ids are ``[0-9a-f-]``,
every key segment is lowercase; there is no uppercase exception.

Content hashes are the lowercase hex SHA-256 of the verbatim received payload,
computed by :func:`content_sha256`. They are the content address for
content-addressed raw objects and the dedup identity for re-ingestion.

Public functions:
    - ``new_id()``         -> a fresh canonical UUIDv7 string (the only id minter).
    - ``content_sha256(data)`` -> lowercase hex SHA-256 of ``data``.

Public predicates (shape checks other modules build on):
    - ``is_canonical_uuid7(id)``   -> True if ``id`` is a canonical UUIDv7 string.
    - ``is_sha256_hex(hash_)``     -> True if ``hash_`` is 64-char lowercase hex.

Regexes:
    - ``_ID_RE``    matches a canonical UUIDv7 posting id.
    - ``_HASH_RE``  matches a 64-char lowercase hex SHA-256.
"""

from __future__ import annotations

import hashlib
import re
import uuid

# Posting id: a canonical UUIDv7 string (RFC 9562 §5.7) — 36-char lowercase
# hyphenated ``8-4-4-4-12``. The regex pins the version nibble (``7``) and the
# RFC 9562 variant (``[89ab]``), so it rejects UUIDv1/3/4/5, uppercase, braces,
# and URN form. Ids are minted only by ``new_id()``.
_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

# Hash is hex (SHA-256), fixed length 64.
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def new_id() -> str:
    """Mint a fresh posting id: a canonical UUIDv7 string (RFC 9562 §5.7).

    This is the **only** id generator. Builders validate ids but never mint
    them, so key construction stays pure and deterministic (important for
    tests). UUIDv7 is time-ordered, so lexicographic order of ids matches
    chronological ingestion order.

    Returns:
        A 36-char lowercase hyphenated UUIDv7, e.g.
        ``'0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d70'``.
    """
    return str(uuid.uuid7())


def content_sha256(data: bytes) -> str:
    """Return the lowercase hex SHA-256 of ``data``.

    The hash is computed from the verbatim posting bytes and is the sole
    content identity used for content-addressed raw objects and the
    re-ingestion dedup check.

    Args:
        data: Verbatim payload bytes.

    Returns:
        A 64-char lowercase hex SHA-256 digest of ``data``.
    """
    return hashlib.sha256(data).hexdigest()


def is_canonical_uuid7(id: str) -> bool:
    """Return True if ``id`` is a canonical UUIDv7 string, else False.

    The shape accepted is identical to :func:`new_id`'s output: 36-char
    lowercase hyphenated ``8-4-4-4-12`` with the version nibble ``7`` and the
    RFC 9562 variant (``[89ab]``). Builders and metadata schemas validate ids
    through this predicate rather than the private :data:`_ID_RE`.

    Args:
        id: Candidate posting-id string.

    Returns:
        True if ``id`` is a canonical UUIDv7.
    """
    return _ID_RE.fullmatch(id) is not None


def is_sha256_hex(hash_: str) -> bool:
    """Return True if ``hash_`` is a 64-char lowercase hex SHA-256, else False.

    The shape accepted is identical to :func:`content_sha256`'s output.
    Builders and metadata schemas validate hashes through this predicate
    rather than the private :data:`_HASH_RE`.

    Args:
        hash_: Candidate hash string.

    Returns:
        True if ``hash_`` is a 64-char lowercase hex string.
    """
    return _HASH_RE.fullmatch(hash_) is not None
