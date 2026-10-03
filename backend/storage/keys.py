"""Pure key builders and validators for the object store.

Keys implement the layout in ``docs/architecture/data-layout.md``. These
functions are the single source of truth for well-formed keys: they are pure
(no I/O) so they can be unit-tested without a store, and every other module
derives keys through them rather than by string interpolation.

This module implements the ``live`` and ``configs`` branches of the layout.
The ``eval`` branch and the deferred eval machinery are intentionally not
built yet; the shapes are reserved by the layout and added here when those
phases arrive.

Posting ids are the first 16 characters of the unpadded base64url encoding of
``sha256(source bytes)``. base64url (RFC 4648 §5) is **case-sensitive**, so an
id segment may contain uppercase letters; every other segment stays lowercase.

Public builders:
    - ``raw_posting(id)``        -> ``live/raw/<id>/source.md``
    - ``raw_metadata(id)``       -> ``live/raw/<id>/metadata.json``
    - ``derived(id, variant, v)``-> ``live/derived/<id>/<variant>/v<v>.json``
    - ``embedding(model, variant, v, id)`` -> ``live/embeddings/<model>/<variant>/v<v>/<id>.npy``
    - ``config(hash)``           -> ``configs/<hash>.json``
    - ``manifest(date)``         -> ``live/meta/datasets/day=<date>/manifest.json``

Validators:
    - ``validate_key(key)`` raises ``ValueError`` on a malformed key.
"""

from __future__ import annotations

import datetime as _dt
import re

# --- Constants ---------------------------------------------------------------

_BUCKETS: frozenset[str] = frozenset({"eval", "live", "configs"})

# Variants that live under live/derived/<id>/<variant>/.
# Kept open: new variants are added here and are additive (layout §5).
_DERIVED_VARIANTS: frozenset[str] = frozenset(
    {"sections", "clean-text", "llm-clean-text", "llm-role-context", "labels"}
)

# Posting id: exactly 16 chars of base64url (RFC 4648 §5), unpadded.
# ``[A-Za-z0-9_-]`` — case-sensitive by construction.
_ID_LENGTH: int = 16
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16}$")

# Regex for a well-formed non-id path segment: lowercase, digits, hyphens, `_`
# (for `config-hash`), and `=` for `key=value` partition pairs
# (e.g. `day=2026-09-23`, Report 005 §3.2). No leading/trailing hyphen.
_SEGMENT_RE = re.compile(r"^[a-z0-9]+(?:[-_=][a-z0-9]+)*$")

# Regex for a well-formed terminal (filename) segment: a bare name or a
# `name.ext`, where the name follows the segment rules and the extension is
# lowercase alphanumeric. Dots are only allowed here, at the leaf.
_FILENAME_RE = re.compile(r"^[a-z0-9]+(?:[-_=][a-z0-9]+)*(?:\.[a-z0-9]+)?$")

# Hash is hex (SHA-256), fixed length 64.
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

# Version token: `v` followed by a positive integer.
_VERSION_RE = re.compile(r"^v[1-9][0-9]*$")


# --- Helpers ----------------------------------------------------------------


def _check(segment: str, *, name: str) -> str:
    """Return ``segment`` if it is a well-formed non-id key segment, else raise."""
    if not segment:
        raise ValueError(f"{name} must be non-empty")
    if _SEGMENT_RE.fullmatch(segment) is None:
        raise ValueError(
            f"{name} {segment!r} is not a valid key segment "
            f"(lowercase letters, digits, hyphens, underscores only)"
        )
    return segment


def _check_id(id: str, *, name: str = "posting id") -> str:
    """Return ``id`` if it is a well-formed posting id, else raise.

    A posting id is exactly :data:`_ID_LENGTH` chars of base64url. base64url is
    case-sensitive, so uppercase is legal **here only** (unlike every other
    segment, which stays lowercase).
    """
    if _ID_RE.fullmatch(id) is None:
        raise ValueError(
            f"{name} {id!r} must be exactly {_ID_LENGTH} chars of base64url "
            f"([A-Za-z0-9_-])"
        )
    return id


def _check_version(version: str, *, name: str = "version") -> str:
    if _VERSION_RE.fullmatch(version) is None:
        raise ValueError(f"{name} must be 'vN' (v1, v2, ...), got {version!r}")
    return version


def _check_hash(hash_: str, *, name: str = "config hash") -> str:
    if _HASH_RE.fullmatch(hash_) is None:
        raise ValueError(f"{name} must be a 64-char hex string, got {hash_!r}")
    return hash_


def _check_date(date: _dt.date) -> str:
    """Return ``date`` as ``YYYY-MM-DD`` for a partition segment."""
    return date.isoformat()


# --- Builders ---------------------------------------------------------------


def raw_posting(id: str) -> str:
    """Key for the verbatim source markdown of a posting.

    ``live/raw/<id>/source.md``
    """
    return f"live/raw/{_check_id(id)}/source.md"


def raw_metadata(id: str) -> str:
    """Key for the ingestion sidecar describing a posting.

    ``live/raw/<id>/metadata.json``
    """
    return f"live/raw/{_check_id(id)}/metadata.json"


def derived(id: str, variant: str, version: str = "v1") -> str:
    """Key for a derived artifact of a posting.

    ``live/derived/<id>/<variant>/v<version>.json``

    ``variant`` must be one of :data:`_DERIVED_VARIANTS`.
    """
    _check_id(id)
    if variant not in _DERIVED_VARIANTS:
        raise ValueError(
            f"unknown derived variant {variant!r}; expected one of "
            f"{sorted(_DERIVED_VARIANTS)}"
        )
    return (
        f"live/derived/{id}/"
        f"{_check(variant, name='variant')}/{_check_version(version)}.json"
    )


def embedding(model: str, variant: str, version: str, id: str) -> str:
    """Key for a posting's embedding vector.

    ``live/embeddings/<model>/<variant>/v<version>/<id>.npy``

    The model leads because embeddings are queried model-first: a vector is a
    row in a model's corpus space, so ``id`` sits at the leaf as a join handle
    back to the source (layout §2.1).
    """
    return (
        f"live/embeddings/{_check(model, name='model')}/"
        f"{_check(variant, name='variant')}/"
        f"{_check_version(version)}/{_check_id(id)}.npy"
    )


def config(hash_: str) -> str:
    """Key for a config manifest entry.

    ``configs/<hash>.json``

    ``hash_`` is the content hash of the config object; the manifest turns a
    version token into reproducible generation parameters (layout §2.3).
    """
    return f"configs/{_check_hash(hash_)}.json"


def manifest(date: _dt.date) -> str:
    """Key for the raw-ingestion manifest on a given day.

    ``live/meta/datasets/day=<date>/manifest.json``
    """
    return f"live/meta/datasets/day={_check_date(date)}/manifest.json"


# --- Validators -------------------------------------------------------------


def _id_index(rest: list[str]) -> int | None:
    """Index in ``rest`` of the posting-id segment for this key shape, else None.

    The id is identified *positionally* by the branch, not by its characters:

    - ``raw/<id>/<file>``                        -> index 1
    - ``derived/<id>/<variant>/<version-file>``  -> index 1
    - ``embeddings/<model>/<variant>/<vN>/<id>.<ext>`` -> leaf (index ``-1``)
    - ``meta/...`` and ``configs/...``           -> no id segment
    """
    if not rest:
        return None
    if rest[0] in {"raw", "derived"}:
        return 1 if len(rest) > 1 else None
    if rest[0] == "embeddings":
        return len(rest) - 1
    return None


def validate_key(key: str) -> str:
    """Validate a key and return it unchanged, or raise ``ValueError``.

    Enforces the structural rules of ``data-layout.md``:
      - first segment is one of the known buckets (``eval``/``live``/``configs``);
      - every segment matches the safe-character rule (no ``..``, no trailing
        slash, no reserved characters);
      - the posting-id segment — where the layout places one — is exactly
        :data:`_ID_LENGTH` chars of base64url and may contain uppercase, while
        every other segment stays lowercase;
      - no trailing slash; no leading slash.
    """
    if not key or key.startswith("/"):
        raise ValueError(f"key must not be empty or start with '/': {key!r}")
    if key.endswith("/"):
        raise ValueError(f"key must not end with '/': {key!r}")
    if "//" in key:
        raise ValueError(f"key must not contain empty segments: {key!r}")

    segments = key.split("/")
    bucket, rest = segments[0], segments[1:]

    if bucket not in _BUCKETS:
        raise ValueError(
            f"key {key!r} starts with unknown bucket {bucket!r}; "
            f"expected one of {sorted(_BUCKETS)}"
        )

    for segment in rest:
        if segment in {".", ".."}:
            raise ValueError(
                f"key {key!r} contains a path-normalization segment {segment!r}"
            )

    id_index = _id_index(rest)
    for index, segment in enumerate(rest):
        if index == id_index:
            # Embeddings place the id at the leaf as ``<id>.npy``; the raw and
            # derived branches place it as a bare segment.
            id_segment = segment
            if rest and rest[0] == "embeddings":
                id_segment, matched, extension = segment.partition(".")
                if not matched or extension != "npy":
                    raise ValueError(
                        f"key {key!r} must end in '<id>.npy', got {segment!r}"
                    )
            _check_id(id_segment)
            continue
        is_last = index == len(rest) - 1
        pattern = _FILENAME_RE if is_last else _SEGMENT_RE
        if pattern.fullmatch(segment) is None:
            raise ValueError(
                f"key {key!r} contains malformed segment {segment!r} "
                f"(lowercase letters, digits, hyphens, underscores; dots only "
                f"in the last segment)"
            )

    return key
