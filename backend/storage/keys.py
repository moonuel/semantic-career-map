"""Pure key builders and validators for the object store.

Keys implement the layout in ``docs/architecture/data-layout.md``. These
functions are the single source of truth for well-formed keys: they are pure
(no I/O) so they can be unit-tested without a store, and every other module
derives keys through them rather than by string interpolation.

This module implements the ``live`` and ``configs`` branches of the layout.
The ``eval`` branch and the deferred eval machinery are intentionally not
built yet; the shapes are reserved by the layout and added here when those
phases arrive.

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

# Regex for a well-formed path segment: lowercase, digits, hyphens, `_`
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
    """Return ``segment`` if it is a well-formed key segment, else raise."""
    if not segment:
        raise ValueError(f"{name} must be non-empty")
    if _SEGMENT_RE.fullmatch(segment) is None:
        raise ValueError(
            f"{name} {segment!r} is not a valid key segment "
            f"(lowercase letters, digits, hyphens, underscores only)"
        )
    return segment


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
    return f"live/raw/{_check(id, name='posting id')}/source.md"


def raw_metadata(id: str) -> str:
    """Key for the ingestion sidecar describing a posting.

    ``live/raw/<id>/metadata.json``
    """
    return f"live/raw/{_check(id, name='posting id')}/metadata.json"


def derived(id: str, variant: str, version: str = "v1") -> str:
    """Key for a derived artifact of a posting.

    ``live/derived/<id>/<variant>/v<version>.json``

    ``variant`` must be one of :data:`_DERIVED_VARIANTS`.
    """
    _check(id, name="posting id")
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
        f"{_check_version(version)}/{_check(id, name='posting id')}.npy"
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


def validate_key(key: str) -> str:
    """Validate a key and return it unchanged, or raise ``ValueError``.

    Enforces the structural rules of ``data-layout.md``:
      - first segment is one of the known buckets (``eval``/``live``/``configs``);
      - every segment matches the safe-character rule (no ``..``, no trailing
        slash, no uppercase, no reserved characters, no PII-like segments);
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

    for index, segment in enumerate(rest):
        if segment in {".", ".."}:
            raise ValueError(f"key {key!r} contains a path-normalization segment {segment!r}")
        is_last = index == len(rest) - 1
        pattern = _FILENAME_RE if is_last else _SEGMENT_RE
        if pattern.fullmatch(segment) is None:
            raise ValueError(
                f"key {key!r} contains malformed segment {segment!r} "
                f"(lowercase, digits, hyphens, underscores; dots only in the last segment)"
            )
        if _is_pii_like(segment):
            raise ValueError(f"key {key!r} contains a PII-like segment {segment!r}")

    return key


def _is_pii_like(segment: str) -> bool:
    """Heuristic guard against obvious PII in a key segment.

    Not a privacy control — keys should simply not *contain* PII. This flags
    email-shaped, phone-shaped, or SSN-shaped segments as an early error.
    """
    return bool(
        re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", segment)
        or re.fullmatch(r"\+?[0-9][0-9 .\-()]{7,}", segment)
        or re.fullmatch(r"\d{3}-\d{2}-\d{4}", segment)
    )
