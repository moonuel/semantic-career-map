"""Pure key builders and validators for the object store.

Keys implement the layout in ``docs/architecture/data.md``. These functions are
the single source of truth for well-formed keys: they are pure (no I/O) so they
can be unit-tested without a store, and every other module derives keys through
them rather than by string interpolation.

This module implements the legacy ``live`` and ``configs`` branches plus the
lakehouse ``lake`` branch. The lakehouse is being migrated branch-by-branch,
key builders first; the ``raw`` and ``manifests/ingestion`` builders have
landed, while ``artifacts``, ``embeddings``, ``eval``, and the remaining
``manifests`` (datasets, runs) are reserved by the layout and added here when
those phases arrive.

Lakehouse ``raw`` objects are **content-addressed**: identity is the SHA-256 of
the verbatim payload, not a posting id. ``raw_payload`` emits
``lake/raw/source=<source>/sha256=<hash>.json``, pruned by the low-cardinality
``source`` partition. ``source`` is validated against an allowlist
(:data:`_RAW_SOURCES`) to keep partitioning governed. There is deliberately no
time partition: identical re-ingested payloads map to the same key and
overwrite idempotently, so each payload is stored once. Ingestion time is not
encoded in the key — objects are never rewritten, so S3 ``Last-Modified`` is
the first-ingestion date, and provenance dates for date-range queries belong
in batch manifests, not here.

Legacy posting ids are time-ordered **UUIDv7** values (RFC 9562 §5.7), rendered
in the canonical 36-char lowercase hyphenated form. Identity minting and
content hashing live in :mod:`backend.storage.ids`; this module only validates
and places id/hash tokens in keys. Because ids are ``[0-9a-f-]``, **every**
segment in a key is lowercase; there is no uppercase exception.

Public builders:
    - ``raw_payload(source, content_hash)`` -> ``lake/raw/source=<source>/sha256=<hash>.json``
    - ``manifest_ingestion(ingested_on, ingestion_id)`` ->
      ``lake/manifests/ingestion/ingested_on=<date>/ingestion_id=<uuid7>.parquet``
    - ``raw_posting(id)``        -> ``live/raw/<id>/source.md``
    - ``raw_metadata(id)``       -> ``live/raw/<id>/metadata.json``
    - ``derived(id, variant, v)``-> ``live/derived/<id>/<variant>/v<v>.json``
    - ``embedding(model, variant, v, id)`` -> ``live/embeddings/<model>/<variant>/v<v>/<id>.npy``
    - ``config(hash)``           -> ``configs/<hash>.json``

Validators:
    - ``validate_key(key)`` raises ``ValueError`` on a malformed key.
"""

from __future__ import annotations

import datetime
import re

from backend.storage.ids import is_canonical_uuid7, is_sha256_hex

# --- Constants ---------------------------------------------------------------

# The single bucket's top-level branches. ``lake`` holds the lakehouse raw
# layer (content-addressed, immutable); ``live`` and ``configs`` are the legacy
# object-per-posting store still in use until each branch migrates.
_BUCKETS: frozenset[str] = frozenset({"eval", "live", "configs", "lake"})

# Sources admitted into ``lake/raw/source=<source>/``. Kept as an explicit
# allowlist so partitioning stays governed — a source is a partition column,
# and an unbounded source set is the partition-explosion / ungoverned-accumulation
# anti-pattern. Sources are lowercase and must satisfy _SEGMENT_RE.
_RAW_SOURCES: frozenset[str] = frozenset({"linkedin", "lever", "greenhouse", "ashby"})

# Variants that live under live/derived/<id>/<variant>/.
# Kept open: new variants are added here and are additive (layout §5).
_DERIVED_VARIANTS: frozenset[str] = frozenset(
    {"sections", "clean-text", "llm-clean-text", "llm-role-context", "labels"}
)

# Regex for a well-formed non-id path segment: lowercase, digits, hyphens, `_`
# (for `config-hash`), and `=` for `key=value` partition pairs
# (e.g. `day=2026-09-23`, Report 005 §3.2). No leading/trailing hyphen.
_SEGMENT_RE = re.compile(r"^[a-z0-9]+(?:[-_=][a-z0-9]+)*$")

# Regex for a well-formed terminal (filename) segment: a bare name or a
# `name.ext`, where the name follows the segment rules and the extension is
# lowercase alphanumeric. Dots are only allowed here, at the leaf.
_FILENAME_RE = re.compile(r"^[a-z0-9]+(?:[-_=][a-z0-9]+)*(?:\.[a-z0-9]+)?$")

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
            f"(lowercase letters, digits, '=', hyphens, underscores only)"
        )
    return segment


def _check_id(id: str, *, name: str = "posting id") -> str:
    """Return ``id`` if it is a well-formed posting id, else raise.

    A posting id is a canonical UUIDv7 in 36-char lowercase hyphenated form
    (``8-4-4-4-12``). Unlike every other segment, hyphens sit at fixed
    positions, so this dedicated shape check is used rather than
    :data:`_SEGMENT_RE`.
    """
    if not is_canonical_uuid7(id):
        raise ValueError(
            f"{name} {id!r} must be a canonical UUIDv7 (8-4-4-4-12), "
            f"e.g. '0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d70'"
        )
    return id


def _check_version(version: str, *, name: str = "version") -> str:
    if _VERSION_RE.fullmatch(version) is None:
        raise ValueError(f"{name} must be 'vN' (v1, v2, ...), got {version!r}")
    return version


def _check_hash(hash_: str, *, name: str = "config hash") -> str:
    if not is_sha256_hex(hash_):
        raise ValueError(f"{name} must be a 64-char hex string, got {hash_!r}")
    return hash_


def _check_date(value: str, *, name: str) -> str:
    """Return ``value`` if a canonical ``YYYY-MM-DD`` calendar date, else raise.

    The string form must be zero-padded and lexically sortable (a Hive
    partition value doubles as a range-query filter), and must name a real
    calendar day. ``datetime.date.fromisoformat`` would accept non-canonical
    forms, so the strict shape is pinned by regex first.
    """
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError(f"{name} {value!r} must be a canonical YYYY-MM-DD date")
    try:
        datetime.date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name} {value!r} is not a calendar date") from None
    return value


# --- Builders ---------------------------------------------------------------


def raw_payload(source: str, content_hash: str) -> str:
    """Key for a verbatim, content-addressed raw payload.

    ``lake/raw/source=<source>/sha256=<content-hash>.json``

    The raw object is addressed by the SHA-256 of the received payload, not by
    a posting id (layout §2.1). ``source`` is a low-cardinality Hive partition
    pruned by Athena. The key carries no time partition: identical re-ingested
    payloads produce the same key, so re-runs overwrite idempotently and each
    payload is stored once. Ingestion time is observable as the object's S3
    ``Last-Modified`` (objects are never rewritten) and lands in batch
    manifests for Athena range queries; it is not encoded here.

    Args:
        source: Origin of the payload; must be in :data:`_RAW_SOURCES`.
        content_hash: 64-char lowercase hex SHA-256 of the verbatim bytes — the
            content address.

    Returns:
        The well-formed raw object key.

    Raises:
        ValueError: if ``source`` is not allowlisted or ``content_hash`` is
            not 64-char hex.
    """
    # Allowlist check first so an unknown source gets the useful "expected one
    # of ..." message, not the generic segment error.
    if source not in _RAW_SOURCES:
        raise ValueError(
            f"unknown raw source {source!r}; expected one of {sorted(_RAW_SOURCES)}"
        )
    source = _check(source, name="source")
    return (
        f"lake/raw/source={source}/"
        f"sha256={_check_hash(content_hash, name='content hash')}.json"
    )


def manifest_ingestion(ingested_on: str, ingestion_id: str) -> str:
    """Key for a batch-scrape ingestion manifest (one Parquet file per run).

    ``lake/manifests/ingestion/ingested_on=<ingested-on>/ingestion_id=<id>.parquet``

    The ingestion event is an append-only Parquet row batch keyed by the run's
    ``ingestion_id`` (layout §2.1, §3): one object per batched scrape — not one
    per event — keeps the atomic-file count at scale down while still recording
    exactly what each run observed and stored. The ``ingested_on`` day partition
    lives *only* here (raw keys deliberately carry no date, §2.1); manifests are
    where batch provenance time is pruned for Athena range queries. Manifests
    are never rewritten, so a re-deduped re-run under the same id is undefined —
    the id is minted once per run.

    Args:
        ingested_on: Canonical ``YYYY-MM-DD`` run date.
        ingestion_id: Canonical UUIDv7 minted once for the batch run.

    Returns:
        The well-formed ingestion-manifest object key.

    Raises:
        ValueError: if the date is not canonical ``YYYY-MM-DD`` or the id is
            not a canonical UUIDv7.
    """
    return (
        f"lake/manifests/ingestion/ingested_on={_check_date(ingested_on, name='ingested_on')}/"
        f"ingestion_id={_check_id(ingestion_id, name='ingestion id')}.parquet"
    )


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


# --- Validators -------------------------------------------------------------


def _id_index(bucket: str, rest: list[str]) -> int | None:
    """Index in ``rest`` of the posting-id segment for this key shape, else None.

    The id is identified *positionally* by the branch, not by its characters:

    - ``live/raw/<id>/<file>``                 -> index 1
    - ``live/derived/<id>/<variant>/<version-file>`` -> index 1
    - ``live/embeddings/<model>/<variant>/<vN>/<id>.<ext>`` -> leaf (index ``-1``)
    - ``lake/raw/...`` (content-addressed)     -> no id segment
    - ``meta/...`` and ``configs/...``         -> no id segment
    """
    if not rest:
        return None
    if bucket == "live" and rest[0] in {"raw", "derived"}:
        return 1 if len(rest) > 1 else None
    if bucket == "live" and rest[0] == "embeddings":
        return len(rest) - 1
    return None


def _validate_lake_manifest(key: str, rest: list[str]) -> None:
    """Semantic checks for the ``lake/manifests/ingestion`` key shape.

    ``lake/manifests/ingestion/ingested_on=<date>/ingestion_id=<uuid>.parquet``

    Pins the branch, the single ``ingested_on`` day partition, and the
    ``ingestion_id=<uuid7>.parquet`` leaf for the same reason
    :func:`_validate_lake_raw` pins the raw shape. Other ``manifests``
    sub-branches (``datasets``, ``runs``) stay reserved and unpinned until
    their builders land.
    """
    if len(rest) != 4 or rest[1] != "ingestion":
        raise ValueError(
            f"key {key!r} must be lake/manifests/ingestion/"
            f"ingested_on=<date>/ingestion_id=<id>.parquet"
        )
    partition = rest[2].partition("=")
    if partition[0] != "ingested_on":
        raise ValueError(
            f"key {key!r} must partition exactly on ingested_on, got {rest[2]!r}"
        )
    _check_date(partition[2], name="ingested_on")
    leaf_name, _, leaf_value = rest[-1].partition("=")
    if leaf_name != "ingestion_id" or not leaf_value.endswith(".parquet"):
        raise ValueError(
            f"key {key!r} must end in 'ingestion_id=<id>.parquet', got {rest[-1]!r}"
        )
    _check_id(leaf_value.removesuffix(".parquet"), name="ingestion id")


def _validate_lake_raw(key: str, rest: list[str]) -> None:
    """Semantic checks for the content-addressed ``lake/raw`` key shape.

    ``lake/raw/source=<source>/sha256=<hash>.json``

    The safe-character loop in :func:`validate_key` only pins the segment
    alphabet; this pins the branch shape itself — the single ``source``
    partition and the hash leaf — because ``validate_key`` also gates
    externally supplied keys that :func:`raw_payload` did not mint. Other
    ``lake/`` branches get the same treatment when their builders land (the
    ingestion-manifest branch is pinned in :func:`_validate_lake_manifest`).
    """
    if len(rest) != 3:
        raise ValueError(
            f"key {key!r} must be lake/raw/source=<source>/sha256=<hash>.json"
        )
    leaf = rest[-1]
    leaf_name, _, leaf_value = leaf.partition("=")
    if leaf_name != "sha256" or not leaf_value.endswith(".json"):
        raise ValueError(
            f"key {key!r} must end in 'sha256=<content-hash>.json', got {leaf!r}"
        )
    _check_hash(leaf_value.removesuffix(".json"), name="content hash")
    partitions = [segment.partition("=") for segment in rest[1:-1]]
    names = [name for name, _, _ in partitions]
    if names != ["source"]:
        raise ValueError(f"key {key!r} must partition exactly on source, got {names}")
    for name, _, value in partitions:
        if value not in _RAW_SOURCES:
            raise ValueError(
                f"key {key!r} has unknown raw source {value!r}; "
                f"expected one of {sorted(_RAW_SOURCES)}"
            )


def validate_key(key: str) -> str:
    """Validate a key and return it unchanged, or raise ``ValueError``.

    Enforces the structural rules of ``data.md``:
      - first segment is one of the known branches (``eval``/``live``/``configs``/``lake``);
      - every segment matches the safe-character rule (no ``..``, no trailing
        slash, no reserved characters);
      - the posting-id segment — where the layout places one — is a canonical
        UUIDv7 (36-char lowercase hyphenated), while every other segment stays
        lowercase;
      - ``lake/raw`` keys additionally pin the whole content-addressed shape
        (allowlisted ``source`` partition, ``sha256=<hash>.json`` leaf), and
        ``lake/manifests/ingestion`` pins the batch-manifest shape (single
        ``ingested_on`` day partition, ``ingestion_id=<uuid7>.parquet`` leaf);
      - no trailing slash; no leading slash.

    Reserved ``lake`` branches (``artifacts``/``embeddings``/``eval``, and the
    ``manifests`` sub-branches other than ``ingestion``) are not yet pinned to
    a shape: they pass the general rules above but their positional layouts are
    only enforced once their builders land.
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

    id_index = _id_index(bucket, rest)
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
                f"(lowercase letters, digits, '=', hyphens, underscores; dots "
                f"only in the last segment)"
            )

    if bucket == "lake" and rest[:1] == ["raw"]:
        _validate_lake_raw(key, rest)

    if bucket == "lake" and rest[:2] == ["manifests", "ingestion"]:
        _validate_lake_manifest(key, rest)

    return key
