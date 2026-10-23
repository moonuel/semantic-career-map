"""Read-side helpers for the content-addressed raw lake layer.

The mirror of :mod:`backend.ingestion.raw`: that module writes payloads under
their content hash, this one lists and fetches them back. Every consumer of
the raw layer (the cleaning runner, the sampler) goes through here rather
than touching the store directly, so the
``lake/raw/source=<source>/sha256=<hash>.json`` layout is parsed in exactly
one place.

The reader is store-agnostic: it speaks the ``get``/``list_keys`` protocol
surface (:class:`RawStore`), which
:class:`backend.storage.store.ObjectStore` satisfies directly and tests
satisfy with an in-memory double. Payloads are parsed JSON dicts carried
alongside their ``source_hash`` — the hash is the posting's join handle
everywhere downstream (design D5, D11-rejected). Selection of *which* keys
to read is the query layer's job (D12); this module is the transform's I/O
path only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from backend.storage import keys

# The raw-layer prefix shared by every source partition.
RAW_PREFIX = "lake/raw/"

# Splits ``lake/raw/source=<source>/sha256=<hash>.json`` into (source, hash).
_RAW_KEY_RE = re.compile(
    r"^lake/raw/source=(?P<source>[a-z0-9-]+)/sha256=(?P<hash>[0-9a-f]{64})\.json$"
)


class RawStore(Protocol):
    """The get/list subset of a storage backend the reader needs.

    ``backend.storage.store.ObjectStore`` satisfies this directly; tests
    inject an in-memory double. Keys arrive pre-validated, and keys minted by
    :func:`keys.raw_payload` are the only ones ever passed in.
    """

    def get(self, key: str) -> bytes: ...

    def list_keys(self, prefix: str) -> Iterator[str]: ...


@dataclass(frozen=True)
class RawPayload:
    """One parsed raw-lake payload with its content-addressed identity.

    Attributes:
        source: Allowlisted source partition the payload lives under.
        source_hash: SHA-256 of the verbatim payload bytes — the posting's
            join handle in rows and manifests (no posting id exists).
        key: The object key the payload was read from.
        data: The parsed payload dict (the full posting record).
    """

    source: str
    source_hash: str
    key: str
    data: dict[str, Any]


def parse_raw_key(key: str) -> tuple[str, str]:
    """Split a raw-layer key into ``(source, source_hash)``.

    Args:
        key: A ``lake/raw/source=<source>/sha256=<hash>.json`` key.

    Returns:
        The source partition value and the 64-char hex content hash.

    Raises:
        ValueError: if ``key`` is not a well-formed raw-layer key.
    """
    match = _RAW_KEY_RE.fullmatch(key)
    if match is None:
        raise ValueError(
            f"key {key!r} is not a raw payload key "
            f"(lake/raw/source=<source>/sha256=<hash>.json)"
        )
    return match.group("source"), match.group("hash")


def list_raw_keys(store: RawStore, *, source: str | None = None) -> list[str]:
    """List raw-layer object keys, optionally within one source partition.

    Keys are returned in sorted order so every consumer iterates the layer in
    a stable order regardless of listing pagination.

    Args:
        store: The object store to list from.
        source: Restrict to one source partition; ``None`` lists all sources.

    Returns:
        Sorted raw-layer keys.
    """
    prefix = RAW_PREFIX + (f"source={source}/" if source else "")
    return sorted(store.list_keys(prefix))


def read_raw_payload(store: RawStore, key: str) -> RawPayload:
    """Fetch and parse one raw payload.

    Args:
        store: The object store to read from.
        key: The raw-layer object key.

    Returns:
        The parsed :class:`RawPayload`.

    Raises:
        FileNotFoundError: if the object is absent (404-class).
        ValueError: if the key is malformed or the payload is unparseable
            JSON.
        TypeError: if the payload parses to a non-object.
    """
    keys.validate_key(key)
    source, source_hash = parse_raw_key(key)
    data = store.get(key)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"payload at {key!r} is not parseable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"payload at {key!r} is not a JSON object")
    return RawPayload(source=source, source_hash=source_hash, key=key, data=payload)


def iter_raw_payloads(
    store: RawStore, *, source: str | None = None
) -> Iterator[RawPayload]:
    """Yield every raw payload (optionally one source) in stable key order.

    Args:
        store: The object store to read from.
        source: Restrict to one source partition; ``None`` iterates all
            sources.

    Yields:
        One parsed :class:`RawPayload` per object, sorted by key.

    Raises:
        ValueError: if any payload is unparseable — callers that must tolerate
            malformed objects catch and record the key instead.
    """
    for key in list_raw_keys(store, source=source):
        yield read_raw_payload(store, key)


def fetch_all(
    store: RawStore,
    source: str,
    *,
    object_keys: list[str] | None = None,
    workers: int = 16,
) -> list[RawPayload | None]:
    """Fetch a source's payloads with a bounded thread pool, keys in order.

    IO-bound object reads; the pool only bounds parallel GETs. Malformed
    payloads return ``None`` at their position instead of raising, so callers
    that must tolerate malformed objects can record or skip them.

    Args:
        store: The object store to read from.
        source: The partition the keys live under.
        object_keys: The keys to fetch; defaults to the source's full
            listing (sorted).
        workers: Concurrent fetch bound.

    Returns:
        RawPayloads (or ``None`` for malformed) in the same order as the
        keys.
    """
    if object_keys is None:
        object_keys = list_raw_keys(store, source=source)

    def read_or_none(key: str) -> RawPayload | None:
        try:
            return read_raw_payload(store, key)
        except ValueError, TypeError:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(read_or_none, object_keys))
