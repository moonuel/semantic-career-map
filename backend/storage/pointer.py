"""The production pointer: the bucket's one mutable control-plane document.

``state/production.json`` names the active production generation for every
scheduled tier as a map of partition segments. It is a *pointer*: it names
addresses; it never searches. Resolution reads it, then walks
pointer → claim → ``configs/<hash>.json`` → rows — so every other authority
over "what is production" (CLI defaults, task-def env, projection enums) is
either derived from this document or deleted.

Mutability is the point — this is the one object in the bucket that is
rewritten rather than added. Mutation runs through two primitives, both
content-addressed so the store arbitrates every race:

- :func:`create` — first bootstrap of a fresh environment
  (``put_if_absent``; a loser of the create race proceeds by reading the
  winner).
- :func:`replace` — every later promotion (``put_if_match``; a stale
  expectation or a lost replace race raises :class:`PointerConflict`;
  replace never creates).

The document is canonicalized before hashing and writing (:func:`content_hash`):
sorted keys, compact separators, non-ASCII escaped — two processes carrying
the same logical document must hash identically, and the hash is what
``previous`` chains (every replacement records the predecessor's content
hash) and what a promotion is predicated on.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from backend.storage import ids, keys
from backend.storage.store import ObjectStore

# The document schema's major version. A newer *major* means the reader no
# longer understands the document; refusing loudly beats misreading it.
POINTER_VERSION = 1

# The scheduled tier map. Exhaustive for this schema version: a missing or
# unknown tier is a refused document, not a skipped tier — a typo like
# ``embedings`` must never silently drop the tier from production.
KNOWN_TIERS: frozenset[str] = frozenset({"cleaning", "embeddings"})

# A pointer names production, and ephemeral tokens are never a production
# address — ``x-...`` prefixes are throwaway runs that nothing scheduled
# consumes. The tier maps admit stable ``vN`` only.
_cleaning_fields = ("artifact_type", "variant", "version")


class PointerConflict(RuntimeError):
    """Raised when a pointer mutation loses arbitration on the live content.

    Attributes:
        expected_hash: The content hash the caller's read was predicated on.
        current_hash: The document's live content hash at refusal time
            (``None`` when the pointer does not exist — replace never
            creates).
    """

    def __init__(self, expected_hash: str, current_hash: str | None) -> None:
        self.expected_hash = expected_hash
        self.current_hash = current_hash
        if current_hash is None:
            message = (
                "the production pointer does not exist; replace never creates "
                "— bootstrap it first: "
                "python scripts/promote_production.py --bootstrap --apply"
            )
        else:
            message = (
                f"pointer moved under the caller: expected {expected_hash[:12]}…, "
                f"current {current_hash[:12]}… — re-read and retry"
            )
        super().__init__(message)


def canonical_bytes(doc: dict[str, Any]) -> bytes:
    """Serialize a pointer document to canonical JSON bytes.

    Sorted keys, compact separators, non-ASCII escaped — the single byte
    form for hashing, writing, and identity comparisons.
    """
    return json.dumps(
        doc, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def content_hash(doc: dict[str, Any]) -> str:
    """Return the canonical 64-char hex SHA-256 digest of a pointer document."""
    return hashlib.sha256(canonical_bytes(doc)).hexdigest()


def parse_document(doc: Any) -> dict[str, Any]:
    """Validate a parsed pointer document; return it unchanged.

    Every tier segment must re-mint through a key builder — a raw address
    string is never trusted: the embeddings entries mint their claim key,
    the cleaning triple mints an artifact part key (with a throwaway run
    id) to prove the name resolves. A segment that any builder rejects
    (misspelled enum, ``x-...`` ephemeral token, ``.``-bearing junk) fails
    the whole document loudly.

    Args:
        doc: The JSON-decoded document (any shape).

    Returns:
        The same document, validated.

    Raises:
        ValueError: on any schema violation — wrong version major, missing
            or unknown tier branch, bad metadata fields, or a tier segment
            that fails its key-builder re-mint.
    """
    if not isinstance(doc, dict):
        # ValueError, not TypeError: a malformed pointer is a data fault the
        # CLI/resolution surfaces exactly like other parse failures.
        raise ValueError(  # noqa: TRY004 — parse failures are one class to callers
            f"pointer must be a JSON object, got {type(doc).__name__}"
        )
    if doc.get("pointer_version") != POINTER_VERSION:
        raise ValueError(
            f"pointer_version must be {POINTER_VERSION} "
            f"(this reader refuses other majors), got {doc.get('pointer_version')!r}"
        )
    _check_field(doc, "updated_by")
    _check_field(doc, "updated_at")
    _check_field(doc, "reason")
    previous = doc.get("previous")
    if previous is not None and not keys.is_sha256_hex(previous):
        raise ValueError(
            f"previous must be null or a 64-char hex content hash, got {previous!r}"
        )

    tiers = doc.get("tiers")
    if not isinstance(tiers, dict):
        raise ValueError(  # noqa: TRY004 — parse failures are one class to callers
            f"pointer tiers must be an object, got {type(tiers).__name__}"
        )
    presented = set(tiers)
    if presented != set(KNOWN_TIERS):
        raise ValueError(
            f"pointer must tier exactly {sorted(KNOWN_TIERS)}; "
            f"got {sorted(presented)} — a missing or unknown tier branch "
            "(a typo would silently drop the tier from production)"
        )
    _parse_cleaning(tiers["cleaning"])
    _parse_embeddings(tiers["embeddings"])
    return doc


def _check_field(doc: dict[str, Any], name: str) -> str:
    """Require one non-empty string field (provenance always names who/why)."""
    value = doc.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"pointer {name} must be a non-empty string, got {value!r}")
    return value


def _parse_cleaning(tier: Any) -> None:
    """Validate the cleaning tier triple by re-minting an artifact key."""
    if not isinstance(tier, dict) or set(tier) != set(_cleaning_fields):
        raise ValueError(
            f"cleaning tier must carry exactly {list(_cleaning_fields)}, "
            f"got {sorted(tier) if isinstance(tier, dict) else tier!r}"
        )
    artifact_type, variant, version = (tier[name] for name in _cleaning_fields)
    # The re-mint is the validation: any malformed or ephemeral segment
    # raises, so the pointer can never name an address no other module
    # would be able to build.
    keys.artifact_part(artifact_type, variant, version, run_id=ids.new_id(), seq=1)


def _parse_embeddings(tier: Any) -> None:
    """Validate the embeddings tier map by re-minting each entry's claim key."""
    if not isinstance(tier, dict) or set(tier) != {"entries"}:
        raise ValueError(
            "embeddings tier must carry exactly ['entries'], "
            f"got {sorted(tier) if isinstance(tier, dict) else tier!r}"
        )
    entries = tier["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("embeddings tier must be a non-empty list of entries")
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "model_segment",
            "variant",
            "version",
        }:
            raise ValueError(
                "embedding entry must carry exactly "
                "['model_segment', 'variant', 'version'], "
                f"got {sorted(entry) if isinstance(entry, dict) else entry!r}"
            )
        model_segment, variant, version = (
            entry["model_segment"],
            entry["variant"],
            entry["version"],
        )
        try:
            # The re-mint is the validation (the cleaning tier's trick): a
            # failed mint is reported against the exact entry, so the
            # operator fixes the document rather than guesses at a segment.
            keys.embedding_version_claim(model_segment, variant, version)
        except ValueError as exc:
            raise ValueError(
                f"embedding entry {entry!r} does not re-mint: {exc}"
            ) from None
        if keys.version_kind(version) != "stable":
            raise ValueError(
                f"embedding entry names version {version!r}: a pointer names "
                "production, and ephemeral x-... tokens are never production "
                "addresses"
            )


def read_pointer(store: ObjectStore) -> dict[str, Any]:
    """Read, decode, and validate the live pointer document.

    The one read path for every consumer (resolution, promote CLI, gate) so
    malformed-JSON handling cannot drift between them.

    Raises:
        FileNotFoundError: when the pointer does not exist (a fresh
            environment; the caller names the bootstrap fix).
        ValueError: when the object is not valid JSON or fails
            :func:`parse_document`.
    """
    key = keys.state_pointer()
    raw = store.get(key)
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"pointer at {key!r} is not valid JSON: {exc}") from None
    return parse_document(doc)


def create(store: ObjectStore, doc: dict[str, Any]) -> bool:
    """Bootstrap the pointer with a conditional first write.

    The document is validated before the write, so an unusable pointer can
    never land. The store arbitrates the fresh-environment race: two
    concurrent bootstrap writers cannot both land — the loser's call
    returns ``False`` and proceeds by reading the winner's document.

    Args:
        store: The object store.
        doc: The pointer document to create.

    Returns:
        ``True`` when this call created the pointer, ``False`` when one
        already existed.

    Raises:
        ValueError: when the document fails :func:`parse_document`.
    """
    parse_document(doc)
    return store.put_if_absent(keys.state_pointer(), canonical_bytes(doc))


def replace(
    store: ObjectStore, doc: dict[str, Any], *, expected_hash: str
) -> dict[str, Any]:
    """Content-addressed replace: apply ``doc`` only over ``expected_hash``.

    The promotion primitive. The live document's content hash is compared
    against ``expected_hash``; a mismatch means someone promoted first and
    the caller's expectation is stale (:class:`PointerConflict` — the
    store's ``If-Match`` arbitrates the physical race the same way, so a
    caller that read a hash a moment ago still cannot stomp a concurrent
    promote that landed between the check and the write). The replacement
    records its predecessor: replace sets ``previous`` to ``expected_hash``
    itself — the chain every operator can walk, enforced in one place.

    Args:
        store: The object store.
        doc: The replacement document (validated; ``previous`` is set here,
            so the caller need not carry it).
        expected_hash: The predecessor document's content hash.

    Returns:
        The applied document (validated).

    Raises:
        ValueError: when the document fails :func:`parse_document`, or its
            ``previous`` does not record ``expected_hash``.
        PointerConflict: when the live pointer's content hash is not
            ``expected_hash``, the pointer is absent (replace never
            creates), or the replace lost the store arbitration race.
    """
    key = keys.state_pointer()
    try:
        current_bytes, etag = store.get_with_etag(key)
    except FileNotFoundError as exc:
        raise PointerConflict(expected_hash, None) from exc
    try:
        current = json.loads(current_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"pointer at {key!r} is not valid JSON: {exc}") from None
    current_hash = content_hash(parse_document(current))
    if current_hash != expected_hash:
        raise PointerConflict(expected_hash, current_hash)

    doc = parse_document(doc)
    # The replacement records its predecessor — the chain discipline lives
    # here, not in the caller's memory.
    doc["previous"] = expected_hash
    if not store.put_if_match(key, canonical_bytes(doc), expected_etag=etag):
        raise PointerConflict(expected_hash, None)
    return doc
