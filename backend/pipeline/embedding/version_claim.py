"""Version-prefix claims: one configuration owns one embedding version.

The embedding writer addresses parts by ``model`` / ``variant`` / ``version``
(``keys.embedding_part``), and ``configuration_hash`` is a row column, never a
path segment. Nothing therefore stopped a second configuration from minting
parts into a prefix a first configuration already owned — which is exactly how
400 deploy-sample rows landed inside production ``version=v1`` (issue #22).

This module closes that hole at the write boundary. Before a run writes any
part, it claims its version prefix with a conditional put; the record names
the owning ``configuration_hash``. A second configuration aimed at the same
prefix fails loudly instead of silently joining the dataset.

Two token classes (``keys.version_kind``) differ only in reuse policy:

- **stable** (``vN``) — a production generation address. One configuration
  owns it, forever; a new configuration takes a new ``vN``.
- **ephemeral** (``x-...``) — a throwaway prefix. Still claimed, so two runs
  cannot interleave parts, but the same configuration may re-claim it after a
  failed or discarded attempt. A *different* configuration still cannot, so a
  throwaway name never means two datasets.

Claims are append-only like the rest of the lake: the registry object records
who created the prefix, not who last used it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from backend.pipeline.embedding.config import EmbeddingConfig
from backend.storage import keys


class ClaimConflict(RuntimeError):
    """Raised when a configuration tries to claim another's version prefix.

    Attributes:
        version: The contended version token.
        owner_hash: The ``configuration_hash`` recorded by the claim.
        incoming_hash: The ``configuration_hash`` of the rejected run.
    """

    def __init__(self, version: str, owner_hash: str, incoming_hash: str) -> None:
        self.version = version
        self.owner_hash = owner_hash
        self.incoming_hash = incoming_hash
        super().__init__(
            f"embedding version {version!r} is already claimed by configuration "
            f"{owner_hash[:12]}…; this run is {incoming_hash[:12]}…. "
            f"A different configuration must write to its own prefix: use a new "
            f"stable 'vN', or an ephemeral 'x-...' token for a throwaway run."
        )


def claim_record(
    config: EmbeddingConfig,
    configuration_hash: str,
    *,
    run_id: str,
    claimed_at: datetime,
    pipeline_version: str,
    note: str | None = None,
) -> dict[str, str]:
    """Build the version-claim document for this run.

    Args:
        config: The run's configuration (supplies model/variant/version).
        configuration_hash: The run's configuration digest.
        run_id: The run id that claimed the prefix.
        claimed_at: Claim wall-clock time (identity-neutral for the config).
        pipeline_version: Producing-code version.
        note: Optional operator note (e.g. ``"backfilled"`` for a claim
            written for a prefix that predates the registry).

    Returns:
        The claim document, JSON-ready.
    """
    document = {
        "dataset": "embeddings",
        "version": config.version,
        "kind": keys.version_kind(config.version),
        "model_segment": config.model_segment,
        "variant": config.variant,
        "configuration_hash": configuration_hash,
        "claimed_by_run_id": run_id,
        "claimed_at": claimed_at.isoformat(),
        "pipeline_version": pipeline_version,
    }
    if note is not None:
        document["note"] = note
    return document


def claim_bytes(record: dict[str, str]) -> bytes:
    """Serialize a claim document to canonical JSON bytes.

    Args:
        record: The claim document from :func:`claim_record`.

    Returns:
        Canonical JSON bytes (sorted keys, compact separators).
    """
    return json.dumps(
        record, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _owner(store, key: str) -> dict[str, str] | None:
    """Return the stored claim at ``key``, or ``None`` when absent."""
    try:
        return json.loads(store.get(key))
    except FileNotFoundError:
        return None


def claim_version(
    store,
    config: EmbeddingConfig,
    configuration_hash: str,
    *,
    run_id: str,
    claimed_at: datetime | None = None,
    pipeline_version: str = "",
    note: str | None = None,
) -> dict[str, str]:
    """Claim ``config.version`` for ``configuration_hash``, or refuse the run.

    Idempotent for the owning configuration: a repeat run against its own
    prefix re-reads the claim and proceeds without a write. The first writer
    is arbitrated by the store (``put_if_absent``), so a concurrent pair
    cannot both believe they created the prefix — the loser re-reads and is
    judged by the winner's record.

    Args:
        store: The object store holding the claim registry.
        config: The run's configuration.
        configuration_hash: The run's configuration digest.
        run_id: The run id claiming the prefix.
        claimed_at: Claim time; ``None`` uses now (UTC).
        pipeline_version: Producing-code version recorded in the claim.
        note: Optional operator note stored with the claim.

    Returns:
        The owning claim document (this run's, or the pre-existing one).

    Raises:
        ClaimConflict: if a different configuration already owns the prefix.
    """
    key = keys.embedding_version_claim(
        config.model_segment, config.variant, config.version
    )
    record = claim_record(
        config,
        configuration_hash,
        run_id=run_id,
        claimed_at=claimed_at or datetime.now(UTC),
        pipeline_version=pipeline_version,
        note=note,
    )

    existing = _owner(store, key)
    if existing is not None:
        return _accept_or_raise(existing, config.version, configuration_hash)

    if store.put_if_absent(key, claim_bytes(record)):
        return record

    # Lost the write race: the winner's record is now authoritative.
    winner = _owner(store, key)
    if winner is None:  # pragma: no cover - a vanished object is a store fault
        raise RuntimeError(f"version claim {key!r} disappeared during the race")
    return _accept_or_raise(winner, config.version, configuration_hash)


def _accept_or_raise(
    existing: dict[str, str], version: str, configuration_hash: str
) -> dict[str, str]:
    """Return ``existing`` when it names this configuration, else raise."""
    if existing.get("configuration_hash") == configuration_hash:
        return existing
    raise ClaimConflict(
        version, str(existing.get("configuration_hash", "")), configuration_hash
    )
