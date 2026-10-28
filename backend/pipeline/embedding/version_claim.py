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


def _refuse_unclaimed_legacy(store, config: EmbeddingConfig) -> None:
    """Refuse a stable-token claim when the prefix already has parts.

    A claim-less prefix with parts on disk is a legacy prefix that predates
    the registry (issue #22) or a run that bypassed the driver. Minting a
    claim for it here would hand a production address to whichever
    configuration runs first; publishing the owner belongs to the operator
    via ``scripts/claim_embedding_version.py``. Ephemeral tokens are
    unaffected: they postdate the registry, so an unclaimed ephemeral prefix
    has no legacy meaning.

    Args:
        store: The object store.
        config: The run's configuration (supplies the prefix).

    Raises:
        RuntimeError: with the exact backfill command when the stable
            prefix already has parts but no claim.
    """
    if keys.version_kind(config.version) != "stable":
        return
    prefix = (
        f"lake/embeddings/model={config.model_segment}/"
        f"variant={config.variant}/version={config.version}/"
    )
    first_part = next(iter(store.list_keys(prefix)), None)
    if first_part is None:
        return
    raise RuntimeError(
        f"embedding version {config.version!r} already has parts but no "
        "ownership claim; refusing to mint. If this is a legacy prefix, "
        "declare its owner with: "
        "python scripts/claim_embedding_version.py "
        f"--model {config.model_segment} --variant {config.variant} "
        f"--version {config.version} --configuration-hash <64-char hex> --apply"
    )


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

    A stable (``vN``) prefix that already has parts but no claim is refused
    instead of adopted: first-writer-wins would silently mint an owner for
    the legacy ``version=v1`` (issue #22). Only
    :func:`adopt_version_prefix` — the operator backfill — may publish a
    claim for such a prefix.

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
        RuntimeError: if a stable prefix has pre-existing parts and no claim.
    """
    return _claim(
        store,
        config,
        configuration_hash,
        run_id=run_id,
        claimed_at=claimed_at,
        pipeline_version=pipeline_version,
        note=note,
        guard_legacy=True,
    )


def adopt_version_prefix(
    store,
    config: EmbeddingConfig,
    configuration_hash: str,
    *,
    run_id: str,
    claimed_at: datetime | None = None,
    pipeline_version: str = "",
    note: str | None = "backfilled",
) -> dict[str, str]:
    """Publish the declared owner of a prefix that already has parts.

    The operator's backfill path, and the only way to claim a *stable* prefix
    written before the registry existed. The owner is a declaration, not an
    inference: it is whatever ``configuration_hash`` the caller names, and it
    can never be changed afterwards (the claim object is write-once).

    Separate from :func:`claim_version` on purpose. A boolean flag on the
    enforcing path would let any future caller switch the legacy guard off;
    reaching this function has to be a deliberate import by the operator
    script.

    Args:
        store: The object store holding the claim registry.
        config: The configuration whose prefix is being adopted.
        configuration_hash: The declared owning configuration digest.
        run_id: The run id recording the backfill event.
        claimed_at: Claim time; ``None`` uses now (UTC).
        pipeline_version: Producing-code version recorded in the claim.
        note: Operator note stored with the claim.

    Returns:
        The owning claim document (this call's, or the pre-existing one).

    Raises:
        ClaimConflict: if a different configuration already owns the prefix.
    """
    return _claim(
        store,
        config,
        configuration_hash,
        run_id=run_id,
        claimed_at=claimed_at,
        pipeline_version=pipeline_version,
        note=note,
        guard_legacy=False,
    )


def _claim(
    store,
    config: EmbeddingConfig,
    configuration_hash: str,
    *,
    run_id: str,
    claimed_at: datetime | None,
    pipeline_version: str,
    note: str | None,
    guard_legacy: bool,
) -> dict[str, str]:
    """Shared claim protocol; ``guard_legacy`` is set only by the public entry."""
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

    if guard_legacy:
        _refuse_unclaimed_legacy(store, config)

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
