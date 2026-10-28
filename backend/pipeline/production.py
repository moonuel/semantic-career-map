"""Production resolution: pointer → claim → ``configs/<hash>.json`` → config.

The feature's core read chain, in one implementation every consumer shares
(scheduled jobs, readers, the promote CLI, the gate): the production
pointer names partition segments per tier; an embeddings tier entry resolves
through its version-claim registry object (the claim name-points the owning
``configuration_hash``) to the content-addressed config object, and the
loaded :class:`EmbeddingConfig` brings the full generation parameters —
model, pinned revision, input lineage, verbatim selection SQL.

Pointers name; they never search. Accordingly every gap is a loud failure
that names the exact operator fix, never a silent fallback: a missing
pointer points at the bootstrap command, a missing claim at
``scripts/claim_embedding_version.py``, an adopted claim without its
``configs/<hash>.json`` object names the orphan outright (an adopted claim
is a declaration — its config must exist before anything consumes it).

The resolved claim hash **is** the production row filter: rows under the
pointer's pair stack whose ``configuration_hash`` equals it. For scheduled
embed runs this is inherent — the config is loaded by hash, so the run's
digest matches the claim and the anti-join is bound to the right
generation. Third-party readers (map visualization, eval) get the hash from
the resolution output and bind ``AND configuration_hash = :config``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from backend.pipeline.embedding.config import EmbeddingConfig, configuration_hash
from backend.storage import keys
from backend.storage.pointer import content_hash, parse_document, read_pointer
from backend.storage.store import ObjectStore

# The operator-facing fix commands (each failure names its own repair).
_BOOTSTRAP_COMMAND = "python scripts/promote_production.py --bootstrap --apply"


@dataclass(frozen=True)
class CleaningProduction:
    """The pointer's cleaning tier: the active artifact-partition stack.

    The cleaning tier has no claim registry yet — the pointer's naming is
    the authority (pipeline-code/config hashing is #17 territory, and an
    artifact claim registry is the planned follow-up). Segments are what
    the cleaning driver and the artifacts table need.
    """

    artifact_type: str
    variant: str
    version: str

    @property
    def stack(self) -> str:
        """The partition stack as one display string."""
        return f"{self.artifact_type}/{self.variant}/{self.version}"


@dataclass(frozen=True)
class EmbeddingProduction:
    """One resolved embeddings tier entry: segments + its owning config.

    Args:
        model_segment: The ``model`` partition segment (the row column
            carries the full model id).
        variant: The producing-method partition segment.
        version: The stable version token.
        configuration_hash: The claim owner — the production row-filter
            value (``configuration_hash = :config`` for every reader).
        config: The full loaded generation parameters (revision, input
            lineage, SQL text) — bound to the hash by construction: a
            scheduled run of this config re-hashes to it.
    """

    model_segment: str
    variant: str
    version: str
    configuration_hash: str
    config: EmbeddingConfig

    @property
    def stack(self) -> str:
        """The partition stack as one display string."""
        return f"{self.model_segment}/{self.variant}/{self.version}"


@dataclass(frozen=True)
class ProductionSet:
    """Everything scheduled production is, resolved once.

    Args:
        pointer: The resolved pointer document (verbatim, for provenance
            prints and gate cross-checks).
        cleaning: The cleaning tier's active stack.
        embeddings: The embeddings tier's active generations, in pointer
            order (the first is the scheduled one).
    """

    pointer: dict[str, Any]
    cleaning: CleaningProduction
    embeddings: list[EmbeddingProduction]


def verify_embedding_entry(
    store: ObjectStore, entry: dict[str, Any]
) -> EmbeddingProduction:
    """Resolve one pointer embeddings entry: claim → config → identity.

    Shared by :func:`resolve_production`, the promote CLI's
    build-then-flip refusal, and the pointer gate — the three places that
    ask "is this pointer entry a real, owned, loadable generation?".

    Args:
        store: The object store.
        entry: A pointer embeddings entry (``model_segment``/``variant``/
            ``version`` strings — a parse-validated tier document).

    Returns:
        The resolved :class:`EmbeddingProduction`.

    Raises:
        RuntimeError: naming the exact operator command for a missing
            claim, a missing config object (orphan adopted claim), or a
            config whose re-computed hash disagrees with the claim owner.
    """
    model_segment, variant, version = (
        entry["model_segment"],
        entry["variant"],
        entry["version"],
    )
    claim_key = keys.embedding_version_claim(model_segment, variant, version)
    try:
        claim = json.loads(store.get(claim_key))
    except FileNotFoundError:
        raise RuntimeError(
            f"embedding stack {model_segment}/{variant}/{version} has no claim "
            f"at {claim_key}; declare its owner with: "
            "python scripts/claim_embedding_version.py "
            f"--model {model_segment} --variant {variant} --version {version} "
            "--configuration-hash <64-char hex> --apply"
        ) from None
    owner = claim.get("configuration_hash")
    if not owner or not isinstance(owner, str):
        raise RuntimeError(f"claim {claim_key} names no configuration_hash")

    config_key = keys.config(owner)
    try:
        config_doc = json.loads(store.get(config_key))
    except FileNotFoundError:
        raise RuntimeError(
            f"orphan claim: {config_key} is absent for claim owner "
            f"{owner[:12]}… — an adopted claim's configuration must be "
            "written before anything resolves it"
        ) from None
    try:
        config = EmbeddingConfig(**config_doc["config"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"config object {config_key} does not carry a config dict: {exc!r}"
        ) from None
    if configuration_hash(config) != owner:
        raise RuntimeError(
            f"config at {config_key} hashes to {configuration_hash(config)[:12]}…, "
            f"not the claim owner {owner[:12]}… — the stored config drifted "
            "from its address"
        )
    return EmbeddingProduction(
        model_segment=model_segment,
        variant=variant,
        version=version,
        configuration_hash=owner,
        config=config,
    )


def resolve_production(
    store: ObjectStore, *, pointer: dict[str, Any] | None = None
) -> ProductionSet:
    """Resolve the active production set from the live pointer.

    Args:
        store: The object store (read-only for resolution).
        pointer: A pre-parsed pointer document, bypassing the read (tests
            holding a fixture document); ``None`` reads the live object.

    Returns:
        The :class:`ProductionSet` — tier stacks and, for the embeddings
        tier, each entry's owning configuration hash and loaded config.

    Raises:
        RuntimeError: naming the exact operator fix when the pointer is
            absent or unreadable, or when any tier resolution fails (the
            per-tier rules live in :func:`verify_embedding_entry`).
        ValueError: when the pointer document fails
            :func:`backend.storage.pointer.parse_document`.
    """
    if pointer is None:
        try:
            doc = read_pointer(store)
        except FileNotFoundError:
            raise RuntimeError(
                "no production pointer at state/production.json — scheduled "
                f"work refuses to guess; bootstrap with: {_BOOTSTRAP_COMMAND}"
            ) from None
    else:
        doc = parse_document(pointer)

    clean = doc["tiers"]["cleaning"]
    cleaning = CleaningProduction(
        artifact_type=clean["artifact_type"],
        variant=clean["variant"],
        version=clean["version"],
    )
    embeddings = [
        verify_embedding_entry(store, entry)
        for entry in doc["tiers"]["embeddings"]["entries"]
    ]
    return ProductionSet(pointer=doc, cleaning=cleaning, embeddings=embeddings)


def resolved_summary(production: ProductionSet) -> dict[str, Any]:
    """Render the ``production.resolved`` observability document.

    The one log point for *"which config did a scheduled run use, and
    why?"*: the pointer's provenance (who named this reality, when, and
    why), the tier stacks as named, and per embeddings entry the claim
    owner (the production row-filter value), the config object's path, and
    the loaded generation parameters. The embed entry path prints this
    before its run; consumers and dashboards read the same shape.

    Args:
        production: A resolved :class:`ProductionSet`.

    Returns:
        The JSON-ready summary dict.
    """
    return {
        "pointer": keys.state_pointer(),
        "pointer_hash": content_hash(production.pointer),
        "updated_by": production.pointer["updated_by"],
        "updated_at": production.pointer["updated_at"],
        "reason": production.pointer["reason"],
        "cleaning": production.cleaning.stack,
        "embeddings": [
            {
                "stack": emb.stack,
                "claim_owner": emb.configuration_hash,
                "config_path": keys.config(emb.configuration_hash),
                "model": emb.config.model_name,
                "model_revision": emb.config.model_revision,
                "input_artifact": emb.config.input_artifact,
                "input_version": emb.config.input_version,
            }
            for emb in production.embeddings
        ],
    }
