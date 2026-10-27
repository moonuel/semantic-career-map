"""Embedding configuration: the hashed generation parameters (data.md §5).

The configuration object carries every parameter that determines what an
embedding dataset version is: which text source (model + pinned revision +
text variant + batch/truncation settings), which run produced it (the input
SQL text, input artifact and version, output variant/version
partitioning), and the normalization contract. Its 64-char SHA-256
``configuration_hash`` is the dataset's identity reference — rows, parts,
and manifests carry the hash instead of the parameters, and the verbatim
config object (SQL text included) is stored content-addressed at
``configs/<hash>.json`` via :func:`backend.storage.keys.config`.

Canonicalization contract: hashing serializes the dataclass fields in their
declared order (no sorting, no timestamps, no runtime-affine values — the
Athena database/workgroup and encoder device are execution context, not
dataset identity), with a fixed JSON form (separators, ensure_ascii). The
same values in two processes must hash identically; any single field change
must change the hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

# Model families are partitioned model-first (data.md §7); the partition
# segment is derived from the model name's last path component, lowercased —
# "sentence-transformers/all-MiniLM-L6-v2" -> "all-minilm-l6-v2".
DEFAULT_INPUT_ARTIFACT = "athena:semantic_career_map/artifacts_cleaned"
DEFAULT_INPUT_VERSION = "v1"


@dataclass(frozen=True)
class EmbeddingConfig:
    """One embedding dataset generation's parameters (all hashed together).

    Attributes:
        model_name: Full sentence-transformers model id.
        model_revision: Pinned HuggingFace revision; the loader asserts the
            resolved revision equals it (no ambient resolution in
            provenance).
        model_segment: Lowercased, slash-free model partition segment.
        variant: The output dataset's ``variant`` partition (producing
            method naming, open).
        text_variant: The artifact column holding the text to embed.
        fallback_text_variant: Column tried when :attr:`text_variant` is
            missing/empty for a row; ``None`` disables fallback.
        batch_size: Encode batch window (rows per part commit in the
            Layer-2 writer).
        max_length: Tokenizer truncation length (pins model.max_seq_length).
        normalize: L2-normalization contract.
        input_artifact: Input lineage string (recorded in rows/manifests).
        input_version: The pinned input artifact generation.
        version: The output dataset's ``version`` partition token.
        sql_text: The verbatim selection SQL (parameterized ``:config``-style
            placeholders), stored with the config in ``configs/``.
    """

    model_name: str
    model_revision: str
    model_segment: str
    variant: str
    text_variant: str
    fallback_text_variant: str | None = None
    batch_size: int = 16
    max_length: int = 256
    normalize: bool = True
    input_artifact: str = DEFAULT_INPUT_ARTIFACT
    input_version: str = DEFAULT_INPUT_VERSION
    version: str = "v1"
    sql_text: str = ""


def configuration_hash(config: EmbeddingConfig) -> str:
    """Return the config's canonical 64-char hex SHA-256 digest.

    Args:
        config: The embedding configuration.

    Returns:
        The 64-char lowercase hex configuration hash.
    """
    payload = json.dumps(
        asdict(config),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_document(config: EmbeddingConfig) -> bytes:
    """Return the content-addressed config-object bytes for ``configs/``.

    The stored object carries the hash beside the exact parameter dict, so a
    manifest reference resolves to full generation parameters (SQL text
    included) with one store read.

    Args:
        config: The embedding configuration.

    Returns:
        Canonical JSON bytes.
    """
    document = {
        "configuration_hash": configuration_hash(config),
        "config": asdict(config),
    }
    return json.dumps(
        document, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def default_model_segment(model_name: str) -> str:
    """Derive the model partition segment from a full model id.

    Args:
        model_name: Full HF-style model id (``org/name`` or ``name``).

    Returns:
        The lowercased last path component (the partition-safe segment).

    Raises:
        ValueError: if the derived segment would be empty.
    """
    segment = model_name.split("/")[-1].strip().lower()
    if not segment:
        raise ValueError(f"cannot derive model segment from {model_name!r}")
    return segment


def dataset_id(config: EmbeddingConfig) -> str:
    """Return the dataset manifest's plain id (unique per generation).

    The id names model/variant/version and carries a 12-char hash suffix so
    two configs that share the partition stack (same version prefix, e.g. a
    prompt-flag change) never share a manifest object — dataset ids are
    addresses, never rewritten (data.md §2.3).

    Args:
        config: The embedding configuration.

    Returns:
        The plain (no `k=v`) dataset id.
    """
    return (
        f"embeddings-{config.model_segment}-{config.variant}-{config.version}-"
        f"{configuration_hash(config)[:12]}"
    )
