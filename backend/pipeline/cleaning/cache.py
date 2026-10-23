"""Artifact-row cache resolution (task 3.5, design D-cache).

The dataset **is** the cache: before an LLM call, the runner looks up the
pair ``(source_hash, configuration_hash)`` among existing artifact rows —
a hit reuses the prior row through no API call and writes nothing new;
a configuration change re-cleans by construction (different hash → no
pair exists). The legacy flat file ``data/.llm_clean_cache.json`` is
never read or written by production code — one storage mechanism, no
dual source of truth (design line: "cache hit = artifact row already
exists for that pair").

Index construction lists the dataset version's Parquet parts and pairs
rows by the two identity columns. Parts are listed/validated through the
same key discipline as everything else: no hand-built part keys; the
version prefix is composed from validated segments and every returned
part key passes :func:`keys.validate_key` before reading.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from typing import Protocol

import pyarrow.parquet as pq

from backend.storage import keys

# Stable probe run id used only to route segment validation through the
# artifact_part builder when composing a version prefix (no I/O; pure).
_PROBE_RUN_ID = "0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d70"


class GetListStore(Protocol):
    """The get/list storage surface cache/runs modules need."""

    def get(self, key: str) -> bytes: ...

    def list_keys(self, prefix: str) -> Iterable[str]: ...


def dataset_prefix(
    artifact_type: str = "cleaned_text",
    variant: str = "llm",
    version: str = "v1",
) -> str:
    """Compose + validate the part-prefix of one dataset version.

    Args:
        artifact_type: Governed artifact family.
        variant: Producing method.
        version: Version token ``vN``.

    Returns:
        The Hive-style prefix (ends with ``/``) under which
        ``part-<run-id>-<seq>.parquet`` parts live.
    """
    keys.artifact_part(artifact_type, variant, version, _PROBE_RUN_ID, 1)
    return f"lake/artifacts/type={artifact_type}/variant={variant}/version={version}/"


def list_dataset_parts(
    store: GetListStore,
    *,
    artifact_type: str = "cleaned_text",
    variant: str = "llm",
    version: str = "v1",
) -> list[str]:
    """List this dataset version's Parquet part keys, sorted.

    Args:
        store: Object store (or testing double).
        artifact_type: Governed artifact family.
        variant: Producing method.
        version: Version token ``vN``.

    Returns:
        Sorted validated part keys — every row of the dataset version is
        inside exactly one of them.
    """
    prefix = dataset_prefix(artifact_type, variant, version)
    parts: list[str] = []
    for key in store.list_keys(prefix):
        if key.endswith(".parquet"):
            parts.append(keys.validate_key(key))
    return parts


def parse_part_key(part_key: str) -> tuple[str, int]:
    """Split a part key into ``(run_id, seq)``.

    Args:
        part_key: A :func:`keys.artifact_part` result.

    Returns:
        The producing run's UUIDv7 and the 1-based flush sequence.

    Raises:
        ValueError: if the leaf is not a part-leaf shape.
    """
    leaf = part_key.rsplit("/", 1)[-1]
    stem = leaf.removesuffix(".parquet")
    if not stem.startswith("part-"):
        raise ValueError(f"{part_key!r} is not an artifact part key")
    run_and_seq = stem[len("part-") :]
    run_id, seq_text = run_and_seq.rsplit("-", 1)
    if not keys.is_canonical_uuid7(run_id):
        raise ValueError(f"{part_key!r} carries a non-UUIDv7 run id")
    return run_id, int(seq_text)


class ArtifactCacheIndex:
    """The set of ``(source_hash, configuration_hash)`` pairs already cleaned."""

    def __init__(self, pairs: set[tuple[str, str]], row_count: int) -> None:
        self.pairs = pairs
        self.row_count = row_count

    def __contains__(self, pair: tuple[str, str]) -> bool:
        return pair in self.pairs

    def __len__(self) -> int:
        return len(self.pairs)


def load_cache_index(
    store: GetListStore,
    *,
    artifact_type: str = "cleaned_text",
    variant: str = "llm",
    version: str = "v1",
) -> ArtifactCacheIndex:
    """Build the cache-resolution index over one dataset version.

    Args:
        store: Object store (or testing double).
        artifact_type: Governed artifact family.
        variant: Producing method.
        version: Version token ``vN``.

    Returns:
        The index: pair set (the cache) + row count (dataset size
        accounting for the dataset manifest).

    Raises:
        ValueError: if a part lacks an identity column (a drifted schema
            must surface here, not inside a run loop).
    """
    pairs: set[tuple[str, str]] = set()
    row_count = 0
    for part_key in list_dataset_parts(
        store, artifact_type=artifact_type, variant=variant, version=version
    ):
        buf = io.BytesIO(store.get(part_key))
        schema = pq.read_schema(buf)
        for column in ("source_hash", "configuration_hash"):
            if column not in schema.names:
                raise ValueError(
                    f"part {part_key!r} lacks the {column!r} identity column"
                )
        # Only the identity columns are decoded; LLM-text payloads stay out
        # of memory (the index is the cache, not the dataset).
        table = pq.read_table(buf, columns=["source_hash", "configuration_hash"])
        for pair in zip(
            table.column("source_hash").to_pylist(),
            table.column("configuration_hash").to_pylist(),
        ):
            pairs.add(pair)
        row_count += table.num_rows
    return ArtifactCacheIndex(pairs, row_count)
