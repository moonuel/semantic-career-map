"""Run-record merging, dataset manifests, orphan healing (task 3.6, D10).

Three products at the end of a cleaning run:

- **Flush checkpoints** — written after each part by the writer
  (:mod:`backend.pipeline.cleaning.writer`); the per-flush truth this
  module aggregates.
- **Run record** — ``manifests/runs/run_id=<run-id>/run.json``, merged
  from the flush checkpoints (cleaned rows, latency, usage, cost) plus
  runner-supplied fields (timestamps, counters that live per-run —
  cache-hit, failed — status, worker count). Cache hits and failed
  postings never enter a checkpoint, so they arrive as runner arguments.
- **Dataset manifest** — ``manifests/datasets/<dataset-id>.json``, the
  dataset-level lineage manifest of data.md §3.

Plus the recovery story: an orphan part (its checkpoint never written —
a crash between the two writer puts) is back-filled from the part's rows
as a ``recovered`` checkpoint by :func:`heal_orphan_checkpoints`, "the
next run's aggregation step" in D10. Rows carry no call telemetry, so a
recovered checkpoint reports zero usage/cost/latency with
``recovered: true`` — durability of the rows is what back-fills, not the
telemetry.
"""

from __future__ import annotations

import datetime
import io
import json
from typing import Any, Protocol

import pyarrow.parquet as pq

from backend.pipeline.cleaning.cache import (  # type: ignore — same package
    list_dataset_parts,
    parse_part_key,
)
from backend.pipeline.cleaning.writer import (  # type: ignore — same package
    build_flush_checkpoint,
    latency_aggregates,
)
from backend.storage import keys


class RunStore(Protocol):
    """The put/get/list surface run management needs."""

    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def list_keys(self, prefix: str) -> list[str]: ...


def _utc_now() -> str:
    """Timestamp helper (ISO-8601 UTC)."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def dataset_id_for(
    artifact_type: str = "cleaned_text", variant: str = "llm", version: str = "v1"
) -> str:
    """Compose the dataset-id token (``cleaned_text-llm-v1`` shape).

    Args:
        artifact_type: Governed artifact family.
        variant: Producing method.
        version: Version token ``vN``.

    Returns:
        The dataset id used by :func:`keys.manifest_dataset`.
    """
    dataset_id = f"{artifact_type}-{variant}-{version}"
    keys.manifest_dataset(dataset_id)  # validate the composed id
    return dataset_id


def list_flush_checkpoints(
    store: RunStore,
    run_id: str,
) -> list[dict[str, Any]]:
    """Read a run's flush checkpoints, in flush order.

    Args:
        store: Object store (or testing double).
        run_id: The run's UUIDv7.

    Returns:
        Checkpoint dicts with ``seq`` ascending.

    Raises:
        ValueError: if a checkpoint object is not valid JSON.
    """
    run_prefix = f"lake/manifests/runs/run_id={run_id}/"
    checkpoints: list[dict[str, Any]] = []
    for key in store.list_keys(run_prefix):
        if not key.endswith(".json"):
            continue
        leaf = key.rstrip("/").rsplit("/", 1)[-1]
        if not leaf.startswith("flush="):
            # the run record itself (``run.json``) lives under the same
            # prefix; it is not a flush checkpoint and carries no ``seq``
            continue
        keys.validate_key(key)
        checkpoints.append(json.loads(store.get(key).decode("utf-8")))
    return sorted(checkpoints, key=lambda c: c["seq"])


def _existing_checkpoints(store: RunStore) -> set[tuple[str, int]]:
    """Collect the ``(run_id, seq)`` pairs of all recorded checkpoints.

    One listing of the run-manifest root instead of one probe per part —
    the set covers every historical run, and malformed leaves contribute
    nothing rather than failing a healing pass.
    """
    existing: set[tuple[str, int]] = set()
    for key in store.list_keys("lake/manifests/runs/"):
        parent, leaf = key.rstrip("/").rsplit("/", 1)
        if not leaf.startswith("flush=") or not leaf.endswith(".json"):
            continue
        run_segment = parent.rsplit("/", 1)[-1]
        if not run_segment.startswith("run_id="):
            continue
        try:
            seq = int(leaf.removeprefix("flush=").removesuffix(".json"))
        except ValueError:
            continue
        existing.add((run_segment.removeprefix("run_id="), seq))
    return existing


def heal_orphan_checkpoints(
    store: RunStore,
    *,
    artifact_type: str = "cleaned_text",
    variant: str = "llm",
    version: str = "v1",
) -> list[dict[str, Any]]:
    """Back-fill checkpoints for parts whose checkpoint never landed.

    Args:
        store: Object store (or testing double).
        artifact_type: Governed artifact family.
        variant: Producing method.
        version: Version token ``vN``.

    Returns:
        The recovery checkpoints written this call (empty when nothing
        was orphaned).
    """
    existing = _existing_checkpoints(store)
    recovered: list[dict[str, Any]] = []
    for part_key in list_dataset_parts(
        store, artifact_type=artifact_type, variant=variant, version=version
    ):
        run_id, seq = parse_part_key(part_key)
        if (run_id, seq) in existing:
            continue
        rows = pq.read_table(
            io.BytesIO(store.get(part_key)),
            columns=["source_hash", "configuration_hash"],
        )
        checkpoint = build_flush_checkpoint(
            run_id=run_id,
            seq=seq,
            trigger="recovered",
            part_key=part_key,
            source_hashes=rows.column("source_hash").to_pylist(),
            configuration_hash=rows.column("configuration_hash")[0].as_py(),
            batch_range=None,
            durations=[],
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            cost_usd=0.0,
        )
        checkpoint["recovered"] = True
        payload = json.dumps(checkpoint, indent=2, ensure_ascii=False).encode("utf-8")
        store.put(keys.manifest_run_checkpoint(run_id, seq), payload)
        recovered.append(checkpoint)
    return recovered


def merge_run_record(
    store: RunStore,
    run_id: str,
    *,
    status: str,
    input_dataset: str,
    output_dataset_id: str,
    model: str,
    configuration_hash: str,
    pipeline_version: str,
    started_at: str,
    completed_at: str,
    worker_count: int = 1,
    cache_hits: int = 0,
    failed: int = 0,
    failed_source_hashes: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate the run record from its flush checkpoints and write it.

    Args:
        store: Object store (or testing double).
        run_id: The run's UUIDv7.
        status: Final status — ``completed`` | ``failed`` | ``interrupted``.
        input_dataset: The pinned input manifest reference (path or key).
        output_dataset_id: The dataset id the parts belong to.
        model: Gateway model the run asked for.
        configuration_hash: The run's configuration digest.
        pipeline_version: Git sha (or dev tag) of the producing code.
        started_at: Run start timestamp (ISO-8601 UTC).
        completed_at: Run end timestamp (ISO-8601 UTC).
        worker_count: Configured bounded-pool size.
        cache_hits: Postings resolved from the cache without a call.
        failed: Postings that errored (fail-fast: 1 when ``failed``).
        failed_source_hashes: The failing postings' hashes — the
            fail-fast failure ledger the selection reader consumes.

    Returns:
        The run record dict (also written to its ``run.json`` key).

    Raises:
        ValueError: if ``status`` is not one of the terminal statuses.
    """
    if status not in {"completed", "failed", "interrupted"}:
        raise ValueError(f"run status {status!r} must be completed|failed|interrupted")
    checkpoints = list_flush_checkpoints(store, run_id)
    durations: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    cost = 0.0
    for c in checkpoints:
        durations.extend(c.get("latency", {}).get("durations", []))
        prompt_tokens += c.get("usage", {}).get("prompt_tokens", 0)
        completion_tokens += c.get("usage", {}).get("completion_tokens", 0)
        cost += c.get("cost_usd") or 0.0
    run_record = {
        "run_id": run_id,
        "kind": "cleaning_run",
        "pipeline_version": pipeline_version,
        "started_at": started_at,
        "completed_at": completed_at,
        "input_dataset": input_dataset,
        "output_dataset_id": output_dataset_id,
        "model": model,
        "configuration_hash": configuration_hash,
        "counters": {
            "cleaned": sum(c["row_count"] for c in checkpoints),
            "cache_hits": cache_hits,
            "failed": failed,
        },
        "worker_count": worker_count,
        "latency": latency_aggregates(durations),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "cost_usd": round(cost, 8),
        "flush_refs": [c["part_key"] for c in checkpoints],
        "status": status,
        "failed_source_hashes": list(failed_source_hashes or []),
    }
    payload = json.dumps(run_record, indent=2, ensure_ascii=False).encode("utf-8")
    store.put(keys.manifest_run(run_id), payload)
    return run_record


def write_dataset_manifest(
    store: RunStore,
    *,
    dataset_id: str,
    input_dataset: str,
    model: str | None,
    configuration_hash: str | None,
    pipeline_version: str,
    record_count: int,
) -> dict[str, Any]:
    """Write the dataset-level lineage manifest (data.md §3).

    Args:
        store: Object store (or testing double).
        dataset_id: The composed dataset id (see :func:`dataset_id_for`).
        input_dataset: The dataset/manifest this version was built from.
        model: The producing model (``None`` for non-LLM datasets).
        configuration_hash: The producing configuration (``None`` when the
            artifact family has no prompt configuration).
        pipeline_version: Git sha (or dev tag).
        record_count: Total rows in the version at write time.

    Returns:
        The dataset manifest dict (also written to its key).
    """
    manifest = {
        "dataset_id": dataset_id,
        "input_dataset": input_dataset,
        "model": model,
        "configuration_hash": configuration_hash,
        "pipeline_version": pipeline_version,
        "record_count": record_count,
        "created_at": _utc_now(),
    }
    payload = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    store.put(keys.manifest_dataset(dataset_id), payload)
    return manifest
