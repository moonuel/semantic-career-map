"""Embedding driver: delta selection → CPU encode → buffered lake writes.

The embedding tier's orchestration: store the run's config object
content-addressed, bind its configuration hash into the ``select_unembedded``
anti-join, stream ``(source_hash, text)`` rows through the shared Athena
runner, encode the delta in batch windows, and hand each batch to the
run's :class:`~backend.pipeline.embedding.writer.FlushWriter` — buffer
thresholds (rows/bytes/dwell/drained/stop) commit one Parquet part per
flush under the run's version prefix, each part immediately followed by its
flush checkpoint. Rows carry the full provenance tuple (data.md §7 columns
+ ``configuration_hash`` + ``run_id``) so the anti-join's cache identity
lives in the data itself. One dataset manifest per run records the lineage
(query execution id, SQL hash, pipeline version, model revision, record
count). A zero-row delta lands no parts and no manifest — the run summary
reports the healthy no-op; the run record (Layer 4) documents it.

Everything durable is idempotent by construction: the anti-join only
selects un-embedded postings, so replays re-embed nothing committed; before
any selection the run heals its configuration's slice (duplicate-row
reconcile, then orphan-checkpoint back-fill).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from backend.pipeline.athena import (
    iter_query_results,
    poll_query_completion,
    results_location,
    submit_query,
)
from backend.pipeline.cleaning.daily import resolve_pipeline_version
from backend.pipeline.embedding import selection
from backend.pipeline.embedding.config import (
    EmbeddingConfig,
    config_document,
    configuration_hash,
    dataset_id,
)
from backend.pipeline.embedding.encoder import EmbeddingEncoder
from backend.pipeline.embedding.reconcile import (
    ReconcileReport,
    heal_orphan_checkpoints,
    reconcile_configuration,
)
from backend.pipeline.embedding.version_claim import claim_version
from backend.pipeline.embedding.writer import FlushWriter
from backend.storage import ids, keys

DATABASE_DEFAULT = "semantic_career_map"
WORKGROUP_DEFAULT = "semantic_career_map"


@dataclass(frozen=True)
class RunSummary:
    """The run's observable outcome (the Layer-4 run record's source).

    Attributes:
        run_id: The minted run id.
        configuration_hash: The run's configuration digest.
        execution_id: The selection's Athena query execution id.
        selected: Rows the anti-join selected (embedded + empty-text).
        embedded: Rows committed to parts.
        empty: Selected rows with no usable text (counted, never dropped).
        zero_delta: Healthy no-op marker (nothing selected).
        part_keys: Committed part keys in flush order.
        dataset_manifest_key: The lineage manifest, ``None`` on no-op.
        reconcile: The pre-selection duplicate-row reconciliation report
            (``None`` when the pass was disabled).
        flushes: This run's flush checkpoints (part → checkpoint books).
        recovered: Orphan-checkpoint back-fills this run performed.
    """

    run_id: str
    configuration_hash: str
    execution_id: str | None
    selected: int
    embedded: int
    empty: int
    zero_delta: bool
    part_keys: list[str]
    dataset_manifest_key: str | None
    reconcile: ReconcileReport | None = None
    flushes: list[dict[str, Any]] = field(default_factory=list)
    recovered: list[dict[str, Any]] = field(default_factory=list)


def ensure_config_object(store, config: EmbeddingConfig) -> str:
    """Store the config document (idempotent) and return its hash.

    The object at ``configs/<hash>.json`` carries the exact parameter dict
    (verbatim SQL included) beside its hash — the manifest rows reference
    only the hash, and this write happens before any selection so a run is
    never the sole holder of its parameters.

    Args:
        store: The object store.
        config: The run's configuration.

    Returns:
        The 64-char lowercase hex configuration hash.
    """
    digest = configuration_hash(config)
    key = keys.config(digest)
    if not store.exists(key):
        store.put(key, config_document(config))
    return digest


def _manifest_bytes(
    config: EmbeddingConfig,
    configuration_hash: str,
    *,
    run_id: str,
    execution_id: str | None,
    sql_hash: str,
    record_count: int,
    part_keys: list[str],
    created_at: datetime,
    pipeline_version: str,
) -> bytes:
    """Serialize the dataset manifest (manifests/datasets discipline).

    Args:
        config: The run's configuration.
        configuration_hash: The run's configuration hash.
        run_id: The minted run id.
        execution_id: The selection's Athena query execution id.
        sql_hash: SHA-256 of the verbatim SQL text.
        record_count: Rows committed across the run's parts.
        part_keys: The committed part keys, in flush order.
        created_at: The run's wall-clock commit time.
        pipeline_version: Producing-code version.

    Returns:
        Canonical JSON manifest bytes.
    """
    document = {
        "dataset_id": dataset_id(config),
        "configuration_hash": configuration_hash,
        "input_dataset": config.input_artifact,
        "input_version": config.input_version,
        "query_execution_id": execution_id,
        "sql_hash": sql_hash,
        "model": config.model_name,
        "model_revision": config.model_revision,
        "outputs": {
            "model_segment": config.model_segment,
            "variant": config.variant,
            "version": config.version,
        },
        "run_id": run_id,
        "record_count": record_count,
        "parts": part_keys,
        "created_at": created_at.isoformat(),
        "pipeline_version": pipeline_version,
    }
    return json.dumps(
        document, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def run_embedding(
    store,
    athena,
    encoder: EmbeddingEncoder,
    config: EmbeddingConfig,
    *,
    database: str = DATABASE_DEFAULT,
    workgroup: str = WORKGROUP_DEFAULT,
    pipeline_version: str | None = None,
    reconcile: bool = True,
    now: Callable[[], datetime] | None = None,
) -> RunSummary:
    """One embedding run over the real (or doubled) wiring.

    Delta-only semantics: the config's SQL text (the anti-join, already
    deployment-bounded where it was generated) selects un-embedded postings
    for the run's configuration; the encoder processes batch windows; one
    part commits per batch; the dataset manifest commits once. A zero-row
    selection is the healthy no-op: the summary reports ``zero_delta`` with
    no parts and no manifest written. Before selecting, the run reconciles
    its configuration's lake slice — concurrent-run duplicate rows from
    earlier runs heal (in-flight runs deferred); the report rides the
    summary.

    Args:
        store: The object store (real or testing double).
        athena: The boto3 Athena client (or testing double).
        encoder: The encode boundary (real or stub).
        config: The run's configuration (its SQL text is hashed identity).
        database: Glue database for unqualified table names (execution
            context; not part of the hashed identity).
        workgroup: Athena workgroup binding (cost cap + pinned output).
        pipeline_version: Producing-code version; ``None`` resolves via
            :func:`backend.pipeline.cleaning.daily.resolve_pipeline_version`.
        reconcile: Run the pre-selection duplicate-row healing pass
            (:func:`reconcile_configuration`).
        now: Wall clock for the run's ``created_at`` and the reconcile's
            reference time; ``None`` uses now (UTC). Identity-neutral — the
            config hash never sees it — so tests pin deterministic stamps.

    Returns:
        The :class:`RunSummary`.

    Raises:
        RuntimeError: from the shared runner on a non-SUCCEEDED query.
    """
    if not config.sql_text.strip():
        # the SQL text is part of the hashed identity — an empty selection
        # text is a configuration defect, never a silently-defaulted SQL
        raise ValueError(
            "config.sql_text is empty; the selection SQL is part of the "
            "hashed identity (build the config through "
            "select_unembedded_sql … or pass --sql-file)"
        )
    now_fn = now if now is not None else (lambda: datetime.now(UTC))
    run_id = ids.new_id()
    created_at = now_fn()
    pipeline_version = pipeline_version or resolve_pipeline_version()
    digest = ensure_config_object(store, config)

    # Bind this configuration to its version prefix before anything durable is
    # written under it. A second configuration aimed at the same prefix fails
    # here instead of silently joining the dataset (issue #22), and the claim
    # precedes the reconcile pass so no run heals a prefix it does not own.
    claim_version(
        store,
        config,
        digest,
        run_id=run_id,
        claimed_at=created_at,
        pipeline_version=pipeline_version,
    )

    report: ReconcileReport | None = None
    recovered: list[dict[str, Any]] = []
    if reconcile:
        report = reconcile_configuration(store, config, as_of=now_fn())
        recovered = heal_orphan_checkpoints(store, config, as_of=now_fn())

    sql = selection.bind_sql_params(config.sql_text, {"config": digest})
    execution_id = submit_query(
        athena,
        sql,
        database=database,
        workgroup=workgroup,
        output_location=results_location(),
    )
    poll_query_completion(athena, execution_id)

    rows: list[list[str]] = list(iter_query_results(athena, execution_id))
    usable, empty = selection.shape_rows(
        rows,
        text_variant=config.text_variant,
        fallback_text_variant=config.fallback_text_variant,
    )

    if not usable:
        return RunSummary(
            run_id=run_id,
            configuration_hash=digest,
            execution_id=execution_id,
            selected=0,
            embedded=0,
            empty=empty,
            zero_delta=True,
            part_keys=[],
            dataset_manifest_key=None,
            reconcile=report,
            recovered=recovered,
        )

    batch_size = max(1, config.batch_size)
    writer = FlushWriter(
        store,
        run_id=run_id,
        config=config,
        configuration_hash=digest,
        created_at=created_at,
    )
    for index, start in enumerate(range(0, len(usable), batch_size), start=1):
        chunk = usable[start : start + batch_size]
        source_hashes = [pair[0] for pair in chunk]
        texts = [pair[1] for pair in chunk]
        vectors = encoder.encode(texts)
        if vectors.shape[0] != len(chunk):
            raise RuntimeError(
                f"encoder returned {vectors.shape[0]} rows for {len(chunk)} texts"
            )
        writer.add_batch(
            batch_id=f"encode-{index:04d}",
            source_hashes=source_hashes,
            vectors=vectors,
        )
        writer.maybe_flush()  # dwell across CPU encode windows
    writer.maybe_flush(queue_empty=True)  # drained: run input exhausted
    writer.flush_final()  # stop: remainder, interrupt-safe
    flushes = writer.results
    part_keys = [checkpoint["part_key"] for checkpoint in flushes]
    embedded_total = sum(checkpoint["row_count"] for checkpoint in flushes)

    manifest = keys.manifest_dataset(dataset_id(config))
    store.put(
        manifest,
        _manifest_bytes(
            config,
            digest,
            run_id=run_id,
            execution_id=execution_id,
            sql_hash=hashlib.sha256(config.sql_text.encode("utf-8")).hexdigest(),
            record_count=embedded_total,
            part_keys=part_keys,
            created_at=created_at,
            pipeline_version=pipeline_version,
        ),
    )
    return RunSummary(
        run_id=run_id,
        configuration_hash=digest,
        execution_id=execution_id,
        selected=len(usable) + empty,
        embedded=embedded_total,
        empty=empty,
        zero_delta=False,
        part_keys=part_keys,
        dataset_manifest_key=manifest,
        reconcile=report,
        flushes=flushes,
        recovered=recovered,
    )
