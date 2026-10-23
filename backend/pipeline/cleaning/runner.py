"""Cleaning-run orchestration (task 3.7, design D10/D9).

Wires the stage end-to-end over a manifest from either producer (§2.4 SQL
adapter or §2.5 sampler — same format): cache resolution (3.5) → bounded
thread pool of LLM calls → single-writer flusher (3.4) → run record +
dataset manifest (3.6). Orphan healing (3.6) runs at start — "the next
run's aggregation step" back-filling any part whose checkpoint is
missing.

Discipline:

- **Fail-fast per posting**: any worker exception cancels the remaining
  futures, flushes the buffer, and the run record lands ``failed`` with
  the failing posting's hash in ``failed_source_hashes`` (the ledger the
  SQL-selection reader consumes).
- **Flush-on-stop**: a SIGINT takes the same path with status
  ``interrupted`` — committed flushes stay durable, only calls that
  never finish are lost, and the rerun resolves their postings as cache
  hits (D10's documented loss window). In-flight calls are joined before
  the record merges, so their rows are counted, not replayed.
- **Zero client calls on second run**: every posting already holding the
  ``(source_hash, configuration_hash)`` pair resolves from the artifact
  cache (3.5).
- **Pacing**: workers jitter-sleep within the ``RATE_LIMIT_SECONDS``
  budget before each call (spacing, not rate-limiting — the client owns
  429/5xx backoff).
- **Abort exit latency**: a failed/interrupted pool joins its in-flight
  calls before returning (rows count rather than replay), so ctrl-c exit
  is bounded by one straggler call — the client's
  ``REQUEST_TIMEOUT_S`` — not instantaneous.

The gateway client is injected — unit tests stub it; the CLI wires the
real :class:`GatewayClient`.
"""

from __future__ import annotations

import datetime
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Protocol

from backend.pipeline.cleaning import cache, llm_client, runs
from backend.pipeline.cleaning.configuration import configuration_hash
from backend.pipeline.cleaning.llm_client import RATE_LIMIT_SECONDS
from backend.pipeline.cleaning.normalization import SHAPE_TEXT_FIELD, normalize_html
from backend.pipeline.cleaning.writer import (
    FLUSH_BYTES_DEFAULT,
    FLUSH_ROWS_DEFAULT,
    FLUSH_SECONDS_DEFAULT,
    FlushWriter,
)
from backend.storage import ids, keys
from backend.storage.raw import read_raw_payload


class CleaningClient(Protocol):
    """The complete() surface the runner asks of any client."""

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]: ...


def load_manifest(path: str) -> dict[str, Any]:
    """Load + validate a pinned manifest from either producer.

    Args:
        path: Manifest file path (``--sample-manifest``).

    Returns:
        The manifest dict (``params`` + ``postings``).

    Raises:
        ValueError: if a posting row misses the selection contract
            (allowlisted source, hex64 hash, key/identity match) — a bad
            manifest fails before any call is made.
    """
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    postings = manifest.get("postings")
    if not isinstance(postings, list):
        raise TypeError(f"manifest {path!r} has no postings list")
    for posting in postings:
        if not isinstance(posting, dict):
            raise TypeError(f"manifest posting is not an object: {posting!r}")
        source = posting.get("source", "")
        if source not in keys.RAW_SOURCES:
            raise ValueError(f"manifest posting has unknown source {source!r}")
        if not keys.is_sha256_hex(posting.get("source_hash", "")):
            raise ValueError(f"manifest posting has bad source_hash {posting!r}")
        if posting.get("key") != keys.raw_payload(source, posting["source_hash"]):
            raise ValueError(
                f"manifest posting key does not match its identity: {posting!r}"
            )
    return manifest


def _utc_now() -> str:
    """Timestamp helper (ISO-8601 UTC)."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def run_cleaning(
    store: Any,
    manifest: dict[str, Any],
    *,
    input_dataset: str,
    model: str,
    system_prompt: str,
    user_template: str,
    client: CleaningClient,
    pipeline_version: str = "dev",
    workers: int = 8,
    artifact_type: str = "cleaned_text",
    variant: str = "llm",
    version: str = "v1",
    flush_rows: int = FLUSH_ROWS_DEFAULT,
    flush_bytes: int = FLUSH_BYTES_DEFAULT,
    flush_seconds: float = FLUSH_SECONDS_DEFAULT,
    pacing_budget: float = RATE_LIMIT_SECONDS,
    heal: bool = True,
) -> dict[str, Any]:
    """Run the cleaning stage over a pinned manifest.

    Args:
        store: Object store (or testing double).
        manifest: A validated producer manifest (:func:`load_manifest`).
        input_dataset: The manifest reference recorded as lineage.
        model: Gateway model identifier.
        system_prompt: The run's cleaning system prompt.
        user_template: The run's user message template — its
            ``{raw_full_text}`` placeholder receives the normalized text.
        client: The gateway client (real or stubbed; only
            ``complete(...)`` is asked of it).
        pipeline_version: Git sha (or dev tag) of the producing code.
        workers: Bounded same-process thread-pool size.
        artifact_type: Governed artifact family.
        variant: Producing method.
        version: Version token ``vN``.
        flush_rows: Writer row trigger.
        flush_bytes: Writer bytes trigger.
        flush_seconds: Writer dwell trigger.
        pacing_budget: Worker jitter budget in seconds (0 disables).
        heal: Back-fill orphan checkpoints at start (D10).

    Returns:
        The persisted run record (see :func:`runs.merge_run_record`).

    Raises:
        KeyboardInterrupt: re-raised after the interrupted-status record
            was made durable (CLI exit semantics preserved).
    """
    config = configuration_hash(model, system_prompt, user_template)
    run_id = ids.new_id()
    started_at = _utc_now()
    if heal:
        runs.heal_orphan_checkpoints(
            store, artifact_type=artifact_type, variant=variant, version=version
        )
    index = cache.load_cache_index(
        store, artifact_type=artifact_type, variant=variant, version=version
    )
    to_run = [
        p for p in manifest["postings"] if (p["source_hash"], config) not in index
    ]
    cache_hits = len(manifest["postings"]) - len(to_run)

    writer = FlushWriter(
        store,
        run_id=run_id,
        pipeline_version=pipeline_version,
        artifact_type=artifact_type,
        variant=variant,
        version=version,
        flush_rows=flush_rows,
        flush_bytes=flush_bytes,
        flush_seconds=flush_seconds,
    )
    failure: dict[str, Any] = {}
    try:
        failure = _run_pool(
            store=store,
            to_run=to_run,
            run_id=run_id,
            writer=writer,
            model=model,
            system_prompt=system_prompt,
            user_template=user_template,
            client=client,
            config=config,
            variant=variant,
            version=version,
            pipeline_version=pipeline_version,
            workers=workers,
            pacing_budget=pacing_budget,
        )
    except KeyboardInterrupt:
        # a SIGINT anywhere inside the pool still lands an interrupted
        # record via the tail below
        failure = _report("interrupted")
    dataset_id = runs.dataset_id_for(artifact_type, variant, version)
    per_run_dataset_id = f"{dataset_id}-{run_id}"  # per-run dataset manifest

    def merge_record(status: str) -> dict[str, Any]:
        return runs.merge_run_record(
            store,
            run_id,
            status=status,
            input_dataset=input_dataset,
            output_dataset_id=per_run_dataset_id,
            model=model,
            configuration_hash=config,
            pipeline_version=pipeline_version,
            started_at=started_at,
            completed_at=_utc_now(),
            worker_count=workers,
            cache_hits=cache_hits,
            failed=len(failure.get("source_hashes", [])),
            failed_source_hashes=failure.get("source_hashes", []),
        )

    try:
        writer.flush_final()
        record = merge_record("completed" if not failure else failure["status"])
        runs.write_dataset_manifest(
            store,
            dataset_id=per_run_dataset_id,
            input_dataset=input_dataset,
            model=model,
            configuration_hash=config,
            pipeline_version=pipeline_version,
            record_count=index.row_count + record["counters"]["cleaned"],
        )
        return record
    except KeyboardInterrupt:
        # SIGINT in the tail: settle the record from the checkpoints
        # already durable, then re-raise (CLI exit semantics preserved)
        merge_record("interrupted" if not failure else failure["status"])
        raise
    except Exception:
        # Infrastructure failure in the tail (store down): still settle the
        # record from whatever checkpoints are durable — the record is the
        # only counters carrier; if even that write fails, parts alone
        # survive and the next run's healing absorbs them
        try:
            merge_record("failed" if not failure else failure["status"])
        except Exception:  # noqa: BLE001,S110 — record write must not mask the tail error
            pass
        raise


def _report(
    status: str,
    *,
    hashes: list[str] | None = None,
    exc: Exception | None = None,
) -> dict[str, Any]:
    """Build one failure/interrupted report for the pool abort paths.

    Args:
        status: ``failed`` or ``interrupted``.
        hashes: The blamed posting hashes (``failed`); empty for
            ``interrupted`` and for infrastructure faults.
        exc: The aborting exception, repr'd into the report.

    Returns:
        The failure report dict.
    """
    report: dict[str, Any] = {"status": status, "source_hashes": list(hashes or [])}
    if exc is not None:
        report["error"] = repr(exc)
    return report


def _run_pool(
    store: Any,
    *,
    to_run: list[dict[str, Any]],
    run_id: str,
    writer: FlushWriter,
    model: str,
    system_prompt: str,
    user_template: str,
    client: Any,
    config: str,
    variant: str,
    version: str,
    pipeline_version: str,
    workers: int,
    pacing_budget: float,
) -> dict[str, Any]:
    """Drive one flushing pool over ``to_run`` (fail-fast; stop-aware).

    Returns:
        ``{}`` when every posting cleaned; otherwise a failure report:
        ``status`` → ``failed`` | ``interrupted`` and (failed only)
        ``source_hashes`` identifying the failing postings.
    """
    rng = random.Random(run_id)  # stable jitter phasing per run

    def work(posting: dict[str, Any]) -> None:
        payload = read_raw_payload(store, posting["key"])
        field = SHAPE_TEXT_FIELD[posting["source"]]
        raw_text = payload.data.get(field) or ""
        text = normalize_html(raw_text)
        user_message = user_template.format(raw_full_text=text)
        if pacing_budget > 0:
            time.sleep(rng.uniform(0.5, 1.5) * pacing_budget)
        started = time.monotonic()
        content, usage = client.complete(system_prompt, user_message, model=model)
        duration = time.monotonic() - started
        writer.add(
            {
                "source_hash": posting["source_hash"],
                "variant": variant,
                "version": version,
                "cleaned_text": content,
                "configuration_hash": config,
                "pipeline_version": pipeline_version,
            },
            batch_id=posting["source_hash"],
            duration_s=duration,
            usage=usage,
            cost_usd=llm_client.price_call(model, usage),
        )

    executor = ThreadPoolExecutor(max_workers=max(workers, 1))
    futures = {executor.submit(work, posting): posting for posting in to_run}
    failure: dict[str, Any] = {}
    try:
        try:
            for future in as_completed(futures):
                posting = futures[future]
                try:
                    future.result()
                except KeyboardInterrupt:
                    # interrupted: no posting failed — the (possibly
                    # finished) in-flight call is the documented loss
                    # window; the ledger must stay empty so the rerun
                    # re-cleans it
                    failure = _report("interrupted")
                    break
                except Exception as exc:  # noqa: BLE001 — any failure is fail-fast
                    failure = _report(
                        "failed", hashes=[posting["source_hash"]], exc=exc
                    )
                    break
                try:
                    writer.maybe_flush()  # dwell chance between completions
                except Exception as exc:  # noqa: BLE001 — fail fast on flush errors
                    # flush/store errors belong to the infrastructure:
                    # fail fast with an empty ledger, never blame a posting
                    failure = _report("failed", exc=exc)
                    break
        except KeyboardInterrupt:
            # SIGINT from the main thread (arrives while the completion
            # iterator is waiting, outside the per-future try)
            failure = _report("interrupted")
    finally:
        if failure:
            # failed/interrupted: cancel whatever is left — no new work
            # starts either way
            for f in futures:
                f.cancel()
        else:
            writer.maybe_flush(queue_empty=True)  # the queue is now drained
        # wait=True joins in-flight workers even on failure/interrupt, so
        # no row can report (or auto-flush) after this function returns
        executor.shutdown(wait=True, cancel_futures=bool(failure))
    return failure
