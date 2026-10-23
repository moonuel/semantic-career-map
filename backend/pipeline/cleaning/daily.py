"""Daily driver: Athena selection → pinned manifest → cleaning run.

The cloud cleaning tier's whole job in one process (report §4.5, plan T3):
the ``select_uncleaned`` anti-join runs on Athena for the run's
``configuration_hash`` (binding the selection to the exact configuration the
runner will hash), :mod:`backend.pipeline.selection` renders the canonical
pinned manifest, and :func:`runner.run_cleaning` consumes it — the existing
flush/checkpoint/run-record durability path, unchanged. An empty selection
is a healthy no-op: the runner still lands a ``completed`` zero-call run
record — the day-2 idempotence proof artifact.

The driver adds only orchestration glue:

- **Secrets**: the one non-IAM secret, ``KILO_API_KEY``, is fetched from SSM
  Parameter Store (a ``SecureString``) into the process environment before
  the gateway client is constructed — absent an already-set value, so local
  ``.env`` callers skip the SSM round trip. Never logged, never baked into
  images or task definitions.
- **Wall-clock guard**: ``--max-run-hours`` (Fargate bills wall-clock, so a
  finished task must exit — roadmap Phase 3b invariant) arms a
  ``SIGINT`` self-kill at the deadline, landing the runner's live-verified
  interrupt path (durable final flush, ``status=interrupted`` record, next
  run replays the remainder via cache hits). Deliberately no runner-side
  stop-event parameter: the SIGINT path *is* the tested mechanism (D10).
- **Exit code**: ``completed`` run record → 0; anything else (``failed``,
  ``interrupted``, infrastructure error) → 1, feeding the ECS
  task-`STOPPED` alarm unchanged.

CLI ``--help`` stays credential-free (the SSM fetch and gateway client live
inside ``main()``/the run, never at import time), and ``--dry-run``
exercises the selection + manifest render without an API key — the same
cheap smoke the gate queries use.

Usage (Fargate ``CMD`` and local runs are the same CLI)::

    python -m backend.pipeline.cleaning.daily \
        [--database semantic_career_map] [--workgroup semantic_career_map] \
        [--model ...] [--workers 8] [--version-root v1] \
        [--max-run-hours 2] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

import boto3
from dotenv import load_dotenv

from backend.pipeline.cleaning import runner
from backend.pipeline.cleaning.configuration import configuration_hash
from backend.pipeline.cleaning.llm_client import GatewayClient
from backend.pipeline.cleaning.prompts import (
    BOILERPLATE_ONLY_SYSTEM_PROMPT,
    USER_MESSAGE_TEMPLATE,
)
from backend.pipeline.selection import (
    build_selection_manifest,
    parse_query_results,
    select_uncleaned_sql,
)

DATABASE_DEFAULT = "semantic_career_map"
WORKGROUP_DEFAULT = "semantic_career_map"
MODEL_DEFAULT = "google/gemini-2.5-flash-lite"
SSM_SECRET_PATH_DEFAULT = "/semantic-career-map/KILO_API_KEY"

_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})
# stable join-handle contract of the selection query's first three columns
_HEADER = ("source", "source_hash", "source_id")


@dataclass(frozen=True)
class Settings:
    """Resolved per-run configuration (task-def env + CLI flags merged)."""

    database: str
    workgroup: str
    model: str
    workers: int
    version: str
    max_run_hours: float
    configuration_hash: str


def _require_hex64(value: str) -> str:
    """Validate the 64-hex digest shape (fail loudly on a drifted env)."""
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"configuration hash {value!r} is not 64-char hex")
    return value


def resolve_configuration_hash(model: str) -> str:
    """Return the run's configuration digest: env pin or computed default.

    The default flows from the same (model, prompts) triple the runner will
    hash, so selection and run are bound by construction.
    ``CONFIGURATION_HASH`` overrides for re-runs that must target an earlier
    dataset generation.

    Args:
        model: Gateway model identifier.

    Returns:
        The 64-char hex configuration hash.

    Raises:
        ValueError: if the ``CONFIGURATION_HASH`` override is not hex64.
    """
    override = os.environ.get("CONFIGURATION_HASH")
    if override:
        return _require_hex64(override.strip())
    return configuration_hash(
        model, BOILERPLATE_ONLY_SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE
    )


def bind_config_sql(sql: str, config: str) -> str:
    """Bind the ``:config`` placeholder (the gate's own binding shape).

    Args:
        sql: The ``select_uncleaned.sql`` text.
        config: The 64-hex configuration digest.

    Returns:
        The SQL text with ``:config`` bound to a quoted literal.
    """
    return sql.replace(":config", f"'{config}'")


def fetch_secret_if_absent(ssm, name: str = SSM_SECRET_PATH_DEFAULT) -> None:
    """Make ``KILO_API_KEY`` available from SSM when the env lacks it.

    Args:
        ssm: A boto3 SSM client (or a testing double).
        name: The ``SecureString`` parameter path.

    Raises:
        ClientError: propagated — a missing secret cannot default to anything.
    """
    if os.environ.get("KILO_API_KEY"):
        return
    parameter = ssm.get_parameter(Name=name, WithDecryption=True)
    os.environ["KILO_API_KEY"] = parameter["Parameter"]["Value"]


def select_query_results(athena, sql: str, results_location: str) -> str:
    """Submit the selection query, poll, and re-serialize rows as strict CSV.

    Cells are collected per row and written with the :mod:`csv` module, so a
    ``source_id`` carrying a comma survives
    :func:`parse_query_results` intact (the gate's earlier
    ``",".join`` preview could and did corrupt that field).

    Args:
        athena: The boto3 Athena client (or a testing double).
        sql: The bound query text (see :func:`bind_config_sql`).
        results_location: The in-bucket output prefix to pin results to
            (``s3://<bucket>/athena-results/``, the workgroup's own).

    Returns:
        CSV text (header + rows), ready for ``parse_query_results``.

    Raises:
        RuntimeError: when the query ends in a non-SUCCEEDED state.
    """
    response = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": results_location},
    )
    execution_id = response["QueryExecutionId"]
    while True:
        status = athena.get_query_execution(QueryExecutionId=execution_id)[
            "QueryExecution"
        ]["Status"]
        if status["State"] in _TERMINAL_STATES:
            break
        time.sleep(2)
    if status["State"] != "SUCCEEDED":
        raise RuntimeError(
            f"query {execution_id} ended {status['State']}: "
            f"{status.get('StateChangeReason', 'no reason given')}"
        )

    rows: list[list[str]] = []
    next_token = None
    while True:
        kwargs: dict[str, Any] = {"QueryExecutionId": execution_id}
        if next_token:
            kwargs["NextToken"] = next_token
        page = athena.get_query_results(**kwargs)
        for row in page["ResultSet"]["Rows"]:
            rows.append([cell.get("VarCharValue", "") for cell in row["Data"]])
        next_token = page.get("NextToken")
        if not next_token:
            break

    if rows and tuple(rows[0][:3]) == _HEADER:
        rows = rows[1:]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_HEADER)
    writer.writerows(rows)
    return buffer.getvalue()


def pipeline_version() -> str:
    """Short git sha, ``PIPELINE_VERSION`` override, or ``dev`` fallback.

    Containers carry no git binary; the ``PIPELINE_VERSION`` env (the
    task-definition's image tag) is the override path there.

    Returns:
        The pipeline version string recorded in run manifests.
    """
    override = os.environ.get("PIPELINE_VERSION")
    if override:
        return override
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except OSError, subprocess.SubprocessError:
        return "dev"


def arm_deadline(max_run_hours: float) -> threading.Timer | None:
    """Arm the wall-clock guard as a self-addressed ``SIGINT`` timer.

    Args:
        max_run_hours: The deadline; ``<= 0`` disables the guard (tests,
            foreground local use).

    Returns:
        The **started** ``Timer``, or ``None`` when disabled. The caller
        must ``cancel()`` it on normal completion.
    """
    if max_run_hours <= 0:
        return None
    timer = threading.Timer(
        max_run_hours * 3600.0,
        lambda: os.kill(os.getpid(), signal.SIGINT),
    )
    timer.start()
    return timer


def _write_manifest(manifest_bytes: bytes) -> str:
    """Persist manifest bytes to a temp file for the runner's path input."""
    with tempfile.NamedTemporaryFile(
        prefix="daily-manifest-", suffix=".json", delete=False
    ) as handle:
        handle.write(manifest_bytes)
    return handle.name


def run_daily(
    athena, store, client, settings: Settings, pipeline_version: str | None = None
) -> dict:
    """Selection → manifest → one cleaning run (the whole tier's cycle).

    Args:
        athena: The boto3 Athena client (or a testing double).
        store: The object store (real or testing double) — read for the
            failure ledger and raw payloads, written by the runner.
        client: The gateway client (real or stubbed) for the runner.
        settings: The resolved per-run configuration.
        pipeline_version: Producing-code version; ``None`` resolves via
            :func:`pipeline_version`.

    Returns:
        The persisted run record dict (see
        :func:`backend.pipeline.cleaning.runs.merge_run_record`).

    Raises:
        KeyboardInterrupt: a deadline/c signal that caught the driver
            outside the runner's own scope; any run record committed in the
            runner's tail remains durable before propagation.
        RuntimeError: upstream on a non-SUCCEEDED Athena query.
    """
    sql = bind_config_sql(select_uncleaned_sql(), settings.configuration_hash)
    csv_text = select_query_results(athena, sql, results_location())
    rows = parse_query_results(csv_text)
    manifest_bytes = build_selection_manifest(
        rows, configuration_hash=settings.configuration_hash, store=store
    )
    manifest_path = _write_manifest(manifest_bytes)
    timer = arm_deadline(settings.max_run_hours)
    try:
        return runner.run_cleaning(
            store,
            runner.load_manifest(manifest_path),
            input_dataset=f"athena:{settings.database}/select_uncleaned",
            model=settings.model,
            system_prompt=BOILERPLATE_ONLY_SYSTEM_PROMPT,
            user_template=USER_MESSAGE_TEMPLATE,
            client=client,
            pipeline_version=pipeline_version or pipeline_version(),
            workers=settings.workers,
            version=settings.version,
        )
    finally:
        if timer is not None:
            timer.cancel()


def run_daily_dry(athena, store, settings: Settings) -> dict:
    """Selection + manifest render, no LLM calls, no store writes (smoke).

    Read-only overall: the manifest render reads the failure ledger from
    the store and writes nothing.

    Args:
        athena: The boto3 Athena client (or a testing double).
        store: The object store (real or testing double) for the ledger.
        settings: The resolved per-run configuration.

    Returns:
        A summary dict: status, digest, and the manifest posting count.
    """
    sql = bind_config_sql(select_uncleaned_sql(), settings.configuration_hash)
    csv_text = select_query_results(athena, sql, results_location())
    rows = parse_query_results(csv_text)
    manifest_bytes = build_selection_manifest(
        rows, configuration_hash=settings.configuration_hash, store=store
    )
    return {
        "status": "dry-run",
        "configuration_hash": settings.configuration_hash,
        "manifest_postings": len(json.loads(manifest_bytes)["postings"]),
    }


def results_location() -> str:
    """The in-bucket Athena results prefix (the workgroup's pinned output)."""
    return f"s3://{os.environ['STORE_S3_BUCKET']}/athena-results/"


def settings_from_env(args: argparse.Namespace) -> Settings:
    """Merge the task-def environment + CLI flags into a :class:`Settings`.

    Flags override their environment equivalents (an explicit flag describes
    a manual run; the env describes the scheduled one).

    Args:
        args: The parsed CLI namespace (``None``-able option fields).

    Returns:
        The resolved settings, configuration digest included.

    Raises:
        ValueError: on a non-hex ``CONFIGURATION_HASH`` override.
    """
    model = args.model or os.environ.get("CLEAN_MODEL") or MODEL_DEFAULT
    workers = args.workers
    if workers is None:
        workers = int(os.environ.get("CLEAN_WORKERS", "8"))
    version = args.version_root or os.environ.get("CLEAN_VERSION") or "v1"
    max_run_hours = args.max_run_hours
    if max_run_hours is None:
        max_run_hours = float(os.environ.get("MAX_RUN_HOURS", "2"))
    return Settings(
        database=args.database,
        workgroup=args.workgroup,
        model=model,
        workers=workers,
        version=version,
        max_run_hours=max_run_hours,
        configuration_hash=resolve_configuration_hash(model),
    )


def main() -> None:
    """CLI flow: resolve settings → (dry) daily run → exit code mapping."""
    load_dotenv()  # local convention; containers rely on task-def env
    parser = argparse.ArgumentParser(
        prog="python -m backend.pipeline.cleaning.daily",
        description="Daily cleaning tier: Athena selection → manifest → "
        "one cleaning run over the real store and gateway.",
    )
    parser.add_argument("--database", default=DATABASE_DEFAULT)
    parser.add_argument("--workgroup", default=WORKGROUP_DEFAULT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--version-root", default=None)
    parser.add_argument("--max-run-hours", type=float, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the selection + manifest render only (no API secret needed)",
    )
    args = parser.parse_args()

    settings = settings_from_env(args)
    if args.dry_run:
        from backend.storage.store import ObjectStore

        summary = run_daily_dry(
            boto3_client("athena"), ObjectStore.from_env(), settings
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    fetch_secret_if_absent(boto3.Session().client("ssm"))
    from backend.storage.store import ObjectStore

    athena = boto3_client("athena")
    try:
        record = run_daily(athena, ObjectStore.from_env(), GatewayClient(), settings)
    except KeyboardInterrupt:
        print(
            "daily: interrupted before/around the run; any committed "
            "flushes are durable and skippable on rerun"
        )
        raise SystemExit(1) from None
    print(json.dumps(record, indent=2, ensure_ascii=False))
    if record["status"] != "completed":
        raise SystemExit(1)


def boto3_client(kind: str) -> Any:
    """A boto3 client pinned to the lake's region."""
    return boto3.Session().client(kind, region_name=os.environ["STORE_S3_REGION"])


if __name__ == "__main__":
    main()
