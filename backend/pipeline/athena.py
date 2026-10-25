"""Shared Athena runner: submit → poll → paginate (the single copy).

One implementation of the boto3 Athena query lifecycle, used by the daily
cleaning driver (:mod:`backend.pipeline.cleaning.daily`) and the
provisioner's verification gates (``scripts.provision_query_layer.gate``):
submission with optional workgroup binding, terminal-state polling, and
NextToken pagination over ``get_query_results``. Callers own everything
before and after — the SQL text and the row shaping — so the lifecycle
itself cannot drift between them again.

The region rule lives here too (Athena endpoints are region-local to the
lake bucket).
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from typing import Any

_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})

_POLL_SECONDS = 2.0

_BUCKET_REGION_RE = re.compile(r"-(us-[a-z]+-\d)-an$")


def aws_region() -> str:
    """The lake bucket's region (Athena/Glue endpoints are region-local).

    ``STORE_S3_REGION`` is authoritative when set; else parse the bucket
    name's region suffix (``…-<region>-an``). AWS CLI defaults never
    override — an ambient us-east-1 default against a us-east-2 bucket was
    observed live: Athena rejects out-of-region results prefixes.

    Returns:
        The AWS region string.

    Raises:
        ValueError: when ``STORE_S3_REGION`` is unset and the bucket name
            carries no region suffix.
    """
    env = os.environ.get("STORE_S3_REGION")
    if env:
        return env
    match = _BUCKET_REGION_RE.search(os.environ["STORE_S3_BUCKET"])
    if match is None:
        raise ValueError(
            "STORE_S3_REGION unset and bucket name carries no <region> suffix"
        )
    return match.group(1)


def results_location() -> str:
    """The workgroup's in-bucket results prefix (read from the store env).

    Returns:
        The ``s3://`` results prefix (the workgroup's pinned output).

    Raises:
        KeyError: when ``STORE_S3_BUCKET`` is unset.
    """
    return f"s3://{os.environ['STORE_S3_BUCKET']}/athena-results/"


def submit_query(
    athena,
    sql: str,
    *,
    database: str,
    workgroup: str | None = None,
    output_location: str | None = None,
) -> str:
    """Submit one query and return its execution id.

    Args:
        athena: The boto3 Athena client (or a testing double).
        sql: The fully bound query text (no ``:config`` placeholders left).
        database: The ``QueryExecutionContext`` database (unqualified table
            names resolve there).
        workgroup: Optional workgroup binding — a provisioned workgroup
            carries the scan cap and pinned output; omitting it runs the
            query in ``primary`` with neither.
        output_location: The results prefix; defaults to
            :func:`results_location`.

    Returns:
        The ``QueryExecutionId``.

    Raises:
        OSError: propagated from the client on submission failure.
    """
    request: dict[str, Any] = {
        "QueryString": sql,
        "QueryExecutionContext": {"Database": database},
        "ResultConfiguration": {
            "OutputLocation": output_location or results_location()
        },
    }
    if workgroup is not None:
        request["WorkGroup"] = workgroup
    return athena.start_query_execution(**request)["QueryExecutionId"]


def poll_query_completion(athena, execution_id: str) -> str:
    """Poll to a terminal state; return ``"SUCCEEDED"`` or raise.

    Args:
        athena: The boto3 Athena client (or a testing double).
        execution_id: The execution id from :func:`submit_query`.

    Returns:
        The terminal state (always ``"SUCCEEDED"`` on return).

    Raises:
        RuntimeError: when the query ends in a non-SUCCEEDED state.
    """
    while True:
        status = athena.get_query_execution(QueryExecutionId=execution_id)[
            "QueryExecution"
        ]["Status"]
        if status["State"] in _TERMINAL_STATES:
            break
        time.sleep(_POLL_SECONDS)
    if status["State"] != "SUCCEEDED":
        raise RuntimeError(
            f"query {execution_id} ended {status['State']}: "
            f"{status.get('StateChangeReason', 'no reason given')}"
        )
    return status["State"]


def iter_query_results(athena, execution_id: str) -> Iterator[list[str]]:
    """Yield result rows as cell-value lists, one row at a time.

    Args:
        athena: The boto3 Athena client (or a testing double).
        execution_id: The execution id from :func:`submit_query`.

    Yields:
        One row: the cell values in column order (VerCharValue cells read
        as ``""`` when absent).
    """
    next_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"QueryExecutionId": execution_id}
        if next_token:
            kwargs["NextToken"] = next_token
        page = athena.get_query_results(**kwargs)
        for row in page["ResultSet"]["Rows"]:
            yield [cell.get("VarCharValue", "") for cell in row["Data"]]
        next_token = page.get("NextToken")
        if not next_token:
            return
