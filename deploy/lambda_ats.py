"""AWS Lambda handler for the ATS scrape tier (report §4.4, plan T4).

Lambda container images are invoked only through the Lambda Runtime API —
the image's ``ENTRYPOINT`` is ``awslambdaric`` and this module is its
``CMD`` handler. The event payload changes nothing (the tier's board list
is the scheduled configuration, not per-event); the handler maps the
batch CLI's exit code onto the Lambda response the async invocation
retry/DLQ machinery understands (``FunctionError`` shapes come from
exceptions — fail loudly for the DLQ, not politely with a 500 body:

- exit 0 → ``{"statusCode": 200, "body": {run digest}}``
- anything else → ``RuntimeError`` re-raise (invocation fails → retry ×2 →
  DLQ → the ``pipeline-dlq-not-empty`` alarm)

Credentials: no static keys — the execution role's default chain resolves
S3 directly (``ObjectStore.from_env``), and the store's endpoint env stays
empty in the image.
"""

from __future__ import annotations

import json
import typing

from scripts.scrape import main as scrape_main

ATS_SOURCES = ("ashby", "greenhouse", "lever")


def handler(event: dict, context: typing.Any) -> dict:
    """Run the whole ATS-board scrape once, then report the batch exit.

    Args:
        event: The triggering event (unused — the schedule carries the
            intent; per-event payloads would fork tier configuration).
        context: The Lambda runtime context (unused).

    Returns:
        ``{"statusCode": 200, "body": "..."}`` with the scrape digest on
        an exit-0 run (the manifest count for the run summary the logs
        capture).

    Raises:
        RuntimeError: on any non-zero batch exit — async-invocation
            failure semantics (retry, then DLQ) require it.
    """
    argv = [flag for source in ATS_SOURCES for flag in ("--source", source)]
    rc = scrape_main(argv)
    if rc != 0:
        raise RuntimeError(f"ATS scrape failed with exit code {rc}")
    return {
        "statusCode": 200,
        "body": json.dumps({"message": "ATS tier complete", "exit": 0}),
    }
