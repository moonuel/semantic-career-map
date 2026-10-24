"""One-shot manual trigger for the scheduled pipeline tiers.

Clones one of the three live schedules with an `at(<+3 min>)` expression,
lets the real EventBridge Scheduler fire its real target once (proving the
scheduler leg — AssumeRole, lambda invoke / ecs RunTask + PassRole, DLQ
wiring — that manual `lambda invoke` / `ecs run-task` cannot), waits for
post-fire evidence, then deletes the clone. A zero-cost invocation after
a clean-tier fire (`n_calls: 0` in the run record) is the designed no-op
read.

Usage (reads the live stack; fires once; never touches the daily crons)::

    python -m scripts.trigger_manual ats|curator|clean
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import boto3

REGION = "us-east-2"
GROUP = "pipeline"
CLUSTER = "pipeline-tasks"
SOURCE = {
    "ats": "daily-ats-scrape",
    "curator": "daily-curator-scrape",
    "clean": "daily-clean",
}
LOG_GROUP = {
    "ats": "/pipeline/ats",
    "curator": "/pipeline/scrape-curator",
    "clean": "/pipeline/clean",
}


def _post_fire_evidence(tier: str, since: datetime) -> list[str]:
    """Evidence lines newer than the fire window (per-tier shape).

    Args:
        tier: Which tier fired (log group + task cluster).
        since: The clone's fire time — only events after it count.

    Returns:
        Human-read lines proving the scheduler leg actually ran.
    """
    logs = boto3.Session(region_name=REGION).client("logs")
    out: list[str] = []
    for stream in logs.describe_log_streams(
        logGroupName=LOG_GROUP[tier],
        orderBy="LastEventTime",
        descending=True,
        limit=5,
    )["logStreams"]:
        ts = stream.get("lastEventTimestamp") or stream.get("creationTime")
        when = datetime.fromtimestamp(ts / 1000, tz=UTC)
        if when > since:
            out.append(
                f"log stream {stream['logStreamName']}: last event {when:%H:%M:%SZ}"
            )
    ecs = boto3.Session(region_name=REGION).client("ecs")
    tasks = ecs.list_tasks(cluster=CLUSTER)["taskArns"]
    if tasks:
        for task in ecs.describe_tasks(cluster=CLUSTER, tasks=tasks)["tasks"]:
            started = task.get("startedAt")
            if started and started > since:
                out.append(
                    f"task {task['taskArn'].rsplit('/', 1)[-1][:12]} started "
                    f"{started:%H:%M:%SZ} lastStatus={task['lastStatus']}"
                )
    return out


def main() -> int:
    """Clone → fire once → await evidence → delete the clone.

    Raises:
        SystemExit: on an unknown tier argument.
    """
    tier = sys.argv[1] if len(sys.argv) > 1 else sys.exit(f"tiers: {sorted(SOURCE)}")
    clone_name = f"manual-{tier}"
    scheduler = boto3.Session(region_name=REGION).client("scheduler")
    fire_at = datetime.now(UTC) + timedelta(minutes=3)
    try:
        source = scheduler.get_schedule(GroupName=GROUP, Name=SOURCE[tier])
    except scheduler.exceptions.ResourceNotFoundException:
        raise SystemExit(
            f"schedule {SOURCE[tier]} missing — run the provisioner first"
        ) from None
    body: dict[str, Any] = {
        "GroupName": GROUP,
        "Name": clone_name,
        "ScheduleExpression": f"at({fire_at.strftime('%Y-%m-%dT%H:%M:%S')})",
        "ScheduleExpressionTimezone": "UTC",
        "FlexibleTimeWindow": {"Mode": "OFF"},
        "State": "ENABLED",
        "Target": {key: value for key, value in source["Target"].items()},
        "Description": f"manual one-shot clone of {SOURCE[tier]}",
    }
    try:
        scheduler.create_schedule(**body)
    except scheduler.exceptions.ConflictException:
        raise SystemExit(
            f"{clone_name} already exists — a previous fire is pending; "
            "delete it first (aws scheduler delete-schedule)"
        ) from None
    print(f"cloned {SOURCE[tier]} -> {clone_name}, firing at {fire_at:%H:%M:%S}")
    try:
        for _ in range(60):
            time.sleep(15)
            lines = _post_fire_evidence(tier, fire_at)
            if lines:
                for line in lines:
                    print("evidence:", line)
                break
        else:
            print("no post-fire evidence after 15 min — check the DLQ + alarms")
    finally:
        try:
            scheduler.delete_schedule(GroupName=GROUP, Name=clone_name)
            print(f"clone deleted: {clone_name}")
        except Exception as exc:  # noqa: BLE001 — never block on cleanup
            print(f"clone delete failed (delete by hand): {str(exc)[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
