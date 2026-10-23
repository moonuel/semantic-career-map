"""Compute-stack provision and verify scripts for the AWS pipeline tiers.

The sibling of ``scripts/provision_query_layer`` (same plan-print → apply,
create-then-catch idempotence style) for everything the daily automation
rides on: IAM roles, the one SSM secret, SQS DLQ, SNS alerts, ECR, Lambda
(ATS tier), Fargate task definitions, EventBridge Scheduler schedules
(DISABLED until smoke gates pass), and the failure alarms
(`docs/architecture/cleaning-automation-report.md` §4.2–4.6; roadmap
Phases 3a/3b/4 + the 4.5 cleaning tier).

Environment contract (loaded from ``.env`` / task env): ``STORE_S3_BUCKET``
and ``STORE_S3_REGION`` name the lake and fix the region —
``ObjectStore.from_env`` already requires both, and one region is a
roadmap §2 guardrail. Region resolution reuses the query layer's helper
(env-authoritative, bucket-suffix fallback).

Usage::

    python -m scripts.provision_compute_stack --create --dry-run   # plan only
    python -m scripts.provision_compute_stack --create             # apply
    python -m scripts.provision_compute_stack --put-secret [--overwrite]
    python -m scripts.provision_compute_stack --schedule [--enable]
    python -m scripts.provision_compute_stack --enable
    python -m scripts.provision_compute_stack --email you@example.com
    python -m scripts.provision_compute_stack --verify
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3

from scripts.provision_query_layer import aws_region

ECR_REPO = "pipeline-images"
IMAGE_TAG_SCRAPE = "scrape"
IMAGE_TAG_CLEAN = "clean"

ROLE_PIPELINE = "pipeline-execution-role"
ROLE_SCHEDULER = "pipeline-scheduler-role"
ROLE_SNS_RULE = "pipeline-sns-rule-role"
ROLE_FARGATE_EXECUTION = "ecsTaskExecutionRole"  # AWS-managed for Fargate

QUEUE_DLQ = "pipeline-ats-dlq"
TOPIC_ALERTS = "pipeline-alerts"

LAMBDA_ATS = "pipeline-ats-scrape"
LAMBDA_MEMORY_MB = 1024
LAMBDA_TIMEOUT_S = 300
LAMBDA_RETRY_ATTEMPTS = 2
LOG_GROUP_ATS = "/pipeline/ats"
LOG_GROUP_CURATOR = "/pipeline/scrape-curator"
LOG_GROUP_CLEAN = "/pipeline/clean"
LOG_RETENTION_DAYS = 30
ALARM_DLQ = "pipeline-dlq-not-empty"
RULE_TASK_FAILED = "pipeline-task-failed"

SCHEDULE_GROUP = "pipeline"
SCHEDULE_ATS = "daily-ats-scrape"
SCHEDULE_CURATOR = "daily-curator-scrape"
SCHEDULE_CLEAN = "daily-clean"
CRON_ATS = "cron(0 4 * * ? *)"
CRON_CURATOR = "cron(0 5 * * ? *)"
CRON_CLEAN = "cron(0 6 * * ? *)"  # offset after the scrape tiers; no chain

SSM_SECRET_PATH = "/semantic-career-map/KILO_API_KEY"
TASK_DEF_FILES = {
    "curator": "deploy/task-definition-scrape-curator.json",
    "clean": "deploy/task-definition-clean.json",
}
TASK_DEF_FAMILIES = {
    "curator": "pipeline-scrape-curator",
    "clean": "pipeline-clean",
}
SCHEDULE_SPECS = [
    (SCHEDULE_ATS, CRON_ATS, "ats"),
    (SCHEDULE_CURATOR, CRON_CURATOR, "curator"),
    (SCHEDULE_CLEAN, CRON_CLEAN, "clean"),
]


def bucket_name() -> str:
    """The lake bucket from the store environment (``ObjectStore`` contract)."""
    bucket = os.environ.get("STORE_S3_BUCKET")
    if not bucket:
        raise RuntimeError("STORE_S3_BUCKET is not set (load .env or task env)")
    return bucket


def _session() -> boto3.Session:
    return boto3.Session(region_name=aws_region())


def account_id(session: boto3.Session | None = None) -> str:
    """The caller's 12-digit account id (the ARN root for every policy)."""
    session = session or _session()
    return session.client("sts").get_caller_identity()["Account"]


def bucket_arn(bucket: str) -> str:
    """The bucket's policy-document ARN."""
    return f"arn:aws:s3:::{bucket}"


def role_arn(acct: str, role_name: str) -> str:
    """An IAM role ARN."""
    return f"arn:aws:iam::{acct}:role/{role_name}"


def assume_role_document(services: list[str]) -> dict[str, Any]:
    """A trust policy for the given service principals (one statement each)."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": [service]},
                "Action": "sts:AssumeRole",
            }
            for service in services
        ],
    }


def pipeline_policy(acct: str, region: str) -> dict[str, Any]:
    """The least-privilege policy the compute tiers carry at runtime.

    Scope: the lake bucket (S3), CloudWatch Logs, the DLQ (Lambda async
    failures), the Athena/Glue read surface the daily driver needs for
    selection, and the one SSM secret path. Everything else is implicit
    deny — the policy-simulator run is the T1 gate.
    """
    bucket = bucket_name()
    statements: list[dict[str, Any]] = [
        {
            "Sid": "LakeBucketReadWrite",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:GetObjectVersion",
                "s3:PutObject",
                "s3:AbortMultipartUpload",
            ],
            "Resource": [f"{bucket_arn(bucket)}/*"],
        },
        {
            "Sid": "LakeBucketList",
            "Effect": "Allow",
            "Action": ["s3:ListBucket"],
            "Resource": [bucket_arn(bucket)],
        },
        {
            "Sid": "CloudWatchLogsWrite",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogStreams",
            ],
            "Resource": [f"arn:aws:logs:{region}:{acct}:log-group:/pipeline/*"],
        },
        {
            "Sid": "SelectionQueryRead",
            "Effect": "Allow",
            "Action": [
                "athena:StartQueryExecution",
                "athena:StopQueryExecution",
                "athena:GetQueryExecution",
                "athena:GetQueryResults",
                "glue:GetDatabase",
                "glue:GetDatabases",
                "glue:GetTable",
                "glue:GetTables",
                "glue:GetPartition",
                "glue:GetPartitions",
            ],
            "Resource": ["*"],
        },
        {
            "Sid": "PushFailuresToDLQ",
            "Effect": "Allow",
            "Action": ["sqs:SendMessage"],
            "Resource": [f"arn:aws:sqs:{region}:{acct}:{QUEUE_DLQ}"],
        },
        {
            "Sid": "ReadTheOneSecret",
            "Effect": "Allow",
            "Action": ["ssm:GetParameter", "ssm:GetParameters"],
            "Resource": [
                f"arn:aws:ssm:{region}:{acct}:parameter/semantic-career-map/*"
            ],
        },
    ]
    return {"Version": "2012-10-17", "Statement": statements}


def scheduler_policy(acct: str, region: str) -> dict[str, Any]:
    """What EventBridge Scheduler's execution role may do per timer fire."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeScrapeLambda",
                "Effect": "Allow",
                "Action": ["lambda:InvokeFunction"],
                "Resource": [f"arn:aws:lambda:{region}:{acct}:function:{LAMBDA_ATS}"],
            },
            {
                "Sid": "RunFargateTasks",
                "Effect": "Allow",
                "Action": ["ecs:RunTask"],
                "Resource": [f"arn:aws:ecs:{region}:{acct}:task-definition/pipeline-*"],
            },
            {
                "Sid": "PassTaskAndExecutionRoles",
                "Effect": "Allow",
                "Action": ["iam:PassRole"],
                "Resource": [
                    role_arn(acct, ROLE_PIPELINE),
                    role_arn(acct, ROLE_FARGATE_EXECUTION),
                ],
            },
        ],
    }


def sns_rule_policy(acct: str, region: str) -> dict[str, Any]:
    """Publish permission for the task-`STOPPED`-failure rule target."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PublishTaskFailures",
                "Effect": "Allow",
                "Action": ["sns:Publish"],
                "Resource": [f"arn:aws:sns:{region}:{acct}:{TOPIC_ALERTS}"],
            }
        ],
    }


def default_network(session: boto3.Session) -> dict[str, list[str]]:
    """Subnets + security groups of the default VPC (Fargate networking).

    The Phase 3b contract: public subnets only, auto-assigned public IP,
    no NAT. A missing default VPC is a stop-and-create-manually condition,
    not something to invent here.

    Returns:
        ``{"subnets": [...], "security_groups": [...]}``.

    Raises:
        RuntimeError: if the account carries no default VPC / default SG.
    """
    ec2 = session.client("ec2")
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])[
        "Vpcs"
    ]
    if not vpcs:
        raise RuntimeError(
            "no default VPC in the region; create one before scheduling tasks"
        )
    vpc_id = vpcs[0]["VpcId"]
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])[
        "Subnets"
    ]
    groups = ec2.describe_security_groups(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "group-name", "Values": ["default"]},
        ]
    )["SecurityGroups"]
    if not groups:
        raise RuntimeError(f"default security group missing in {vpc_id}")
    return {
        "subnets": [s["SubnetId"] for s in subnets],
        "security_groups": [g["GroupId"] for g in groups],
    }


def stage_plan(acct: str, region: str) -> dict[str, Any]:
    """The blueprint printed before any use of the credentials."""
    return {
        "account_id": acct,
        "region": region,
        "bucket": bucket_name(),
        "roles": {
            "pipeline": role_arn(acct, ROLE_PIPELINE),
            "scheduler": role_arn(acct, ROLE_SCHEDULER),
            "sns_rule": role_arn(acct, ROLE_SNS_RULE),
            "fargate_execution": role_arn(acct, ROLE_FARGATE_EXECUTION)
            + " (AWS-managed)",
        },
        "ssm_secret": SSM_SECRET_PATH,
        "queue_dlq": f"arn:aws:sqs:{region}:{acct}:{QUEUE_DLQ}",
        "topic_alerts": f"arn:aws:sns:{region}:{acct}:{TOPIC_ALERTS}",
        "ecr_repo": f"arn:aws:ecr:{region}:{acct}:repository/{ECR_REPO}",
        "log_groups": [LOG_GROUP_ATS, LOG_GROUP_CURATOR, LOG_GROUP_CLEAN],
        "lambda": {
            "name": LAMBDA_ATS,
            "image": f"{acct}.dkr.ecr.{region}.amazonaws.com/{ECR_REPO}:{IMAGE_TAG_SCRAPE}",
            "memory_mb": LAMBDA_MEMORY_MB,
            "timeout_s": LAMBDA_TIMEOUT_S,
            "retry_attempts": LAMBDA_RETRY_ATTEMPTS,
            "dlq": QUEUE_DLQ,
        },
        "task_definitions": TASK_DEF_FILES,
        "schedules": {
            "group": SCHEDULE_GROUP,
            SCHEDULE_ATS: {"when": CRON_ATS, "target": LAMBDA_ATS},
            SCHEDULE_CURATOR: {
                "when": CRON_CURATOR,
                "target": TASK_DEF_FAMILIES["curator"],
            },
            SCHEDULE_CLEAN: {"when": CRON_CLEAN, "target": TASK_DEF_FAMILIES["clean"]},
            "state_at_creation": "DISABLED (arm via --enable after smoke gates)",
        },
        "alarm": ALARM_DLQ,
        "failure_rule": RULE_TASK_FAILED,
    }


def render_task_definitions(acct: str, region: str) -> dict[str, dict[str, Any]]:
    """The task-definition JSONs with the placeholders substituted.

    The rendered payloads are exactly ``register_task_definition`` bodies
    (``family`` included; the API takes it as the named argument).

    Args:
        acct: The AWS account id (fills the ARNs).
        region: The bucket region (fills the ECR image Uris and logs).

    Returns:
        A mapping of tier name → the register payload.
    """
    rendered: dict[str, dict[str, Any]] = {}
    for tier, path in TASK_DEF_FILES.items():
        with open(path, encoding="utf-8") as f:
            spec = f.read()
        for old, new in (
            ("<ACCOUNT_ID>", acct),
            ("<REGION>", region),
            ("<STORE_S3_BUCKET>", bucket_name()),
            ("<PIPELINE_ROLE_NAME>", ROLE_PIPELINE),
        ):
            spec = spec.replace(old, new)
        payload: dict[str, Any] = json.loads(spec)
        if "<" in json.dumps(payload):
            outstanding = [
                token
                for token in json.dumps(payload).split('"')
                if token.startswith("<")
            ]
            raise ValueError(f"unresolved placeholders in {path}: {outstanding}")
        rendered[tier] = payload
    return rendered


# --- apply helpers (create-then-catch idempotence throughout) ---


def ensure_role(iam, name: str, services: list[str], policy: dict[str, Any]) -> str:
    """Create-or-ignore the IAM role, then upsert its inline policy.

    Args:
        iam: The boto3 IAM client.
        name: The role name.
        services: Trust-policy service principals.
        policy: The inline policy document.

    Returns:
        The role ARN.
    """
    try:
        arn = role_arn(account_id(), name)
        iam.get_role(RoleName=name)
        print(f"role exists: {name}")
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=json.dumps(assume_role_document(services)),
        )
        print(f"created role: {name}")
    iam.put_role_policy(
        RoleName=name,
        PolicyName=f"{name}-policy",
        PolicyDocument=json.dumps(policy),
    )
    return arn


def ensure_dlq(session: boto3.Session) -> str:
    """Create-or-ignore the async-failure DLQ; return its ARN."""
    sqs = session.client("sqs")
    try:
        url = sqs.get_queue_url(QueueName=QUEUE_DLQ)["QueueUrl"]
        print(f"queue exists: {QUEUE_DLQ}")
    except sqs.exceptions.QueueDoesNotExistError:
        url = sqs.create_queue(
            QueueName=QUEUE_DLQ,
            Attributes={"MessageRetentionPeriod": "1209600"},
        )["QueueUrl"]
        print(f"created queue: {QUEUE_DLQ}")
    return sqs.get_queue_attributes(QueueUrl=url, AttributeNames=["QueueArn"])[
        "Attributes"
    ]["QueueArn"]


def ensure_topic(session: boto3.Session, email: str | None = None) -> str:
    """Create-or-ignore the alert topic; optionally subscribe an email."""
    sns = session.client("sns")
    topic = sns.create_topic(Name=TOPIC_ALERTS)["TopicArn"]
    print(f"topic ready: {TOPIC_ALERTS}")
    if email:
        confirmation = sns.subscribe(TopicArn=topic, Protocol="email", Endpoint=email)
        print(
            "subscription sent — confirm the email in the inbox"
            f" ({confirmation['SubscriptionArn']})"
        )
    return topic


def ensure_ecr(session: boto3.Session) -> str:
    """Create-or-ignore the image repo; return its registry URI."""
    ecr = session.client("ecr")
    print(f"ecr repo: {ECR_REPO}")
    try:
        repo = ecr.describe_repositories(repositoryNames=[ECR_REPO])
        print(f"ecr exists: {ECR_REPO}")
    except ecr.exceptions.RepositoryNotFoundException:
        ecr.create_repository(
            repositoryName=ECR_REPO,
            imageTagMutability="MUTABLE",
            imageScanningConfiguration={"scanOnPush": True},
        )
        print(f"created ecr: {ECR_REPO}")
        repo = ecr.describe_repositories(repositoryNames=[ECR_REPO])
    return repo["repositories"][0]["repositoryUri"]


def ensure_log_groups(session: boto3.Session) -> None:
    """Create-or-ignore the three log groups with retention pinned."""
    logs = session.client("logs")
    for group in (LOG_GROUP_ATS, LOG_GROUP_CURATOR, LOG_GROUP_CLEAN):
        try:
            logs.create_log_group(logGroupName=group)
            print(f"created log group: {group}")
        except logs.exceptions.ResourceAlreadyExistsException:
            print(f"log group exists: {group}")
        logs.put_retention_policy(
            logGroupName=group, retentionInDays=LOG_RETENTION_DAYS
        )


def ensure_lambda(session: boto3.Session, acct: str) -> str:
    """Create-or-ignore the ATS Lambda function; return its ARN.

    Raises:
        RuntimeError: if the ``scrape`` image has not been pushed yet —
            the T5 gate order is build/push (deploy/README.md) before
            register.
    """
    region = aws_region()
    image_uri = f"{acct}.dkr.ecr.{region}.amazonaws.com/{ECR_REPO}:{IMAGE_TAG_SCRAPE}"
    lam = session.client("lambda")
    params: dict[str, Any] = {
        "FunctionName": LAMBDA_ATS,
        "Description": "Whole-board ATS scrape tier (image CMD = the ATS run)",
        "PackageType": "Image",
        "Code": {"ImageUri": image_uri},
        "Role": role_arn(acct, ROLE_PIPELINE),
        "Architectures": ["arm64"],
        "MemorySize": LAMBDA_MEMORY_MB,
        "Timeout": LAMBDA_TIMEOUT_S,
        "Environment": {
            "Variables": {
                "STORE_S3_BUCKET": bucket_name(),
                "STORE_S3_REGION": region,
                "PIPELINE_VERSION": IMAGE_TAG_SCRAPE,
            }
        },
        "DeadLetterConfig": {"TargetArn": f"arn:aws:sqs:{region}:{acct}:{QUEUE_DLQ}"},
        "MaximumRetryAttempts": LAMBDA_RETRY_ATTEMPTS,
        "MaximumEventAgeInSeconds": 3600,
    }
    try:
        lam.get_function(FunctionName=LAMBDA_ATS)
        print(f"lambda exists: {LAMBDA_ATS}")
    except lam.exceptions.ResourceNotFoundException:
        try:
            lam.create_function(**params)
            print(f"created lambda: {LAMBDA_ATS}")
        except lam.exceptions.InvalidParameterValueException:
            raise RuntimeError(
                f"image {image_uri} missing — push it first (deploy/README.md)"
            ) from None
    return f"arn:aws:lambda:{region}:{acct}:function:{LAMBDA_ATS}"


def _register_or_skip(session, family: str, payload: dict[str, Any]) -> str:
    """Register the task definition (idempotent by family revision).

    An existing revision is left alone; updating a task-definition
    revision is a T5-gate action, not a provisioner default.

    Returns:
        The task-definition ARN in ``family:revision`` form.
    """
    ecs = session.client("ecs")
    try:
        existing = ecs.describe_task_definition(taskDefinition=family, include=[])[
            "taskDefinition"
        ]
        print(f"task definition exists: {family} (rev {existing['revision']})")
        return existing["taskDefinitionArn"]
    except ecs.exceptions.ClientError as exc:
        if "does not exist" not in str(exc):
            raise
    response = ecs.register_task_definition(**payload)
    print(
        f"registered task definition: {family} "
        f"(rev {response['taskDefinition']['revision']})"
    )
    return response["taskDefinition"]["taskDefinitionArn"]


def register_task_definitions(session) -> dict[str, str]:
    """Register both Fargate task definitions from the ``deploy/`` JSONs.

    Returns:
        Mapping tier name → task-definition ARN.
    """
    rendered = render_task_definitions(account_id(), aws_region())
    return {
        tier: _register_or_skip(session, TASK_DEF_FAMILIES[tier], payload)
        for tier, payload in rendered.items()
    }


def ensure_schedule_group(scheduler) -> None:
    """Create-or-ignore the schedule group holding all three targets."""
    try:
        scheduler.get_schedule_group(GroupName=SCHEDULE_GROUP)
        print(f"schedule group exists: {SCHEDULE_GROUP}")
    except scheduler.exceptions.ResourceNotFoundException:
        scheduler.create_schedule_group(Name=SCHEDULE_GROUP)
        print(f"created schedule group: {SCHEDULE_GROUP}")


def _schedule_target(
    tier: str, acct: str, region: str, arns: dict[str, str]
) -> dict[str, Any]:
    """One EventBridge Scheduler target body (Lambda or Fargate flavor)."""
    target: dict[str, Any] = {
        "RoleArn": role_arn(acct, ROLE_SCHEDULER),
        "RetryPolicy": {
            "MaximumRetryAttempts": LAMBDA_RETRY_ATTEMPTS,
            "MaximumEventAgeInSeconds": 3600,
        },
        "DeadLetterConfig": {"Arn": f"arn:aws:sqs:{region}:{acct}:{QUEUE_DLQ}"},
    }
    if tier == "ats":
        target["Arn"] = arns["lambda"]
        return target
    network = default_network(boto3.Session(region_name=aws_region()))
    target["Arn"] = arns["task_definition"]
    target["EcsParameters"] = {
        "TaskDefinitionArn": arns["task_definition"],
        "LaunchType": "FARGATE",
        "NetworkConfiguration": {
            "awsvpcConfiguration": {
                "subnets": network["subnets"],
                "securityGroups": network["security_groups"],
                "assignPublicIp": "ENABLED",
            }
        },
    }
    return target


def create_schedules(
    session, acct: str, region: str, arns: dict[str, str], *, state: str
) -> None:
    """Create-or-ignore the three daily schedules (each created DISABLED).

    Args:
        session: The boto3 session.
        acct: The account id.
        region: The bucket region.
        arns: The ``lambda`` and ``task_definition`` target ARNs.
        state: ``DISABLED`` (smoke-first contract) or ``ENABLED``.
    """
    scheduler = session.client("scheduler")
    ensure_schedule_group(scheduler)
    for name, cron, tier in SCHEDULE_SPECS:
        try:
            scheduler.get_schedule(GroupName=SCHEDULE_GROUP, Name=name)
            print(f"schedule exists: {name}")
            continue
        except scheduler.exceptions.ResourceNotFoundException:
            pass
        scheduler.create_schedule(
            Name=name,
            GroupName=SCHEDULE_GROUP,
            ScheduleExpression=cron,
            ScheduleExpressionTimezone="UTC",
            FlexibleTimeWindow={"Mode": "OFF"},
            State=state,
            Target=_schedule_target(tier, acct, region, arns),
        )
        print(f"created schedule: {name} [{state}]")


def enable_schedules(session) -> None:
    """Flip all three schedules to ``ENABLED`` (the T5 step-6 arming)."""
    scheduler = session.client("scheduler")
    for name, _, _ in SCHEDULE_SPECS:
        schedule = scheduler.get_schedule(GroupName=SCHEDULE_GROUP, Name=name)[
            "Schedule"
        ]
        if schedule.get("State") == "ENABLED":
            print(f"schedule already enabled: {name}")
            continue
        scheduler.update_schedule(
            GroupName=SCHEDULE_GROUP,
            Name=name,
            State="ENABLED",
            ScheduleExpression=schedule["ScheduleExpression"],
            ScheduleExpressionTimezone=schedule["ScheduleExpressionTimezone"],
            FlexibleTimeWindow=schedule["FlexibleTimeWindow"],
            Target=schedule["Target"],
        )
        print(f"schedule enabled: {name}")


def put_alarm(session, topic_arn: str) -> None:
    """Alarm on DLQ depth (non-empty = the ATS tier failed its async run).

    ``put_metric_alarm`` upserts — safe to repeat.
    """
    session.client("cloudwatch").put_metric_alarm(
        AlarmName=ALARM_DLQ,
        Namespace="AWS/SQS",
        MetricName="ApproximateNumberOfMessagesVisible",
        Dimensions=[{"Name": "QueueName", "Value": QUEUE_DLQ}],
        Statistic="Average",
        Period=300,
        EvaluationPeriods=1,
        Threshold=0,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=[topic_arn],
        AlarmDescription="Non-empty ATS DLQ — the async tier is failing",
    )
    print(f"alarm ready: {ALARM_DLQ}")


def put_failure_rule(session, acct: str, region: str, topic_arn: str) -> None:
    """Bridge ECS task STOPPED-with-nonzero-exit onto SNS (curator/clean).

    ``anything-but:0`` matches failure exit codes without listing them; a
    clean STOPPED (exit 0) never publishes.
    """
    events = session.client("events")
    pattern = {
        "source": ["aws.ecs"],
        "detail-type": ["ECS Task State Change"],
        "detail": {
            "lastStatus": ["STOPPED"],
            "containers": [{"exitCode": [{"anything-but": 0}]}],
        },
    }
    events.put_rule(
        Name=RULE_TASK_FAILED,
        EventPattern=json.dumps(pattern),
        State="ENABLED",
        EventBusName="default",
        Description="Non-zero-exit Fargate pipeline tasks (curator/clean)",
    )
    events.put_targets(
        Rule=RULE_TASK_FAILED,
        EventBusName="default",
        Targets=[
            {
                "Id": "alerts",
                "Arn": topic_arn,
                "RoleArn": role_arn(acct, ROLE_SNS_RULE),
            }
        ],
    )
    print(f"failure rule ready: {RULE_TASK_FAILED}")


def put_secret(session, value: str, *, overwrite: bool = False) -> None:
    """Write the LLM key as the one SecureString (value never printed).

    Args:
        session: The boto3 session.
        value: The KILO_API_KEY value (read from the environment by the
            CLI; never logged).
        overwrite: SSM's default fails on an existing parameter unless
            set; rotation is manual by design.
    """
    session.client("ssm").put_parameter(
        Name=SSM_SECRET_PATH,
        Type="SecureString",
        Value=value,
        Overwrite=overwrite,
    )
    print(f"secret written: {SSM_SECRET_PATH} (value not printed)")


def stage_create(session, acct: str, region: str, email: str | None = None) -> None:
    """Apply the whole creation set except scheduling (which arms it)."""
    iam = session.client("iam")
    ensure_role(
        iam,
        ROLE_PIPELINE,
        ["lambda.amazonaws.com", "ecs-tasks.amazonaws.com"],
        pipeline_policy(acct, region),
    )
    ensure_role(
        iam,
        ROLE_SCHEDULER,
        ["scheduler.amazonaws.com"],
        scheduler_policy(acct, region),
    )
    ensure_role(
        iam,
        ROLE_SNS_RULE,
        ["events.amazonaws.com"],
        sns_rule_policy(acct, region),
    )
    dlq_arn = ensure_dlq(session)
    topic_arn = ensure_topic(session, email)
    ensure_ecr(session)
    ensure_log_groups(session)
    lambda_arn = ensure_lambda(session, acct)
    task_arns = register_task_definitions(session)
    put_alarm(session, topic_arn)
    put_failure_rule(session, acct, region, topic_arn)
    arns = {"lambda": lambda_arn, "task_definition": task_arns["curator"]}
    create_schedules(session, acct, region, arns, state="DISABLED")
    print(f"dlq arn: {dlq_arn}")


def verify(
    extra_checks: list[tuple[str, bool]] | None = None,
    prefix: str = "  ",
) -> bool:
    """Probe every provisioned resource; print OK/FAIL per check.

    Args:
        extra_checks: Caller-added rows (appended to the printed list).
        prefix: Print indentation (the CLI prints one indent).

    Returns:
        True when every check passed.
    """
    checks: list[tuple[str, bool]] = []
    try:
        session = _session()
        iam = session.client("iam")
        for role_name in (ROLE_PIPELINE, ROLE_SCHEDULER, ROLE_SNS_RULE):
            try:
                found = iam.get_role(RoleName=role_name)["Role"]
                checks.append((f"role {role_name}", bool(found)))
            except iam.exceptions.NoSuchEntityException:
                checks.append((f"role {role_name} missing", False))
        ssm_ready = True
        try:
            session.client("ssm").get_parameter(Name=SSM_SECRET_PATH)
        except session.client("ssm").exceptions.ParameterNotFound:
            ssm_ready = False
        checks.append((f"ssm secret {SSM_SECRET_PATH}", ssm_ready))
        try:
            url = session.client("sqs").get_queue_url(QueueName=QUEUE_DLQ)["QueueUrl"]
            checks.append(("dlq exists", bool(url)))
        except session.client("sqs").exceptions.QueueDoesNotExistError:
            checks.append(("dlq missing", False))
        ensure_ecr(session)
        checks.append(("ecr repo", True))
        try:
            lam_state = session.client("lambda").get_function(FunctionName=LAMBDA_ATS)[
                "Configuration"
            ]["State"]
            checks.append(
                (f"lambda {LAMBDA_ATS} state={lam_state}", lam_state == "Active")
            )
        except session.client("lambda").exceptions.ResourceNotFoundException:
            checks.append((f"lambda {LAMBDA_ATS} missing", False))
        ecs = session.client("ecs")
        for tier, family in TASK_DEF_FAMILIES.items():
            try:
                defn = ecs.describe_task_definition(taskDefinition=family, include=[])[
                    "taskDefinition"
                ]
                checks.append((f"task def {tier}", defn["status"] == "ACTIVE"))
            except ecs.exceptions.ClientError as exc:
                checks.append((f"task def {tier} missing: {str(exc)[:40]}", False))
        scheduler = session.client("scheduler")
        for name, _, _ in SCHEDULE_SPECS:
            try:
                schedule = scheduler.get_schedule(GroupName=SCHEDULE_GROUP, Name=name)[
                    "Schedule"
                ]
                checks.append((f"schedule {name}: {schedule.get('State')}", True))
            except scheduler.exceptions.ResourceNotFoundException:
                checks.append((f"schedule {name} missing", False))
        alarms = session.client("cloudwatch").describe_alarms(AlarmNames=[ALARM_DLQ])
        checks.append((f"alarm {ALARM_DLQ}", bool(alarms["MetricAlarms"])))
    except Exception as exc:  # noqa: BLE001 — any hard failure is a FAIL row
        checks.append((f"session/region: {repr(exc)[:60]}", False))
    for label, good in extra_checks or []:
        checks.append((label, good))
    ok = all(good for _, good in checks)
    for label, good in checks:
        print(f"{prefix}{label:<46} {'OK' if good else 'FAIL'}")
    print(f"{prefix}VERIFICATION: {'PASS' if ok else 'FAIL'}")
    return ok
