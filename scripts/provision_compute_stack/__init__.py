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
import time
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
CLUSTER_TASKS = "pipeline-tasks"
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
    """What EventBridge Scheduler's execution role may do per timer fire.

    The DLQ statement is required: every target names the queue as its
    ``DeadLetterConfig`` and Scheduler delivers through this role —
    without it failed invocations are dropped silently and the
    ``pipeline-dlq-not-empty`` alarm can never fire for schedule targets.
    """
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
            {
                "Sid": "PushScheduleFailuresToDLQ",
                "Effect": "Allow",
                "Action": ["sqs:SendMessage"],
                "Resource": [f"arn:aws:sqs:{region}:{acct}:{QUEUE_DLQ}"],
            },
        ],
    }


def image_arn(acct: str, region: str, tag: str) -> str:
    """The ECR image URI for one tag (the single tag source of truth)."""
    return f"{acct}.dkr.ecr.{region}.amazonaws.com/{ECR_REPO}:{tag}"


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


def ensure_cluster(session) -> str:
    """Create-or-ignore the Fargate cluster the schedules RunTask into.

    Scheduler ECS targets demand a **cluster** ARN as the target ``Arn``
    (the task-definition ARN rides in ``EcsParameters``) — targeting the
    task definition ARN directly fails CreateSchedule with a misleading
    "Provided Arn is not in correct format".

    Returns:
        The cluster ARN.
    """
    ecs = session.client("ecs")
    try:
        response = ecs.describe_clusters(clusters=[CLUSTER_TASKS])["clusters"]
        if response:
            arn = response[0]["clusterArn"]
            print(f"cluster exists: {CLUSTER_TASKS}")
            return arn
    except ecs.exceptions.ClientError as exc:
        if "ClusterNotFound" not in str(exc):
            raise
    arn = ecs.create_cluster(clusterName=CLUSTER_TASKS)["cluster"]["clusterArn"]
    print(f"created cluster: {CLUSTER_TASKS}")
    return arn


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


def render_task_definitions(
    acct: str,
    region: str,
    image_tags: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """The task-definition JSONs with the placeholders substituted.

    The rendered payloads are exactly ``register_task_definition`` bodies
    (``family`` included). Image tags are the variables — the JSON's
    static ``:scrape``/``:clean`` image values are overwritten, and the
    ``PIPELINE_VERSION`` env is upserted (added when absent) from the
    ``image_tags`` mapping (defaults: the ``IMAGE_TAG_*`` constants;
    promotion reads pinned ``scrape:<sha>``-style tags from the
    ``--image-tags`` CLI flag).

    Args:
        acct: The AWS account id (fills the ARNs).
        region: The bucket region (fills the ECR image Uris and logs).
        image_tags: Per-tier tag override (tier name → tag).

    Returns:
        A mapping of tier name → the register payload.
    """
    tags = {
        "curator": IMAGE_TAG_SCRAPE,
        "clean": IMAGE_TAG_CLEAN,
        **(image_tags or {}),
    }
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
        if "<" in spec:
            raise ValueError(
                f"unresolved placeholders in {path}: "
                + str([token for token in spec.split('"') if token.startswith("<")])
            )
        payload: dict[str, Any] = json.loads(spec)
        container = payload["containerDefinitions"][0]
        container["image"] = image_arn(acct, region, tags[tier])
        envs = container.setdefault("environment", [])
        for env in envs:
            if env["name"] == "PIPELINE_VERSION":
                env["value"] = tags[tier]
                break
        else:
            envs.append({"name": "PIPELINE_VERSION", "value": tags[tier]})
        rendered[tier] = payload
    return rendered


# --- apply helpers (create-then-catch idempotence throughout) ---


def ensure_role(iam, name: str, services: list[str], policy: dict[str, Any]) -> str:
    """Create-or-ignore the IAM role, then reconcile its inline policy.

    On an existing role the current document is fetched and compared: an
    identical policy skips the put (no churn), a differing one warns
    before the managed version replaces any manual drift — post-provision
    tuning (the T1 policy-simulator outcome) must not quietly revert.
    A statement-less policy (execution roles carry only an attached managed
    policy) skips the inline put entirely — an empty ``Statement`` list is
    malformed to IAM.

    Args:
        iam: The boto3 IAM client.
        name: The role name.
        services: Trust-policy service principals.
        policy: The inline policy document (may be statement-less).

    Returns:
        The role ARN.
    """
    if not policy.get("Statement"):
        try:
            iam.get_role(RoleName=name)
            print(f"role exists: {name} (no inline policy)")
        except iam.exceptions.NoSuchEntityException:
            iam.create_role(
                RoleName=name,
                AssumeRolePolicyDocument=json.dumps(assume_role_document(services)),
            )
            print(f"created role: {name}")
        return role_arn(account_id(), name)
    try:
        arn = role_arn(account_id(), name)
        iam.get_role(RoleName=name)
        managed = _current_policy(iam, name)
        if managed is None or managed != policy:
            print(
                f"WARNING: managed policy for {name} differs from"
                " the on-role version — overwriting"
            )
            iam.put_role_policy(
                RoleName=name,
                PolicyName=f"{name}-policy",
                PolicyDocument=json.dumps(policy),
            )
        else:
            print(f"role exists, policy unchanged: {name}")
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=json.dumps(assume_role_document(services)),
        )
        print(f"created role: {name}")
    return arn


def _current_policy(iam, name: str) -> dict[str, Any] | None:
    """The role's current inline policy document (``None`` when absent).

    Botocore hands back ``PolicyDocument`` already URL-decoded (a dict);
    older serialized-string forms are still parsed defensively.
    """
    try:
        response = iam.get_role_policy(RoleName=name, PolicyName=f"{name}-policy")
    except iam.exceptions.NoSuchEntityException:
        return None
    document = response["PolicyDocument"]
    if isinstance(document, dict):
        return document
    from urllib.parse import unquote_plus

    return json.loads(unquote_plus(document))


def ensure_dlq(session: boto3.Session) -> str:
    """Create-or-ignore the async-failure DLQ; return its ARN."""
    sqs = session.client("sqs")
    try:
        url = sqs.get_queue_url(QueueName=QUEUE_DLQ)["QueueUrl"]
        print(f"queue exists: {QUEUE_DLQ}")
    except sqs.exceptions.QueueDoesNotExist:
        url = sqs.create_queue(
            QueueName=QUEUE_DLQ,
            Attributes={"MessageRetentionPeriod": "1209600"},
        )["QueueUrl"]
        print(f"created queue: {QUEUE_DLQ}")
    return sqs.get_queue_attributes(QueueUrl=url, AttributeNames=["QueueArn"])[
        "Attributes"
    ]["QueueArn"]


def _subscribed_endpoints(sns, topic_arn: str) -> set[str]:
    """Every (protocol, endpoint) already on the topic (any state)."""
    subscribed: set[str] = set()
    next_token = None
    while True:
        kwargs: dict[str, Any] = {"TopicArn": topic_arn}
        if next_token:
            kwargs["NextToken"] = next_token
        page = sns.list_subscriptions_by_topic(**kwargs)
        for sub in page.get("Subscriptions", []):
            subscribed.add(f"{sub['Protocol']}:{sub['Endpoint']}")
        next_token = page.get("NextToken")
        if not next_token:
            return subscribed


def ensure_topic(session: boto3.Session, email: str | None = None) -> str:
    """Create-or-ignore the alert topic; optionally subscribe an email (deduped).

    Re-subscribing a confirmed endpoint would duplicate every failure
    alert and mint a fresh pending-confirmation email on each apply — the
    (protocol, endpoint) pair gates the call.
    """
    sns = session.client("sns")
    topic = sns.create_topic(Name=TOPIC_ALERTS)["TopicArn"]
    print(f"topic ready: {TOPIC_ALERTS}")
    if email:
        key = f"email:{email}"
        if key in _subscribed_endpoints(sns, topic):
            print(f"email already subscribed: {email}")
            return topic
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


def ensure_lambda(session: boto3.Session, acct: str, tag: str | None = None) -> str:
    """Create-or-ignore the ATS Lambda function; return its ARN.

    The image URI is tag-pinned (see :func:`image_arn`) — on an existing
    function the code is re-pointed and polled so a re-apply after a
    promotion push cannot silently keep serving the stale digest.

    Args:
        session: The boto3 session.
        acct: The account id.
        tag: The ``scrape`` image tag override; defaults to
            :data:`IMAGE_TAG_SCRAPE`.

    Raises:
        RuntimeError: if the image has not been pushed yet — the T5 gate
            order is build/push (deploy/README.md) before register.
    """
    region = aws_region()
    image_uri = image_arn(acct, region, tag or IMAGE_TAG_SCRAPE)
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
            }
        },
        "DeadLetterConfig": {"TargetArn": f"arn:aws:sqs:{region}:{acct}:{QUEUE_DLQ}"},
    }
    try:
        lam.get_function(FunctionName=LAMBDA_ATS)
        print(f"lambda exists: {LAMBDA_ATS}")
        # the tag is MUTABLE (README promotion flow): re-point the function
        # at the freshly pushed digest so `--create` after a push is not a
        # silent no-op running the stale image
        lam.update_function_code(FunctionName=LAMBDA_ATS, ImageUri=image_uri)
        for _ in range(30):
            state = lam.get_function_configuration(FunctionName=LAMBDA_ATS)[
                "LastUpdateStatus"
            ]
            if state == "Successful":
                print(f"lambda image updated: {image_uri}")
                break
            if state == "Failed":
                raise RuntimeError(
                    f"lambda {LAMBDA_ATS} rejected the image update "
                    f"(LastUpdateStatus=Failed) — check the pushed :{image_uri.rsplit(':', 1)[1]} tag"
                )
            time.sleep(2)
        else:
            raise RuntimeError(
                f"lambda update did not settle after {30 * 2}s — inspect "
                f"the console (LastUpdateStatus={state})"
            )
    except lam.exceptions.ResourceNotFoundException:
        try:
            lam.create_function(**params)
            print(f"created lambda: {LAMBDA_ATS}")
            lam.put_function_event_invoke_config(
                FunctionName=LAMBDA_ATS,
                MaximumRetryAttempts=LAMBDA_RETRY_ATTEMPTS,
                MaximumEventAgeInSeconds=3600,
            )
            print(f"async retry policy set ({LAMBDA_RETRY_ATTEMPTS} retries)")
        except lam.exceptions.InvalidParameterValueException:
            raise RuntimeError(
                f"image {image_uri} missing — push it first (deploy/README.md)"
            ) from None
    return f"arn:aws:lambda:{region}:{acct}:function:{LAMBDA_ATS}"


def _register_or_skip(
    session, family: str, payload: dict[str, Any], *, force: bool = False
) -> str:
    """Register the task definition (idempotent by family revision).

    An existing revision is left alone; updating a task-definition
    revision is a T5-gate action (--reregister / ``force``), not a
    provisioner default.

    Returns:
        The task-definition ARN in ``family:revision`` form.
    """
    ecs = session.client("ecs")
    if not force:
        try:
            existing = ecs.describe_task_definition(taskDefinition=family, include=[])[
                "taskDefinition"
            ]
            print(f"task definition exists: {family} (rev {existing['revision']})")
            return existing["taskDefinitionArn"]
        except ecs.exceptions.ClientError as exc:
            message = str(exc)
            # Amazon phrases a missing family three ways across API revisions
            if not any(
                marker in message
                for marker in (
                    "Task definition not found",
                    "does not exist",
                    "Unable to describe task definition",
                )
            ):
                raise
    response = ecs.register_task_definition(**payload)
    print(
        f"registered task definition: {family} "
        f"(rev {response['taskDefinition']['revision']})"
    )
    return response["taskDefinition"]["taskDefinitionArn"]


def register_task_definitions(
    session, image_tags: dict[str, str] | None = None, *, force: bool = False
) -> dict[str, str]:
    """Register both Fargate task definitions from the ``deploy/`` JSONs.

    Args:
        session: The boto3 session.
        image_tags: Override image tag per tier (``scrape``/``clean``) —
            the promotion path for pinned per-iteration tags; defaults to
            the moving constants.
        force: Re-register as a new revision even when a current one exists
            (``--reregister``; picks up JSON changes like runtimePlatform).

    Returns:
        Mapping tier name → task-definition ARN.
    """
    rendered = render_task_definitions(account_id(), aws_region(), image_tags)
    return {
        tier: _register_or_skip(session, TASK_DEF_FAMILIES[tier], payload, force=force)
        for tier, payload in rendered.items()
    }


def ensure_schedule_group(scheduler) -> None:
    """Create-or-ignore the schedule group holding all three targets."""
    try:
        scheduler.get_schedule_group(Name=SCHEDULE_GROUP)
        print(f"schedule group exists: {SCHEDULE_GROUP}")
    except scheduler.exceptions.ResourceNotFoundException:
        scheduler.create_schedule_group(Name=SCHEDULE_GROUP)
        print(f"created schedule group: {SCHEDULE_GROUP}")


def _schedule_target(
    tier: str, acct: str, region: str, arns: dict[str, Any]
) -> dict[str, Any]:
    """One EventBridge Scheduler target body (Lambda or Fargate flavor).

    The `ats` tier runs the Lambda; every other tier is an ECS RunTask
    target: ``Arn`` is the **cluster** ARN, ``EcsParameters.TaskDefinitionArn``
    the revisioned task definition — the task-def ARN as the target Arn
    fails CreateSchedule with a misleading ARN-format error. A missing
    per-tier ``task_defs`` key raises (loud) rather than scheduling the
    wrong task definition (the finding-3 bug took the daily-clean target
    from the curator's ARN this way).
    """
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
    task_arn = arns["task_defs"][tier]  # missing tier = loud failure
    target["Arn"] = arns["cluster"]
    target["EcsParameters"] = {
        "TaskDefinitionArn": task_arn,
        "LaunchType": "FARGATE",
        "NetworkConfiguration": {
            "awsvpcConfiguration": {
                "Subnets": network["subnets"],
                "SecurityGroups": network["security_groups"],
                "AssignPublicIp": "ENABLED",
            }
        },
    }
    return target


def create_schedules(
    session, acct: str, region: str, arns: dict[str, Any], *, state: str
) -> None:
    """Create-or-ignore the three daily schedules (each created DISABLED).

    Args:
        session: The boto3 session.
        acct: The account id.
        region: The bucket region.
        arns: The ``lambda`` ARN plus the per-tier ``task_defs`` mapping
            (tagged via :func:`register_task_definitions`); the clean and
            curator schedules index their own tier's task-definition ARN.
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
        # GetSchedule returns the schedule object at top level (no wrapper)
        schedule = scheduler.get_schedule(GroupName=SCHEDULE_GROUP, Name=name)
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

    The ``containers`` match value is the bare-object form — API-proven
    (``test_event_pattern``): a numeric ``[">", 0]`` leaf under it matches
    nonzero exits and ignores exit-0 and missing-exitCode containers, while
    the array-of-object form is rejected outright (``InvalidEventPattern:
    Unrecognized match type``). A clean STOPPED (exit 0) never publishes.
    """
    events = session.client("events")
    pattern = {
        "source": ["aws.ecs"],
        "detail-type": ["ECS Task State Change"],
        "detail": {
            "lastStatus": ["STOPPED"],
            "containers": {"exitCode": [{"numeric": [">", 0]}]},
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


def ensure_fargate_execution_role(iam) -> str:
    """Create-or-ignore the Fargate execution role (task defs Require it).

    ``ecsTaskExecutionRole`` is the conventional name but it is account-
    local, not an AWS-managed guarantee — a tooling-first account has no
    ECS console history to have minted it, and the first task-def
    registration would abort mid-apply. The managed policy is exactly the
    two things the runtime needs: ECR reads + CloudWatch Logs writes.

    Returns:
        The role ARN.
    """
    name = ROLE_FARGATE_EXECUTION
    policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
    arn = ensure_role(
        iam,
        name,
        ["ecs-tasks.amazonaws.com"],
        {"Version": "2012-10-17", "Statement": []},
    )
    attached = iam.list_attached_role_policies(RoleName=name)["AttachedPolicies"]
    if not any(p["PolicyArn"] == policy_arn for p in attached):
        iam.attach_role_policy(RoleName=name, PolicyArn=policy_arn)
        print(f"attached execution policy: {name}")
    return arn


def test_failure_rule_pattern(events) -> tuple[bool, bool]:
    """Prove the task-failure rule's event pattern against both shapes.

    The ECS STOPPED matcher's array-of-object semantics are exactly the
    subtle part of the alert path: a captured failure event must MATCH
    and a clean exit must NOT (otherwise failures publish nothing and the
    alarm promise is hollow).

    Args:
        events: The boto3 events client.

    Returns:
        ``(failure_matches, clean_ignored)``.
    """

    def sample(exit_code: int) -> dict[str, Any]:
        return {
            "id": "pattern-probe",
            "source": "aws.ecs",
            "detail-type": "ECS Task State Change",
            "account": account_id(),
            "time": "2026-10-24T00:00:00Z",
            "region": aws_region(),
            "detail": {
                "lastStatus": "STOPPED",
                "stoppedReason": "Essential container in task exited",
                "containers": [{"name": "pipeline-clean", "exitCode": exit_code}],
            },
        }

    rule = events.describe_rule(Name=RULE_TASK_FAILED)
    # test_event_pattern answers with a bare boolean under "Result"
    failure = events.test_event_pattern(
        EventPattern=rule["EventPattern"], Event=json.dumps(sample(exit_code=1))
    )["Result"]
    clean = events.test_event_pattern(
        EventPattern=rule["EventPattern"], Event=json.dumps(sample(exit_code=0))
    )["Result"]
    if not failure or clean:
        print(
            f"WARNING: rule {RULE_TASK_FAILED} pattern "
            f"(failure matched={failure}, clean matched={clean}) — "
            "the alert path fires hollow; fix the pattern's arrays"
        )
    return failure, not clean


def stage_create(
    session,
    acct: str,
    region: str,
    email: str | None = None,
    image_tags: dict[str, str] | None = None,
    *,
    reregister: bool = False,
) -> None:
    """Apply the whole creation set except scheduling (which arms it).

    Order: roles first (pipeline, scheduler, sns-rule, and the Fargate
    execution role the registrations Require), then the lake plumbing
    (DLQ, topic, ECR, log groups), then the registering consumers
    (Lambda, task definitions), then the alerting + schedules
    (disabled). ``image_tags`` rides the promotion path
    (``--image-tags``; the Lambda tag key is ``scrape``).

    Args:
        session: The boto3 session.
        acct: The account id.
        region: The bucket region.
        email: Optional alert email to subscribe (deduped).
        image_tags: Optional per-tier image tag overrides.
    """
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
    ensure_fargate_execution_role(iam)
    dlq_arn = ensure_dlq(session)
    topic_arn = ensure_topic(session, email)
    ensure_ecr(session)
    ensure_log_groups(session)
    lambda_arn = ensure_lambda(session, acct, (image_tags or {}).get("scrape"))
    task_arns = register_task_definitions(session, image_tags, force=reregister)
    put_alarm(session, topic_arn)
    put_failure_rule(session, acct, region, topic_arn)
    cluster_arn = ensure_cluster(session)
    arns = {"lambda": lambda_arn, "task_defs": task_arns, "cluster": cluster_arn}
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
        for role_name in (
            ROLE_PIPELINE,
            ROLE_SCHEDULER,
            ROLE_SNS_RULE,
            ROLE_FARGATE_EXECUTION,
        ):
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
        except session.client("sqs").exceptions.QueueDoesNotExist:
            checks.append(("dlq missing", False))
        ecr = session.client("ecr")
        try:
            ecr.describe_repositories(repositoryNames=[ECR_REPO])
            checks.append(("ecr repo exists", True))
        except ecr.exceptions.RepositoryNotFoundException:
            checks.append(("ecr repo missing (push step)", False))
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
                schedule = scheduler.get_schedule(GroupName=SCHEDULE_GROUP, Name=name)
                checks.append((f"schedule {name}: {schedule.get('State')}", True))
            except scheduler.exceptions.ResourceNotFoundException:
                checks.append((f"schedule {name} missing", False))
        alarms = session.client("cloudwatch").describe_alarms(AlarmNames=[ALARM_DLQ])
        checks.append((f"alarm {ALARM_DLQ}", bool(alarms["MetricAlarms"])))
        try:
            failure_matched, clean_ignored = test_failure_rule_pattern(
                session.client("events")
            )
            checks.append(
                (
                    f"failure rule pattern {RULE_TASK_FAILED}",
                    failure_matched and clean_ignored,
                )
            )
        except Exception as exc:  # noqa: BLE001 — a broken rule is a FAIL row
            checks.append((f"failure rule: {repr(exc)[:50]}", False))
    except Exception as exc:  # noqa: BLE001 — any hard failure is a FAIL row
        checks.append((f"session/region: {repr(exc)[:60]}", False))
    for label, good in extra_checks or []:
        checks.append((label, good))
    ok = all(good for _, good in checks)
    for label, good in checks:
        print(f"{prefix}{label:<46} {'OK' if good else 'FAIL'}")
    print(f"{prefix}VERIFICATION: {'PASS' if ok else 'FAIL'}")
    return ok
