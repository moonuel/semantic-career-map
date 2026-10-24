"""Regression tests for the compute-stack provisioner findings (review
2026-10-23): per-tier schedule targets (finding 3), the scheduler role's
DLQ statement (finding 9), the Fargate execution role (finding 10), the
Lambda image-update re-point (finding 12), the read-only ECR verify row
(finding 13 — via the pattern helper for finding 15), policy-diff warn
(finding 16), email dedupe (finding 17), and tag rendering (finding 19).
"""

from __future__ import annotations

import json

import pytest

import scripts.provision_compute_stack as pcs

ACCT = "111111111111"
REGION_OK = "us-east-2"
NETWORK = {"subnets": ["subnet-a"], "security_groups": ["sg-a"]}


class _RNFE(Exception):
    """Stand-in for every client's ResourceNotFoundException."""


class _Exceptions:
    ResourceNotFoundException = _RNFE
    NoSuchEntityException = _RNFE


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORE_S3_BUCKET", "lake-bucket")
    monkeypatch.setenv("STORE_S3_REGION", REGION_OK)
    monkeypatch.setattr(pcs, "account_id", lambda session=None: ACCT)
    monkeypatch.setattr(pcs, "default_network", lambda session: NETWORK)


class FakeScheduler:
    exceptions = _Exceptions()

    def __init__(self) -> None:
        self.created: dict[str, dict] = {}

    def get_schedule_group(self, *, Name): ...
    def get_schedule(self, *, GroupName, Name):
        raise _RNFE(Name)

    def create_schedule(self, **body):
        self.created[body["Name"]] = body["Target"]


class FakeSession:
    def __init__(self) -> None:
        self.scheduler = FakeScheduler()

    def client(self, kind: str) -> FakeScheduler:
        assert kind == "scheduler"
        return self.scheduler


def _targets(scheduler: FakeScheduler) -> dict[str, str]:
    out = {}
    for name, target in scheduler.created.items():
        if "task-definition/" in target.get("Arn", ""):
            out[name] = target["Arn"].split("task-definition/", 1)[1]
        else:
            td = target.get("EcsParameters", {}).get("TaskDefinitionArn", "")
            out[name] = td.split("task-definition/", 1)[1] if td else target["Arn"]
    return out


def test_schedules_index_their_own_task_definition(env) -> None:
    session = FakeSession()
    task_arns = {
        "curator": f"arn:aws:ecs:{REGION_OK}:{ACCT}:task-definition/pipeline-scrape-curator:1",
        "clean": f"arn:aws:ecs:{REGION_OK}:{ACCT}:task-definition/pipeline-clean:1",
    }
    lambda_arn = f"arn:aws:lambda:{REGION_OK}:{ACCT}:function:{pcs.LAMBDA_ATS}"
    cluster_arn = f"arn:aws:ecs:{REGION_OK}:{ACCT}:cluster/{pcs.CLUSTER_TASKS}"
    pcs.create_schedules(
        session,
        ACCT,
        REGION_OK,
        {"lambda": lambda_arn, "task_defs": task_arns, "cluster": cluster_arn},
        state="DISABLED",
    )
    targets = _targets(session.scheduler)
    assert targets[pcs.SCHEDULE_ATS].endswith(f"function:{pcs.LAMBDA_ATS}")
    assert targets[pcs.SCHEDULE_CURATOR].startswith("pipeline-scrape-curator")
    assert targets[pcs.SCHEDULE_CLEAN].startswith("pipeline-clean")
    for name, target in session.scheduler.created.items():
        assert target["DeadLetterConfig"]["Arn"].endswith(pcs.QUEUE_DLQ)
        if name != pcs.SCHEDULE_ATS:
            # ECS targets: Arn = cluster, task def rides EcsParameters
            assert target["Arn"] == cluster_arn
            assert target["EcsParameters"]["TaskDefinitionArn"].endswith(":1")


def test_schedules_fail_loud_on_missing_tier(env) -> None:
    session = FakeSession()
    with pytest.raises(KeyError, match="curator"):
        pcs.create_schedules(
            session,
            ACCT,
            REGION_OK,
            {
                "lambda": f"arn:aws:lambda:{REGION_OK}:{ACCT}:function:x",
                "task_defs": {},
                "cluster": f"arn:aws:ecs:{REGION_OK}:{ACCT}:cluster/{pcs.CLUSTER_TASKS}",
            },
            state="DISABLED",
        )


def test_scheduler_policy_carries_the_dlq_statement() -> None:
    policy = pcs.scheduler_policy(ACCT, REGION_OK)
    statements = [
        s for s in policy["Statement"] if s["Sid"] == "PushScheduleFailuresToDLQ"
    ]
    assert statements and statements[0]["Action"] == ["sqs:SendMessage"]
    assert statements[0]["Resource"] == [
        f"arn:aws:sqs:{REGION_OK}:{ACCT}:{pcs.QUEUE_DLQ}"
    ]


class FakeIam:
    exceptions = _Exceptions()

    def __init__(self) -> None:
        self.roles: set[str] = set()
        self.policies: dict[str, str] = {}
        self.attached: dict[str, list[str]] = {}

    def get_role(self, *, RoleName):
        if RoleName not in self.roles:
            raise _RNFE(RoleName)
        return {"Role": {"RoleName": RoleName}}

    def create_role(self, **kw):
        self.roles.add(kw["RoleName"])

    def get_role_policy(self, *, RoleName, PolicyName):
        key = f"{RoleName}/{PolicyName}"
        if key not in self.policies:
            raise _RNFE(key)
        import urllib.parse

        return {"PolicyDocument": urllib.parse.quote_plus(self.policies[key])}

    def put_role_policy(self, *, RoleName, PolicyName, PolicyDocument):
        self.policies[f"{RoleName}/{PolicyName}"] = PolicyDocument

    def list_attached_role_policies(self, *, RoleName):
        return {
            "AttachedPolicies": [
                {"PolicyArn": arn} for arn in self.attached.get(RoleName, [])
            ]
        }

    def attach_role_policy(self, *, RoleName, PolicyArn):
        self.attached.setdefault(RoleName, []).append(PolicyArn)


def test_ensure_fargate_execution_role_creates_and_attaches(env) -> None:
    iam = FakeIam()
    arn = pcs.ensure_fargate_execution_role(iam)
    assert arn == pcs.role_arn(ACCT, pcs.ROLE_FARGATE_EXECUTION)
    assert pcs.ROLE_FARGATE_EXECUTION in iam.roles
    assert iam.attached[pcs.ROLE_FARGATE_EXECUTION] == [
        "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
    ]
    # idempotent second pass: no duplicate attachment
    pcs.ensure_fargate_execution_role(iam)
    assert len(iam.attached[pcs.ROLE_FARGATE_EXECUTION]) == 1


def test_pipeline_policy_has_no_static_secrets(env) -> None:
    policy_doc = json.dumps(pcs.pipeline_policy(ACCT, REGION_OK))
    assert "STORE_S3_ACCESS_KEY_ID" not in policy_doc
    assert pcs.bucket_name() in policy_doc
    statements = {s["Sid"] for s in pcs.pipeline_policy(ACCT, REGION_OK)["Statement"]}
    assert {
        "LakeBucketReadWrite",
        "SelectionQueryRead",
        "ReadTheOneSecret",
    } <= statements


class FakeLambda:
    exceptions = _Exceptions()

    def __init__(self, exists: bool, last_status: str = "Successful") -> None:
        self._exists = exists
        self._last = last_status
        self.updated: list[dict] = []

    def get_function(self, *, FunctionName):
        if not self._exists:
            raise _RNFE(FunctionName)
        return {"Configuration": {"State": "Active"}}

    def update_function_code(self, **kw):
        self.updated.append(kw)

    def get_function_configuration(self, *, FunctionName):
        return {"LastUpdateStatus": self._last}

    def create_function(self, **kw):
        self.created = kw


def test_ensure_lambda_repoints_stale_image(env) -> None:
    region = pcs.aws_region()
    lam = FakeLambda(exists=True)
    arn = pcs.ensure_lambda(FakeLambdaSession(lam), ACCT)
    assert arn.endswith(f"function:{pcs.LAMBDA_ATS}")
    expected = pcs.image_arn(ACCT, region, pcs.IMAGE_TAG_SCRAPE)
    assert lam.updated == [{"FunctionName": pcs.LAMBDA_ATS, "ImageUri": expected}]


def test_ensure_lambda_repoints_failing_image_raises(env) -> None:
    lam = FakeLambda(exists=True, last_status="Failed")
    with pytest.raises(RuntimeError, match="rejected the image update"):
        pcs.ensure_lambda(FakeLambdaSession(lam), ACCT)


class FakeLambdaSession:
    """Session double exposing only the lambda client."""

    def __init__(self, lam) -> None:
        self._lam = lam

    def client(self, kind: str):
        assert kind == "lambda"
        return self._lam


class FakeSNS:
    def __init__(self) -> None:
        self.subscriptions: list[dict] = []
        self.subscribed: list[dict] = []

    def create_topic(self, *, Name):
        return {"TopicArn": f"arn:aws:sns:{REGION_OK}:{ACCT}:{Name}"}

    def list_subscriptions_by_topic(self, **kw):
        return {"Subscriptions": self.subscriptions, "NextToken": None}

    def subscribe(self, **kw):
        self.subscribed.append(kw)
        return {"SubscriptionArn": "pending"}


def test_ensure_topic_dedupes_subscriptions(env) -> None:
    sns = FakeSNS()
    pcs.ensure_topic(FakeSNS_Session(sns), email="ops@example.com")
    assert len(sns.subscribed) == 1
    sns.subscriptions.append(
        {
            "Protocol": "email",
            "Endpoint": "ops@example.com",
            "SubscriptionArn": "confirmed",
        }
    )
    pcs.ensure_topic(FakeSNS_Session(sns), email="ops@example.com")
    assert len(sns.subscribed) == 1  # no duplicate subscription


class FakeSNS_Session:
    def __init__(self, sns) -> None:
        self._sns = sns

    def client(self, kind: str):
        assert kind == "sns"
        return self._sns


class FakeEvents:
    exceptions = _Exceptions()

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern

    def describe_rule(self, *, Name):
        return {"EventPattern": self._pattern}


def test_failure_rule_pattern_helper_contract(env) -> None:
    class FakeEventsMatcher(FakeEvents):
        """Adds the AWS pattern matcher — the real array-semantics gate."""

        def __init__(self, pattern: str, matches: dict[int, bool]) -> None:
            super().__init__(pattern)
            self._matches = matches

        def test_event_pattern(self, *, EventPattern, Event):
            event = json.loads(Event)
            code = event["detail"]["containers"][0]["exitCode"]
            return {"Result": self._matches[code], "ErrorEntries": []}

    events = FakeEventsMatcher(
        '{"source":["aws.ecs"],"detail-type":["ECS Task State Change"],'
        '"detail":{"lastStatus":["STOPPED"],"containers":[{"exitCode":[{"anything-but":0}]}]}}',
        {1: True, 0: False},
    )
    failure, clean_ignored = pcs.test_failure_rule_pattern(events)
    assert failure is True
    assert clean_ignored is True

    hollow = FakeEventsMatcher(
        '{"source":["aws.ecs"],"detail-type":["ECS Task State Change"],'
        '"detail":{"lastStatus":["STOPPED"],"containers":[{"exitCode":[{"anything-but":0}]}]}}',
        {1: False, 0: True},
    )
    failure, clean_ignored = pcs.test_failure_rule_pattern(hollow)
    assert failure is False
    assert clean_ignored is False


def test_render_task_definitions_wires_tags_and_version(env, tmp_path) -> None:
    rendered = pcs.render_task_definitions(
        ACCT, REGION_OK, {"curator": "scrape-abc1234", "clean": "clean-def5678"}
    )
    curator = rendered["curator"]["containerDefinitions"][0]
    clean = rendered["clean"]["containerDefinitions"][0]
    assert curator["image"].endswith(":scrape-abc1234")
    assert clean["image"].endswith(":clean-def5678")
    assert curator["entryPoint"] == ["python", "scripts/scrape.py"]
    assert curator["command"] == ["--source", "dataroles"]
    version = {
        e["name"]: e["value"]
        for e in clean.get("environment", [])
        if e["name"] == "PIPELINE_VERSION"
    }
    assert version["PIPELINE_VERSION"] == "clean-def5678"
    curator_version = {
        e["name"]: e["value"]
        for e in curator.get("environment", [])
        if e["name"] == "PIPELINE_VERSION"
    }
    assert curator_version["PIPELINE_VERSION"] == "scrape-abc1234"
    rendered_default = pcs.render_task_definitions(ACCT, REGION_OK)
    assert rendered_default["clean"]["containerDefinitions"][0]["image"].endswith(
        ":clean"
    )


def test_render_task_definitions_adds_version_when_absent(env) -> None:
    """The curator def carries no static PIPELINE_VERSION env entry — the
    renderer must add it (scrape manifests record the tag, not 'unknown')."""
    rendered = pcs.render_task_definitions(ACCT, REGION_OK)
    for tier, tag in (
        ("curator", pcs.IMAGE_TAG_SCRAPE),
        ("clean", pcs.IMAGE_TAG_CLEAN),
    ):
        envs = rendered[tier]["containerDefinitions"][0]["environment"]
        assert {"name": "PIPELINE_VERSION", "value": tag} in envs


def test_render_task_definitions_rejects_unresolved_placeholders(
    env, tmp_path, monkeypatch
) -> None:
    bogus = tmp_path / "bogus-task-def.json"
    bogus.write_text('{"family": "x", "image": "<UNREPLACED>"}', encoding="utf-8")
    monkeypatch.setattr(
        pcs, "TASK_DEF_FILES", {"curator": str(bogus), "clean": str(bogus)}
    )
    with pytest.raises(ValueError, match="UNREPLACED"):
        pcs.render_task_definitions(ACCT, REGION_OK)
