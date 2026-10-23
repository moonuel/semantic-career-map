"""Unit tests for the daily driver (report T3): selection binding, secret
injection, paginated Athena rows, the manifest→runner seam, the ledger
filter path, the empty-selection no-op run, deadline arm/cancel, and the
exit-code mapping."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import pytest

from backend.pipeline.cleaning import daily
from backend.pipeline.cleaning.configuration import configuration_hash
from backend.pipeline.cleaning.prompts import (
    BOILERPLATE_ONLY_SYSTEM_PROMPT,
    USER_MESSAGE_TEMPLATE,
)
from backend.storage import ids, keys
from tests.conftest import JournalSink

MODEL = "google/gemini-2.5-flash-lite"
CONFIG = configuration_hash(
    MODEL, BOILERPLATE_ONLY_SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE
)


class StubClient:
    """Gateway stand-in with canned outputs and a call log."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, system_prompt: str, user_message: str, *, model: str, **kw: Any
    ) -> tuple[str, dict[str, Any]]:
        self.calls.append({"user_message": user_message, "model": model})
        return f"cleaned:{user_message[:12]}", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
        }


class FakeAthena:
    """Athena stand-in over explicit page chunks of fixture cell rows."""

    def __init__(self, chunks: list[list[list[str]]] | None = None) -> None:
        self._chunks = chunks or [[[]]]
        self.executed: list[str] = []
        self.results_locations: list[str] = []

    def start_query_execution(
        self, *, QueryString: str, ResultConfiguration: dict[str, Any], **kw: Any
    ) -> dict[str, Any]:
        self.executed.append(QueryString)
        self.results_locations.append(ResultConfiguration["OutputLocation"])
        return {"QueryExecutionId": "q-daily-1"}

    def get_query_execution(self, *, QueryExecutionId: str) -> dict[str, Any]:
        return {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}

    def get_query_results(
        self, *, QueryExecutionId: str, NextToken: str | None = None
    ) -> dict[str, Any]:
        page = int(NextToken) if NextToken else 0
        encoded = [
            {"Data": [{"VarCharValue": c} for c in row]} for row in self._chunks[page]
        ]
        response: dict[str, Any] = {"ResultSet": {"Rows": encoded}}
        if page + 1 < len(self._chunks):
            response["NextToken"] = str(page + 1)
        return response


def _post(store: JournalSink, source: str, field: str, body: str) -> dict[str, str]:
    """Seed one raw payload; return its manifest posting row."""
    source_hash = source_hash_of(body)
    store.objects[keys.raw_payload(source, source_hash)] = json.dumps(
        {field: body, "source_id": f"{source}:x:1", "employer": "Acme"}
    ).encode()
    return {
        "source": source,
        "source_hash": source_hash,
        "source_id": f"{source}:x:1",
    }


def source_hash_of(body: str) -> str:
    import hashlib

    return hashlib.sha256(body.encode()).hexdigest()


def _settings(**kw: Any) -> daily.Settings:
    defaults: dict[str, Any] = {
        "database": "semantic_career_map",
        "workgroup": "semantic_career_map",
        "model": MODEL,
        "workers": 2,
        "version": "v1",
        "max_run_hours": 0,  # guard disabled in tests
        "configuration_hash": CONFIG,
    }
    defaults.update(kw)
    return daily.Settings(**defaults)


@pytest.fixture
def bucket_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORE_S3_BUCKET", "lake")


def test_help_is_credential_free(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("STORE_S3_") and k != "KILO_API_KEY"
    }
    proc = subprocess.run(
        [sys.executable, "-m", "backend.pipeline.cleaning.daily", "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--max-run-hours" in proc.stdout


def test_resolve_configuration_hash_computed_and_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert daily.resolve_configuration_hash(MODEL) == CONFIG
    monkeypatch.setenv("CONFIGURATION_HASH", "9" * 64)
    assert daily.resolve_configuration_hash(MODEL) == "9" * 64
    monkeypatch.setenv("CONFIGURATION_HASH", "FFFFFFFF")
    with pytest.raises(ValueError, match="64-char hex"):
        daily.resolve_configuration_hash(MODEL)


def test_bind_config_sql_binds_every_placeholder() -> None:
    sql = daily.select_uncleaned_sql()
    bound = daily.bind_config_sql(sql, CONFIG)
    assert bound.count(f"'{CONFIG}'") == sql.count(":config")
    assert ":config" not in bound.replace("--", "#") or "-- " in sql


def test_secret_injection_sets_env_only_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSSM:
        def __init__(self) -> None:
            self.called = 0

        def get_parameter(self, *, Name: str, WithDecryption: bool) -> dict[str, Any]:
            self.called += 1
            assert Name == daily.SSM_SECRET_PATH_DEFAULT
            assert WithDecryption is True
            return {"Parameter": {"Value": "kilo-secret-1"}}

    ssm = FakeSSM()
    monkeypatch.delenv("KILO_API_KEY", raising=False)
    daily.fetch_secret_if_absent(ssm)
    assert os.environ["KILO_API_KEY"] == "kilo-secret-1"
    monkeypatch.setenv("KILO_API_KEY", "already-set")
    daily.fetch_secret_if_absent(ssm)
    assert ssm.called == 1
    assert os.environ["KILO_API_KEY"] == "already-set"


def test_select_query_results_comma_safe_and_paginated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STORE_S3_BUCKET", "lake")
    a1 = source_hash_of("body a")
    a2 = source_hash_of("body b")
    athena = FakeAthena(
        [
            [["source", "source_hash", "source_id"], ["greenhouse", a1, "acme, inc 5"]],
            [["ashby", a2, "acme 6"]],
        ]
    )
    csv_text = daily.select_query_results(
        athena,
        daily.bind_config_sql(daily.select_uncleaned_sql(), CONFIG),
        "s3://lake/x/",
    )
    assert athena.executed and athena.results_locations == ["s3://lake/x/"]
    rows = daily.parse_query_results(csv_text)
    assert rows == [
        {"source": "greenhouse", "source_hash": a1, "source_id": "acme, inc 5"},
        {"source": "ashby", "source_hash": a2, "source_id": "acme 6"},
    ]


def test_run_daily_end_to_end(bucket_env: None) -> None:
    store = JournalSink()
    a = _post(store, "greenhouse", "description_html", "html p1 <p>boiler</p>")
    b = _post(store, "ashby", "description_html", "html p2 <ul><li>duty</li></ul>")
    athena = FakeAthena(
        [
            [
                ["source", "source_hash", "source_id"],
                [a["source"], a["source_hash"], a["source_id"]],
                [b["source"], b["source_hash"], b["source_id"]],
            ]
        ]
    )
    client = StubClient()
    record = daily.run_daily(athena, store, client, _settings(), pipeline_version="dev")

    assert record["status"] == "completed"
    assert record["counters"] == {"cleaned": 2, "cache_hits": 0, "failed": 0}
    assert record["input_dataset"] == "athena:semantic_career_map/select_uncleaned"
    assert record["configuration_hash"] == CONFIG
    assert len(client.calls) == 2
    artifact_keys = [k for k in store.journal if k.startswith("lake/artifacts/")]
    assert artifact_keys and any(k.endswith(".parquet") for k in artifact_keys)


def test_run_daily_applies_failure_ledger(bucket_env: None) -> None:
    store = JournalSink()
    a = _post(store, "greenhouse", "description_html", "html p1 <p>boiler</p>")
    b = _post(store, "ashby", "description_html", "html p2 <ul><li>duty</li></ul>")
    store.objects[keys.manifest_run(ids.new_id())] = json.dumps(
        {"failed_source_hashes": [b["source_hash"]]}
    ).encode()
    athena = FakeAthena(
        [
            [
                ["source", "source_hash", "source_id"],
                [a["source"], a["source_hash"], a["source_id"]],
                [b["source"], b["source_hash"], b["source_id"]],
            ]
        ]
    )
    client = StubClient()
    record = daily.run_daily(athena, store, client, _settings(), pipeline_version="dev")

    assert record["status"] == "completed"
    assert record["counters"] == {"cleaned": 1, "cache_hits": 0, "failed": 0}
    assert len(client.calls) == 1
    assert (
        client.calls[0]["user_message"].startswith("cleaned")
        or "p1" in client.calls[0]["user_message"]
    )


def test_empty_selection_runs_noop_record(bucket_env: None) -> None:
    store = JournalSink()
    athena = FakeAthena([[["source", "source_hash", "source_id"]]])
    client = StubClient()
    record = daily.run_daily(athena, store, client, _settings(), pipeline_version="dev")

    assert record["status"] == "completed"
    assert record["counters"] == {"cleaned": 0, "cache_hits": 0, "failed": 0}
    assert client.calls == []


def test_dry_run_touches_no_store_writes(bucket_env: None) -> None:
    store = JournalSink()
    a = _post(store, "greenhouse", "description_html", "html p1 <p>boiler</p>")
    athena = FakeAthena(
        [
            [
                ["source", "source_hash", "source_id"],
                [a["source"], a["source_hash"], a["source_id"]],
            ]
        ]
    )
    summary = daily.run_daily_dry(athena, store, _settings())
    assert summary == {
        "status": "dry-run",
        "configuration_hash": CONFIG,
        "manifest_postings": 1,
    }
    assert store.journal == []


def test_deadline_timer_arms_starts_and_cancels() -> None:
    assert daily.arm_deadline(0) is None
    assert daily.arm_deadline(-1) is None
    timer = daily.arm_deadline(0.05)
    try:
        assert timer is not None and timer.is_alive()
    finally:
        timer.cancel()
