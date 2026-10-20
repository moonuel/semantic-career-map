"""Environment-contract tests for ``ObjectStore.from_env``.

The pairing rule is the AWS-migration guardrail: a set ``STORE_S3_ENDPOINT``
means a local S3-compatible server (Garage) that needs explicit keys; an
empty or absent endpoint means real S3, where credentials must come from
boto3's default chain and static keys in the environment are refused.
"""

from __future__ import annotations

import pytest

from backend.storage.store import ObjectStore

_STORE_VARS = (
    "STORE_S3_BUCKET",
    "STORE_S3_ENDPOINT",
    "STORE_S3_REGION",
    "STORE_S3_ACCESS_KEY_ID",
    "STORE_S3_SECRET_ACCESS_KEY",
)

_GARAGE_ENV = {
    "STORE_S3_BUCKET": "scm-data",
    "STORE_S3_ENDPOINT": "http://127.0.0.1:3900",
    "STORE_S3_REGION": "garage",
    "STORE_S3_ACCESS_KEY_ID": "key-id",
    "STORE_S3_SECRET_ACCESS_KEY": "secret",
}

_AWS_ENV = {
    "STORE_S3_BUCKET": "lake",
    "STORE_S3_ENDPOINT": "",
    "STORE_S3_REGION": "us-east-1",
}


def _load_env(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    for name in _STORE_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_garage_mode_requires_explicit_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Endpoint set without the key pair fails, naming the missing variables."""
    _load_env(monkeypatch, {k: v for k, v in _GARAGE_ENV.items() if "KEY" not in k})
    with pytest.raises(
        RuntimeError, match="STORE_S3_ACCESS_KEY_ID, STORE_S3_SECRET_ACCESS_KEY"
    ):
        ObjectStore.from_env()


def test_garage_mode_sets_path_style_and_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Endpoint set: the client pins path-style addressing and the endpoint."""
    _load_env(monkeypatch, _GARAGE_ENV)
    store = ObjectStore.from_env()
    assert store._client.meta.endpoint_url == "http://127.0.0.1:3900"
    assert store._client.meta.config.s3["addressing_style"] == "path"


def test_aws_mode_refuses_static_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Endpoint empty with keys present is rejected — no static keys toward real S3."""
    _load_env(
        monkeypatch,
        {
            **_AWS_ENV,
            "STORE_S3_ACCESS_KEY_ID": "key-id",
            "STORE_S3_SECRET_ACCESS_KEY": "secret",
        },
    )
    with pytest.raises(RuntimeError, match="static keys"):
        ObjectStore.from_env()


def test_aws_mode_bare_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Endpoint empty and keys absent: real S3 with default-chain credentials."""
    _load_env(monkeypatch, _AWS_ENV)
    store = ObjectStore.from_env()
    assert store._bucket == "lake"
    assert store._client.meta.region_name == "us-east-1"
    assert not (store._client.meta.config and store._client.meta.config.s3)


def test_missing_required_variables_are_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without bucket or region the error names both, before any pairing check."""
    _load_env(monkeypatch, {})
    with pytest.raises(RuntimeError, match="STORE_S3_BUCKET, STORE_S3_REGION"):
        ObjectStore.from_env()
