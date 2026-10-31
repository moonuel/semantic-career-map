"""Titan encoder tests: response contract, order, retries — fake client, no
network."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from backend.pipeline.embedding.config import EmbeddingConfig
from backend.pipeline.embedding.titan import (
    TITAN_DIMENSIONS,
    TitanContractError,
    TitanEncoder,
    is_titan_model,
)


def _config(**kw) -> EmbeddingConfig:
    defaults = {
        "model_name": "amazon.titan-embed-text-v2",
        "model_revision": "amazon.titan-embed-text-v2:0",
        "model_segment": "amazon-titan-embed-text-v2",
        "variant": "titan-v2",
        "text_variant": "variant_text",
        "max_length": 8192,
        "normalize": True,
        "version": "v1",
        "sql_text": "SELECT 1",
    }
    defaults.update(kw)
    return EmbeddingConfig(**defaults)


class BedrockError(Exception):
    """Shape of a botocore ClientError without importing botocore."""

    def __init__(self, code: str, message: str = "boom") -> None:
        super().__init__(f"{code}: {message}")
        self.response = {"Error": {"Code": code, "Message": message}}


def _row(text: str, *, norm: float = 1.0) -> list[float]:
    """A deterministic 1,024-d row derived from the text, scaled to ``norm``."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    raw = rng.normal(size=TITAN_DIMENSIONS)
    return (raw / np.linalg.norm(raw) * norm).astype(float).tolist()


class FakeBedrock:
    """Records requests; answers per-text or raises scripted errors."""

    def __init__(self, *, errors: list[Exception] | None = None) -> None:
        self.requests: list[dict] = []
        self.errors = list(errors or [])
        self.tokens_of = lambda text: len(text)

    def invoke_model(self, **kw) -> dict:
        self.requests.append(kw)
        if self.errors:
            raise self.errors.pop(0)
        text = json.loads(kw["body"])["inputText"]
        payload = {
            "embedding": _row(text),
            "inputTextTokenCount": self.tokens_of(text),
        }
        return {"body": io.BytesIO(json.dumps(payload).encode())}


def _encoder(client, sleeps: list[float] | None = None, **config_kw) -> TitanEncoder:
    recorded = sleeps if sleeps is not None else []
    return TitanEncoder(
        _config(**config_kw), client_factory=lambda: client, sleep=recorded.append
    )


def test_is_titan_model_family() -> None:
    assert is_titan_model("amazon.titan-embed-text-v2")
    assert is_titan_model("amazon.titan-embed-text-v1")
    assert not is_titan_model("sentence-transformers/all-MiniLM-L6-v2")


def test_encode_shape_and_order() -> None:
    client = FakeBedrock()
    vectors = _encoder(client).encode(["alpha", "beta", "gamma"])
    assert vectors.shape == (3, TITAN_DIMENSIONS)
    assert vectors.dtype == np.float32
    sent = [json.loads(r["body"])["inputText"] for r in client.requests]
    assert sent == ["alpha", "beta", "gamma"]
    for index, text in enumerate(sent):
        assert np.allclose(vectors[index], _row(text), atol=1e-6)


def test_request_body_pins_dimensions_and_normalization() -> None:
    client = FakeBedrock()
    _encoder(client).encode(["x"])
    (request,) = client.requests
    body = json.loads(request["body"])
    assert body == {"inputText": "x", "dimensions": 1024, "normalize": True}
    assert request["modelId"] == "amazon.titan-embed-text-v2:0"


def test_serial_one_request_per_text() -> None:
    client = FakeBedrock()
    _encoder(client, batch_size=16).encode(["a", "b", "c", "d"])
    assert len(client.requests) == 4


def test_normalized_rows_are_unit_norm() -> None:
    client = FakeBedrock()
    vectors = _encoder(client).encode(["one", "two"])
    assert np.all(np.abs(np.linalg.norm(vectors, axis=1) - 1.0) < 1e-5)


def test_drifted_norm_is_renormalized_once() -> None:
    class DriftingClient(FakeBedrock):
        def invoke_model(self, **kw) -> dict:
            self.requests.append(kw)
            text = json.loads(kw["body"])["inputText"]
            payload = {
                "embedding": _row(text, norm=3.7),
                "inputTextTokenCount": 1,
            }
            return {"body": io.BytesIO(json.dumps(payload).encode())}

    vectors = _encoder(DriftingClient()).encode(["drifty"])
    assert np.all(np.abs(np.linalg.norm(vectors, axis=1) - 1.0) < 1e-5)


def test_normalize_false_is_honored() -> None:
    class UnnormedClient(FakeBedrock):
        def invoke_model(self, **kw) -> dict:
            self.requests.append(kw)
            assert json.loads(kw["body"])["normalize"] is False
            text = json.loads(kw["body"])["inputText"]
            payload = {"embedding": _row(text, norm=5.0), "inputTextTokenCount": 1}
            return {"body": io.BytesIO(json.dumps(payload).encode())}

    vectors = _encoder(UnnormedClient(), normalize=False).encode(["x"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 5.0)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"embedding": [0.0] * TITAN_DIMENSIONS, "inputTextTokenCount": 1}, "zero"),
        ({"embedding": [0.5] * 384, "inputTextTokenCount": 1}, "shape"),
        (
            {"embedding": [0.5] * TITAN_DIMENSIONS, "inputTextTokenCount": 9000},
            "token count 9000",
        ),
        ({"inputTextTokenCount": 1}, "no decodable embedding"),
    ],
)
def test_contract_violations_raise(payload: dict, match: str) -> None:
    class BadClient(FakeBedrock):
        def invoke_model(self, **kw) -> dict:
            return {"body": io.BytesIO(json.dumps(payload).encode())}

    with pytest.raises(TitanContractError, match=match):
        _encoder(BadClient()).encode(["x"])


def test_token_bound_is_config_max_length() -> None:
    class TokenClient(FakeBedrock):
        tokens_of = staticmethod(lambda text: 8192)

    vectors = _encoder(TokenClient()).encode(["fits"])
    assert vectors.shape == (1, TITAN_DIMENSIONS)


def test_retries_transient_errors_with_backoff() -> None:
    sleeps: list[float] = []
    errors = [BedrockError("ThrottlingException"), BedrockError("InternalServerError")]
    client = FakeBedrock(errors=errors)
    vectors = _encoder(client, sleeps).encode(["retry-me"])
    assert len(client.requests) == 3
    assert sleeps == [1.0, 2.0]
    assert vectors.shape == (1, TITAN_DIMENSIONS)


def test_transients_exhausted_raise_runtime_error() -> None:
    errors = [BedrockError("ThrottlingException") for _ in range(6)]
    client = FakeBedrock(errors=errors)
    with pytest.raises(RuntimeError, match="failed after 6 attempts"):
        _encoder(client).encode(["nope"])
    assert len(client.requests) == 6


@pytest.mark.parametrize(
    "error",
    [
        BedrockError("AccessDeniedException"),
        BedrockError("ValidationException", "input too long"),
        BedrockError("ThrottlingException", "You don't have access to the model"),
    ],
)
def test_fatal_errors_are_not_retried(error: Exception) -> None:
    client = FakeBedrock(errors=[error])
    sleeps: list[float] = []
    with pytest.raises(Exception) as excinfo:
        _encoder(client, sleeps).encode(["x"])
    assert excinfo.value is error
    assert len(client.requests) == 1
    assert sleeps == []


def test_empty_input_is_shape_safe() -> None:
    client = FakeBedrock()
    vectors = _encoder(client).encode([])
    assert vectors.shape == (0, 0)
    assert client.requests == []


def test_low_max_length_rejected() -> None:
    with pytest.raises(ValueError, match="must be at least 8192"):
        TitanEncoder(_config(max_length=256), client_factory=FakeBedrock)


def test_client_is_built_lazily() -> None:
    calls: list[int] = []

    def factory():
        calls.append(1)
        return FakeBedrock()

    encoder = TitanEncoder(_config(), client_factory=factory)
    assert calls == []
    encoder.encode(["x"])
    assert calls == [1]
