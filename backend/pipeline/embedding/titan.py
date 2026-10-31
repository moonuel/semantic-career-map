"""Amazon Titan V2 encoder: Bedrock ``InvokeModel`` behind the encode protocol.

The production Titan generation (experiment 012 arm C) embeds each cleaned
posting with a single Bedrock ``InvokeModel`` call against the pinned
``model_revision`` (the Bedrock model ID, e.g.
``amazon.titan-embed-text-v2:0``). This module is the extraction of the
011 experiment's proven request contract into the production tier: one
request per text in input order, 1,024 dimensions, native normalization,
and the same per-row postconditions the experiment enforced (non-zero norm,
token count within the configured bound, unit norm within the shared
1e-5 tolerance).

Failure policy is split deliberately:

- Transient faults (throttling, 5xx, connection/timeouts) retry with capped
  exponential backoff — the backoff, not concurrency, is what makes the
  serial ~1 s/request cadence survive Bedrock's quota.
- Contract faults (access denied / model access, validation, a malformed
  body) never retry: they are deterministic and burn quota. They raise
  immediately with a message an operator can act on.

``boto3`` is imported lazily inside the client factory so importing this
module (and ``build_encoder``'s non-Titan branch) stays cheap.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from backend.pipeline.embedding.config import EmbeddingConfig
from backend.pipeline.embedding.encoder import EmbeddingEncoder

TITAN_MODEL_PREFIX = "amazon.titan-embed-text"
TITAN_DIMENSIONS = 1024
TITAN_MAX_TOKENS = 8192
TITAN_MAX_ATTEMPTS = 6
TITAN_BACKOFF_CAP_SECONDS = 30.0

_NORM_TOLERANCE = 1e-5
_ZERO_NORM = 1e-12

# Deterministic contract faults: retrying them only multiplies the wait.
_FATAL_CODES = frozenset(
    {
        "AccessDeniedException",
        "ResourceNotFoundException",
        "ValidationException",
        "ModelNotReadyException",
        "UnrecognizedClientException",
        "InvalidSignatureException",
        "ModelErrorException",
    }
)
_FATAL_MESSAGES = (
    "access to the model",
    "model access",
    "not authorized",
    "not authorized to perform",
)


class TitanContractError(ValueError):
    """A response or request violated the Titan embedding contract."""


def is_titan_model(model_name: str) -> bool:
    """Whether ``model_name`` names a Bedrock Titan text-embedding model.

    Args:
        model_name: The config's ``model_name`` (HF id or Bedrock family id).

    Returns:
        ``True`` for the Titan text-embedding family.
    """
    return model_name.startswith(TITAN_MODEL_PREFIX)


def _fatal(exc: BaseException) -> bool:
    """Whether a Bedrock exception is a deterministic contract fault."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error") or {}
        code = error.get("Code")
        if isinstance(code, str) and code in _FATAL_CODES:
            return True
        message = str(error.get("Message") or "").lower()
        if any(marker in message for marker in _FATAL_MESSAGES):
            return True
    text = str(exc).lower()
    return any(marker in text for marker in _FATAL_MESSAGES)


def _default_client():
    """Build a region-local Bedrock Runtime client (lazy ``boto3`` import)."""
    import boto3
    from botocore.config import Config

    from backend.pipeline.athena import aws_region

    return boto3.Session().client(
        "bedrock-runtime",
        region_name=aws_region(),
        config=Config(
            retries={"max_attempts": 1, "mode": "standard"},
            read_timeout=60,
            connect_timeout=10,
        ),
    )


class TitanEncoder(EmbeddingEncoder):
    """Bedrock Titan V2 encoder: serial, retrying, norm-postconditioned."""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        client_factory: Callable[[], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Bind the client and configuration (no network at construction).

        Args:
            config: The embedding configuration; ``model_revision`` is the
                Bedrock model ID, ``max_length`` the input-token bound, and
                ``normalize`` the L2 contract.
            client_factory: ``() -> bedrock-runtime client`` double for
                tests; defaults to a region-local client built on first use.
            sleep: Backoff sleep (tests inject a recorder).

        Raises:
            ValueError: if the configured ``max_length`` is below Titan's
                bound.
        """
        if config.max_length < TITAN_MAX_TOKENS:
            # The bound is Titan's contract, not a truncation knob: a lower
            # value would reject postings inside the measured range.
            raise ValueError(
                "Titan config max_length is the input-token bound and must be "
                f"at least {TITAN_MAX_TOKENS} (got {config.max_length})"
            )
        self._config = config
        self._client_factory = client_factory or _default_client
        self._sleep = sleep
        self._client: Any | None = None

    def _get_client(self):
        """The lazily built Bedrock Runtime client."""
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def _embed_one(self, text: str) -> np.ndarray:
        """Invoke Titan for one text; retry transients, fail fast on contract.

        Args:
            text: The text to embed.

        Returns:
            The validated, configured-normalized ``(1024,)`` float32 row.

        Raises:
            TitanContractError: on a malformed or out-of-contract response.
            Exception: the last *fatal* Bedrock error (never retried), or
                the last transient error after the attempt budget.
        """
        body = json.dumps(
            {
                "inputText": text,
                "dimensions": TITAN_DIMENSIONS,
                "normalize": bool(self._config.normalize),
            }
        )
        client = self._get_client()
        last_error: BaseException | None = None
        for attempt in range(1, TITAN_MAX_ATTEMPTS + 1):
            try:
                response = client.invoke_model(
                    modelId=self._config.model_revision,
                    accept="application/json",
                    contentType="application/json",
                    body=body,
                )
            except Exception as exc:
                if _fatal(exc):
                    raise
                last_error = exc
                if attempt == TITAN_MAX_ATTEMPTS:
                    break
                self._sleep(min(2.0 ** (attempt - 1), TITAN_BACKOFF_CAP_SECONDS))
                continue
            return self._validate(json.loads(response["body"].read()))
        raise RuntimeError(
            f"Titan embedding failed after {TITAN_MAX_ATTEMPTS} attempts for "
            f"model {self._config.model_revision!r}: {last_error}"
        ) from last_error

    def _validate(self, payload: dict[str, Any]) -> np.ndarray:
        """Enforce the response contract and return the normalized row.

        Args:
            payload: The decoded ``InvokeModel`` response body.

        Returns:
            The ``(1024,)`` float32 row.

        Raises:
            TitanContractError: on a missing/mis-shaped embedding, a token
                count beyond ``max_length``, a zero vector, or a norm that
                stays off-unit after one re-normalization.
        """
        try:
            embedding = np.asarray(payload["embedding"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise TitanContractError(
                f"Titan response carries no decodable embedding: {exc}"
            ) from exc
        if embedding.shape != (TITAN_DIMENSIONS,):
            raise TitanContractError(
                f"Titan returned shape {embedding.shape}, expected "
                f"({TITAN_DIMENSIONS},)"
            )
        tokens = int(payload.get("inputTextTokenCount") or 0)
        if tokens > self._config.max_length:
            raise TitanContractError(
                f"Titan input token count {tokens} exceeds the configured "
                f"bound {self._config.max_length} (no silent truncation)"
            )
        norm = float(np.linalg.norm(embedding))
        if norm < _ZERO_NORM:
            raise TitanContractError("Titan returned a zero vector")
        if not self._config.normalize:
            return embedding.astype(np.float32)
        embedding = embedding / norm
        drift = abs(float(np.linalg.norm(embedding)) - 1.0)
        if drift >= _NORM_TOLERANCE:
            raise TitanContractError(
                f"L2 normalization violated: norm drift {drift:.8f} beyond "
                f"{_NORM_TOLERANCE}"
            )
        return embedding.astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode ``texts`` serially into a float32 ``(N, 1024)`` matrix.

        Args:
            texts: The texts to embed (falsy-safe: ``(0, 0)`` empty matrix).

        Returns:
            The ``(N, 1024)`` float32 matrix in input order; unit-norm rows
            when the config normalizes.

        Raises:
            TitanContractError: when any row violates the response contract.
            Exception: a fatal Bedrock error, or transients exhausted.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = np.empty((len(texts), TITAN_DIMENSIONS), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index] = self._embed_one(text)
        return vectors
