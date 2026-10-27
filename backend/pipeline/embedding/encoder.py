"""Encoder wrapper: pinned-revision CPU encode with norm postcondition.

The embed step's single identity check lives here: the model is loaded
against the configuration's pinned ``model_revision``, the actually-resolved
revision is checked against it (fail fast — ambient resolution is banned
from provenance), the tokenizer truncation is set to the configuration's
``max_length``, and encode returns order-preserving float32 rows whose L2
norm meets the 1e-5 tolerance whenever ``normalize`` is set. The real model
loads only inside :class:`SentenceEncoder`; tests inject factories and stub
models, never download weights.
"""

from __future__ import annotations

import hashlib

import numpy as np

from backend.pipeline.embedding.config import EmbeddingConfig

_NORM_TOLERANCE = 1e-5


def _hash_row(text: str, dimension: int) -> np.ndarray:
    """Deterministic unit-norm-able row for a text (doubles only)."""
    raw = hashlib.sha256(text.encode("utf-8")).digest()
    need = dimension * 2
    chunk = (raw * ((need // len(raw)) + 1))[:need]
    return (np.frombuffer(chunk, dtype=np.int16).astype(np.float32) + 1)[:dimension]


class RevisionMismatch(RuntimeError):
    """The resolved model revision does not equal the pinned one."""


def resolve_pinned_revision(model_name: str) -> str:
    """Resolve the model's current latest revision (CLI convenience).

    This is the *explicit* resolution path the CLI's ``--revision auto``
    uses: it resolves once, prints, and the resulting string becomes the
    config's pin (hashed identity, asserted at load). This is not ambient
    resolution — a run that skips it must carry a pinned revision, and
    every run fails fast on a drifted load regardless.

    Args:
        model_name: Full HF-style model id.

    Returns:
        The resolved revision (a commit hash).

    Raises:
        OSError: propagated from the hub when the model id is unknown.
    """
    from huggingface_hub import model_info

    return model_info(model_name).sha


class EmbeddingEncoder:
    """The encode protocol the driver consumes (encode only)."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode ``texts`` to a float32 ``(N, D)`` matrix in input order."""
        raise NotImplementedError


def revision_of(model: object) -> str | None:
    """Best-effort resolved HuggingFace revision of a loaded ST model.

    Walks the duck-typed chain sentence-transformers builds (the first
    ``Transformer`` module's auto model carries the loaded commit on its
    config). Returns ``None`` when unresolvable — e.g. an offline snapshot
    load without metadata — in which case the caller keeps the pin and logs
    the unresolved provenance rather than inventing one.

    Args:
        model: A loaded sentence-transformers model (or a test double).

    Returns:
        The resolved revision string, or ``None``.
    """
    try:
        first = model[0]  # type: ignore[index]
        auto_model = first.auto_model  # type: ignore[attr-defined]
        config = auto_model.config  # type: ignore[attr-defined]
        commit = config._commit_hash  # type: ignore[attr-defined]
    except AttributeError, IndexError, KeyError, TypeError, NotImplementedError:
        return None
    if isinstance(commit, str) and commit:
        return commit
    return None


class SentenceEncoder(EmbeddingEncoder):
    """Real encoder: pinned-revision load, CPU-only, norm-postconditions."""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        model_factory=None,
    ) -> None:
        """Load the configured model (or take an injected factory).

        Args:
            config: The embedding configuration.
            model_factory: ``(model_name, revision) -> loaded model``
                double for tests; defaults to a real CPU
                sentence-transformers load, resolving and asserting the
                revision before any encode can run.

        Raises:
            RevisionMismatch: if the resolved revision differs from the pin.
            OSError: propagated from the loader on a bad name/revision.
        """
        self._config = config
        if model_factory is None:

            def model_factory(name: str, revision: str) -> object:
                from sentence_transformers import SentenceTransformer

                return SentenceTransformer(
                    name, device="cpu", revision=revision or None
                )

        self._model = model_factory(config.model_name, config.model_revision)
        resolved = revision_of(self._model)
        if (
            resolved is not None
            and config.model_revision
            and resolved != config.model_revision
        ):
            raise RevisionMismatch(
                f"resolved model revision {resolved!r} != pinned "
                f"{config.model_revision!r} for {config.model_name!r}"
            )
        try:
            self._model.max_seq_length = config.max_length  # type: ignore[index]
        except AttributeError, TypeError:
            pass  # double without the attribute; encoding assertions still hold

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode in configuration-normalized float32 rows, input order.

        Args:
            texts: The texts to embed (falsy-safe: returns a ``(0, 0)``
                float32 empty matrix).

        Returns:
            The ``(N, D)`` float32 matrix; unit-norm rows when the config
            normalizes.

        Raises:
            ValueError: on an all-empty-text selection producing a
                zero-norm row, or a postcondition norm violation.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=self._config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if self._config.normalize:
            norms = np.linalg.norm(vectors, axis=1)
            if np.any(norms < 1e-12):
                raise ValueError("zero-norm vector produced; cannot normalize")
            vectors = vectors / norms[:, None]
        post = np.linalg.norm(vectors, axis=1) if self._config.normalize else None
        if post is not None and not np.all(np.abs(post - 1.0) < _NORM_TOLERANCE):
            raise ValueError(
                "L2 normalization violated: norms drift beyond "
                f"{_NORM_TOLERANCE} (min {post.min():.8f}, max {post.max():.8f})"
            )
        return vectors


class StubEncoder(EmbeddingEncoder):
    """Deterministic double: hash-derived unit vectors (tests only)."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        self.calls.append(list(texts))
        rows = np.stack([_hash_row(text, self.dimension) for text in texts])
        norms = np.linalg.norm(rows, axis=1)
        return (rows / norms[:, np.newaxis]).astype(np.float32)
