"""Encoder-layer tests: pinned-revision fail-fast, normalization
postconditions, order preservation — all against stub models/factories, no
downloads."""

from __future__ import annotations

import numpy as np
import pytest

from backend.pipeline.embedding.config import EmbeddingConfig
from backend.pipeline.embedding.encoder import (
    RevisionMismatch,
    SentenceEncoder,
    StubEncoder,
    _hash_row,
    revision_of,
)


def _config(**kw) -> EmbeddingConfig:
    defaults = {
        "model_name": "m",
        "model_revision": "rev-1",
        "model_segment": "m",
        "variant": "v",
        "text_variant": "t",
        "version": "v1",
        "sql_text": "SELECT 1",
    }
    defaults.update(kw)
    return EmbeddingConfig(**defaults)


class StubModel:
    """Duck-typed stand-in: per-text hash rows, resolvable revision chain."""

    def __init__(self, revision: str | None, dimension: int = 4) -> None:
        self.max_seq_length = 999
        self.windows: list[int] = []
        self.dimension = dimension
        self._revision = revision

    def encode(self, texts, *, batch_size, convert_to_numpy, show_progress_bar):
        self.windows.append(batch_size)
        return np.stack([_hash_row(text, self.dimension) for text in texts])

    def __getitem__(self, index):
        class AutoModel:
            config = type("Config", (), {"_commit_hash": self._revision})()

        class Transformer:
            auto_model = AutoModel()

        return Transformer()


def _factory(model: StubModel):
    return lambda name, revision: model


def test_revision_of_walks_the_chain() -> None:
    class AutoModel:
        config = type("Config", (), {"_commit_hash": "abc123def"})()

    class Transformer:
        auto_model = AutoModel()

    class Model:
        def __getitem__(self, index):
            return Transformer()

    assert revision_of(Model()) == "abc123def"
    assert revision_of(object()) is None
    # the double's chain resolves to its own revision
    assert revision_of(StubModel(revision="rev-1")) == "rev-1"


def test_revision_mismatch_fails_at_load() -> None:
    with pytest.raises(RevisionMismatch):
        SentenceEncoder(
            _config(), model_factory=_factory(StubModel(revision="other-rev"))
        )


def test_revision_unresolvable_proceeds() -> None:
    encoder = SentenceEncoder(
        _config(), model_factory=_factory(StubModel(revision=None))
    )
    assert encoder is not None


@pytest.mark.parametrize("dimension", [4, 8])
def test_encode_normalizes_to_unit_norm(dimension: int) -> None:
    model = StubModel(revision="rev-1", dimension=dimension)
    encoder = SentenceEncoder(_config(), model_factory=_factory(model))
    vectors = encoder.encode(["a", "b"])
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert np.all(np.abs(norms - 1.0) < 1e-5)


def test_encode_order_preserved_and_window_ridden() -> None:
    model = StubModel(revision="rev-1", dimension=4)
    encoder = SentenceEncoder(_config(batch_size=3), model_factory=_factory(model))
    texts = ["t1", "t2", "t3", "t4"]
    vectors = encoder.encode(texts)
    assert vectors.shape == (len(texts), model.dimension)
    # order preserved: each row is the unit vector of that text's hash row
    for index, text in enumerate(texts):
        row = _hash_row(text, model.dimension)
        unit = (row / np.linalg.norm(row)).astype(np.float32)
        assert np.allclose(vectors[index], unit, atol=1e-5)
    # one encode call carries the config's batch window (the driver slices
    # chunks; the model call sees batch_size)
    assert model.windows == [3]


def test_empty_encode_is_shape_safe() -> None:
    encoder = SentenceEncoder(_config(), model_factory=_factory(StubModel(None)))
    assert encoder.encode([]).shape == (0, 0)


def test_stub_encoder_is_deterministic_unit_norm() -> None:
    first = StubEncoder().encode(["hello", "world"])
    second = StubEncoder().encode(["hello", "world"])
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
