"""Key-layer tests for the embeddings branch: builder round trips and the
shape pinner's rejections."""

from __future__ import annotations

import pytest

from backend.storage import keys

RUN_ID = "0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d70"


def key(model: str = "all-minilm-l6-v2", variant: str = "minilm") -> str:
    return keys.embedding_part(model, variant, "v1", RUN_ID, 1)


EXPECTED = (
    "lake/embeddings/model=all-minilm-l6-v2/variant=minilm/version=vN/"
    "part-<run-id>-<seq>.parquet"
)


def test_builder_round_trips_through_validate() -> None:
    built = key()
    assert built.startswith("lake/embeddings/model=")
    assert keys.validate_key(built) == built


def test_builder_part_leaf_pads_sequence() -> None:
    assert keys.embedding_part("all-minilm-l6-v2", "minilm", "v2", RUN_ID, 5).endswith(
        "-00005.parquet"
    )


@pytest.mark.parametrize(
    "bad",
    [
        # wrong partition order
        (
            "lake/embeddings/variant=minilm/model=all-minilm-l6-v2/version=v1/"
            f"part-{RUN_ID}-00001.parquet"
        ),
        # non-version token
        (
            "lake/embeddings/model=all-minilm-l6-v2/variant=minilm/version=1/"
            f"part-{RUN_ID}-00001.parquet"
        ),
        # non-run-minted leaf
        (
            "lake/embeddings/model=all-minilm-l6-v2/variant=minilm/version=v1/"
            "data.parquet"
        ),
        # extra partition level
        (
            "lake/embeddings/day=2026-10-26/model=all-minilm-l6-v2/variant=minilm/"
            f"version=v1/part-{RUN_ID}-00001.parquet"
        ),
        # wrong leaf extension form
        (
            "lake/embeddings/model=all-minilm-l6-v2/variant=minilm/version=v1/"
            f"{RUN_ID}.npy"
        ),
    ],
)
def test_pinner_rejects_malformed_shapes(bad: str) -> None:
    with pytest.raises(ValueError):
        keys.validate_key(bad)


@pytest.mark.parametrize(
    "kw",
    [
        {"model": "with slash/not ok"},
        {"variant": "UpperCase"},
        {"version": "v0"},
        {"version": "v1x"},
    ],
)
def test_builder_validate_token_shapes(kw: dict) -> None:
    with pytest.raises(ValueError):
        keys.embedding_part(
            kw.get("model", "ok-model"),
            kw.get("variant", "ok-variant"),
            kw.get("version", "v1"),
            RUN_ID,
            1,
        )


def test_builder_rejects_run_id_and_sequence() -> None:
    with pytest.raises(ValueError):
        keys.embedding_part("m", "v", "v1", "not-a-uuid7", 1)
    with pytest.raises(ValueError):
        keys.embedding_part("m", "v", "v1", RUN_ID, 0)
