"""Selection-layer tests: the anti-join SQL text, binding, and row shaping
(header contract, fallback, empty accounting, validation)."""

from __future__ import annotations

import pytest

from backend.pipeline.embedding import selection

H1 = "0" * 64
H2 = "1" * 64


def test_select_unembedded_sql_is_the_anti_join() -> None:
    sql = selection.select_unembedded_sql()
    assert "LEFT JOIN postings_embeddings" in sql
    # partition scoping is the pinned registration's job; the query filters
    # payload columns only — the configuration hash IS the cache identity
    assert "e.configuration_hash = :config" in sql
    assert ":model" not in sql and ":variant" not in sql and ":version" not in sql
    assert "WHERE e.source_hash IS NULL" in sql
    assert "LIMIT" in sql  # the deterministic row cap default


def test_select_unembedded_sql_caps_deterministically() -> None:
    sql = selection.select_unembedded_sql(max_postings=25)
    assert sql.endswith("LIMIT 25")
    sql = selection.select_unembedded_sql(max_postings=0)
    assert "LIMIT" not in sql


def test_bind_sql_params_binds_every_named_param() -> None:
    bound = selection.bind_sql_params("x = :config AND y = :other", {"config": "c", "other": "v"})
    assert bound == "x = 'c' AND y = 'v'"


def test_bind_sql_params_rejects_leftovers() -> None:
    with pytest.raises(ValueError):
        selection.bind_sql_params("x = :config AND y = :unknown", {"config": "c"})


def test_shape_rows_skips_header_and_builds_pairs() -> None:
    rows = [
        ["source_hash", "variant_text"],
        [H1, "texts one"],
        [H2, "texts two"],
    ]
    usable, empty = selection.shape_rows(rows, text_variant="variant_text")
    assert usable == [(H1, "texts one"), (H2, "texts two")]
    assert empty == 0


def test_shape_rows_counts_empty_not_silent() -> None:
    rows = [
        ["source_hash", "variant_text"],
        [H1, "   "],
        [H2, ""],
    ]
    usable, empty = selection.shape_rows(rows, text_variant="variant_text")
    assert usable == []
    assert empty == 2


def test_shape_rows_fallback_applied_per_row() -> None:
    rows = [
        ["source_hash", "variant_text", "fallback_text"],
        [H1, "", "fallback text"],
        [H2, "primary kept", "ignored"],
        ["2" * 64, "", "  "],  # both empty -> counted
    ]
    usable, empty = selection.shape_rows(
        rows, text_variant="variant_text", fallback_text_variant="fallback_text"
    )
    assert usable == [(H1, "fallback text"), (H2, "primary kept")]
    assert empty == 1


def test_shape_rows_rejects_drift() -> None:
    with pytest.raises(ValueError):
        selection.shape_rows([["source_hash", 64 * "f"]], text_variant="other")
    with pytest.raises(ValueError):
        selection.shape_rows(
            [["notheader", "variant_text"], ["short", "x"]], text_variant="variant_text"
        )
    with pytest.raises(ValueError):
        selection.shape_rows(
            [["source_hash", "variant_text"], ["bad-hash", "text"]],
            text_variant="variant_text",
        )
