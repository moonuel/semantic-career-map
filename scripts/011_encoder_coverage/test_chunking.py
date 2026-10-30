"""Single-responsibility tests for Experiment 011 window chunking."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "run_experiment.py"
spec = importlib.util.spec_from_file_location("exp011_run_experiment", SCRIPT)
assert spec is not None and spec.loader is not None
run_experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_experiment)

chunk_ids = run_experiment.chunk_ids


def test_chunk_ids_no_overlap_covers_all_tokens() -> None:
    ids = list(range(10))
    assert chunk_ids(ids, 4, 0) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_chunk_ids_overlap_repeats_boundary_tokens() -> None:
    ids = list(range(10))
    assert chunk_ids(ids, 4, 2) == [
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [4, 5, 6, 7],
        [6, 7, 8, 9],
        [8, 9],
    ]


def test_chunk_ids_empty_input_returns_one_empty_window() -> None:
    assert chunk_ids([], 4) == [[]]


@pytest.mark.parametrize("overlap", [-1, 4, 5])
def test_chunk_ids_rejects_bad_overlap(overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_ids(list(range(10)), 4, overlap)
