"""Cleaning-quality result datasets: evaluation of cleaned rows vs golden.

Closes the eval loop (task 4.3): join golden references and cleaned rows by
``source_hash``, compute the verbatim exp 003 metrics
(:mod:`backend.pipeline.eval.metrics`), and append the results to
``eval/results/type=cleaning_quality/variant=<variant>/version=<vN>/`` as
Parquet — machine output under generated data, versions add and never
overwrite (data.md §8).

Lineage contract: every row pins the posting identity (``source``,
``source_hash``, ``source_id`` from the golden row — ``source_hash`` is the
join handle design D uses), per-metric values, the ``evaluation_version``
(version/path token) and an ``evaluation_configuration_hash`` — the SHA-256
of the rubric shape (evaluation version + marker regexes + thresholds), so
"the same dataset re-evaluated under a different rubric" is distinguishable
in the data. Rubric upgrades mean a new ``evaluation_version``/new prefix,
never mutated rows.

The raw text for the hallucination metric is read per posting from the
store's raw payload (its shape text field through the same normalizer the
cleaning stage used) — the lake-native meaning of exp 003's
``raw_full_text`` snapshot.
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from backend.pipeline.eval import metrics as m
from backend.storage import keys

RESULT_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("source_hash", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("jaccard", pa.float64()),
        pa.field("boilerplate_hits", pa.list_(pa.string())),
        pa.field("hallucination_count", pa.int64()),
        pa.field("hallucination_sample", pa.list_(pa.string())),
        pa.field("over_deletion_pct", pa.float64()),
        pa.field("over_deletion_flagged", pa.bool_()),
        pa.field("over_deletion_sample", pa.list_(pa.string())),
        pa.field("golden_words", pa.int64()),
        pa.field("clean_words", pa.int64()),
        pa.field("evaluation_configuration_hash", pa.string()),
        pa.field("evaluation_version", pa.string()),
        pa.field("created_at", pa.string()),
    ]
)


def evaluation_hash(evaluation_version: str) -> str:
    """Hash the rubric shape: version + marker regexes + thresholds.

    Args:
        evaluation_version: The results version token (``vN``).

    Returns:
        sha256 hex of the canonical JSON carrying the frozen rubric
        (markers, flag threshold, sample limit) — two evaluations that pair
        the same cleaned rows with the same golden rows under the same
        version hash identically; a rubric change is a new version.
    """
    rubric = {
        "rubric": "exp003-cleaning-quality",
        "evaluation_version": evaluation_version,
        "boilerplate_markers": m.BOILERPLATE_MARKERS,
        "over_deletion_flag_pct": m.OVER_DELETION_FLAG_PCT,
        "sample_limit": m.SAMPLE_LIMIT,
        "word_definition": m.word_set.__name__,
        "jaccard_definition": m.jaccard_similarity.__name__,
    }
    payload = json.dumps(rubric, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_posting(
    *,
    golden_text: str,
    cleaned_text: str,
    raw_text: str,
    evaluation_configuration_hash: str,
    evaluation_version: str,
    created_at: str,
) -> dict[str, Any]:
    """Compute every cleaning-quality metric for one posting.

    Metric definitions are the verbatim exp 003 semantics
    (:mod:`backend.pipeline.eval.metrics`); the match fields mirror exp
    003's per-posting table.

    Args:
        golden_text: The hand-cleaned reference text.
        cleaned_text: One cleaned row's text.
        raw_text: The posting's raw text (shape field, normalized).
        evaluation_configuration_hash: The rubric hash (from
            :func:`evaluation_hash`).
        evaluation_version: The results version token.
        created_at: UTC ISO-8601 timestamp of the evaluation run.

    Returns:
        One result row (keyed to :data:`RESULT_SCHEMA`).
    """
    halluc = m.hallucinated_words(cleaned_text, raw_text)
    pct, missing = m.over_deleted_words(golden_text, cleaned_text)
    return {
        "jaccard": m.jaccard_similarity(cleaned_text, golden_text),
        "boilerplate_hits": m.check_boilerplate(cleaned_text),
        "hallucination_count": len(halluc),
        "hallucination_sample": m.first_sample(halluc),
        "over_deletion_pct": pct,
        "over_deletion_flagged": m.over_deletion_flagged(pct),
        "over_deletion_sample": m.first_sample(missing),
        "golden_words": len(golden_text.split()),
        "clean_words": len(cleaned_text.split()),
        "evaluation_configuration_hash": evaluation_configuration_hash,
        "evaluation_version": evaluation_version,
        "created_at": created_at,
    }


def write_results(
    store: Any,
    rows: list[dict[str, Any]],
    *,
    eval_type: str = "cleaning_quality",
    variant: str = "exp003",
    evaluation_version: str = "v1",
    run_id: str,
    part_seq: int = 1,
) -> str:
    """Write one evaluation's result rows as a Parquet part.

    Args:
        store: Object store (or testing double).
        rows: Result rows (:func:`evaluate_posting` products plus the
            golden row's ``source``/``source_hash``/``source_id``).
        variant: The rubric variant name (``eval_results`` variant).
        evaluation_version: The results version token (``vN``).
        run_id: Canonical UUIDv7 of this evaluation run.
        part_seq: 1-based sequence within the run.

    Returns:
        The written part key.

    Raises:
        ValueError: if a row's lineage is incomplete (identity fields and
            the evaluation fields are mandatory — drift must surface at
            write time, not in Athena).
    """
    for row in rows:
        for column in (
            "source",
            "source_hash",
            "evaluation_configuration_hash",
            "evaluation_version",
            "created_at",
        ):
            if row.get(column) in (None, ""):
                raise ValueError(f"result row missing {column!r} lineage: {row!r}")
    table = pa.Table.from_pylist(rows, schema=RESULT_SCHEMA)
    sink = io.BytesIO()
    pq.write_table(table, sink)
    key = keys.eval_result(eval_type, variant, evaluation_version, run_id, part_seq)
    store.put(key, sink.getvalue())
    return key
