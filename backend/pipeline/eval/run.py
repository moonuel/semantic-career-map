"""Evaluation orchestration: golden ∩ cleaned joins, result writes, summary.

Task 4.4's engine under a thin ``__main__`` CLI: load the authored golden
dataset version and one artifacts (cleaned) dataset version, join rows by
``source_hash``, evaluate every matched posting with the verbatim exp 003
metrics, append one Parquet part to the results version, and render the
exp-003-style per-posting summary table.

Join rules that keep the job honest:

- Golden rows join cleaned rows by ``source_hash`` (design D5 — the
  posting's identity). ``source_id`` rides along from the golden row for
  the human-facing table.
- A cleaned dataset version may legitimately carry several
  ``configuration_hash``es (versions append, they don't overwrite), so the
  evaluation pins ONE cleaning configuration: pass
  ``configuration_hash`` when the version mixes more than one, else the
  join raises — ambiguity must surface, not silently double-report.
- Postings missing from either side are reported in the summary (never
  fabricated as rows).
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pyarrow.parquet as pq

from backend.pipeline.cleaning.normalization import SHAPE_TEXT_FIELD, normalize_html
from backend.pipeline.eval.results import (
    evaluate_posting,
    evaluation_hash,
    write_results,
    write_run_manifest,
)
from backend.storage import ids, keys

# Probing run-id for pure prefix composition (same trick as the cleaning
# cache module: the builder validates; no I/O).
_PROBE_RUN_ID = "0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d70"


def golden_prefix(variant: str = "author", version: str = "v1") -> str:
    """Compose + validate the golden dataset version's part prefix."""
    keys.eval_dataset("cleaning_golden", variant, version, _PROBE_RUN_ID, 1)
    return (
        f"lake/eval/datasets/type=cleaning_golden/variant={variant}/version={version}/"
    )


def results_prefix(variant: str = "exp003", version: str = "v1") -> str:
    """Compose + validate the results dataset version's part prefix."""
    keys.eval_result("cleaning_quality", variant, version, _PROBE_RUN_ID, 1)
    return (
        f"lake/eval/results/type=cleaning_quality/variant={variant}/version={version}/"
    )


def _read_rows(store: Any, part_keys: list[str]) -> list[dict[str, Any]]:
    """Read every row of the given Parquet parts, part order preserved."""
    rows: list[dict[str, Any]] = []
    for part_key in part_keys:
        table = pq.read_table(io.BytesIO(store.get(part_key)))
        columns = [column.to_pylist() for column in table.columns]
        rows.extend(dict(zip(table.column_names, values)) for values in zip(*columns))
    return rows


def _list_parts(store: Any, prefix: str) -> list[str]:
    """Sorted ``.parquet`` part keys under a validated dataset prefix."""
    return [
        keys.validate_key(key)
        for key in store.list_keys(prefix)
        if key.endswith(".parquet")
    ]


@dataclass
class GoldenRow:
    """One authored golden reference (loaded from
    ``eval/datasets/type=cleaning_golden/…``)."""

    source: str
    source_hash: str
    source_id: str
    golden_text: str


@dataclass
class CleanedRow:
    """One cleaned artifact row joined for evaluation."""

    source_hash: str
    cleaned_text: str
    configuration_hash: str


@dataclass(frozen=True)
class EvaluationSummary:
    """The outcome of one evaluation run.

    Attributes:
        part_key: The results Parquet part written (empty when ``write`` is
            false).
        rows: Every evaluated result row (posting identity + metrics).
        missing_golden: ``source_hash`` values with a golden row but no
            cleaned row (not evaluated).
        missing_cleaned: ``source_hash`` values with a cleaned row but no
            golden row.
        configuration_hash: The cleaning configuration the evaluation
            pinned (the cleaned rows' shared hash).
        evaluation_configuration_hash: The rubric hash
            (:func:`results.evaluation_hash`).
    """

    part_key: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    missing_golden: list[str] = field(default_factory=list)
    missing_cleaned: list[str] = field(default_factory=list)
    configuration_hash: str = ""
    evaluation_configuration_hash: str = ""


def load_golden(
    store: Any,
    *,
    variant: str = "author",
    version: str = "v1",
) -> list[GoldenRow]:
    """Load the authored golden references of one dataset version.

    Args:
        store: Object store (or testing double).
        variant: Authoring method (``author``).
        version: Version token ``vN``.

    Returns:
        Golden rows in part order (writing order).

    Raises:
        ValueError: if a row lacks any mandatory identity/reference column
            (a golden dataset with a broken schema must fail loudly, not
            evaluate a phantom reference).
    """
    mandatory = ("source", "source_hash", "golden_text")
    rows: list[GoldenRow] = []
    for row in _read_rows(store, _list_parts(store, golden_prefix(variant, version))):
        for column in mandatory:
            if not row.get(column):
                raise ValueError(f"golden row missing {column!r}: {row!r}")
        rows.append(
            GoldenRow(
                source=row["source"],
                source_hash=row["source_hash"],
                source_id=row.get("source_id", ""),
                golden_text=row["golden_text"],
            )
        )
    return rows


def load_cleaned(
    store: Any,
    *,
    artifact_type: str = "cleaned_text",
    variant: str = "llm",
    version: str = "v1",
    configuration_hash: str | None = None,
) -> list[CleanedRow]:
    """Load one cleaned-dataset version's rows, pinned to one configuration.

    Args:
        store: Object store (or testing double).
        artifact_type: Governed artifact family (``cleaned_text``).
        variant: Producing method (``llm``).
        version: Version token ``vN``.
        configuration_hash: Pin the cleaning configuration when the version
            carries several.

    Returns:
        Cleaned rows keyed ``source_hash``/``cleaned_text``
        (``configuration_hash`` carried for the summary).

    Raises:
        ValueError: if the version carries rows under more than one
            ``configuration_hash`` without an explicit pin.
    """
    part_prefix = (
        f"lake/artifacts/type={artifact_type}/variant={variant}/version={version}/"
    )
    keys.artifact_part(artifact_type, variant, version, _PROBE_RUN_ID, 1)
    rows: list[CleanedRow] = []
    seen_configs: set[str] = set()
    for row in _read_rows(store, _list_parts(store, part_prefix)):
        config = row.get("configuration_hash", "")
        if config:
            seen_configs.add(config)
        rows.append(
            CleanedRow(
                source_hash=row["source_hash"],
                cleaned_text=row.get("cleaned_text", ""),
                configuration_hash=config,
            )
        )
    if len(seen_configs) > 1 and configuration_hash is None:
        raise ValueError(
            "cleaned version mixes configurations "
            f"{sorted(seen_configs)}; pass --configuration-hash to pin one"
        )
    if configuration_hash is not None:
        rows = [r for r in rows if r.configuration_hash == configuration_hash]
    return rows


def _raw_text(store: Any, source: str, source_hash: str) -> str:
    """Fetch + normalize a posting's shape text field from the raw layer."""
    payload = json.loads(store.get(keys.raw_payload(source, source_hash)))
    field = SHAPE_TEXT_FIELD[source]
    return normalize_html(payload.get(field, ""))


def run_evaluation(
    store: Any,
    *,
    golden: list[GoldenRow] | None = None,
    cleaned: list[CleanedRow] | None = None,
    write: bool = True,
    results_variant: str = "exp003",
    evaluation_version: str = "v1",
    run_id: str | None = None,
    created_at: str | None = None,
) -> EvaluationSummary:
    """Evaluate golden ∩ cleaned and persist the results part.

    Args:
        store: Object store (or testing double).
        golden: Loaded golden rows (default
            :func:`load_golden` on the store).
        cleaned: Loaded cleaned rows (default
            :func:`load_cleaned` on the store). When given explicitly with
            mixed configurations, filter before passing.
        write: Persist the Parquet part in the real store when True (menus
            in tests write to the double; the summary is dry when False).
        results_variant: The results rubric variant.
        evaluation_version: Results version token (``vN``).
        run_id: Canonical UUIDv7 for the evaluation run (minted when None).
        created_at: UTC ISO-8601 timestamp (minted now when None).

    Returns:
        :class:`EvaluationSummary` — rows, join gaps, the written part key
        (empty when ``write`` is False), and the rubric hash.
    """
    if created_at is None:
        created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    run_id = run_id or ids.new_id()
    golden_by_hash = {g.source_hash: g for g in (golden or load_golden(store))}
    cleaned_by_hash = {c.source_hash: c for c in (cleaned or load_cleaned(store))}

    missing_golden = sorted(h for h in cleaned_by_hash if h not in golden_by_hash)
    missing_cleaned = sorted(h for h in golden_by_hash if h not in cleaned_by_hash)

    eval_config_hash = evaluation_hash(evaluation_version)
    configs = {c.configuration_hash for c in cleaned_by_hash.values()}
    if len(configs) == 1:
        pinned = next(iter(configs))
    else:
        pinned = ""

    rows: list[dict[str, Any]] = []
    for source_hash, cleaned_row in cleaned_by_hash.items():
        golden_row = golden_by_hash.get(source_hash)
        if golden_row is None:
            continue
        raw = _raw_text(store, golden_row.source, source_hash)
        result = evaluate_posting(
            golden_text=golden_row.golden_text,
            cleaned_text=cleaned_row.cleaned_text,
            raw_text=raw,
            evaluation_configuration_hash=eval_config_hash,
            evaluation_version=evaluation_version,
            created_at=created_at,
        )
        row_entry = {
            "source": golden_row.source,
            "source_hash": source_hash,
            "source_id": golden_row.source_id,
            **result,
        }
        row_entry["evaluation_configuration_hash"] = eval_config_hash
        row_entry["evaluation_version"] = evaluation_version
        row_entry["created_at"] = created_at
        rows.append(row_entry)

    part_key = ""
    if write and rows:
        part_key = write_results(
            store,
            rows,
            variant=results_variant,
            evaluation_version=evaluation_version,
            run_id=run_id,
        )
        write_run_manifest(
            store,
            run_id=run_id,
            evaluation_version=evaluation_version,
            evaluation_configuration_hash=eval_config_hash,
            results_variant=results_variant,
            part_key=part_key,
            row_count=len(rows),
            created_at=created_at,
        )
    return EvaluationSummary(
        part_key=part_key,
        rows=rows,
        missing_golden=missing_golden,
        missing_cleaned=missing_cleaned,
        configuration_hash=pinned,
        evaluation_configuration_hash=eval_config_hash,
    )


def render_table(summary: EvaluationSummary) -> str:
    """Render the exp-003-style per-posting summary table.

    Args:
        summary: An evaluation summary (:func:`run_evaluation` result).

    Returns:
        Multi-line table: one row per evaluated posting (jaccard,
        boilerplate/hallucination/over-deletion), plus a final means row —
        the exp 003 print layout the report's table came from.
    """
    lines = [
        (f"CLEANING QUALITY EVALUATION — {len(summary.rows)} evaluated postings"),
        (
            "  (exp 003 metrics: Jaccard vs golden, boilerplate markers, "
            "hallucination vs raw, golden-over-deletion %)"
        ),
        (
            f"{'#':>3}  {'source':<10}  {'jaccard':>7}  {'boilerplate':<10}  "
            f"{'halluc':>6}  {'overdel%':>8}  {'flag':<4}"
        ),
    ]
    jaccards: list[float] = []
    pcts: list[float] = []
    halluc_counts: list[int] = []
    for i, row in enumerate(summary.rows, start=1):
        flags = (
            ("bp=" + ",".join(row["boilerplate_hits"])[:12])
            if row["boilerplate_hits"]
            else "CLEAN"
        )
        lines.append(
            f"{i:>3}  {row['source']:<10.10}  {row['jaccard']:>7.3f}  {flags:<10.10}  "
            f"{row['hallucination_count']:>6}  {row['over_deletion_pct']:>8.1f}  "
            f"{'FLAG' if row['over_deletion_flagged'] else ''}"
        )
        jaccards.append(row["jaccard"])
        pcts.append(row["over_deletion_pct"])
        halluc_counts.append(row["hallucination_count"])
    if jaccards:
        flagged = sum(1 for row in summary.rows if row["over_deletion_flagged"])
        lines.append(
            f"  means  jaccard={sum(jaccards) / len(jaccards):.3f}  "
            f"overdel%={sum(pcts) / len(pcts):.1f}  "
            f"flagged={flagged}  "
            f"halluc-mean={sum(halluc_counts) / len(halluc_counts):.1f}"
        )
    if summary.missing_cleaned:
        lines.append(
            f"  golden rows without cleaned output: {len(summary.missing_cleaned)}"
        )
    if summary.missing_golden:
        lines.append(
            f"  cleaned rows without golden row: {len(summary.missing_golden)}"
        )
    return "\n".join(lines)
