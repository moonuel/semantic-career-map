"""CLI entry point: ``python -m backend.pipeline.eval`` (task 4.4).

Evaluates one cleaned-dataset version against the golden set and writes a
``eval/results/type=cleaning_quality/…`` Parquet version. Loads the
project-root ``.env`` (003 convention) and builds the real store; see
:func:`backend.pipeline.eval.run.run_evaluation` for the injectable variant
tests use. Prints the exp-003-style summary table after a successful write.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from backend.pipeline.eval import run
from backend.storage.store import ObjectStore


def main() -> None:
    """CLI flow: load golden + cleaned, evaluate, write results, print."""
    load_dotenv()  # STORE_S3_* live in .env (003 convention)
    parser = argparse.ArgumentParser(
        prog="python -m backend.pipeline.eval",
        description="Cleaning-quality evaluation over an artifacts dataset "
        "version (exp 003 metrics; results appended per version).",
    )
    parser.add_argument(
        "--artifacts-variant",
        default="llm",
        help="Producing method of the cleaned dataset version.",
    )
    parser.add_argument(
        "--artifacts-version",
        default="v1",
        help="Cleaned dataset version token (vN).",
    )
    parser.add_argument(
        "--golden-variant",
        default="author",
        help="Authoring method of the golden dataset version.",
    )
    parser.add_argument(
        "--golden-version",
        default="v1",
        help="Golden dataset version token (vN).",
    )
    parser.add_argument(
        "--results-variant",
        default="exp003",
        help="Results rubric variant name (path partition).",
    )
    parser.add_argument(
        "--evaluation-version",
        default="v1",
        help="Results version token (vN) — versions append, never overwrite.",
    )
    parser.add_argument(
        "--configuration-hash",
        default=None,
        help="Pin the cleaning configuration when the cleaned version "
        "carries more than one.",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Evaluate + print the table without writing a results part.",
    )
    args = parser.parse_args()

    store = ObjectStore.from_env()
    golden = run.load_golden(
        store, variant=args.golden_variant, version=args.golden_version
    )
    cleaned = run.load_cleaned(
        store,
        variant=args.artifacts_variant,
        version=args.artifacts_version,
        configuration_hash=args.configuration_hash,
        text_hashes={g.source_hash for g in golden},
    )
    summary = run.run_evaluation(
        store,
        golden=golden,
        cleaned=cleaned,
        write=not args.dry,
        results_variant=args.results_variant,
        evaluation_version=args.evaluation_version,
    )
    print(run.render_table(summary))
    if summary.part_key:
        print(f"  results part: {summary.part_key}")
    elif args.dry:
        print("  (dry run: no results part written)")


if __name__ == "__main__":
    main()
