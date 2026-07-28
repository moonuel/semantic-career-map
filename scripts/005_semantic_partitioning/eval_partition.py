"""Evaluate LLM semantic partitioning against hand-curated golden fields.

Phase 2.0 — Validates that the two semantic dimensions (job-context and role-context)
are genuinely separable via LLM prompting. Computes Jaccard similarity, partition
overlap, and partition coverage against golden-set references.

Evaluates six phases:
  1. Golden partition validation — confirms hand-curated split is clean
  2. Job-context extraction quality — Pass A vs golden job-context
  3. Role-context extraction quality — Pass B vs golden role-context
  4. LLM partition integrity — confirms LLM outputs form a clean split
  5. Hallucination detection — checks for words not in undifferentiated source
  6. Per-posting detail — word counts and cross-contamination breakdowns

Usage:
    python scripts/005_semantic_partitioning/eval_partition.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PARTITIONS_PATH = PROJECT_ROOT / "data" / "005_semantic_partitions.json"


def jaccard_similarity(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def partition_overlap(text_a: str, text_b: str) -> float:
    return jaccard_similarity(text_a, text_b)


def partition_coverage(part_a: str, part_b: str, source: str) -> float:
    union = set(part_a.lower().split()) | set(part_b.lower().split())
    source_words = set(source.lower().split())
    if not source_words:
        return 1.0
    return len(union & source_words) / len(source_words)


def header(text: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}")


def main() -> None:
    with open(PARTITIONS_PATH, "r") as f:
        results = json.load(f)

    print(f"{'=' * 80}")
    print(f"SEMANTIC PARTITION EVALUATION — {len(results)} postings")
    print(f"{'=' * 80}")

    # ── Phase 1: Golden partition validation ──
    header("Phase 1 — Golden Partition Validation")
    print(f'{"Posting":<45s} {"Overlap":>8s} {"Coverage":>9s}')
    print(f"{'-' * 64}")
    golden_overlaps = []
    golden_coverages = []
    for r in results:
        overlap = partition_overlap(r["golden_job_context"], r["golden_role_context"])
        coverage = partition_coverage(
            r["golden_job_context"],
            r["golden_role_context"],
            r["undifferentiated_text"],
        )
        golden_overlaps.append(overlap)
        golden_coverages.append(coverage)
        print(f'{r["posting_id"]:<45s} {overlap:>8.4f} {coverage:>8.4f}')
    print(f"{'-' * 64}")
    print(
        f'{"MEAN":<45s} '
        f"{sum(golden_overlaps) / len(golden_overlaps):>8.4f} "
        f"{sum(golden_coverages) / len(golden_coverages):>8.4f}"
    )

    # ── Phase 2: Job-context extraction quality ──
    header("Phase 2 — Job-Context Extraction Quality (Pass A)")
    print(
        f'{"Posting":<45s} '
        f'{"Jacc(job,golden)":>18s} '
        f'{"Cross-contam":>13s}'
    )
    print(f"{'-' * 80}")
    job_jaccards = []
    job_cross = []
    for r in results:
        jacc = jaccard_similarity(
            r["llm_job_context"], r["golden_job_context"]
        )
        cross = jaccard_similarity(
            r["llm_job_context"], r["golden_role_context"]
        )
        job_jaccards.append(jacc)
        job_cross.append(cross)
        print(
            f'{r["posting_id"]:<45s} '
            f"{jacc:>18.4f} "
            f"{cross:>13.4f}"
        )
    print(f"{'-' * 80}")
    print(
        f'{"MEAN":<45s} '
        f"{sum(job_jaccards) / len(job_jaccards):>18.4f} "
        f"{sum(job_cross) / len(job_cross):>13.4f}"
    )

    # ── Phase 3: Role-context extraction quality ──
    header("Phase 3 — Role-Context Extraction Quality (Pass B)")
    print(
        f'{"Posting":<45s} '
        f'{"Jacc(role,golden)":>19s} '
        f'{"Cross-contam":>13s}'
    )
    print(f"{'-' * 81}")
    role_jaccards = []
    role_cross = []
    for r in results:
        jacc = jaccard_similarity(
            r["llm_role_context"], r["golden_role_context"]
        )
        cross = jaccard_similarity(
            r["llm_role_context"], r["golden_job_context"]
        )
        role_jaccards.append(jacc)
        role_cross.append(cross)
        print(
            f'{r["posting_id"]:<45s} '
            f"{jacc:>19.4f} "
            f"{cross:>13.4f}"
        )
    print(f"{'-' * 81}")
    print(
        f'{"MEAN":<45s} '
        f"{sum(role_jaccards) / len(role_jaccards):>19.4f} "
        f"{sum(role_cross) / len(role_cross):>13.4f}"
    )

    # ── Phase 4: LLM partition integrity ──
    header("Phase 4 — LLM Partition Integrity")
    print(
        f'{"Posting":<45s} '
        f'{"Overlap":>8s} '
        f'{"Coverage":>9s}'
    )
    print(f"{'-' * 66}")
    llm_overlaps = []
    llm_coverages = []
    for r in results:
        if not r["llm_job_context"] or not r["llm_role_context"]:
            continue
        overlap = partition_overlap(
            r["llm_job_context"], r["llm_role_context"]
        )
        coverage = partition_coverage(
            r["llm_job_context"],
            r["llm_role_context"],
            r["undifferentiated_text"],
        )
        llm_overlaps.append(overlap)
        llm_coverages.append(coverage)
        print(
            f'{r["posting_id"]:<45s} '
            f"{overlap:>8.4f} "
            f"{coverage:>8.4f}"
        )
    if llm_overlaps:
        print(f"{'-' * 66}")
        print(
            f'{"MEAN":<45s} '
            f"{sum(llm_overlaps) / len(llm_overlaps):>8.4f} "
            f"{sum(llm_coverages) / len(llm_coverages):>8.4f}"
        )

    # ── Phase 5: Hallucination detection ──
    header("Phase 5 — Hallucination Detection (LLM words not in source)")
    for r in results:
        source_words = set(r["undifferentiated_text"].lower().split())
        llm_job_words = set(r.get("llm_job_context", "").lower().split())
        llm_role_words = set(r.get("llm_role_context", "").lower().split())

        job_hallucinations = llm_job_words - source_words
        role_hallucinations = llm_role_words - source_words

        status_job = (
            f"*** {len(job_hallucinations)} HALLUCINATED WORDS"
            if job_hallucinations
            else "CLEAN"
        )
        status_role = (
            f"*** {len(role_hallucinations)} HALLUCINATED WORDS"
            if role_hallucinations
            else "CLEAN"
        )

        print(f'  [{r["posting_id"]}]')
        print(f"    Job-context (Pass A):  {status_job}")
        if job_hallucinations:
            print(
                f"      Sample: "
                f"{sorted(job_hallucinations)[:10]}"
            )
        print(f"    Role-context (Pass B): {status_role}")
        if role_hallucinations:
            print(
                f"      Sample: "
                f"{sorted(role_hallucinations)[:10]}"
            )

    # ── Phase 6: Per-posting detail ──
    header("Phase 6 — Per-Posting Detail")
    for r in results:
        print(f"\n  [{r['posting_id']}]")
        print(
            f"    Undifferentiated text:     "
            f"{len(r['undifferentiated_text'].split()):>5d} words"
        )
        print(
            f"    Golden job-context:        "
            f"{len(r['golden_job_context'].split()):>5d} words"
        )
        print(
            f"    Golden role-context:       "
            f"{len(r['golden_role_context'].split()):>5d} words"
        )
        print(
            f"    LLM job-context:           "
            f"{len(r.get('llm_job_context', '').split()):>5d} words"
        )
        print(
            f"    LLM role-context:          "
            f"{len(r.get('llm_role_context', '').split()):>5d} words"
        )
        print(
            f"    Jaccard(LLM job, golden job): "
            f"{jaccard_similarity(r['llm_job_context'], r['golden_job_context']):.4f}"
        )
        print(
            f"    Jaccard(LLM role, golden role): "
            f"{jaccard_similarity(r['llm_role_context'], r['golden_role_context']):.4f}"
        )
        print(
            f"    Cross-contam (job→role):   "
            f"{jaccard_similarity(r['llm_job_context'], r['golden_role_context']):.4f}"
        )
        print(
            f"    Cross-contam (role→job):   "
            f"{jaccard_similarity(r['llm_role_context'], r['golden_job_context']):.4f}"
        )

    print()


if __name__ == "__main__":
    main()
