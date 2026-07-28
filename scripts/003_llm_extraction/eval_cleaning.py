"""Evaluate LLM cleaning against hand-cleaned golden set.

Phase 1.4b — Compares llm_clean_text output to the golden_cleaned.json references.
Checks for boilerplate retention, skill deletion, and word-level similarity.

Usage:
    python scripts/003_llm_extraction/eval_cleaning.py
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
JOBS_PATH = PROJECT_ROOT / "data" / "jobs.json"
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_cleaned.json"

BOILERPLATE_MARKERS = [
    r"salary|pay\s*(grade|range|type)|base\s+pay|compensation|equity\s+grade",
    r"benefits|health\s+insurance|dental|vision|vacation|PTO|parental\s+leave",
    r"equal\s+opportunity\s+employer|EEO|\bdiversity\s+statement\b",
    r"why\s+you('|’)\s*ll\s+love|testimonials|perks",
    r"note\s+to\s+recruiters|unsolicited\s+resumes|recruiting\s+agency",
    r"about\s+us\b.*mission|values|purpose",
    r"\$[\d,]+.*(?:CAD|USD)|[\d,]+.*(?:CAD|USD).*\bsalary\b",
]


def jaccard_similarity(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def check_boilerplate(text: str) -> list[str]:
    matches = []
    lower = text.lower()
    for pattern in BOILERPLATE_MARKERS:
        if re.search(pattern, lower):
            matches.append(pattern.split("|")[0].strip())
    return matches


def main() -> None:
    with open(JOBS_PATH) as f:
        jobs = {j["id"]: j for j in json.load(f)}
    with open(GOLDEN_PATH) as f:
        golden_list = json.load(f)
    golden = {g["posting_id"]: g["text"] for g in golden_list}

    print(f"{'=' * 80}")
    print(f"LLM CLEANING EVALUATION — Golden Set ({len(golden)} postings)")
    print(f"{'=' * 80}\n")

    for posting_id in golden:
        if posting_id not in jobs:
            print(f"  MISSING: {posting_id} not in jobs.json\n")
            continue

        job = jobs[posting_id]
        llm_text = job.get("llm_clean_text", "")
        golden_text = golden[posting_id]

        if not llm_text:
            print(
                f"[{posting_id}] >>> No llm_clean_text field yet — run extract_clean_text.py first\n"
            )
            continue

        print(f"[{posting_id}]")

        jaccard = jaccard_similarity(llm_text, golden_text)
        print(f"  Jaccard similarity: {jaccard:.3f}")

        boilerplate_hits = check_boilerplate(llm_text)
        if boilerplate_hits:
            print(
                f"  *** UNDER-DELETION: boilterplate markers found: {boilerplate_hits}"
            )
        else:
            print("  Boilerplate check: CLEAN (no markers found)")

        raw_words = set(job["raw_full_text"].lower().split())
        llm_words = set(llm_text.lower().split())
        hallucinations = llm_words - raw_words
        if hallucinations:
            sample = list(hallucinations)[:10]
            print(
                f"  *** HALLUCINATION: {len(hallucinations)} words not in raw text. Sample: {sample}"
            )
        else:
            print("  Hallucination check: CLEAN (all words from raw text)")

        golden_words = set(golden_text.lower().split())
        missing_from_llm = golden_words - llm_words
        missing_pct = (
            len(missing_from_llm) / len(golden_words) * 100 if golden_words else 0
        )
        if missing_pct > 5:
            sample = list(missing_from_llm)[:10]
            print(
                f"  *** OVER-DELETION: {missing_pct:.1f}% of golden words missing. Sample: {sample}"
            )
        else:
            print(
                f"  Over-deletion: {missing_pct:.1f}% ({len(missing_from_llm)} words)"
            )

        print(f"  Golden length: {len(golden_text.split())} words")
        print(f"  LLM output:    {len(llm_text.split())} words")
        print()


if __name__ == "__main__":
    main()
