"""Evaluate Exp 008 arm outputs against the hand-curated golden set.

Phase 1.4c — Experiment 008, Step 4. Text-level metrics only (pure
stdlib), reusing Exp 003's whitespace-tokenization methodology verbatim so
008 numbers stay comparable with 003's published rows:

- Jaccard similarity: word-set overlap of llm output vs golden text.
- Over-deletion share: |golden_words − llm_words| / |golden_words| (003
  flags > 5 percent).
- Hallucination count: |llm_words − raw_words| (003 counts raw vocab).
- Boilerplate marker hits on the llm output, reported twice: against a
  "raw" (003's full marker list, kept for regression comparability) and
  "raw-unambiguous" set (003's list minus the over-broad
  ``about us.*mission|values|purpose`` pattern, which false-positives on
  intentionally kept team-mission sentences; the unambiguous set is the
  plan's primary under-deletion signal).

Primary comparisons:
  C2 (old prompt, gpt-5.4-nano) vs 003 published numbers → rerun
  reproducibility (also byte-compares against jobs.json llm_clean_text).
  C1 vs C2 → isolated prompt effect. A1–A5 vs C1 → isolated model effect.

Writes scripts/008_boilerplate_only_cleaning/data/008_eval_results.json
(config, per-posting rows, per-arm summary, limitations) and prints a
per-arm summary table.

Usage:
    python scripts/008_boilerplate_only_cleaning/eval_cleaning.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from statistics import mean

from llm_client import price_per_token, pricing_source

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = SCRIPT_DATA_DIR / "008_corpus.json"
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_cleaned.json"  # shared golden set
OUTPUTS_PATH = SCRIPT_DATA_DIR / "008_clean_outputs.json"
JOBS_PATH = PROJECT_ROOT / "data" / "jobs.json"
RESULTS_PATH = SCRIPT_DATA_DIR / "008_eval_results.json"

# Exp 003's full marker list, verbatim (regression comparability).
MARKERS_003_TOKENS = [
    (
        "salary_comp",
        r"salary|pay\s*(grade|range|type)|base\s+pay|compensation|equity\s+grade",
    ),
    (
        "benefits",
        r"benefits|health\s+insurance|dental|vision|vacation|PTO|parental\s+leave",
    ),
    ("eeo", r"equal\s+opportunity\s+employer|EEO|\bdiversity\s+statement\b"),
    ("perks_testimonials", r"why\s+you('|’)\s*ll\s+love|testimonials|perks"),
    (
        "recruiter_notes",
        r"note\s+to\s+recruiters|unsolicited\s+resumes|recruiting\s+agency",
    ),
    ("about_us_mission", r"about\s+us\b.*mission|values|purpose"),
    ("salary_currency", r"\$[\d,]+.*(?:CAD|USD)|[\d,]+.*(?:CAD|USD).*\bsalary\b"),
]
# Unambiguous-only set (plan T4): 003's list minus the over-broad line.
MARKERS_UNAMBIGUOUS = [
    (token, pattern)
    for token, pattern in MARKERS_003_TOKENS
    if token != "about_us_mission"
]


def word_set(text: str) -> set[str]:
    """Whitespace-tokenize exactly as 003's eval (raw .lower().split())."""
    return set(text.lower().split())


def jaccard_similarity(words_a: set[str], words_b: set[str]) -> float:
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def marker_hits(text: str, markers: list[tuple[str, str]]) -> list[str]:
    lower = text.lower()
    return [token for token, pattern in markers if re.search(pattern, lower)]


def main() -> None:
    corpus = json.loads(CORPUS_PATH.read_text())
    raw_by_id = {p["posting_id"]: p["raw_full_text"] for p in corpus["postings"]}
    role_by_id = {p["posting_id"]: p["role_category"] for p in corpus["postings"]}
    golden_list = json.loads(GOLDEN_PATH.read_text())
    golden = {g["posting_id"]: g["text"] for g in golden_list}
    outputs = json.loads(OUTPUTS_PATH.read_text())
    jobs_003 = {j["id"]: j for j in json.loads(JOBS_PATH.read_text())}

    repro: list[dict] = []
    arm_words: dict[str, list[dict]] = {}
    per_posting: list[dict] = []

    for arm_id, arm in outputs["arms"].items():
        price_in, price_out = price_per_token(arm["model"])
        rows: list[dict] = []
        for posting_id, out in arm["outputs"].items():
            if posting_id not in golden:
                continue  # golden not curated yet for this posting
            golden_text = golden[posting_id]
            llm_text = out["text"]
            raw_text = raw_by_id[posting_id]
            g_words, l_words, r_words = (
                word_set(golden_text),
                word_set(llm_text),
                word_set(raw_text),
            )
            missing = g_words - l_words
            halluc = l_words - r_words
            cost = (
                out["usage"]["prompt_tokens"] * price_in
                + out["usage"]["completion_tokens"] * price_out
            )
            row = {
                "arm": arm_id,
                "posting_id": posting_id,
                "role_category": role_by_id[posting_id],
                "jaccard": jaccard_similarity(l_words, g_words),
                "over_deletion_pct": len(missing) / len(g_words) * 100
                if g_words
                else 0.0,
                "over_deletion_flagged": (
                    len(missing) / len(g_words) * 100 if g_words else 0
                )
                > 5,
                "over_deletion_sample": sorted(missing)[:10],
                "hallucination_count": len(halluc),
                "hallucination_sample": sorted(halluc)[:10],
                "markers_003": marker_hits(llm_text, MARKERS_003_TOKENS),
                "markers_unambiguous": marker_hits(llm_text, MARKERS_UNAMBIGUOUS),
                "output_words": len(llm_text.split()),
                "golden_words": len(g_words),
                "cost_usd": cost,
            }
            rows.append(row)
            per_posting.append(row)

            # Reproducibility: C2 temp-0.0 should byte-reproduce 003's output.
            if arm_id == "C2" and posting_id in jobs_003:
                ref = jobs_003[posting_id].get("llm_clean_text", "")
                repro.append(
                    {
                        "posting_id": posting_id,
                        "byte_identical_to_003": llm_text == ref,
                        "jaccard_to_003_output": jaccard_similarity(
                            word_set(llm_text), word_set(ref)
                        ),
                    }
                )
        if rows:
            arm_words[arm_id] = rows

    per_arm = {}
    LIMITATIONS = [
        (
            "Golden curation covers signed-off postings only; per-arm means use "
            "postings with a golden entry (39 target, 5 initially present)."
        ),
        (
            "12 of 39 postings are md-only and role-unlabeled; separation-gap "
            "style role analysis stays on the labeled 27 (see compare_variants.py)."
        ),
        (
            "Over-deletion threshold inherits 003's >5% flag heuristic; "
            "flags reviewed against the unambiguous marker set before judging "
            "an arm failed."
        ),
        f"Costs use {pricing_source()}",
    ]
    for arm_id, rows in arm_words.items():
        flagged = [r for r in rows if r["over_deletion_flagged"]]
        per_arm[arm_id] = {
            "model": outputs["arms"][arm_id]["model"],
            "prompt_variant": outputs["arms"][arm_id]["prompt_variant"],
            "n_postings_evaluated": len(rows),
            "mean_jaccard": mean(r["jaccard"] for r in rows),
            "mean_over_deletion_pct": mean(r["over_deletion_pct"] for r in rows),
            "n_over_deletion_flagged": len(flagged),
            "mean_hallucination_count": mean(r["hallucination_count"] for r in rows),
            "any_boilerplate_hits_003": sum(1 for r in rows if r["markers_003"]),
            "any_boilerplate_hits_unambiguous": sum(
                1 for r in rows if r["markers_unambiguous"]
            ),
            "mean_output_words": mean(r["output_words"] for r in rows),
            "mean_cost_usd_per_posting": mean(r["cost_usd"] for r in rows),
        }

    artifact = {
        "experiment": "008_boilerplate_only_cleaning",
        "evaluated_utc": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prompt_sha256_per_arm": {
            arm_id: arm.get("prompt_sha256") for arm_id, arm in outputs["arms"].items()
        },
        "models": {arm_id: arm["model"] for arm_id, arm in outputs["arms"].items()},
        "temperature": 0.0,
        "pricing_source": pricing_source(),
        "c2_reproducibility_vs_003": repro,
        "c2_byte_identical_share": (
            sum(1 for r in repro if r["byte_identical_to_003"]) / len(repro)
            if repro
            else None
        ),
        "per_arm": per_arm,
        "per_posting": per_posting,
        "limitations": LIMITATIONS,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    print(f"{'=' * 96}")
    print(f"EXP 008 EVAL — {len(per_posting)} (arm, posting) rows scored vs golden")
    print(f"{'=' * 96}")
    if repro:
        ident = sum(1 for r in repro if r["byte_identical_to_003"])
        print(
            f"\nC2 rerun reproducibility vs 003 jobs.json llm_clean_text: "
            f"{ident}/{len(repro)} byte-identical"
        )
    header = (
        f"{'arm':<5}{'model':<38}{'n':>3}{'Jac':>8}{'OverDel%':>10}"
        f"{'Flag':>5}{'Halluc':>8}{'Markers':>9}{'Words':>7}{'$/post':>9}"
    )
    print(f"\n{header}")
    print(f"{'-' * 96}")
    for arm_id, stats in per_arm.items():
        print(
            f"{arm_id:<5}{stats['model']:<38}{stats['n_postings_evaluated']:>3}"
            f"{stats['mean_jaccard']:>8.3f}{stats['mean_over_deletion_pct']:>10.1f}"
            f"{stats['n_over_deletion_flagged']:>5}"
            f"{stats['mean_hallucination_count']:>8.1f}"
            f"{stats['any_boilerplate_hits_unambiguous']:>9}"
            f"{stats['mean_output_words']:>7.0f}"
            f"{stats['mean_cost_usd_per_posting']:>9.5f}"
        )
    print(f"\nPer-posting detail → {RESULTS_PATH}")
    print("\nNote:", LIMITATIONS[2])


if __name__ == "__main__":
    main()
