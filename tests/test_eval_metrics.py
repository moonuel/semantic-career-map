"""Tests for the verbatim exp 003 cleaning-quality metrics (task 4.2).

Two verification layers:

1. Hand-computed synthetic fixtures pin exact metric semantics (empty-input
   law, marker identification, sample truncation, flag threshold).
2. ``tests/fixtures/exp003/`` reproduces exp 003's per-posting table from
   ``docs/experiments/003-llm-extraction.md`` on the legacy inputs: the
   golden texts pinned verbatim at the exp-003-era commit (c5f4f5b) and the
   ``llm_clean_text`` fields of ``data/jobs.json`` — the same inputs the
   experiment's table was computed from.
"""

import json
from pathlib import Path

import pytest

from backend.pipeline.eval import metrics

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "exp003"


# --- hand-computed synthetic semantics -----------------------------------


def test_jaccard_identical_and_disjoint_and_empty() -> None:
    assert metrics.jaccard_similarity("same words set", "words set same") == 1.0
    assert metrics.jaccard_similarity("alpha beta", "gamma delta") == 0.0
    assert metrics.jaccard_similarity("", "") == 1.0
    assert metrics.jaccard_similarity("", "words") == 0.0
    assert metrics.jaccard_similarity("words", "") == 0.0


def test_jaccard_exact_intersection_value() -> None:
    # words: A={alpha beta gamma}, B={beta gamma delta} → 2/4
    assert metrics.jaccard_similarity("alpha beta gamma", "gamma delta beta") == 0.5


def test_word_set_is_lowercased_whitespace_split() -> None:
    assert metrics.word_set("Alpha,  BETA beta!") == {"alpha,", "beta", "beta!"}


def test_check_boilerplate_reports_first_alternative() -> None:
    # The id of a matched family is the pattern's first alternative,
    # taken verbatim from the regex (escapes included — exp 003 printed
    # the raw ``pattern.split("|")[0].strip()`` strings).
    assert metrics.check_boilerplate("includes health insurance.") == ["benefits"]
    assert metrics.check_boilerplate("the pay range is generous.") == ["salary"]
    assert metrics.check_boilerplate("We are an equal opportunity employer.") == [
        "equal\\s+opportunity\\s+employer"
    ]
    assert metrics.check_boilerplate("About us: our mission drives us.") == [
        "about\\s+us\\b.*mission"
    ]
    assert metrics.check_boilerplate("why you'll love working here") == [r"why\s+you('"]
    assert metrics.check_boilerplate("Note to recruiters: no calls.") == [
        r"note\s+to\s+recruiters"
    ]


def test_check_boilerplate_clean_and_multiple() -> None:
    assert metrics.check_boilerplate("Build great systems. Ship.") == []
    # Two independent families report two ids in marker order (salary,
    # then benefits)
    hits = metrics.check_boilerplate("salary bands plus health insurance and PTO.")
    assert hits == ["salary", "benefits"]


def test_hallucinated_words_is_word_set_difference() -> None:
    assert metrics.hallucinated_words("alpha beta GAMMA", "alpha beta raw") == {"gamma"}
    assert metrics.hallucinated_words("alpha", "alpha") == set()


def test_over_deleted_words_ratio_and_flag() -> None:
    # golden={a..f} six words, cleaned={a b} → 4/6 = 66.7%
    pct, missing = metrics.over_deleted_words("a b c d e f", "a b")
    assert round(pct, 1) == 66.7
    assert missing == {"c", "d", "e", "f"}
    assert metrics.over_deletion_flagged(pct)
    assert not metrics.over_deletion_flagged(1.0)
    assert metrics.over_deletion_flagged(metrics.OVER_DELETION_FLAG_PCT + 0.01)
    # Strictly-greater-than flag line: exactly 5 of 45 words gone = 11.11%… still
    # above; a case at exactly the threshold needs 1/20 words → flagged anyway,
    # so pin the boundary with a direct constant check
    pct5 = 100 / 20  # 5.0 exactly
    assert metrics.over_deletion_flagged(5.0001)
    assert not metrics.over_deletion_flagged(5.0)
    assert pct5 == 5.0
    # Empty golden: 0.0 by the exp's law
    assert metrics.over_deleted_words("", "any text")[0] == 0.0


def test_first_sample_caps_at_ten_in_iteration_order() -> None:
    words = {f"w{i}" for i in range(25)}
    sample = metrics.first_sample(words)
    assert len(sample) == 10
    assert sample == list(words)[:10]
    assert metrics.first_sample({"one"}) == ["one"]


# --- exp 003 per-posting table reproduction ------------------------------


def _load_fixture(posting_id: str) -> dict[str, str]:
    return json.loads((FIXTURES / f"{posting_id}.json").read_text(encoding="utf-8"))


def test_marker_count_is_seven() -> None:
    assert len(metrics.BOILERPLATE_MARKERS) == 7


@pytest.mark.parametrize(
    "posting_id",
    [
        "affirm-ml-engineer-2",
        "bmo-associate-data-scientist",
        "hellofresh-ml-engineer-operations-technology",
        "mastercard-data-scientist-2",
        "scribd-data-scientist-2",
    ],
)
def test_reproduces_exp_003_table_row(posting_id: str) -> None:
    expected = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))[
        posting_id
    ]
    fx = _load_fixture(posting_id)
    golden_text, llm_text = fx["golden_text"], fx["llm_clean_text"]

    jaccard = metrics.jaccard_similarity(llm_text, golden_text)
    assert round(jaccard, 3) == expected["jaccard"], posting_id

    pct, _ = metrics.over_deleted_words(golden_text, llm_text)
    assert round(pct, 1) == expected["over_deletion_pct"], posting_id
    # exp 003 flagged all five rows as OVER-DELETION (> 5% of golden words
    # missing) — that flag is the doc table's over-deletion column meaning
    assert metrics.over_deletion_flagged(pct), posting_id

    assert len(golden_text.split()) == expected["golden_words"], posting_id
    assert len(llm_text.split()) == expected["llm_words"], posting_id

    # exp 003 found no boilerplate in any of the five outputs
    assert metrics.check_boilerplate(llm_text) == []
