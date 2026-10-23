"""Cleaning-quality metrics, extracted verbatim from exp 003 (task 4.2).

Exp 003 (`scripts/003_llm_extraction/eval_cleaning.py`, untouched) compared
LLM-cleaned text to the hand-cleaned golden set: Jaccard similarity, a
7-pattern boilerplate scan (under-deletion), hallucination detection
(word-set difference vs the raw text, first-10 sample), and over-deletion
(golden−cleaned word ratio with a 5% flag, first-10 sample). The metric
definitions below carry over verbatim for comparability across the legacy →
lake migration (design: metric definitions stay frozen at the exp 003
shapes; thresholds are data, not code drift).

Verbatim discipline: "words" are ``text.lower().split()`` terms everywhere;
``check_boilerplate`` reports the first alternative of each matched pattern
(the ``pattern.split("|")[0]`` idiom is exp 003's table identity); samples
are the first 10 words in set-iteration order, exactly as the exp printed
them (exp 008's ``sorted()[:10]`` is a later refinement — not ported).
"""

from __future__ import annotations

import re

# The 7 boilerplate markers (salary, benefits, EEO, testimonials, recruiter
# notes, company mission, salary amounts) — exp 003's list, unstirred.
BOILERPLATE_MARKERS: list[str] = [
    r"salary|pay\s*(grade|range|type)|base\s+pay|compensation|equity\s+grade",
    r"benefits|health\s+insurance|dental|vision|vacation|PTO|parental\s+leave",
    r"equal\s+opportunity\s+employer|EEO|\bdiversity\s+statement\b",
    r"why\s+you('|’)\s*ll\s+love|testimonials|perks",
    r"note\s+to\s+recruiters|unsolicited\s+resumes|recruiting\s+agency",
    r"about\s+us\b.*mission|values|purpose",
    r"\$[\d,]+.*(?:CAD|USD)|[\d,]+.*(?:CAD|USD).*\bsalary\b",
]

# Over-deletion above this golden-word percentage means the cleaning was
# too aggressive (exp 003 printed values, flagged when > 5%).
OVER_DELETION_FLAG_PCT = 5.0

# Sample truncation for human-facing word lists (hallucinations and
# over-deletions): exp 003 printed the first ten.
SAMPLE_LIMIT = 10


def word_set(text: str) -> set[str]:
    """Return the exp 003 word set: lowercased whitespace-split terms.

    Args:
        text: Any text.

    Returns:
        The set of lowercased, whitespace-separated words — the only
        tokenization every metric below uses.
    """
    return set(text.lower().split())


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity of two texts' word sets.

    Args:
        text_a: First text (e.g. the LLM-cleaned output).
        text_b: Second text (e.g. the golden reference).

    Returns:
        ``|A∩B| / |A∪B|``; 1.0 when both texts are empty, 0.0 when exactly
        one is empty.
    """
    words_a = word_set(text_a)
    words_b = word_set(text_b)
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def check_boilerplate(text: str) -> list[str]:
    """Scan text for boilerplate markers (the under-deletion signal).

    Args:
        text: The cleaned text under evaluation.

    Returns:
        One id per matched marker — the first alternative of the matched
        regex (``pattern.split("|")[0].strip()``), order of
        :data:`BOILERPLATE_MARKERS`.
    """
    matches = []
    lower = text.lower()
    for pattern in BOILERPLATE_MARKERS:
        if re.search(pattern, lower):
            matches.append(pattern.split("|")[0].strip())
    return matches


def hallucinated_words(cleaned_text: str, raw_text: str) -> set[str]:
    """Words the cleaned text introduced that the raw text never had.

    Args:
        cleaned_text: The LLM-cleaned output.
        raw_text: The posting's raw text (the shape text field, viewed
            through the same normalizer the cleaning stage saw).

    Returns:
        The word-set difference ``cleaned − raw`` (possibly empty).
    """
    return word_set(cleaned_text) - word_set(raw_text)


def over_deleted_words(golden_text: str, cleaned_text: str) -> tuple[float, set[str]]:
    """Percentage of golden words the cleaned text dropped, plus the words.

    Args:
        golden_text: The hand-cleaned golden reference.
        cleaned_text: The LLM-cleaned output.

    Returns:
        ``(pct, missing_word_set)`` where ``pct`` is the percent of golden
        words absent from the cleaned text (0.0 when the golden text is
        empty) — the exp 003 ratio, flagged when > :data:`OVER_DELETION_FLAG_PCT`.
    """
    golden_words = word_set(golden_text)
    missing = golden_words - word_set(cleaned_text)
    pct = len(missing) / len(golden_words) * 100 if golden_words else 0.0
    return pct, missing


def over_deletion_flagged(pct: float) -> bool:
    """Whether an over-deletion percentage crosses the exp 003 flag line.

    Args:
        pct: Over-deletion percentage (from :func:`over_deleted_words`).

    Returns:
        True when ``pct`` exceeds :data:`OVER_DELETION_FLAG_PCT`.
    """
    return pct > OVER_DELETION_FLAG_PCT


def first_sample(words: set[str]) -> list[str]:
    """The first :data:`SAMPLE_LIMIT` words of a word set, iteration order.

    Args:
        words: The word set (hallucinated or over-deleted words).

    Returns:
        Up to ten words in set-iteration order — exp 003's ``list(...)[:10]``.
    """
    return list(words)[:SAMPLE_LIMIT]
