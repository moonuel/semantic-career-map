"""HTML → plain-text normalization for the cleaning stage (task 3.2).

Deterministic, in-process (design D-normalize: no standalone
``normalized/`` Parquet layer at this scale): shape-1 payloads carry raw
``description_html`` and the LLM stage must receive markup-free plain
text, so normalization runs inside the cleaning module character by
character. The no-HTML invariant ("cleaned-artifacts" spec) is asserted
by unit tests over synthetic fixture HTML.

Guarantees and their bounds:

- **No HTML tags survive** — well-formed tags and ``<script>``/
  ``<style>`` bodies are removed, including tags that surface from
  double-encoded source text.
- **Content is never summarized** — normalization touches markup only;
  boilerplate sections (compensation, benefits, EEO) pass through intact
  for the LLM stage to delete, per the exp-008 prompt contract.
- **Structures become whitespace** — block closes/newlines, list items
  gain a line-leading ``- `` bullet (tags may carry meaning; headers and
  list structure are the LLM's section context), cells gain a space.
- **Entities are resolved two levels deep** — the exp-009 noise
  inventory found greenhouse ``description_html`` double
  entity-encoded (``&lt;div&gt;`` literal), so the pass is
  unescape → strip → unescape → strip, capped at two rounds. Text that
  intentionally displays markup-like strings (``<div>`` as prose) is
  therefore stripped too — accepted loss, made safe because raw is
  immutable (design risk table).

Shape contract (design.md Context): all six sources normalize through
their own text field — ``description_html`` for the ScrapedJob shape
(ashby, dataroles, greenhouse, lever), plain-text ``description`` for
hiringcafe, markdown ``content`` for linkedin (normalization is
near-identity there; markdown constructs are not HTML).
"""

from __future__ import annotations

import html
import re

# The text field each source partition normalizes from (design.md Context
# — shapes 2–3 have no description_html).
SHAPE_TEXT_FIELD: dict[str, str] = {
    "ashby": "description_html",
    "dataroles": "description_html",
    "greenhouse": "description_html",
    "lever": "description_html",
    "hiringcafe": "description",
    "linkedin": "content",
}

# Fault containers go with their content (JS/CSS bodies are not posting text).
_SCRIPT_OR_STYLE = re.compile(
    r"<(script|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_DOCTYPE = re.compile(r"<!\s*doctype[^>]*>", re.IGNORECASE)

# Structure-bearing marks → whitespace/bullets.
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HR = re.compile(r"<hr\s*/?>", re.IGNORECASE)
_LIST_ITEM = re.compile(r"<li\b[^>]*>", re.IGNORECASE)
_BLOCK_BREAK = re.compile(
    r"</?(?:p|div|h[1-6]|table|tr|ul|ol|blockquote|section|header|footer)\b[^>]*>",
    re.IGNORECASE,
)
_CELL = re.compile(r"</?t[hd]\b[^>]*>", re.IGNORECASE)

# Well-formed tags only: the name starts with a letter, so prose like
# "<2 years" survives the strip (unlike a greedy "<[^>]+" pattern).
_ANY_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9:._-]*\b[^<>]*>")


def normalize_html(text: str) -> str:
    """Reduce HTML (or already-plain text) to markup-free plain text.

    Two unescape→strip rounds handle the double-encoded noise the raw
    inventory recorded; round two is usually a no-op. Whitespace is then
    normalized: line-leading/trailing spaces stripped, intra-line runs
    collapsed to one space, a paragraph break capped at one blank line.

    Args:
        text: The raw text field (HTML, plain text, or markdown).

    Returns:
        Plain text with no HTML tags and normalized whitespace.
    """
    for _ in range(2):
        text = html.unescape(text)
        text = _SCRIPT_OR_STYLE.sub(" ", text)
        text = _COMMENT.sub(" ", text)
        text = _DOCTYPE.sub(" ", text)
        text = _BR.sub("\n", text)
        text = _HR.sub("\n", text)
        text = _LIST_ITEM.sub("\n- ", text)
        text = _BLOCK_BREAK.sub("\n", text)
        text = _CELL.sub(" ", text)
        text = _ANY_TAG.sub("", text)
    return _normalize_whitespace(text)


def _normalize_whitespace(text: str) -> str:
    """Collapse whitespace artifacts left by tag removal.

    Args:
        text: Post-strip text, possibly with ragged line breaks and
            spacing runs from removed inline tags.

    Returns:
        Text with ``\\r``-family breaks unified, per-line padding
        stripped, intra-line space runs collapsed, and blank-line runs
        capped at one paragraph break. Leading/trailing blank lines are
        removed.
    """
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[\xa0\t]+", " ", text)
    lines = [re.sub(r" {2,}", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
