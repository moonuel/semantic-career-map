"""Tests for html→plain-text normalization (task 3.2).

Synthetic fixture HTML covers the verification clause: tags never
survive, boilerplate sections stay intact for the LLM stage, entities
resolve (including the greenhouse double-encoding case), and structure
becomes whitespace/bullets — never deleted content.
"""

from __future__ import annotations

import re

from backend.pipeline.cleaning.normalization import (
    SHAPE_TEXT_FIELD,
    normalize_html,
)
from backend.storage import keys

# Invariant: no well-formed tag can survive (letter/! after <).
_NOT_HTML = re.compile(r"</?[a-zA-Z!]")


def assert_no_tags(text: str) -> None:
    assert not _NOT_HTML.search(text), f"tag survived: {text!r}"


def test_headers_lists_entities_fixture() -> None:
    fixture = (
        "<h2>Role</h2>"
        "<ul><li>Build pipelines</li><li>Ship models</li></ul>"
        '<h2>About the Company</h2><p>We&#39;re funded &amp; growing &mdash; '
        "salary $120k, full benefits &amp; equity.</p>"
        "<script>track(1);</script>"
    )
    text = normalize_html(fixture)
    assert_no_tags(text)
    assert "Role" in text
    assert "\n- Build pipelines" in text
    assert "\n- Ship models" in text
    assert "About the Company" in text
    # boilerplate section intact for the LLM stage (LLM deletes it later)
    assert "salary $120k, full benefits & equity." in text
    assert "We're funded & growing —" in text
    assert "track(1);" not in text  # script body removed with the tags


def test_double_entity_encoded_greenhouse_case() -> None:
    fixture = (
        "&lt;div&gt;We are a steeped-in-data team.&lt;/div&gt;"
        "&lt;p&gt;Requirements&lt;/p&gt; 5+ years experience"
    )
    text = normalize_html(fixture)
    assert_no_tags(text)
    assert "&lt;" not in text
    assert "We are a steeped-in-data team." in text
    assert "Requirements" in text
    assert "5+ years experience" in text


def test_angle_bracket_prose_survives() -> None:
    text = normalize_html("<p>Note: requires &lt;2 years of experience</p>")
    assert_no_tags(text)
    # a digit name can't form a tag; prose keeps its bracket
    assert "<2 years of experience" in text


def test_whitespace_normalization() -> None:
    fixture = "<p>First&nbsp;&nbsp;section\r\n   </p>\n<br><br>\n\n\n\n<p>Second</p>"
    text = normalize_html(fixture)
    assert_no_tags(text)
    assert "First section" in text
    assert "\n\n\n" not in text
    assert not any(line != line.strip() for line in text.split("\n"))


def test_table_cells_spaced_and_rows_broken() -> None:
    text = normalize_html("<table><tr><td>Base</td><td>120k</td></tr></table>")
    assert_no_tags(text)
    assert "Base" in text and "120k" in text


def test_plain_text_is_whitespace_identity() -> None:
    assert normalize_html("clean plain text stays") == "clean plain text stays"


def test_markdown_content_is_not_mangled() -> None:
    fixture = "# Data Engineer\n\nExperienced with **python** and `sql`.\n"
    text = normalize_html(fixture)
    assert "**python**" in text
    assert "`sql`" in text


def test_inline_tags_removed_and_text_glued() -> None:
    text = normalize_html('<a href="https://x.example">Data</a> <b>Engineer</b>')
    assert_no_tags(text)
    assert "Data Engineer" in text
    assert "href" not in text


def test_empty_and_whitespace_degenerate_input() -> None:
    assert normalize_html("") == ""
    assert normalize_html("&nbsp;   <p></p> ") == ""


def test_shape_text_field_covers_all_sources() -> None:
    assert set(SHAPE_TEXT_FIELD) == set(keys.RAW_SOURCES)
    html_fields = {s for s, f in SHAPE_TEXT_FIELD.items() if f == "description_html"}
    assert html_fields == {"ashby", "dataroles", "greenhouse", "lever"}
    assert SHAPE_TEXT_FIELD["hiringcafe"] == "description"
    assert SHAPE_TEXT_FIELD["linkedin"] == "content"
