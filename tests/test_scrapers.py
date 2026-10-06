"""Tests for :mod:`backend.ingestion.scrapers`.

The network-dependent functions (``list_job_urls``, ``scrape_job``) hit a live
source and are excluded from this hermetic suite; ``parse_job`` and the
JSON-LD extraction are pure and covered here against synthetic fixture HTML.
"""

from __future__ import annotations

import pytest

from backend.ingestion.scrapers import (
    _LDJSON_RE,
    _loc_urls,
    parse_job,
)

_JOB_URL = "https://dataroles.co/jobs/acme-senior-data-analyst-a1b2c3d4e5f6"

_JOBPAGE_HTML = """<html>
<head>
<script type="application/ld+json">{"@context":"https://schema.org/","@type":"Organization","name":"ACME"}</script>
<script type="application/ld+json">{
  "@context": "https://schema.org/",
  "@type": "JobPosting",
  "title": "Senior Data Analyst",
  "description": "<p>Analyze data</p>",
  "datePosted": "2026-10-05",
  "employmentType": "FULL_TIME",
  "directApply": false,
  "hiringOrganization": {"@type": "Organization", "name": "ACME"},
  "jobLocation": {"@type": "Place", "address": {
      "addressLocality": "McLean", "addressCountry": "US"}},
  "baseSalary": {"@type": "MonetaryAmount", "currency": "USD", "value": {
      "minValue": 70000, "maxValue": 90000, "unitText": "YEAR"}},
  "identifier": {"@type": "PropertyValue", "name": "workday",
      "value": "workday:acme:R1002"}
}</script>
</head>
<body></body>
</html>"""


class TestParseJob:
    """``parse_job`` extracts the ``JobPosting`` JSON-LD from a page."""

    def test_parses_full_job(self) -> None:
        job = parse_job(_JOBPAGE_HTML, _JOB_URL)
        assert job.url == _JOB_URL
        assert job.title == "Senior Data Analyst"
        assert job.description_html == "<p>Analyze data</p>"
        assert job.employer == "ACME"
        assert job.location == "McLean"
        assert job.country == "US"
        assert job.employment_type == "FULL_TIME"
        assert job.date_posted == "2026-10-05"
        assert job.salary_min == 70000
        assert job.salary_max == 90000
        assert job.salary_currency == "USD"
        assert job.source_id == "workday:acme:R1002"

    def test_missing_optional_fields_default_to_none(self) -> None:
        minimal = (
            '{"@type": "JobPosting", "title": "T", "description": "D",'
            ' "hiringOrganization": {"name": "X"}}'
        )
        html = f'<script type="application/ld+json">{minimal}</script>'
        job = parse_job(html, _JOB_URL)
        assert job.title == "T"
        assert job.location is None
        assert job.country is None
        assert job.employment_type is None
        assert job.date_posted is None
        assert job.salary_min is None
        assert job.salary_max is None
        assert job.salary_currency is None
        assert job.source_id is None

    def test_skips_non_jobposting_blocks(self) -> None:
        html = (
            '<script type="application/ld+json">{"@type":"Organization"}</script>'
            '<script type="application/ld+json">{"@type":"JobPosting",'
            '"title":"T","description":"D"}</script>'
        )
        job = parse_job(html, _JOB_URL)
        assert job.employer == ""

    def test_no_jobposting_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_job('<script type="application/ld+json">{"@type":"X"}</script>', _JOB_URL)

    def test_malformed_json_block_is_skipped(self) -> None:
        html = (
            '<script type="application/ld+json">{bad json}</script>'
            '<script type="application/ld+json">{"@type":"JobPosting",'
            '"title":"T","description":"D"}</script>'
        )
        job = parse_job(html, _JOB_URL)
        assert job.title == "T"


class TestLocUrls:
    """``_loc_urls`` extracts ``<loc>`` URLs from sitemap XML."""

    def test_extracts_locs(self) -> None:
        xml = (
            "<?xml version=\"1.0\"?>"
            "<urlset><url><loc>https://dataroles.co/a</loc></url>"
            "<url><loc>https://dataroles.co/b</loc></url></urlset>"
        )
        assert _loc_urls(xml) == ["https://dataroles.co/a", "https://dataroles.co/b"]

    def test_filters_other_namespaces(self) -> None:
        xml = (
            "<sitemapindex>"
            "<sitemap><loc>https://dataroles.co/sitemaps/jobs-data.xml</loc></sitemap>"
            "</sitemapindex>"
        )
        assert _loc_urls(xml) == ["https://dataroles.co/sitemaps/jobs-data.xml"]


class TestLdjsonRegex:
    """``_LDJSON_RE`` matches the ld+json blocks regardless of attribute order."""

    def test_matches_and_captures_body(self) -> None:
        html = (
            '<script type="application/ld+json">{"a":1}</script>'
            '<script src="x" type="application/ld+json"> {"b":2} </script>'
        )
        blocks = _LDJSON_RE.findall(html)
        assert len(blocks) == 2
        assert '{"a":1}' in blocks[0]
        assert '{"b":2}' in blocks[1]