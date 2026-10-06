"""Tests for :mod:`backend.ingestion.scrapers`.

The network-dependent functions (``list_job_urls``, ``scrape_job``,
``fetch_ashby_jobs``, ``fetch_greenhouse_jobs``, ``fetch_lever_jobs``) hit live
sources and are excluded from this hermetic suite. The pure per-posting parsers
(``parse_job`` and the ``_from_*`` helpers) are covered here against synthetic
fixtures.
"""

from __future__ import annotations

import pytest

from backend.ingestion.scrapers import (
    _LDJSON_RE,
    _from_ashby_job,
    _from_greenhouse_job,
    _from_lever_job,
    _lever_description,
    _loc_urls,
    _salary_from_components,
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
            parse_job(
                '<script type="application/ld+json">{"@type":"X"}</script>', _JOB_URL
            )

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
            '<?xml version="1.0"?>'
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


# --- Ashby -------------------------------------------------------------------


class TestFromAshbyJob:
    """``_from_ashby_job`` normalizes one Ashby posting."""

    def _job(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "id": "7458d4e9-da2e-47bd-98cb-adfda43d42b2",
            "title": "Engineering Manager - EU",
            "locationName": "Remote - EU",
            "employmentType": "FullTime",
            "publishedAt": "2024-03-04T14:29:08.532+00:00",
            "jobUrl": "https://jobs.ashbyhq.com/acme/7458d4e9",
            "descriptionHtml": "<p>manage engineers</p>",
            "address": {"postalAddress": {"addressCountry": "European Union"}},
            "compensation": {
                "summaryComponents": [
                    {
                        "compensationType": "Salary",
                        "interval": "1 YEAR",
                        "currencyCode": "EUR",
                        "minValue": 110000,
                        "maxValue": 185000,
                        "id": "c1",
                    },
                    {
                        "compensationType": "Bonus",
                        "currencyCode": "EUR",
                        "minValue": None,
                        "maxValue": None,
                        "id": "c2",
                    },
                ]
            },
        }
        base.update(overrides)
        return base

    def test_full_job(self) -> None:
        job = _from_ashby_job(self._job(), "acme", "acme")
        assert job.employer == "acme"
        assert job.title == "Engineering Manager - EU"
        assert job.description_html == "<p>manage engineers</p>"
        assert job.location == "Remote - EU"
        assert job.country == "European Union"
        assert job.employment_type == "FullTime"
        assert job.date_posted == "2024-03-04T14:29:08.532+00:00"
        assert job.salary_min == 110000.0
        assert job.salary_max == 185000.0
        assert job.salary_currency == "EUR"
        assert job.source_id == "ashby:acme:7458d4e9-da2e-47bd-98cb-adfda43d42b2"

    def test_salary_selects_salary_component_over_bonus(self) -> None:
        job = _from_ashby_job(self._job(), "acme", "acme")
        assert job.salary_min == 110000.0

    def test_no_compensation_yields_none_salary(self) -> None:
        job = _from_ashby_job(self._job(compensation={}), "acme", "acme")
        assert job.salary_min is None
        assert job.salary_max is None
        assert job.salary_currency is None

    def test_missing_coveralls(self) -> None:
        job = _from_ashby_job({}, "acme", "acme")
        assert job.source_id == "ashby:acme:None"
        assert job.title == ""
        assert job.employer == "acme"


# --- Greenhouse --------------------------------------------------------------


class TestFromGreenhouseJob:
    """``_from_greenhouse_job`` normalizes one Greenhouse posting."""

    def _job(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "id": 8234044,
            "title": "Enterprise Sales Development Representative",
            "content": "<p>discipline</p>",
            "location": {"name": "Anywhere in the United States"},
            "updated_at": "2026-10-05T09:49:23-04:00",
            "first_published": "2026-09-28T09:18:45-04:00",
            "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/8234044",
            "company_name": "ACME",
            "offices": [{"location": "US"}],
        }
        base.update(overrides)
        return base

    def test_full_job(self) -> None:
        job = _from_greenhouse_job(self._job(), "ACME", "acme")
        assert job.employer == "ACME"
        assert job.title == "Enterprise Sales Development Representative"
        assert job.description_html == "<p>discipline</p>"
        assert job.location == "Anywhere in the United States"
        assert job.country == "US"
        assert job.date_posted == "2026-09-28T09:18:45-04:00"
        assert job.salary_min is None
        assert job.source_id == "greenhouse:acme:8234044"

    def test_missing_employment_type_ok(self) -> None:
        job = _from_greenhouse_job(self._job(employment_type=None), "ACME", "acme")
        assert job.employment_type is None

    def test_empty_job(self) -> None:
        job = _from_greenhouse_job({}, "ACME", "acme")
        assert job.source_id == "greenhouse:acme:None"
        assert job.title == ""


# --- Lever -------------------------------------------------------------------


class TestFromLeverJob:
    """``_from_lever_job`` normalizes one Lever posting."""

    def _job(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "id": "681fbc53-1e34-4a46-8677-3a78118674eb",
            "text": "Product Manager",
            "hostedUrl": "https://jobs.lever.co/acme/681fbc53-1e34",
            "createdAt": 1565990241800,
            "categories": {
                "location": "San Francisco, CA",
                "commitment": "Regular Full Time (Salary)",
                "team": "Product",
            },
            "country": "US",
            "workplaceType": "remote",
            "salaryRange": {
                "min": 150000,
                "max": 180000,
                "currency": "USD",
                "interval": "per-year-salary",
            },
            "description": "<p>lead product</p>",
            "lists": [{"text": "You have:", "content": "<ul><li>5yrs</li></ul>"}],
            "additional": "<p>EOE</p>",
        }
        base.update(overrides)
        return base

    def test_full_job(self) -> None:
        job = _from_lever_job(self._job(), "acme", "acme")
        assert job.employer == "acme"
        assert job.title == "Product Manager"
        assert job.location == "San Francisco, CA"
        assert job.country == "US"
        assert job.employment_type == "Regular Full Time (Salary)"
        assert job.date_posted == "2019-08-16T21:17:21Z"
        assert job.salary_min == 150000.0
        assert job.salary_max == 180000.0
        assert job.salary_currency == "USD"
        assert job.source_id == "lever:acme:681fbc53-1e34-4a46-8677-3a78118674eb"
        assert "lead product" in job.description_html
        assert "5yrs" in job.description_html
        assert "EOE" in job.description_html

    def test_no_salary(self) -> None:
        job = _from_lever_job(self._job(salaryRange=None), "acme", "acme")
        assert job.salary_min is None
        assert job.salary_max is None
        assert job.salary_currency is None

    def test_created_at_missing(self) -> None:
        job = _from_lever_job(self._job(createdAt=None), "acme", "acme")
        assert job.date_posted is None


class TestLeverDescription:
    """``_lever_description`` concatenates description + lists + additional."""

    def test_concatenates_all_parts(self) -> None:
        out = _lever_description(
            {
                "description": "<p>intro</p>",
                "lists": [
                    {"text": "Requirements", "content": "<ul><li>python</li></ul>"},
                    {"text": "About", "content": "<p>x</p>"},
                ],
                "additional": "<p>footer</p>",
            }
        )
        assert "<p>intro</p>" in out
        assert "Requirements" in out
        assert "<ul><li>python</li></ul>" in out
        assert "About" in out
        assert "<p>footer</p>" in out

    def test_handles_empty_lists_and_empty_additional(self) -> None:
        out = _lever_description(
            {"description": "<p>intro</p>", "lists": [], "additional": ""}
        )
        assert out == "<p>intro</p>"

    def test_ignores_non_dict_list_entries(self) -> None:
        out = _lever_description(
            {
                "description": "<p>i</p>",
                "lists": [{"text": "A", "content": "b"}, "junk"],
            }
        )
        assert "junk" not in out
        assert "A" in out


# --- Salary helpers ----------------------------------------------------------


class TestSalaryFromComponents:
    """``_salary_from_components`` picks the Salary component."""

    def test_picks_salary_component(self) -> None:
        comps = [
            {
                "compensationType": "Bonus",
                "minValue": None,
                "maxValue": None,
                "currencyCode": "EUR",
            },
            {
                "compensationType": "Salary",
                "minValue": 110000,
                "maxValue": 185000,
                "currencyCode": "EUR",
            },
        ]
        assert _salary_from_components(comps) == (110000.0, 185000.0, "EUR")

    def test_falls_back_to_first_bounded_component(self) -> None:
        comps = [
            {
                "compensationType": "Equity",
                "minValue": 50000,
                "maxValue": None,
                "currencyCode": "USD",
            }
        ]
        assert _salary_from_components(comps) == (50000.0, None, "USD")

    def test_none_for_empty(self) -> None:
        assert _salary_from_components([]) == (None, None, None)
        assert _salary_from_components(None) == (None, None, None)
