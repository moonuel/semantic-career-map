"""Tests for :mod:`backend.ingestion.scrapers`.

Three layers:

- **Fixture tests** capture real live payloads at capture time, store them
  under ``tests/fixtures/scrapers/`` (trimmed: one or two postings per source,
  descriptions truncated), and monkeypatch them into ``requests.get`` so the
  full fetch→normalize path runs hermetically against real-world shapes. The
  fixtures are the source of truth for each API's field names — the synthetic
  dataclasses below only cover edge cases the fixtures don't exercise.
- **Pure unit tests** cover regexes, salary-component selection, and
  missing/empty field defaults with hand-built inputs.
- **Connection tests** (marked ``network``) hit the live endpoints to confirm
  they are alive and still shaped like the fixtures. They are excluded by
  default (pyproject ``addopts``); run them with ``pytest -m network``.

Fixtures were captured with:
  - ``GET api.ashbyhq.com/posting-api/job-board/ashby?includeCompensation=true``
  - ``GET boards-api.greenhouse.io/v1/boards/greenhouse/jobs?content=true``
  - ``GET api.lever.co/v0/postings/leverdemo?mode=json``
  - a real ``dataroles.co/jobs/...`` page (ld+json blocks kept verbatim)
  - real ``sitemap.xml`` and a trimmed ``sitemaps/jobs-data.xml``
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import requests

from backend.ingestion.scrapers import (
    _ASHBY_API,
    _GREENHOUSE_API,
    _LDJSON_RE,
    _LEVER_API,
    _dataroles_board_fetch,
    _from_ashby_job,
    _from_dataroles_job,
    _from_greenhouse_job,
    _from_lever_job,
    _lever_description,
    _loc_urls,
    _salary_from_components,
    fetch_ashby_jobs,
    fetch_dataroles_job,
    fetch_dataroles_jobs,
    fetch_greenhouse_jobs,
    fetch_lever_jobs,
    list_dataroles_job_urls,
    parse_dataroles_job,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "scrapers"


def load_fixture(name: str) -> str:
    """Return the raw text of a captured fixture."""
    return (_FIXTURES / name).read_text()


class _FakeResponse:
    """Stand-in for ``requests.Response`` serving stored fixture data."""

    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def patch_routes(
    monkeypatch: pytest.MonkeyPatch,
    routes: list[tuple[Callable[[str], bool], _FakeResponse]],
) -> None:
    """Replace ``requests.get`` with a route table (first match wins)."""

    def fake_get(url: str, params: Any = None, timeout: Any = None) -> _FakeResponse:
        for match, response in routes:
            if match(url):
                return response
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr("backend.ingestion.scrapers.requests.get", fake_get)


def _json_ok(payload: dict[str, Any] | list[Any]) -> _FakeResponse:
    return _FakeResponse(json.dumps(payload))


# --- Ashby: fixture-pinned fetch ---------------------------------------------


class TestFetchAshbyJobsFixture:
    """``fetch_ashby_jobs`` over the captured Ashby board payload."""

    _payload = json.loads(load_fixture("ashby_board.json"))

    def test_normalizes_first_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_routes(
            monkeypatch,
            [
                (
                    lambda url: url.startswith(_ASHBY_API),
                    _FakeResponse(load_fixture("ashby_board.json")),
                )
            ],
        )
        jobs = fetch_ashby_jobs("ashby")
        assert len(jobs) == 1
        job = jobs[0]
        # Values below are the captured reality, not synthetic inventions.
        assert job.title == "Engineering Manager - EU"
        assert job.employer == "ashby"
        assert job.location == "Remote - European Union"
        assert job.employment_type == "FullTime"
        assert job.date_posted == "2024-03-04T14:29:08.532+00:00"
        assert (
            job.url
            == "https://jobs.ashbyhq.com/ashby/7458d4e9-da2e-47bd-98cb-adfda43d42b2"
        )
        assert job.salary_min == 110000.0
        assert job.salary_max == 185000.0
        assert job.salary_currency == "EUR"
        assert job.source_id == "ashby:ashby:7458d4e9-da2e-47bd-98cb-adfda43d42b2"
        assert job.description_html  # captured (truncated) description kept

    def test_employer_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_routes(
            monkeypatch,
            [
                (
                    lambda url: url.startswith(_ASHBY_API),
                    _FakeResponse(load_fixture("ashby_board.json")),
                )
            ],
        )
        assert fetch_ashby_jobs("ashby", employer="Ashby")[0].employer == "Ashby"


# --- Greenhouse: fixture-pinned fetch ----------------------------------------


class TestFetchGreenhouseJobsFixture:
    """``fetch_greenhouse_jobs`` over the captured Greenhouse board payload."""

    def test_normalizes_first_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_routes(
            monkeypatch,
            [
                (
                    lambda url: url.startswith(_GREENHOUSE_API),
                    _FakeResponse(load_fixture("greenhouse_board.json")),
                )
            ],
        )
        jobs = fetch_greenhouse_jobs("greenhouse")
        assert len(jobs) == 1
        job = jobs[0]
        # The real payload carries company_name on the job, not the envelope.
        assert job.employer == "Greenhouse"
        assert job.title == "Enterprise Sales Development Representative"
        assert job.location == "Anywhere in the United States"
        assert job.date_posted == "2026-09-28T09:18:45-04:00"
        assert job.source_id == "greenhouse:greenhouse:8234044"
        assert job.description_html  # captured (truncated) content kept

    def test_employer_override_beats_company_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_routes(
            monkeypatch,
            [
                (
                    lambda url: url.startswith(_GREENHOUSE_API),
                    _FakeResponse(load_fixture("greenhouse_board.json")),
                )
            ],
        )
        assert fetch_greenhouse_jobs("greenhouse", employer="GH")[-1].employer == "GH"


# --- Lever: fixture-pinned fetch ---------------------------------------------


class TestFetchLeverJobsFixture:
    """``fetch_lever_jobs`` over the captured Lever board payload."""

    _payload = json.loads(load_fixture("lever_board.json"))

    def test_normalizes_both_jobs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_routes(
            monkeypatch,
            [
                (
                    lambda url: url.startswith(_LEVER_API),
                    _FakeResponse(load_fixture("lever_board.json")),
                )
            ],
        )
        jobs = fetch_lever_jobs("leverdemo")
        assert len(jobs) == 2
        salary_job, plain_job = jobs

        assert salary_job.title == "Customer Success Manager"
        assert salary_job.location == "Atlanta, Georgia"
        assert salary_job.employment_type == "Regular Full Time (Salary)"
        assert salary_job.country == "US"
        assert salary_job.salary_min == 85000.0
        assert salary_job.salary_max == 175000.0
        assert salary_job.salary_currency == "USD"
        assert salary_job.date_posted == "2026-08-31T18:12:19Z"
        assert salary_job.source_id.startswith("lever:leverdemo:")
        assert salary_job.description_html

        assert plain_job.salary_min is None
        assert plain_job.salary_max is None
        assert plain_job.salary_currency is None
        assert plain_job.source_id.startswith("lever:leverdemo:")


# --- dataroles: fixture-pinned sitemap walk + page scrape --------------------


class TestDatarolesFetched:
    """``list_dataroles_job_urls`` + ``fetch_dataroles_job`` over fixtures."""

    def test_list_job_urls_walks_sitemaps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        index = load_fixture("sitemap_index.xml")
        jobs_sitemap = load_fixture("jobs_sitemap.xml")

        def fake_get(
            url: str, params: Any = None, timeout: Any = None
        ) -> _FakeResponse:
            if url.endswith("sitemap.xml"):
                return _FakeResponse(index)
            return _FakeResponse(jobs_sitemap)

        monkeypatch.setattr("backend.ingestion.scrapers.requests.get", fake_get)
        urls = list_dataroles_job_urls()
        # The index lists 4 non-job sitemaps (filtered) + 4 jobs sitemaps
        # (each returned as the same trimmed fixture in this test).
        assert len(urls) == 20  # trimmed fixture holds 20 real job URLs
        assert urls[0].startswith("https://dataroles.co/jobs/")
        assert all("/jobs/" in u for u in urls)

    def test_fetch_job_parses_captured_page(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = load_fixture("dataroles_job_page.html")

        def fake_get(
            url: str, params: Any = None, timeout: Any = None
        ) -> _FakeResponse:
            return _FakeResponse(page)

        monkeypatch.setattr("backend.ingestion.scrapers.requests.get", fake_get)
        job = fetch_dataroles_job(
            "https://dataroles.co/jobs/capital-one-senior-manager-data-analyst-product-analytics-mclean-c56e662b9a225015"
        )
        assert job.title == "Senior Manager, Data Analyst - Product Analytics"
        assert job.employer == "Capital One"
        assert job.location == "McLean"
        assert job.country == "US"
        assert job.employment_type == "FULL_TIME"
        assert job.date_posted == "2026-10-05"
        assert job.salary_min == 200700.0
        assert job.salary_max == 229100.0
        assert job.salary_currency == "USD"
        assert job.source_id == "workday:capitalone/wd12/capital_one:R1001957"
        assert job.url.startswith("https://dataroles.co/jobs/")
        assert job.description_html

    def test_end_to_end_walk_then_scrape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        index = load_fixture("sitemap_index.xml")
        jobs_sitemap = load_fixture("jobs_sitemap.xml")
        page = load_fixture("dataroles_job_page.html")

        def fake_get(
            url: str, params: Any = None, timeout: Any = None
        ) -> _FakeResponse:
            if url.endswith("sitemap.xml"):
                return _FakeResponse(index)
            if "jobs-data" in url:
                return _FakeResponse(jobs_sitemap)
            return _FakeResponse(page)

        monkeypatch.setattr("backend.ingestion.scrapers.requests.get", fake_get)
        urls = list_dataroles_job_urls()
        job = fetch_dataroles_job(urls[0])
        assert job.employer == "Capital One"
        assert job.title == "Senior Manager, Data Analyst - Product Analytics"


# --- dataroles: bounded sitemap walk -----------------------------------------


def _dataroles_routes() -> list[tuple[Callable[[str], bool], _FakeResponse]]:
    """Route table serving the captured dataroles index, sitemaps, and pages."""
    return [
        (
            lambda url: url.endswith("sitemap.xml"),
            _FakeResponse(load_fixture("sitemap_index.xml")),
        ),
        (
            lambda url: "jobs-data" in url,
            _FakeResponse(load_fixture("jobs_sitemap.xml")),
        ),
        (
            lambda url: "/jobs/" in url,
            _FakeResponse(load_fixture("dataroles_job_page.html")),
        ),
    ]


def _patch_dataroles_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the walk wiring: fixture routes plus a zero page delay."""
    patch_routes(monkeypatch, _dataroles_routes())
    monkeypatch.setattr("backend.ingestion.scrapers.time.sleep", lambda _s: None)


class TestFetchDatarolesJobs:
    """``fetch_dataroles_jobs`` — bounded walk of the curated corpus."""

    def test_unbounded_walk_fetches_whole_fixture_sitemap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dataroles_walk(monkeypatch)
        jobs = fetch_dataroles_jobs()
        assert len(jobs) == 20
        assert jobs[0].employer == "Capital One"

    def test_limit_bounds_pages_fetched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dataroles_walk(monkeypatch)
        jobs = fetch_dataroles_jobs(limit=3)
        assert len(jobs) == 3
        assert all(job.employer == "Capital One" for job in jobs)

    def test_page_delay_param_paces_each_get(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(
            "backend.ingestion.scrapers.time.sleep", lambda s: sleeps.append(s)
        )
        patch_routes(monkeypatch, _dataroles_routes())
        jobs = fetch_dataroles_jobs(limit=2, page_delay_s=0.5)
        assert len(jobs) == 2
        assert sleeps == [0.5, 0.5]

    def test_failed_page_skips_without_stopping_walk(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        page = load_fixture("dataroles_job_page.html")
        seen: list[str] = []

        def fake_get(
            url: str, params: Any = None, timeout: Any = None
        ) -> _FakeResponse:
            if url.endswith("sitemap.xml"):
                return _FakeResponse(load_fixture("sitemap_index.xml"))
            if "jobs-data" in url:
                return _FakeResponse(load_fixture("jobs_sitemap.xml"))
            seen.append(url)
            if len(seen) == 1:
                return _FakeResponse("", status_code=500)
            return _FakeResponse(page)

        monkeypatch.setattr("backend.ingestion.scrapers.requests.get", fake_get)
        monkeypatch.setattr("backend.ingestion.scrapers.time.sleep", lambda _s: None)

        jobs = fetch_dataroles_jobs()
        assert len(jobs) == 19
        captured = capsys.readouterr().err
        assert "skipping page" in captured
        assert seen[0] in captured

    def test_unparseable_page_skips_without_stopping_walk(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        page = load_fixture("dataroles_job_page.html")
        seen: list[str] = []

        def fake_get(
            url: str, params: Any = None, timeout: Any = None
        ) -> _FakeResponse:
            if url.endswith("sitemap.xml"):
                return _FakeResponse(load_fixture("sitemap_index.xml"))
            if "jobs-data" in url:
                return _FakeResponse(load_fixture("jobs_sitemap.xml"))
            seen.append(url)
            if len(seen) == 1:
                return _FakeResponse(
                    '<script type="application/ld+json">{"@type":"Organization"}</script>'
                )
            return _FakeResponse(page)

        monkeypatch.setattr("backend.ingestion.scrapers.requests.get", fake_get)
        monkeypatch.setattr("backend.ingestion.scrapers.time.sleep", lambda _s: None)

        jobs = fetch_dataroles_jobs()
        assert len(jobs) == 19
        assert "skipping page" in capsys.readouterr().err

    def test_budget_counts_attempts_not_successes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = load_fixture("dataroles_job_page.html")
        seen: list[str] = []

        def fake_get(
            url: str, params: Any = None, timeout: Any = None
        ) -> _FakeResponse:
            if url.endswith("sitemap.xml"):
                return _FakeResponse(load_fixture("sitemap_index.xml"))
            if "jobs-data" in url:
                return _FakeResponse(load_fixture("jobs_sitemap.xml"))
            seen.append(url)
            if len(seen) == 1:
                return _FakeResponse("", status_code=500)
            return _FakeResponse(page)

        monkeypatch.setattr("backend.ingestion.scrapers.requests.get", fake_get)
        monkeypatch.setattr("backend.ingestion.scrapers.time.sleep", lambda _s: None)

        jobs = fetch_dataroles_jobs(limit=2)
        assert len(jobs) == 1
        assert len(seen) == 2

    def test_all_failed_pages_fail_the_board(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seen: list[str] = []

        def fake_get(
            url: str, params: Any = None, timeout: Any = None
        ) -> _FakeResponse:
            if url.endswith("sitemap.xml"):
                return _FakeResponse(load_fixture("sitemap_index.xml"))
            if "jobs-data" in url:
                return _FakeResponse(load_fixture("jobs_sitemap.xml"))
            seen.append(url)
            return _FakeResponse("", status_code=500)

        monkeypatch.setattr("backend.ingestion.scrapers.requests.get", fake_get)
        monkeypatch.setattr("backend.ingestion.scrapers.time.sleep", lambda _s: None)

        with pytest.raises(requests.RequestException, match="none of"):
            fetch_dataroles_jobs()
        assert len(seen) == 20
        assert capsys.readouterr().err.count("skipping page") == 20

    def test_non_dataroles_page_url_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        page = load_fixture("dataroles_job_page.html")
        sitemap = (
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            "<url><loc>https://other.example.net/jobs/sneaky</loc></url>"
            "<url><loc>https://dataroles.co/jobs/real</loc></url>"
            "</urlset>"
        )
        seen: list[str] = []

        def fake_get(
            url: str, params: Any = None, timeout: Any = None
        ) -> _FakeResponse:
            if url.endswith("sitemap.xml"):
                return _FakeResponse(load_fixture("sitemap_index.xml"))
            if "jobs-data" in url:
                return _FakeResponse(sitemap)
            seen.append(url)
            return _FakeResponse(page)

        monkeypatch.setattr("backend.ingestion.scrapers.requests.get", fake_get)
        monkeypatch.setattr("backend.ingestion.scrapers.time.sleep", lambda _s: None)

        jobs = fetch_dataroles_jobs()
        assert len(jobs) == 1
        assert "https://dataroles.co/jobs/real" in seen
        assert all("other.example.net" not in got for got in seen)
        err = capsys.readouterr().err
        assert "skipping page" in err
        assert "other.example.net" in err

    def test_non_https_dataroles_page_url_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        page = load_fixture("dataroles_job_page.html")
        sitemap = (
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            "<url><loc>http://dataroles.co/jobs/downgraded</loc></url>"
            "<url><loc>https://dataroles.co/jobs/real</loc></url>"
            "</urlset>"
        )
        seen: list[str] = []

        def fake_get(
            url: str, params: Any = None, timeout: Any = None
        ) -> _FakeResponse:
            if url.endswith("sitemap.xml"):
                return _FakeResponse(load_fixture("sitemap_index.xml"))
            if "jobs-data" in url:
                return _FakeResponse(sitemap)
            seen.append(url)
            return _FakeResponse(page)

        monkeypatch.setattr("backend.ingestion.scrapers.requests.get", fake_get)
        monkeypatch.setattr("backend.ingestion.scrapers.time.sleep", lambda _s: None)

        jobs = fetch_dataroles_jobs()
        assert len(jobs) == 1
        assert seen == ["https://dataroles.co/jobs/real"]
        assert "skipping page" in capsys.readouterr().err


class TestBoardFetchDispatch:
    """``_dataroles_board_fetch`` — slug validation at the dispatch edge."""

    def test_rejects_uncurated_slug(self) -> None:
        with pytest.raises(ValueError, match="curated"):
            _dataroles_board_fetch("acme")


# --- Pure unit tests ---------------------------------------------------------


class TestParseDatarolesJob:
    """Edge cases around the JSON-LD extraction."""

    def _parse(self, html: str) -> Any:
        return parse_dataroles_job(html, "https://dataroles.co/jobs/x")

    def test_missing_optional_fields_default_to_none(self) -> None:
        minimal = (
            '{"@type": "JobPosting", "title": "T", "description": "D",'
            ' "hiringOrganization": {"name": "X"}}'
        )
        job = self._parse(f'<script type="application/ld+json">{minimal}</script>')
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
        assert self._parse(html).employer == ""

    def test_no_jobposting_raises(self) -> None:
        with pytest.raises(ValueError):
            self._parse('<script type="application/ld+json">{"@type":"X"}</script>')

    def test_direct_helper_matches_parse(self) -> None:
        obj = {
            "@type": "JobPosting",
            "title": "T",
            "description": "D",
            "hiringOrganization": {"name": "X"},
        }
        html = (
            '<script type="application/ld+json">{"@type":"JobPosting",'
            '"title":"T","description":"D","hiringOrganization":{"name":"X"}}'
            "</script>"
        )
        assert _from_dataroles_job(obj, "u") == parse_dataroles_job(html, "u")

    def test_malformed_json_block_is_skipped(self) -> None:
        html = (
            '<script type="application/ld+json">{bad json}</script>'
            '<script type="application/ld+json">{"@type":"JobPosting",'
            '"title":"T","description":"D"}</script>'
        )
        assert self._parse(html).title == "T"


class TestLocUrls:
    """``_loc_urls`` extracts ``<loc>`` URLs from sitemap XML."""

    def test_parses_captured_jobs_sitemap(self) -> None:
        urls = _loc_urls(load_fixture("jobs_sitemap.xml"))
        assert len(urls) == 20
        assert all(u.startswith("https://dataroles.co/jobs/") for u in urls)

    def test_parses_captured_index(self) -> None:
        urls = _loc_urls(load_fixture("sitemap_index.xml"))
        assert len(urls) == 8  # pages, hubs, salaries, companies, jobs-data×4
        assert sum("jobs-data" in u for u in urls) == 4


class TestLdjsonRegex:
    """``_LDJSON_RE`` finds ld+json blocks regardless of attribute order."""

    def test_matches_and_captures_body(self) -> None:
        html = (
            '<script type="application/ld+json">{"a":1}</script>'
            '<script src="x" type="application/ld+json"> {"b":2} </script>'
        )
        blocks = _LDJSON_RE.findall(html)
        assert len(blocks) == 2
        assert '{"a":1}' in blocks[0]
        assert '{"b":2}' in blocks[1]


class TestFromAshbyJob:
    """``_from_ashby_job`` edge cases on hand-built payloads."""

    def _job(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "id": "7458d4e9-da2e-47bd-98cb-adfda43d42b2",
            "title": "Engineering Manager - EU",
            "location": "Remote - EU",
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

    def test_salary_selects_salary_component_over_bonus(self) -> None:
        assert _from_ashby_job(self._job(), "acme", "acme").salary_min == 110000.0

    def test_no_compensation_yields_none_salary(self) -> None:
        job = _from_ashby_job(self._job(compensation={}), "acme", "acme")
        assert job.salary_min is None
        assert job.salary_max is None
        assert job.salary_currency is None

    def test_location_falls_back_to_postal_address(self) -> None:
        job = _from_ashby_job(self._job(location=None), "acme", "acme")
        # postal addressLocality is empty in the fixture; falls through to None.
        assert job.location is None

    def test_missing_coveralls(self) -> None:
        job = _from_ashby_job({}, "acme", "acme")
        assert job.source_id == "ashby:acme:None"
        assert job.title == ""
        assert job.employer == "acme"


class TestFromGreenhouseJob:
    """``_from_greenhouse_job`` edge cases on hand-built payloads."""

    def _job(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "id": 8234044,
            "title": "Enterprise Sales Development Representative",
            "content": "<p>discipline</p>",
            "location": {"name": "Anywhere in the United States"},
            "updated_at": "2026-10-05T09:49:23-04:00",
            "first_published": "2026-09-28T09:18:45-04:00",
            "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/8234044",
            "offices": [{"location": "US"}],
        }
        base.update(overrides)
        return base

    def test_first_published_beats_updated_at(self) -> None:
        job = _from_greenhouse_job(self._job(), "ACME", "acme")
        assert job.date_posted == "2026-09-28T09:18:45-04:00"

    def test_falls_back_to_updated_at(self) -> None:
        job = _from_greenhouse_job(self._job(first_published=None), "ACME", "acme")
        assert job.date_posted == "2026-10-05T09:49:23-04:00"

    def test_country_from_first_office(self) -> None:
        job = _from_greenhouse_job(self._job(), "ACME", "acme")
        assert job.country == "US"
        job = _from_greenhouse_job(self._job(offices=[]), "ACME", "acme")
        assert job.country is None

    def test_empty_job(self) -> None:
        job = _from_greenhouse_job({}, "ACME", "acme")
        assert job.source_id == "greenhouse:acme:None"
        assert job.title == ""


class TestFromLeverJob:
    """``_from_lever_job`` edge cases on hand-built payloads."""

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

    def test_millis_to_iso(self) -> None:
        job = _from_lever_job(self._job(), "acme", "acme")
        assert job.date_posted == "2019-08-16T21:17:21Z"

    def test_no_salary(self) -> None:
        job = _from_lever_job(self._job(salaryRange=None), "acme", "acme")
        assert job.salary_min is None
        assert job.salary_max is None
        assert job.salary_currency is None

    def test_created_at_missing(self) -> None:
        job = _from_lever_job(self._job(createdAt=None), "acme", "acme")
        assert job.date_posted is None

    def test_created_at_non_numeric(self) -> None:
        job = _from_lever_job(self._job(createdAt="not-a-number"), "acme", "acme")
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


# --- Connection tests (live) -------------------------------------------------


@pytest.mark.network
class TestConnections:
    """The live endpoints are up and still shaped like the fixtures.

    Run with ``pytest -m network``. These are smoke tests only: they pin the
    global envelope (status, top-level shape, non-emptiness) and assert each
    posting parses into the documented field types. Field VALUES still live
    in the fixtures — a live change should turn these on and re-capture the
    fixtures alongside them.
    """

    def test_ashby_live(self) -> None:
        jobs = fetch_ashby_jobs("ashby")
        assert jobs
        job = jobs[0]
        assert job.title
        assert job.employer == "ashby"
        assert job.source_id.startswith("ashby:ashby:")
        assert isinstance(job.location, str)

    def test_greenhouse_live(self) -> None:
        jobs = fetch_greenhouse_jobs("greenhouse")
        assert jobs
        job = jobs[0]
        assert job.title
        assert job.employer  # company_name from the live payload
        assert job.source_id.startswith("greenhouse:greenhouse:")
        assert job.description_html

    def test_lever_live(self) -> None:
        jobs = fetch_lever_jobs("leverdemo")
        assert jobs
        job = jobs[0]
        assert job.title
        assert job.employer == "leverdemo"
        assert job.source_id.startswith("lever:leverdemo:")

    def test_dataroles_live_bounded_walk(self) -> None:
        jobs = fetch_dataroles_jobs(limit=2)
        assert len(jobs) == 2
        assert all(job.title for job in jobs)
        assert all(job.url.startswith("https://dataroles.co/jobs/") for job in jobs)
