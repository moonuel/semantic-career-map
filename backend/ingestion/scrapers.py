"""Scrapers for job-posting sources.

Currently two sources:

- :func:`fetch_job_board` reads a single company's job board from Ashby's
  public posting API.
- :func:`scrape_job` (and :func:`list_job_urls`) read postings from
  dataroles.co through its documented read interface — the sitemap enumerates
  every job-page URL, and each job page embeds a schema.org ``JobPosting``
  JSON-LD block whose ``description`` (raw HTML) and accompanying metadata are
  sufficient for ingestion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests

_SITEMAP_INDEX = "https://dataroles.co/sitemap.xml"
_JOB_SITEMAP_PREFIX = "https://dataroles.co/sitemaps/jobs-data"

# A ``<loc>`` URL in any sitemap namespace; matches both ``urlset`` and
# ``sitemapindex`` documents.
_LOC_RE = re.compile(r"<loc>\s*([^<]+?)\s*</loc>", re.DOTALL)

# Each job page carries one ``JobPosting`` JSON-LD block among others.
_LDJSON_RE = re.compile(
    r'<script\b[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', re.DOTALL
)


def fetch_job_board(job_board_name: str) -> requests.Response:
    """Fetch the posting list for an Ashby job board.

    Args:
        job_board_name: The Ashby board identifier, i.e. the slug after
            ``jobs.ashbyhq.com/``.

    Returns:
        The raw ``requests.Response``; callers call ``.json()`` themselves.
    """
    url = (
        "https://api.ashbyhq.com/posting-api/job-board/"
        f"{job_board_name}?includeCompensation=true"
    )
    return requests.get(url)


@dataclass(frozen=True)
class ScrapedJob:
    """A job posting parsed from dataroles.co's ``JobPosting`` JSON-LD.

    Attributes:
        url: The dataroles.co job-page URL that announced this posting.
        title: Role title, e.g. ``Senior Data Analyst``.
        description_html: Verbatim job description HTML from ``description``.
        employer: Organization name from ``hiringOrganization``.
        location: Locality from ``jobLocation.address.addressLocality``.
        country: Country code from ``jobLocation.address.addressCountry``.
        employment_type: e.g. ``FULL_TIME``.
        date_posted: ISO date from ``datePosted`` (listings confirmed open).
        salary_min: Employer-stated lower bound from ``baseSalary``, if any.
        salary_max: Employer-stated upper bound from ``baseSalary``, if any.
        salary_currency: ISO currency from ``baseSalary``, if any.
        source_id: The ATS id from ``identifier``, if present.
    """

    url: str
    title: str
    description_html: str
    employer: str
    location: str | None
    country: str | None
    employment_type: str | None
    date_posted: str | None
    salary_min: float | None
    salary_max: float | None
    salary_currency: str | None
    source_id: str | None


def list_job_urls(base: str = _SITEMAP_INDEX) -> list[str]:
    """Return every job-page URL on dataroles.co, via its sitemaps.

    This is the documented read path for automated readers
    (``docs/architecture``): the sitemap index references paged ``jobs-data``
    sitemaps, whose entries point at individual ``/jobs/`` pages.

    Args:
        base: The sitemap index URL; default is dataroles.co's.

    Returns:
        A list of job-page URLs, de-duplicated in document order.
    """
    seen: set[str] = set()
    urls: list[str] = []
    index = requests.get(base, timeout=30)
    index.raise_for_status()
    for sitemap_url in _loc_urls(index.text):
        if not sitemap_url.startswith(_JOB_SITEMAP_PREFIX):
            continue
        sitemap = requests.get(sitemap_url, timeout=30)
        sitemap.raise_for_status()
        for loc in _loc_urls(sitemap.text):
            if "/jobs/" in loc and loc not in seen:
                seen.add(loc)
                urls.append(loc)
    return urls


def scrape_job(url: str) -> ScrapedJob:
    """Fetch and parse one dataroles.co job page into a :class:`ScrapedJob`.

    Args:
        url: A job-page URL as returned by :func:`list_job_urls`.

    Returns:
        The parsed posting.

    Raises:
        requests.HTTPError: if the page cannot be retrieved.
        ValueError: if no ``JobPosting`` JSON-LD block is found.
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return parse_job(response.text, url)


def parse_job(html: str, url: str) -> ScrapedJob:
    """Parse the ``JobPosting`` JSON-LD block out of a dataroles.co page.

    Pure and I/O-free, so it is unit-testable against a fixture.

    Args:
        html: The raw response body of a dataroles.co job page.
        url: The page URL, recorded on the result.

    Returns:
        The parsed posting.

    Raises:
        ValueError: if the page has no ``JobPosting`` JSON-LD block.
    """
    for block in _LDJSON_RE.findall(html):
        try:
            obj: Any = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("@type") != "JobPosting":
            continue
        return _from_jobposting(obj, url)
    raise ValueError(f"no JobPosting JSON-LD found at {url}")


def _loc_urls(xml_text: str) -> list[str]:
    """Return the ``<loc>`` URL strings in a sitemap document."""
    return [loc.strip() for loc in _LOC_RE.findall(xml_text)]


def _from_jobposting(obj: dict[str, Any], url: str) -> ScrapedJob:
    """Build a :class:`ScrapedJob` from a parsed ``JobPosting`` object."""
    org = obj.get("hiringOrganization") or {}
    org_name = org.get("name") if isinstance(org, dict) else None

    place = obj.get("jobLocation") or {}
    address = place.get("address") if isinstance(place, dict) else None
    address = address if isinstance(address, dict) else {}

    salary = obj.get("baseSalary") or {}
    salary = salary if isinstance(salary, dict) else {}
    value = salary.get("value") or {}
    if isinstance(value, list):
        value = value[0] if value else {}
    value = value if isinstance(value, dict) else {}

    identifier = obj.get("identifier") or {}
    source_id = identifier.get("value") if isinstance(identifier, dict) else None

    return ScrapedJob(
        url=url,
        title=obj.get("title") or "",
        description_html=obj.get("description") or "",
        employer=org_name or "",
        location=address.get("addressLocality"),
        country=address.get("addressCountry"),
        employment_type=obj.get("employmentType"),
        date_posted=obj.get("datePosted"),
        salary_min=value.get("minValue"),
        salary_max=value.get("maxValue"),
        salary_currency=salary.get("currency"),
        source_id=source_id,
    )


if __name__ == "__main__":

    jobs = list_job_urls()

    for job in jobs[:1]:
        val = scrape_job(job)

        print(val)