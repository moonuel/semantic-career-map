"""Scrapers for job-posting sources.

Each established ATS has a public, key-free read API that returns a whole
company board in one call. The three supported here — Ashby, Greenhouse, and
Lever — all normalize into the same :class:`ScrapedJob` shape, with
``source_id`` set from each ATS's native posting id as ``<ats>:<board>:<id>``
so that postings can be de-duplicated across sources later:

- :func:`fetch_ashby_jobs` — ``api.ashbyhq.com/posting-api/job-board/{slug}``
- :func:`fetch_greenhouse_jobs` — ``boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true``
- :func:`fetch_lever_jobs` — ``api.lever.co/v0/postings/{slug}?mode=json``

dataroles.co, a curator over 16 ATS systems, is also supported via its
documented read interface (:func:`list_dataroles_job_urls` and
:func:`fetch_dataroles_job`), whose
``identifier.value`` carries the origin embedded on every posting.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests

_SITEMAP_INDEX = "https://dataroles.co/sitemap.xml"
_JOB_SITEMAP_PREFIX = "https://dataroles.co/sitemaps/jobs-data"

_ASHBY_API = "https://api.ashbyhq.com/posting-api/job-board"
_GREENHOUSE_API = "https://boards-api.greenhouse.io/v1/boards"
_LEVER_API = "https://api.lever.co/v0/postings"

# A ``<loc>`` URL in any sitemap namespace; matches both ``urlset`` and
# ``sitemapindex`` documents.
_LOC_RE = re.compile(r"<loc>\s*([^<]+?)\s*</loc>", re.DOTALL)

# Each job page carries one ``JobPosting`` JSON-LD block among others.
_LDJSON_RE = re.compile(
    r'<script\b[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', re.DOTALL
)


def fetch_ashby_jobs(
    job_board_name: str, employer: str | None = None
) -> list[ScrapedJob]:
    """Fetch and normalize all postings on an Ashby job board.

    ``GET api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true``
    returns the whole board. ``includeCompensation=true`` is required for the
    ``compensation`` object to be present at all.

    Args:
        job_board_name: The Ashby board identifier (slug after ``jobs.ashbyhq.com/``).
        employer: Optional override; Ashby does not name the employer, so it
            defaults to ``job_board_name``.

    Returns:
        A list of :class:`ScrapedJob`, one per open posting.

    Raises:
        requests.HTTPError: if the board cannot be retrieved (unknown boards
            return HTTP 404).
    """
    response = requests.get(
        f"{_ASHBY_API}/{job_board_name}?includeCompensation=true", timeout=30
    )
    response.raise_for_status()
    payload = response.json()
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else payload
    return [
        _from_ashby_job(job, employer or job_board_name, job_board_name)
        for job in jobs
        if isinstance(job, dict)
    ]


def fetch_greenhouse_jobs(
    board_token: str, employer: str | None = None
) -> list[ScrapedJob]:
    """Fetch and normalize all postings on a Greenhouse job board.

    ``GET boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true``
    returns the whole board with full HTML descriptions in one call.
    ``content=true`` is what makes ``content`` (the description) non-empty.

    Args:
        board_token: The Greenhouse board token (slug after
            ``job-boards.greenhouse.io/``).
        employer: Optional override; defaults to the API's ``company_name``.

    Returns:
        A list of :class:`ScrapedJob`, one per open posting.

    Raises:
        requests.HTTPError: if the board cannot be retrieved (unknown tokens
            return HTTP 404).
    """
    response = requests.get(
        f"{_GREENHOUSE_API}/{board_token}/jobs", params={"content": "true"}, timeout=30
    )
    response.raise_for_status()
    payload = response.json()
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
    base_employer = employer or (
        next(
            (
                job["company_name"]
                for job in jobs
                if isinstance(job, dict) and job.get("company_name")
            ),
            None,
        )
    )
    return [
        _from_greenhouse_job(job, base_employer or board_token, board_token)
        for job in jobs
        if isinstance(job, dict)
    ]


def fetch_lever_jobs(site: str, employer: str | None = None) -> list[ScrapedJob]:
    """Fetch and normalize all postings on a Lever job site.

    ``GET api.lever.co/v0/postings/{site}?mode=json`` returns the full board as
    a bare JSON array. ``mode=json`` (or an ``Accept: application/json``
    header) is required — without it the endpoint serves a rendered page.

    Args:
        site: The Lever board slug (after ``jobs.lever.co/``).
        employer: Optional override; Lever never names the company, so it
            defaults to ``site``.

    Returns:
        A list of :class:`ScrapedJob`, one per open posting.

    Raises:
        requests.HTTPError: if the board cannot be retrieved (unknown sites
            return HTTP 404).
    """
    response = requests.get(f"{_LEVER_API}/{site}", params={"mode": "json"}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    jobs = payload if isinstance(payload, list) else []
    return [
        _from_lever_job(job, employer or site, site)
        for job in jobs
        if isinstance(job, dict)
    ]


@dataclass(frozen=True)
class ScrapedJob:
    """A job posting normalized from any supported source.

    The canonical output shape for every source (Ashby, Greenhouse, Lever,
    dataroles.co). For the ATS scrapers, ``source_id`` is the native posting id
    as ``<ats>:<board>:<id>`` so postings can be joined back to a specific
    origin and de-duplicated across the individual scrapers and the
    dataroles.co curator.

    Attributes:
        url: The public posting URL on the source/employer site.
        title: Role title, e.g. ``Senior Data Analyst``.
        description_html: Verbatim job description HTML.
        employer: Organization name.
        location: Free-text location or locality.
        country: Country code, if the source provides a structured one.
        employment_type: e.g. ``FULL_TIME`` (or the source's equivalent).
        date_posted: ISO date/timestamp of posting, if provided.
        salary_min: Employer-stated lower bound, if any.
        salary_max: Employer-stated upper bound, if any.
        salary_currency: ISO currency, if any.
        source_id: Native posting id as ``<ats>:<board>:<id>``, e.g.
            ``ashby:acme:7458d4e9``; a dataroles.co posting carries its origin
            system's identifier value verbatim instead.
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


def list_dataroles_job_urls(base: str = _SITEMAP_INDEX) -> list[str]:
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


def fetch_dataroles_job(url: str) -> ScrapedJob:
    """Fetch and parse one dataroles.co job page into a :class:`ScrapedJob`.

    Args:
        url: A job-page URL as returned by :func:`list_dataroles_job_urls`.

    Returns:
        The parsed posting.

    Raises:
        requests.HTTPError: if the page cannot be retrieved.
        ValueError: if no ``JobPosting`` JSON-LD block is found.
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return parse_dataroles_job(response.text, url)


def parse_dataroles_job(html: str, url: str) -> ScrapedJob:
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
        return _from_dataroles_job(obj, url)
    raise ValueError(f"no JobPosting JSON-LD found at {url}")


def _loc_urls(xml_text: str) -> list[str]:
    """Return the ``<loc>`` URL strings in a sitemap document."""
    return [loc.strip() for loc in _LOC_RE.findall(xml_text)]


def _from_dataroles_job(obj: dict[str, Any], url: str) -> ScrapedJob:
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


def _from_ashby_job(job: dict[str, Any], employer: str, slug: str) -> ScrapedJob:
    """Build a :class:`ScrapedJob` from one Ashby posting object."""
    compensation = job.get("compensation") or {}
    compensation = compensation if isinstance(compensation, dict) else {}
    comp = _salary_from_components(compensation.get("summaryComponents"))

    address = job.get("address") or {}
    postal = address.get("postalAddress") if isinstance(address, dict) else None
    postal = postal if isinstance(postal, dict) else {}

    return ScrapedJob(
        url=job.get("jobUrl") or "",
        title=job.get("title") or "",
        description_html=job.get("descriptionHtml") or "",
        employer=employer,
        location=job.get("location") or postal.get("addressLocality"),
        country=postal.get("addressCountry"),
        employment_type=job.get("employmentType"),
        date_posted=job.get("publishedAt"),
        salary_min=comp[0],
        salary_max=comp[1],
        salary_currency=comp[2],
        source_id=f"ashby:{slug}:{job.get('id')}",
    )


def _from_greenhouse_job(job: dict[str, Any], employer: str, token: str) -> ScrapedJob:
    """Build a :class:`ScrapedJob` from one Greenhouse posting object."""
    location = job.get("location") or {}
    location = location if isinstance(location, dict) else {}
    offices = job.get("offices") or []
    country = next(
        (o.get("location") for o in offices if isinstance(o, dict)),
        None,
    )

    return ScrapedJob(
        url=job.get("absolute_url") or "",
        title=job.get("title") or "",
        description_html=job.get("content") or "",
        employer=employer,
        location=location.get("name"),
        country=country,
        employment_type=job.get("employment_type"),
        date_posted=job.get("first_published") or job.get("updated_at"),
        salary_min=None,
        salary_max=None,
        salary_currency=None,
        source_id=f"greenhouse:{token}:{job.get('id')}",
    )


def _from_lever_job(job: dict[str, Any], employer: str, site: str) -> ScrapedJob:
    """Build a :class:`ScrapedJob` from one Lever posting object."""
    categories = job.get("categories") or {}
    categories = categories if isinstance(categories, dict) else {}
    salary = job.get("salaryRange") or {}
    salary = salary if isinstance(salary, dict) else {}
    commitment = categories.get("commitment")
    country = job.get("country")

    source_id = f"lever:{site}:{job.get('id')}"
    return ScrapedJob(
        url=job.get("hostedUrl") or "",
        title=job.get("text") or "",
        description_html=_lever_description(job),
        employer=employer,
        location=categories.get("location"),
        country=country,
        employment_type=commitment,
        date_posted=_from_millis(job.get("createdAt")),
        salary_min=_as_float(salary.get("min")),
        salary_max=_as_float(salary.get("max")),
        salary_currency=salary.get("currency"),
        source_id=source_id,
    )


def _lever_description(job: dict[str, Any]) -> str:
    """Assemble a Lever posting's full HTML body.

    Lever splits a posting across ``description``, ``lists`` (requirements /
    responsibilities), and ``additional``. Concatenating them gives the full
    body; ``description`` alone would look truncated.
    """
    parts: list[str] = []
    if job.get("description"):
        parts.append(job["description"])
    lists = job.get("lists") or []
    for block in lists:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        content = block.get("content")
        if text or content:
            parts.append(
                f"{text}<br>{content}" if text and content else text or content
            )
    if job.get("additional"):
        parts.append(job["additional"])
    return "<br>".join(parts)


def _salary_from_components(
    components: Any,
) -> tuple[float | None, float | None, str | None]:
    """Reduce an Ashby ``summaryComponents`` array to (min, max, currency).

    A posting can carry several components (base salary plus separate equity /
    bonus tiers). The base pay is the component whose ``compensationType`` is
    ``Salary``; when absent, fall back to the first component that has bounds.
    ``minValue``/``maxValue`` are ``None`` for equity/bonus tiers.
    """
    if not isinstance(components, list):
        return None, None, None
    salary = next(
        (
            c
            for c in components
            if isinstance(c, dict) and c.get("compensationType") == "Salary"
        ),
        None,
    )
    pick = salary or next(
        (
            c
            for c in components
            if isinstance(c, dict) and c.get("minValue") is not None
        ),
        None,
    )
    if not isinstance(pick, dict):
        return None, None, None
    return (
        _as_float(pick.get("minValue")),
        _as_float(pick.get("maxValue")),
        pick.get("currencyCode"),
    )


def _as_float(value: Any) -> float | None:
    """Coerce an int/float to ``float``, else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _from_millis(value: Any) -> str | None:
    """Format a millisecond Unix epoch as an ISO UTC string, else ``None``."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return datetime.fromtimestamp(value / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    jobs = list_dataroles_job_urls()

    for job in jobs[:1]:
        val = fetch_dataroles_job(job)

        print(val)
