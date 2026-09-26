"""Fetch a single posting from a URL the user pasted.

Used by ``jobsearch run <url>``. Recognises the three public ATS URL shapes and
uses their APIs rather than scraping the rendered page. For anything else it
falls back to a plain fetch of the page, but only when robots.txt permits it -
and it refuses outright for hosts whose terms forbid automated access.
"""

from __future__ import annotations

import json
import re
import urllib.parse

from ..config import BoardRef, Config
from ..models import JobPosting
from .sources import (
    DiscoveryError,
    Fetcher,
    is_public_ats_api,
    parse_ashby,
    parse_greenhouse,
    parse_lever,
    strip_html,
)


def _api_json(fetcher: Fetcher, url: str):
    """Fetch a documented public ATS API (robots.txt exempt - see sources.py)."""
    return fetcher.get_json(url, check_robots=not is_public_ats_api(url))

#: Hosts whose terms of service forbid automated collection. The tool refuses
#: rather than pretending it is a browser.
FORBIDDEN_HOSTS = (
    "linkedin.com",
    "indeed.com",
    "glassdoor.com",
    "monster.com",
    "ziprecruiter.com",
    "welcometothejungle.com",
)


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.lower()


def _board(company: str, ats: str, token: str, cfg: Config | None) -> BoardRef:
    """Prefer the configured board (it carries tier / sponsor / gaming flags)."""
    if cfg is not None:
        for board in cfg.boards:
            if board.ats == ats and board.token.lower() == token.lower():
                return board
    return BoardRef(company=company or token, ats=ats, token=token)


def fetch_single_posting(
    url: str, fetcher: Fetcher, cfg: Config | None = None
) -> JobPosting:
    """Resolve one posting URL into a :class:`JobPosting`."""
    host = _host(url)
    if any(host == h or host.endswith("." + h) for h in FORBIDDEN_HOSTS):
        raise DiscoveryError(
            f"Refusing to fetch {host}: its terms of service forbid automated access. "
            "Open the posting in a browser and use `jobsearch add --file <text-file>` "
            "with the description pasted in."
        )

    if "greenhouse.io" in host or "job-boards.greenhouse.io" in host:
        return _from_greenhouse(url, fetcher, cfg)
    if "lever.co" in host:
        return _from_lever(url, fetcher, cfg)
    if "ashbyhq.com" in host:
        return _from_ashby(url, fetcher, cfg)
    return _from_page(url, fetcher)


def _from_greenhouse(url: str, fetcher: Fetcher, cfg: Config | None) -> JobPosting:
    match = re.search(r"(?:boards|job-boards)\.greenhouse\.io/([^/?#]+)", url)
    job_match = re.search(r"/jobs/(\d+)", url)
    if not match:
        raise DiscoveryError(f"Could not read a Greenhouse board token from {url}")
    token = match.group(1)
    board = _board(token, "greenhouse", token, cfg)
    if job_match:
        payload = _api_json(
            fetcher,
            f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs/{job_match.group(1)}",
        )
        postings = parse_greenhouse({"jobs": [payload]}, board)
    else:
        postings = parse_greenhouse(
            _api_json(
                fetcher, f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
            ),
            board,
        )
    return _pick(postings, url)


def _from_lever(url: str, fetcher: Fetcher, cfg: Config | None) -> JobPosting:
    match = re.search(r"jobs\.lever\.co/([^/?#]+)", url)
    if not match:
        raise DiscoveryError(f"Could not read a Lever company token from {url}")
    token = match.group(1)
    board = _board(token, "lever", token, cfg)
    postings = parse_lever(
        _api_json(fetcher, f"https://api.lever.co/v0/postings/{token}?mode=json"), board
    )
    return _pick(postings, url)


def _from_ashby(url: str, fetcher: Fetcher, cfg: Config | None) -> JobPosting:
    match = re.search(r"jobs\.ashbyhq\.com/([^/?#]+)", url)
    if not match:
        raise DiscoveryError(f"Could not read an Ashby board name from {url}")
    token = match.group(1)
    board = _board(token, "ashby", token, cfg)
    postings = parse_ashby(
        _api_json(
            fetcher,
            f"https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true",
        ),
        board,
    )
    return _pick(postings, url)


def _from_page(url: str, fetcher: Fetcher) -> JobPosting:
    """Last resort: read the company's own careers page, robots permitting.

    Prefers the page's schema.org ``JobPosting`` block, which most career sites
    embed for search engines, over the rendered text. Locations come from every
    source on the page, because the structured block often names only the
    first site of a multi-location role: Phenom-hosted careers sites (Genmab,
    for one) list the rest in a separate ``multi_location`` array, and a
    Utrecht role recorded as "Princeton" fails the location constraint.
    """
    body = fetcher.get(url)
    return posting_from_page(url, body)


def posting_from_page(url: str, body: str) -> JobPosting:
    """Build a :class:`JobPosting` from a fetched careers page. Pure, no I/O."""
    ld = _ld_job_posting(body)
    locations = _dedupe_locations(
        [*(_ld_locations(ld) if ld else []), *_phenom_locations(body)]
    )
    if ld:
        org = ld.get("hiringOrganization")
        company = org.get("name", "") if isinstance(org, dict) else (org or "")
        return JobPosting(
            company=company or _company_from_host(url),
            title=strip_html(str(ld.get("title", ""))) or url,
            url=url,
            source="career_page",
            location="; ".join(locations),
            description=strip_html(str(ld.get("description", "")))[:20000],
            salary_text=_ld_salary(ld),
        )
    title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", body)
    title = strip_html(title_match.group(1)) if title_match else url
    return JobPosting(
        company=_company_from_host(url),
        title=title,
        url=url,
        source="career_page",
        location="; ".join(locations),
        description=strip_html(body)[:20000],
    )


def _company_from_host(url: str) -> str:
    host = _host(url).removeprefix("www.")
    parts = host.split(".")
    # careers.genmab.com -> Genmab, not "Careers"
    if len(parts) > 2 and parts[0] in {"careers", "jobs", "career", "work", "join"}:
        parts = parts[1:]
    return parts[0].title()


def _ld_job_posting(body: str) -> dict | None:
    """The first schema.org JobPosting in the page's JSON-LD blocks."""
    for match in re.finditer(
        r'(?is)<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', body
    ):
        try:
            data = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            item = stack.pop(0)
            if not isinstance(item, dict):
                continue
            kind = item.get("@type")
            if kind == "JobPosting" or (isinstance(kind, list) and "JobPosting" in kind):
                return item
            graph = item.get("@graph")
            if isinstance(graph, list):
                stack.extend(graph)
    return None


def _ld_locations(ld: dict) -> list[str]:
    raw = ld.get("jobLocation") or []
    places = raw if isinstance(raw, list) else [raw]
    out: list[str] = []
    for place in places:
        if not isinstance(place, dict):
            continue
        address = place.get("address") or {}
        if isinstance(address, str):
            out.append(address)
            continue
        country = address.get("addressCountry")
        if isinstance(country, dict):
            country = country.get("name", "")
        parts = [address.get("addressLocality"), address.get("addressRegion"), country]
        text = ", ".join(str(p) for p in parts if p)
        if text:
            out.append(text)
    if ld.get("jobLocationType") == "TELECOMMUTE":
        out.append("Remote")
    return out


def _phenom_locations(body: str) -> list[str]:
    """Locations from a Phenom page's ``multi_location`` array, if present."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r'"multi_location"\s*:\s*\[', body):
        try:
            items, _ = decoder.raw_decode(body, match.end() - 1)
        except json.JSONDecodeError:
            continue
        out: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            city = item.get("city") or item.get("cityState") or ""
            country = item.get("country") or ""
            text = ", ".join(p for p in (city, country) if p)
            if text:
                out.append(text)
        if out:
            return out
    return []


def _dedupe_locations(locations: list[str]) -> list[str]:
    """Keep the first mention of each city, in page order."""
    seen: set[str] = set()
    out: list[str] = []
    for loc in locations:
        key = loc.split(",")[0].strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(loc.strip())
    return out


def _ld_salary(ld: dict) -> str:
    salary = ld.get("baseSalary")
    if not isinstance(salary, dict):
        return ""
    value = salary.get("value") or {}
    currency = salary.get("currency", "")
    if isinstance(value, dict):
        low, high = value.get("minValue"), value.get("maxValue")
        unit = value.get("unitText", "")
        if low or high:
            span = " - ".join(f"{v:,.0f}" if isinstance(v, (int, float)) else str(v) for v in (low, high) if v)
            return " ".join(p for p in (currency, span, unit.lower() if unit else "") if p)
    return ""


_TRACKING_PARAMS = re.compile(r"^(utm_\w+|src|source|ref|gh_src|lever-source|trk|fbclid|gclid)$", re.I)


def clean_posting_url(url: str) -> str:
    """Drop tracking parameters so the same role always gets the same job id."""
    parts = urllib.parse.urlsplit(url)
    query = [
        (k, v) for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if not _TRACKING_PARAMS.match(k)
    ]
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(query), "")
    )


def _pick(postings: list[JobPosting], url: str) -> JobPosting:
    """Choose the posting whose URL matches, else the only one, else error."""
    for posting in postings:
        if posting.url and posting.url.rstrip("/") == url.rstrip("/"):
            return posting
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    for posting in postings:
        if tail and tail in posting.url:
            return posting
    if len(postings) == 1:
        return postings[0]
    raise DiscoveryError(
        f"Could not match {url} to a posting on that board "
        f"({len(postings)} postings seen). The role may have been taken down."
    )
