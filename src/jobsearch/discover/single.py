"""Fetch a single posting from a URL the user pasted.

Used by ``jobsearch run <url>``. Recognises the three public ATS URL shapes and
uses their APIs rather than scraping the rendered page. For anything else it
falls back to a plain fetch of the page, but only when robots.txt permits it -
and it refuses outright for hosts whose terms forbid automated access.
"""

from __future__ import annotations

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
    """Last resort: read the company's own careers page, robots permitting."""
    body = fetcher.get(url)
    title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", body)
    title = strip_html(title_match.group(1)) if title_match else url
    company = _host(url).removeprefix("www.").split(".")[0].title()
    return JobPosting(
        company=company,
        title=title,
        url=url,
        source="career_page",
        description=strip_html(body)[:20000],
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
