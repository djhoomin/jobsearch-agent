"""Job discovery from *public, permitted* sources only.

What this module will talk to:

* Greenhouse public job board API - ``boards-api.greenhouse.io``
* Lever public postings API - ``api.lever.co/v0/postings``
* Ashby public job board API - ``api.ashbyhq.com/posting-api/job-board``
* Plain JSON feeds a company publishes on its own careers page

What it will **never** talk to: LinkedIn, Indeed, Glassdoor, or any site whose
terms of service forbid automated access. Those are handled by handing the user
a URL to click, not by a scraper. See the README's "What this deliberately does
not do".

Politeness: a real User-Agent identifying the tool and a contact address, an
inter-request delay, and a robots.txt check before fetching any *page*.

A note on robots.txt and the three ATS APIs. ``api.ashbyhq.com/robots.txt``
disallows everything, because it is written for web crawlers indexing pages.
The endpoints used here are the vendors' own *documented public job-board APIs*,
published precisely so that employers' listings can be consumed
programmatically. Applying a crawler-directed robots.txt to a documented API
would block the legitimate use it exists for, so those three hosts are exempt -
and only those three. Every other fetch (a company's own JSON feed, a careers
page) is robots-checked, and the hosts whose terms forbid automation are refused
outright in ``single.py``. The rate limit and User-Agent apply to everything.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from html import unescape
from typing import Any, Callable, Iterable

from ..config import BoardRef, Config
from ..models import JobPosting

log = logging.getLogger(__name__)

#: Documented public job-board APIs. See the module docstring for why these are
#: exempt from the robots.txt check.
PUBLIC_ATS_HOSTS = frozenset(
    {"boards-api.greenhouse.io", "api.lever.co", "api.ashbyhq.com"}
)

GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
LEVER_URL = "https://api.lever.co/v0/postings/{token}?mode=json"
ASHBY_URL = "https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true"


class DiscoveryError(RuntimeError):
    """A source could not be read. Never fatal: other sources still run."""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


@dataclass
class Fetcher:
    """Rate-limited, robots-aware HTTP GET returning decoded text."""

    user_agent: str = "jobsearch-agent/0.1"
    rate_limit_seconds: float = 1.0
    timeout_seconds: float = 20.0
    respect_robots_txt: bool = True
    _last_request: float = 0.0
    _robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._robots_cache is None:
            self._robots_cache = {}

    @classmethod
    def from_config(cls, cfg: Config) -> "Fetcher":
        section = cfg.section("discover")
        return cls(
            user_agent=section.get("user_agent", "jobsearch-agent/0.1"),
            rate_limit_seconds=float(section.get("rate_limit_seconds", 1.0)),
            timeout_seconds=float(section.get("timeout_seconds", 20)),
            respect_robots_txt=bool(section.get("respect_robots_txt", True)),
        )

    def allowed(self, url: str) -> bool:
        """Check robots.txt for ``url``. Fail open only when robots is absent."""
        if not self.respect_robots_txt:
            return True
        parts = urllib.parse.urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots_cache:
            parser: urllib.robotparser.RobotFileParser | None = urllib.robotparser.RobotFileParser()
            parser.set_url(f"{origin}/robots.txt")
            try:
                parser.read()
            except Exception:  # no robots.txt served, or unreachable
                parser = None
            self._robots_cache[origin] = parser
        parser = self._robots_cache[origin]
        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)

    def get(self, url: str, check_robots: bool = True) -> str:
        if check_robots and not self.allowed(url):
            raise DiscoveryError(f"robots.txt disallows fetching {url}")
        self._throttle()
        request = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept": "application/json, */*"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except urllib.error.HTTPError as exc:
            raise DiscoveryError(f"HTTP {exc.code} from {url}") from exc
        except urllib.error.URLError as exc:
            raise DiscoveryError(f"Could not reach {url}: {exc.reason}") from exc

    def get_json(self, url: str, check_robots: bool = True) -> Any:
        body = self.get(url, check_robots=check_robots)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise DiscoveryError(f"{url} did not return JSON: {exc}") from exc

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if self._last_request and elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self._last_request = time.monotonic()


# ---------------------------------------------------------------------------
# Per-ATS adapters. Each takes the decoded JSON payload and yields JobPostings.
# Kept pure (no I/O) so they are trivially testable against recorded payloads.
# ---------------------------------------------------------------------------


def strip_html(value: str) -> str:
    """Flatten an HTML job description to readable plain text."""
    if not value:
        return ""
    text = unescape(value)
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|h[1-6]|tr)>", "\n", text)
    text = re.sub(r"(?i)<li[^>]*>", "- ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_greenhouse(payload: Any, board: BoardRef) -> list[JobPosting]:
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
    out: list[JobPosting] = []
    for job in jobs:
        location = (job.get("location") or {}).get("name", "")
        offices = ", ".join(o.get("name", "") for o in job.get("offices", []) if o.get("name"))
        departments = ", ".join(
            d.get("name", "") for d in job.get("departments", []) if d.get("name")
        )
        out.append(
            JobPosting(
                company=board.company,
                title=job.get("title", ""),
                url=job.get("absolute_url", ""),
                source="greenhouse",
                location=location or offices,
                department=departments,
                description=strip_html(job.get("content", "")),
                posted_at=(job.get("updated_at") or job.get("first_published") or "")[:10],
                board_tier=board.tier,
                ind_sponsor=board.sponsor_state,
                gaming=board.gaming,
                raw=job,
            )
        )
    return out


def parse_lever(payload: Any, board: BoardRef) -> list[JobPosting]:
    postings = payload if isinstance(payload, list) else []
    out: list[JobPosting] = []
    for job in postings:
        categories = job.get("categories", {}) or {}
        description = job.get("descriptionPlain") or strip_html(job.get("description", ""))
        lists_text = "\n".join(
            f"{item.get('text', '')}\n{strip_html(item.get('content', ''))}"
            for item in job.get("lists", []) or []
        )
        created = job.get("createdAt")
        out.append(
            JobPosting(
                company=board.company,
                title=job.get("text", ""),
                url=job.get("hostedUrl") or job.get("applyUrl", ""),
                source="lever",
                location=categories.get("location", "") or "",
                department=categories.get("team", "") or categories.get("department", "") or "",
                description="\n\n".join(p for p in (description, lists_text) if p),
                posted_at=_epoch_ms_to_date(created),
                salary_text=str(job.get("salaryRange") or ""),
                board_tier=board.tier,
                ind_sponsor=board.sponsor_state,
                gaming=board.gaming,
                raw=job,
            )
        )
    return out


def parse_ashby(payload: Any, board: BoardRef) -> list[JobPosting]:
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
    out: list[JobPosting] = []
    for job in jobs:
        compensation = job.get("compensation") or {}
        salary_bits = []
        for tier in compensation.get("summaryComponents", []) or []:
            label = tier.get("summary") or ""
            if label:
                salary_bits.append(label)
        if compensation.get("compensationTierSummary"):
            salary_bits.append(compensation["compensationTierSummary"])
        out.append(
            JobPosting(
                company=board.company,
                title=job.get("title", ""),
                url=job.get("jobUrl") or job.get("applyUrl", ""),
                source="ashby",
                location=ashby_location(job),
                department=job.get("department", "") or job.get("team", "") or "",
                description=job.get("descriptionPlain") or strip_html(job.get("descriptionHtml", "")),
                posted_at=(job.get("publishedAt") or "")[:10],
                salary_text=" | ".join(dict.fromkeys(salary_bits)),
                remote=job.get("isRemote"),
                board_tier=board.tier,
                ind_sponsor=board.sponsor_state,
                gaming=board.gaming,
                raw=job,
            )
        )
    return out


def ashby_location(job: dict[str, Any]) -> str:
    """Primary plus secondary locations, joined.

    Ashby posts one record for a multi-city role: `location` is the primary
    office and `secondaryLocations` carries the rest. Keeping only the primary
    mislabels a role that is workable elsewhere - an EMEA role headed "Paris"
    that also lists Amsterdam reads as a French role and gets dismissed.
    """
    primary = (job.get("location") or "").strip()
    names: list[str] = [primary] if primary else []
    for entry in job.get("secondaryLocations") or []:
        if isinstance(entry, dict):
            name = (entry.get("location") or "").strip()
        else:
            name = str(entry).strip()
        if name and name not in names:
            names.append(name)
    # Deliberately does NOT fold in `isRemote`. "remote" is an allowed location
    # pattern, so appending it makes any country that is not explicitly
    # blocklisted - Australia, Brazil, UAE - read as workable. The remote flag
    # stays on JobPosting.remote for callers that want it.
    return " / ".join(names)


def parse_json_feed(payload: Any, board: BoardRef) -> list[JobPosting]:
    """Best-effort adapter for a company's own JSON careers feed.

    Accepts either a bare list or a dict with a ``jobs``/``items``/``results``
    key, and maps the common field spellings.
    """
    if isinstance(payload, dict):
        for key in ("jobs", "items", "results", "data", "postings"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return []
    out: list[JobPosting] = []
    for job in payload:
        if not isinstance(job, dict):
            continue
        title = _first(job, "title", "name", "text", "jobTitle")
        url = _first(job, "url", "absolute_url", "hostedUrl", "jobUrl", "link", "applyUrl")
        description = _first(job, "descriptionPlain", "description", "content", "body")
        out.append(
            JobPosting(
                company=board.company,
                title=title,
                url=url,
                source="json_feed",
                location=_first(job, "location", "city", "office"),
                department=_first(job, "department", "team", "category"),
                description=strip_html(description),
                posted_at=_first(job, "publishedAt", "posted_at", "createdAt")[:10],
                board_tier=board.tier,
                ind_sponsor=board.sponsor_state,
                gaming=board.gaming,
                raw=job,
            )
        )
    return out


PARSERS: dict[str, Callable[[Any, BoardRef], list[JobPosting]]] = {
    "greenhouse": parse_greenhouse,
    "lever": parse_lever,
    "ashby": parse_ashby,
    "json_feed": parse_json_feed,
}


def board_url(board: BoardRef) -> str:
    if board.ats == "greenhouse":
        return GREENHOUSE_URL.format(token=board.token)
    if board.ats == "lever":
        return LEVER_URL.format(token=board.token)
    if board.ats == "ashby":
        return ASHBY_URL.format(token=board.token)
    if board.ats == "json_feed":
        if not board.url:
            raise DiscoveryError(f"{board.company}: ats='json_feed' requires a `url`")
        return board.url
    raise DiscoveryError(
        f"{board.company}: unknown ats {board.ats!r}. "
        f"Use one of: {', '.join(sorted(PARSERS))}"
    )


def is_public_ats_api(url: str) -> bool:
    return urllib.parse.urlsplit(url).netloc.lower() in PUBLIC_ATS_HOSTS


def fetch_board(board: BoardRef, fetcher: Fetcher) -> list[JobPosting]:
    """Fetch and normalise one company's public job board."""
    url = board_url(board)
    payload = fetcher.get_json(url, check_robots=not is_public_ats_api(url))
    parser = PARSERS[board.ats]
    return parser(payload, board)


# ---------------------------------------------------------------------------
# Title filtering
# ---------------------------------------------------------------------------


def _word(term: str, needle: str) -> bool:
    """True when ``term`` appears in ``needle`` as a whole word or phrase."""
    term = (term or "").strip().lower()
    if not term:
        return False
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", needle) is not None


def title_matches(
    title: str,
    include: Iterable[str],
    exclude: Iterable[str],
    require_any: Iterable[str] = (),
) -> bool:
    """Cheap pre-filter so only plausible roles reach the (paid) scorer.

    ``require_any`` is a second gate, applied after ``include``. Seniority
    words alone are ambiguous: "Staff Software Engineer - Backend" and "Staff
    Applied AI Researcher" both match "staff", but only one is the job you
    want. Requiring at least one subject-matter word separates them without
    having to enumerate every platform speciality in ``exclude``.

    All three lists are matched on word boundaries, so "ai" does not fire on
    "maintain", "Ukraine" or "supply chain", and "ml" does not fire on "html".
    Substring matching here was a real defect rather than a nicety: "ai" alone
    admitted 213 postings whose only qualification was the letters inside
    another word.
    """
    needle = (title or "").lower()
    if any(_word(term, needle) for term in exclude):
        return False
    include = list(include)
    if include and not any(_word(term, needle) for term in include):
        return False
    if require_any and not any(_word(term, needle) for term in require_any):
        return False
    return True


def filter_postings(postings: Iterable[JobPosting], cfg: Config) -> list[JobPosting]:
    section = cfg.section("discover")
    include = section.get("title_include", [])
    exclude = section.get("title_exclude", [])
    require_any = section.get("title_require_any", [])
    return [p for p in postings if title_matches(p.title, include, exclude, require_any)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first(job: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = job.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict) and isinstance(value.get("name"), str):
            return value["name"]
    return ""


def _epoch_ms_to_date(value: Any) -> str:
    try:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError):
        return ""
