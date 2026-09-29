"""Re-check roles that no board sweep covers, at their own posting URL.

Discovery refreshes ``last_seen_at`` for everything a configured board returns.
A role added by hand (``jobsearch add``) from a company with no configured board
is never swept, so it could vanish and still look live. This visits each such
role's URL once per discover run and records two facts:

- ``last_checked_at``: when the URL was checked, whatever the answer.
- ``last_seen_at``: refreshed only when the posting is still there.

A role is gone when it was checked after it was last seen. Only open roles are
checked (not rejected or withdrawn), so a run costs a handful of requests.

What counts as still listed:

- Greenhouse, Lever and Ashby URLs: the posting is returned by the public API.
- Other career pages: the page loads, shows no closed-posting wording, and
  carries a schema.org ``JobPosting``. A page with no structured data and no
  closed wording is left unknown rather than called gone, because some sites
  never had the structured block. A 404 or 410 is gone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .discover.single import _host, _ld_job_posting, fetch_single_posting
from .discover.sources import DiscoveryError, Fetcher

RECHECK_STATUSES = frozenset(
    {"Not started", "Parked", "Applied", "Outreach sent", "In conversation", "Interviewing", "Offer"}
)

#: Wording a career site shows on a closed posting it has not taken down.
CLOSED_WORDING = re.compile(
    r"no longer (?:accepting|available|open)|position has been filled|this (?:job|vacancy|role) (?:is|has) (?:closed|expired)"
    r"|vacature is (?:gesloten|vervuld)|niet meer (?:beschikbaar|vacant)",
    re.IGNORECASE,
)
_ATS_HOSTS = ("greenhouse.io", "lever.co", "ashbyhq.com")


@dataclass
class RecheckReport:
    listed: list[str] = field(default_factory=list)
    gone: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)


def page_listed(body: str) -> bool | None:
    """True, False, or None (cannot tell) for a fetched career page. Pure.

    ``validThrough`` is deliberately ignored: bol's live Head of Data Science
    posting carried a validThrough a month in the past (2026-09-29), and sites
    leave that field stale often enough that it cannot mark a role gone.
    """
    if CLOSED_WORDING.search(body):
        return False
    return True if _ld_job_posting(body) else None


def is_listed(url: str, fetcher: Fetcher, cfg: Config | None = None) -> bool | None:
    """Whether the posting at ``url`` is still up. None when it cannot be told."""
    host = _host(url)
    try:
        if any(h in host for h in _ATS_HOSTS):
            fetch_single_posting(url, fetcher, cfg)
            return True
        return page_listed(fetcher.get(url))
    except DiscoveryError as exc:
        message = str(exc)
        if "HTTP 404" in message or "HTTP 410" in message:
            return False
        if "not found" in message.lower() or "no posting" in message.lower():
            return False
        return None


def recheck_unswept(tracker: Any, cfg: Config, fetcher: Fetcher | None = None) -> RecheckReport:
    """Check every open role whose company has no configured board."""
    fetcher = fetcher or Fetcher.from_config(cfg)
    swept = {b.company.strip().lower() for b in cfg.boards}
    report = RecheckReport()
    for row in tracker.list_jobs(limit=100000):
        if str(row["status"]) not in RECHECK_STATUSES:
            continue
        if str(row["company"] or "").strip().lower() in swept or not row["url"]:
            continue
        job_id = str(row["job_id"])
        listed = is_listed(str(row["url"]), fetcher, cfg)
        if listed is None:
            report.unknown.append(job_id)
            continue
        if listed:
            tracker.mark_seen([job_id])
            report.listed.append(job_id)
        else:
            report.gone.append(job_id)
        tracker.mark_checked(job_id)
    return report
