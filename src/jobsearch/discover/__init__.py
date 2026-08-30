"""Discovery stage: find candidate postings from permitted public sources."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import BoardRef, Config
from ..models import JobPosting
from .sources import (
    DiscoveryError,
    Fetcher,
    board_url,
    fetch_board,
    filter_postings,
    strip_html,
    title_matches,
)
from .single import fetch_single_posting
from .websearch import web_search_discover

log = logging.getLogger(__name__)


@dataclass
class DiscoveryReport:
    """Result of a discovery sweep."""

    postings: list[JobPosting] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    boards_checked: int = 0
    raw_count: int = 0

    def dedupe(self) -> "DiscoveryReport":
        seen: set[str] = set()
        unique: list[JobPosting] = []
        for posting in self.postings:
            key = posting.url or posting.job_id
            if key in seen:
                continue
            seen.add(key)
            unique.append(posting)
        self.postings = unique
        return self


def discover(
    cfg: Config,
    *,
    companies: list[str] | None = None,
    tiers: list[int] | None = None,
    fetcher: Fetcher | None = None,
    apply_title_filter: bool = True,
) -> DiscoveryReport:
    """Sweep every configured public board and return matching postings.

    ``companies`` and ``tiers`` narrow the sweep. Failures on individual boards
    are collected in ``report.errors`` rather than aborting the run - a company
    that renamed its board token should not stop the other twenty.
    """
    fetcher = fetcher or Fetcher.from_config(cfg)
    report = DiscoveryReport()

    wanted = _select_boards(cfg, companies, tiers)
    for board in wanted:
        report.boards_checked += 1
        try:
            found = fetch_board(board, fetcher)
        except DiscoveryError as exc:
            log.warning("%s: %s", board.company, exc)
            report.errors.append(f"{board.company}: {exc}")
            continue
        report.raw_count += len(found)
        report.postings.extend(found)

    if apply_title_filter:
        report.postings = filter_postings(report.postings, cfg)
    return report.dedupe()


def _select_boards(
    cfg: Config, companies: list[str] | None, tiers: list[int] | None
) -> list[BoardRef]:
    boards = cfg.boards
    if companies:
        wanted = {c.strip().lower() for c in companies}
        boards = [b for b in boards if b.company.strip().lower() in wanted]
        missing = wanted - {b.company.strip().lower() for b in boards}
        for name in sorted(missing):
            log.warning("No board configured for %r - add it to [[discover.boards]]", name)
    if tiers:
        boards = [b for b in boards if b.tier in set(tiers)]
    return boards


__all__ = [
    "DiscoveryError",
    "DiscoveryReport",
    "Fetcher",
    "board_url",
    "discover",
    "fetch_board",
    "fetch_single_posting",
    "filter_postings",
    "strip_html",
    "title_matches",
    "web_search_discover",
]
