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
    find_moved_board,
    moved_board_hint,
    strip_html,
    title_matches,
)
from .single import clean_posting_url, fetch_single_posting, posting_from_page
from .websearch import web_search_discover

log = logging.getLogger(__name__)


@dataclass
class DiscoveryReport:
    """Result of a discovery sweep."""

    postings: list[JobPosting] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    boards_checked: int = 0
    raw_count: int = 0
    #: Every job_id the boards returned this run, before title filtering. This
    #: is what "still listed" means; `postings` is only what also matched.
    seen_job_ids: set[str] = field(default_factory=set)
    #: Postings that matched the title filter but sit outside the workable
    #: locations, so they never reach the tracker.
    location_dropped: list[JobPosting] = field(default_factory=list)

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
            hint = moved_board_hint(board, fetcher) if "HTTP 404" in str(exc) else ""
            log.warning("%s: %s%s", board.company, exc, hint)
            report.errors.append(f"{board.company}: {exc}{hint}")
            continue
        if not found:
            # An empty 200 after a migration looks like a company with no jobs.
            hint = moved_board_hint(board, fetcher)
            if hint:
                report.errors.append(f"{board.company}: board is empty{hint}")
        report.raw_count += len(found)
        report.postings.extend(found)

    # Record what the boards actually returned BEFORE filtering. last_seen_at
    # must mean "the board still lists this", not "it still matches my title
    # filters": those are different facts, and conflating them reports a live
    # posting as delisted the moment the filters are tightened.
    report.seen_job_ids = {p.job_id for p in report.postings}
    if apply_title_filter:
        report.postings = filter_postings(report.postings, cfg)
    if cfg.section("discover").get("location_filter", True):
        report.postings, report.location_dropped = filter_locations(report.postings, cfg)
    return report.dedupe()


def filter_locations(
    postings: list[JobPosting], cfg: Config
) -> tuple[list[JobPosting], list[JobPosting]]:
    """Split postings into (workable, out of region) before they are stored.

    Scoring runs the same location check later, but only on roles someone
    chooses to score, so without this every US or Krakow posting a board
    returns sits in the tracker as "Not started". A posting is dropped when
    the check fails outright, or when it states a location that matches none
    of the allowed patterns (a bare "Krakow" or "Singapore"). A posting that
    states no location is kept: missing information is not a mismatch.
    """
    from ..scoring import check_location  # scoring imports the Claude client; keep discover light
    from ..models import Verdict

    kept: list[JobPosting] = []
    dropped: list[JobPosting] = []
    for posting in postings:
        verdict = check_location(posting, cfg).verdict
        if verdict == Verdict.PASS or (verdict == Verdict.UNKNOWN and not posting.location.strip()):
            kept.append(posting)
        else:
            dropped.append(posting)
    return kept, dropped


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
    "clean_posting_url",
    "fetch_single_posting",
    "posting_from_page",
    "filter_locations",
    "filter_postings",
    "strip_html",
    "title_matches",
    "web_search_discover",
]
