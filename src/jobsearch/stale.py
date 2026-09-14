"""Find tracked roles a board has stopped returning.

The tracker only ever adds and refreshes. Nothing detected a delisting, so a
role pulled from a board simply stopped being updated and sat at "Not started"
looking exactly like a live posting nobody had got to yet. Across 889 rows that
is not a rounding error: it is a backlog that quietly fills with dead roles and
makes the pipeline look healthier than it is.

The fix is ``job.last_seen_at``, written on every upsert. A row is delisted when
a sweep covered its company and did not return it. The distinction that matters,
and the one the old schema could not express, is between **gone** and **not
checked**: a row with no recent ``last_seen_at`` on a board that has not been
swept says nothing at all.

This module only reports. Changing status automatically is a separate decision:
a bug there silently withdraws live roles and leaves no trace that it happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

#: Statuses safe to consider delisted. Anything further along stays put: a role
#: vanishing after you applied usually means it was filled, which is
#: information about your application, not clutter to be tidied away.
REPORTABLE_STATUSES: frozenset[str] = frozenset({"Not started", "Parked"})


@dataclass
class StaleRow:
    job_id: str
    company: str
    title: str
    status: str
    last_seen_at: str | None
    days_unseen: int | None

    @property
    def never_seen(self) -> bool:
        """True for rows predating last_seen_at, which prove nothing."""
        return self.last_seen_at is None


@dataclass
class StaleReport:
    """What a sweep did and did not find."""

    swept_companies: list[str]
    delisted: list[StaleRow]
    unknown: list[StaleRow]

    def render(self) -> str:
        lines: list[str] = []
        if not self.swept_companies:
            return (
                "No company has been swept since last_seen_at was introduced.\n"
                "Run `jobsearch discover` first; until a board has been swept once, "
                "nothing can be said about whether its postings are still listed."
            )
        lines.append(
            f"Swept and still tracked: {len(self.swept_companies)} compan"
            f"{'y' if len(self.swept_companies) == 1 else 'ies'}"
        )
        if self.delisted:
            lines.append("")
            lines.append(f"DELISTED ({len(self.delisted)}): swept for, not returned")
            for row in sorted(self.delisted, key=lambda r: (r.company, r.title)):
                age = f"{row.days_unseen}d" if row.days_unseen is not None else "?"
                lines.append(
                    f"  [{age:>4} unseen] {row.status:<12} {row.company} - {row.title[:52]}"
                )
        else:
            lines.append("")
            lines.append("No delisted roles found on the boards swept so far.")
        if self.unknown:
            lines.append("")
            lines.append(
                f"UNKNOWN ({len(self.unknown)}): no sweep has covered these since "
                "last_seen_at was added, so their state is unproven"
            )
            counts: dict[str, int] = {}
            for row in self.unknown:
                counts[row.company] = counts.get(row.company, 0) + 1
            for company, count in sorted(counts.items(), key=lambda kv: -kv[1])[:15]:
                lines.append(f"  {count:>4}  {company}")
            if len(counts) > 15:
                lines.append(f"  ... and {len(counts) - 15} more companies")
        return "\n".join(lines)


def _parse(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def find_stale(
    rows: Iterable[Any],
    *,
    now: datetime | None = None,
    grace_hours: int = 36,
    statuses: frozenset[str] = REPORTABLE_STATUSES,
) -> StaleReport:
    """Split tracked roles into delisted and unproven.

    A company counts as swept when at least one of its rows has a
    ``last_seen_at``. Within such a company, a row whose ``last_seen_at`` is
    older than the company's most recent sighting by more than ``grace_hours``
    was not returned by that sweep, and is therefore delisted.

    Comparing each row against its own company's latest sighting, rather than
    against a global clock, is what keeps a board nobody has swept lately from
    being reported as a mass delisting.
    """
    now = now or datetime.now(timezone.utc)
    rows = list(rows)

    latest_by_company: dict[str, datetime] = {}
    for row in rows:
        seen = _parse(row["last_seen_at"])
        if seen is None:
            continue
        company = str(row["company"] or "")
        if company not in latest_by_company or seen > latest_by_company[company]:
            latest_by_company[company] = seen

    delisted: list[StaleRow] = []
    unknown: list[StaleRow] = []
    for row in rows:
        if str(row["status"]) not in statuses:
            continue
        company = str(row["company"] or "")
        seen = _parse(row["last_seen_at"])
        sweep = latest_by_company.get(company)
        entry = StaleRow(
            job_id=str(row["job_id"]),
            company=company,
            title=str(row["title"] or ""),
            status=str(row["status"]),
            last_seen_at=row["last_seen_at"],
            days_unseen=(now - seen).days if seen else None,
        )
        if sweep is None:
            unknown.append(entry)
        elif seen is None or (sweep - seen) > timedelta(hours=grace_hours):
            delisted.append(entry)
    return StaleReport(
        swept_companies=sorted(latest_by_company),
        delisted=delisted,
        unknown=unknown,
    )
