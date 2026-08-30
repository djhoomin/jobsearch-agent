"""Core domain objects shared across the pipeline stages."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Status enum. The string values mirror the Status column already in use in the
# user's job-search-tracker.xlsx so the export round-trips.
# ---------------------------------------------------------------------------


class Status(str, Enum):
    NOT_STARTED = "Not started"
    PARKED = "Parked"
    OUTREACH_SENT = "Outreach sent"
    APPLIED = "Applied"
    IN_CONVERSATION = "In conversation"
    INTERVIEWING = "Interviewing"
    OFFER = "Offer"
    REJECTED = "Rejected"
    WITHDRAWN = "Withdrawn"

    @classmethod
    def parse(cls, value: str | "Status") -> "Status":
        if isinstance(value, cls):
            return value
        needle = str(value).strip().lower()
        for member in cls:
            if member.value.lower() == needle or member.name.lower() == needle:
                return member
        raise ValueError(
            f"Unknown status {value!r}. Valid: {', '.join(m.value for m in cls)}"
        )


#: Allowed status transitions. Terminal states accept no outgoing edge except
#: a deliberate re-open back to PARKED.
TRANSITIONS: dict[Status, set[Status]] = {
    Status.NOT_STARTED: {Status.PARKED, Status.OUTREACH_SENT, Status.APPLIED, Status.WITHDRAWN},
    Status.PARKED: {Status.NOT_STARTED, Status.OUTREACH_SENT, Status.APPLIED, Status.WITHDRAWN},
    Status.OUTREACH_SENT: {
        Status.APPLIED,
        Status.IN_CONVERSATION,
        Status.PARKED,
        Status.REJECTED,
        Status.WITHDRAWN,
    },
    Status.APPLIED: {
        Status.IN_CONVERSATION,
        Status.INTERVIEWING,
        Status.REJECTED,
        Status.WITHDRAWN,
        Status.PARKED,
    },
    Status.IN_CONVERSATION: {
        Status.INTERVIEWING,
        Status.APPLIED,
        Status.REJECTED,
        Status.WITHDRAWN,
        Status.PARKED,
    },
    Status.INTERVIEWING: {Status.OFFER, Status.REJECTED, Status.WITHDRAWN, Status.PARKED},
    Status.OFFER: {Status.WITHDRAWN, Status.REJECTED},
    Status.REJECTED: {Status.PARKED},
    # Dismissing a role (TUI `d`) parks it here, so this must be reversible:
    # a role dismissed on a skim is sometimes one you later apply to.
    Status.WITHDRAWN: {Status.PARKED, Status.NOT_STARTED, Status.APPLIED},
}


def can_transition(current: Status, target: Status) -> bool:
    """True if ``current -> target`` is a legal status move."""
    if current is target:
        return True
    return target in TRANSITIONS.get(current, set())


# ---------------------------------------------------------------------------
# Job postings
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_job_id(company: str, title: str, url: str = "") -> str:
    """A stable, human-readable job id.

    ``weaviate-director-of-product-3f2a`` - slug plus a short hash of the URL so
    two similarly titled roles at one company never collide.
    """
    slug_parts = [_slug(company), _slug(title)]
    base = "-".join(p for p in slug_parts if p) or "job"
    digest = hashlib.sha256((url or base).encode("utf-8")).hexdigest()[:4]
    return f"{base[:60]}-{digest}"


def _slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").lower())
    return text.strip("-")


@dataclass
class JobPosting:
    """A single job posting, normalised across ATS providers."""

    company: str
    title: str
    url: str
    source: str = "manual"  # greenhouse | lever | ashby | json_feed | web_search | manual
    location: str = ""
    description: str = ""
    posted_at: str = ""
    department: str = ""
    salary_text: str = ""
    remote: bool | None = None
    board_tier: int | None = None
    ind_sponsor: str = "unknown"  # yes | no | unknown
    gaming: bool = False
    job_id: str = ""
    discovered_at: str = field(default_factory=_now)
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.job_id:
            self.job_id = make_job_id(self.company, self.title, self.url)

    @property
    def haystack(self) -> str:
        """All free text of the posting, lowercased, for pattern matching."""
        return " \n".join(
            [self.title, self.location, self.department, self.description, self.salary_text]
        ).lower()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        return d

    def summary_line(self) -> str:
        loc = f" - {self.location}" if self.location else ""
        return f"{self.job_id}  {self.company}: {self.title}{loc}"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass
class ConstraintResult:
    """Outcome of one hard-constraint filter."""

    name: str
    verdict: Verdict
    reason: str
    evidence: str = ""

    @property
    def eliminates(self) -> bool:
        return self.verdict is Verdict.FAIL


@dataclass
class ConstraintReport:
    """All hard constraints for one posting."""

    results: list[ConstraintResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True unless some constraint definitively failed."""
        return not any(r.eliminates for r in self.results)

    @property
    def failures(self) -> list[ConstraintResult]:
        return [r for r in self.results if r.eliminates]

    @property
    def unknowns(self) -> list[ConstraintResult]:
        return [r for r in self.results if r.verdict is Verdict.UNKNOWN]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "results": [asdict(r) | {"verdict": r.verdict.value} for r in self.results],
        }


@dataclass
class DimensionScore:
    """One rubric dimension: a 1-5 score with its reasoning."""

    name: str
    score: float
    weight: float
    reasoning: str = ""
    evidence: list[str] = field(default_factory=list)

    @property
    def contribution(self) -> float:
        return self.score * self.weight


@dataclass
class ScoreReport:
    """Hard constraints plus the weighted rubric score for one posting."""

    job_id: str
    constraints: ConstraintReport
    dimensions: list[DimensionScore] = field(default_factory=list)
    weighted: float | None = None
    recommendation: str = ""
    notes: str = ""
    scored_at: str = field(default_factory=_now)

    @property
    def eliminated(self) -> bool:
        return not self.constraints.passed

    def dimension(self, name: str) -> DimensionScore | None:
        return next((d for d in self.dimensions if d.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "constraints": self.constraints.to_dict(),
            "dimensions": [asdict(d) for d in self.dimensions],
            "weighted": self.weighted,
            "recommendation": self.recommendation,
            "notes": self.notes,
            "scored_at": self.scored_at,
        }


# ---------------------------------------------------------------------------
# Tailoring / grounding
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """A factual claim made by the tailored CV, checked against the fact base."""

    text: str
    grounded: bool
    evidence: str = ""
    source: str = ""  # "dossier" | "base_cv" | ""
    severity: str = "info"  # info | warn | block

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TailorResult:
    job_id: str
    html_path: str
    pdf_path: str | None = None
    headline: str = ""
    claims: list[Claim] = field(default_factory=list)
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def ungrounded(self) -> list[Claim]:
        return [c for c in self.claims if not c.grounded]


# ---------------------------------------------------------------------------
# Outreach
# ---------------------------------------------------------------------------


@dataclass
class Contact:
    """A likely contact, inferred - never scraped."""

    title: str
    name: str = ""
    rationale: str = ""
    linkedin_search_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OutreachDraft:
    job_id: str
    contacts: list[Contact] = field(default_factory=list)
    linkedin_connection_note: str = ""
    linkedin_message: str = ""
    email_subject: str = ""
    email_body: str = ""
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "contacts": [c.to_dict() for c in self.contacts],
            "linkedin_connection_note": self.linkedin_connection_note,
            "linkedin_message": self.linkedin_message,
            "email_subject": self.email_subject,
            "email_body": self.email_body,
            "created_at": self.created_at,
        }


__all__ = [
    "Claim",
    "Contact",
    "ConstraintReport",
    "ConstraintResult",
    "DimensionScore",
    "JobPosting",
    "OutreachDraft",
    "ScoreReport",
    "Status",
    "TailorResult",
    "Verdict",
    "can_transition",
    "make_job_id",
]

def today() -> date:
    return datetime.now(timezone.utc).date()
