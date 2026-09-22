"""A System One reader for the hard constraints.

The constraint checks in :mod:`jobsearch.scoring` are phrase lists. They are
right when they fire and silent when the posting says the same thing in words
the list does not contain: "authorized to work in the United States without
the need for new visa sponsorship" is a refusal, and no list catches every
phrasing of it. This module asks TypeSafe's Jev, a decision model that returns
calibrated probabilities and no text, three yes/no questions about the posting
text and feeds the answers back into the constraint report.

The rules are deliberately narrow:

- A definite FAIL from the code is never touched.
- An UNKNOWN becomes FAIL above ``fail_above``, PASS below ``pass_below``, and
  otherwise stays UNKNOWN with the probability noted in its reason. Visa is
  the exception: its UNKNOWN means the employer is not on the IND register,
  which posting text cannot clear, so Jev can only fail or flag it.
- A PASS the code reached by lookup (an IND-listed employer, say) is kept, but
  gets an advisory when the posting text contradicts it above ``flag_above``.

Only posting text is sent: company, title, location, department, stated pay
and the description. Never the dossier, the CV, or the candidate's details.

The reader is optional. With no ``[systemone]`` section, no key in the
environment, or the SDK not installed, scoring behaves exactly as before.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

from .config import Config
from .models import ConstraintReport, ConstraintResult, JobPosting, Verdict

BOILERPLATE = (
    "generic equal-opportunity boilerplate such as 'applicants must be authorized "
    "to work in the country in which they apply'"
)


@dataclass(frozen=True)
class Question:
    """One yes/no question about the posting and the constraint it informs."""

    constraint: str
    #: What a "yes" means, in words that read well after "the posting suggests".
    meaning: str
    instructions: str
    #: Descriptions of what a yes and a no look like. Jev reads them; they are
    #: what keeps boilerplate from counting as a refusal.
    criteria: dict[str, str] | None = None
    #: Whether a confident "no" may turn UNKNOWN into PASS. Posting text can
    #: clear a travel or location doubt, because those doubts came from the
    #: text. It cannot clear a visa doubt: that one means the employer is not
    #: on the IND register, and no wording in a posting changes that.
    can_clear: bool = True


NOULS: dict[str, Question] = {
    "rules_out_sponsorship": Question(
        constraint="visa",
        meaning="a candidate who needs Dutch sponsorship could not get it for this role",
        instructions=(
            "Would a candidate who needs a Dutch (Netherlands) work visa be unable "
            "to get sponsorship for this role?"
        ),
        criteria={
            "true": (
                "The employer says it will not sponsor, requires existing work "
                "authorisation for a named country, or sponsors visas only for a "
                "country other than the Netherlands (e.g. 'we can sponsor visas to "
                "Germany; for any other country you need an existing right to work')."
            ),
            "false": (
                "The employer offers sponsorship without restricting it to another "
                f"country, or says nothing beyond {BOILERPLATE}."
            ),
        },
        can_clear=False,
    ),
    "weekly_travel": Question(
        constraint="travel",
        meaning="the posting requires weekly travel or more than 25% travel",
        instructions="Does the posting require weekly travel, or travel above 25% of the time?",
    ),
    "outside_europe": Question(
        constraint="location",
        meaning="the role cannot be done from the Netherlands or remote within Europe",
        instructions=(
            "Is this role impossible to do from the Netherlands or remote within Europe "
            "(for example US-only, on-site outside Europe, or remote restricted to "
            "non-European countries)?"
        ),
    ),
}

Reader = Callable[[JobPosting], "dict[str, float] | None"]


@dataclass(frozen=True)
class SystemOneSettings:
    enabled: bool = False
    api_key_env: str = "TYPESAFE_API_KEY"
    model: str = "jev-latest"
    fail_above: float = 0.8
    pass_below: float = 0.2
    flag_above: float = 0.7
    max_description_chars: int = 20000
    timeout: float = 30.0

    @classmethod
    def from_config(cls, cfg: Config) -> "SystemOneSettings":
        s = cfg.section("systemone")
        return cls(
            enabled=bool(s.get("enabled", False)),
            api_key_env=str(s.get("api_key_env", "TYPESAFE_API_KEY")),
            model=str(s.get("model", "jev-latest")),
            fail_above=float(s.get("fail_above", 0.8)),
            pass_below=float(s.get("pass_below", 0.2)),
            flag_above=float(s.get("flag_above", 0.7)),
            max_description_chars=int(s.get("max_description_chars", 20000)),
            timeout=float(s.get("timeout", 30.0)),
        )

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None


def posting_state(posting: JobPosting, settings: SystemOneSettings) -> dict[str, str]:
    """The posting as Jev sees it. Posting fields only."""
    return {
        "company": posting.company,
        "title": posting.title,
        "location": posting.location or "",
        "department": posting.department or "",
        "stated_compensation": posting.salary_text or "not stated",
        "description": (posting.description or "")[: settings.max_description_chars],
    }


def reader_for(settings: SystemOneSettings) -> Reader | None:
    """A callable that asks Jev the three questions, or None when it cannot run."""
    if not settings.enabled or not settings.api_key:
        return None
    try:
        from typesafe_sdk import Noul, TypeSafeClient
    except ImportError:
        _warn("[systemone] enabled but typesafe-sdk is not installed; pip install -e '.[systemone]'")
        return None

    questions = {
        name: Noul(instructions=q.instructions, criteria=q.criteria) for name, q in NOULS.items()
    }
    client = TypeSafeClient(api_key=settings.api_key, model=settings.model, timeout=settings.timeout)

    def read(posting: JobPosting) -> dict[str, float] | None:
        response = client.system_one(state=posting_state(posting, settings), questions=questions)
        return {name: float(response.answers[name].noul) for name in NOULS}

    return read


def apply_readings(
    report: ConstraintReport, probs: dict[str, float], settings: SystemOneSettings
) -> list[ConstraintResult]:
    """Fold Jev's probabilities into the constraint results, in place.

    Returns the results that changed or gained an advisory.
    """
    by_name = {r.name: r for r in report.results}
    touched: list[ConstraintResult] = []
    for noul, q in NOULS.items():
        p = probs.get(noul)
        result = by_name.get(q.constraint)
        meaning = q.meaning
        if p is None or result is None:
            continue
        tag = f"jev {noul}={p:.2f}"
        if result.verdict is Verdict.UNKNOWN:
            if p >= settings.fail_above:
                result.verdict = Verdict.FAIL
                result.reason = f"Jev reads the posting: {meaning} (p={p:.2f}). Code said: {result.reason}"
                result.evidence = tag
                touched.append(result)
            elif p <= settings.pass_below and q.can_clear:
                result.verdict = Verdict.PASS
                result.reason = f"Jev reads no sign that {meaning} (p={p:.2f}). Code said: {result.reason}"
                result.evidence = tag
                touched.append(result)
            else:
                result.reason = f"{result.reason} (Jev inconclusive, p={p:.2f})"
                result.evidence = tag
        elif result.verdict is Verdict.PASS and p >= settings.flag_above:
            result.advisory = f"posting text suggests {meaning} (p={p:.2f})"
            result.evidence = tag
            touched.append(result)
    return touched


def refine_constraints(
    posting: JobPosting,
    cfg: Config,
    report: ConstraintReport,
    *,
    reader: Reader | None = None,
) -> bool:
    """Ask Jev about the posting and fold the answers into ``report``.

    ``reader`` overrides the SDK-backed one (tests). Returns True when Jev was
    consulted. Any failure to reach it is reported and leaves the code's
    verdicts untouched: this is a refinement, never a dependency.
    """
    settings = SystemOneSettings.from_config(cfg)
    if not settings.enabled:
        return False
    if reader is None:
        reader = reader_for(settings)
        if reader is None:
            return False
    try:
        probs = reader(posting)
    except Exception as exc:  # noqa: BLE001 - never let the reader block scoring
        _warn(f"[systemone] {type(exc).__name__}: {exc}; constraints left as the code found them")
        return False
    if not probs:
        return False
    apply_readings(report, probs, settings)
    return True


def _warn(message: str) -> None:
    print(message, file=sys.stderr)
