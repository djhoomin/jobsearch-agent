"""Scoring: hard constraints first, then the weighted rubric.

This module encodes ``search-strategy.md``.

Section 0 of the strategy defines four pass/fail filters that are applied
*before* any scoring happens: visa, non-compete, compensation floor, and
location/travel. They are implemented here as pure functions over a
:class:`~jobsearch.models.JobPosting`, with a three-valued verdict - a posting
that simply does not state its salary is UNKNOWN, not a failure. Only a
definite FAIL eliminates a role, and every verdict carries the evidence string
that produced it.

The five weighted dimensions (Buyer 20%, Role fit 25%, Company 25%, Domain 15%,
Talent density 15%) are qualitative, so Claude scores them 1-5 with the rubric
text in the prompt and returns structured output. The arithmetic that turns
those five numbers into a weighted score is done here, deterministically, and is
unit-tested.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Sequence

from .claude import ClaudeClient, stable_context_for
from .config import Config
from .models import (
    ConstraintReport,
    ConstraintResult,
    DimensionScore,
    JobPosting,
    ScoreReport,
    Verdict,
)

DIMENSIONS = ("buyer", "role_fit", "company", "domain", "talent")

DIMENSION_LABELS = {
    "buyer": "Buyer (business pain + budget)",
    "role_fit": "Role fit (altitude + hands-on room)",
    "company": "Company (stage, founders, stability)",
    "domain": "Domain (edge match + tailwind)",
    "talent": "Talent density / learning",
}


# ===========================================================================
# Hard constraints
# ===========================================================================


def _match(patterns: Iterable[str], haystack: str) -> str | None:
    """Return the first pattern found in ``haystack``, else None."""
    for pattern in patterns:
        p = pattern.strip().lower()
        if p and p in haystack:
            return p
    return None


def _match_word(patterns: Iterable[str], haystack: str) -> str | None:
    """Substring match, but on word boundaries.

    Location patterns are short and collide badly otherwise: a bare "nl" matches
    inside "Finland", which would classify a Helsinki role as Dutch.
    """
    for pattern in patterns:
        p = pattern.strip().lower()
        if p and re.search(rf"(?<!\w){re.escape(p)}(?!\w)", haystack):
            return p
    return None


def check_visa(posting: JobPosting, cfg: Config) -> ConstraintResult:
    """IND recognised sponsor, or explicit sponsorship language in the posting.

    A US-headquartered company with no NL entity is not automatically a fail:
    remote-EU employment through an EOR can work, it just does not carry
    kennismigrant status. The filter therefore fails only where the posting
    *rules sponsorship out* or the board is flagged as a non-sponsor for a role
    that requires being in the Netherlands.
    """
    constraints = cfg.section("constraints")
    if not constraints.get("require_ind_sponsor", True):
        return ConstraintResult("visa", Verdict.PASS, "Sponsor requirement disabled in config")

    text = posting.haystack
    refusal = _match(
        [
            "we do not sponsor",
            "we are unable to sponsor",
            "cannot provide visa sponsorship",
            "no visa sponsorship",
            "not able to sponsor",
            "must have existing work authorization",
            "must already have the right to work",
        ],
        text,
    )
    if refusal:
        return ConstraintResult(
            "visa",
            Verdict.FAIL,
            "Posting explicitly rules out visa sponsorship",
            evidence=refusal,
        )

    offer = _match(
        ["visa sponsorship", "relocation support", "we sponsor", "kennismigrant",
         "highly skilled migrant", "ind recognised sponsor", "ind recognized sponsor"],
        text,
    )
    if posting.ind_sponsor == "yes":
        return ConstraintResult(
            "visa", Verdict.PASS, "Employer is a known IND recognised sponsor"
        )
    if offer:
        return ConstraintResult(
            "visa", Verdict.PASS, "Posting offers sponsorship or relocation", evidence=offer
        )
    if posting.ind_sponsor == "no":
        return ConstraintResult(
            "visa",
            Verdict.UNKNOWN,
            "Employer has no NL entity on the IND register - viable only as "
            "remote-EU or via an EOR without kennismigrant status. Legal check needed.",
        )
    return ConstraintResult(
        "visa",
        Verdict.UNKNOWN,
        "Sponsor status unverified - check the IND public register before applying",
    )


def check_noncompete(posting: JobPosting, cfg: Config) -> ConstraintResult:
    """Competing employers are gated until the non-compete waiver is signed."""
    constraints = cfg.section("constraints")
    if constraints.get("non_compete_waiver_signed", False):
        return ConstraintResult(
            "non_compete", Verdict.PASS, "Non-compete waiver signed - no gate"
        )

    keywords = constraints.get("gaming_adjacent_keywords", [])
    hit = _match(keywords, posting.haystack)
    if posting.gaming or hit:
        return ConstraintResult(
            "non_compete",
            Verdict.FAIL,
            "Gaming-adjacent employer, gated until the non-compete waiver is signed. "
            "Sequence this conversation after signature, not before.",
            evidence=hit or "board flagged gaming=true",
        )
    return ConstraintResult("non_compete", Verdict.PASS, "Not gaming-competing")


#: A currency marker anywhere near an amount.
_CURRENCY = re.compile(r"€|eur\b|\$|usd\b|£|gbp\b", re.IGNORECASE)

#: A plausible salary figure: 90,000 / 90.000 / 90000 / 90k / 90K.
_AMOUNT = re.compile(r"\b(\d{1,3}(?:[.,]\d{3})+|\d{2,3}\s?[kK]\b|\d{5,7})\b")

#: What separates the two ends of a stated range.
_RANGE_SEP = re.compile(r"^[\s]*(?:-|–|—|to|and|tot|up to|until|/)[\s]*$", re.IGNORECASE)


def _amount_value(raw: str) -> int | None:
    raw = raw.strip().lower().replace(" ", "")
    try:
        if raw.endswith("k"):
            return int(float(raw[:-1].replace(",", ".")) * 1000)
        return int(re.sub(r"[.,]", "", raw))
    except ValueError:
        return None


def parse_salaries(text: str) -> list[int]:
    """Extract plausible annual base-salary figures from free text.

    A figure counts when a currency marker sits within a few characters of it,
    or when it is the other end of a range whose first end was anchored - so
    "EUR 90,000 - 110,000" and "110k - 140k EUR" both yield both numbers, while
    "a team of 25 people and 500 customers" yields nothing.

    Deliberately conservative: figures under 20,000 (hourly rates, equity unit
    counts) and over 1,000,000 (funding rounds) are discarded.
    """
    text = text or ""
    matches = list(_AMOUNT.finditer(text))
    anchored: list[bool] = []
    values: list[int | None] = []
    for match in matches:
        before = text[max(0, match.start() - 8) : match.start()]
        after = text[match.end() : match.end() + 8]
        anchored.append(bool(_CURRENCY.search(before) or _CURRENCY.search(after)))
        values.append(_amount_value(match.group(1)))

    # Propagate anchoring across range separators, in both directions.
    for _ in range(2):
        for i in range(len(matches) - 1):
            gap = text[matches[i].end() : matches[i + 1].start()]
            if _RANGE_SEP.match(gap):
                if anchored[i] or anchored[i + 1]:
                    anchored[i] = anchored[i + 1] = True

    found: list[int] = []
    for value, is_anchored in zip(values, anchored):
        if value is None or not is_anchored:
            continue
        if 20_000 <= value <= 1_000_000:
            found.append(value)
    return found


def check_comp(posting: JobPosting, cfg: Config) -> ConstraintResult:
    """Fail only when a stated range's TOP is below the floor."""
    constraints = cfg.section("constraints")
    floor = int(constraints.get("comp_floor_eur", 125_000))
    text = f"{posting.salary_text}\n{posting.description}"
    salaries = parse_salaries(text)
    if not salaries:
        if constraints.get("comp_fail_on_unknown", False):
            return ConstraintResult(
                "compensation", Verdict.FAIL, "No salary stated and comp_fail_on_unknown is set"
            )
        return ConstraintResult(
            "compensation",
            Verdict.UNKNOWN,
            f"No salary range stated - confirm base clears EUR {floor:,} early in the process",
        )
    top = max(salaries)
    if top < floor:
        return ConstraintResult(
            "compensation",
            Verdict.FAIL,
            f"Top of stated range ({top:,}) is below the EUR {floor:,} floor",
            evidence=", ".join(f"{s:,}" for s in sorted(set(salaries))),
        )
    return ConstraintResult(
        "compensation",
        Verdict.PASS,
        f"Stated range tops out at {top:,}, at or above the EUR {floor:,} floor",
        evidence=", ".join(f"{s:,}" for s in sorted(set(salaries))),
    )


def check_location(posting: JobPosting, cfg: Config) -> ConstraintResult:
    """Amsterdam / NL-hybrid / remote-EU only.

    The subtlety worth getting right: a bare "Remote" does NOT rescue a role
    whose location is otherwise the United States. "Remote - United States" is a
    US role. Only an explicitly European anchor (Netherlands, Amsterdam, EMEA,
    remote-EU, a named Dutch city, a CET requirement) overrides a blocked
    location, which is what makes "San Francisco or Remote (Europe)" workable
    and "Remote - United States" not.
    """
    constraints = cfg.section("constraints")
    allowed = constraints.get("allowed_location_patterns", [])
    blocked = constraints.get("blocked_location_patterns", [])
    anchors = constraints.get("eu_anchor_patterns", [])
    location = (posting.location or "").lower()
    if not location:
        return ConstraintResult(
            "location", Verdict.UNKNOWN, "Posting states no location"
        )
    allow_hit = _match_word(allowed, location)
    block_hit = _match_word(blocked, location)
    anchor_hit = _match_word(anchors, location)

    if block_hit and not anchor_hit:
        return ConstraintResult(
            "location",
            Verdict.FAIL,
            f"Location is outside Amsterdam / NL-hybrid / remote-EU: {posting.location}",
            evidence=block_hit,
        )
    if block_hit and anchor_hit:
        return ConstraintResult(
            "location",
            Verdict.PASS,
            f"Lists a non-EU base but also a European option: {posting.location}",
            evidence=anchor_hit,
        )
    if allow_hit:
        return ConstraintResult(
            "location", Verdict.PASS, f"Workable location: {posting.location}", evidence=allow_hit
        )
    return ConstraintResult(
        "location",
        Verdict.UNKNOWN,
        f"Could not classify location {posting.location!r} - confirm it allows Amsterdam basing",
    )


_TRAVEL_PERCENT = re.compile(r"(\d{1,3})\s?%\s*(?:of\s+(?:the\s+)?time\s+)?travel|travel[^.]{0,30}?(\d{1,3})\s?%")


def check_travel(posting: JobPosting, cfg: Config) -> ConstraintResult:
    """Travel cap of roughly monthly. Weekly-travel roles are out."""
    constraints = cfg.section("constraints")
    text = posting.haystack
    hit = _match(constraints.get("travel_fail_patterns", []), text)
    if hit:
        return ConstraintResult(
            "travel",
            Verdict.FAIL,
            "Travel load exceeds the ~monthly cap",
            evidence=hit,
        )
    cap = int(constraints.get("travel_max_percent", 30))
    for match in _TRAVEL_PERCENT.finditer(text):
        value = next((int(g) for g in match.groups() if g), None)
        if value is not None and value > cap:
            return ConstraintResult(
                "travel",
                Verdict.FAIL,
                f"Posting states {value}% travel, above the {cap}% cap",
                evidence=match.group(0).strip(),
            )
    if "travel" in text:
        return ConstraintResult(
            "travel", Verdict.UNKNOWN, "Travel mentioned but not quantified - probe on the first call"
        )
    return ConstraintResult("travel", Verdict.PASS, "No travel load flagged in the posting")


CONSTRAINT_CHECKS: Sequence[Callable[[JobPosting, Config], ConstraintResult]] = (
    check_visa,
    check_noncompete,
    check_comp,
    check_location,
    check_travel,
)


def check_constraints(posting: JobPosting, cfg: Config) -> ConstraintReport:
    """Run every hard constraint. Order is stable for reproducible output."""
    return ConstraintReport(results=[check(posting, cfg) for check in CONSTRAINT_CHECKS])


# ===========================================================================
# Weighted rubric
# ===========================================================================


def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean of the 1-5 dimension scores.

    Raises on a missing dimension rather than silently scoring the role low.
    """
    missing = [d for d in weights if d not in scores]
    if missing:
        raise ValueError(f"Missing dimension scores: {', '.join(sorted(missing))}")
    for name, value in scores.items():
        if name in weights and not 1.0 <= float(value) <= 5.0:
            raise ValueError(f"Dimension {name} score {value} outside 1-5")
    total = sum(float(scores[d]) * w for d, w in weights.items())
    return round(total, 2)


def build_dimensions(payload: dict[str, Any], weights: dict[str, float]) -> list[DimensionScore]:
    """Turn the model's structured response into DimensionScore objects."""
    dims: list[DimensionScore] = []
    for name in DIMENSIONS:
        block = payload.get(name) or {}
        dims.append(
            DimensionScore(
                name=name,
                score=_clamp_score(block.get("score", 0)),
                weight=weights[name],
                reasoning=block.get("reasoning", ""),
                evidence=list(block.get("evidence", []) or []),
            )
        )
    return dims


def _clamp_score(raw: Any) -> float:
    """Coerce a model-supplied dimension score into the documented 1-5 band.

    The schema constrains this, but a value outside the band would otherwise
    propagate silently into the weighted total.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return min(5.0, max(0.0, value))


def _dimension_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                # Structured outputs reject `minimum`/`maximum` on integers
                # (400 invalid_request_error). An enum expresses the same
                # 1-5 bound and is supported.
                "enum": [1, 2, 3, 4, 5],
            },
            "reasoning": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["score", "reasoning", "evidence"],
        "additionalProperties": False,
    }


SCORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **{name: _dimension_schema() for name in DIMENSIONS},
        "recommendation": {
            "type": "string",
            "enum": ["apply", "network_in", "park", "pass"],
        },
        "notes": {"type": "string"},
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [*DIMENSIONS, "recommendation", "notes", "open_questions"],
    "additionalProperties": False,
}


SCORING_INSTRUCTIONS = """\
You are scoring one job posting against a specific person's job-search rubric.
The rubric, his career dossier, and his current CV are supplied below and do not
change between jobs.

Score each of the five dimensions 1-5 using the rubric's own definitions:

- buyer (20%): does the pain AND the budget sit with a business owner, with a
  concrete case? His whole pricing thesis is that value must land in a business
  P&L; he has watched the alternative fail from the vendor side twice.
- role_fit (25%): altitude and shape. Primary archetype is Head/Director of AI
  owning a research-to-product function. Secondary: AI-platform product-line
  ownership, founding EMEA leadership, and SE/FDE leadership at major AI/data
  platforms. Senior-IC roles are an altitude step DOWN from running a 10-person
  research org and should score low. Does the role leave hands-on room?
- company (25%): stage, founders, runway. Series B-D hypergrowth with revenue
  and PMF scores highest. Pre-PMF labs and big-cog incumbents score low. Founder
  quality is a first-class criterion, not a tiebreaker.
- domain (15%): 1) sovereign/enterprise agentic AI for regulated industries,
  2) AI platform / agent infrastructure, 3) games/entertainment AI
  (opportunistic only). Score against that ranking plus any regulatory tailwind.
- talent (15%): will he learn there? Are the best people builders, and do
  builders lead? He has spent two years mostly being the teacher.

Rules:
- Ground every score in something the posting or the dossier actually says.
  Put the specific quote or fact in `evidence`. An empty evidence list means you
  are guessing; say so in `reasoning` and score conservatively.
- Do not restate the rubric back. Reason about THIS company and THIS posting.
- `notes` is for the one thing he should verify before spending time on it.
- `open_questions` are the questions to ask on the first call.
"""


def score_posting(
    posting: JobPosting,
    cfg: Config,
    claude: ClaudeClient,
    *,
    skip_constraints: bool = False,
) -> ScoreReport:
    """Score one posting: hard constraints, then (if it survives) the rubric.

    A posting eliminated by a hard constraint is not sent to the model at all -
    that is the point of running the filters first.
    """
    constraints = ConstraintReport() if skip_constraints else check_constraints(posting, cfg)
    report = ScoreReport(job_id=posting.job_id, constraints=constraints)

    if not constraints.passed:
        report.recommendation = "pass"
        report.notes = "; ".join(f"{r.name}: {r.reason}" for r in constraints.failures)
        return report

    weights = cfg.weights.as_dict()
    payload = claude.structured(
        instructions=SCORING_INSTRUCTIONS,
        stable_context=stable_context_for(cfg),
        user_content=_posting_prompt(posting, constraints),
        schema=SCORE_SCHEMA,
        stage="score",
        dry_run_value=_dry_run_score(),
    )

    report.dimensions = build_dimensions(payload, weights)
    report.weighted = weighted_score(
        {d.name: d.score for d in report.dimensions}, weights
    )
    report.recommendation = payload.get("recommendation", "")
    notes = [payload.get("notes", "")]
    questions = payload.get("open_questions") or []
    if questions:
        notes.append("Ask: " + "; ".join(questions))
    if constraints.unknowns:
        notes.append(
            "Unverified: " + "; ".join(f"{r.name} ({r.reason})" for r in constraints.unknowns)
        )
    report.notes = "\n".join(n for n in notes if n)
    return report


def _posting_prompt(posting: JobPosting, constraints: ConstraintReport) -> str:
    lines = [
        "<job_posting>",
        f"Company: {posting.company}",
        f"Title: {posting.title}",
        f"Location: {posting.location}",
        f"Department: {posting.department}",
        f"Source: {posting.source}",
        f"URL: {posting.url}",
        f"Stated compensation: {posting.salary_text or 'not stated'}",
        "",
        posting.description or "(no description text available)",
        "</job_posting>",
        "",
        "<hard_constraint_results>",
    ]
    for result in constraints.results:
        lines.append(f"- {result.name}: {result.verdict.value} - {result.reason}")
    lines.append("</hard_constraint_results>")
    lines.append("")
    lines.append("Score this posting on the five rubric dimensions.")
    return "\n".join(lines)


def _dry_run_score() -> dict[str, Any]:
    """Deterministic placeholder so --dry-run exercises the whole code path."""
    block = {"score": 3, "reasoning": "[dry-run] no API call made", "evidence": []}
    return {
        **{name: dict(block) for name in DIMENSIONS},
        "recommendation": "park",
        "notes": "[dry-run] no API call made",
        "open_questions": [],
    }
