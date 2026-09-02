"""CV tailoring: the stage that must never invent experience.

The pipeline is deliberately three steps, not one:

1. **Generate.** A streaming call produces a complete tailored HTML CV, using
   the base CV as the structural template and the career dossier as the fact
   base. Both are in the cached prompt prefix.
2. **Harden.** Deterministic post-processing re-applies the ATS CSS rules the
   template depends on (no letter-spacing/small-caps on headings, inline bullet
   markers, inline education dates, ``.nb { white-space: nowrap }``) in case the
   model dropped one, and normalises the file so the renderer sees what the
   verifier expects.
3. **Verify grounding.** A second, structured call takes the *generated* CV back
   and checks every factual claim against the dossier and base CV, returning
   each claim with the evidence that grounds it. Anything it cannot ground is
   surfaced to the user - loudly - rather than silently shipped.

Step 3 exists because step 1 is the single highest-risk operation in the tool:
a fabricated number on a CV is a fireable offence in an interview, and a model
asked to "tailor" will happily reach for a plausible one.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Sequence

from .claude import ClaudeClient, stable_context_for
from .config import Config
from .models import Claim, Critique, JobPosting, TailorResult, make_job_id

log = logging.getLogger(__name__)

#: CSS the ATS verifier depends on. Injected if the generated CV lacks it.
ATS_CSS_GUARDS = {
    "nb_nowrap": (
        r"\.nb\s*\{[^}]*white-space\s*:\s*nowrap",
        "  /* ATS: hyphenated keywords must not break across a line, or a literal\n"
        '     match like "human-in-the-loop" is lost at the wrap point. */\n'
        "  .nb { white-space: nowrap; }\n",
    ),
}

#: Patterns that must NOT appear: they are the known ATS-breaking styles.
FORBIDDEN_CSS = (
    (r"h2\s*\{[^}]*font-variant\s*:\s*small-caps", "small-caps on h2 corrupts heading extraction"),
    (
        r"h2\s*\{[^}]*letter-spacing\s*:\s*(?!0)[0-9.]+",
        "letter-spacing on h2 injects stray spaces into headings",
    ),
    (
        r"li::before\s*\{[^}]*position\s*:\s*absolute",
        "absolutely positioned bullet markers orphan onto their own line",
    ),
)


TAILOR_INSTRUCTIONS = """\
You tailor one specific person's CV to one specific job posting. His career
dossier (the fact base) and his ATS-hardened base CV (the template) are supplied
below and never change.

YOUR OUTPUT IS A COMPLETE HTML DOCUMENT. Nothing else - no markdown fences, no
commentary before or after. Start at the leading HTML comment and end at the
final closing tag.

## The one rule that overrides every other rule

NEVER INVENT EXPERIENCE. Every factual claim in the output - every number, date,
title, employer, technology, team size, outcome - must be traceable to the
career dossier or the base CV. You may:
  - reword, reorder, compress, and re-emphasise existing facts
  - promote a fact from the dossier that the base CV omitted
  - adopt the posting's vocabulary for something he genuinely did
You may NOT:
  - add a technology, tool, metric, employer, or responsibility that is not in
    the source documents
  - inflate a number, widen a scope, or convert "helped build" into "built"
  - state seniority, headcount, or budget figures not present in the sources
If a job requirement is not met, leave it out. A gap is honest; an invention is
disqualifying. Some of his strongest material is confidential: respect the
dossier's red-list guardrails (no revenue figures, no named pipeline companies,
the semiconductor partner stays anonymous, no compensation specifics).

## House style

Never use an em dash (-) or an en dash (-). Use a plain hyphen, a comma, or
two sentences. This is not negotiable: the candidate does not write that way.

## Density and wording

These are the rules a sharp reader applies. Each one was learned from a real
review of this CV.

- **One bullet, one outcome.** A bullet carrying four achievements hides all
  four. Split it. Aim for 25-35 words; never exceed 45.
- **Lead with the outcome, then how.** "Cut forecasting error 50%, a measured
  $6M saving" beats three clauses of setup before the number.
- **Put leadership scope first in a leadership role.** Headcount, team shape
  and company growth belong at the start of the bullet, not inside it.
- **Never make a claim that is trivially true.** "A small local model beat a
  frontier API on latency" is true of any local model and reads as padding.
  State the condition that makes it hard: at matched quality, on their
  benchmark, with the magnitude.
- **Use the conventional term.** "Product-market fit", not "market-fit".
- **Do not soften a claim the dossier records as measured** into "estimated".
  Equally, do not harden an estimate into a measurement.
- Cut connective tissue: "took the initiative to", "extensively used ... to
  help make customers successful", "a project which had previously". The
  outcome survives without them.

## What to tailor

1. `.subtitle` headline: the posting's EXACT job title, optionally with one
   adjacent title after a pipe. This is the single highest-value ATS edit.
2. `<title>`: "__CANDIDATE_NAME__ - <posting title>".
3. The leading HTML comment: "Tailored for: <Company> - <Title>. Base version:
   __BASE_CV_FILENAME__".
4. Professional Summary: rewritten to lead with what THIS posting is buying.
   Same facts, different order and emphasis.
5. Bullet emphasis and ordering: promote the bullets that answer the posting,
   demote or cut the ones that do not. Prefer cutting a whole bullet to
   shortening every bullet into mush. The `<span class="r">` lead-in label may
   be renamed from "Result:" to something the posting cares about
   (e.g. "Agentic AI in production:", "Observability:"), as the existing
   tailored example does.
6. The Skills block: re-group and re-title the three lines around the posting's
   own vocabulary, keeping only skills evidenced in the sources. Wrap any
   hyphenated or multi-word term the ATS will scan for literally in
   `<span class="nb">...</span>`.
7. Keep Education and Patents/Publications intact.

## ATS constraints on the HTML you emit

- Keep the section headings EXACTLY: "Professional Summary", "Professional
  Experience", "Skills", "Education". An ATS matches these literally.
- Never put `letter-spacing` or `font-variant: small-caps` on h2. Both corrupt
  PDF text extraction and are why an earlier version of this CV failed parsing.
- Keep the inline `li::before` bullet. Never make it absolutely positioned.
- Keep education entries as one inline row each, dates inline - never a
  right-aligned flex column.
- Keep `.nb { white-space: nowrap; }` in the stylesheet and use it.
- The result must fit on TWO A4 pages. Cutting bullets is how you achieve this.
"""


GROUNDING_INSTRUCTIONS = """\
You are auditing a tailored CV for fabrication, before it is sent to an
employer. The career dossier and the base CV supplied below are the ONLY
admissible evidence.

Extract every factual claim the tailored CV makes - each bullet, each summary
sentence, each skill term, each headline title - and for each one decide whether
the supplied sources support it.

Judge strictly:
  - `grounded: true` requires you to quote the supporting text in `evidence`.
    A paraphrase of a dossier fact is grounded. A reasonable-sounding inference
    is NOT.
  - Any number, date, headcount, employer, job title, technology or named
    outcome that does not appear in the sources is `grounded: false`.
  - A skill term is grounded if the sources show him doing that thing, even if
    the exact phrase differs. A skill term with no supporting activity is not.
  - Severity: "block" for an invented fact or inflated metric; "warn" for a
    claim that stretches the evidence or that a red-list guardrail covers;
    "info" for a wording nuance worth a look.

Do not list claims you judged grounded and unremarkable in bulk - report at most
the 12 most load-bearing grounded claims, and EVERY ungrounded one. Missing a
fabrication is the only unacceptable failure here.
"""


GROUNDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "grounded": {"type": "boolean"},
                    "evidence": {"type": "string"},
                    "source": {"type": "string", "enum": ["dossier", "base_cv", "none"]},
                    "severity": {"type": "string", "enum": ["info", "warn", "block"]},
                },
                "required": ["text", "grounded", "evidence", "source", "severity"],
                "additionalProperties": False,
            },
        },
        "verdict": {"type": "string", "enum": ["clean", "review", "reject"]},
        "summary": {"type": "string"},
    },
    "required": ["claims", "verdict", "summary"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Deterministic HTML hardening
# ---------------------------------------------------------------------------


def strip_code_fences(text: str) -> str:
    """Remove a ```html fence if the model wrapped its output in one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*\n", "", stripped)
        stripped = re.sub(r"\n```\s*$", "", stripped)
    return stripped.strip()


ATS_REPAIRS: tuple[tuple[str, str, str], ...] = (
    (
        r"font-variant:\s*small-caps",
        "font-variant: normal",
        "replaced small-caps on headings (it corrupts extracted text)",
    ),
    (
        r"(h2\s*\{[^}]*letter-spacing:\s*)[0-9.]+pt",
        r"\g<1>0",
        "zeroed heading letter-spacing (it injects stray spaces between characters)",
    ),
    (
        r"li\s*\{([^}]*?)position:\s*relative;?",
        r"li {\1text-indent: -8pt;",
        "made bullets inline rather than absolutely positioned",
    ),
    (
        r'li::before\s*\{\s*content:\s*"\\2022";\s*position:\s*absolute;\s*left:\s*[0-9.]+pt;',
        'li::before { content: "\\2022\\00a0\\00a0";',
        "attached bullet markers to their text",
    ),
    (
        r"<h2>Experience</h2>",
        "<h2>Professional Experience</h2>",
        "renamed Experience to the ATS-standard Professional Experience",
    ),
)


def repair_ats_html(html: str) -> tuple[str, list[str]]:
    """Fix known ATS hazards in a CV written against an older template.

    `harden_html` used to only warn about these. Warning is no use for a CV
    that already exists: the defects are mechanical and so is the repair.
    Every rule is a no-op on a CV that is already hardened.
    """
    notes: list[str] = []
    for pattern, replacement, description in ATS_REPAIRS:
        html, count = re.subn(pattern, replacement, html, flags=re.IGNORECASE | re.DOTALL)
        if count:
            notes.append(f"repaired: {description}")
    return html, notes


DASHES: tuple[tuple[str, str], ...] = (
    ("&mdash;", "-"),
    ("&ndash;", "-"),
    ("\u2014", "-"),   # em dash
    ("\u2013", "-"),   # en dash
)


def normalise_dashes(text: str) -> str:
    """Replace em and en dashes with a plain hyphen.

    A house-style rule, applied mechanically because asking the model not to
    reach for an em dash does not reliably stop it. Spacing is preserved, so
    "a - b" stays spaced and "well-known" stays closed up; the middot used as a
    separator in the CV template is left alone.
    """
    for needle, replacement in DASHES:
        text = text.replace(needle, replacement)
    return re.sub(r" +- +", " - ", text)


#: A bullet past this is carrying more than one outcome and hides all of them.
MAX_BULLET_WORDS = 45


def overlong_bullets(html: str, limit: int = MAX_BULLET_WORDS) -> list[tuple[int, str]]:
    """Bullets long enough that the outcome gets lost inside them.

    Reported rather than rewritten: which achievement to split out is a
    judgement call, and a mechanical cut would lose the wrong half.
    """
    # Skills and publications lines are legitimately long: they are delimited
    # lists, not prose, and splitting them would be wrong.
    skip: list[tuple[int, int]] = [
        (m.start(), m.end())
        for m in re.finditer(r'<ul class="(?:skills|pub)".*?</ul>', html, re.S | re.I)
    ]

    findings: list[tuple[int, str]] = []
    for match in re.finditer(r"<li>(.*?)</li>", html, re.S | re.I):
        if any(start <= match.start() < end for start, end in skip):
            continue
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", match.group(1))).strip()
        words = len(text.split())
        if words > limit:
            findings.append((words, text[:70]))
    return findings


def protect_keywords(html: str, keywords: Sequence[str]) -> tuple[str, int]:
    """Wrap protected keywords in ``<span class="nb">`` so they cannot wrap.

    The prompt asks the model to do this; it does not do it reliably, and a
    keyword split across a line ("cross-" / "functional") is not a literal
    match for an ATS. Doing it mechanically afterwards is the only way it is
    actually guaranteed.

    Only text between tags is touched - never tag internals, and never the
    contents of <style>, <title> or an existing .nb span.
    """
    terms = sorted({k.strip() for k in keywords if k and k.strip()}, key=len, reverse=True)
    if not terms:
        return html, 0
    pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)

    parts = re.split(r"(<[^>]*>)", html)
    skip_depth = 0
    in_nb = False
    wrapped = 0
    for index, part in enumerate(parts):
        if part.startswith("<"):
            tag = part.lower()
            if tag.startswith(("<style", "<title", "<script")):
                skip_depth += 1
            elif tag.startswith(("</style", "</title", "</script")):
                skip_depth = max(0, skip_depth - 1)
            elif "class=\"nb\"" in tag:
                in_nb = True
            elif tag.startswith("</span"):
                in_nb = False
            continue
        if skip_depth or in_nb or not part.strip():
            continue

        def _wrap(match: re.Match[str]) -> str:
            nonlocal wrapped
            wrapped += 1
            return f'<span class="nb">{match.group(0)}</span>'

        parts[index] = pattern.sub(_wrap, part)
    return "".join(parts), wrapped


def harden_html(html: str, nowrap_keywords: Sequence[str] = ()) -> tuple[str, list[str]]:
    """Re-apply the ATS CSS invariants. Returns ``(html, notes)``."""
    notes: list[str] = []

    for _, (pattern, snippet) in ATS_CSS_GUARDS.items():
        if not re.search(pattern, html, re.IGNORECASE | re.DOTALL):
            html = _inject_css(html, snippet)
            notes.append("injected missing .nb { white-space: nowrap } rule")

    dashed = normalise_dashes(html)
    if dashed != html:
        notes.append("replaced em and en dashes with hyphens")
        html = dashed

    html, repairs = repair_ats_html(html)
    notes.extend(repairs)

    # Warn only about hazards the repairs could not reach.
    for pattern, why in FORBIDDEN_CSS:
        if re.search(pattern, html, re.IGNORECASE | re.DOTALL):
            notes.append(f"WARNING: CSS still contains an ATS hazard - {why}")

    for words, snippet in overlong_bullets(html):
        notes.append(f"dense: a {words}-word bullet hides its outcome - {snippet}...")

    html, wrapped = protect_keywords(html, nowrap_keywords)
    if wrapped:
        notes.append(f"wrapped {wrapped} protected keyword(s) in .nb")

    return html, notes


def _inject_css(html: str, snippet: str) -> str:
    """Add a CSS rule just before the closing </style>, or a new style block."""
    if "</style>" in html:
        return html.replace("</style>", snippet + "</style>", 1)
    return f"<style>\n{snippet}</style>\n" + html


def html_to_text(html: str) -> str:
    """Flatten HTML to text for the grounding audit."""
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"(?i)</(p|div|li|h[1-6]|tr)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&middot;", "-").replace("&ouml;", "o")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_headline(html: str) -> str:
    match = re.search(r'class="subtitle"[^>]*>(.*?)</div>', html, re.IGNORECASE | re.DOTALL)
    return html_to_text(match.group(1)) if match else ""


def role_slug(posting: JobPosting) -> str:
    """A unique, readable identifier for one posting.

    Company alone is not enough. A single employer can have a dozen roles open,
    and the same title in several cities - Databricks currently lists "Manager,
    Forward Deployed Engineering" in Amsterdam, Singapore and Tokyo. Naming by
    company alone means the second CV silently overwrites the first and both
    job rows point at the survivor.

    The trailing hash comes from the job id, which is what actually guarantees
    uniqueness; the company and title words are there so a human can tell which
    file to attach.
    """
    company = re.sub(r"[^A-Za-z0-9]+", "", posting.company) or "Role"
    words = re.findall(r"[A-Za-z0-9]+", posting.title)[:4]
    title = "".join(word.capitalize() for word in words)
    job_id = posting.job_id or make_job_id(posting.company, posting.title, posting.url)
    suffix = job_id.rsplit("-", 1)[-1]
    return "_".join(part for part in (company, title, suffix) if part)[:70]


def output_stem(posting: JobPosting) -> str:
    """CV filename stem, e.g. ``DJ_Human_CV_Databricks_ManagerForwardDeployed_4597``."""
    return f"DJ_Human_CV_{role_slug(posting)}"


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


# Whitespace compaction steps, least damaging first. Each is (regex, replacement,
# description). Applied one at a time, re-rendering after each, until the CV
# fits. Never touches content and never reintroduces an ATS hazard - the same
# levers you would reach for by hand.
FIT_STEPS: tuple[tuple[str, str, str], ...] = (
    (r"(line-height:\s*)1\.3[5-9]", r"\g<1>1.32", "tightened line height"),
    (r"(li\s*\{[^}]*margin-bottom:\s*)1\.[6-9]pt", r"\g<1>1.3pt", "tightened bullet spacing"),
    (r"(h2\s*\{[^}]*margin:\s*)7pt 0 3\.5pt 0", r"\g<1>5.5pt 0 3pt 0", "tightened heading margins"),
    (r"(\.entry\s*\{[^}]*margin-bottom:\s*)4pt", r"\g<1>2.5pt", "tightened entry spacing"),
    (r"(font-size:\s*)9\.5pt", r"\g<1>9.2pt", "reduced body font size"),
    (r"(line-height:\s*)1\.32", r"\g<1>1.28", "tightened line height further"),
)


def fit_to_pages(
    cfg,
    html: str,
    html_path: Path,
    pdf_path: Path,
    max_pages: int,
) -> tuple[str, int, list[str]]:
    """Render, and compact whitespace until the CV fits the page limit.

    Returns ``(html, page_count, notes)``. Content is never cut - that is a
    judgement call for a human. If whitespace alone cannot make it fit, the
    caller is told how many pages remain so it can say so plainly.
    """
    from .ats import extract_text
    from .render import render_from_config

    html_path.write_text(html, encoding="utf-8")
    render_from_config(cfg, html_path, pdf_path)
    pages = extract_text(pdf_path)[1]
    notes: list[str] = []

    for pattern, replacement, description in FIT_STEPS:
        if pages <= max_pages:
            break
        compacted, count = re.subn(pattern, replacement, html, flags=re.IGNORECASE | re.DOTALL)
        if not count:
            continue
        html = compacted
        html_path.write_text(html, encoding="utf-8")
        render_from_config(cfg, html_path, pdf_path)
        pages = extract_text(pdf_path)[1]
        notes.append(f"{description} -> {pages} page(s)")

    return html, pages, notes


ADVERSARIAL_INSTRUCTIONS = """\
You are a hostile reader of one tailored CV. Not a proofreader and not a
cheerleader: a sceptical hiring manager who has read six hundred of these and
is looking for the reason to stop reading this one.

The candidate's dossier and base CV are supplied so you can tell a weak claim
from a badly-worded strong one. Do not check whether claims are TRUE - a
separate grounding pass does that. Check whether they are DEFENSIBLE, CLEAR and
WORTH THE READER'S TIME.

Find, in order of value:

1. **Claims that invite a question the candidate would struggle to answer.**
   A number with no baseline. A comparison whose conditions are unstated. A
   causal chain that skips a step, such as a model improvement producing a
   labour saving with nothing in between.
2. **Trivially true claims.** "A small local model beat a frontier API on
   latency" is true of every local model. If the impressive part is a
   condition (at matched quality, on their benchmark), the claim is hiding it.
3. **Bullets carrying more than one outcome**, where the best achievement is
   buried mid-sentence.
4. **Vague seniority or scope language** where a specific one exists: "worked
   with", "was involved in", "helped to", "supported".
5. **Anything that reads as inflation** to someone predisposed to think so:
   aspirational titles, borrowed credit, plural where the truth is singular.
6. **Internal inconsistencies**: dates, titles, headcounts that disagree with
   each other or with the dossier.

Be specific and quote the text. Do not invent problems to fill a quota; an
empty list is a valid and useful answer. Severity is "blocking" only if you
would stop reading or doubt the whole document.
"""

ADVERSARIAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "critiques": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue": {"type": "string"},
                    "severity": {"type": "string", "enum": ["blocking", "major", "minor"]},
                    "quote": {"type": "string"},
                    "why": {"type": "string"},
                    "fix": {"type": "string"},
                },
                "required": ["issue", "severity", "quote", "why", "fix"],
                "additionalProperties": False,
            },
        },
        "overall": {"type": "string"},
    },
    "required": ["critiques", "overall"],
    "additionalProperties": False,
}


def adversarial_review(
    tailored_html: str, posting: JobPosting, cfg: Config, claude: ClaudeClient
) -> list[Critique]:
    """Read the generated CV as a sceptical hiring manager would."""
    payload = claude.structured(
        instructions=ADVERSARIAL_INSTRUCTIONS,
        stable_context=stable_context_for(cfg),
        user_content=(
            f"<target_role>{posting.company} - {posting.title}</target_role>\n\n"
            f"<tailored_cv_text>\n{html_to_text(tailored_html)}\n</tailored_cv_text>\n\n"
            "Attack this CV. Quote what you object to."
        ),
        schema=ADVERSARIAL_SCHEMA,
        stage="adversarial",
        dry_run_value={"critiques": [], "overall": "[dry-run]"},
    )
    order = {"blocking": 0, "major": 1, "minor": 2}
    critiques = [
        Critique(
            issue=str(c.get("issue", "")),
            severity=str(c.get("severity", "minor")),
            quote=str(c.get("quote", "")),
            why=str(c.get("why", "")),
            fix=str(c.get("fix", "")),
        )
        for c in payload.get("critiques", [])
    ]
    return sorted(critiques, key=lambda c: order.get(c.severity, 3))


def tailor_instructions(cfg: Config) -> str:
    """Fill the candidate-specific placeholders in TAILOR_INSTRUCTIONS.

    Uses str.replace rather than str.format: the instructions contain literal
    CSS braces that would need escaping.
    """
    name = cfg.get("candidate", "name", "the candidate")
    return TAILOR_INSTRUCTIONS.replace("__CANDIDATE_NAME__", str(name)).replace(
        "__BASE_CV_FILENAME__", cfg.base_cv.name
    )


def _tailor_once(
    posting: JobPosting,
    cfg: Config,
    claude: ClaudeClient,
    *,
    render: bool = True,
    verify_claims: bool = True,
    adversarial: bool = True,
    prior_critiques: Sequence[Critique] = (),
    on_delta: Callable[[str], None] | None = None,
) -> TailorResult:
    """Generate, harden, render and ground-check a tailored CV. One pass."""
    out_dir = cfg.ensure_output_dir() / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(posting)
    html_path = out_dir / f"{stem}.html"
    pdf_path = out_dir / f"{stem}.pdf"

    stable = stable_context_for(cfg)

    html = claude.stream_text(
        instructions=tailor_instructions(cfg),
        stable_context=stable,
        user_content=_tailor_prompt(posting, prior_critiques),
        stage="tailor",
        dry_run_value=cfg.read_base_cv(),
        on_delta=on_delta,
    )
    usage = claude.last_usage

    html = strip_code_fences(html)
    if "<h2>" not in html and not claude.dry_run:
        raise RuntimeError(
            "Tailoring produced something that is not an HTML CV. "
            "Re-run; if it persists, check the model id in config.toml."
        )
    html, notes = harden_html(html, cfg.get("ats", "nowrap_keywords", []) or [])
    for note in notes:
        log.warning("%s", note)

    if claude.dry_run:
        log.info("[dry-run] would write %s", html_path)
    else:
        html_path.write_text(html, encoding="utf-8")

    result = TailorResult(
        job_id=posting.job_id,
        html_path=str(html_path),
        headline=extract_headline(html),
        cache_read_tokens=usage.cache_read_input_tokens,
        cache_creation_tokens=usage.cache_creation_input_tokens,
    )

    if render and not claude.dry_run:
        # Render, then compact whitespace until it fits. The model is not
        # reliable about length, and a three-page CV fails the page check no
        # matter how good the prose is.
        max_pages = int(cfg.get("ats", "max_pages", 2) or 2)
        html, pages, fit_notes = fit_to_pages(cfg, html, html_path, pdf_path, max_pages)
        for note in fit_notes:
            log.info("fit: %s", note)
        result.pages = pages
        result.fit_notes = fit_notes
        if pages > max_pages:
            log.warning(
                "still %d pages after compaction (limit %d) - cut a bullet",
                pages, max_pages,
            )
        result.pdf_path = str(pdf_path)
    elif render:
        log.info("[dry-run] would render %s", pdf_path)
        result.pdf_path = str(pdf_path)

    if verify_claims:
        result.claims = ground_claims(html, posting, cfg, claude)
    if adversarial and not claude.dry_run:
        result.critiques = adversarial_review(html, posting, cfg, claude)

    return result


def tailor_cv(
    posting: JobPosting,
    cfg: Config,
    claude: ClaudeClient,
    *,
    render: bool = True,
    verify_claims: bool = True,
    adversarial: bool = True,
    prior_critiques: Sequence[Critique] = (),
    on_delta: Callable[[str], None] | None = None,
    max_passes: int = 2,
) -> TailorResult:
    """Tailor a CV, and re-tailor once if the review finds a blocking problem.

    A blocking critique means the CV should not be sent. Writing it anyway and
    printing a red count leaves the fix depending on the reader noticing a
    status line and knowing that the remedy is to run the same command again.
    The critiques were already being fed forward; they were just being fed to
    the next run the user happened to start, which might be never.

    Each pass costs a full generation, so this is capped rather than looped
    until clean. If a blocking issue survives the second pass it is reported
    and left to a person, which is the right place for a judgement the model
    has already failed once.
    """
    critiques: list[Critique] = list(prior_critiques)
    result = None
    for attempt in range(1, max(1, max_passes) + 1):
        result = _tailor_once(
            posting, cfg, claude,
            render=render, verify_claims=verify_claims, adversarial=adversarial,
            prior_critiques=critiques, on_delta=on_delta,
        )
        result.passes = attempt
        result.prior_addressed = len(critiques)
        if not result.blocking or attempt >= max(1, max_passes) or claude.dry_run:
            break
        log.warning(
            "pass %d found %d blocking critique(s); re-tailoring: %s",
            attempt, len(result.blocking),
            "; ".join(c.issue for c in result.blocking)[:200],
        )
        # Feed everything forward, not just the blockers: a major issue left
        # unfixed on a rewrite tends to come back as a blocking one.
        critiques = critiques + list(result.critiques)
    return result


def ground_claims(
    tailored_html: str, posting: JobPosting, cfg: Config, claude: ClaudeClient
) -> list[Claim]:
    """Audit the generated CV against the fact base. Returns every claim."""
    payload = claude.structured(
        instructions=GROUNDING_INSTRUCTIONS,
        stable_context=stable_context_for(cfg),
        user_content=(
            f"<target_role>{posting.company} - {posting.title}</target_role>\n\n"
            f"<tailored_cv_text>\n{html_to_text(tailored_html)}\n</tailored_cv_text>\n\n"
            "Audit every claim in the tailored CV against the sources."
        ),
        schema=GROUNDING_SCHEMA,
        stage="ground",
        dry_run_value={"claims": [], "verdict": "clean", "summary": "[dry-run]"},
    )
    claims = [
        Claim(
            text=item.get("text", ""),
            grounded=bool(item.get("grounded")),
            evidence=item.get("evidence", ""),
            source=item.get("source", ""),
            severity=item.get("severity", "info"),
        )
        for item in payload.get("claims", [])
    ]
    log.info(
        "grounding verdict=%s: %d claims, %d ungrounded",
        payload.get("verdict"),
        len(claims),
        sum(1 for c in claims if not c.grounded),
    )
    return claims


def _tailor_prompt(
    posting: JobPosting, prior_critiques: Sequence[Critique] = ()
) -> str:
    critique_block: list[str] = []
    if prior_critiques:
        critique_block = [
            "",
            "<previous_version_was_criticised_for>",
            "A sceptical reader raised these against the last version of this CV.",
            "Address each one. Do NOT invent anything to do so: if a finding can",
            "only be answered with a fact that is not in the sources, cut the",
            "claim instead of strengthening it.",
            "",
        ]
        for c in prior_critiques:
            critique_block.append(f"- [{c.severity}] {c.issue}")
            if c.quote:
                critique_block.append(f'    text: "{c.quote}"')
            if c.fix:
                critique_block.append(f"    suggested: {c.fix}")
        critique_block.append("</previous_version_was_criticised_for>")

    return "\n".join(
        [
            "<job_posting>",
            f"Company: {posting.company}",
            f"Exact job title: {posting.title}",
            f"Location: {posting.location}",
            f"Department: {posting.department}",
            f"URL: {posting.url}",
            "",
            posting.description or "(no description text available)",
            "</job_posting>",
            *critique_block,
            "",
            "Produce the complete tailored HTML CV now. Output HTML only.",
        ]
    )


def format_claim_report(result: TailorResult) -> str:
    """Human-readable grounding summary for the CLI."""
    ungrounded = result.ungrounded
    lines: list[str] = []
    if not result.claims:
        return "  (no grounding audit was run)"
    blocking = [c for c in ungrounded if c.severity == "block"]
    lines.append(
        f"  Grounding audit: {len(result.claims)} claims checked, "
        f"{len(ungrounded)} could not be grounded ({len(blocking)} blocking)."
    )
    if not ungrounded:
        lines.append("  Every checked claim traces back to the dossier or the base CV.")
        return "\n".join(lines)
    lines.append("")
    lines.append("  UNGROUNDED CLAIMS - read these before sending anything:")
    for claim in ungrounded:
        lines.append(f"    [{claim.severity.upper()}] {claim.text}")
        if claim.evidence:
            lines.append(f"       closest evidence: {claim.evidence}")
    return "\n".join(lines)


__all__ = [
    "extract_headline",
    "format_claim_report",
    "ground_claims",
    "harden_html",
    "html_to_text",
    "output_stem",
    "strip_code_fences",
    "tailor_cv",
]
