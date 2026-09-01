"""ATS verifier: does the rendered PDF survive machine parsing?

This is the component that stops a beautifully formatted CV from being silently
mangled by an applicant tracking system. It reads the PDF's *text layer* with
``pypdf`` - the same thing an ATS parser sees - and asserts a set of properties
that were each learned the hard way on this CV:

1. **Page count** within the configured limit.
2. **Standard section headings present and cleanly extractable.** A parser that
   cannot find "Education" does not merely mis-render it; it reports the
   candidate as having no degree.
3. **No heading corrupted by letter-spacing artifacts.** The base CV originally
   set ``letter-spacing`` plus ``font-variant: small-caps`` on ``h2``. Chrome's
   PDF writer emits those as individually positioned glyphs, and extraction
   turns "Patents, Publications" into "Patents , Publications" - which is why
   Jobscan could not find the Education section. The template no longer does
   this; this check is the regression test.
4. **Education entries on one line each, with their own date range.** The dates
   used to live in a right-aligned flex column, which extraction reorders,
   detaching every date from its degree.
5. **Bullet markers inline with their text.** An absolutely positioned
   ``li::before`` orphans every bullet glyph onto its own line, so the parser
   sees a column of dots and a column of unattributed sentences.
6. **No hyphenated keyword broken across a line wrap.** "human-in-the-loop"
   split as "human-in-" / "the-loop" is not a literal match for the JD term.
   The template wraps these in ``<span class="nb">``.
7. **Keyword coverage** against the job description, reported honestly as
   present-vs-missing rather than as a single reassuring number.

Runnable standalone: ``jobsearch verify path/to/cv.pdf [--jd-file jd.txt]``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from pypdf import PdfReader

SEVERITY_ORDER = {"pass": 0, "warn": 1, "fail": 2}

#: A date range as it appears on this CV: "Mar 2016 - Sep 2019", "May 2024 - Present".
DATE_RANGE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*"
    r"[-‐-―−]\s*"
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|Present|Current|Now)\b",
    re.IGNORECASE,
)

BULLET_CHARS = "•‣▪●·⁃-"

STOPWORDS = {
    "a", "about", "above", "across", "after", "all", "also", "an", "and", "any", "are", "as",
    "at", "be", "been", "being", "both", "but", "by", "can", "could", "do", "does", "doing",
    "each", "either", "else", "etc", "every", "for", "from", "get", "great", "had", "has",
    "have", "help", "her", "here", "his", "how", "in", "into", "is", "it", "its", "just",
    "like", "make", "many", "may", "more", "most", "must", "new", "not", "of", "on", "one",
    "only", "or", "other", "our", "out", "over", "own", "per", "role", "same", "see", "she",
    "should", "so", "some", "such", "team", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "up", "us", "use",
    "using", "very", "want", "was", "we", "well", "were", "what", "when", "where", "which",
    "while", "who", "why", "will", "with", "within", "work", "working", "would", "you",
    "your", "years", "year", "experience", "job", "company", "position", "candidate",
    "including", "ability", "strong", "excellent", "good", "plus", "nice", "must", "you'll",
    "we're", "you're", "join", "looking", "someone", "people", "day", "days", "week",
}


@dataclass
class Check:
    """One verifier assertion."""

    name: str
    status: str  # pass | warn | fail
    message: str
    details: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status != "fail"

    def render(self) -> str:
        icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[self.status]
        lines = [f"  [{icon}] {self.name}: {self.message}"]
        lines += [f"         - {d}" for d in self.details[:12]]
        if len(self.details) > 12:
            lines.append(f"         ... and {len(self.details) - 12} more")
        return "\n".join(lines)


@dataclass
class KeywordCoverage:
    """JD terms present in, and missing from, the CV."""

    present: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.present) + len(self.missing)

    @property
    def ratio(self) -> float:
        return len(self.present) / self.total if self.total else 1.0


@dataclass
class ATSReport:
    """Full verification result for one PDF."""

    pdf_path: str
    page_count: int
    checks: list[Check] = field(default_factory=list)
    coverage: KeywordCoverage | None = None
    text: str = field(default="", repr=False)

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == "warn"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pdf_path": self.pdf_path,
            "page_count": self.page_count,
            "passed": self.passed,
            "checks": [
                {"name": c.name, "status": c.status, "message": c.message, "details": c.details}
                for c in self.checks
            ],
            "coverage": (
                {
                    "present": self.coverage.present,
                    "missing": self.coverage.missing,
                    "ratio": round(self.coverage.ratio, 3),
                }
                if self.coverage
                else None
            ),
        }

    def render(self) -> str:
        head = "ATS VERIFICATION: " + ("PASS" if self.passed else "FAIL")
        lines = [head, f"  file: {self.pdf_path}", f"  pages: {self.page_count}", ""]
        lines += [c.render() for c in self.checks]
        if self.coverage and self.coverage.total:
            lines.append("")
            lines.append(
                f"  Keyword coverage: {len(self.coverage.present)}/{self.coverage.total} "
                f"({self.coverage.ratio:.0%}) of job-description terms appear in the CV"
            )
            if self.coverage.present:
                lines.append("    present: " + ", ".join(self.coverage.present[:25]))
            if self.coverage.missing:
                lines.append("    MISSING: " + ", ".join(self.coverage.missing[:25]))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text(pdf_path: str | Path) -> tuple[str, int]:
    """Return ``(text, page_count)`` from a PDF's text layer."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"No such PDF: {path}")
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages), len(reader.pages)


def _lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines()]


def _norm(value: str) -> str:
    """Collapse whitespace and lowercase, for tolerant heading comparison."""
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_page_count(page_count: int, max_pages: int) -> Check:
    if page_count > max_pages:
        return Check(
            "page_count",
            "fail",
            f"{page_count} pages exceeds the {max_pages}-page limit",
        )
    return Check("page_count", "pass", f"{page_count} page(s), limit {max_pages}")


def check_headings(text: str, required: Sequence[str], optional: Sequence[str] = ()) -> Check:
    """Every required section heading must appear, whitespace-normalised."""
    flat = _norm(text)
    missing = [h for h in required if _norm(h) not in flat]
    missing_optional = [h for h in optional if _norm(h) not in flat]
    if missing:
        return Check(
            "section_headings",
            "fail",
            f"{len(missing)} required heading(s) not found in the extracted text",
            details=[f"missing: {h!r}" for h in missing],
        )
    details = [f"optional heading not found: {h!r}" for h in missing_optional]
    return Check(
        "section_headings",
        "warn" if details else "pass",
        f"all {len(required)} required headings extract cleanly",
        details=details,
    )


def check_heading_spacing(text: str, headings: Sequence[str]) -> Check:
    """Detect letter-spacing / small-caps corruption around headings.

    Two symptoms, both produced by per-glyph positioning:

    * a space before a comma or other punctuation ("Patents , Publications")
    * a run of single characters separated by spaces ("E d u c a t i o n")
    """
    problems: list[str] = []
    for line in _lines(text):
        stripped = line.strip()
        if not stripped or len(stripped) > 90:
            continue
        looks_like_heading = any(
            _norm(h).split()[0] in _norm(stripped) for h in headings if h
        ) or bool(re.fullmatch(r"[A-Z][A-Za-z,&' ]{3,60}", stripped))
        if not looks_like_heading:
            continue
        if re.search(r"\s+[,;:.]", stripped):
            problems.append(f"stray space before punctuation: {stripped!r}")
        tokens = stripped.split()
        singles = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
        if len(tokens) >= 4 and singles / len(tokens) > 0.5:
            problems.append(f"letter-spaced glyph run: {stripped!r}")
    if problems:
        return Check(
            "heading_integrity",
            "fail",
            "heading text is corrupted - check for letter-spacing or small-caps on h2",
            details=sorted(set(problems)),
        )
    return Check(
        "heading_integrity", "pass", "no letter-spacing or small-caps artifacts in headings"
    )


#: A line that looks like a section heading even if we were not told about it:
#: short, title-case, no digits. Stops a section body from swallowing the next
#: section when the caller did not list every heading on the page.
_HEADING_SHAPE = re.compile(r"^[A-Z][A-Za-z,&'()/ -]{3,60}$")


def looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    return bool(_HEADING_SHAPE.fullmatch(stripped)) and not any(
        ch.isdigit() for ch in stripped
    )


def section_text(text: str, heading: str, all_headings: Sequence[str]) -> str:
    """Return the body of one section, up to the next heading."""
    lines = _lines(text)
    others = {_norm(h) for h in all_headings if _norm(h) != _norm(heading)}
    start = None
    for i, line in enumerate(lines):
        if _norm(line).startswith(_norm(heading)):
            start = i + 1
            break
    if start is None:
        return ""
    body: list[str] = []
    for line in lines[start:]:
        if _norm(line) in others or any(_norm(line).startswith(o) for o in others if o):
            break
        if body and looks_like_heading(line):
            break
        body.append(line)
    return "\n".join(body).strip()


#: Tokens that mark a line under Education as an actual qualification. A
#: Languages or Certifications line legitimately carries no date range, and
#: failing it would train you to ignore this check.
DEGREE_TOKENS = (
    "university", "college", "institute", "school",
    "bsc", "msc", "ba ", "bcom", "mba", "phd", "bachelor", "master", "honours",
    "doctorate", "diploma", "degree",
)


def looks_like_a_qualification(line: str) -> bool:
    """True when a line under Education names an institution or a degree."""
    lowered = line.lower()
    return any(token in lowered for token in DEGREE_TOKENS)


def check_education_lines(text: str, all_headings: Sequence[str]) -> Check:
    """Each education entry must extract on ONE line, with its own date range.

    Only qualification lines are checked. A Languages line under the same
    heading has no date by design, and flagging it would be a false positive
    on a document that is correct.
    """
    body = section_text(text, "Education", all_headings)
    if not body:
        return Check(
            "education_entries",
            "fail",
            "no Education section body found after the heading",
        )
    entries = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and looks_like_a_qualification(line)
    ]
    if not entries:
        return Check("education_entries", "fail", "Education section is empty")
    problems = [e for e in entries if not DATE_RANGE.search(e)]
    if problems:
        return Check(
            "education_entries",
            "fail",
            f"{len(problems)} of {len(entries)} education line(s) have no date range on the "
            "same line - dates detached from degrees (right-aligned flex column?)",
            details=[f"no date on: {p!r}" for p in problems],
        )
    return Check(
        "education_entries",
        "pass",
        f"all {len(entries)} education entries extract on one line with their dates",
    )


def check_bullets_inline(text: str) -> Check:
    """No line may consist of a bullet marker alone."""
    orphans = [
        (i + 1, line)
        for i, line in enumerate(_lines(text))
        if line.strip() and all(ch in BULLET_CHARS + " " for ch in line.strip())
    ]
    if orphans:
        return Check(
            "bullets_inline",
            "fail",
            f"{len(orphans)} bullet marker(s) extracted onto their own line - "
            "use an inline li::before, not an absolutely positioned one",
            details=[f"line {n}: {line!r}" for n, line in orphans],
        )
    return Check("bullets_inline", "pass", "bullet markers stay attached to their text")


def check_hyphen_wraps(text: str, keywords: Sequence[str]) -> Check:
    """No hyphenated *keyword* may be split across a line wrap.

    A generic hyphenated word wrapping ("knowledge-" / "graph engine") costs a
    literal keyword match too, but only for terms an ATS is actually scanning
    for. So a break that reconstructs into one of the configured
    ``nowrap_keywords`` is a FAIL; any other hyphen wrap is reported as a WARN
    with the reconstructed word, so the user can decide whether to add a
    ``<span class="nb">`` around it.
    """
    lines = _lines(text)
    failures: list[str] = []
    warnings: list[str] = []
    needles = [k.lower() for k in keywords if k]

    # Symptom 1: a line ends mid-hyphenated-word. Reconstruct the broken word
    # and check it against the keyword list.
    for i, line in enumerate(lines[:-1]):
        stripped = line.rstrip()
        if not re.search(r"\w-$", stripped):
            continue
        nxt = lines[i + 1].strip()
        if not nxt or not (nxt[0].islower() or nxt[0].isdigit()):
            continue
        head = stripped.split()[-1] if stripped.split() else stripped
        tail = nxt.split()[0] if nxt.split() else nxt
        rejoined = (head + tail).strip(".,;:()").lower()
        hit = next((k for k in needles if k in rejoined), None)
        entry = f"{head!r} + {tail!r} -> {rejoined!r}"
        if hit:
            failures.append(f"{entry} splits protected keyword {hit!r}")
        else:
            warnings.append(entry)

    # Symptom 2: a configured keyword survives whitespace-flattening but is
    # absent from the raw text, i.e. it only exists across a line break.
    raw = text.lower()
    flat = re.sub(r"\s+", "", raw)
    for keyword in needles:
        if keyword not in raw and keyword.replace(" ", "") in flat:
            failures.append(f"keyword {keyword!r} only matches across a line break")

    if failures:
        return Check(
            "hyphen_wraps",
            "fail",
            "hyphenated keyword broken across a line wrap - wrap it in "
            '<span class="nb"> (white-space: nowrap)',
            details=sorted(set(failures)),
        )
    if warnings:
        return Check(
            "hyphen_wraps",
            "warn",
            f"{len(warnings)} hyphenated word(s) wrap across a line but none is a "
            "protected keyword - add them to [ats].nowrap_keywords if an ATS "
            "should match them literally",
            details=sorted(set(warnings)),
        )
    return Check("hyphen_wraps", "pass", "no hyphenated keyword split across a line wrap")


# ---------------------------------------------------------------------------
# Keyword coverage
# ---------------------------------------------------------------------------


def extract_jd_terms(jd_text: str, max_terms: int = 40) -> list[str]:
    """Pull candidate keyword terms out of a job description.

    Deterministic: lowercase, strip punctuation, build 1-3 word n-grams, drop
    anything containing only stopwords or short tokens, then rank by frequency
    (longer n-grams first, so "knowledge graphs" beats "knowledge").
    """
    text = re.sub(r"[^a-z0-9+#./\- ]+", " ", (jd_text or "").lower())
    words = [w for w in text.split() if w]
    counts: dict[str, int] = {}
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            gram = words[i : i + size]
            if any(w in STOPWORDS for w in gram):
                continue
            if any(len(w) < 3 for w in gram):
                continue
            if all(w.isdigit() for w in gram):
                continue
            term = " ".join(gram).strip("-./")
            if len(term) < 4:
                continue
            counts[term] = counts.get(term, 0) + (size * 2)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen: list[str] = []
    for term, _ in ranked:
        # Skip a term already covered by a longer chosen phrase.
        if any(term in longer for longer in chosen):
            continue
        chosen.append(term)
        if len(chosen) >= max_terms:
            break
    return chosen


def keyword_coverage(cv_text: str, jd_text: str, max_terms: int = 40) -> KeywordCoverage:
    """Which JD terms appear in the CV text, and which do not."""
    haystack = re.sub(r"\s+", " ", (cv_text or "").lower())
    present: list[str] = []
    missing: list[str] = []
    for term in extract_jd_terms(jd_text, max_terms=max_terms):
        (present if term in haystack else missing).append(term)
    return KeywordCoverage(present=present, missing=missing)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def verify_pdf(
    pdf_path: str | Path,
    *,
    max_pages: int = 2,
    required_headings: Sequence[str] = (
        "Professional Summary",
        "Professional Experience",
        "Skills",
        "Education",
    ),
    optional_headings: Sequence[str] = (),
    nowrap_keywords: Sequence[str] = (),
    jd_text: str = "",
    min_keyword_coverage: float = 0.0,
) -> ATSReport:
    """Run every ATS assertion against a rendered PDF."""
    text, page_count = extract_text(pdf_path)
    all_headings = [*required_headings, *optional_headings]

    report = ATSReport(pdf_path=str(pdf_path), page_count=page_count, text=text)
    report.checks = [
        check_page_count(page_count, max_pages),
        check_headings(text, required_headings, optional_headings),
        check_heading_spacing(text, all_headings),
        check_education_lines(text, all_headings),
        check_bullets_inline(text),
        check_hyphen_wraps(text, nowrap_keywords),
    ]

    if not text.strip():
        report.checks.insert(
            0,
            Check(
                "text_layer",
                "fail",
                "the PDF has no extractable text layer at all - an ATS sees an empty document",
            ),
        )

    if jd_text.strip():
        report.coverage = keyword_coverage(text, jd_text)
        status = "pass"
        if min_keyword_coverage and report.coverage.ratio < min_keyword_coverage:
            status = "warn"
        report.checks.append(
            Check(
                "keyword_coverage",
                status,
                f"{len(report.coverage.present)}/{report.coverage.total} job-description "
                f"terms present ({report.coverage.ratio:.0%})",
                details=[f"missing: {t}" for t in report.coverage.missing[:20]],
            )
        )
    return report


def verify_from_config(cfg, pdf_path: str | Path, jd_text: str = "") -> ATSReport:
    """Convenience wrapper reading the thresholds out of ``config.toml``."""
    ats = cfg.section("ats")
    return verify_pdf(
        pdf_path,
        max_pages=int(ats.get("max_pages", 2)),
        required_headings=ats.get(
            "required_headings",
            ["Professional Summary", "Professional Experience", "Skills", "Education"],
        ),
        optional_headings=ats.get("optional_headings", []),
        nowrap_keywords=ats.get("nowrap_keywords", []),
        jd_text=jd_text,
        min_keyword_coverage=float(ats.get("min_keyword_coverage", 0.0)),
    )


__all__ = [
    "ATSReport",
    "Check",
    "KeywordCoverage",
    "extract_jd_terms",
    "extract_text",
    "keyword_coverage",
    "verify_from_config",
    "verify_pdf",
]
