"""The ATS verifier.

Two layers of testing:

* **Unit level** - each check against synthetic extracted text, so the assertion
  logic is pinned regardless of what any particular Chrome build does.
* **End to end** - two fixture HTML files are rendered to PDF with the real
  headless Chrome and verified. ``good_cv.html`` must pass everything;
  ``bad_cv.html`` carries the five defects the real CV actually had and must
  fail. These are skipped where Chrome is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jobsearch.ats import (
    check_bullets_inline,
    check_education_lines,
    check_headings,
    check_heading_spacing,
    check_hyphen_wraps,
    check_page_count,
    extract_jd_terms,
    keyword_coverage,
    section_text,
    verify_pdf,
)

from .conftest import FIXTURES, requires_chrome

HEADINGS = ["Professional Summary", "Professional Experience", "Skills", "Education"]


# ---------------------------------------------------------------------------
# Unit level
# ---------------------------------------------------------------------------


class TestPageCount:
    def test_within_limit_passes(self):
        assert check_page_count(2, 2).status == "pass"

    def test_over_limit_fails(self):
        check = check_page_count(3, 2)
        assert check.status == "fail"
        assert "3 pages" in check.message


class TestHeadings:
    def test_all_present_passes(self):
        text = "\n".join(HEADINGS)
        assert check_headings(text, HEADINGS).status == "pass"

    def test_missing_heading_fails(self):
        text = "Professional Summary\nProfessional Experience\nSkills"
        check = check_headings(text, HEADINGS)
        assert check.status == "fail"
        assert any("Education" in d for d in check.details)

    def test_matching_is_whitespace_tolerant(self):
        text = "Professional   Summary\nProfessional Experience\nSkills\nEducation"
        assert check_headings(text, HEADINGS).status == "pass"

    def test_missing_optional_heading_only_warns(self):
        text = "\n".join(HEADINGS)
        check = check_headings(text, HEADINGS, ["Patents, Publications and Presentations"])
        assert check.status == "warn"


class TestHeadingIntegrity:
    """The letter-spacing + small-caps regression, the reason Jobscan failed."""

    def test_clean_headings_pass(self):
        text = "Professional Summary\nSome body text here.\nEducation"
        assert check_heading_spacing(text, HEADINGS).status == "pass"

    def test_stray_space_before_comma_fails(self):
        # What letter-spacing did to "Patents, Publications and Presentations".
        text = "Patents , Publications and Presentations"
        check = check_heading_spacing(text, ["Patents, Publications and Presentations"])
        assert check.status == "fail"
        assert any("stray space" in d for d in check.details)

    def test_letter_spaced_glyph_run_fails(self):
        text = "E D U C A T I O N"
        check = check_heading_spacing(text, HEADINGS)
        assert check.status == "fail"
        assert any("letter-spaced" in d for d in check.details)

    def test_ordinary_body_text_is_not_flagged(self):
        text = (
            "Built the evaluation stack behind the platform, including scoring "
            "and regression tests, across two product lines."
        )
        assert check_heading_spacing(text, HEADINGS).status == "pass"


class TestEducationEntries:
    def test_dates_on_the_same_line_pass(self):
        text = (
            "Education\n"
            "Example University - Masters degree: Operations Research - Mar 2016 - Sep 2019\n"
            "Example University - BSc: Mathematical Sciences - Feb 2011 - Nov 2013\n"
        )
        assert check_education_lines(text, HEADINGS).status == "pass"

    def test_detached_dates_fail(self):
        # The right-aligned flex column failure mode: degrees, then dates.
        text = (
            "Education\n"
            "Example University - Masters degree: Operations Research\n"
            "Example University - BSc: Mathematical Sciences\n"
            "Mar 2016 - Sep 2019\n"
        )
        check = check_education_lines(text, HEADINGS)
        assert check.status == "fail"
        assert "detached" in check.message

    def test_missing_section_fails(self):
        assert check_education_lines("Skills\nPython", HEADINGS).status == "fail"

    def test_present_end_date_is_accepted(self):
        text = "Education\nExample University - PhD - Mar 2016 - Present\n"
        assert check_education_lines(text, HEADINGS).status == "pass"

    def test_section_text_stops_at_the_next_heading(self):
        text = "Education\nA degree - Mar 2016 - Sep 2019\nSkills\nPython\n"
        assert "Python" not in section_text(text, "Education", HEADINGS)


class TestBulletsInline:
    def test_inline_bullets_pass(self):
        text = "• Own the research-to-product portfolio\n• Took the platform to market-fit"
        assert check_bullets_inline(text).status == "pass"

    def test_orphaned_bullets_fail(self):
        text = "•\n•\nOwn the research-to-product portfolio\nTook the platform to market-fit"
        check = check_bullets_inline(text)
        assert check.status == "fail"
        assert "2 bullet marker" in check.message


class TestHyphenWraps:
    KEYWORDS = ["human-in-the-loop", "multi-agent", "on-device"]

    def test_intact_keyword_passes(self):
        text = "Experienced with human-in-the-loop review and multi-agent systems."
        assert check_hyphen_wraps(text, self.KEYWORDS).status == "pass"

    def test_protected_keyword_split_fails(self):
        text = "governance and human-in-\nthe-loop review of model output"
        check = check_hyphen_wraps(text, self.KEYWORDS)
        assert check.status == "fail"
        assert any("human-in-the-loop" in d for d in check.details)

    def test_unprotected_hyphen_wrap_only_warns(self):
        # The real base CV wraps "knowledge-graph" and still passes overall.
        text = "conceived a knowledge-\ngraph engine for narrative consistency"
        check = check_hyphen_wraps(text, self.KEYWORDS)
        assert check.status == "warn"
        assert check.ok

    def test_keyword_only_matching_across_a_break_fails(self):
        text = "we use multi-\nagent orchestration"
        assert check_hyphen_wraps(text, self.KEYWORDS).status == "fail"


class TestKeywordCoverage:
    JD = (
        "We are hiring a Director of AI to own our agent platform. You will build "
        "evaluation pipelines, knowledge graphs, and retrieval augmented generation "
        "systems. Experience with knowledge graphs and evaluation pipelines required."
    )

    def test_extracts_repeated_domain_terms(self):
        terms = extract_jd_terms(self.JD)
        joined = " ".join(terms)
        assert "knowledge graphs" in joined
        assert "evaluation pipelines" in joined

    def test_drops_stopwords_and_boilerplate(self):
        terms = extract_jd_terms(self.JD)
        assert "you will" not in terms
        assert "the" not in terms

    def test_coverage_splits_present_and_missing(self):
        cv = "I built knowledge graphs and evaluation pipelines."
        coverage = keyword_coverage(cv, self.JD)
        assert "knowledge graphs" in coverage.present
        assert coverage.missing, "a real CV never covers every JD term"
        assert 0.0 < coverage.ratio < 1.0

    def test_empty_jd_is_full_coverage_not_a_crash(self):
        assert keyword_coverage("anything", "").ratio == 1.0

    def test_ranking_prefers_longer_phrases(self):
        terms = extract_jd_terms(self.JD, max_terms=10)
        # "knowledge graphs" should be chosen instead of a bare "knowledge".
        assert not any(t == "knowledge" for t in terms)


# ---------------------------------------------------------------------------
# End to end: render fixture HTML with Chrome, then verify the PDF
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> dict[str, Path]:
    """Render both fixture CVs once for the whole module."""
    from jobsearch.render import html_to_pdf

    out = tmp_path_factory.mktemp("pdfs")
    paths: dict[str, Path] = {}
    for name in ("good_cv", "bad_cv"):
        paths[name] = html_to_pdf(FIXTURES / f"{name}.html", out / f"{name}.pdf")
    return paths


NOWRAP = ["human-in-the-loop", "multi-agent", "retrieval-augmented", "on-device"]


@requires_chrome
class TestEndToEnd:
    def test_hardened_cv_passes_every_check(self, rendered):
        report = verify_pdf(
            rendered["good_cv"],
            nowrap_keywords=NOWRAP,
            optional_headings=["Patents, Publications and Presentations"],
        )
        assert report.passed, report.render()
        assert report.page_count <= 2

    def test_broken_cv_fails(self, rendered):
        report = verify_pdf(rendered["bad_cv"], nowrap_keywords=NOWRAP)
        assert not report.passed
        failed = {c.name for c in report.failures}
        # Every planted defect must be caught, not just one of them.
        assert "section_headings" in failed
        assert "heading_integrity" in failed
        assert "education_entries" in failed
        assert "bullets_inline" in failed
        assert "hyphen_wraps" in failed

    def test_broken_cv_names_the_split_keyword(self, rendered):
        report = verify_pdf(rendered["bad_cv"], nowrap_keywords=NOWRAP)
        check = next(c for c in report.checks if c.name == "hyphen_wraps")
        assert any("human-in-the-loop" in d for d in check.details)

    def test_coverage_report_runs_against_a_jd(self, rendered):
        report = verify_pdf(
            rendered["good_cv"],
            nowrap_keywords=NOWRAP,
            jd_text="We need knowledge graphs, multi-agent systems and Kubernetes at scale.",
        )
        assert report.coverage is not None
        assert report.coverage.total > 0

    def test_report_serialises(self, rendered):
        payload = verify_pdf(rendered["good_cv"]).to_dict()
        assert payload["passed"] is True
        assert {c["name"] for c in payload["checks"]} >= {"page_count", "bullets_inline"}


def test_missing_pdf_raises():
    with pytest.raises(FileNotFoundError):
        verify_pdf("/nonexistent/file.pdf")
