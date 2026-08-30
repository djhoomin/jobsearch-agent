"""Tailoring: HTML hardening, grounding audit, and the prompt-cache layout.

The generation call itself is mocked - these tests pin the deterministic parts
around it, which are the parts that keep a fabricated claim from shipping.
"""

from __future__ import annotations

import pytest

from jobsearch.models import Claim, TailorResult
from jobsearch.tailor import (
    extract_headline,
    format_claim_report,
    ground_claims,
    harden_html,
    html_to_text,
    output_stem,
    strip_code_fences,
    tailor_cv,
)


class TestCodeFences:
    def test_strips_a_fenced_block(self):
        assert strip_code_fences("```html\n<p>hi</p>\n```") == "<p>hi</p>"

    def test_leaves_plain_html_alone(self):
        assert strip_code_fences("<p>hi</p>") == "<p>hi</p>"


class TestHarden:
    def test_injects_the_missing_nowrap_rule(self, base_cv_html):
        stripped = base_cv_html.replace("  .nb { white-space: nowrap; }\n", "")
        assert ".nb { white-space: nowrap" not in stripped
        hardened, notes = harden_html(stripped)
        assert ".nb { white-space: nowrap" in hardened
        assert any("injected" in n for n in notes)

    def test_leaves_a_correct_stylesheet_untouched(self, base_cv_html):
        hardened, notes = harden_html(base_cv_html)
        assert hardened == base_cv_html
        assert notes == []

    def test_flags_reintroduced_small_caps(self):
        html = "<style>h2 { font-variant: small-caps; }\n.nb { white-space: nowrap; }</style>"
        _, notes = harden_html(html)
        assert any("small-caps" in n for n in notes)

    def test_flags_reintroduced_letter_spacing(self):
        html = "<style>h2 { letter-spacing: 2.4pt; }\n.nb { white-space: nowrap; }</style>"
        _, notes = harden_html(html)
        assert any("letter-spacing" in n for n in notes)

    def test_flags_absolutely_positioned_bullets(self):
        html = (
            "<style>li::before { content: '•'; position: absolute; }\n"
            ".nb { white-space: nowrap; }</style>"
        )
        _, notes = harden_html(html)
        assert any("bullet" in n for n in notes)

    def test_injection_works_without_a_style_block(self):
        hardened, _ = harden_html("<p>no styles here</p>")
        assert "white-space: nowrap" in hardened


class TestHelpers:
    def test_extracts_the_headline(self, base_cv_html):
        assert extract_headline(base_cv_html).startswith("Head of AI")

    def test_html_to_text_drops_markup(self, base_cv_html):
        text = html_to_text(base_cv_html)
        assert "<div" not in text
        assert "Professional Summary" in text

    def test_output_stem_is_filesystem_safe(self):
        from jobsearch.models import JobPosting

        posting = JobPosting(company="Dash0 / Acme, Inc.", title="Director", url="https://x")
        assert output_stem(posting) == "DJ_Human_CV_Dash0AcmeInc"


class TestTailorStage:
    """The stage function, with generation and grounding both mocked."""

    @pytest.fixture
    def wired(self, fake_claude, base_cv_html):
        fake_claude.stream_response = base_cv_html
        fake_claude.structured_responses["ground"] = {
            "claims": [
                {
                    "text": "Director of Research since May 2024",
                    "grounded": True,
                    "evidence": "Director of Engineering at Northwind Labs since May 2024",
                    "source": "dossier",
                    "severity": "info",
                },
                {
                    "text": "Managed a budget of EUR 4 million",
                    "grounded": False,
                    "evidence": "",
                    "source": "none",
                    "severity": "block",
                },
            ],
            "verdict": "review",
            "summary": "one invented figure",
        }
        return fake_claude

    def test_writes_html_and_reports_ungrounded_claims(self, cfg, wired, posting):
        result = tailor_cv(posting, cfg, wired, render=False)
        assert result.html_path.endswith("DJ_Human_CV_Weaviate.html")
        assert len(result.claims) == 2
        assert len(result.ungrounded) == 1
        assert result.ungrounded[0].severity == "block"

    def test_the_generated_file_lands_in_the_output_dir(self, cfg, wired, posting):
        result = tailor_cv(posting, cfg, wired, render=False)
        from pathlib import Path

        assert Path(result.html_path).is_file()
        assert cfg.output_dir in Path(result.html_path).parents

    def test_never_writes_outside_the_output_dir(self, cfg, wired, posting):
        """The user's real source documents must be untouched by tailoring."""
        before = cfg.base_cv.read_text(encoding="utf-8")
        tailor_cv(posting, cfg, wired, render=False)
        assert cfg.base_cv.read_text(encoding="utf-8") == before

    def test_grounding_can_be_skipped(self, cfg, wired, posting):
        result = tailor_cv(posting, cfg, wired, render=False, verify_claims=False)
        assert result.claims == []

    def test_cacheable_prefix_precedes_the_job_content(self, cfg, wired, posting):
        tailor_cv(posting, cfg, wired, render=False)
        generation = next(c for c in wired.calls if c["kind"] == "stream")
        labels = [label for label, _ in generation["stable_context"]]
        assert labels == ["career_dossier", "base_cv_html", "search_strategy"]
        assert "Director of Product" not in "".join(
            text for _, text in generation["stable_context"]
        )
        assert "Director of Product" in generation["user_content"]

    def test_grounding_reuses_the_same_prefix(self, cfg, wired, posting):
        """Same prefix, same order: otherwise the second call misses the cache."""
        tailor_cv(posting, cfg, wired, render=False)
        stream = next(c for c in wired.calls if c["kind"] == "stream")
        ground = next(c for c in wired.calls if c["stage"] == "ground")
        assert stream["stable_context"] == ground["stable_context"]

    def test_reports_cache_usage(self, cfg, wired, posting):
        result = tailor_cv(posting, cfg, wired, render=False)
        assert result.cache_read_tokens == 9000

    def test_non_html_output_is_rejected(self, cfg, fake_claude, posting):
        fake_claude.stream_response = "I'm sorry, I can't help with that."
        with pytest.raises(RuntimeError, match="not an HTML CV"):
            tailor_cv(posting, cfg, fake_claude, render=False, verify_claims=False)


class TestGroundClaims:
    def test_parses_the_audit_payload(self, cfg, fake_claude, posting, base_cv_html):
        fake_claude.structured_responses["ground"] = {
            "claims": [
                {"text": "a", "grounded": True, "evidence": "e", "source": "dossier",
                 "severity": "info"},
                {"text": "b", "grounded": False, "evidence": "", "source": "none",
                 "severity": "warn"},
            ],
            "verdict": "review",
            "summary": "",
        }
        claims = ground_claims(base_cv_html, posting, cfg, fake_claude)
        assert [c.grounded for c in claims] == [True, False]
        assert claims[1].severity == "warn"

    def test_the_audit_sees_flattened_cv_text(self, cfg, fake_claude, posting, base_cv_html):
        fake_claude.structured_responses["ground"] = {"claims": [], "verdict": "clean", "summary": ""}
        ground_claims(base_cv_html, posting, cfg, fake_claude)
        content = fake_claude.calls[-1]["user_content"]
        assert "<style>" not in content
        assert "Professional Summary" in content


class TestClaimReport:
    def test_clean_result_says_so(self):
        result = TailorResult(
            job_id="x",
            html_path="x.html",
            claims=[Claim(text="a", grounded=True, evidence="e")],
        )
        assert "traces back" in format_claim_report(result)

    def test_ungrounded_claims_are_printed_in_full(self):
        result = TailorResult(
            job_id="x",
            html_path="x.html",
            claims=[Claim(text="Managed EUR 4M", grounded=False, severity="block")],
        )
        report = format_claim_report(result)
        assert "UNGROUNDED" in report
        assert "Managed EUR 4M" in report
        assert "BLOCK" in report

    def test_no_audit_is_reported_honestly(self):
        report = format_claim_report(TailorResult(job_id="x", html_path="x.html"))
        assert "no grounding audit" in report


class TestProtectKeywords:
    """The prompt asks the model to wrap protected keywords; it does not do it
    reliably, and a split keyword is not a literal ATS match. So the hardening
    step does it mechanically.
    """

    KEYWORDS = ["cross-functional", "on-device", "human-in-the-loop"]

    def test_it_wraps_a_bare_keyword(self):
        from jobsearch.tailor import protect_keywords

        html, n = protect_keywords("<p>led cross-functional teams</p>", self.KEYWORDS)
        assert html == '<p>led <span class="nb">cross-functional</span> teams</p>'
        assert n == 1

    def test_it_is_idempotent(self):
        """Hardening twice must not nest spans."""
        from jobsearch.tailor import protect_keywords

        once, _ = protect_keywords("<p>on-device</p>", self.KEYWORDS)
        twice, n = protect_keywords(once, self.KEYWORDS)
        assert twice == once
        assert n == 0

    def test_it_ignores_style_and_title(self):
        from jobsearch.tailor import protect_keywords

        html, n = protect_keywords(
            "<style>.a{on-device}</style><title>on-device</title><p>on-device</p>",
            self.KEYWORDS,
        )
        assert n == 1, "only the body text should be wrapped"
        assert "<style>.a{on-device}</style>" in html

    def test_it_does_not_touch_tag_internals(self):
        from jobsearch.tailor import protect_keywords

        html, _ = protect_keywords('<a href="/on-device">link</a>', self.KEYWORDS)
        assert 'href="/on-device"' in html

    def test_it_matches_case_insensitively_and_preserves_case(self):
        from jobsearch.tailor import protect_keywords

        html, n = protect_keywords("<p>On-Device work</p>", self.KEYWORDS)
        assert '<span class="nb">On-Device</span>' in html
        assert n == 1

    def test_longer_keywords_win(self):
        from jobsearch.tailor import protect_keywords

        html, _ = protect_keywords("<p>human-in-the-loop</p>", ["human-in-the-loop", "loop"])
        assert '<span class="nb">human-in-the-loop</span>' in html

    def test_no_keywords_is_a_no_op(self):
        from jobsearch.tailor import protect_keywords

        assert protect_keywords("<p>x</p>", []) == ("<p>x</p>", 0)

    def test_harden_html_applies_it_and_reports(self):
        from jobsearch.tailor import harden_html

        html, notes = harden_html("<p>cross-functional</p>", self.KEYWORDS)
        assert '<span class="nb">cross-functional</span>' in html
        assert any("protected keyword" in n for n in notes)


class TestFitToPages:
    """The model is not reliable about length; a 3-page CV fails the page
    check no matter how good the prose is. Whitespace is compacted until it
    fits - content is never cut, because that is a judgement call.
    """

    def _html(self, cfg, body_repeats: int) -> str:
        base = cfg.base_cv.read_text(encoding="utf-8")
        filler = "<div class='entry'><ul>" + "".join(
            f"<li>Padding bullet number {i} with enough text to occupy a full line of the page.</li>"
            for i in range(body_repeats)
        ) + "</ul></div>"
        return base.replace("</body>", filler + "</body>") if "</body>" in base else base + filler

    def test_a_short_cv_is_left_alone(self, cfg, tmp_path):
        from jobsearch.tailor import fit_to_pages

        html = cfg.base_cv.read_text(encoding="utf-8")
        out, pages, notes = fit_to_pages(
            cfg, html, tmp_path / "a.html", tmp_path / "a.pdf", max_pages=2
        )
        assert pages <= 2
        assert notes == [], "nothing to compact when it already fits"
        assert out == html

    def test_an_overlong_cv_is_compacted_until_it_fits(self, cfg, tmp_path):
        from jobsearch.tailor import fit_to_pages

        html = self._html(cfg, 90)
        out, pages, notes = fit_to_pages(
            cfg, html, tmp_path / "b.html", tmp_path / "b.pdf", max_pages=2
        )
        assert notes, "compaction steps should have been applied"
        assert out != html

    def test_it_gives_up_rather_than_shrinking_forever(self, cfg, tmp_path):
        """Whitespace cannot fix everything; the caller is told what remains."""
        from jobsearch.tailor import FIT_STEPS, fit_to_pages

        html = self._html(cfg, 600)
        _, pages, notes = fit_to_pages(
            cfg, html, tmp_path / "c.html", tmp_path / "c.pdf", max_pages=2
        )
        assert len(notes) <= len(FIT_STEPS)
        assert pages > 2, "this fixture is far too long to fit"

    def test_compaction_only_touches_whitespace(self, cfg, tmp_path):
        """The base CV legitimately sets `letter-spacing: 0`; compaction must
        leave the ATS-critical declarations exactly as it found them."""
        import re

        from jobsearch.tailor import fit_to_pages

        html = self._html(cfg, 90)
        out, _, _ = fit_to_pages(
            cfg, html, tmp_path / "d.html", tmp_path / "d.pdf", max_pages=2
        )

        def critical(text: str) -> list[str]:
            return re.findall(r"(letter-spacing:\s*[^;]+|font-variant:\s*[^;]+)", text)

        assert critical(out) == critical(html)
        assert "white-space: nowrap" in out
