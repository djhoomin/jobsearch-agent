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
        # Company alone is no longer the whole stem: a role discriminator was
        # added so two roles at one employer cannot overwrite each other.
        stem = output_stem(posting)
        assert stem.startswith("DJ_Human_CV_Dash0AcmeInc")
        assert not set(stem) & set('/\\:*?"<>| ')


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
        assert result.html_path.endswith(".html")
        assert "DJ_Human_CV_Weaviate" in result.html_path
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


class TestRepairAtsHtml:
    """A CV written against an older template can be rescued mechanically.

    Warning about a hazard is no use for a file that already exists; every
    defect here is deterministic, so the repair is too.
    """

    OLD = (
        "<style>\n"
        '  h2 { font-variant: small-caps; letter-spacing: 1.2pt; margin: 8pt 0 4pt 0; }\n'
        "  li { padding-left: 9pt; position: relative; margin-bottom: 1.8pt; }\n"
        '  li::before { content: "\\2022"; position: absolute; left: 1pt; color: #555; }\n'
        "</style>\n<h2>Experience</h2>\n"
    )

    def test_it_fixes_every_known_hazard(self):
        from jobsearch.tailor import repair_ats_html

        out, notes = repair_ats_html(self.OLD)
        assert "small-caps" not in out
        assert "letter-spacing: 0" in out
        assert "position: absolute" not in out
        assert "<h2>Professional Experience</h2>" in out
        assert len(notes) == 5

    def test_it_is_a_no_op_on_an_already_hardened_cv(self, cfg):
        """The base CV must pass through untouched."""
        from jobsearch.tailor import repair_ats_html

        html = cfg.base_cv.read_text(encoding="utf-8")
        out, notes = repair_ats_html(html)
        assert out == html
        assert notes == []

    def test_repairing_twice_changes_nothing_further(self):
        from jobsearch.tailor import repair_ats_html

        once, _ = repair_ats_html(self.OLD)
        twice, notes = repair_ats_html(once)
        assert twice == once
        assert notes == []

    def test_harden_reports_repairs_before_warnings(self):
        """A hazard that was repaired must not also be warned about."""
        from jobsearch.tailor import harden_html

        _, notes = harden_html(self.OLD, [])
        assert any(n.startswith("repaired:") for n in notes)
        assert not any("still contains an ATS hazard" in n for n in notes)


class TestNormaliseDashes:
    """House style: no em or en dashes. Asking the model does not reliably
    stop it, so the rule is applied mechanically as well.
    """

    def test_a_spaced_em_dash_becomes_a_spaced_hyphen(self):
        from jobsearch.tailor import normalise_dashes

        assert normalise_dashes("led the team — and shipped") == "led the team - and shipped"

    def test_a_closed_up_em_dash_stays_closed_up(self):
        from jobsearch.tailor import normalise_dashes

        assert normalise_dashes("New York—based") == "New York-based"

    def test_en_dashes_in_ranges(self):
        from jobsearch.tailor import normalise_dashes

        assert normalise_dashes("2020–2022") == "2020-2022"

    def test_html_entities(self):
        from jobsearch.tailor import normalise_dashes

        assert normalise_dashes("a &mdash; b &ndash; c") == "a - b - c"

    def test_existing_hyphens_are_untouched(self):
        from jobsearch.tailor import normalise_dashes

        assert normalise_dashes("cross-functional, well-known") == "cross-functional, well-known"

    def test_the_middot_separator_survives(self):
        """The CV template uses · between education fields."""
        from jobsearch.tailor import normalise_dashes

        assert "·" in normalise_dashes("Stellenbosch · MSc · 2019")

    def test_it_is_idempotent(self):
        from jobsearch.tailor import normalise_dashes

        once = normalise_dashes("a — b")
        assert normalise_dashes(once) == once

    def test_harden_html_applies_it_and_says_so(self):
        from jobsearch.tailor import harden_html

        html, notes = harden_html("<p>led the team — and shipped</p>", [])
        assert "—" not in html
        assert any("em and en dashes" in n for n in notes)


class TestFilenamesAreUniquePerRole:
    """Company-only names collide: one employer can have a dozen roles open,
    and the same title in several cities. The second write would overwrite the
    first and leave both job rows pointing at the survivor.
    """

    def _posting(self, title, location, job_id):
        from jobsearch.models import JobPosting

        return JobPosting(
            company="Databricks", title=title, url=f"https://x.test/{job_id}",
            location=location, job_id=job_id,
        )

    def test_same_title_different_cities_do_not_collide(self):
        from jobsearch.tailor import output_stem

        a = self._posting("Manager, Forward Deployed Engineering", "Amsterdam", "db-manager-4597")
        b = self._posting("Manager, Forward Deployed Engineering", "Singapore", "db-manager-87f6")
        assert output_stem(a) != output_stem(b)

    def test_different_titles_at_one_company_do_not_collide(self):
        from jobsearch.tailor import output_stem

        a = self._posting("Manager, Forward Deployed Engineering", "Amsterdam", "db-a-1111")
        b = self._posting("Senior Manager, Forward Deployed Engineering", "Amsterdam", "db-b-2222")
        assert output_stem(a) != output_stem(b)

    def test_letters_and_cvs_use_the_same_discriminator(self, cfg):
        from jobsearch.letter import letter_path_for
        from jobsearch.tailor import role_slug

        posting = self._posting("Manager, FDE", "Amsterdam", "db-manager-4597")
        assert role_slug(posting) in letter_path_for(cfg, posting).name

    def test_the_name_stays_readable(self):
        from jobsearch.tailor import output_stem

        stem = output_stem(self._posting("Manager, Forward Deployed Engineering", "Amsterdam", "db-x-4597"))
        assert stem.startswith("DJ_Human_CV_Databricks")
        assert stem.endswith("4597")
        assert len(stem) <= 82

    def test_it_is_deterministic(self):
        from jobsearch.tailor import output_stem

        posting = self._posting("Manager, FDE", "Amsterdam", "db-manager-4597")
        assert output_stem(posting) == output_stem(posting)

    def test_a_posting_without_a_job_id_still_gets_a_unique_stem(self):
        """`add` builds the id on save; a stem may be needed before that."""
        from jobsearch.models import JobPosting
        from jobsearch.tailor import output_stem

        a = JobPosting(company="Databricks", title="Manager, FDE", url="https://x.test/1")
        b = JobPosting(company="Databricks", title="Manager, FDE", url="https://x.test/2")
        assert output_stem(a) != output_stem(b)

    def test_punctuation_is_stripped_from_the_filename(self):
        from jobsearch.models import JobPosting
        from jobsearch.tailor import output_stem

        stem = output_stem(
            JobPosting(company="Abacus.AI", title="Director, Data Science", url="", job_id="x-1234")
        )
        assert "." not in stem and "," not in stem


class TestOverlongBullets:
    """A bullet carrying four achievements hides all four."""

    def test_a_long_experience_bullet_is_reported(self):
        from jobsearch.tailor import overlong_bullets

        html = "<ul><li>" + " ".join(["word"] * 50) + "</li></ul>"
        assert overlong_bullets(html)[0][0] == 50

    def test_a_short_one_is_not(self):
        from jobsearch.tailor import overlong_bullets

        assert overlong_bullets("<ul><li>Cut error 50%, a $6M saving</li></ul>") == []

    def test_skills_and_publication_lists_are_exempt(self):
        """They are delimited lists, not prose; splitting them would be wrong."""
        from jobsearch.tailor import overlong_bullets

        long_line = " | ".join(["keyword"] * 40)
        assert overlong_bullets(f'<ul class="skills"><li>{long_line}</li></ul>') == []
        assert overlong_bullets(f'<ul class="pub"><li>{long_line}</li></ul>') == []

    def test_markup_does_not_count_towards_the_word_budget(self):
        from jobsearch.tailor import overlong_bullets

        html = "<ul><li>" + " ".join(f"<span>{w}</span>" for w in ["word"] * 20) + "</li></ul>"
        assert overlong_bullets(html) == []

    def test_hardening_reports_density(self):
        from jobsearch.tailor import harden_html

        html = "<ul><li>" + " ".join(["word"] * 60) + "</li></ul>"
        _, notes = harden_html(html, [])
        assert any("dense:" in n for n in notes)


class TestAdversarialReview:
    """A sceptical reader, not a proofreader. Checks defensibility, not truth."""

    def _client(self, payload):
        from types import SimpleNamespace

        return SimpleNamespace(
            structured=lambda **kw: payload, dry_run=False,
            last_usage=SimpleNamespace(cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )

    def test_findings_are_ordered_by_severity(self, cfg):
        from jobsearch.models import JobPosting
        from jobsearch.tailor import adversarial_review

        payload = {"critiques": [
            {"issue": "c", "severity": "minor", "quote": "", "why": "", "fix": ""},
            {"issue": "a", "severity": "blocking", "quote": "", "why": "", "fix": ""},
            {"issue": "b", "severity": "major", "quote": "", "why": "", "fix": ""},
        ], "overall": ""}
        posting = JobPosting(company="X", title="Y", url="", job_id="j")
        result = adversarial_review("<p>cv</p>", posting, cfg, self._client(payload))
        assert [c.issue for c in result] == ["a", "b", "c"]

    def test_an_empty_list_is_a_valid_answer(self, cfg):
        from jobsearch.models import JobPosting
        from jobsearch.tailor import adversarial_review

        posting = JobPosting(company="X", title="Y", url="", job_id="j")
        assert adversarial_review("<p>cv</p>", posting, cfg, self._client({"critiques": [], "overall": "fine"})) == []

    def test_blocking_findings_are_exposed_separately(self):
        from jobsearch.models import Critique, TailorResult

        result = TailorResult(job_id="j", html_path="x")
        result.critiques = [Critique(issue="a", severity="blocking"), Critique(issue="b", severity="minor")]
        assert [c.issue for c in result.blocking] == ["a"]

    def test_the_instructions_separate_defensibility_from_truth(self):
        """Grounding checks whether claims are true; this checks whether they
        survive a sceptical reader. Conflating them wastes both passes."""
        import re

        from jobsearch.tailor import ADVERSARIAL_INSTRUCTIONS

        flat = re.sub(r"\s+", " ", ADVERSARIAL_INSTRUCTIONS.lower())
        assert "do not check whether claims are true" in flat
        assert "trivially true" in flat
        assert "an empty list is a valid" in flat


class TestCritiqueFeedbackLoop:
    """Re-tailoring without the previous criticism just re-rolls the dice."""

    def _posting(self):
        from jobsearch.models import JobPosting

        return JobPosting(company="Acme", title="Head of AI", url="", job_id="j")

    def test_prior_findings_reach_the_prompt(self):
        from jobsearch.models import Critique
        from jobsearch.tailor import _tailor_prompt

        prompt = _tailor_prompt(self._posting(), [
            Critique(issue="Trivially true", severity="major",
                     quote="beat a frontier model", fix="state the condition")
        ])
        assert "previous_version_was_criticised_for" in prompt
        assert "Trivially true" in prompt
        assert "beat a frontier model" in prompt
        assert "state the condition" in prompt

    def test_no_findings_means_no_block(self):
        from jobsearch.tailor import _tailor_prompt

        assert "previous_version" not in _tailor_prompt(self._posting())

    def test_the_prompt_forbids_inventing_a_fix(self):
        """The obvious failure mode: answering a critique with a new claim."""
        from jobsearch.models import Critique
        from jobsearch.tailor import _tailor_prompt

        prompt = _tailor_prompt(self._posting(), [Critique(issue="x", severity="minor")])
        assert "Do NOT invent anything" in prompt
        assert "cut the" in prompt and "claim instead" in prompt

    def test_stored_findings_are_loaded_back(self, cfg):
        import json

        from jobsearch.tui import load_critiques

        row = {"critique_json": json.dumps([
            {"issue": "i", "severity": "blocking", "quote": "q", "why": "w", "fix": "f"}
        ])}
        loaded = load_critiques(row)
        assert len(loaded) == 1 and loaded[0].severity == "blocking"

    def test_a_role_with_no_prior_review_loads_nothing(self):
        from jobsearch.tui import load_critiques

        assert load_critiques({}) == []
        assert load_critiques({"critique_json": None}) == []


# --- a blocking critique must be acted on, not merely reported --------------


class _StubClaude:
    """Returns a fixed CV, and critiques that clear after the first rewrite."""

    dry_run = False

    def __init__(self, blocking_passes: int = 1):
        self.blocking_passes = blocking_passes
        self.calls = 0
        self.prompts: list[str] = []

        class _U:
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0

        self.last_usage = _U()

    def stream_text(self, *, user_content="", **kw):
        self.calls += 1
        self.prompts.append(user_content)
        return "<html><body><h2>Professional Experience</h2><p>x</p></body></html>"

    def structured(self, *, stage="", **kw):
        if stage == "ground":
            return {"claims": [], "verdict": "clean", "summary": ""}
        if self.calls <= self.blocking_passes:
            return {"critiques": [{"issue": "Six years is really four",
                                   "severity": "blocking", "quote": "Six years",
                                   "why": "arithmetic", "fix": "say four"}]}
        return {"critiques": []}


def test_a_blocking_critique_triggers_a_second_pass(cfg, posting):
    from jobsearch.tailor import tailor_cv

    claude = _StubClaude(blocking_passes=1)
    result = tailor_cv(posting, cfg, claude, render=False, verify_claims=False)
    assert claude.calls == 2, "a blocking critique must be re-tailored, not just reported"
    assert result.passes == 2
    assert not result.blocking, "the second pass cleared it"


def test_the_second_pass_is_told_what_was_wrong(cfg, posting):
    from jobsearch.tailor import tailor_cv

    claude = _StubClaude(blocking_passes=1)
    tailor_cv(posting, cfg, claude, render=False, verify_claims=False)
    assert "Six years is really four" in claude.prompts[1]


def test_it_gives_up_rather_than_looping(cfg, posting):
    """A blocking issue surviving two passes is a judgement for a person."""
    from jobsearch.tailor import tailor_cv

    claude = _StubClaude(blocking_passes=99)
    result = tailor_cv(posting, cfg, claude, render=False, verify_claims=False)
    assert claude.calls == 2
    assert result.passes == 2
    assert result.blocking, "still reported, so the user knows to look"
