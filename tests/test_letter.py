"""Cover letters: short, grounded, and named for what they are."""

from __future__ import annotations

import pytest

from jobsearch.letter import LetterResult, letter_path_for, strip_stray_markdown, write_letter
from jobsearch.models import Claim, JobPosting


def posting(company="Databricks", title="Manager, Forward Deployed Engineering"):
    return JobPosting(
        company=company, title=title, url="https://example.com/j/1",
        location="Amsterdam, Netherlands", description="Lead a team of FDEs.",
        job_id="job-1",
    )


class TestNaming:
    def test_a_letter_is_not_named_like_a_cv(self, cfg):
        """DJ_Human_CV_Databricks.txt gets attached to the wrong form field."""
        path = letter_path_for(cfg, posting())
        assert path.name.startswith("Cover_Letter_Databricks")
        assert path.name.endswith(".txt")
        assert "CV" not in path.name

    def test_punctuation_in_a_company_name_is_stripped(self, cfg):
        name = letter_path_for(cfg, posting("Abacus.AI")).name
        assert name.startswith("Cover_Letter_AbacusAI")
        assert "." not in name[:-4], "only the extension may contain a dot"

    def test_an_empty_company_still_yields_a_path(self, cfg):
        assert letter_path_for(cfg, posting("")).name.startswith("Cover_Letter_Role")

    def test_letters_live_beside_the_cvs_not_among_them(self, cfg):
        assert letter_path_for(cfg, posting()).parent.name == "letters"


class TestStripStrayMarkdown:
    def test_headings_are_removed(self):
        assert "## Intro" not in strip_stray_markdown("## Intro\nHello,\n")

    def test_bold_and_italics_are_unwrapped(self):
        assert strip_stray_markdown("I **led** a *team*.").strip() == "I led a team."

    def test_a_code_fence_is_removed(self):
        assert strip_stray_markdown("```\nHello,\n```").strip() == "Hello,"

    def test_plain_text_is_left_alone(self):
        text = "Hello,\n\nA sentence.\n\nBest regards,\nDJ\n"
        assert strip_stray_markdown(text) == text

    def test_an_asterisk_inside_a_word_survives(self):
        """MLflow* and Spark™ style markers must not be mangled."""
        assert "2*3" in strip_stray_markdown("Maths: 2*3 = 6")


class TestWriteLetter:
    def test_dry_run_writes_nothing_and_needs_no_credentials(self, cfg):
        from jobsearch.claude import ClaudeClient

        client = ClaudeClient(dry_run=True)
        result = write_letter(posting(), cfg, client)
        assert not letter_path_for(cfg, posting()).exists()
        assert "dry-run" in result.text

    def test_it_writes_the_file_and_counts_words(self, cfg, monkeypatch):
        from jobsearch.claude import ClaudeClient
        import jobsearch.letter as letter_mod

        client = ClaudeClient()
        monkeypatch.setattr(
            type(client), "stream_text",
            lambda self, **kw: "Hello,\n\nOne two three four five.\n\nBest,\nDJ\n",
        )
        monkeypatch.setattr(letter_mod, "ground_claims", lambda *a, **k: [])
        result = write_letter(posting(), cfg, client)
        assert letter_path_for(cfg, posting()).read_text(encoding="utf-8").startswith("Hello,")
        assert result.word_count == 8  # Hello, One two three four five. Best, DJ

    def test_ungrounded_claims_are_surfaced(self, cfg, monkeypatch):
        from jobsearch.claude import ClaudeClient
        import jobsearch.letter as letter_mod

        client = ClaudeClient()
        monkeypatch.setattr(type(client), "stream_text", lambda self, **kw: "Hello,\n")
        monkeypatch.setattr(
            letter_mod, "ground_claims",
            lambda *a, **k: [
                Claim(text="grounded thing", grounded=True),
                Claim(text="20% travel suits me", grounded=False),
            ],
        )
        result = write_letter(posting(), cfg, client)
        assert [c.text for c in result.ungrounded] == ["20% travel suits me"]

    def test_an_extra_instruction_reaches_the_prompt(self, cfg, monkeypatch):
        from jobsearch.claude import ClaudeClient
        import jobsearch.letter as letter_mod

        seen = {}
        client = ClaudeClient()
        monkeypatch.setattr(
            type(client), "stream_text",
            lambda self, **kw: seen.update(kw) or "Hello,\n",
        )
        monkeypatch.setattr(letter_mod, "ground_claims", lambda *a, **k: [])
        write_letter(posting(), cfg, client, instruction="lead on the FDE angle")
        assert "lead on the FDE angle" in seen["user_content"]


class TestLetterHouseStyle:
    def test_em_dashes_never_reach_the_letter(self, cfg, monkeypatch):
        from jobsearch.claude import ClaudeClient
        import jobsearch.letter as letter_mod

        client = ClaudeClient()
        monkeypatch.setattr(
            type(client), "stream_text",
            lambda self, **kw: "Hello,\n\nI led a team — and shipped.\n\nBest,\nDJ\n",
        )
        monkeypatch.setattr(letter_mod, "ground_claims", lambda *a, **k: [])
        result = write_letter(posting(), cfg, client)
        assert "—" not in result.text
        assert "team - and shipped" in result.text

    def test_the_instructions_forbid_them_too(self):
        """Post-processing is the backstop; the prompt is the first line."""
        import re

        from jobsearch.letter import LETTER_INSTRUCTIONS

        # Normalise whitespace: the rule is line-wrapped in the prompt, so a
        # plain substring check fails on a wrap rather than on a missing rule.
        flat = re.sub(r"\s+", " ", LETTER_INSTRUCTIONS.lower())
        assert "em dash" in flat


class TestLetterTone:
    """Confidence and arrogance differ in specific, nameable ways."""

    def test_the_instructions_name_each_tipping_point(self):
        import re

        from jobsearch.letter import LETTER_INSTRUCTIONS

        flat = re.sub(r"\s+", " ", LETTER_INSTRUCTIONS.lower())
        for rule in (
            "do not tell the reader what their own job",   # no lecturing them
            "no slogans",                                   # no "not demos"
            "at most three numbers",                        # no achievement stacking
            'say "we" where the work was shared',           # credit the team
            "one genuine question",                         # curiosity, not just supply
            "modest next step",                             # no presuming their process
        ):
            assert rule in flat, f"the tone guidance lost: {rule}"
