"""Tests for the guided first-run wizard.

The wizard is the one command that runs before any configuration exists, so
these tests drive it with a scripted reader instead of a real terminal.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from jobsearch.config import EXAMPLE_CONFIG_NAME, LOCAL_CONFIG_NAME, Config
from jobsearch.setup_wizard import (
    GUESSES,
    SetupAborted,
    ask,
    ask_yes_no,
    guess_document,
    render_config,
    run_setup,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (REPO_ROOT / EXAMPLE_CONFIG_NAME).read_text(encoding="utf-8")


def scripted(answers: list[str]):
    """A reader that replays canned answers, then raises EOF like a closed stdin."""
    queue = list(answers)

    def reader(_prompt: str) -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    return reader


class TestPrompting:
    def test_empty_input_takes_the_default(self):
        assert ask("q", "fallback", reader=scripted([""])) == "fallback"

    def test_typed_input_wins_over_the_default(self):
        assert ask("q", "fallback", reader=scripted(["typed"])) == "typed"

    def test_it_reprompts_until_validation_passes(self):
        reader = scripted(["nope", "still nope", "42"])
        assert ask("n", "", reader=reader, validate=lambda v: None if v.isdigit() else "digits only") == "42"

    def test_closed_stdin_aborts_rather_than_looping(self):
        with pytest.raises(SetupAborted):
            ask("q", "", reader=scripted([]))

    @pytest.mark.parametrize(
        "typed,default,expected",
        [("y", False, True), ("yes", False, True), ("n", True, False), ("", True, True), ("", False, False)],
    )
    def test_yes_no(self, typed, default, expected):
        assert ask_yes_no("q", default, reader=scripted([typed])) is expected


class TestDocumentDiscovery:
    def test_it_finds_a_matching_document_in_the_parent_directory(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (tmp_path / "my-career-dossier.md").write_text("x", encoding="utf-8")
        guess = next(g for g in GUESSES if g.key == "dossier")
        assert guess_document(repo, guess) == "../my-career-dossier.md"

    def test_it_falls_back_when_nothing_matches(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        guess = next(g for g in GUESSES if g.key == "dossier")
        assert guess_document(repo, guess) == guess.fallback

    def test_the_most_recent_match_wins(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        older = tmp_path / "old-dossier.md"
        newer = tmp_path / "new-dossier.md"
        older.write_text("x", encoding="utf-8")
        newer.write_text("x", encoding="utf-8")
        import os
        os.utime(older, (1_000_000, 1_000_000))
        guess = next(g for g in GUESSES if g.key == "dossier")
        assert guess_document(repo, guess) == "../new-dossier.md"


ANSWERS = {
    "dossier": "../d.md",
    "base_cv": "../cv.html",
    "search_strategy": "../s.md",
    "target_companies": "../t.md",
    "tracker_xlsx": "../t.xlsx",
    "name": "Jane Q. Testcandidate",
    "email": "jane@example.com",
    "location": "Testville",
    "linkedin": "https://www.linkedin.com/in/jane/",
    "current_title": "Director of Engineering",
    "current_company": "Northwind Labs",
    "comp_floor_eur": "175000",
    "non_compete_waiver_signed": "true",
    "chrome_binary": "/usr/bin/chromium",
    "user_agent": "jobsearch-agent/0.1 (+mailto:jane@example.com)",
}


class TestRenderConfig:
    def test_the_result_is_valid_toml_with_the_answers_in_it(self):
        raw = tomllib.loads(render_config(TEMPLATE, ANSWERS))
        assert raw["candidate"]["name"] == "Jane Q. Testcandidate"
        assert raw["constraints"]["comp_floor_eur"] == 175000
        assert raw["constraints"]["non_compete_waiver_signed"] is True
        assert raw["paths"]["dossier"] == "../d.md"
        assert raw["render"]["chrome_binary"] == "/usr/bin/chromium"

    def test_it_preserves_the_templates_comments(self):
        out = render_config(TEMPLATE, ANSWERS)
        assert "# ---" in out
        assert out.count("#") > 20, "explanatory comments should survive into the user's config"

    def test_it_still_loads_as_a_Config(self, tmp_path):
        raw = tomllib.loads(render_config(TEMPLATE, ANSWERS))
        cfg = Config(root=tmp_path, raw=raw)
        assert cfg.weights.as_dict()["role_fit"] == 0.25

    def test_quotes_in_an_answer_do_not_break_the_toml(self):
        answers = dict(ANSWERS, current_company='North"wind')
        raw = tomllib.loads(render_config(TEMPLATE, answers))
        assert raw["candidate"]["current_company"] == 'North"wind'

    def test_only_the_first_occurrence_of_a_key_is_replaced(self):
        """`name` appears in [candidate] and again in each [[discover.boards]]."""
        out = render_config(TEMPLATE, ANSWERS)
        raw = tomllib.loads(out)
        assert raw["candidate"]["name"] == "Jane Q. Testcandidate"
        # Board company names must be untouched by the [candidate] substitution.
        assert [b["company"] for b in raw["discover"]["boards"]] == [
            "Example Ashby Co",
            "Example Greenhouse Co",
            "Example Lever Co",
        ]


class TestRunSetup:
    def _repo(self, tmp_path: Path) -> Path:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / EXAMPLE_CONFIG_NAME).write_text(TEMPLATE, encoding="utf-8")
        return repo

    def _answers(self) -> list[str]:
        # 5 documents, 6 candidate fields, comp floor, waiver y/n, chrome, boards y/n
        return ["", "", "", "", "", "Jane Q. Testcandidate", "jane@example.com",
                "Testville", "https://linkedin.test/jane", "Director of Engineering",
                "Northwind Labs", "175000", "y", "/usr/bin/chromium", "n"]

    def test_it_writes_a_loadable_config(self, tmp_path, capsys):
        repo = self._repo(tmp_path)
        assert run_setup(repo, reader=scripted(self._answers())) == 0
        written = repo / LOCAL_CONFIG_NAME
        assert written.is_file()
        raw = tomllib.loads(written.read_text(encoding="utf-8"))
        assert raw["candidate"]["email"] == "jane@example.com"
        assert raw["constraints"]["comp_floor_eur"] == 175000
        assert raw["constraints"]["non_compete_waiver_signed"] is True

    def test_it_declines_to_overwrite_without_confirmation(self, tmp_path):
        repo = self._repo(tmp_path)
        existing = repo / LOCAL_CONFIG_NAME
        existing.write_text("# do not clobber me\n", encoding="utf-8")
        assert run_setup(repo, reader=scripted(["n"])) == 0
        assert existing.read_text(encoding="utf-8") == "# do not clobber me\n"

    def test_force_skips_the_overwrite_prompt(self, tmp_path):
        repo = self._repo(tmp_path)
        (repo / LOCAL_CONFIG_NAME).write_text("# old\n", encoding="utf-8")
        assert run_setup(repo, force=True, reader=scripted(self._answers())) == 0
        assert "# old" not in (repo / LOCAL_CONFIG_NAME).read_text(encoding="utf-8")

    def test_missing_template_is_an_error_not_a_traceback(self, tmp_path, capsys):
        repo = tmp_path / "empty"
        repo.mkdir()
        assert run_setup(repo, reader=scripted([])) == 2
        assert "no config.example.toml" in capsys.readouterr().err

    def test_it_warns_about_paths_that_do_not_exist(self, tmp_path, capsys):
        repo = self._repo(tmp_path)
        run_setup(repo, reader=scripted(self._answers()))
        assert "do not exist yet" in capsys.readouterr().out
