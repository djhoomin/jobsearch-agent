"""Config loading, outreach drafting, dry-run behaviour, and the CLI surface."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from jobsearch.cli import build_parser, main
from jobsearch.config import BoardRef, Config, ConfigError, Weights, load_config
from jobsearch.models import JobPosting
from jobsearch.outreach import draft_outreach, format_draft, linkedin_search_url

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestConfig:
    def test_example_config_parses(self):
        """The committed template must stay loadable: it is what a new user copies."""
        raw = tomllib.loads((REPO_ROOT / "config.example.toml").read_text(encoding="utf-8"))
        cfg = Config(root=REPO_ROOT, raw=raw)
        assert cfg.weights.as_dict()["role_fit"] == 0.25
        assert len(cfg.boards) >= 1

    def test_example_config_is_placeholders_only(self):
        """Regression guard: the template is public; config.local.toml is not.

        Asserts the placeholders are intact rather than blocklisting specific
        personal strings - a blocklist would have to contain the very values it
        protects.
        """
        import re

        text = (REPO_ROOT / "config.example.toml").read_text(encoding="utf-8")
        assert 'name = "Your Name"' in text
        assert 'current_company = "Your Current Employer"' in text
        # Every email address in the template must be a reserved example domain.
        for address in re.findall(r"[\w.+-]+@[\w.-]+", text):
            assert address.endswith("@example.com"), f"real address in template: {address}"

    def test_loaded_config_records_its_own_source_path(self, tmp_path):
        """`doctor` reports this path; it must be the file actually read."""
        from jobsearch.config import load_config

        src = tmp_path / "config.local.toml"
        src.write_text((REPO_ROOT / "config.example.toml").read_text(encoding="utf-8"),
                       encoding="utf-8")
        assert load_config(src).source == src

    def test_paths_resolve_against_the_config_file(self, cfg, tmp_path):
        assert cfg.dossier == tmp_path / "dossier.md"
        assert cfg.dossier.is_file()

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ConfigError, match="must sum to 1.0"):
            Weights(buyer=0.5, role_fit=0.5, company=0.5, domain=0.5, talent=0.5)

    def test_board_lookup_is_case_insensitive(self, cfg):
        assert cfg.board_for("example ashby co") is not None
        assert cfg.board_for("EXAMPLE ASHBY CO").company == "Example Ashby Co"

    def test_unknown_company_returns_none(self, cfg):
        assert cfg.board_for("Nonexistent Corp") is None

    def test_sponsor_state_normalises(self):
        assert BoardRef("A", "ashby", "a", ind_sponsor=True).sponsor_state == "yes"
        assert BoardRef("A", "ashby", "a", ind_sponsor=False).sponsor_state == "no"
        assert BoardRef("A", "ashby", "a").sponsor_state == "unknown"

    def test_missing_source_document_is_a_clear_error(self, cfg):
        cfg.dossier = cfg.root / "gone.md"
        with pytest.raises(ConfigError, match="does not exist"):
            cfg.read_dossier()

    def test_missing_config_file_raises(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(tmp_path / "nope.toml")


class TestOutreach:
    def test_search_url_is_built_not_fetched(self):
        url = linkedin_search_url("Weaviate", "VP Engineering")
        assert url.startswith("https://www.linkedin.com/search/results/people/")
        assert "Weaviate" in url and "VP%20Engineering" in url

    @pytest.fixture
    def wired(self, fake_claude):
        fake_claude.structured_responses["outreach"] = {
            "contacts": [
                {"title": "CPTO", "name": "", "rationale": "no AI exec exists, so the CPTO is the buyer"}
            ],
            "linkedin_connection_note": "n" * 400,
            "linkedin_message": "m" * 400,
            "email_subject": "Your agent bet needs an AI org",
            "email_body": "Body text.",
        }
        return fake_claude

    def test_caps_are_enforced(self, cfg, wired, posting):
        draft = draft_outreach(posting, cfg, wired)
        assert len(draft.linkedin_connection_note) <= 200
        assert len(draft.linkedin_message) <= 300

    def test_each_contact_gets_a_search_url(self, cfg, wired, posting):
        draft = draft_outreach(posting, cfg, wired)
        assert draft.contacts[0].linkedin_search_url.startswith("https://www.linkedin.com/search")

    def test_target_company_notes_join_the_cached_prefix(self, cfg, wired, posting):
        draft_outreach(posting, cfg, wired)
        labels = [label for label, _ in wired.calls[0]["stable_context"]]
        assert labels[-1] == "target_company_notes"

    def test_rendered_draft_says_nothing_was_sent(self, cfg, wired, posting):
        rendered = format_draft(draft_outreach(posting, cfg, wired), posting)
        assert "Nothing has been sent" in rendered
        assert "LINKEDIN CONNECTION NOTE" in rendered


class TestCLI:
    def test_every_subcommand_has_help(self, capsys):
        parser = build_parser()
        commands = parser._subparsers._group_actions[0].choices  # noqa: SLF001
        assert set(commands) >= {
            "discover", "score", "tailor", "verify", "outreach",
            "track", "status", "export", "sync", "run", "doctor",
        }
        for name, sub in commands.items():
            assert sub.format_help(), f"{name} has no help"

    def test_verify_runs_standalone_on_a_pdf(self, cfg, tmp_path, capsys):
        pytest.importorskip("pypdf")
        from jobsearch.render import RenderError, html_to_pdf

        try:
            pdf = html_to_pdf(
                Path(__file__).parent / "fixtures" / "good_cv.html", tmp_path / "cv.pdf"
            )
        except RenderError:
            pytest.skip("Chrome unavailable")
        code = main(["--config", str(cfg.root / "config.local.toml"), "verify", str(pdf)])
        assert code == 0
        assert "ATS VERIFICATION: PASS" in capsys.readouterr().out

    def test_verify_exits_nonzero_on_failure(self, cfg, tmp_path):
        from jobsearch.render import RenderError, html_to_pdf

        try:
            pdf = html_to_pdf(
                Path(__file__).parent / "fixtures" / "bad_cv.html", tmp_path / "bad.pdf"
            )
        except RenderError:
            pytest.skip("Chrome unavailable")
        assert main(["--config", str(cfg.root / "config.local.toml"), "verify", str(pdf)]) == 1


class TestDryRun:
    """--dry-run must make no API call and write nothing."""

    def test_discover_lists_boards_without_fetching(self, cfg, capsys, monkeypatch):
        import jobsearch.cli as cli

        called = []
        monkeypatch.setattr(
            "jobsearch.discover.discover", lambda *a, **k: called.append(1)
        )
        args = build_parser().parse_args(["--dry-run", "discover", "--tier", "1"])
        cli.cmd_discover(cfg, args)
        out = capsys.readouterr().out
        assert "[dry-run]" in out
        assert "boards-api.greenhouse.io" in out
        assert called == [], "dry-run must not hit the network"

    def test_dry_run_creates_no_database(self, cfg, capsys):
        import jobsearch.cli as cli

        args = build_parser().parse_args(
            ["--dry-run", "add", "--company", "X", "--title", "Head of AI"]
        )
        cli.cmd_add(cfg, args)
        assert not cfg.db_path.exists()

    def test_dry_run_client_makes_no_call(self, cfg):
        from jobsearch.claude import ClaudeClient
        from jobsearch.scoring import score_posting

        client = ClaudeClient.from_config(cfg, dry_run=True)
        posting = JobPosting(
            company="X", title="Head of AI", url="https://x/1", location="Amsterdam"
        )
        report = score_posting(posting, cfg, client)
        # The dry-run placeholder scores everything 3 and never builds a client.
        assert report.weighted == 3.0
        assert client._client is None  # noqa: SLF001 - asserting no client was built
