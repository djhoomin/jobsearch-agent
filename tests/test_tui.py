"""Tests for the terminal UI.

The pure rendering helpers and the stage dispatcher are tested directly. The
app itself is exercised through Textual's headless ``run_test`` harness, so no
real terminal is involved.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from jobsearch.models import JobPosting, Status
from jobsearch.tracker import Tracker
from jobsearch.tui import (
    build_app,
    is_eliminated,
    run_stage_blocking,
    score_cell,
    status_glyph,
    truncate,
)

textual = pytest.importorskip("textual", reason="TUI extra not installed")


def seed(cfg, **overrides) -> str:
    """Put one posting in the tracker and return its job id."""
    posting = JobPosting(
        company=overrides.pop("company", "Northwind"),
        title=overrides.pop("title", "Director of Engineering, AI"),
        url="https://example.com/job/1",
        location=overrides.pop("location", "Amsterdam, Netherlands"),
        description=overrides.pop("description", "We need someone to run the AI team."),
        **overrides,
    )
    with Tracker.from_config(cfg) as tracker:
        return tracker.upsert_job(posting)


class TestRenderHelpers:
    def test_known_status_has_a_glyph(self):
        assert status_glyph(Status.APPLIED.value) == "●"

    def test_unknown_status_falls_back(self):
        assert status_glyph("Something New") == "·"

    def test_truncate_leaves_short_text_alone(self):
        assert truncate("short", 20) == "short"

    def test_truncate_marks_elision(self):
        assert truncate("abcdefghij", 5) == "abcd…"

    def test_truncate_handles_none(self):
        assert truncate(None, 5) == ""

    @pytest.mark.parametrize(
        "row,expected",
        [
            ({"constraints_ok": 0, "score_weighted": None}, "elim"),
            ({"constraints_ok": 1, "score_weighted": 4.153}, "4.15"),
            ({"constraints_ok": 1, "score_weighted": None}, "-"),
            ({}, "-"),
        ],
    )
    def test_score_cell(self, row, expected):
        assert score_cell(row) == expected

    def test_unscored_job_is_not_treated_as_eliminated(self):
        """A missing constraints_ok means 'not scored yet', not 'failed'."""
        assert is_eliminated({}) is False
        assert is_eliminated({"constraints_ok": None}) is False
        assert is_eliminated({"constraints_ok": 0}) is True


class TestStageDispatch:
    def test_verify_without_a_cv_explains_itself(self, cfg):
        job_id = seed(cfg)
        assert "tailor first" in run_stage_blocking(cfg, "verify", job_id)

    def test_unknown_stage_raises(self, cfg):
        job_id = seed(cfg)
        with pytest.raises(ValueError, match="unknown stage"):
            run_stage_blocking(cfg, "nonsense", job_id)

    def test_a_missing_job_raises_rather_than_returning_a_string(self, cfg):
        seed(cfg)
        with pytest.raises(Exception):
            run_stage_blocking(cfg, "verify", "no-such-job")


class TestApp:
    """Headless mount tests: prove the layout composes and reads the tracker."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_it_mounts_and_lists_seeded_jobs(self, cfg):
        seed(cfg, company="Northwind")
        seed(cfg, company="Contoso AI", title="Head of AI")

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                from textual.widgets import DataTable

                table = app.query_one("#table", DataTable)
                assert table.row_count == 2
                companies = {str(app.rows[i]["company"]) for i in range(2)}
                assert companies == {"Northwind", "Contoso AI"}

        self._run(scenario())

    def test_empty_pipeline_shows_guidance_not_a_crash(self, cfg):
        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                assert "No roles yet" in app.detail_text

        self._run(scenario())

    def test_filtering_narrows_the_table(self, cfg):
        seed(cfg, company="Northwind")
        seed(cfg, company="Contoso AI", title="Head of AI")

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.filter_text = "contoso"
                app.action_refresh_rows()
                await pilot.pause()
                from textual.widgets import DataTable

                assert app.query_one("#table", DataTable).row_count == 1

        self._run(scenario())

    def test_a_stage_key_on_an_empty_pipeline_is_a_no_op(self, cfg):
        """Pressing a stage key with nothing selected must not raise."""

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.action_stage("score")
                await pilot.pause()
                assert app.busy is False

        self._run(scenario())

    def test_dry_run_is_announced(self, cfg):
        async def scenario():
            app = build_app(cfg, dry_run=True)
            async with app.run_test() as pilot:
                await pilot.pause()
                assert app.dry_run is True

        self._run(scenario())


class TestOutreachView:
    """Regression: drafted contacts and messages were generated but unreadable.

    The outreach stage reported "4 likely contact(s)" and then offered no way
    to see them, in the TUI or in `jobsearch show`.
    """

    def test_it_lists_contacts_and_drafts(self, cfg):
        from jobsearch.models import Contact, OutreachDraft
        from jobsearch.tui import outreach_detail_text

        job_id = seed(cfg)
        with Tracker.from_config(cfg) as tracker:
            # Contacts ride on the draft: save_outreach replaces whatever the
            # job already had, so saving them separately first would be undone.
            tracker.save_outreach(
                OutreachDraft(
                    job_id=job_id,
                    contacts=[Contact(title="VP Engineering", linkedin_search_url="https://x.test/1")],
                    linkedin_message="Saw the posting.",
                    email_subject="Subject line",
                    email_body="Hi [name],\n\nBody.",
                )
            )
        text = outreach_detail_text(cfg, job_id)
        assert "VP Engineering" in text
        assert "https://x.test/1" in text
        assert "Saw the posting." in text
        assert "Subject line" in text
        # The literal placeholder must survive: Textual markup would eat it.
        assert "[name]" in text

    def test_it_says_so_when_there_is_no_outreach_yet(self, cfg):
        from jobsearch.tui import outreach_detail_text

        assert "No contacts yet" in outreach_detail_text(cfg, seed(cfg))

    def test_pressing_enter_opens_the_role_screen(self, cfg):
        """Press the actual key.

        DataTable consumes Enter and emits RowSelected, so an App-level
        Binding("enter", ...) never fires. The previous version of this test
        called the action directly and passed while the key did nothing.
        """
        seed(cfg)

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                assert len(app.screen_stack) == 2, "enter must open the role screen"
                await pilot.press("escape")
                await pilot.pause()
                assert len(app.screen_stack) == 1

        asyncio.run(scenario())

    def test_role_detail_shows_scoring_and_drafts(self, cfg):
        from jobsearch.models import Contact, OutreachDraft, ScoreReport
        from jobsearch.tui import role_detail_text

        job_id = seed(cfg)
        with Tracker.from_config(cfg) as tracker:
            tracker.save_outreach(
                OutreachDraft(
                    job_id=job_id,
                    contacts=[Contact(title="VP Engineering", linkedin_search_url="https://x.test/1")],
                    linkedin_message="Saw the posting.",
                )
            )
        text = role_detail_text(cfg, job_id)
        assert "Northwind" in text
        assert "Not scored yet" in text
        assert "VP Engineering" in text
        assert "Saw the posting." in text


def test_save_outreach_replaces_contacts(cfg):
    """Documented sharp edge: a draft owns its contacts and overwrites them.

    Saving a draft whose `contacts` list is empty clears any contacts the job
    already had. Real callers attach contacts to the draft, which is why this
    is documented rather than guarded.
    """
    from jobsearch.models import Contact, OutreachDraft

    job_id = seed(cfg)
    with Tracker.from_config(cfg) as tracker:
        tracker.save_contacts(job_id, [Contact(title="VP Engineering")])
        assert len(tracker.contacts(job_id)) == 1
        tracker.save_outreach(OutreachDraft(job_id=job_id, linkedin_message="x"))
        assert tracker.contacts(job_id) == []


class TestLocationColumn:
    def test_a_dutch_location_passes(self, cfg):
        from jobsearch.tui import location_cell

        assert location_cell({"location": "Amsterdam, Netherlands"}, cfg).startswith("✓")

    def test_us_remote_is_not_rescued_by_the_word_remote(self, cfg):
        """The trap: 'Remote - United States' is a US role, not a remote-EU one."""
        from jobsearch.tui import location_cell

        cell = location_cell({"location": "Remote - United States"}, cfg)
        assert cell.startswith("✗")
        assert "remote" in cell

    def test_european_remote_passes(self, cfg):
        from jobsearch.tui import location_cell

        assert location_cell({"location": "Remote (Europe)"}, cfg).startswith("✓")

    def test_unclassifiable_location_is_a_question_not_a_guess(self, cfg):
        from jobsearch.tui import location_cell

        assert location_cell({"location": "CET, GMT or EST timezones"}, cfg).startswith("?")

    def test_missing_location_does_not_crash(self, cfg):
        from jobsearch.tui import location_cell

        assert location_cell({}, cfg).startswith("?")


class TestStatusPicker:
    def test_it_offers_only_legal_transitions(self):
        from jobsearch.tui import allowed_statuses

        options = allowed_statuses("Not started")
        assert "Applied" in options
        assert "Not started" not in options, "the current status is not a transition"
        assert "Offer" not in options, "cannot jump straight to Offer"

    def test_an_unknown_current_status_degrades_to_everything(self):
        from jobsearch.tui import allowed_statuses

        assert len(allowed_statuses("Nonsense")) > 0

    def test_marking_applied_persists(self, cfg):
        from jobsearch.tui import set_status_blocking

        job_id = seed(cfg)
        set_status_blocking(cfg, job_id, "Applied")
        with Tracker.from_config(cfg) as tracker:
            assert tracker.get_job(job_id)["status"] == "Applied"

    def test_an_illegal_transition_raises(self, cfg):
        from jobsearch.tui import set_status_blocking

        job_id = seed(cfg)
        with pytest.raises(Exception):
            set_status_blocking(cfg, job_id, "Offer")


class TestAddRole:
    def test_it_inserts_a_hand_entered_role(self, cfg):
        from jobsearch.tui import add_role_blocking

        job_id = add_role_blocking(
            cfg,
            company="Dash0",
            title="Director of Engineering, AI",
            url="https://example.com/j/1",
            location="Amsterdam / remote-first",
            description="Own Agent0.",
        )
        with Tracker.from_config(cfg) as tracker:
            row = tracker.get_job(job_id)
            assert row["company"] == "Dash0"
            assert row["source"] == "manual"
            assert row["status"] == "Not started"

    @pytest.mark.parametrize("company,title", [("", "Title"), ("Company", ""), ("  ", "Title")])
    def test_company_and_title_are_required(self, cfg, company, title):
        from jobsearch.tui import add_role_blocking

        with pytest.raises(ValueError, match="required"):
            add_role_blocking(cfg, company=company, title=title)


class TestNewScreens:
    def test_the_table_has_a_location_column(self, cfg):
        seed(cfg)

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                from textual.widgets import DataTable

                table = app.query_one("#table", DataTable)
                labels = [str(c.label) for c in table.columns.values()]
                assert "Location" in labels

        asyncio.run(scenario())

    def test_n_opens_the_add_role_screen(self, cfg):
        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.action_new_role()
                await pilot.pause()
                assert len(app.screen_stack) == 2

        asyncio.run(scenario())

    def test_a_opens_the_status_screen_when_a_row_is_selected(self, cfg):
        seed(cfg)

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.action_set_status()
                await pilot.pause()
                assert len(app.screen_stack) == 2

        asyncio.run(scenario())

    def test_a_on_an_empty_pipeline_does_not_open_a_screen(self, cfg):
        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.action_set_status()
                await pilot.pause()
                assert len(app.screen_stack) == 1

        asyncio.run(scenario())


class TestDismissAndDelete:
    def test_dismiss_marks_withdrawn(self, cfg):
        from jobsearch.tui import dismiss_blocking

        job_id = seed(cfg)
        assert dismiss_blocking(cfg, job_id) == "Withdrawn"
        with Tracker.from_config(cfg) as tracker:
            assert tracker.get_job(job_id)["status"] == "Withdrawn"

    def test_dismiss_survives_rediscovery(self, cfg):
        """The whole reason dismiss is a status and not a delete.

        A re-discovered posting refreshes its text but must not resurrect a
        role the user has already said no to.
        """
        from jobsearch.models import JobPosting
        from jobsearch.tui import dismiss_blocking

        job_id = seed(cfg)
        dismiss_blocking(cfg, job_id)
        with Tracker.from_config(cfg) as tracker:
            tracker.upsert_job(
                JobPosting(
                    company="Northwind",
                    title="Director of Engineering, AI",
                    url="https://example.com/job/1",
                    location="Amsterdam, Netherlands",
                    description="Refreshed text from a later sweep.",
                    job_id=job_id,
                )
            )
            row = tracker.get_job(job_id)
        assert row["status"] == "Withdrawn", "re-discovery must not undo a dismissal"
        assert "Refreshed text" in row["description"], "but the posting text should refresh"

    def test_dismiss_falls_back_when_withdrawn_is_not_a_legal_move(self, cfg):
        """From Rejected only Parked is reachable; dismiss must not blow up."""
        from jobsearch.tui import dismiss_blocking, set_status_blocking

        job_id = seed(cfg)
        set_status_blocking(cfg, job_id, "Applied")
        set_status_blocking(cfg, job_id, "Rejected")
        assert dismiss_blocking(cfg, job_id) == "Parked"

    def test_dismissing_twice_is_harmless(self, cfg):
        from jobsearch.tui import dismiss_blocking

        job_id = seed(cfg)
        dismiss_blocking(cfg, job_id)
        assert dismiss_blocking(cfg, job_id) == "already dismissed"

    def test_delete_removes_the_row(self, cfg):
        from jobsearch.tui import delete_blocking

        job_id = seed(cfg)
        assert delete_blocking(cfg, job_id) == "Northwind"
        with Tracker.from_config(cfg) as tracker:
            assert tracker.get_job(job_id) is None

    def test_dismissed_roles_are_hidden_until_toggled(self, cfg):
        from jobsearch.tui import dismiss_blocking

        keep = seed(cfg, company="Northwind")
        drop = seed(cfg, company="Irrelevant Corp", title="VP Sales")
        dismiss_blocking(cfg, drop)

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                from textual.widgets import DataTable

                table = app.query_one("#table", DataTable)
                assert table.row_count == 1
                app.action_toggle_hidden()
                await pilot.pause()
                assert app.query_one("#table", DataTable).row_count == 2

        asyncio.run(scenario())

    def test_delete_asks_before_destroying(self, cfg):
        """x opens a confirmation; it must not delete on the keypress alone."""
        job_id = seed(cfg)

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.action_delete_role()
                await pilot.pause()
                assert len(app.screen_stack) == 2
                with Tracker.from_config(cfg) as tracker:
                    assert tracker.get_job(job_id) is not None, "not deleted yet"

        asyncio.run(scenario())


class TestScan:
    """A sweep must never disturb a role you have already decided about."""

    def _fake_report(self, postings, monkeypatch):
        from types import SimpleNamespace

        import jobsearch.discover as discover_mod

        report = SimpleNamespace(
            postings=postings, boards_checked=3, raw_count=len(postings), errors=[]
        )
        monkeypatch.setattr(discover_mod, "discover", lambda cfg, **kw: report)
        return report

    def _posting(self, job_id, company="Northwind"):
        from jobsearch.models import JobPosting

        return JobPosting(
            company=company,
            title="Director of Engineering, AI",
            url="https://example.com/job/1",
            location="Amsterdam, Netherlands",
            description="Fresh text from this sweep.",
            job_id=job_id,
        )

    def test_new_postings_are_added(self, cfg, monkeypatch):
        from jobsearch.tui import scan_blocking

        self._fake_report([self._posting("brand-new-1")], monkeypatch)
        assert "1 new" in scan_blocking(cfg)
        with Tracker.from_config(cfg) as tracker:
            assert tracker.get_job("brand-new-1") is not None

    def test_a_dismissed_role_is_skipped_entirely(self, cfg, monkeypatch):
        from jobsearch.tui import dismiss_blocking, scan_blocking

        job_id = seed(cfg)
        dismiss_blocking(cfg, job_id)
        with Tracker.from_config(cfg) as tracker:
            before = tracker.get_job(job_id)["description"]

        self._fake_report([self._posting(job_id)], monkeypatch)
        summary = scan_blocking(cfg)

        assert "left dismissed" in summary
        with Tracker.from_config(cfg) as tracker:
            row = tracker.get_job(job_id)
        assert row["status"] == "Withdrawn", "a sweep must not resurrect a dismissal"
        assert row["description"] == before, "and must not rewrite it either"

    def test_a_role_in_progress_is_left_alone(self, cfg, monkeypatch):
        from jobsearch.tui import scan_blocking, set_status_blocking

        job_id = seed(cfg)
        set_status_blocking(cfg, job_id, "Applied")
        with Tracker.from_config(cfg) as tracker:
            before = tracker.get_job(job_id)["description"]

        self._fake_report([self._posting(job_id)], monkeypatch)
        summary = scan_blocking(cfg)

        assert "in progress, untouched" in summary
        with Tracker.from_config(cfg) as tracker:
            row = tracker.get_job(job_id)
        assert row["status"] == "Applied"
        assert row["description"] == before

    def test_an_untouched_role_is_refreshed(self, cfg, monkeypatch):
        from jobsearch.tui import scan_blocking

        job_id = seed(cfg)
        self._fake_report([self._posting(job_id)], monkeypatch)
        assert "refreshed" in scan_blocking(cfg)
        with Tracker.from_config(cfg) as tracker:
            assert "Fresh text" in tracker.get_job(job_id)["description"]

    def test_board_errors_are_reported_not_swallowed(self, cfg, monkeypatch):
        from types import SimpleNamespace

        import jobsearch.discover as discover_mod
        from jobsearch.tui import scan_blocking

        monkeypatch.setattr(
            discover_mod,
            "discover",
            lambda cfg, **kw: SimpleNamespace(
                postings=[], boards_checked=2, raw_count=0, errors=["acme: HTTP 404"]
            ),
        )
        assert "acme: HTTP 404" in scan_blocking(cfg)


class TestTierPicker:
    def test_options_come_from_the_configured_boards(self, cfg):
        from jobsearch.tui import tier_options

        options = tier_options(cfg)
        assert options[0][1] is None, "the first option scans everything"
        tiers = [t for _, t in options[1:]]
        configured = sorted({b.tier for b in cfg.boards})
        assert tiers == [[t] for t in configured]

    def test_labels_carry_board_counts(self, cfg):
        from jobsearch.tui import tier_options

        label, _ = tier_options(cfg)[0]
        assert f"({len(cfg.boards)} board" in label

    def test_it_singularises_a_lone_board(self, cfg):
        from jobsearch.config import BoardRef
        from jobsearch.tui import tier_options

        cfg.boards = [BoardRef(company="Solo", ats="ashby", token="solo", tier=1)]
        assert "(1 board)" in tier_options(cfg)[0][0]

    def test_f_opens_the_picker_rather_than_scanning(self, cfg):
        """The keypress must not start a network sweep on its own."""

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.action_scan()
                await pilot.pause()
                assert len(app.screen_stack) == 2
                assert app.busy is False, "nothing runs until a tier is chosen"

        asyncio.run(scenario())

    def test_cancelling_the_picker_runs_nothing(self, cfg, monkeypatch):
        import jobsearch.tui as tui_mod

        called = []
        monkeypatch.setattr(tui_mod, "scan_blocking", lambda *a, **k: called.append(1))

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.after_scan_choice(None)
                await pilot.pause()
                assert called == []
                assert app.busy is False

        asyncio.run(scenario())

    def test_choosing_a_tier_scans_only_that_tier(self, cfg, monkeypatch):
        from types import SimpleNamespace

        import jobsearch.discover as discover_mod

        seen: dict = {}

        def fake_discover(_cfg, **kw):
            seen.update(kw)
            return SimpleNamespace(postings=[], boards_checked=1, raw_count=0, errors=[])

        monkeypatch.setattr(discover_mod, "discover", fake_discover)
        from jobsearch.tui import scan_blocking

        scan_blocking(cfg, [2])
        assert seen["tiers"] == [2]


class TestRolePage:
    """The page advertises stage keys, so they have to work from inside it.

    A ModalScreen's bindings shadow the app's: without its own, the page said
    "press s" while s did nothing.
    """

    def test_the_page_binds_every_stage_key_it_advertises(self, cfg):
        seed(cfg)

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                keys = set(app.screen.active_bindings)
                for key in ("s", "t", "o", "v", "w", "escape"):
                    assert key in keys, f"the role page does not bind {key!r}"

        asyncio.run(scenario())

    def test_a_stage_key_on_the_page_runs_that_stage(self, cfg, monkeypatch):
        import jobsearch.tui as tui_mod

        calls = []
        monkeypatch.setattr(
            tui_mod, "run_stage_blocking",
            lambda cfg, stage, job_id, **kw: calls.append(stage) or f"ran {stage}",
        )
        seed(cfg)

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                await pilot.press("s")
                for _ in range(20):
                    await pilot.pause()
                    if calls:
                        break
                assert calls == ["score"], "pressing s on the page must score"

        asyncio.run(scenario())

    def test_markup_escapes_untrusted_text(self, cfg):
        """Reasoning and drafts contain literal brackets; markup would eat them."""
        from jobsearch.models import Contact, OutreachDraft
        from jobsearch.tui import role_detail_markup

        job_id = seed(cfg)
        with Tracker.from_config(cfg) as tracker:
            tracker.save_outreach(
                OutreachDraft(
                    job_id=job_id,
                    contacts=[Contact(title="VP Eng", linkedin_search_url="https://x.test/1")],
                    email_body="Hi [name], regards",
                )
            )
        markup = role_detail_markup(cfg, job_id)
        assert "\\[name]" in markup, "the placeholder must be escaped, not swallowed"

    def test_it_renders_without_a_score(self, cfg):
        from jobsearch.tui import role_detail_markup

        markup = role_detail_markup(cfg, seed(cfg))
        assert "Not scored yet" in markup
        assert "No contacts yet" in markup


class TestAttachCv:
    """For CVs tailored by hand before the tool existed."""

    def _pdf(self, cfg, tmp_path):
        """Render a fixture CV to a real PDF via the same path the tool uses."""
        from jobsearch.render import render_from_config

        html = tmp_path / "hand_made.html"
        html.write_text((cfg.base_cv).read_text(encoding="utf-8"), encoding="utf-8")
        return render_from_config(cfg, html, tmp_path / "hand_made.pdf")

    def test_it_records_the_pdf_against_the_role(self, cfg, tmp_path):
        from jobsearch.tui import attach_cv_blocking

        job_id = seed(cfg)
        pdf = self._pdf(cfg, tmp_path)
        message = attach_cv_blocking(cfg, job_id, pdf)
        assert "hand_made.pdf" in message
        with Tracker.from_config(cfg) as tracker:
            assert tracker.get_job(job_id)["cv_pdf_path"] == str(Path(pdf).resolve())

    def test_it_picks_up_a_sibling_html(self, cfg, tmp_path):
        from jobsearch.tui import attach_cv_blocking

        job_id = seed(cfg)
        pdf = self._pdf(cfg, tmp_path)
        attach_cv_blocking(cfg, job_id, pdf)
        with Tracker.from_config(cfg) as tracker:
            assert tracker.get_job(job_id)["cv_html_path"].endswith("hand_made.html")

    def test_it_verifies_by_default_and_stores_the_report(self, cfg, tmp_path):
        from jobsearch.tui import attach_cv_blocking

        job_id = seed(cfg)
        message = attach_cv_blocking(cfg, job_id, self._pdf(cfg, tmp_path))
        assert "ATS" in message
        with Tracker.from_config(cfg) as tracker:
            assert tracker.get_job(job_id)["ats_json"], "the report should be stored"

    def test_no_verify_skips_the_check(self, cfg, tmp_path):
        from jobsearch.tui import attach_cv_blocking

        job_id = seed(cfg)
        message = attach_cv_blocking(cfg, job_id, self._pdf(cfg, tmp_path), verify=False)
        assert "ATS" not in message

    def test_a_missing_file_is_a_clear_error(self, cfg):
        from jobsearch.tui import attach_cv_blocking

        with pytest.raises(FileNotFoundError, match="No such file"):
            attach_cv_blocking(cfg, seed(cfg), "/nope/missing.pdf")

    def test_candidate_files_are_newest_first(self, cfg, tmp_path):
        import os

        from jobsearch.tui import candidate_cv_files

        folder = cfg.base_cv.parent
        old, new = folder / "old_cv.pdf", folder / "new_cv.pdf"
        old.write_bytes(b"%PDF-1.4\n"); new.write_bytes(b"%PDF-1.4\n")
        os.utime(old, (1_000_000, 1_000_000))
        try:
            names = [p.name for p in candidate_cv_files(cfg)]
            assert names.index("new_cv.pdf") < names.index("old_cv.pdf")
        finally:
            old.unlink(); new.unlink()


class TestSettingsScreen:
    def test_comma_opens_settings(self, cfg):
        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press(",")
                await pilot.pause()
                assert len(app.screen_stack) == 2

        asyncio.run(scenario())

    def test_saving_writes_the_file_and_reloads_the_config(self, cfg):
        """A settings change must take effect without restarting the app."""
        from textual.widgets import Input

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                assert app.cfg.raw["constraints"]["comp_floor_eur"] != 199000
                await pilot.press(",")
                await pilot.pause()
                app.screen.query_one("#set-comp_floor_eur", Input).value = "199000"
                app.screen.action_save()
                await pilot.pause()
                assert app.cfg.raw["constraints"]["comp_floor_eur"] == 199000

        asyncio.run(scenario())

    def test_an_invalid_value_does_not_write(self, cfg):
        from textual.widgets import Input

        before = cfg.source.read_text(encoding="utf-8")

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press(",")
                await pilot.pause()
                app.screen.query_one("#w-buyer", Input).value = "0.9"
                app.screen.action_save()
                await pilot.pause()
                assert len(app.screen_stack) == 2, "the screen stays open on error"
                assert cfg.source.read_text(encoding="utf-8") == before

        asyncio.run(scenario())


class TestNotesAndContacts:
    def test_a_note_is_stored_and_shown_on_the_page(self, cfg):
        from jobsearch.tui import add_note_blocking, role_detail_markup

        job_id = seed(cfg)
        assert "None yet" in role_detail_markup(cfg, job_id)
        add_note_blocking(cfg, job_id, "Referred by a former colleague.")
        assert "Referred by a former colleague." in role_detail_markup(cfg, job_id)

    def test_an_empty_note_is_refused(self, cfg):
        from jobsearch.tui import add_note_blocking

        with pytest.raises(ValueError, match="cannot be empty"):
            add_note_blocking(cfg, seed(cfg), "   ")

    def test_notes_accumulate(self, cfg):
        from jobsearch.tui import add_note_blocking, role_detail_markup

        job_id = seed(cfg)
        add_note_blocking(cfg, job_id, "first")
        add_note_blocking(cfg, job_id, "second")
        markup = role_detail_markup(cfg, job_id)
        assert "first" in markup and "second" in markup

    def test_a_manual_contact_does_not_replace_inferred_ones(self, cfg):
        """save_outreach owns its contacts; one you add yourself must survive."""
        from jobsearch.models import Contact, OutreachDraft
        from jobsearch.tui import add_contact_blocking

        job_id = seed(cfg)
        with Tracker.from_config(cfg) as tracker:
            tracker.save_outreach(
                OutreachDraft(job_id=job_id, contacts=[Contact(title="VP Engineering")])
            )
        add_contact_blocking(cfg, job_id, name="Real Person", title="CTO")
        with Tracker.from_config(cfg) as tracker:
            titles = {str(c["title"]) for c in tracker.contacts(job_id)}
        assert titles == {"VP Engineering", "CTO"}

    def test_a_contact_needs_a_name_or_title(self, cfg):
        from jobsearch.tui import add_contact_blocking

        with pytest.raises(ValueError, match="name or a title"):
            add_contact_blocking(cfg, seed(cfg), name="  ", title="  ")


class TestPresenceColumns:
    def test_the_table_has_cv_and_outreach_columns(self, cfg):
        seed(cfg)

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                from textual.widgets import DataTable

                labels = [str(c.label) for c in app.query_one("#table", DataTable).columns.values()]
                assert "CV" in labels and "Out" in labels

        asyncio.run(scenario())

    def test_outreach_ids_are_fetched_in_one_query(self, cfg):
        from jobsearch.models import Contact, OutreachDraft
        from jobsearch.tui import tracker_outreach_ids

        with_outreach = seed(cfg, company="HasOutreach")
        seed(cfg, company="Bare", title="Nothing Here")
        with Tracker.from_config(cfg) as tracker:
            tracker.save_outreach(
                OutreachDraft(job_id=with_outreach, contacts=[Contact(title="VP Eng")])
            )
        ids = tracker_outreach_ids(cfg)
        assert with_outreach in ids
        assert len(ids) == 1


class TestScreensActuallyOpen:
    """Press the key. Every modal key must open its screen.

    A helper named `_attach` once overrode MessagePump._attach - how Textual
    attaches a node to the tree - so the screen could never mount, and the
    helper's own `except Exception` swallowed the cause. Nothing caught it
    because the tests called the underlying functions instead of pressing keys.
    """

    @pytest.mark.parametrize("key", ["c", "n", "p"])
    def test_role_page_keys_open_a_screen(self, cfg, key):
        seed(cfg)

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                await pilot.press(key)
                await pilot.pause()
                assert len(app.screen_stack) == 3, f"{key!r} did not open a screen"
                await pilot.press("escape")
                await pilot.pause()
                assert len(app.screen_stack) == 2

        asyncio.run(scenario())

    @pytest.mark.parametrize("key", ["a", "n", "f", "comma"])
    def test_main_screen_keys_open_a_screen(self, cfg, key):
        seed(cfg)

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press(key)
                await pilot.pause()
                assert len(app.screen_stack) == 2, f"{key!r} did not open a screen"

        asyncio.run(scenario())

    def test_no_screen_method_shadows_a_textual_internal(self):
        """Guard the whole class of bug, not just the one instance."""
        import re
        from pathlib import Path

        from textual.screen import ModalScreen

        source = (Path(__file__).resolve().parents[1] / "src/jobsearch/tui.py").read_text()
        defined = set(re.findall(r"^        def (\w+)\(", source, re.MULTILINE))
        allowed = {"__init__"}
        clashes = {
            name
            for name in defined
            if name not in allowed
            and not name.startswith(("on_", "action_", "key_"))
            and name != "compose"
            and hasattr(ModalScreen, name)
        }
        assert not clashes, f"these shadow Textual internals: {sorted(clashes)}"


class TestCopy:
    """Textual captures the mouse, so terminal selection does not reach the app.
    Copying has to be an explicit action.
    """

    def test_it_shells_out_to_the_platform_clipboard(self, monkeypatch):
        import subprocess

        import jobsearch.tui as tui_mod

        seen = {}

        def fake_run(command, input=None, check=False):
            seen["command"] = command
            seen["payload"] = input.decode("utf-8")
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/pbcopy" if name == "pbcopy" else None)
        monkeypatch.setattr("subprocess.run", fake_run)
        message = tui_mod.copy_to_clipboard("hello")
        assert seen["command"] == ["pbcopy"]
        assert seen["payload"] == "hello"
        assert "copied 5 characters" in message

    def test_it_reports_when_no_clipboard_exists(self, monkeypatch):
        import jobsearch.tui as tui_mod

        monkeypatch.setattr("shutil.which", lambda name: None)
        assert "no clipboard command" in tui_mod.copy_to_clipboard("hello")

    def test_empty_text_is_not_copied(self):
        import jobsearch.tui as tui_mod

        assert tui_mod.copy_to_clipboard("   ") == "nothing to copy"

    def test_a_clipboard_failure_is_reported_not_raised(self, monkeypatch):
        import subprocess

        import jobsearch.tui as tui_mod

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/pbcopy")

        def boom(*a, **k):
            raise OSError("pipe closed")

        monkeypatch.setattr("subprocess.run", boom)
        assert "clipboard failed" in tui_mod.copy_to_clipboard("hello")

    def test_contact_links_are_one_per_line(self, cfg):
        from jobsearch.models import Contact, OutreachDraft
        from jobsearch.tui import contact_links_text

        job_id = seed(cfg)
        with Tracker.from_config(cfg) as tracker:
            tracker.save_outreach(
                OutreachDraft(
                    job_id=job_id,
                    contacts=[
                        Contact(title="VP Eng", linkedin_search_url="https://x.test/1"),
                        Contact(title="", name="A Person", linkedin_search_url="https://x.test/2"),
                    ],
                )
            )
        lines = contact_links_text(cfg, job_id).splitlines()
        assert lines == ["VP Eng: https://x.test/1", "A Person: https://x.test/2"]

    @pytest.mark.parametrize("key", ["y", "l"])
    def test_copy_keys_are_bound_on_the_role_page(self, cfg, key):
        seed(cfg)

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                assert key in set(app.screen.active_bindings)

        asyncio.run(scenario())


class TestProviderInSettingsScreen:
    def test_the_screen_exposes_the_provider_fields(self, cfg):
        from textual.widgets import Input

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press(",")
                await pilot.pause()
                for key in ("provider", "model", "base_url", "api_key_env"):
                    assert app.screen.query_one(f"#set-{key}", Input) is not None

        asyncio.run(scenario())

    def test_switching_provider_without_a_base_url_does_not_write(self, cfg):
        from textual.widgets import Input

        before = cfg.source.read_text(encoding="utf-8")

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press(",")
                await pilot.pause()
                app.screen.query_one("#set-provider", Input).value = "openai_compatible"
                app.screen.query_one("#set-base_url", Input).value = ""
                app.screen.action_save()
                await pilot.pause()
                assert len(app.screen_stack) == 2, "the screen stays open on error"
                assert cfg.source.read_text(encoding="utf-8") == before

        asyncio.run(scenario())

    def test_switching_provider_persists_and_reloads(self, cfg):
        from textual.widgets import Input

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press(",")
                await pilot.pause()
                app.screen.query_one("#set-provider", Input).value = "openai_compatible"
                app.screen.query_one("#set-base_url", Input).value = "https://openrouter.ai/api/v1"
                app.screen.action_save()
                await pilot.pause()
                assert app.cfg.raw["claude"]["provider"] == "openai_compatible"

        asyncio.run(scenario())


class TestWholeCompanyContact:
    """A hiring manager is a fact about the company, not about one posting."""

    def _company(self, cfg, name="Northwind", n=3):
        return [seed(cfg, company=name, title=f"Role {i}") for i in range(n)]

    def test_it_applies_to_every_live_role_at_the_company(self, cfg):
        from jobsearch.tui import add_contact_blocking

        ids = self._company(cfg)
        seed(cfg, company="Other Co", title="Unrelated")
        message = add_contact_blocking(
            cfg, ids[0], name="A Person", title="CTO", whole_company=True
        )
        assert "3 role(s) at Northwind" in message
        with Tracker.from_config(cfg) as tracker:
            for job_id in ids:
                assert any(str(c["name"]) == "A Person" for c in tracker.contacts(job_id))

    def test_it_does_not_touch_other_companies(self, cfg):
        from jobsearch.tui import add_contact_blocking

        ids = self._company(cfg)
        other = seed(cfg, company="Other Co", title="Unrelated")
        add_contact_blocking(cfg, ids[0], name="A Person", whole_company=True)
        with Tracker.from_config(cfg) as tracker:
            assert tracker.contacts(other) == []

    def test_dismissed_roles_are_skipped(self, cfg):
        """You already said no to those, and a company can have thirty."""
        from jobsearch.tui import add_contact_blocking, dismiss_blocking

        ids = self._company(cfg)
        dismiss_blocking(cfg, ids[2])
        message = add_contact_blocking(cfg, ids[0], name="A Person", whole_company=True)
        assert "2 role(s)" in message
        with Tracker.from_config(cfg) as tracker:
            assert tracker.contacts(ids[2]) == []

    def test_it_does_not_duplicate_an_existing_contact(self, cfg):
        from jobsearch.tui import add_contact_blocking

        ids = self._company(cfg)
        add_contact_blocking(cfg, ids[1], name="A Person", title="CTO")
        message = add_contact_blocking(
            cfg, ids[0], name="A Person", title="CTO", whole_company=True
        )
        assert "1 already had them" in message
        with Tracker.from_config(cfg) as tracker:
            names = [str(c["name"]) for c in tracker.contacts(ids[1])]
        assert names.count("A Person") == 1

    def test_without_the_flag_only_one_role_gets_it(self, cfg):
        from jobsearch.tui import add_contact_blocking

        ids = self._company(cfg)
        add_contact_blocking(cfg, ids[0], name="A Person")
        with Tracker.from_config(cfg) as tracker:
            assert tracker.contacts(ids[1]) == []

    def test_the_checkbox_exists_on_the_contact_screen(self, cfg):
        from textual.widgets import Checkbox

        seed(cfg)

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press("enter"); await pilot.pause()
                await pilot.press("p"); await pilot.pause()   # contacts manager
                await pilot.press("a"); await pilot.pause()   # the add form
                box = app.screen.query_one("#c-all", Checkbox)
                assert box.value is False, "must be opt-in"

        asyncio.run(scenario())


class TestEditContacts:
    """Contacts arrive inferred or hand-typed; both need correcting."""

    def _one(self, cfg):
        from jobsearch.models import Contact, OutreachDraft

        job_id = seed(cfg)
        with Tracker.from_config(cfg) as tracker:
            tracker.save_outreach(
                OutreachDraft(
                    job_id=job_id,
                    contacts=[Contact(title="VP Engineering", linkedin_search_url="https://x.test/1")],
                )
            )
            return job_id, tracker.contacts(job_id)[0]["id"]

    def test_an_inferred_contact_can_be_named(self, cfg):
        """The outreach stage names a role; you find out who holds it."""
        from jobsearch.tui import update_contact_blocking

        job_id, contact_id = self._one(cfg)
        message = update_contact_blocking(
            cfg, contact_id, name="Kayne Putman", title="Director, Solutions Engineering",
            url="https://linkedin.test/kayne",
        )
        assert "Kayne Putman" in message
        with Tracker.from_config(cfg) as tracker:
            row = tracker.contacts(job_id)[0]
        assert row["name"] == "Kayne Putman"
        assert row["search_url"] == "https://linkedin.test/kayne"

    def test_editing_does_not_create_a_second_contact(self, cfg):
        from jobsearch.tui import update_contact_blocking

        job_id, contact_id = self._one(cfg)
        update_contact_blocking(cfg, contact_id, name="A Person")
        with Tracker.from_config(cfg) as tracker:
            assert len(tracker.contacts(job_id)) == 1

    def test_a_contact_cannot_be_emptied(self, cfg):
        from jobsearch.tui import update_contact_blocking

        _, contact_id = self._one(cfg)
        with pytest.raises(ValueError, match="name or a title"):
            update_contact_blocking(cfg, contact_id, name="  ", title="  ")

    def test_removing_one(self, cfg):
        from jobsearch.tui import delete_contact_blocking

        job_id, contact_id = self._one(cfg)
        assert "VP Engineering" in delete_contact_blocking(cfg, contact_id)
        with Tracker.from_config(cfg) as tracker:
            assert tracker.contacts(job_id) == []

    def test_the_tracker_refuses_an_unknown_field(self, cfg):
        from jobsearch.tracker import TrackerError

        _, contact_id = self._one(cfg)
        with Tracker.from_config(cfg) as tracker:
            with pytest.raises(TrackerError, match="unknown field"):
                tracker.update_contact(contact_id, nickname="K")

    def test_p_opens_the_manager_and_enter_opens_the_editor(self, cfg):
        self._one(cfg)

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press("enter"); await pilot.pause()
                await pilot.press("p"); await pilot.pause()
                assert type(app.screen).__name__ == "ContactsScreen"
                await pilot.press("enter"); await pilot.pause()
                assert type(app.screen).__name__ == "ContactScreen"
                from textual.widgets import Input
                assert app.screen.query_one("#c-title", Input).value == "VP Engineering"

        asyncio.run(scenario())

    def test_the_edit_form_has_no_whole_company_checkbox(self, cfg):
        """Applying an edit to every role would be a different operation."""
        from textual.widgets import Checkbox

        self._one(cfg)

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press("enter"); await pilot.pause()
                await pilot.press("p"); await pilot.pause()
                await pilot.press("enter"); await pilot.pause()
                assert not app.screen.query(Checkbox)

        asyncio.run(scenario())


class TestShareContactAcrossCompany:
    """Editing and sharing have to compose: correct someone once, then push
    the correction to every role at that employer.
    """

    def _setup(self, cfg, n=3):
        from jobsearch.models import Contact

        ids = [seed(cfg, company="Northwind", title=f"Role {i}") for i in range(n)]
        with Tracker.from_config(cfg) as tracker:
            tracker.add_contact(ids[0], Contact(title="VP Engineering"))
            contact_id = tracker.contacts(ids[0])[0]["id"]
        return ids, contact_id

    def test_it_copies_to_the_other_roles(self, cfg):
        from jobsearch.tui import share_contact_blocking

        ids, contact_id = self._setup(cfg)
        assert "2 added" in share_contact_blocking(cfg, ids[0], contact_id)
        with Tracker.from_config(cfg) as tracker:
            for job_id in ids[1:]:
                assert [c["title"] for c in tracker.contacts(job_id)] == ["VP Engineering"]

    def test_an_edit_then_a_share_updates_rather_than_duplicating(self, cfg):
        """The exact flow: an inferred role name, later given a person."""
        from jobsearch.tui import share_contact_blocking, update_contact_blocking

        ids, contact_id = self._setup(cfg)
        share_contact_blocking(cfg, ids[0], contact_id)
        update_contact_blocking(cfg, contact_id, name="Kayne Putman", title="Director, SE")
        message = share_contact_blocking(cfg, ids[0], contact_id)

        assert "2 updated" in message
        with Tracker.from_config(cfg) as tracker:
            rows = tracker.contacts(ids[1])
        assert len(rows) == 1, "must update in place, not add a second entry"
        assert rows[0]["name"] == "Kayne Putman"

    def test_sharing_twice_changes_nothing_further(self, cfg):
        from jobsearch.tui import share_contact_blocking

        ids, contact_id = self._setup(cfg)
        share_contact_blocking(cfg, ids[0], contact_id)
        share_contact_blocking(cfg, ids[0], contact_id)
        with Tracker.from_config(cfg) as tracker:
            assert len(tracker.contacts(ids[1])) == 1

    def test_dismissed_roles_are_skipped(self, cfg):
        from jobsearch.tui import dismiss_blocking, share_contact_blocking

        ids, contact_id = self._setup(cfg)
        dismiss_blocking(cfg, ids[2])
        assert "1 added" in share_contact_blocking(cfg, ids[0], contact_id)
        with Tracker.from_config(cfg) as tracker:
            assert tracker.contacts(ids[2]) == []

    def test_other_companies_are_untouched(self, cfg):
        from jobsearch.tui import share_contact_blocking

        ids, contact_id = self._setup(cfg)
        other = seed(cfg, company="Other Co", title="Unrelated")
        share_contact_blocking(cfg, ids[0], contact_id)
        with Tracker.from_config(cfg) as tracker:
            assert tracker.contacts(other) == []

    def test_a_missing_contact_is_a_clear_error(self, cfg):
        from jobsearch.tui import share_contact_blocking

        with pytest.raises(ValueError, match="no contact with id"):
            share_contact_blocking(cfg, seed(cfg), 999999)

    def test_c_is_bound_on_the_contacts_manager(self, cfg):
        self._setup(cfg)

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press("enter"); await pilot.pause()
                await pilot.press("p"); await pilot.pause()
                assert "c" in set(app.screen.active_bindings)

        asyncio.run(scenario())


class TestSharedContactsSurviveRenames:
    """The rename is exactly what prompts a share, so it must not break the
    link between copies. Text matching cannot do that; a group id can.
    """

    def _two_roles(self, cfg):
        from jobsearch.models import Contact

        ids = [seed(cfg, company="Northwind", title=f"Role {i}") for i in range(2)]
        with Tracker.from_config(cfg) as tracker:
            tracker.add_contact(ids[0], Contact(title="VP Engineering"))
            return ids, tracker.contacts(ids[0])[0]["id"]

    def test_a_rename_then_a_share_updates_the_copy(self, cfg):
        from jobsearch.tui import share_contact_blocking, update_contact_blocking

        ids, contact_id = self._two_roles(cfg)
        share_contact_blocking(cfg, ids[0], contact_id)
        update_contact_blocking(cfg, contact_id, name="Kayne Putman", title="Director, SE")
        assert "1 updated" in share_contact_blocking(cfg, ids[0], contact_id)
        with Tracker.from_config(cfg) as tracker:
            rows = tracker.contacts(ids[1])
        assert len(rows) == 1
        assert rows[0]["name"] == "Kayne Putman"

    def test_repeated_renames_never_accumulate_copies(self, cfg):
        from jobsearch.tui import share_contact_blocking, update_contact_blocking

        ids, contact_id = self._two_roles(cfg)
        for name in ("First Name", "Second Name", "Third Name"):
            update_contact_blocking(cfg, contact_id, name=name, title="Director, SE")
            share_contact_blocking(cfg, ids[0], contact_id)
        with Tracker.from_config(cfg) as tracker:
            rows = tracker.contacts(ids[1])
        assert len(rows) == 1, f"accumulated {len(rows)} copies"
        assert rows[0]["name"] == "Third Name"

    def test_a_copy_made_before_group_ids_is_adopted(self, cfg):
        """Contacts already in the database predate the group column."""
        from jobsearch.models import Contact
        from jobsearch.tui import share_contact_blocking

        ids, contact_id = self._two_roles(cfg)
        with Tracker.from_config(cfg) as tracker:
            tracker.add_contact(ids[1], Contact(title="VP Engineering"))  # no group
        assert "1 updated" in share_contact_blocking(cfg, ids[0], contact_id)
        with Tracker.from_config(cfg) as tracker:
            rows = tracker.contacts(ids[1])
        assert len(rows) == 1, "the pre-existing copy should be adopted, not duplicated"
        assert rows[0]["group_id"] == contact_id

    def test_two_different_people_stay_separate(self, cfg):
        from jobsearch.models import Contact
        from jobsearch.tui import share_contact_blocking

        ids, first = self._two_roles(cfg)
        with Tracker.from_config(cfg) as tracker:
            tracker.add_contact(ids[0], Contact(title="Head of Talent"))
            second = [c["id"] for c in tracker.contacts(ids[0]) if c["title"] == "Head of Talent"][0]
        share_contact_blocking(cfg, ids[0], first)
        share_contact_blocking(cfg, ids[0], second)
        with Tracker.from_config(cfg) as tracker:
            titles = sorted(str(c["title"]) for c in tracker.contacts(ids[1]))
        assert titles == ["Head of Talent", "VP Engineering"]


class TestLetterRendering:
    """A stored path can go stale - the file gets renamed or moved. Rendering
    an empty section reads as "no letter" when the truth is "cannot find it".
    """

    def _with_letter(self, cfg, body="Hello,\n\nA letter.\n"):
        path = cfg.ensure_output_dir() / "letters" / "Cover_Letter_Northwind.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        job_id = seed(cfg)
        with Tracker.from_config(cfg) as tracker:
            tracker.save_letter(job_id, str(path))
        return job_id, path

    def test_the_letter_body_is_shown(self, cfg):
        from jobsearch.tui import role_detail_markup

        job_id, _ = self._with_letter(cfg)
        markup = role_detail_markup(cfg, job_id)
        assert "COVER LETTER" in markup
        assert "A letter." in markup

    def test_a_missing_file_says_so_rather_than_rendering_nothing(self, cfg):
        from jobsearch.tui import role_detail_markup

        job_id, path = self._with_letter(cfg)
        path.unlink()
        markup = role_detail_markup(cfg, job_id)
        assert "cannot read it" in markup
        assert "press [b]b[/b] to write it again" in markup

    def test_an_empty_file_is_distinguished_from_a_missing_one(self, cfg):
        from jobsearch.tui import role_detail_markup

        job_id, path = self._with_letter(cfg, body="   \n")
        path.write_text("   \n", encoding="utf-8")
        assert "the file is empty" in role_detail_markup(cfg, job_id)

    def test_no_letter_means_no_section(self, cfg):
        from jobsearch.tui import role_detail_markup

        assert "COVER LETTER" not in role_detail_markup(cfg, seed(cfg))

    def test_brackets_in_a_letter_survive(self, cfg):
        """Letters contain [name] placeholders; markup would swallow them."""
        from jobsearch.tui import role_detail_markup

        job_id, _ = self._with_letter(cfg, body="Hi [name],\n\nRegards.\n")
        assert "\\[name]" in role_detail_markup(cfg, job_id)


class TestClosedRolesSortLast:
    """A rejection is history worth keeping, but it should not sit above roles
    still in play. Dismissed roles are hidden; rejected ones sink.
    """

    def test_rejected_sinks_below_live_roles(self, cfg):
        from jobsearch.tui import set_status_blocking

        live = seed(cfg, company="Live Co", title="Open Role")
        dead = seed(cfg, company="Dead Co", title="Closed Role")
        set_status_blocking(cfg, dead, "Applied")
        set_status_blocking(cfg, dead, "Rejected")

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                companies = [str(r["company"]) for r in app.rows]
                assert companies.index("Live Co") < companies.index("Dead Co")

        asyncio.run(scenario())

    def test_a_rejected_role_stays_visible(self, cfg):
        """Unlike dismissal, rejection is not a reason to hide it."""
        from jobsearch.tui import set_status_blocking

        job_id = seed(cfg, company="Dead Co")
        set_status_blocking(cfg, job_id, "Applied")
        set_status_blocking(cfg, job_id, "Rejected")

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                assert any(str(r["company"]) == "Dead Co" for r in app.rows)

        asyncio.run(scenario())

    def test_ranking_within_the_live_group_is_untouched(self, cfg):
        """The sort must not disturb the score ordering the query produced."""
        from jobsearch.models import ScoreReport
        from jobsearch.tui import is_closed

        rows = [
            {"status": "Applied", "score_weighted": 4.0},
            {"status": "Rejected", "score_weighted": 4.5},
            {"status": "Not started", "score_weighted": 3.0},
        ]
        rows.sort(key=is_closed)
        assert [r["score_weighted"] for r in rows] == [4.0, 3.0, 4.5]

    def test_is_closed_covers_both_endings(self):
        from jobsearch.tui import is_closed

        assert is_closed({"status": "Rejected"})
        assert is_closed({"status": "Withdrawn"})
        assert not is_closed({"status": "Applied"})
        assert not is_closed({"status": "Offer"})
        assert not is_closed({})

    def test_the_subtitle_counts_live_and_closed_separately(self, cfg):
        from jobsearch.tui import set_status_blocking

        seed(cfg, company="Live Co")
        dead = seed(cfg, company="Dead Co", title="Closed Role")
        set_status_blocking(cfg, dead, "Applied")
        set_status_blocking(cfg, dead, "Rejected")

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                assert "1 live" in app.sub_title
                assert "1 closed" in app.sub_title

        asyncio.run(scenario())


class TestRowNumbersAndCounts:
    def test_rows_are_numbered_from_one(self, cfg):
        for i in range(3):
            seed(cfg, company=f"Co {i}", title=f"Role {i}")

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                from textual.widgets import DataTable

                table = app.query_one("#table", DataTable)
                assert table.show_row_labels
                assert [str(r.label) for r in table.rows.values()] == ["1", "2", "3"]

        asyncio.run(scenario())

    def test_numbering_follows_the_filter(self, cfg):
        """Numbers count what is on screen, not the whole database."""
        seed(cfg, company="Keep Co")
        seed(cfg, company="Drop Co", title="Other")

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.filter_text = "keep"
                app.action_refresh_rows()
                await pilot.pause()
                from textual.widgets import DataTable

                table = app.query_one("#table", DataTable)
                assert [str(r.label) for r in table.rows.values()] == ["1"]

        asyncio.run(scenario())

    def test_the_subtitle_counts_applications_in_flight(self, cfg):
        from jobsearch.tui import set_status_blocking

        seed(cfg, company="Untouched")
        applied = seed(cfg, company="Applied Co", title="A")
        interviewing = seed(cfg, company="Interviewing Co", title="B")
        set_status_blocking(cfg, applied, "Applied")
        set_status_blocking(cfg, interviewing, "Applied")
        set_status_blocking(cfg, interviewing, "Interviewing")

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                assert "2 applied" in app.sub_title, app.sub_title

        asyncio.run(scenario())

    def test_a_rejected_role_is_not_counted_as_in_flight(self, cfg):
        """It was applied for, but it is no longer out with anyone."""
        from jobsearch.tui import is_in_flight, set_status_blocking

        job_id = seed(cfg)
        set_status_blocking(cfg, job_id, "Applied")
        set_status_blocking(cfg, job_id, "Rejected")
        with Tracker.from_config(cfg) as tracker:
            assert not is_in_flight(tracker.get_job(job_id))

    def test_in_flight_covers_the_later_stages(self):
        from jobsearch.tui import is_in_flight

        for status in ("Applied", "In conversation", "Interviewing", "Offer"):
            assert is_in_flight({"status": status}), status
        for status in ("Not started", "Parked", "Rejected", "Withdrawn"):
            assert not is_in_flight({"status": status}), status


class TestDefaultOrdering:
    """Lifecycle stage first, then score, then company. A parked role sitting
    among live applications is the thing this fixes.
    """

    def _row(self, status, score=None, company="Co", title="Role"):
        return {"status": status, "score_weighted": score, "company": company, "title": title}

    def test_stage_beats_score(self, cfg):
        from jobsearch.tui import sort_key

        rows = [
            self._row("Parked", 5.0, "Parked Co"),
            self._row("Applied", 1.0, "Applied Co"),
        ]
        rows.sort(key=sort_key)
        assert [r["company"] for r in rows] == ["Applied Co", "Parked Co"]

    def test_the_full_lifecycle_order(self):
        from jobsearch.tui import sort_key

        statuses = ["Withdrawn", "Rejected", "Parked", "Not started",
                    "Outreach sent", "Applied", "In conversation",
                    "Interviewing", "Offer"]
        rows = [self._row(s) for s in statuses]
        rows.sort(key=sort_key)
        assert [r["status"] for r in rows] == [
            "Offer", "Interviewing", "In conversation", "Applied",
            "Outreach sent", "Not started", "Parked", "Rejected", "Withdrawn",
        ]

    def test_score_orders_within_a_stage(self):
        from jobsearch.tui import sort_key

        rows = [self._row("Applied", 3.0, "Low"), self._row("Applied", 4.5, "High")]
        rows.sort(key=sort_key)
        assert [r["company"] for r in rows] == ["High", "Low"]

    def test_unscored_sinks_within_its_stage(self):
        """An unscored role must never outrank a scored one beside it."""
        from jobsearch.tui import sort_key

        rows = [self._row("Applied", None, "Unscored"), self._row("Applied", 1.0, "Scored")]
        rows.sort(key=sort_key)
        assert [r["company"] for r in rows] == ["Scored", "Unscored"]

    def test_company_breaks_the_tie(self):
        """Stable between refreshes rather than dependent on insertion order."""
        from jobsearch.tui import sort_key

        rows = [self._row("Applied", 4.0, "Zeta"), self._row("Applied", 4.0, "Alpha")]
        rows.sort(key=sort_key)
        assert [r["company"] for r in rows] == ["Alpha", "Zeta"]

    def test_an_unknown_status_sorts_last_rather_than_crashing(self):
        from jobsearch.tui import sort_key

        rows = [self._row("Something New"), self._row("Applied")]
        rows.sort(key=sort_key)
        assert rows[0]["status"] == "Applied"

    def test_a_parked_role_lands_below_the_backlog_in_the_app(self, cfg):
        from jobsearch.tui import set_status_blocking

        parked = seed(cfg, company="Parked Co", title="Parked Role")
        seed(cfg, company="Backlog Co", title="Untouched Role")
        set_status_blocking(cfg, parked, "Parked")

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                companies = [str(r["company"]) for r in app.rows]
                assert companies.index("Backlog Co") < companies.index("Parked Co")

        asyncio.run(scenario())


class TestLocationFilter:
    """Most of a large backlog is roles you can never take. They are hidden by
    default, but engaging with one overrides that.
    """

    def test_an_untouched_us_role_is_hidden(self, cfg):
        from jobsearch.tui import is_hidden

        job_id = seed(cfg, company="US Co", title="Director", location="Remote - United States")
        with Tracker.from_config(cfg) as tracker:
            assert is_hidden(tracker.get_job(job_id), cfg)

    def test_a_workable_role_is_not_hidden(self, cfg):
        from jobsearch.tui import is_hidden

        job_id = seed(cfg, location="Amsterdam, Netherlands")
        with Tracker.from_config(cfg) as tracker:
            assert not is_hidden(tracker.get_job(job_id), cfg)

    def test_an_unclassifiable_location_stays_visible(self, cfg):
        """A '?' is a question to ask, not an answer. Mistral labels EMEA roles
        by office city, and those turned out to be workable."""
        from jobsearch.tui import is_hidden

        job_id = seed(cfg, company="Vague Co", title="Lead", location="CET, GMT or EST timezones")
        with Tracker.from_config(cfg) as tracker:
            assert not is_hidden(tracker.get_job(job_id), cfg)

    def test_applying_to_a_blocked_location_overrides_the_filter(self, cfg):
        """If you applied to something outside the EU on purpose, that decision
        is yours and the table must keep showing it."""
        from jobsearch.tui import is_hidden, set_status_blocking

        job_id = seed(cfg, company="US Co", title="Director", location="Remote - United States")
        set_status_blocking(cfg, job_id, "Applied")
        with Tracker.from_config(cfg) as tracker:
            assert not is_hidden(tracker.get_job(job_id), cfg)

    def test_a_missing_location_is_not_grounds_for_hiding(self, cfg):
        from jobsearch.tui import is_hidden

        job_id = seed(cfg, company="No Location Co", title="Lead", location="")
        with Tracker.from_config(cfg) as tracker:
            assert not is_hidden(tracker.get_job(job_id), cfg)

    def test_h_reveals_them_and_the_subtitle_counts_them(self, cfg):
        seed(cfg, company="Keep Co", location="Amsterdam, Netherlands")
        seed(cfg, company="US Co", title="Other", location="Remote - United States")

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                assert len(app.rows) == 1
                assert "1 hidden" in app.sub_title
                await pilot.press("h")
                await pilot.pause()
                assert len(app.rows) == 2
                assert "including hidden" in app.sub_title

        asyncio.run(scenario())


class TestBulkScoring:
    """53 unscored roles is past what anyone triages by eye. Scoring is the
    cheap stage, and hard constraints run locally before the model is touched.
    """

    def test_it_only_picks_unscored_roles(self, cfg):
        from jobsearch.models import ConstraintReport, ScoreReport
        from jobsearch.tui import unscored_job_ids

        a = seed(cfg, company="Unscored Co")
        b = seed(cfg, company="Scored Co", title="Other")
        with Tracker.from_config(cfg) as tracker:
            tracker.save_score(ScoreReport(job_id=b, constraints=ConstraintReport(), weighted=4.0))
            rows = tracker.list_jobs()
        assert unscored_job_ids(cfg, rows) == [a]

    def test_a_partial_run_keeps_what_it_finished(self, cfg):
        """Stopping must not discard roles already paid for."""
        from jobsearch.tui import score_many_blocking

        ids = [seed(cfg, company=f"Co {i}", title=f"Role {i}") for i in range(3)]
        seen: list[int] = []
        message = score_many_blocking(
            cfg, ids, dry_run=True,
            on_progress=lambda i, n, note: seen.append(i),
            should_stop=lambda: len(seen) >= 2,
        )
        assert "stopped after" in message
        assert len(seen) == 2

    def test_one_bad_role_does_not_stop_the_batch(self, cfg):
        from jobsearch.tui import score_many_blocking

        ids = [seed(cfg, company="Good Co"), "no-such-job", seed(cfg, company="Also Good", title="X")]
        message = score_many_blocking(cfg, ids, dry_run=True)
        assert "failed" in message
        assert "2 scored" in message

    def test_it_reports_eliminations_separately(self, cfg):
        """A role failing a hard constraint never reaches the model."""
        from jobsearch.tui import score_many_blocking

        blocked = seed(cfg, company="US Co", title="Director", location="Remote - United States")
        message = score_many_blocking(cfg, [blocked], dry_run=True)
        assert "1 eliminated" in message

    def test_capital_s_asks_before_spending(self, cfg):
        seed(cfg)

        async def scenario():
            app = build_app(cfg, dry_run=True)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press("S")
                await pilot.pause()
                assert type(app.screen).__name__ == "ScoreAllScreen"
                assert app.busy is False, "nothing runs until confirmed"
                await pilot.press("n")
                await pilot.pause()
                assert app.busy is False

        asyncio.run(scenario())

    def test_nothing_unscored_is_a_no_op(self, cfg):
        from jobsearch.models import ConstraintReport, ScoreReport

        job_id = seed(cfg)
        with Tracker.from_config(cfg) as tracker:
            tracker.save_score(ScoreReport(job_id=job_id, constraints=ConstraintReport(), weighted=4.0))

        async def scenario():
            app = build_app(cfg, dry_run=True)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press("S")
                await pilot.pause()
                assert len(app.screen_stack) == 1, "no dialog when there is nothing to do"

        asyncio.run(scenario())


class TestCopyLog:
    """Errors land in the log pane, which RichLog renders and does not hand
    back. Copying is exactly what you want at the moment something fails.
    """

    def test_log_lines_are_captured_as_plain_text(self, cfg):
        async def scenario():
            app = build_app(cfg, dry_run=True)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.log_line("[red]NotFoundError:[/] 404 page not found")
                await pilot.pause()
                assert "NotFoundError: 404 page not found" in app.log_history
                assert "[red]" not in "".join(app.log_history), "markup should be stripped"

        asyncio.run(scenario())

    def test_y_copies_the_log(self, cfg, monkeypatch):
        import jobsearch.tui as tui_mod

        copied = {}
        monkeypatch.setattr(
            tui_mod, "copy_to_clipboard",
            lambda text: copied.setdefault("text", text) and "" or "copied",
        )

        async def scenario():
            app = build_app(cfg, dry_run=True)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.log_line("something went wrong")
                await pilot.press("y")
                await pilot.pause()
                assert "something went wrong" in copied["text"]

        asyncio.run(scenario())

    def test_the_history_is_capped(self, cfg):
        async def scenario():
            app = build_app(cfg, dry_run=True)
            async with app.run_test() as pilot:
                await pilot.pause()
                for i in range(600):
                    app.log_line(f"line {i}")
                assert len(app.log_history) <= 500

        asyncio.run(scenario())
