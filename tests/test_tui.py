"""Tests for the terminal UI.

The pure rendering helpers and the stage dispatcher are tested directly. The
app itself is exercised through Textual's headless ``run_test`` harness, so no
real terminal is involved.
"""

from __future__ import annotations

import asyncio

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
        location="Amsterdam, Netherlands",
        description="We need someone to run the AI team.",
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

    def test_enter_opens_the_screen(self, cfg):
        from jobsearch.models import Contact

        job_id = seed(cfg)
        with Tracker.from_config(cfg) as tracker:
            tracker.save_contacts(job_id, [Contact(title="VP Engineering", linkedin_search_url="https://x.test/1")])

        async def scenario():
            app = build_app(cfg)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.action_open_outreach()
                await pilot.pause()
                assert len(app.screen_stack) == 2

        asyncio.run(scenario())


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
                app.action_toggle_dismissed()
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
