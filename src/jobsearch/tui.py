"""A terminal UI over the pipeline.

The CLI is the contract; this is a front end onto exactly the same functions.
Every stage key here calls the same ``score_posting`` / ``tailor_cv`` /
``draft_outreach`` / ``verify_from_config`` the subcommands call, so behaviour
cannot drift between the two.

Textual is an optional dependency (``pip install -e '.[tui]'``). Import errors
are turned into a plain instruction rather than a traceback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
from .models import Status

EMPTY_STATE = (
    "[dim]No roles yet.[/]\n\n"
    "Run [b]jobsearch discover[/] to pull from the configured boards, "
    "or [b]jobsearch add[/] to enter one by hand."
)

TEXTUAL_MISSING = (
    "The TUI needs Textual, which is an optional extra.\n"
    "Install it with:  pip install -e '.[tui]'"
)


def _require_textual() -> None:
    try:
        import textual  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by hand
        raise RuntimeError(TEXTUAL_MISSING) from exc


# Status -> a single glyph, so the table stays scannable at a glance.
STATUS_GLYPH: dict[str, str] = {
    Status.NOT_STARTED.value: "○",
    Status.PARKED.value: "◔",
    Status.OUTREACH_SENT.value: "◑",
    Status.APPLIED.value: "●",
    Status.IN_CONVERSATION.value: "◕",
    Status.INTERVIEWING.value: "◕",
    Status.OFFER.value: "★",
    Status.REJECTED.value: "✗",
    Status.WITHDRAWN.value: "✗",
}


def status_glyph(status: str) -> str:
    return STATUS_GLYPH.get(status, "·")


def is_eliminated(row: Any) -> bool:
    """A stored constraints_ok of 0 means a hard constraint eliminated the role."""
    value = _get(row, "constraints_ok", None)
    return value is not None and not value


def score_cell(row: Any) -> str:
    """Render the weighted score, or why there isn't one."""
    if is_eliminated(row):
        return "elim"
    value = _get(row, "score_weighted")
    return f"{value:.2f}" if value is not None else "-"


def _get(row: Any, key: str, default: Any = None) -> Any:
    """Read a column from a sqlite3.Row or a plain mapping."""
    try:
        value = row[key]
    except (IndexError, KeyError, TypeError):
        return default
    return default if value is None else value


def truncate(text: str, width: int) -> str:
    text = str(text or "")
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def allowed_statuses(current: str) -> list[str]:
    """Statuses the tracker will actually accept from `current`, minus itself."""
    from .models import TRANSITIONS

    try:
        status = Status.parse(current)
    except Exception:
        return [s.value for s in Status]
    return sorted(s.value for s in TRANSITIONS.get(status, set()))


def set_status_blocking(cfg: Config, job_id: str, target: str) -> str:
    """Apply a status transition. Raises if the tracker rejects it."""
    from .tracker import Tracker

    with Tracker.from_config(cfg) as tracker:
        job_id = tracker.resolve_job_id(job_id)
        tracker.set_status(job_id, Status.parse(target), reason="set from the TUI")
    return target


def add_role_blocking(
    cfg: Config,
    *,
    company: str,
    title: str,
    url: str = "",
    location: str = "",
    description: str = "",
) -> str:
    """Insert a hand-entered role, the same way `jobsearch add` does."""
    from .models import JobPosting, make_job_id
    from .tracker import Tracker

    if not company.strip() or not title.strip():
        raise ValueError("company and title are required")
    posting = JobPosting(
        company=company.strip(),
        title=title.strip(),
        url=url.strip(),
        location=location.strip(),
        description=description.strip(),
        source="manual",
        job_id=make_job_id(company, title, url),
    )
    with Tracker.from_config(cfg) as tracker:
        return tracker.upsert_job(posting)


DISMISSED_STATUS = Status.WITHDRAWN


def dismiss_blocking(cfg: Config, job_id: str) -> str:
    """Mark a role as not being pursued, stickily.

    Uses a status rather than a delete on purpose: `upsert_job` never clobbers
    the status of an existing job, so a dismissed role stays dismissed when
    `discover` sweeps its board again. A deleted one would simply come back.
    """
    from .models import TRANSITIONS
    from .tracker import Tracker

    with Tracker.from_config(cfg) as tracker:
        job_id = tracker.resolve_job_id(job_id)
        current = Status.parse(str(tracker.get_job(job_id)["status"]))
        if current is DISMISSED_STATUS:
            return "already dismissed"
        target = (
            DISMISSED_STATUS
            if DISMISSED_STATUS in TRANSITIONS.get(current, set())
            else Status.PARKED
        )
        tracker.set_status(job_id, target, reason="dismissed from the TUI as not relevant")
    return target.value


def delete_blocking(cfg: Config, job_id: str) -> str:
    """Permanently remove a job row. `discover` may re-add it from its board."""
    from .tracker import Tracker

    with Tracker.from_config(cfg) as tracker:
        job_id = tracker.resolve_job_id(job_id)
        company = str(tracker.get_job(job_id)["company"])
        tracker.delete_job(job_id)
    return company


def location_cell(row: Any, cfg: Config) -> str:
    """Location plus a fitness glyph, using the same check the scorer uses.

    Deliberately calls check_location rather than re-deriving the rules, so the
    table can never disagree with what elimination will decide. A bare "remote"
    is not a pass: "Remote - United States" is a US role.
    """
    from .models import JobPosting, Verdict
    from .scoring import check_location

    location = str(_get(row, "location", "") or "")
    if not location:
        return "?  —"
    verdict = check_location(JobPosting(company="", title="", url="", location=location), cfg).verdict
    glyph = {Verdict.PASS: "✓", Verdict.FAIL: "✗", Verdict.UNKNOWN: "?"}[verdict]
    remote = " · remote" if "remote" in location.lower() else ""
    return f"{glyph} {truncate(location, 26)}{remote}"


def outreach_detail_text(cfg: Config, job_id: str) -> str:
    """Plain text of the contacts and drafted messages for one job.

    Returned without Textual markup: drafts contain literal square brackets
    (``Hi [name],``) that markup parsing would swallow.
    """
    from .tracker import Tracker

    with Tracker.from_config(cfg) as tracker:
        job_id = tracker.resolve_job_id(job_id)
        row = tracker.get_job(job_id)
        contacts = tracker.contacts(job_id)
        draft = tracker.latest_outreach(job_id)

    lines = [f"{_get(row, 'company', '')} — {_get(row, 'title', '')}", ""]

    if contacts:
        lines.append(f"LIKELY CONTACTS ({len(contacts)}) — open these searches yourself:")
        for contact in contacts:
            label = _get(contact, "name") or _get(contact, "title", "contact")
            lines.append(f"  · {label}")
            url = _get(contact, "search_url")
            if url:
                lines.append(f"    {url}")
        lines.append("")
    else:
        lines += ["No contacts yet — press o to draft outreach.", ""]

    if draft:
        sent = "marked sent" if _get(draft, "sent") else "nothing sent"
        lines.append(f"DRAFTS (created {str(_get(draft, 'created_at', ''))[:10]}, {sent})")
        for label, key in (
            ("Connection note", "connection_note"),
            ("LinkedIn message", "linkedin_message"),
        ):
            body = str(_get(draft, key, "")).strip()
            if body:
                lines += ["", f"-- {label} --", body]
        subject = str(_get(draft, "email_subject", "")).strip()
        body = str(_get(draft, "email_body", "")).strip()
        if subject or body:
            lines += ["", "-- Email --"]
            if subject:
                lines.append(f"Subject: {subject}")
            if body:
                lines += ["", body]

    lines += ["", f"Full copy-pasteable version:  jobsearch show {job_id}"]
    return "\n".join(lines)


def build_app(cfg: Config, *, dry_run: bool = False) -> Any:
    """Construct the Textual app. Imports Textual lazily so import is cheap."""
    _require_textual()

    from textual import work
    from textual.app import App, ComposeResult
    from textual.screen import ModalScreen
    from textual.binding import Binding
    from textual.containers import Vertical, VerticalScroll
    from textual.widgets import (
        DataTable, Footer, Header, Input, OptionList, RichLog, Static, TextArea,
    )

    class JobSearchTUI(App):  # type: ignore[misc]
        """Pipeline browser: pick a role, run a stage, watch the result."""

        CSS = """
        Screen { layout: vertical; }
        #table { height: 1fr; min-height: 6; }
        #detail { height: auto; max-height: 16; border-top: solid $accent; padding: 0 1; }
        #outreach { padding: 1 2; height: 1fr; }
        #picker { padding: 1 2; height: auto; background: $surface; border: solid $accent; }
        #newrole { padding: 1 2; height: 1fr; background: $surface; border: solid $accent; }
        #newrole Input { margin-bottom: 1; }
        #f-description { height: 12; }
        #log { height: 8; border-top: solid $accent; padding: 0 1; }
        #filter { display: none; }
        #filter.visible { display: block; }
        """

        BINDINGS = [
            Binding("s", "stage('score')", "Score"),
            Binding("t", "stage('tailor')", "Tailor"),
            Binding("o", "stage('outreach')", "Outreach"),
            Binding("v", "stage('verify')", "Verify"),
            Binding("enter", "open_outreach", "Read drafts"),
            Binding("a", "set_status", "Status"),
            Binding("n", "new_role", "Add role"),
            Binding("d", "dismiss", "Dismiss"),
            Binding("x", "delete_role", "Delete"),
            Binding("h", "toggle_dismissed", "Show dismissed"),
            Binding("r", "refresh_rows", "Refresh"),
            Binding("slash", "focus_filter", "Filter"),
            Binding("escape", "clear_filter", "Clear", show=False),
            Binding("q", "quit", "Quit"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.cfg = cfg
            self.dry_run = dry_run
            self.filter_text = ""
            self.show_dismissed = False
            self.rows: list[Any] = []
            self.busy = False
            # The text last rendered into the detail pane. Kept on the app so
            # tests can assert on it without reaching into widget internals,
            # which change shape between Textual versions.
            self.detail_text = ""

        # -- layout ------------------------------------------------------

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            yield Input(placeholder="filter by company or title…", id="filter")
            yield DataTable(id="table", cursor_type="row")
            with Vertical(id="detail"):
                yield Static("", id="detail-body")
            yield RichLog(id="log", markup=True, wrap=True)
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one("#table", DataTable)
            table.add_columns("Company", "Title", "Location", "Score", "Status")
            table.focus()
            self.title = "jobsearch"
            self.sub_title = str(self.cfg.source or self.cfg.root)
            self.action_refresh_rows()
            if self.dry_run:
                self.log_line("[yellow]dry-run: no API calls, nothing written[/]")

        # -- data --------------------------------------------------------

        def load_rows(self) -> list[Any]:
            from .tracker import Tracker

            with Tracker.from_config(self.cfg) as tracker:
                rows = tracker.list_jobs()
            if not self.show_dismissed:
                rows = [r for r in rows if str(_get(r, "status", "")) != DISMISSED_STATUS.value]
            needle = self.filter_text.strip().lower()
            if needle:
                rows = [
                    r
                    for r in rows
                    if needle in str(_get(r, "company", "")).lower()
                    or needle in str(_get(r, "title", "")).lower()
                ]
            return rows

        def action_refresh_rows(self) -> None:
            table = self.query_one("#table", DataTable)
            saved = table.cursor_row
            table.clear()
            self.rows = self.load_rows()
            for row in self.rows:
                table.add_row(
                    truncate(_get(row, "company", ""), 20),
                    truncate(_get(row, "title", ""), 32),
                    location_cell(row, self.cfg),
                    score_cell(row),
                    f"{status_glyph(str(_get(row, 'status', '')))} {_get(row, 'status', '')}",
                )
            if self.rows:
                table.move_cursor(row=min(saved, len(self.rows) - 1))
            self.update_detail()
            bits = [f"{len(self.rows)} role(s)"]
            if self.filter_text:
                bits.append(f"filter: {self.filter_text}")
            if self.show_dismissed:
                bits.append("including dismissed")
            self.sub_title = "  ·  ".join(bits)

        def selected_row(self) -> Any | None:
            table = self.query_one("#table", DataTable)
            if not self.rows or table.cursor_row is None:
                return None
            if not (0 <= table.cursor_row < len(self.rows)):
                return None
            return self.rows[table.cursor_row]

        # -- detail pane --------------------------------------------------

        def update_detail(self) -> None:
            body = self.query_one("#detail-body", Static)
            row = self.selected_row()
            self.detail_text = EMPTY_STATE if row is None else self.render_detail(row)
            body.update(self.detail_text)

        def render_detail(self, row: Any) -> str:
            company = _get(row, "company", "")
            title = _get(row, "title", "")
            lines = [f"[b]{company}[/] — {title}", ""]

            location = _get(row, "location") or "location not stated"
            lines.append(f"[dim]{location}[/]")
            url = _get(row, "url")
            if url:
                lines.append(f"[dim]{url}[/]")

            if is_eliminated(row):
                lines += ["", "[red]ELIMINATED[/] by a hard constraint"]
            elif _get(row, "score_weighted") is not None:
                dims = " │ ".join(
                    f"{name} {_get(row, f'score_{name}', '-')}"
                    for name in ("buyer", "role_fit", "company", "domain", "talent")
                )
                lines += ["", f"score [b]{_get(row, 'score_weighted'):.2f}[/]   {dims}"]
                rec = _get(row, "recommendation")
                if rec:
                    lines.append(f"recommendation: [b]{rec}[/]")
            else:
                lines += ["", "[dim]not scored yet — press [b]s[/b][/]"]

            cv = _get(row, "cv_pdf_path") or _get(row, "cv_html_path")
            lines.append(f"cv   {cv}" if cv else "[dim]cv   none yet — press [b]t[/b][/]")

            count = self.contact_count(str(_get(row, "job_id", "")))
            if count:
                lines.append(f"outreach  {count} contact(s) drafted — press [b]enter[/] to read")
            else:
                lines.append("[dim]outreach  none yet — press [b]o[/b][/]")
            return "\n".join(lines)

        def contact_count(self, job_id: str) -> int:
            if not job_id:
                return 0
            from .tracker import Tracker

            with Tracker.from_config(self.cfg) as tracker:
                return len(tracker.contacts(job_id))

        def on_data_table_row_highlighted(self, _event: Any) -> None:
            self.update_detail()

        def action_set_status(self) -> None:
            row = self.selected_row()
            if row is None:
                self.log_line("[yellow]nothing selected[/]")
                return
            self.push_screen(
                StatusScreen(str(_get(row, "job_id", "")), str(_get(row, "status", ""))),
                self.after_status,
            )

        def after_status(self, result: str | None) -> None:
            if result:
                self.log_line(result)
            self.action_refresh_rows()

        def action_toggle_dismissed(self) -> None:
            self.show_dismissed = not self.show_dismissed
            self.action_refresh_rows()

        def action_dismiss(self) -> None:
            row = self.selected_row()
            if row is None:
                self.log_line("[yellow]nothing selected[/]")
                return
            company = _get(row, "company", "")
            try:
                target = dismiss_blocking(self.cfg, str(_get(row, "job_id", "")))
            except Exception as exc:  # noqa: BLE001 - reported into the log pane
                self.log_line(f"[red]{type(exc).__name__}:[/] {exc}")
                return
            self.log_line(
                f"dismissed [b]{company}[/] → {target}  "
                f"[dim]h to show, a to restore[/]"
            )
            self.action_refresh_rows()

        def action_delete_role(self) -> None:
            row = self.selected_row()
            if row is None:
                self.log_line("[yellow]nothing selected[/]")
                return
            self.push_screen(
                ConfirmDeleteScreen(str(_get(row, "job_id", "")), str(_get(row, "company", ""))),
                self.after_delete,
            )

        def after_delete(self, result: str | None) -> None:
            if result:
                self.log_line(result)
            self.action_refresh_rows()

        def action_new_role(self) -> None:
            self.push_screen(NewRoleScreen(), self.after_new_role)

        def after_new_role(self, result: str | None) -> None:
            if result:
                self.log_line(result)
            self.action_refresh_rows()

        def action_open_outreach(self) -> None:
            row = self.selected_row()
            if row is None:
                return
            self.push_screen(OutreachScreen(self.cfg, str(_get(row, "job_id", ""))))

        # -- filtering ----------------------------------------------------

        def action_focus_filter(self) -> None:
            box = self.query_one("#filter", Input)
            box.add_class("visible")
            box.focus()

        def action_clear_filter(self) -> None:
            box = self.query_one("#filter", Input)
            box.value = ""
            box.remove_class("visible")
            self.filter_text = ""
            self.action_refresh_rows()
            self.query_one("#table", DataTable).focus()

        def on_input_submitted(self, event: Any) -> None:
            self.filter_text = event.value
            self.action_refresh_rows()
            self.query_one("#table", DataTable).focus()

        # -- stages -------------------------------------------------------

        def log_line(self, text: str) -> None:
            self.query_one("#log", RichLog).write(text)

        def action_stage(self, stage: str) -> None:
            row = self.selected_row()
            if row is None:
                self.log_line("[yellow]nothing selected[/]")
                return
            if self.busy:
                self.log_line("[yellow]a stage is already running[/]")
                return
            job_id = str(_get(row, "job_id", ""))
            company = _get(row, "company", "")
            self.busy = True
            self.log_line(f"[b]{stage}[/] → {company} [dim]({job_id})[/]")
            self.run_stage(stage, job_id)

        @work(thread=True, exclusive=True)
        def run_stage(self, stage: str, job_id: str) -> None:
            """Run one pipeline stage off the UI thread.

            Opens its own Tracker: a sqlite3 connection may only be used from
            the thread that created it.
            """
            try:
                message = run_stage_blocking(self.cfg, stage, job_id, dry_run=self.dry_run)
                self.call_from_thread(self.log_line, message)
            except Exception as exc:  # noqa: BLE001 - surfaced into the log pane
                self.call_from_thread(self.log_line, f"[red]{type(exc).__name__}:[/] {exc}")
            finally:
                self.call_from_thread(self.finish_stage)

        def finish_stage(self) -> None:
            self.busy = False
            self.action_refresh_rows()

    class ConfirmDeleteScreen(ModalScreen):  # type: ignore[misc]
        """Deletion is irreversible, so it asks. Dismissal (d) does not."""

        BINDINGS = [
            Binding("escape,n", "cancel", "Cancel"),
            Binding("y", "confirm", "Delete"),
        ]

        def __init__(self, job_id: str, company: str) -> None:
            super().__init__()
            self.job_id = job_id
            self.company = company

        def compose(self) -> ComposeResult:
            with Vertical(id="picker"):
                yield Static(
                    f"Permanently delete [b]{self.company}[/], with its scores, "
                    f"CV paths and outreach drafts?\n\n"
                    f"[dim]discover may re-add it from its board. To make it stay "
                    f"gone, press escape and use [b]d[/b] to dismiss instead.[/]\n\n"
                    f"[b]y[/] delete    [b]n[/] cancel"
                )
            yield Footer()

        def action_confirm(self) -> None:
            try:
                company = delete_blocking(cfg, self.job_id)
            except Exception as exc:  # noqa: BLE001 - reported into the log pane
                self.dismiss(f"[red]{type(exc).__name__}:[/] {exc}")
                return
            self.dismiss(f"deleted [b]{company}[/]")

        def action_cancel(self) -> None:
            self.dismiss(None)

    class StatusScreen(ModalScreen):  # type: ignore[misc]
        """Pick a new status, offering only transitions the tracker allows."""

        BINDINGS = [Binding("escape", "cancel", "Cancel")]

        def __init__(self, job_id: str, current: str) -> None:
            super().__init__()
            self.job_id = job_id
            self.current = current

        def compose(self) -> ComposeResult:
            options = allowed_statuses(self.current)
            with Vertical(id="picker"):
                yield Static(f"Status — currently [b]{self.current}[/]")
                yield OptionList(*options, id="statuses")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#statuses", OptionList).focus()

        def on_option_list_option_selected(self, event: Any) -> None:
            target = str(event.option.prompt)
            try:
                set_status_blocking(cfg, self.job_id, target)
            except Exception as exc:  # noqa: BLE001 - reported into the log pane
                self.dismiss(f"[red]{type(exc).__name__}:[/] {exc}")
                return
            self.dismiss(f"status → [b]{target}[/]")

        def action_cancel(self) -> None:
            self.dismiss(None)

    class NewRoleScreen(ModalScreen):  # type: ignore[misc]
        """Add a role found outside the configured boards."""

        BINDINGS = [
            Binding("escape", "cancel", "Cancel"),
            Binding("ctrl+s", "save", "Save"),
        ]

        def compose(self) -> ComposeResult:
            with Vertical(id="newrole"):
                yield Static("[b]Add a role[/]  —  company and title required, ctrl+s to save")
                yield Input(placeholder="company *", id="f-company")
                yield Input(placeholder="title *", id="f-title")
                yield Input(placeholder="url", id="f-url")
                yield Input(placeholder="location", id="f-location")
                yield Static("[dim]description — paste the posting[/]")
                yield TextArea(id="f-description")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#f-company", Input).focus()

        def action_save(self) -> None:
            def value(name: str) -> str:
                return self.query_one(f"#f-{name}", Input).value.strip()

            company, title = value("company"), value("title")
            if not company or not title:
                self.notify("company and title are required", severity="warning")
                return
            try:
                job_id = add_role_blocking(
                    cfg,
                    company=company,
                    title=title,
                    url=value("url"),
                    location=value("location"),
                    description=self.query_one("#f-description", TextArea).text,
                )
            except Exception as exc:  # noqa: BLE001 - reported into the log pane
                self.dismiss(f"[red]{type(exc).__name__}:[/] {exc}")
                return
            self.dismiss(f"added [b]{company}[/] [dim]({job_id})[/]")

        def action_cancel(self) -> None:
            self.dismiss(None)

    class OutreachScreen(ModalScreen):  # type: ignore[misc]
        """Full contacts and drafted messages for one job."""

        BINDINGS = [Binding("escape,q,enter", "dismiss_screen", "Back")]

        def __init__(self, cfg: Config, job_id: str) -> None:
            super().__init__()
            self.cfg = cfg
            self.job_id = job_id

        def compose(self) -> ComposeResult:
            with VerticalScroll(id="outreach"):
                # markup=False: drafts contain literal [name] placeholders.
                yield Static(outreach_detail_text(self.cfg, self.job_id), markup=False)
            yield Footer()

        def action_dismiss_screen(self) -> None:
            self.dismiss()

    return JobSearchTUI()


def run_stage_blocking(cfg: Config, stage: str, job_id: str, *, dry_run: bool = False) -> str:
    """Execute one stage synchronously and return a one-line result summary.

    Kept free of Textual so it is testable without a terminal.
    """
    from .claude import ClaudeClient
    from .tracker import Tracker

    with Tracker.from_config(cfg) as tracker:
        job_id = tracker.resolve_job_id(job_id)
        posting = tracker.get_posting(job_id)

        if stage == "verify":
            from .ats import verify_from_config

            row = tracker.get_job(job_id)
            cv_path = _get(row, "cv_pdf_path") or _get(row, "cv_html_path")
            if not cv_path:
                return "[yellow]no CV yet — tailor first[/]"
            report = verify_from_config(cfg, cv_path, posting.description or "")
            verdict = "[green]PASS[/]" if report.passed else "[red]FAIL[/]"
            return f"verify {verdict}  {Path(cv_path).name}  {report.page_count}pp"

        client = ClaudeClient.from_config(cfg, dry_run=dry_run)

        if stage == "score":
            from .scoring import score_posting

            report = score_posting(posting, cfg, client)
            if not dry_run:
                tracker.save_score(report)
            if report.eliminated:
                failures = report.constraints.failures
                reason = failures[0].reason if failures else "a hard constraint"
                return f"[red]eliminated[/] — {reason}"
            return f"scored [b]{report.weighted:.2f}[/] → {report.recommendation}"

        if stage == "tailor":
            from .tailor import tailor_cv

            result = tailor_cv(posting, cfg, client)
            if not dry_run:
                tracker.save_cv(job_id, str(result.html_path), result.pdf_path)
            ungrounded = len(result.ungrounded)
            flag = (
                "[green]0 ungrounded claims[/]"
                if ungrounded == 0
                else f"[yellow]{ungrounded} ungrounded claim(s) — review[/]"
            )
            name = Path(str(result.pdf_path or result.html_path)).name
            return f"tailored {name}  {flag}  [dim]press v to verify[/]"

        if stage == "outreach":
            from .outreach import draft_outreach

            draft = draft_outreach(posting, cfg, client)
            if not dry_run:
                tracker.save_outreach(draft)
            count = len(draft.contacts)
            return f"drafted outreach  {count} likely contact(s)  [dim]nothing sent[/]"

    raise ValueError(f"unknown stage: {stage}")


def run_tui(cfg: Config, *, dry_run: bool = False) -> int:
    """Entry point for ``jobsearch tui``."""
    app = build_app(cfg, dry_run=dry_run)
    app.run()
    return 0
