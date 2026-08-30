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


def tier_options(cfg: Config) -> list[tuple[str, list[int] | None]]:
    """Scan choices derived from the configured boards, with their counts.

    Built from the config rather than hardcoded, so a config with only tier 1
    and 3 offers exactly those.
    """
    counts: dict[int, int] = {}
    for board in cfg.boards:
        counts[board.tier] = counts.get(board.tier, 0) + 1
    total = sum(counts.values())
    options: list[tuple[str, list[int] | None]] = [
        (f"All tiers  ({total} board{'s' if total != 1 else ''})", None)
    ]
    for tier in sorted(counts):
        n = counts[tier]
        options.append((f"Tier {tier}  ({n} board{'s' if n != 1 else ''})", [tier]))
    return options


def scan_blocking(cfg: Config, tiers: list[int] | None = None) -> str:
    """Sweep the configured boards and fold results into the tracker.

    Roles you have dismissed, or are already working on, are skipped entirely
    rather than upserted. `upsert_job` would preserve their status anyway, but
    skipping makes it impossible for a sweep to touch a role you have already
    made a decision about.
    """
    from .discover import discover
    from .tracker import Tracker

    report = discover(cfg, tiers=tiers)
    new = refreshed = dismissed = in_progress = 0

    with Tracker.from_config(cfg) as tracker:
        for posting in report.postings:
            existing = tracker.get_job(posting.job_id)
            if existing is None:
                tracker.upsert_job(posting)
                new += 1
                continue
            status = str(existing["status"])
            if status == DISMISSED_STATUS.value:
                dismissed += 1
                continue
            if status != Status.NOT_STARTED.value:
                in_progress += 1
                continue
            tracker.upsert_job(posting)
            refreshed += 1

    parts = [
        f"{report.boards_checked} board(s), {report.raw_count} posting(s), "
        f"{len(report.postings)} matched the title filter"
    ]
    parts.append(f"[b]{new} new[/]")
    if refreshed:
        parts.append(f"{refreshed} refreshed")
    if dismissed:
        parts.append(f"[dim]{dismissed} left dismissed[/]")
    if in_progress:
        parts.append(f"[dim]{in_progress} in progress, untouched[/]")
    summary = "scan: " + "  ·  ".join(parts)
    for error in report.errors[:5]:
        summary += f"\n  [yellow]![/] {error}"
    return summary


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


def candidate_cv_files(cfg: Config) -> list[Path]:
    """CVs already on disk: alongside the base CV, and in the output folder.

    Sorted newest first, because the one you just made is the one you want.
    """
    seen: dict[Path, float] = {}
    for folder in (cfg.base_cv.parent, cfg.output_dir / "cv", cfg.output_dir):
        if not folder.is_dir():
            continue
        for pattern in ("*.pdf", "*.html"):
            for path in folder.glob(pattern):
                if path.is_file() and not path.name.startswith("."):
                    seen[path.resolve()] = path.stat().st_mtime
    return sorted(seen, key=lambda p: -seen[p])


def attach_cv_blocking(
    cfg: Config, job_id: str, path: str | Path, *, verify: bool = True
) -> str:
    """Record an existing CV against a role, verifying it if it is a PDF.

    For CVs written outside the tool - the ones you tailored by hand before any
    of this existed - so the tracker and the xlsx export know they exist.
    """
    from .tracker import Tracker

    cv = Path(path).expanduser()
    if not cv.is_file():
        raise FileNotFoundError(f"No such file: {cv}")

    html_path, pdf_path = "", ""
    if cv.suffix.lower() == ".pdf":
        pdf_path = str(cv.resolve())
        sibling = cv.with_suffix(".html")
        html_path = str(sibling.resolve()) if sibling.is_file() else ""
    else:
        html_path = str(cv.resolve())
        sibling = cv.with_suffix(".pdf")
        pdf_path = str(sibling.resolve()) if sibling.is_file() else ""

    report = None
    note = ""
    if verify and pdf_path:
        from .ats import verify_from_config

        with Tracker.from_config(cfg) as tracker:
            posting = tracker.get_posting(tracker.resolve_job_id(job_id))
        report = verify_from_config(cfg, pdf_path, posting.description or "")
        note = f"  ATS {'[green]PASS[/]' if report.passed else '[red]FAIL[/]'} ({report.page_count}pp)"

    with Tracker.from_config(cfg) as tracker:
        resolved = tracker.resolve_job_id(job_id)
        tracker.save_cv(
            resolved, html_path, pdf_path or None, report.to_dict() if report else None
        )
    return f"attached [b]{cv.name}[/]{note}"


def role_detail_markup(cfg: Config, job_id: str) -> str:
    """The role page as Rich markup, with every dynamic value escaped.

    Escaping matters: postings, model reasoning and drafts all contain literal
    square brackets (``Hi [name],``) that markup would otherwise swallow. With
    them escaped we can use markup for structure and colour safely.
    """
    import json

    from rich.markup import escape

    from .tracker import Tracker

    with Tracker.from_config(cfg) as tracker:
        job_id = tracker.resolve_job_id(job_id)
        row = tracker.get_job(job_id)
        contacts = tracker.contacts(job_id)
        draft = tracker.latest_outreach(job_id)
        score_json = row["score_json"]

    def e(value: Any) -> str:
        return escape(str(value or ""))

    def rule(label: str) -> str:
        return f"\n[bold]{label}[/]\n[dim]{'─' * 58}[/]"

    out: list[str] = [
        f"[b]{e(_get(row, 'company', ''))}[/]  [dim]—[/]  {e(_get(row, 'title', ''))}",
        "",
        f"[dim]status  [/] {e(_get(row, 'status', ''))}",
        f"[dim]place   [/] {e(_get(row, 'location', '')) or '[dim]—[/]'}",
        f"[dim]link    [/] {e(_get(row, 'url', '')) or '[dim]—[/]'}",
    ]
    cv = _get(row, "cv_pdf_path") or _get(row, "cv_html_path")
    out.append(f"[dim]cv      [/] {e(cv)}" if cv else "[dim]cv       none yet — press t[/]")

    if score_json:
        data = json.loads(score_json)
        out.append(rule("HARD CONSTRAINTS"))
        for c in data.get("constraints", {}).get("results", []):
            colour, mark = {
                "pass": ("green", "ok "),
                "fail": ("red", "FAIL"),
                "unknown": ("yellow", " ? "),
            }.get(c["verdict"], ("white", "   "))
            out.append(f"  [{colour}]{mark}[/]  {e(c['name'])}: {e(c['reason'])}")

        weighted = data.get("weighted")
        rec = e(data.get("recommendation") or "—")
        head = f"{weighted:.2f}" if weighted is not None else "—"
        out.append(rule(f"SCORE {head}   ·   recommendation: {rec}"))
        for dim in data.get("dimensions", []):
            score = dim["score"]
            colour = "green" if score >= 4 else "yellow" if score >= 3 else "red"
            out.append(f"\n  [b]{e(dim['name'])}[/]  [{colour}]{score}/5[/]")
            out.append(f"  [dim]{e(dim['reasoning'])}[/]")
            for ev in dim.get("evidence", [])[:4]:
                out.append(f"    [dim]·[/] {e(ev)}")

        notes = (data.get("notes") or "").strip()
        if notes:
            out.append(rule("NOTES AND QUESTIONS"))
            out.append(f"  [dim]{e(notes)}[/]")
    else:
        out.append(rule("SCORE"))
        out.append("  [dim]Not scored yet — press [b]s[/b].[/]")

    out.append(rule("OUTREACH"))
    if contacts:
        out.append(f"  [dim]{len(contacts)} likely contact(s) — open these searches yourself[/]")
        for contact in contacts:
            out.append(f"  [b]·[/] {e(_get(contact, 'name') or _get(contact, 'title', 'contact'))}")
            url = _get(contact, "search_url")
            if url:
                out.append(f"    [dim]{e(url)}[/]")
    else:
        out.append("  [dim]No contacts yet — press [b]o[/b].[/]")

    if draft:
        for label, key in (
            ("Connection note", "connection_note"),
            ("LinkedIn message", "linkedin_message"),
        ):
            body = str(_get(draft, key, "")).strip()
            if body:
                out += ["", f"  [b]{label}[/]", f"  [dim]{e(body)}[/]"]
        subject = str(_get(draft, "email_subject", "")).strip()
        body = str(_get(draft, "email_body", "")).strip()
        if subject or body:
            out += ["", "  [b]Email[/]"]
            if subject:
                out.append(f"  [dim]Subject: {e(subject)}[/]")
            if body:
                out.append(f"  [dim]{e(body)}[/]")
    return "\n".join(out)


def role_detail_text(cfg: Config, job_id: str) -> str:
    """Everything known about one role, as plain text.

    Markup-free: postings and drafts contain literal square brackets that
    Textual markup would swallow.
    """
    import json

    from .tracker import Tracker

    with Tracker.from_config(cfg) as tracker:
        job_id = tracker.resolve_job_id(job_id)
        row = tracker.get_job(job_id)
        score_json = row["score_json"]

    lines = [
        f"{_get(row, 'company', '')} — {_get(row, 'title', '')}",
        "",
        f"status    {_get(row, 'status', '')}",
        f"location  {_get(row, 'location', '') or '—'}",
        f"url       {_get(row, 'url', '') or '—'}",
    ]
    cv = _get(row, "cv_pdf_path") or _get(row, "cv_html_path")
    if cv:
        lines.append(f"cv        {cv}")

    if score_json:
        data = json.loads(score_json)
        lines += ["", "HARD CONSTRAINTS"]
        for c in data.get("constraints", {}).get("results", []):
            mark = {"pass": "ok ", "fail": "FAIL", "unknown": " ? "}.get(c["verdict"], "   ")
            lines.append(f"  [{mark}] {c['name']}: {c['reason']}")

        weighted = data.get("weighted")
        header = f"SCORE {weighted:.2f}" if weighted is not None else "SCORE —"
        rec = data.get("recommendation")
        lines += ["", f"{header}   recommendation: {rec or '—'}"]
        for dim in data.get("dimensions", []):
            lines += ["", f"  {dim['name']} {dim['score']}/5", f"    {dim['reasoning']}"]
            for ev in dim.get("evidence", [])[:4]:
                lines.append(f"      · {ev}")

        notes = (data.get("notes") or "").strip()
        if notes:
            lines += ["", "NOTES AND QUESTIONS", f"  {notes}"]
    else:
        lines += ["", "Not scored yet — press s."]

    lines += ["", "-" * 60, "", outreach_detail_text(cfg, job_id)]
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
        #outreach { padding: 1 2; height: 1fr; background: $surface; }
        #rolestatus { height: auto; padding: 1 2 0 2; background: $surface; }
        #settings { padding: 1 2; height: 1fr; background: $surface; }
        #settings Input { margin-bottom: 1; }
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
            Binding("w", "open_url", "Open posting"),
            Binding("a", "set_status", "Status"),
            Binding("n", "new_role", "Add role"),
            Binding("d", "dismiss", "Dismiss"),
            Binding("x", "delete_role", "Delete"),
            Binding("h", "toggle_dismissed", "Show dismissed"),
            Binding("f", "scan", "Scan boards"),
            Binding("comma", "settings", "Settings"),
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

        def action_scan(self) -> None:
            if self.busy:
                self.log_line("[yellow]a stage is already running[/]")
                return
            self.push_screen(ScanScreen(self.cfg), self.after_scan_choice)

        def after_scan_choice(self, choice: tuple[str, list[int] | None] | None) -> None:
            if choice is None:
                return
            label, tiers = choice
            boards = [b for b in self.cfg.boards if tiers is None or b.tier in tiers]
            self.busy = True
            self.log_line(f"[b]scan[/] → {label.split('  ')[0]}, sweeping {len(boards)} board(s)…")
            self.run_scan(tiers)

        @work(thread=True, exclusive=True)
        def run_scan(self, tiers: list[int] | None = None) -> None:
            try:
                self.call_from_thread(self.log_line, scan_blocking(self.cfg, tiers))
            except Exception as exc:  # noqa: BLE001 - surfaced into the log pane
                self.call_from_thread(self.log_line, f"[red]{type(exc).__name__}:[/] {exc}")
            finally:
                self.call_from_thread(self.finish_stage)

        def action_settings(self) -> None:
            self.push_screen(SettingsScreen(self.cfg), self.after_settings)

        def after_settings(self, message: str | None) -> None:
            if message:
                self.log_line(message)
                self.cfg.reload()
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

        def on_data_table_row_selected(self, _event: Any) -> None:
            """Enter. DataTable consumes the key itself, so an App-level
            Binding("enter", ...) never fires - this event is the hook."""
            self.action_open_role()

        def action_open_role(self) -> None:
            row = self.selected_row()
            if row is None:
                return
            self.push_screen(RoleScreen(self.cfg, str(_get(row, "job_id", ""))))

        # Kept so the old name still works from tests and any muscle memory.
        def action_open_outreach(self) -> None:
            self.action_open_role()

        def action_open_url(self) -> None:
            import webbrowser

            row = self.selected_row()
            url = str(_get(row, "url", "")) if row is not None else ""
            if not url:
                self.log_line("[yellow]no URL on this role[/]")
                return
            webbrowser.open(url)
            self.log_line(f"opened [dim]{url}[/]")

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

    class SettingsScreen(ModalScreen):  # type: ignore[misc]
        """Edit the settings that decide what the search looks for.

        Writes back into config.local.toml in place, so its comments survive.
        """

        BINDINGS = [
            Binding("escape", "cancel", "Cancel"),
            Binding("ctrl+s", "save", "Save"),
            Binding("ctrl+e", "open_strategy", "Edit strategy doc"),
        ]

        def __init__(self, cfg: Config) -> None:
            super().__init__()
            self.cfg = cfg

        def compose(self) -> ComposeResult:
            from .settings import SETTINGS, WEIGHT_KEYS, current_values

            values = current_values(self.cfg.raw)
            with VerticalScroll(id="settings"):
                yield Static(
                    "[b]Settings[/]  [dim]ctrl+s save · esc cancel · "
                    "ctrl+e open search-strategy.md[/]\n"
                    "[dim]Lists are comma separated. Judgement lives in the "
                    "strategy doc, which the scorer reads directly.[/]"
                )
                for spec in SETTINGS:
                    value = values.get(spec.key)
                    shown = ", ".join(value) if isinstance(value, list) else str(
                        "true" if value is True else "false" if value is False else value
                    )
                    yield Static(f"\n[b]{spec.label}[/]  [dim]{spec.help}[/]")
                    yield Input(value=shown, id=f"set-{spec.key}")
                yield Static("\n[b]Rubric weights[/]  [dim]must sum to 1.0[/]")
                for key in WEIGHT_KEYS:
                    yield Static(f"[dim]{key}[/]")
                    yield Input(value=str(values.get(f"weight_{key}", "")), id=f"w-{key}")
            yield Footer()

        def on_mount(self) -> None:
            from .settings import SETTINGS

            self.query_one(f"#set-{SETTINGS[0].key}", Input).focus()

        def action_open_strategy(self) -> None:
            import os
            import subprocess

            editor = os.environ.get("EDITOR") or "open"
            with self.app.suspend():
                subprocess.call([editor, str(self.cfg.search_strategy)])

        def action_save(self) -> None:
            from .settings import SETTINGS, WEIGHT_KEYS, SettingsError, apply_edits

            edits = {
                spec.key: self.query_one(f"#set-{spec.key}", Input).value
                for spec in SETTINGS
            }
            weights = {
                key: self.query_one(f"#w-{key}", Input).value for key in WEIGHT_KEYS
            }
            path = self.cfg.source
            if path is None:
                self.notify(
                    "this config was not loaded from a file, so it cannot be saved",
                    severity="error",
                )
                return
            try:
                updated = apply_edits(path.read_text(encoding="utf-8"), edits, weights)
            except SettingsError as exc:
                self.notify(str(exc), severity="error", timeout=8)
                return
            path.write_text(updated, encoding="utf-8")
            self.dismiss(f"settings saved → [dim]{path.name}[/]")

        def action_cancel(self) -> None:
            self.dismiss(None)

    class ScanScreen(ModalScreen):  # type: ignore[misc]
        """Choose which board tiers to sweep."""

        BINDINGS = [Binding("escape", "cancel", "Cancel")]

        def __init__(self, cfg: Config) -> None:
            super().__init__()
            self.options = tier_options(cfg)

        def compose(self) -> ComposeResult:
            with Vertical(id="picker"):
                yield Static("Scan which boards?")
                yield OptionList(*[label for label, _ in self.options], id="tiers")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#tiers", OptionList).focus()

        def on_option_list_option_selected(self, event: Any) -> None:
            self.dismiss(self.options[event.option_index])

        def action_cancel(self) -> None:
            self.dismiss(None)

    class AttachCvScreen(ModalScreen):  # type: ignore[misc]
        """Pick a CV already on disk, or type a path."""

        BINDINGS = [Binding("escape", "cancel", "Cancel")]

        def __init__(self, cfg: Config, job_id: str) -> None:
            super().__init__()
            self.cfg = cfg
            self.job_id = job_id
            self.files = candidate_cv_files(cfg)

        def compose(self) -> ComposeResult:
            with Vertical(id="picker"):
                yield Static("Attach a CV  [dim]— newest first, or type a path below[/]")
                yield OptionList(*[f.name for f in self.files] or ["(none found)"], id="cvs")
                yield Input(placeholder="…or a full path, then Enter", id="cvpath")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#cvs", OptionList).focus()

        def _attach(self, path: Any) -> None:
            try:
                self.dismiss(attach_cv_blocking(self.cfg, self.job_id, path))
            except Exception as exc:  # noqa: BLE001 - shown on the role page
                self.dismiss(f"[red]{type(exc).__name__}:[/] {exc}")

        def on_option_list_option_selected(self, event: Any) -> None:
            if not self.files:
                self.dismiss(None)
                return
            self._attach(self.files[event.option_index])

        def on_input_submitted(self, event: Any) -> None:
            if event.value.strip():
                self._attach(event.value.strip())

        def action_cancel(self) -> None:
            self.dismiss(None)

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

    class RoleScreen(ModalScreen):  # type: ignore[misc]
        """Everything known about one role, and the stage keys that change it.

        A modal's bindings shadow the app's, so without these the page would
        say "press s" while s did nothing.
        """

        BINDINGS = [
            Binding("escape,q", "dismiss_screen", "Back"),
            Binding("s", "stage('score')", "Score"),
            Binding("t", "stage('tailor')", "Tailor"),
            Binding("o", "stage('outreach')", "Outreach"),
            Binding("v", "stage('verify')", "Verify"),
            Binding("w", "open_url", "Open posting"),
            Binding("c", "attach_cv", "Attach CV"),
        ]

        def __init__(self, cfg: Config, job_id: str) -> None:
            super().__init__()
            self.cfg = cfg
            self.job_id = job_id
            self.busy = False

        def compose(self) -> ComposeResult:
            yield Static("", id="rolestatus")
            with VerticalScroll(id="outreach"):
                # Dynamic values are escaped by role_detail_markup, so markup
                # is safe to leave on here.
                yield Static(role_detail_markup(self.cfg, self.job_id), id="rolebody")
            yield Footer()

        def refresh_body(self) -> None:
            self.query_one("#rolebody", Static).update(
                role_detail_markup(self.cfg, self.job_id)
            )

        def action_stage(self, stage: str) -> None:
            if self.busy:
                return
            self.busy = True
            self.query_one("#rolestatus", Static).update(f"  [yellow]running {stage}…[/]")
            self.run_stage(stage)

        @work(thread=True, exclusive=True)
        def run_stage(self, stage: str) -> None:
            try:
                message = run_stage_blocking(self.cfg, stage, self.job_id, dry_run=dry_run)
            except Exception as exc:  # noqa: BLE001 - shown on the page
                message = f"[red]{type(exc).__name__}:[/] {exc}"
            # call_from_thread lives on the App, not on a Screen.
            self.app.call_from_thread(self.finish_stage, message)

        def finish_stage(self, message: str) -> None:
            self.busy = False
            self.query_one("#rolestatus", Static).update(f"  {message}")
            self.refresh_body()
            self.app.action_refresh_rows()

        def action_dismiss_screen(self) -> None:
            self.dismiss()

        def action_attach_cv(self) -> None:
            self.app.push_screen(AttachCvScreen(self.cfg, self.job_id), self.after_attach)

        def after_attach(self, message: str | None) -> None:
            if message:
                self.query_one("#rolestatus", Static).update(f"  {message}")
                self.refresh_body()

        def action_open_url(self) -> None:
            import webbrowser

            from .tracker import Tracker

            with Tracker.from_config(self.cfg) as tracker:
                url = str(tracker.get_job(tracker.resolve_job_id(self.job_id))["url"] or "")
            if url:
                webbrowser.open(url)

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
