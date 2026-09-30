"""``jobsearch`` command-line interface.

Every stage is independently invocable and chainable: ``discover`` writes to the
tracker, ``score`` / ``tailor`` / ``outreach`` read a job id back out of it, and
``run`` chains them agentically. A global ``--dry-run`` shows what each stage
would do without calling the API or writing anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import Config, ConfigError, load_config
from .models import JobPosting, Status

log = logging.getLogger("jobsearch")


class CLIError(RuntimeError):
    """An error to report to the user without a traceback."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(cfg: Config, args: argparse.Namespace):
    from .claude import make_client

    client = make_client(cfg, dry_run=args.dry_run)
    if args.dry_run:
        client.dry_run_hook = lambda stage, prompt: print(
            f"  [dry-run] would call Claude ({client.model}) for stage {stage!r}"
        )
    return client


def _tracker(cfg: Config):
    from .tracker import Tracker

    return Tracker.from_config(cfg)


def _print_usage(client) -> None:
    usage = client.last_usage
    if usage.input_tokens or usage.cache_read_input_tokens:
        note = "cache HIT" if usage.cache_hit else "cache miss (first call warms it)"
        print(f"  tokens: {usage.describe()}  [{note}]")
    # The headless-CLI provider reports what the same tokens would have cost on
    # the metered API. Nothing is billed, so it is shown as avoided, not spent.
    avoided = getattr(client, "last_cost_usd", 0.0)
    if avoided:
        print(f"  ${avoided:.4f} of API spend avoided (ran on your subscription)")


def _fmt_score(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "  - "


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------


def cmd_discover(cfg: Config, args: argparse.Namespace) -> int:
    from .discover import discover

    companies = args.company or None
    tiers = [int(t) for t in args.tier] if args.tier else None

    if args.dry_run:
        boards = [
            b
            for b in cfg.boards
            if (not companies or b.company.lower() in {c.lower() for c in companies})
            and (not tiers or b.tier in tiers)
        ]
        print(f"[dry-run] would query {len(boards)} public job board(s):")
        from .discover import board_url

        for board in boards:
            print(f"  {board.company:<18} {board.ats:<11} {board_url(board)}")
        return 0

    report = discover(cfg, companies=companies, tiers=tiers)
    print(
        f"Checked {report.boards_checked} board(s): {report.raw_count} postings, "
        f"{len(report.postings)} matched the title and location filters."
    )
    if report.location_dropped:
        print(f"  {len(report.location_dropped)} title match(es) skipped as outside NL / remote-EU.")
    for error in report.errors:
        print(f"  ! {error}")

    if args.web:
        from .discover import web_search_discover

        client = _client(cfg, args)
        query = args.web if isinstance(args.web, str) else (
            "Head of AI / Director of AI leadership roles, Netherlands or remote-EU, "
            "Series B-D AI companies, opened in the last 30 days"
        )
        leads = web_search_discover(cfg, client, query)
        print(f"web_search returned {len(leads)} additional lead(s).")
        report.postings.extend(leads)

    tracker = _tracker(cfg)
    try:
        new = 0
        for posting in report.postings:
            if tracker.get_job(posting.job_id) is None:
                new += 1
            tracker.upsert_job(posting)
            print(f"  {posting.summary_line()}")
        # Refresh last_seen_at for everything the boards returned, including
        # postings the title filter dropped. Otherwise tightening the filters
        # makes live roles look delisted.
        refreshed = tracker.mark_seen(getattr(report, "seen_job_ids", ()) or ())
        tracker.mark_seen_urls(getattr(report, "seen_urls", ()) or ())
        print(f"\n{new} new posting(s) added to {cfg.db_path}")
        if refreshed:
            print(f"{refreshed} tracked posting(s) confirmed still listed")
        from .recheck import recheck_unswept
        from .stale import note_in_flight_delistings

        checked = recheck_unswept(tracker, cfg)
        if checked.listed or checked.gone or checked.unknown:
            print(f"{len(checked.listed)} hand-added role(s) still listed at their own URL"
                  + (f", {len(checked.gone)} gone" if checked.gone else "")
                  + (f", {len(checked.unknown)} could not be checked" if checked.unknown else ""))

        for row in note_in_flight_delistings(tracker):
            print(f"  ! delisted after you applied: {row.company} - {row.title} ({row.status}); note added")
    finally:
        tracker.close()
    return 0


# ---------------------------------------------------------------------------
# add (manual entry, for postings behind a site we will not scrape)
# ---------------------------------------------------------------------------


def cmd_add(cfg: Config, args: argparse.Namespace) -> int:
    from .discover import DiscoveryError, Fetcher, clean_posting_url, fetch_single_posting

    description = ""
    if args.file:
        description = Path(args.file).read_text(encoding="utf-8")
    elif args.stdin:
        description = sys.stdin.read()
    url = clean_posting_url(args.url) if args.url else ""

    # With a URL, read the posting page and let the flags override what it
    # says. Company, title and every listed location come from the page when
    # the flags leave them out.
    fetched = None
    needs_page = url and not (args.company and args.title and args.location and description)
    if needs_page and args.dry_run:
        print(f"[dry-run] would read {url} for company, title, locations and description")
        if not (args.company and args.title):
            return 0
        needs_page = False
    if needs_page:
        try:
            fetched = fetch_single_posting(url, Fetcher.from_config(cfg), cfg)
        except DiscoveryError as exc:
            if not (args.company and args.title):
                print(f"Could not read the posting: {exc}", file=sys.stderr)
                print("Pass --company and --title, and the text via --file or --stdin.", file=sys.stderr)
                return 1
            print(f"warning: could not read the posting page ({exc}); using the flags only", file=sys.stderr)
    if not fetched and not (args.company and args.title):
        print("Pass --company and --title, or a --url the tool can read.", file=sys.stderr)
        return 1

    posting = JobPosting(
        company=args.company or fetched.company,
        title=args.title or fetched.title,
        url=url,
        source="manual" if not fetched else fetched.source,
        location=args.location or (fetched.location if fetched else ""),
        description=description or (fetched.description if fetched else ""),
        salary_text=args.salary or (fetched.salary_text if fetched else ""),
    )
    if fetched and not args.location:
        print(f"location from the page: {posting.location or 'none stated'}")
    board = cfg.board_for(posting.company)
    if board:
        posting.board_tier = board.tier
        posting.ind_sponsor = board.sponsor_state
        posting.gaming = board.gaming

    if args.dry_run:
        print(f"[dry-run] would add {posting.summary_line()}")
        return 0

    tracker = _tracker(cfg)
    try:
        tracker.upsert_job(posting)
    finally:
        tracker.close()
    print(f"Added {posting.job_id}")
    print(f"Next: jobsearch score {posting.job_id}")
    return 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def cmd_score(cfg: Config, args: argparse.Namespace) -> int:
    from .scoring import DIMENSION_LABELS, score_posting

    tracker = _tracker(cfg)
    client = _client(cfg, args)
    try:
        job_id = tracker.resolve_job_id(args.job_id)
        posting = tracker.get_posting(job_id)
        report = score_posting(posting, cfg, client)

        print(f"{posting.company} - {posting.title}")
        print(f"  {posting.location or 'location not stated'}  |  {posting.url}")
        print()
        print("HARD CONSTRAINTS (applied before scoring):")
        for result in report.constraints.results:
            mark = {"pass": "ok  ", "fail": "FAIL", "unknown": "?   "}[result.verdict.value]
            print(f"  [{mark}] {result.name}: {result.reason}")
            if result.evidence:
                print(f"         evidence: {result.evidence}")
            if result.advisory:
                print(f"         CHECK: {result.advisory}")

        if report.eliminated:
            print()
            print("ELIMINATED by a hard constraint. Not scored, not tailored.")
            if not args.dry_run:
                tracker.save_score(report)
            return 0

        print()
        print("WEIGHTED SCORE:")
        for dim in report.dimensions:
            print(
                f"  {DIMENSION_LABELS[dim.name]:<42} {dim.score:.0f}/5  "
                f"x{dim.weight:.0%} = {dim.contribution:.2f}"
            )
            if dim.reasoning:
                for line in _wrap(dim.reasoning, 74):
                    print(f"      {line}")
            for item in dim.evidence:
                print(f"      - {item}")
        print(f"  {'TOTAL':<42} {report.weighted:.2f}/5")
        threshold = float(cfg.get("scoring", "shortlist_threshold", 3.5))
        verdict = "SHORTLIST" if (report.weighted or 0) >= threshold else "below threshold"
        print(f"  recommendation: {report.recommendation}  ({verdict}, cutoff {threshold})")
        if report.notes:
            print()
            for line in report.notes.splitlines():
                for wrapped in _wrap(line, 76):
                    print(f"  {wrapped}")
        _print_usage(client)

        if not args.dry_run:
            tracker.save_score(report)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
    finally:
        tracker.close()
    return 0


# ---------------------------------------------------------------------------
# tailor
# ---------------------------------------------------------------------------


def cmd_tailor(cfg: Config, args: argparse.Namespace) -> int:
    from .ats import verify_from_config
    from .tailor import format_claim_report, tailor_cv

    tracker = _tracker(cfg)
    client = _client(cfg, args)
    try:
        job_id = tracker.resolve_job_id(args.job_id)
        posting = tracker.get_posting(job_id)
        print(f"Tailoring for {posting.company} - {posting.title}")

        from .tui import load_critiques, load_notes

        prior = [] if args.fresh else load_critiques(tracker.get_job(job_id))
        if prior:
            print(f"  addressing {len(prior)} finding(s) from the previous version")
        notes = load_notes(tracker, job_id)
        if notes:
            print(f"  using {len(notes)} note(s) you wrote on this role")

        on_delta = (lambda text: print(text, end="", flush=True)) if args.stream else None
        result = tailor_cv(
            posting,
            cfg,
            client,
            prior_critiques=prior,
            notes=notes,
            render=not args.no_render,
            verify_claims=not args.no_verify_claims,
            on_delta=on_delta,
        )
        if args.stream:
            print()

        print(f"  headline: {result.headline}")
        print(f"  html: {result.html_path}")
        if result.pdf_path:
            print(f"  pdf:  {result.pdf_path}")
        _print_usage(client)

        print()
        print(format_claim_report(result))

        ats_payload = None
        if result.pdf_path and not args.dry_run and not args.no_render:
            print()
            report = verify_from_config(cfg, result.pdf_path, jd_text=posting.description)
            print(report.render())
            ats_payload = report.to_dict()
            if not report.passed:
                print()
                print("The rendered PDF FAILED ATS verification. Fix before sending.")

        if not args.dry_run:
            tracker.save_cv(
                job_id,
                result.html_path,
                result.pdf_path,
                ats_payload,
                [c.to_dict() for c in result.claims],
                [c.to_dict() for c in result.critiques],
            )

        if result.critiques:
            print(f"\n  adversarial review: {len(result.critiques)} finding(s)")
            for c in result.critiques:
                print(f"    [{c.severity}] {c.issue}")
                if c.quote:
                    print(f"        \"{c.quote[:100]}\"")
                if c.fix:
                    print(f"        fix: {c.fix}")
        if result.prior_addressed:
            # Counted against what the final pass was actually shown: on a
            # re-tailor that is the prior findings plus the first pass's own.
            fed, stayed = result.prior_addressed, result.unchanged
            print(
                f"\n  prior findings: {fed - len(stayed)} of {fed} "
                f"rewritten, {len(stayed)} quote(s) unchanged"
            )
            for c in stayed:
                print(f"    [{c.severity}] still says: \"{c.quote[:100]}\"")
        if result.ungrounded:
            print()
            print("Review the ungrounded claims above before sending this CV.")
            return 2
    finally:
        tracker.close()
    return 0


# ---------------------------------------------------------------------------
# verify (standalone)
# ---------------------------------------------------------------------------


def cmd_letter(cfg: Config, args: argparse.Namespace) -> int:
    from .letter import write_letter

    tracker = _tracker(cfg)
    client = _client(cfg, args)
    try:
        job_id = tracker.resolve_job_id(args.job_id)
        posting = tracker.get_posting(job_id)
        result = write_letter(
            posting, cfg, client,
            verify_claims=not args.no_verify_claims,
            instruction=args.instruction or "",
        )
        print(result.text)
        print(f"\n  {result.word_count} words -> {result.path}")
        if result.critiques:
            print(f"\n  adversarial review: {len(result.critiques)} finding(s)")
            for c in result.critiques:
                print(f"    [{c.severity}] {c.issue}")
                if c.quote:
                    print(f"        \"{c.quote[:100]}\"")
                if c.fix:
                    print(f"        fix: {c.fix}")
        if result.ungrounded:
            print(f"\n  {len(result.ungrounded)} ungrounded claim(s):")
            for claim in result.ungrounded:
                print(f"    - {claim.text}")
            print("  Review these before sending.")
        if not args.dry_run:
            tracker.save_letter(job_id, result.path, [c.to_dict() for c in result.claims])
        _print_usage(client)
    finally:
        tracker.close()
    return 0


def cmd_verify(cfg: Config, args: argparse.Namespace) -> int:
    from .ats import verify_from_config

    jd_text = ""
    if args.jd_file:
        jd_text = Path(args.jd_file).read_text(encoding="utf-8")
    elif args.job_id:
        tracker = _tracker(cfg)
        try:
            jd_text = tracker.get_posting(tracker.resolve_job_id(args.job_id)).description
        finally:
            tracker.close()

    report = verify_from_config(cfg, args.pdf, jd_text=jd_text)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())
    return 0 if report.passed else 1


# ---------------------------------------------------------------------------
# outreach
# ---------------------------------------------------------------------------


def cmd_outreach(cfg: Config, args: argparse.Namespace) -> int:
    from .outreach import draft_outreach, format_draft

    tracker = _tracker(cfg)
    client = _client(cfg, args)
    try:
        job_id = tracker.resolve_job_id(args.job_id)
        posting = tracker.get_posting(job_id)
        draft = draft_outreach(posting, cfg, client)
        print(f"{posting.company} - {posting.title}")
        print()
        print(format_draft(draft, posting))
        _print_usage(client)

        if not args.dry_run:
            tracker.save_outreach(draft)
            out = cfg.ensure_output_dir() / "outreach"
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{job_id}.json"
            path.write_text(json.dumps(draft.to_dict(), indent=2), encoding="utf-8")
            print(f"\nSaved to {path}")

        if args.gmail_draft:
            _gmail_draft(cfg, args, draft, posting, tracker)
    finally:
        tracker.close()
    return 0


def _gmail_draft(cfg, args, draft, posting, tracker) -> None:
    from .sync import GoogleSync, GoogleSyncError

    if not args.to:
        raise CLIError("--gmail-draft needs --to <address>")
    try:
        sync = GoogleSync.from_config(cfg, dry_run=args.dry_run)
        row = tracker.get_job(draft.job_id)
        attachments = [Path(row["cv_pdf_path"])] if row and row["cv_pdf_path"] else []
        draft_id = sync.create_draft(
            args.to, draft.email_subject, draft.email_body, attachments
        )
        print(f"Gmail DRAFT created (id {draft_id}). Nothing was sent.")
    except GoogleSyncError as exc:
        raise CLIError(str(exc)) from exc


# ---------------------------------------------------------------------------
# track / status
# ---------------------------------------------------------------------------


def cmd_stale(cfg: Config, args: argparse.Namespace) -> int:
    """Report roles a board has stopped returning. Reports only, changes nothing."""
    from .stale import find_stale

    tracker = _tracker(cfg)
    try:
        report = find_stale(
            tracker.list_jobs(limit=100000), grace_hours=args.grace_hours
        )
        print(report.render())
        from .stale import in_flight_delisted

        gone = in_flight_delisted(tracker.list_jobs(limit=100000), grace_hours=args.grace_hours)
        if gone:
            print()
            print(f"IN FLIGHT, NO LONGER LISTED ({len(gone)}): probably filled or paused; status left as is")
            for row in sorted(gone, key=lambda r: (r.company, r.title)):
                print(f"  [{row.days_unseen:>3}d unseen] {row.status:<15} {row.company} - {row.title[:52]}")
        if report.delisted:
            print()
            print(
                "Nothing has been changed. To close one: "
                "jobsearch track <job-id> --status Withdrawn --reason 'no longer listed'"
            )
    finally:
        tracker.close()
    return 0


def cmd_track(cfg: Config, args: argparse.Namespace) -> int:
    tracker = _tracker(cfg)
    try:
        job_id = tracker.resolve_job_id(args.job_id)
        if args.dry_run:
            print(f"[dry-run] would update {job_id}")
            return 0
        if args.status:
            new = tracker.set_status(job_id, Status.parse(args.status), reason=args.reason or "")
            print(f"{job_id}: status -> {new.value}")
        fields = {
            k: v
            for k, v in {
                "next_action": args.next_action,
                "due": args.due,
                "warm_path": args.warm_path,
            }.items()
            if v
        }
        if fields:
            tracker.set_fields(job_id, **fields)
            print(f"{job_id}: {', '.join(fields)} updated")
        if args.note:
            tracker.add_note(job_id, args.note)
            print(f"{job_id}: note added")
        if args.history:
            print("\nStatus history:")
            for row in tracker.history(job_id):
                arrow = f"{row['from_status'] or 'new'} -> {row['to_status']}"
                print(f"  {row['changed_at']}  {arrow}  {row['reason'] or ''}")
    finally:
        tracker.close()
    return 0


def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    tracker = _tracker(cfg)
    try:
        status = Status.parse(args.status) if args.status else None
        jobs = tracker.list_jobs(
            status=status, company=args.company, min_score=args.min_score, limit=args.limit
        )
        if args.json:
            print(json.dumps([dict(j) for j in jobs], indent=2, default=str))
            return 0

        stats = tracker.stats()
        print(
            f"{stats['total']} tracked, {stats['scored']} scored, "
            f"{stats['active']} active. Average score "
            f"{stats['avg_score'] if stats['avg_score'] is not None else '-'}."
        )
        print()
        # Size the id column to the widest id actually present, so long
        # generated ids do not shear the table.
        width = max([len(j["job_id"]) for j in jobs] + [len("JOB ID")])
        header = f"{'JOB ID':<{width}} {'SCORE':>6} {'STATUS':<16} COMPANY - TITLE"
        print(header)
        print("-" * min(len(header), 120))
        for job in jobs:
            flag = "" if job["constraints_ok"] in (1, None) else " [constraint FAIL]"
            print(
                f"{job['job_id']:<{width}} {_fmt_score(job['score_weighted']):>6} "
                f"{(job['status'] or ''):<16} {job['company']} - {job['title']}{flag}"
            )
        if not jobs:
            print("(nothing tracked yet - run `jobsearch discover`)")
    finally:
        tracker.close()
    return 0


def _age(ts: str | None) -> str:
    """'12d ago' for an ISO timestamp, '-' for none."""
    if not ts:
        return "-"
    try:
        then = datetime.fromisoformat(ts)
    except ValueError:
        return ts
    days = (datetime.now(timezone.utc) - then).days
    return "today" if days <= 0 else f"{days}d ago"


def cmd_recruiter(cfg: Config, args: argparse.Namespace) -> int:
    """Search firms and recruiters: who covers what, when you last spoke, what they pitched."""
    from .tracker import TrackerError

    tracker = _tracker(cfg)
    try:
        action = args.recruiter_action
        if action == "add":
            rid = tracker.add_recruiter(
                args.firm, name=args.name, specialism=args.specialism, email=args.email,
                phone=args.phone, profile_url=args.url, notes=args.notes,
                next_action=args.next, due=args.due,
            )
            print(f"Added recruiter {rid}: {args.name or '(no name)'} at {args.firm}")
            return 0
        if action == "list":
            rows = tracker.list_recruiters(status=args.status, cold_days=args.cold)
            if args.json:
                print(json.dumps([dict(r) for r in rows], indent=2, default=str))
                return 0
            if not rows:
                print("No recruiters" + (" going cold." if args.cold is not None else " yet. Add one with: jobsearch recruiter add FIRM --name NAME"))
                return 0
            print(f"{'ID':>3}  {'FIRM':<24} {'NAME':<22} {'STATUS':<10} {'LAST CONTACT':<13} NEXT")
            for r in rows:
                nxt = " ".join(x for x in (r["next_action"] or "", f"(due {r['due']})" if r["due"] else "") if x)
                print(f"{r['id']:>3}  {r['firm'][:24]:<24} {(r['name'] or '')[:22]:<22} {r['status']:<10} "
                      f"{_age(r['last_contact']):<13} {nxt}")
            return 0
        rid = args.id
        if action == "show":
            r = tracker.get_recruiter(rid)
            print(f"{r['name'] or '(no name)'} - {r['firm']}   [{r['status']}]")
            for label, key in (("specialism", "specialism"), ("email", "email"), ("phone", "phone"),
                               ("profile", "profile_url"), ("next", "next_action"), ("due", "due"),
                               ("notes", "notes")):
                if r[key]:
                    print(f"  {label + ':':<12}{r[key]}")
            print(f"  {'last:':<12}{_age(r['last_contact'])}")
            events = tracker.recruiter_events(rid)
            if events:
                print()
                for e in events:
                    role = " - ".join(x for x in (e["company"], e["title"]) if x)
                    job = f" [{e['job_id']}]" if e["job_id"] else ""
                    text = "  ".join(x for x in (role + job, e["body"] or "") if x)
                    print(f"  {e['created_at'][:10]}  {e['kind']:<8} {text}")
            return 0
        if action == "log":
            tracker.log_recruiter(rid, args.kind, args.body or "", status=args.status,
                                  next_action=args.next, due=args.due)
            print(f"Logged {args.kind} for recruiter {rid}")
            return 0
        if action == "pitch":
            tracker.log_recruiter(rid, "pitch", args.note or "", company=args.company,
                                  title=args.title, job_id=args.job, status=args.status,
                                  next_action=args.next, due=args.due)
            print(f"Logged pitch from recruiter {rid}: {args.company} - {args.title}")
            return 0
        raise TrackerError(f"Unknown recruiter action {action!r}")
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        tracker.close()


def cmd_tidy_outputs(cfg: Config, args: argparse.Namespace) -> int:
    """Move each role's CV and letter into output/applications/<Company>/<Role - id>/."""
    import shutil

    from .letter import letter_path_for
    from .tailor import cv_paths

    tracker = _tracker(cfg)
    moved = skipped = missing = kept = 0
    try:
        for row in tracker.list_jobs():
            job_id = row["job_id"]
            if not (row["cv_pdf_path"] or row["cv_html_path"] or row["letter_path"]):
                continue
            posting = tracker.get_posting(job_id)
            html_new, pdf_new = cv_paths(cfg, posting)
            letter_new = html_new.parent / letter_path_for(cfg, posting).name
            updates: dict[str, str] = {}
            for field, new in (("cv_html_path", html_new), ("cv_pdf_path", pdf_new), ("letter_path", letter_new)):
                old = row[field]
                if not old:
                    continue
                old_path = Path(old)
                if not old_path.is_absolute():
                    old_path = cfg.root / old_path
                if old_path.resolve() == new.resolve():
                    continue
                # Only files the tool wrote. A CV attached from elsewhere (the base
                # CV, one tailored by hand) stays where it is.
                if cfg.output_dir.resolve() not in old_path.resolve().parents:
                    kept += 1
                    print(f"  kept     {job_id}: {old_path.name} (outside the output folder)")
                    continue
                if not old_path.is_file():
                    missing += 1
                    print(f"  missing  {job_id}: {old}")
                    continue
                if new.exists():
                    skipped += 1
                    print(f"  exists   {new}")
                    continue
                print(f"  {'would move' if args.dry_run else 'move'}  {old_path.name}  ->  {new.relative_to(cfg.output_dir)}")
                if not args.dry_run:
                    new.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old_path), str(new))
                    updates[field] = str(new)
                moved += 1
            if updates:
                tracker.set_output_paths(job_id, **updates)
        verb = "would move" if args.dry_run else "moved"
        print(f"{verb} {moved} file(s); {kept} kept outside the output folder; "
              f"{skipped} already in place; {missing} missing on disk")
        return 0
    finally:
        tracker.close()


def cmd_show(cfg: Config, args: argparse.Namespace) -> int:
    tracker = _tracker(cfg)
    try:
        job_id = tracker.resolve_job_id(args.job_id)
        row = tracker.get_job(job_id)
        print(f"{row['company']} - {row['title']}")
        print(f"  id:        {row['job_id']}")
        print(f"  location:  {row['location']}")
        print(f"  url:       {row['url']}")
        print(f"  status:    {row['status']}")
        print(f"  score:     {_fmt_score(row['score_weighted'])}  ({row['recommendation'] or '-'})")
        print(f"  cv:        {row['cv_pdf_path'] or '-'}")
        if row["score_json"]:
            data = json.loads(row["score_json"])
            print("  dimensions:")
            for dim in data.get("dimensions", []):
                print(f"    {dim['name']:<10} {dim['score']}/5  {dim['reasoning'][:90]}")
        contacts = tracker.contacts(job_id)
        if contacts:
            print("  contacts:")
            for contact in contacts:
                print(f"    {contact['name'] or contact['title']}: {contact['search_url']}")
        draft = tracker.latest_outreach(job_id)
        if draft:
            print(f"  outreach:  drafted {draft['created_at'][:10]}"
                  f"{'  (marked sent)' if draft['sent'] else '  (nothing sent)'}")
            for label, key in (
                ("connection note", "connection_note"),
                ("linkedin message", "linkedin_message"),
            ):
                body = (draft[key] or "").strip()
                if body:
                    print(f"    {label}:")
                    for line in body.splitlines():
                        print(f"      {line}")
            if (draft["email_subject"] or "").strip():
                print(f"    email subject: {draft['email_subject']}")
            body = (draft["email_body"] or "").strip()
            if body:
                print("    email body:")
                for line in body.splitlines():
                    print(f"      {line}")
        notes = tracker.notes(job_id)
        if notes:
            print("  notes:")
            for note in notes:
                print(f"    {note['created_at'][:10]}  {note['body']}")
    finally:
        tracker.close()
    return 0


def cmd_export(cfg: Config, args: argparse.Namespace) -> int:
    from .tracker.export import export_xlsx

    out = Path(args.out) if args.out else cfg.ensure_output_dir() / "job-search-tracker-export.xlsx"
    if args.dry_run:
        print(f"[dry-run] would export the tracker to {out}")
        return 0
    tracker = _tracker(cfg)
    try:
        result = export_xlsx(
            tracker,
            out,
            template_xlsx=cfg.tracker_xlsx if cfg.tracker_xlsx.is_file() else None,
            protect_path=cfg.tracker_xlsx,
        )
    finally:
        tracker.close()
    print(f"Exported {result.describe()}")
    print("Your own job-search-tracker.xlsx was not touched.")
    return 0


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


def cmd_sync(cfg: Config, args: argparse.Namespace) -> int:
    from .sync import GoogleSync, GoogleSyncError
    from .tracker.export import PIPELINE_COLUMNS

    try:
        sync = GoogleSync.from_config(cfg, dry_run=args.dry_run)
    except GoogleSyncError as exc:
        raise CLIError(str(exc)) from exc

    tracker = _tracker(cfg)
    try:
        if args.drive:
            uploaded = 0
            for job in tracker.list_jobs():
                pdf = job["cv_pdf_path"]
                if pdf and Path(pdf).is_file():
                    link = sync.upload_cv(pdf)
                    print(f"  uploaded {Path(pdf).name} -> {link}")
                    uploaded += 1
            print(f"Uploaded {uploaded} CV(s) to Drive folder {sync.drive_folder_name!r}")

        if args.sheets:
            from .tracker.export import _pipeline_row  # noqa: PLC2701 - internal by design

            rows = [_pipeline_row(job) for job in tracker.list_jobs()]
            url = sync.mirror_to_sheet(PIPELINE_COLUMNS, rows)
            print(f"Mirrored {len(rows)} row(s) to {url}")

        if args.replies:
            replies = sync.find_replies(args.query)
            print(f"{len(replies)} recent message(s) matching {args.query!r}:")
            for message in replies:
                print(f"  {message['date'][:16]}  {message['from']}")
                print(f"      {message['subject']}")
            print("\nStatus is NOT auto-updated from these. Use `jobsearch track`.")
    except GoogleSyncError as exc:
        raise CLIError(str(exc)) from exc
    finally:
        tracker.close()
    return 0


# ---------------------------------------------------------------------------
# run (agentic)
# ---------------------------------------------------------------------------


def cmd_attach_cv(cfg: Config, args: argparse.Namespace) -> int:
    """Record a CV written outside the tool against a role."""
    from .tui import attach_cv_blocking

    message = attach_cv_blocking(cfg, args.job_id, args.path, verify=not args.no_verify)
    print(re.sub(r"\[/?[a-z]*\]", "", message))
    return 0


def cmd_setup(cfg: Config | None, args: argparse.Namespace) -> int:
    """Guided first run. Deliberately does not require an existing config."""
    from .setup_wizard import SetupAborted, default_repo_root, run_setup

    root = Path(args.config).expanduser().resolve().parent if args.config else default_repo_root()
    try:
        return run_setup(root, force=args.force)
    except SetupAborted as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def cmd_tui(cfg: Config, args: argparse.Namespace) -> int:
    from .tui import run_tui

    try:
        return run_tui(cfg, dry_run=args.dry_run)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def cmd_doctor(cfg: Config, args: argparse.Namespace) -> int:
    """Check that everything the tool depends on is actually present."""
    import os
    import shutil

    ok = True

    def check(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"  [{'ok  ' if good else 'FAIL'}] {label}{': ' + detail if detail else ''}")

    print(f"config: {cfg.source or cfg.root}")
    print("source documents:")
    for label, path in [
        ("dossier", cfg.dossier),
        ("base CV", cfg.base_cv),
        ("search strategy", cfg.search_strategy),
        ("target companies", cfg.target_companies),
        ("tracker xlsx", cfg.tracker_xlsx),
    ]:
        check(label, path.is_file(), str(path))

    print("rendering:")
    from .render import RenderError, find_chrome

    try:
        binary = find_chrome(cfg.get("render", "chrome_binary"))
        check("chrome", True, binary)
    except RenderError as exc:
        check("chrome", False, str(exc))

    print("api:")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    has_cli = bool(shutil.which("ant"))
    check(
        "Anthropic credentials",
        has_key or has_cli,
        "ANTHROPIC_API_KEY set" if has_key else ("`ant` CLI found" if has_cli else "set ANTHROPIC_API_KEY or run `ant auth login`"),
    )

    from .systemone import SystemOneSettings

    so = SystemOneSettings.from_config(cfg)
    if so.enabled:
        try:
            import typesafe_sdk  # noqa: F401

            has_sdk = True
        except ImportError:
            has_sdk = False
        check(
            "System One (Jev) reader",
            bool(so.api_key) and has_sdk,
            f"{so.api_key_env} set, typesafe-sdk installed" if (so.api_key and has_sdk)
            else (f"set {so.api_key_env}" if not so.api_key else "pip install -e '.[systemone]'"),
        )
    else:
        print("  [ok  ] System One (Jev) reader: disabled (constraints use the phrase lists only)")
    print("scoring:")
    check("weights sum to 1.0", True, str(cfg.weights.as_dict()))
    print("boards:")
    print(f"  {len(cfg.boards)} configured across tiers "
          f"{sorted({b.tier for b in cfg.boards})}")
    if args.boards:
        from .discover import Fetcher, board_url, fetch_board, moved_board_hint
        from .discover.sources import DiscoveryError

        fetcher = Fetcher.from_config(cfg)
        for board in cfg.boards:
            try:
                found = fetch_board(board, fetcher)
                if found:
                    check(f"  {board.company}", True, f"{len(found)} postings")
                else:
                    # A board that answers 200 with nothing is usually a stale
                    # token after an ATS migration, not a company with no jobs.
                    # Reporting it as ok hides that; it is the quietest way for
                    # a target to fall out of the sweep.
                    hint = moved_board_hint(board, fetcher)
                    check(
                        f"  {board.company}",
                        False,
                        "0 postings - the token answers but is empty"
                        + (hint or "; check whether they moved ATS"),
                    )
            except DiscoveryError as exc:
                hint = moved_board_hint(board, fetcher) if "HTTP 404" in str(exc) else ""
                check(f"  {board.company}", False, f"{exc}{hint or ' - fix the token in config.local.toml'}")
    print("google sync:")
    enabled = cfg.get("google", "enabled", False)
    print(f"  {'enabled' if enabled else 'disabled (everything else still works)'}")

    print()
    print("READY" if ok else "Some checks failed - see above.")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jobsearch",
        description=(
            "Agent harness for a personal senior-AI-leadership job search: "
            "discover, score, tailor, ATS-verify, draft outreach, and track."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical flow:\n"
            "  jobsearch doctor\n"
            "  jobsearch discover --tier 1\n"
            "  jobsearch status\n"
            "  jobsearch score <job-id>\n"
            "  jobsearch tailor <job-id>\n"
            "  jobsearch outreach <job-id>\n"
            "  jobsearch track <job-id> --status Applied\n"
            "  jobsearch export\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"jobsearch {__version__}")
    parser.add_argument("--config", help="Path to config.toml (default: nearest one)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen. No API calls, no writes.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    # discover
    p = sub.add_parser(
        "discover",
        help="Find postings on the configured public ATS boards",
        description=(
            "Queries Greenhouse, Lever and Ashby public job-board APIs for the "
            "companies in config.toml. Never scrapes LinkedIn or Indeed."
        ),
    )
    p.add_argument("--company", action="append", help="Limit to a company (repeatable)")
    p.add_argument("--tier", action="append", help="Limit to a target tier (repeatable)")
    p.add_argument(
        "--web",
        nargs="?",
        const=True,
        help="Also run a Claude web_search pass, optionally with a custom query",
    )
    p.set_defaults(func=cmd_discover)

    # add
    p = sub.add_parser(
        "add",
        help="Add a posting by hand (paste the text on stdin or via --file)",
        description=(
            "Add one role. With --url alone the tool reads the posting page "
            "(Greenhouse, Lever and Ashby through their APIs; anything else "
            "through its schema.org data, robots.txt permitting) and records "
            "every location it lists. For sites it will not read, paste the "
            "description on stdin or point --file at a text file."
        ),
    )
    p.add_argument("--company", help="Read from the page when --url is given")
    p.add_argument("--title", help="Read from the page when --url is given")
    p.add_argument("--url", help="Posting URL; tracking parameters are dropped")
    p.add_argument(
        "--location",
        help='Overrides the page. Several sites: "Utrecht, Netherlands; Copenhagen, Denmark"',
    )
    p.add_argument("--salary")
    p.add_argument("--file", help="File containing the job description text")
    p.add_argument(
        "--stdin", action="store_true", help="Read the job description from stdin"
    )
    p.set_defaults(func=cmd_add)

    # score
    p = sub.add_parser(
        "score",
        help="Score a job: hard constraints first, then the weighted rubric",
        description=(
            "Applies the visa / non-compete / compensation / location / travel "
            "filters, then scores the five weighted dimensions with per-dimension "
            "reasoning. A role failing a hard constraint is never sent to the model."
        ),
    )
    p.add_argument("job_id", help="Job id, unique prefix, or URL")
    p.add_argument("--json", action="store_true", help="Also print the raw JSON report")
    p.set_defaults(func=cmd_score)

    # tailor
    p = sub.add_parser(
        "tailor",
        help="Generate a role-tailored CV, render to PDF, and verify it",
        description=(
            "Tailors the base CV using the career dossier as the fact base, "
            "renders with headless Chrome, audits every claim for grounding, and "
            "runs the ATS verifier on the output PDF."
        ),
    )
    p.add_argument("job_id")
    p.add_argument("--no-render", action="store_true", help="Skip the PDF render")
    p.add_argument(
        "--no-verify-claims",
        action="store_true",
        help="Skip the grounding audit (not recommended)",
    )
    p.add_argument("--stream", action="store_true", help="Print the CV as it generates")
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore the previous version's adversarial findings instead of addressing them",
    )
    p.set_defaults(func=cmd_tailor)

    # verify
    p = sub.add_parser(
        "letter",
        help="Write a short, grounded cover letter for a role",
    )
    p.add_argument("job_id")
    p.add_argument("--instruction", help="Extra steer, e.g. 'lead on the FDE angle'")
    p.add_argument("--no-verify-claims", action="store_true", help="Skip the grounding audit")
    p.set_defaults(func=cmd_letter)

    p = sub.add_parser(
        "verify",
        help="Run the ATS verifier against any PDF",
        description=(
            "Checks page count, section headings, letter-spacing corruption, "
            "education date attachment, inline bullet markers, hyphenated keyword "
            "wraps, and job-description keyword coverage."
        ),
    )
    p.add_argument("pdf", help="Path to the PDF to verify")
    p.add_argument("--jd-file", help="Job description text file, for keyword coverage")
    p.add_argument("--job-id", help="Tracked job id, for keyword coverage")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_verify)

    # outreach
    p = sub.add_parser(
        "outreach",
        help="Infer likely contacts and draft LinkedIn / email outreach",
        description=(
            "Emits LinkedIn search URLs for you to click, plus drafted messages. "
            "Never scrapes LinkedIn. Never sends anything."
        ),
    )
    p.add_argument("job_id")
    p.add_argument(
        "--gmail-draft",
        action="store_true",
        help="Also create a Gmail DRAFT (requires Google setup). Never sends.",
    )
    p.add_argument("--to", help="Recipient address for --gmail-draft")
    p.set_defaults(func=cmd_outreach)

    # track
    p = sub.add_parser(
        "track", help="Update a tracked job: status, next action, notes"
    )
    p.add_argument("job_id")
    p.add_argument(
        "--status",
        help="New status: " + ", ".join(s.value for s in Status),
    )
    p.add_argument("--reason", help="Why the status changed (logged in history)")
    p.add_argument("--next-action")
    p.add_argument("--due", help="ISO date, e.g. 2026-09-15")
    p.add_argument("--warm-path")
    p.add_argument("--note")
    p.add_argument("--history", action="store_true", help="Print the status history")
    p.set_defaults(func=cmd_track)

    # stale
    p = sub.add_parser(
        "stale",
        help="Roles a board has stopped returning",
        description=(
            "Lists tracked roles that a board was swept for and did not return, so "
            "they are very likely delisted. Reports only; nothing is changed. Roles "
            "on boards not swept since last_seen_at was introduced are listed "
            "separately as unproven rather than assumed dead."
        ),
    )
    p.add_argument(
        "--grace-hours", type=int, default=36,
        help="How far behind its company's latest sighting a row may fall before "
             "it counts as delisted (default 36)",
    )
    p.set_defaults(func=cmd_stale)

    # status
    p = sub.add_parser("status", help="List tracked jobs")
    p.add_argument("--status", help="Filter by status")
    p.add_argument("--company")
    p.add_argument("--min-score", type=float)
    p.add_argument("--limit", type=int)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    # show
    p = sub.add_parser("show", help="Show everything known about one job")
    p.add_argument("job_id")
    p.set_defaults(func=cmd_show)

    # export
    p = sub.add_parser(
        "export",
        help="Export the tracker to xlsx in your existing tracker's shape",
        description=(
            "Writes a new xlsx to the output directory using the column shape of "
            "your job-search-tracker.xlsx. Your own file is never modified."
        ),
    )
    p.add_argument("--out", help="Output path (default: output/)")
    p.set_defaults(func=cmd_export)

    # sync
    p = sub.add_parser(
        "sync",
        help="Optional Gmail / Drive / Sheets sync (feature-flagged)",
        description=(
            "Requires the Google setup in the README. Gmail creates DRAFTS only - "
            "this codebase has no send path."
        ),
    )
    p.add_argument("--drive", action="store_true", help="Upload tailored CVs to Drive")
    p.add_argument("--sheets", action="store_true", help="Mirror the pipeline to Sheets")
    p.add_argument("--replies", action="store_true", help="List recent Gmail replies")
    p.add_argument("--query", default="newer_than:30d", help="Gmail search query")
    p.set_defaults(func=cmd_sync)

    # run
    
    # doctor
    p = sub.add_parser(
        "attach-cv",
        help="Point a role at a CV you already made",
        description=(
            "Record an existing CV file against a role, and verify it if it is "
            "a PDF. For CVs tailored by hand outside this tool."
        ),
    )
    p.add_argument("job_id")
    p.add_argument("path", help="Path to the .pdf or .html CV")
    p.add_argument("--no-verify", action="store_true", help="Skip the ATS check")
    p.set_defaults(func=cmd_attach_cv)

    p = sub.add_parser(
        "setup",
        help="Guided first-run setup: write config.local.toml",
        description=(
            "Interactively create config.local.toml from config.example.toml. "
            "Finds your source documents, asks for your details and hard "
            "constraints, and checks Chrome and Claude credentials."
        ),
    )
    p.add_argument("--force", action="store_true", help="Overwrite an existing config.local.toml without asking")
    p.set_defaults(func=cmd_setup, needs_config=False)

    p = sub.add_parser(
        "tui",
        help="Interactive pipeline browser",
        description=(
            "A terminal UI over the pipeline: pick a role and run a stage on it. "
            "Needs the optional extra: pip install -e '.[tui]'"
        ),
    )
    p.set_defaults(func=cmd_tui)

    # tidy-outputs
    p = sub.add_parser(
        "tidy-outputs",
        help="Move tailored CVs and letters into one folder per company and role",
        description="Moves each role's files to output/applications/<Company>/<Role - id>/ as "
        "<Name>-CV.pdf and <Name>-Cover-Letter.txt, and updates the tracker. Use --dry-run first.",
    )
    p.set_defaults(func=cmd_tidy_outputs)

    # recruiter
    p = sub.add_parser(
        "recruiter",
        help="Log search firms and recruiters: coverage, contact history, roles pitched",
        description=(
            "Relationships with recruiters, kept apart from roles. 'list --cold 30' shows "
            "open relationships with no contact for 30 days."
        ),
    )
    rsub = p.add_subparsers(dest="recruiter_action", required=True)
    r = rsub.add_parser("add", help="Add a recruiter")
    r.add_argument("firm")
    r.add_argument("--name")
    r.add_argument("--specialism", help="e.g. 'data & AI leadership, Benelux'")
    r.add_argument("--email")
    r.add_argument("--phone")
    r.add_argument("--url", help="Profile or firm page")
    r.add_argument("--notes")
    r.add_argument("--next", help="Next action")
    r.add_argument("--due", help="YYYY-MM-DD")
    r = rsub.add_parser("list", help="List recruiters, most overdue first")
    r.add_argument("--status", choices=("new", "contacted", "active", "cold", "closed"))
    r.add_argument("--cold", type=int, metavar="DAYS", help="Open relationships with no contact for DAYS days")
    r.add_argument("--json", action="store_true")
    r = rsub.add_parser("show", help="One recruiter with their history")
    r.add_argument("id", type=int)
    r = rsub.add_parser("log", help="Record a call, email, meeting, message or note")
    r.add_argument("id", type=int)
    r.add_argument("kind", choices=("call", "email", "meeting", "message", "note"))
    r.add_argument("body", nargs="?", default="")
    r.add_argument("--status", choices=("new", "contacted", "active", "cold", "closed"))
    r.add_argument("--next", help="Next action ('' clears it)")
    r.add_argument("--due", help="YYYY-MM-DD ('' clears it)")
    r = rsub.add_parser("pitch", help="Record a role a recruiter put forward")
    r.add_argument("id", type=int)
    r.add_argument("--company", required=True)
    r.add_argument("--title", required=True)
    r.add_argument("--job", help="Tracked job id (or a unique prefix) to link")
    r.add_argument("--note")
    r.add_argument("--status", choices=("new", "contacted", "active", "cold", "closed"))
    r.add_argument("--next")
    r.add_argument("--due")
    p.set_defaults(func=cmd_recruiter)

    p = sub.add_parser("doctor", help="Check config, source documents, Chrome, credentials")
    p.add_argument(
        "--boards",
        action="store_true",
        help="Also ping every configured ATS board (hits the network)",
    )
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # `setup` writes the config, so it must run before one exists.
    if getattr(args, "needs_config", True):
        try:
            cfg = load_config(args.config)
        except ConfigError as exc:
            print(f"config error: {exc}", file=sys.stderr)
            print("run `jobsearch setup` to create one", file=sys.stderr)
            return 2
    else:
        cfg = None

    try:
        return args.func(cfg, args)
    except CLIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary
        from .claude import ClaudeError
        from .discover import DiscoveryError
        from .render import RenderError
        from .tracker import TrackerError

        if isinstance(exc, (ClaudeError, DiscoveryError, RenderError, TrackerError, FileNotFoundError)):
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.verbose:
            raise
        print(f"unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("(re-run with -v for a traceback)", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
