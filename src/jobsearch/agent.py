"""The agentic end-to-end command, built on the SDK's Tool Runner.

``jobsearch run <url-or-id>`` hands Claude the pipeline stages as tools and lets
it decide the order and the stopping point - which matters because the right
sequence genuinely varies. A role that fails a hard constraint should never be
tailored for. A role scoring 4.4 with a warm path should get a CV and outreach.
A senior-IC posting at a great company should produce outreach aimed at the
leadership seat instead of an application.

Implementation notes:

* ``client.beta.messages.tool_runner`` with ``@beta_tool`` functions, rather
  than a hand-rolled loop. Tool schemas come from the type hints and docstrings.
* The runner's tools need access to config, tracker and client. They get it from
  a module-level :class:`RunContext` set by :func:`run_agent` - the decorator
  turns a plain function into a tool, so the state cannot be passed as an
  argument without leaking into the tool schema.
* ``pause_turn`` is handled by mirroring the message history and restarting the
  runner, as the Python runner does not auto-resume a paused turn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from anthropic import beta_tool

from .claude import ClaudeClient
from .config import Config
from .models import Status
from .tracker import Tracker

log = logging.getLogger(__name__)

MAX_RESTARTS = 5


@dataclass
class RunContext:
    """Everything the tools need. Set once per ``run`` invocation."""

    cfg: Config
    claude: ClaudeClient
    tracker: Tracker
    dry_run: bool = False
    transcript: list[str] = field(default_factory=list)

    def log(self, line: str) -> None:
        self.transcript.append(line)
        log.info("%s", line)


_CONTEXT: RunContext | None = None


def _ctx() -> RunContext:
    if _CONTEXT is None:  # pragma: no cover - guarded by run_agent
        raise RuntimeError("Agent tools called outside of a run context")
    return _CONTEXT


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@beta_tool
def fetch_posting(url_or_id: str) -> str:
    """Fetch a job posting and store it in the tracker.

    Accepts a public ATS URL (Greenhouse, Lever, Ashby) or the id of a posting
    already tracked. Refuses LinkedIn and Indeed URLs, whose terms forbid
    automated access.

    Args:
        url_or_id: The posting URL, or a tracked job id / id prefix.
    """
    ctx = _ctx()
    if not url_or_id.lower().startswith("http"):
        job_id = ctx.tracker.resolve_job_id(url_or_id)
        posting = ctx.tracker.get_posting(job_id)
    else:
        from .discover import Fetcher, fetch_single_posting

        posting = fetch_single_posting(url_or_id, Fetcher.from_config(ctx.cfg), ctx.cfg)
        ctx.tracker.upsert_job(posting)
    ctx.log(f"fetched {posting.job_id}: {posting.company} - {posting.title}")
    payload = posting.to_dict()
    payload["description"] = (payload.get("description") or "")[:6000]
    return json.dumps(payload)


@beta_tool
def score_job(job_id: str) -> str:
    """Score a tracked posting: hard constraints first, then the weighted rubric.

    Returns the per-dimension scores with their reasoning, and whether any hard
    constraint (visa, non-compete, compensation floor, location, travel)
    eliminated the role. A role that fails a hard constraint must not be
    tailored for.

    Args:
        job_id: The tracked job id.
    """
    ctx = _ctx()
    from .scoring import score_posting

    job_id = ctx.tracker.resolve_job_id(job_id)
    posting = ctx.tracker.get_posting(job_id)
    report = score_posting(posting, ctx.cfg, ctx.claude)
    if not ctx.dry_run:
        ctx.tracker.save_score(report)
    ctx.log(
        f"scored {job_id}: weighted={report.weighted} "
        f"constraints={'ok' if report.constraints.passed else 'FAILED'}"
    )
    return json.dumps(report.to_dict())


@beta_tool
def tailor_cv_for(job_id: str) -> str:
    """Generate a role-tailored CV, render it to PDF, and verify it.

    Runs the grounding audit: every claim in the generated CV is checked against
    the career dossier and base CV, and anything ungrounded is returned so it
    can be reported to the user rather than silently shipped. Also runs the ATS
    verifier on the rendered PDF.

    Args:
        job_id: The tracked job id.
    """
    ctx = _ctx()
    from .ats import verify_from_config
    from .tailor import tailor_cv

    job_id = ctx.tracker.resolve_job_id(job_id)
    posting = ctx.tracker.get_posting(job_id)
    result = tailor_cv(posting, ctx.cfg, ctx.claude)

    payload: dict[str, Any] = {
        "html_path": result.html_path,
        "pdf_path": result.pdf_path,
        "headline": result.headline,
        "ungrounded_claims": [c.to_dict() for c in result.ungrounded],
        "cache_read_tokens": result.cache_read_tokens,
    }
    if result.pdf_path and not ctx.dry_run:
        report = verify_from_config(ctx.cfg, result.pdf_path, jd_text=posting.description)
        payload["ats"] = report.to_dict()
        ctx.tracker.save_cv(job_id, result.html_path, result.pdf_path, report.to_dict())
    ctx.log(
        f"tailored {job_id}: {result.pdf_path} "
        f"({len(result.ungrounded)} ungrounded claims)"
    )
    return json.dumps(payload)


@beta_tool
def verify_cv_pdf(pdf_path: str, job_id: str = "") -> str:
    """Run the ATS verifier standalone against a rendered PDF.

    Checks page count, section headings, heading corruption from letter-spacing,
    education entries keeping their dates, inline bullet markers, hyphenated
    keywords broken across line wraps, and job-description keyword coverage.

    Args:
        pdf_path: Path to the PDF to verify.
        job_id: Optional tracked job id, to compare against its description.
    """
    ctx = _ctx()
    from .ats import verify_from_config

    jd = ""
    if job_id:
        jd = ctx.tracker.get_posting(ctx.tracker.resolve_job_id(job_id)).description
    report = verify_from_config(ctx.cfg, pdf_path, jd_text=jd)
    ctx.log(f"verified {pdf_path}: {'PASS' if report.passed else 'FAIL'}")
    return json.dumps(report.to_dict())


@beta_tool
def draft_outreach_for(job_id: str) -> str:
    """Infer likely contacts and draft LinkedIn and email outreach.

    Emits LinkedIn *search URLs* for the user to click. It never scrapes
    LinkedIn and never sends anything.

    Args:
        job_id: The tracked job id.
    """
    ctx = _ctx()
    from .outreach import draft_outreach

    job_id = ctx.tracker.resolve_job_id(job_id)
    posting = ctx.tracker.get_posting(job_id)
    draft = draft_outreach(posting, ctx.cfg, ctx.claude)
    if not ctx.dry_run:
        ctx.tracker.save_outreach(draft)
    ctx.log(f"drafted outreach for {job_id}: {len(draft.contacts)} contacts")
    return json.dumps(draft.to_dict())


@beta_tool
def update_tracker(job_id: str, status: str = "", next_action: str = "", note: str = "") -> str:
    """Record progress against a tracked job.

    Args:
        job_id: The tracked job id.
        status: One of: Not started, Parked, Outreach sent, Applied,
            In conversation, Interviewing, Offer, Rejected, Withdrawn.
        next_action: The single next thing the user should do.
        note: A free-text note to append to the job's history.
    """
    ctx = _ctx()
    job_id = ctx.tracker.resolve_job_id(job_id)
    changes: list[str] = []
    if status:
        target = Status.parse(status)
        if not ctx.dry_run:
            ctx.tracker.set_status(job_id, target, reason="set by agent run")
        changes.append(f"status={target.value}")
    if next_action and not ctx.dry_run:
        ctx.tracker.set_fields(job_id, next_action=next_action)
        changes.append("next_action set")
    if note and not ctx.dry_run:
        ctx.tracker.add_note(job_id, note)
        changes.append("note added")
    ctx.log(f"tracker {job_id}: {', '.join(changes) or 'no change'}")
    return json.dumps({"job_id": job_id, "changes": changes})


TOOLS = [
    fetch_posting,
    score_job,
    tailor_cv_for,
    verify_cv_pdf,
    draft_outreach_for,
    update_tracker,
]


SYSTEM = """\
You are running one job application end to end for a senior AI leader based in
Amsterdam, using the tools provided.

The order that usually makes sense:

1. fetch_posting - get the role into the tracker.
2. score_job - ALWAYS before tailoring. If a hard constraint failed (visa,
   non-compete, compensation floor, location, travel), STOP. Do not tailor a CV
   for a role he cannot take. Explain which constraint failed and what would
   have to change for the role to become viable.
3. If it survives and scores at or above the shortlist threshold, tailor_cv_for.
   Report every ungrounded claim the grounding audit returned, verbatim. Do not
   summarise them away: an unverifiable claim on a CV is the worst outcome this
   tool can produce.
4. Report the ATS verification result. If it failed, say exactly which check.
5. draft_outreach_for, then update_tracker with the status and next action.

Judgement calls that are yours to make:
- A senior-IC posting at a strong company is an altitude step down. Do not
  simply apply: draft outreach aimed at the leadership conversation instead,
  and say why.
- A role scoring below the threshold is worth stopping on. Say so plainly.

Finish with a short brief: what the role is, what it scored and why, where the
CV and PDF are, what to check before sending, and the single next action. Be
concise and honest. Do not tell him a weak role is strong.
"""


def run_agent(
    target: str,
    cfg: Config,
    claude: ClaudeClient,
    tracker: Tracker,
    *,
    instruction: str = "",
    max_restarts: int = MAX_RESTARTS,
) -> str:
    """Run the agentic pipeline for one posting. Returns the final brief."""
    global _CONTEXT
    _CONTEXT = RunContext(cfg=cfg, claude=claude, tracker=tracker, dry_run=claude.dry_run)

    if claude.dry_run:
        return (
            "[dry-run] would run the agentic pipeline for "
            f"{target!r} with tools: {', '.join(t.name for t in TOOLS)}"
        )

    user = f"Work this role end to end: {target}"
    if instruction:
        user += f"\n\nExtra instruction from the user: {instruction}"

    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    final_text = ""
    restarts = 0

    try:
        while True:
            runner = claude.client.beta.messages.tool_runner(
                model=claude.model,
                max_tokens=claude.max_tokens,
                thinking={"type": "adaptive"},
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
            )
            last = None
            for message in runner:
                last = message
                # Mirror history: the runner keeps its own copy and does not
                # expose it, and we need it to restart on a paused turn.
                messages.append({"role": "assistant", "content": message.content})
                tool_response = runner.generate_tool_call_response()
                if tool_response is not None:
                    messages.append(tool_response)
                text = "".join(b.text for b in message.content if b.type == "text")
                if text.strip():
                    final_text = text

            if last is None or last.stop_reason != "pause_turn":
                break
            restarts += 1
            if restarts > max_restarts:
                raise RuntimeError(
                    f"Agent run still paused after {max_restarts} restarts; giving up."
                )
    finally:
        _CONTEXT = None

    return final_text


__all__ = ["RunContext", "TOOLS", "run_agent"]
