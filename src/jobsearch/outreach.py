"""Outreach drafting. Nothing is scraped and nothing is sent.

LinkedIn's terms forbid automated access, so this module never touches it.
Instead it does the two things that are both legitimate and actually useful:

* **Infers** who the right contact *role* is for this company and this posting
  (hiring manager title, VP Eng, Head of Talent, founder) and emits a LinkedIn
  **search URL** the user clicks himself. A human opening a search page in a
  browser is normal use; a bot harvesting profiles is not.
* **Drafts** a short personalised message grounded in the dossier and the
  specific posting - plus an email version - and prints them for the user to
  copy, edit and send. There is no send path anywhere in this codebase.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from .claude import ClaudeClient, stable_context_for
from .config import Config
from .models import Contact, JobPosting, OutreachDraft

LINKEDIN_PEOPLE_SEARCH = "https://www.linkedin.com/search/results/people/?keywords={q}"


def linkedin_search_url(company: str, title: str) -> str:
    """A people-search URL for the user to open. Not fetched by this tool."""
    query = " ".join(part for part in [company, title] if part).strip()
    return LINKEDIN_PEOPLE_SEARCH.format(q=urllib.parse.quote(query))


OUTREACH_INSTRUCTIONS = """\
You draft outreach for one specific person applying to one specific role. His
career dossier and CV are supplied below and never change.

Produce:

1. `contacts`: 2-4 likely contact ROLES at this company for this posting, most
   promising first. Give the job title you would search for and why that person
   is the right door (e.g. "the hiring manager owns the mandate", "no AI
   executive exists, so the CPTO is the buyer"). Include a specific person's
   name ONLY if the job posting itself names them or the supplied target-company
   notes do. Never guess a name.
2. `linkedin_connection_note`: <= 200 characters. A connection request note.
3. `linkedin_message`: <= 300 characters. Assumes the connection was accepted.
4. `email_subject` and `email_body`: the same approach as an email. 120-180
   words. Plain text, no markdown.

How to write them:

- Lead with the specific thing about THIS company that made him reach out - a
  detail from the posting or the company notes, not a generality. If the notes
  say they bet the company on agents and have no AI leader, say that.
- Follow with the single most relevant proof point from the dossier. One. The
  right one. Not a list of achievements.
- Close with a low-friction ask: a 20-minute conversation, or a specific
  question. Never "I would love the opportunity to discuss how my skills...".
- Sound like a director-level peer writing to another, not a candidate
  petitioning. He is currently employed and choosing carefully.
- No flattery, no superlatives, no "passionate about". No em dashes.
- Respect the dossier's red-list: no revenue figures, no named pipeline
  companies, the semiconductor partner stays anonymous, no compensation talk.
- Every factual claim must be traceable to the dossier or the CV.
"""


OUTREACH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "contacts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "name": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["title", "name", "rationale"],
                "additionalProperties": False,
            },
        },
        "linkedin_connection_note": {"type": "string"},
        "linkedin_message": {"type": "string"},
        "email_subject": {"type": "string"},
        "email_body": {"type": "string"},
    },
    "required": [
        "contacts",
        "linkedin_connection_note",
        "linkedin_message",
        "email_subject",
        "email_body",
    ],
    "additionalProperties": False,
}


def draft_outreach(
    posting: JobPosting, cfg: Config, claude: ClaudeClient
) -> OutreachDraft:
    """Infer likely contacts and draft the LinkedIn and email approaches."""
    section = cfg.section("outreach")
    payload = claude.structured(
        instructions=OUTREACH_INSTRUCTIONS,
        stable_context=[
            *stable_context_for(cfg),
            ("target_company_notes", cfg.read_target_companies()),
        ],
        user_content=_prompt(posting, section),
        schema=OUTREACH_SCHEMA,
        stage="outreach",
        dry_run_value=_dry_run_payload(posting),
    )

    contacts = [
        Contact(
            title=item.get("title", ""),
            name=item.get("name", ""),
            rationale=item.get("rationale", ""),
            linkedin_search_url=linkedin_search_url(
                posting.company, item.get("name") or item.get("title", "")
            ),
        )
        for item in payload.get("contacts", [])
    ]

    return OutreachDraft(
        job_id=posting.job_id,
        contacts=contacts,
        linkedin_connection_note=_truncate(
            payload.get("linkedin_connection_note", ""),
            int(section.get("connection_note_max_chars", 200)),
        ),
        linkedin_message=_truncate(
            payload.get("linkedin_message", ""),
            int(section.get("linkedin_message_max_chars", 300)),
        ),
        email_subject=payload.get("email_subject", ""),
        email_body=payload.get("email_body", ""),
    )


def _truncate(text: str, limit: int) -> str:
    """Hard character caps matter: LinkedIn silently truncates over them."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 3].rstrip()
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut + "..."


def _prompt(posting: JobPosting, section: dict[str, Any]) -> str:
    titles = ", ".join(section.get("target_contact_titles", []))
    return "\n".join(
        [
            "<job_posting>",
            f"Company: {posting.company}",
            f"Title: {posting.title}",
            f"Location: {posting.location}",
            f"URL: {posting.url}",
            "",
            (posting.description or "(no description text available)")[:12000],
            "</job_posting>",
            "",
            f"<contact_titles_of_interest>{titles}</contact_titles_of_interest>",
            "",
            "Draft the outreach.",
        ]
    )


def _dry_run_payload(posting: JobPosting) -> dict[str, Any]:
    return {
        "contacts": [
            {
                "title": "Hiring manager for this role",
                "name": "",
                "rationale": "[dry-run] no API call made",
            }
        ],
        "linkedin_connection_note": "[dry-run] no API call made",
        "linkedin_message": "[dry-run] no API call made",
        "email_subject": f"[dry-run] {posting.title} at {posting.company}",
        "email_body": "[dry-run] no API call made",
    }


def format_draft(draft: OutreachDraft, posting: JobPosting | None = None) -> str:
    """Render the draft for the terminal, with the click-yourself URLs."""
    lines: list[str] = []
    company = posting.company if posting else ""
    lines.append("LIKELY CONTACTS (inferred, not scraped - open the search yourself):")
    for contact in draft.contacts:
        who = f"{contact.name} ({contact.title})" if contact.name else contact.title
        lines.append(f"  - {who}")
        if contact.rationale:
            lines.append(f"    why: {contact.rationale}")
        lines.append(f"    search: {contact.linkedin_search_url}")
    if company:
        lines.append(f"    company page: {linkedin_search_url(company, 'Head of Talent')}")
    lines.append("")
    lines.append(
        f"LINKEDIN CONNECTION NOTE ({len(draft.linkedin_connection_note)} chars):"
    )
    lines.append(f"  {draft.linkedin_connection_note}")
    lines.append("")
    lines.append(f"LINKEDIN MESSAGE ({len(draft.linkedin_message)} chars):")
    lines.append(f"  {draft.linkedin_message}")
    lines.append("")
    lines.append("EMAIL")
    lines.append(f"  Subject: {draft.email_subject}")
    lines.append("")
    for line in (draft.email_body or "").splitlines():
        lines.append(f"  {line}")
    lines.append("")
    lines.append("Nothing has been sent. Copy, edit, and send these yourself.")
    return "\n".join(lines)


__all__ = ["draft_outreach", "format_draft", "linkedin_search_url"]
