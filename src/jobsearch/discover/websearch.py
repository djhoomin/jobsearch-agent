"""Broader discovery via Claude's server-side ``web_search`` tool.

Optional. Configured boards are the primary source because they are exact and
free; this is for finding roles at companies not yet on the board list, or
newly created leadership seats that were never posted to a board the tool knows.

The model returns *leads* - company, title, URL, why it matched - which the
user then confirms. It is not treated as ground truth and nothing is written to
the tracker from here without the user's confirmation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from ..claude import ClaudeClient, ClaudeError
from ..config import Config
from ..models import JobPosting

log = logging.getLogger(__name__)

WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": 8,
}

INSTRUCTIONS = """\
You are a job-search researcher for a senior AI leader based in Amsterdam.

Use web_search to find CURRENTLY OPEN job postings that match the brief. Rules:

- Only report roles you actually saw on a company careers page, a public ATS
  board (Greenhouse, Lever, Ashby, Workable, SmartRecruiters), or the company's
  own announcement. Do not report roles from aggregators you could not verify.
- Never report a LinkedIn or Indeed listing URL as the source of truth; if you
  found it there, locate the company's own posting and cite that instead.
- If you cannot verify a role is currently open, say so in `confidence`.
- Prefer Netherlands / remote-EU roles at Series B-D companies.

Return your findings using the required JSON shape. Be honest about weak
matches: an empty list is a valid and useful answer.
"""

LEADS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "leads": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "location": {"type": "string"},
                    "summary": {"type": "string"},
                    "why_it_matches": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": [
                    "company",
                    "title",
                    "url",
                    "location",
                    "summary",
                    "why_it_matches",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["leads", "notes"],
    "additionalProperties": False,
}


def web_search_discover(
    cfg: Config,
    claude: ClaudeClient,
    query: str,
    *,
    max_results: int = 10,
) -> list[JobPosting]:
    """Ask Claude to search the open web for matching roles.

    Server tools cannot be combined with ``output_config.format`` in one call,
    so this runs the search turn first and then asks the model to restate its
    findings as JSON in a second, cheap turn.
    """
    if claude.dry_run:
        log.info("[dry-run] would run a web_search discovery pass for: %s", query)
        return []

    brief = _brief(cfg, query, max_results)

    try:
        research = claude.client.messages.create(
            model=claude.model,
            max_tokens=claude.max_tokens,
            thinking={"type": "adaptive"},
            system=INSTRUCTIONS,
            tools=[WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": brief}],
        )
    except anthropic.NotFoundError as exc:
        raise ClaudeError(
            "web_search is not available for this API key or model. Board-based "
            "discovery still works: run `jobsearch discover` without --web."
        ) from exc
    except anthropic.APIStatusError as exc:
        raise ClaudeError(f"web_search discovery failed: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise ClaudeError(f"web_search discovery: network error. {exc}") from exc

    if research.stop_reason == "pause_turn":
        log.warning(
            "web_search paused before finishing (stop_reason=pause_turn); "
            "results may be partial."
        )

    findings = "\n".join(b.text for b in research.content if b.type == "text")
    if not findings.strip():
        return []

    structured = claude.structured(
        instructions="Restate the supplied research findings as JSON. Do not invent roles.",
        stable_context=(),
        user_content=f"<research_findings>\n{findings}\n</research_findings>",
        schema=LEADS_SCHEMA,
        stage="web_search_extract",
        dry_run_value={"leads": [], "notes": ""},
    )

    postings: list[JobPosting] = []
    for lead in structured.get("leads", [])[:max_results]:
        board = cfg.board_for(lead.get("company", ""))
        postings.append(
            JobPosting(
                company=lead.get("company", ""),
                title=lead.get("title", ""),
                url=lead.get("url", ""),
                source="web_search",
                location=lead.get("location", ""),
                description="\n\n".join(
                    filter(
                        None,
                        [
                            lead.get("summary", ""),
                            f"Why it matches: {lead.get('why_it_matches', '')}",
                            f"Discovery confidence: {lead.get('confidence', 'low')}",
                        ],
                    )
                ),
                board_tier=board.tier if board else None,
                ind_sponsor=board.sponsor_state if board else "unknown",
                gaming=board.gaming if board else False,
                raw=lead,
            )
        )
    return postings


def _brief(cfg: Config, query: str, max_results: int) -> str:
    section = cfg.section("discover")
    include = ", ".join(section.get("title_include", [])[:20])
    constraints = cfg.section("constraints")
    return (
        f"<search_request>{query}</search_request>\n\n"
        f"<target_titles>{include}</target_titles>\n\n"
        "<hard_constraints>\n"
        f"- Location: {cfg.get('candidate', 'location', 'unspecified')}, "
        "hybrid there, or remote in the same region. Not onsite elsewhere.\n"
        f"- Travel: roughly monthly maximum; weekly-travel roles are out.\n"
        + (
            f"- Base compensation floor: EUR {constraints['comp_floor_eur']:,}.\n"
            if constraints.get("comp_floor_eur")
            else ""
        )
        +
        f"- Netherlands employment needs an IND recognised sponsor or equivalent EOR.\n"
        "</hard_constraints>\n\n"
        f"<target_company_context>\n{cfg.read_target_companies()[:6000]}\n"
        "</target_company_context>\n\n"
        f"Return at most {max_results} leads."
    )
