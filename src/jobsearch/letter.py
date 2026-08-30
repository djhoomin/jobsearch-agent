"""Cover letters.

Short on purpose. Most are skimmed at best, so the only version worth sending
is one that could not have been sent to anyone else: it names the specific
thing this posting is buying and puts one piece of evidence behind it.

Grounded the same way the CV is - every claim is audited against the dossier
before it reaches you - because a letter is where invented detail is easiest to
write and hardest to notice.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .claude import ClaudeClient, stable_context_for
from .config import Config
from .models import Claim, JobPosting
from .tailor import ground_claims, normalise_dashes

log = logging.getLogger(__name__)

__all__ = ["LETTER_INSTRUCTIONS", "LetterResult", "write_letter", "letter_path_for"]

LETTER_INSTRUCTIONS = """\
You write one short cover letter for one specific job posting. The candidate's
career dossier (the fact base) and his ATS-hardened base CV are supplied below.

OUTPUT PLAIN TEXT. No markdown headings, no bullet list, no subject line, no
commentary before or after. Start at the greeting and end at the sign-off.

## The one rule that overrides every other rule

NEVER INVENT ANYTHING. Every fact - every number, employer, title, technology,
outcome - must be traceable to the sources. If a requirement is not met, leave
it out. A gap is honest; an invention is disqualifying. Respect the dossier's
red-list guardrails: no revenue figures, no named pipeline companies, the
semiconductor partner stays anonymous, no compensation specifics.

## Shape

- 200-280 words. Four short paragraphs at most. Shorter is better.
- Open with the specific thing this posting is buying, in the posting's own
  terms. Never "I am writing to apply for" or "I was excited to see".
- One paragraph of evidence: the two or three items from the dossier that most
  directly answer this posting, each with its number. Not a CV summary.
- One paragraph on why this company and not a similar one, using something
  specific to the posting or the company. If there is nothing specific to say,
  omit the paragraph rather than write filler.
- Close with a concrete next step. No "I look forward to hearing from you".

## Register

Plain, direct, first person. No superlatives about yourself, no "passionate",
no "thrilled", no "leverage" as a verb, no three-adjective strings. Never an em dash or an
en dash - use a plain hyphen, a comma, or two sentences. Write the
way a competent person emails a peer. Contractions are fine.

If the posting names a hiring manager, address them. Otherwise use a plain
"Hello," - never "To Whom It May Concern" or "Dear Sir/Madam".
"""


class LetterResult:
    """A generated letter and its grounding audit."""

    def __init__(self, job_id: str, path: str, text: str, claims: list[Claim] | None = None):
        self.job_id = job_id
        self.path = path
        self.text = text
        self.claims: list[Claim] = claims or []

    @property
    def ungrounded(self) -> list[Claim]:
        return [c for c in self.claims if not c.grounded]

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def letter_path_for(cfg: Config, posting: JobPosting) -> Path:
    """Where this role's letter lives.

    Named for what it is. `output_stem` produces a CV filename, and a cover
    letter called DJ_Human_CV_Databricks.txt is the kind of thing that ends up
    attached to the wrong field of an application form.
    """
    company = re.sub(r"[^A-Za-z0-9]+", "", posting.company) or "Role"
    return cfg.ensure_output_dir() / "letters" / f"Cover_Letter_{company}.txt"


def strip_stray_markdown(text: str) -> str:
    """Remove markdown the model adds despite being told not to.

    Cheaper and more reliable than another round trip: a letter with a stray
    '## Introduction' heading is not a letter.
    """
    lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", cleaned)
    if cleaned.strip().startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*\n", "", cleaned.strip())
        cleaned = re.sub(r"\n```\s*$", "", cleaned)
    cleaned = normalise_dashes(cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"


def write_letter(
    posting: JobPosting,
    cfg: Config,
    claude: ClaudeClient,
    *,
    verify_claims: bool = True,
    instruction: str = "",
) -> LetterResult:
    """Generate one cover letter, audit it, and write it to the output folder."""
    path = letter_path_for(cfg, posting)
    path.parent.mkdir(parents=True, exist_ok=True)

    prompt = [
        "<job_posting>",
        f"Company: {posting.company}",
        f"Exact job title: {posting.title}",
        f"Location: {posting.location}",
        f"URL: {posting.url}",
        "",
        posting.description or "(no description text available)",
        "</job_posting>",
    ]
    if instruction:
        prompt += ["", f"<extra_instruction>{instruction}</extra_instruction>"]
    prompt += ["", "Write the letter now. Plain text only."]

    text = claude.stream_text(
        instructions=LETTER_INSTRUCTIONS,
        stable_context=stable_context_for(cfg),
        user_content="\n".join(prompt),
        stage="letter",
        dry_run_value="Hello,\n\n[dry-run] no letter was generated.\n",
    )
    text = strip_stray_markdown(text)

    if not claude.dry_run:
        path.write_text(text, encoding="utf-8")

    result = LetterResult(job_id=posting.job_id, path=str(path), text=text)
    if verify_claims and not claude.dry_run:
        result.claims = ground_claims(text, posting, cfg, claude)
    if result.word_count > 400:
        log.warning("letter is %d words; the brief asks for under 280", result.word_count)
    return result
