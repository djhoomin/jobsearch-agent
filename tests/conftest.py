"""Shared fixtures.

No test in this suite makes a live Claude API call. The Claude client is
replaced by :class:`FakeClaude`, which returns canned structured payloads and
records what it was asked for - including whether the cacheable prefix was
placed before the volatile per-job content.
"""

from __future__ import annotations

import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pytest

from jobsearch.claude import Usage
from jobsearch.config import Config
from jobsearch.models import JobPosting

FIXTURES = Path(__file__).parent / "fixtures"

MINIMAL_DOSSIER = """\
# Career dossier (test fixture)

Jane Q. Testcandidate, Director of Engineering at Northwind Labs since May 2024.
Previously Contoso AI (Director of Data Science, Oct 2022 - May 2024), Initech
(Mar 2020 - Oct 2022), Example Bank (2015-2020).
MSc Operations Research, cum laude, Fictional University.
Patent XX/2025/00000, approved for PCT international filing.
"""

MINIMAL_STRATEGY = """\
# Strategy (test fixture)

Hard constraints: IND sponsor, non-compete, EUR 125K floor, Amsterdam/remote-EU,
travel roughly monthly.
Weights: Buyer 20%, Role fit 25%, Company 25%, Domain 15%, Talent 15%.
"""

MINIMAL_TARGETS = "# Targets (test fixture)\n\nWeaviate, Mistral, DataSnipper.\n"


@pytest.fixture
def base_cv_html() -> str:
    """The ATS-hardened CV template used by the verifier fixtures."""
    return (FIXTURES / "good_cv.html").read_text(encoding="utf-8")


LOCAL_PATHS = """\
[paths]
dossier = "dossier.md"
base_cv = "base_cv.html"
search_strategy = "strategy.md"
target_companies = "targets.md"
tracker_xlsx = "tracker.xlsx"
output_dir = "output"
db_path = "output/test.db"

"""


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """A fully working config pointing at throwaway source documents."""
    (tmp_path / "dossier.md").write_text(MINIMAL_DOSSIER, encoding="utf-8")
    (tmp_path / "strategy.md").write_text(MINIMAL_STRATEGY, encoding="utf-8")
    (tmp_path / "targets.md").write_text(MINIMAL_TARGETS, encoding="utf-8")
    shutil.copy(FIXTURES / "good_cv.html", tmp_path / "base_cv.html")

    # Take the committed template config and repoint [paths] at the throwaway
    # copies, then write it to disk so `jobsearch --config` can load it too.
    # Using the template (not the gitignored config.local.toml) keeps the test
    # suite runnable on a fresh clone and makes the shipped example a tested
    # artifact.
    repo_config = Path(__file__).resolve().parents[1] / "config.example.toml"
    text = repo_config.read_text(encoding="utf-8")
    # Anchor on the line start: "[paths]" also appears inside a comment above it.
    start = text.index("\n[paths]") + 1
    end = text.index("\n[claude]") + 1
    text = text[:start] + LOCAL_PATHS + text[end:]
    (tmp_path / "config.local.toml").write_text(text, encoding="utf-8")

    return Config(root=tmp_path, raw=tomllib.loads(text), source=tmp_path / "config.local.toml")


@pytest.fixture
def posting() -> JobPosting:
    return JobPosting(
        company="Weaviate",
        title="Director of Product",
        url="https://jobs.ashbyhq.com/weaviate/abc-123",
        source="ashby",
        location="Amsterdam, Netherlands (remote CET)",
        description=(
            "We are looking for a Director of Product to own our database, cloud "
            "and agent products. You will work with knowledge graphs, vector "
            "search and agentic AI. EUR 150,000 - 180,000 base. We are an IND "
            "recognised sponsor and offer visa sponsorship."
        ),
        salary_text="EUR 150,000 - 180,000",
        ind_sponsor="yes",
        board_tier=1,
    )


@dataclass
class FakeClaude:
    """Stand-in for :class:`jobsearch.claude.ClaudeClient`.

    Records every call so tests can assert on prompt construction, especially
    that the cacheable stable context precedes the volatile job content.
    """

    model: str = "claude-opus-5"
    max_tokens: int = 16000
    streaming_max_tokens: int = 64000
    effort: str = "high"
    cache_ttl: str = "1h"
    dry_run: bool = False
    dry_run_hook: Any = None
    structured_responses: dict[str, Any] = field(default_factory=dict)
    stream_response: str = ""
    calls: list[dict[str, Any]] = field(default_factory=list)
    _usage: Usage = field(default_factory=lambda: Usage(input_tokens=100, cache_read_input_tokens=9000))

    @property
    def last_usage(self) -> Usage:
        return self._usage

    def system_blocks(
        self, instructions: str, stable_context: Sequence[tuple[str, str]] = ()
    ) -> list[dict[str, Any]]:
        from jobsearch.claude import ClaudeClient

        return ClaudeClient(cache_ttl=self.cache_ttl).system_blocks(instructions, stable_context)

    def structured(
        self,
        *,
        instructions: str,
        stable_context: Sequence[tuple[str, str]],
        user_content: str,
        schema: dict[str, Any],
        stage: str = "structured",
        dry_run_value: Any = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "kind": "structured",
                "stage": stage,
                "instructions": instructions,
                "stable_context": list(stable_context),
                "user_content": user_content,
                "schema": schema,
            }
        )
        if stage not in self.structured_responses:
            raise AssertionError(f"FakeClaude has no canned response for stage {stage!r}")
        return self.structured_responses[stage]

    def stream_text(
        self,
        *,
        instructions: str,
        stable_context: Sequence[tuple[str, str]],
        user_content: str,
        stage: str = "stream",
        dry_run_value: str = "",
        on_delta: Any = None,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "kind": "stream",
                "stage": stage,
                "instructions": instructions,
                "stable_context": list(stable_context),
                "user_content": user_content,
            }
        )
        if on_delta:
            on_delta(self.stream_response)
        return self.stream_response


@pytest.fixture
def fake_claude() -> FakeClaude:
    return FakeClaude()


def chrome_available() -> bool:
    from jobsearch.render import RenderError, find_chrome

    try:
        find_chrome()
        return True
    except RenderError:
        return False


requires_chrome = pytest.mark.skipif(
    not chrome_available(), reason="headless Chrome not available on this machine"
)
