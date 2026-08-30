"""Thin, opinionated wrapper around the Anthropic Python SDK.

Design decisions worth knowing:

* **Zero-arg client.** ``anthropic.Anthropic()`` resolves ``ANTHROPIC_API_KEY``,
  ``ANTHROPIC_AUTH_TOKEN`` or an ``ant auth login`` profile. This module never
  prompts for or stores a key.
* **Adaptive thinking.** ``thinking={"type": "adaptive"}``. ``budget_tokens`` was
  removed on Claude Opus 5 and returns a 400 - it is never sent.
* **No assistant prefill.** Prefill returns a 400 on this model, so structure is
  obtained with ``output_config={"format": {...}}`` instead.
* **Prompt caching.** The dossier and base CV form a ~40KB stable prefix that is
  re-sent on every scoring / tailoring / grounding call. It is placed in the
  *system* blocks, before any per-job content, with ``cache_control`` on the last
  stable block. :meth:`ClaudeClient.last_usage` exposes
  ``cache_read_input_tokens`` so a cache regression is visible.
* **Streaming for long generations.** A full tailored CV is ~10K output tokens;
  ``messages.stream()`` + ``get_final_message()`` is used there.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import anthropic

log = logging.getLogger(__name__)

MODEL = "claude-opus-5"


class ClaudeError(RuntimeError):
    """A Claude API failure that the CLI should report, not traceback on."""


@dataclass
class Usage:
    """Token accounting for one call, including cache effectiveness."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def cache_hit(self) -> bool:
        return self.cache_read_input_tokens > 0

    def describe(self) -> str:
        return (
            f"in={self.input_tokens} out={self.output_tokens} "
            f"cache_read={self.cache_read_input_tokens} "
            f"cache_write={self.cache_creation_input_tokens}"
        )


@dataclass
class ClaudeClient:
    """Wrapper carrying the stable cached prefix for this job search."""

    model: str = MODEL
    max_tokens: int = 16000
    streaming_max_tokens: int = 64000
    effort: str = "high"
    cache_ttl: str = "1h"
    dry_run: bool = False
    #: Called with (stage_name, prompt_summary) when ``dry_run`` is on.
    dry_run_hook: Callable[[str, str], None] | None = None
    _client: anthropic.Anthropic | None = field(default=None, repr=False)
    _last_usage: Usage = field(default_factory=Usage, repr=False)

    # -- lifecycle ---------------------------------------------------------
    @property
    def client(self) -> anthropic.Anthropic:
        """Lazily construct the SDK client, so --dry-run needs no credentials."""
        if self._client is None:
            try:
                self._client = anthropic.Anthropic()
            except Exception as exc:  # pragma: no cover - env-dependent
                raise ClaudeError(
                    "Could not initialise the Anthropic client. Set ANTHROPIC_API_KEY "
                    "or run `ant auth login`."
                ) from exc
        return self._client

    @classmethod
    def from_config(cls, cfg, dry_run: bool = False) -> "ClaudeClient":
        section = cfg.section("claude")
        return cls(
            model=section.get("model", MODEL),
            max_tokens=int(section.get("max_tokens", 16000)),
            streaming_max_tokens=int(section.get("streaming_max_tokens", 64000)),
            effort=section.get("effort", "high"),
            cache_ttl=section.get("cache_ttl", "1h"),
            dry_run=dry_run,
        )

    @property
    def last_usage(self) -> Usage:
        return self._last_usage

    # -- prompt assembly ---------------------------------------------------
    def system_blocks(
        self,
        instructions: str,
        stable_context: Sequence[tuple[str, str]] = (),
    ) -> list[dict[str, Any]]:
        """Build system blocks with the cache breakpoint in the right place.

        ``instructions`` is the frozen role description. ``stable_context`` is a
        sequence of ``(label, text)`` pairs - the dossier, the base CV, the
        rubric - that do not change between jobs. ``cache_control`` goes on the
        LAST stable block so everything before it is cached; per-job content
        goes into ``messages`` and therefore after the breakpoint.

        Ordering matters and must be deterministic: caching is a prefix match,
        so a reordered or re-timestamped block silently invalidates the cache.
        """
        blocks: list[dict[str, Any]] = [{"type": "text", "text": instructions}]
        for label, text in stable_context:
            blocks.append({"type": "text", "text": f"<{label}>\n{text}\n</{label}>"})
        if len(blocks) > 1:
            blocks[-1]["cache_control"] = {"type": "ephemeral", "ttl": self.cache_ttl}
        return blocks

    # -- calls -------------------------------------------------------------
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
        """One structured-output call. Returns the parsed JSON object.

        Uses ``output_config={"format": {"type": "json_schema", ...}}`` - the
        supported mechanism; the deprecated ``output_format`` top-level
        parameter is deliberately not used.
        """
        if self.dry_run:
            self._note_dry_run(stage, user_content)
            return dry_run_value if dry_run_value is not None else {}

        response = self._call(
            stage,
            lambda: self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                system=self.system_blocks(instructions, stable_context),
                messages=[{"role": "user", "content": user_content}],
            ),
        )
        self._record_usage(response)
        text = next((b.text for b in response.content if b.type == "text"), "")
        if not text.strip():
            raise ClaudeError(f"{stage}: model returned no text block")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ClaudeError(f"{stage}: response was not valid JSON: {exc}") from exc

    def stream_text(
        self,
        *,
        instructions: str,
        stable_context: Sequence[tuple[str, str]],
        user_content: str,
        stage: str = "stream",
        dry_run_value: str = "",
        on_delta: Callable[[str], None] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Stream a long generation (a full CV) and return the final text."""
        if self.dry_run:
            self._note_dry_run(stage, user_content)
            return dry_run_value

        def run() -> str:
            chunks: list[str] = []
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens or self.streaming_max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                system=self.system_blocks(instructions, stable_context),
                messages=[{"role": "user", "content": user_content}],
            ) as stream:
                for text in stream.text_stream:
                    chunks.append(text)
                    if on_delta:
                        on_delta(text)
                final = stream.get_final_message()
            self._record_usage(final)
            if final.stop_reason == "max_tokens":
                raise ClaudeError(
                    f"{stage}: generation hit max_tokens; raise "
                    "[claude].streaming_max_tokens"
                )
            return "".join(chunks)

        return self._call(stage, run)

    # -- error handling ----------------------------------------------------
    def _call(self, stage: str, fn: Callable[[], Any]) -> Any:
        """Run an API call with the typed exception chain, narrowest first."""
        try:
            return fn()
        except anthropic.NotFoundError as exc:
            raise ClaudeError(
                f"{stage}: model or endpoint not found ({self.model}). {exc.message}"
            ) from exc
        except anthropic.AuthenticationError as exc:
            raise ClaudeError(
                f"{stage}: authentication failed. Set ANTHROPIC_API_KEY or run "
                "`ant auth login`."
            ) from exc
        except anthropic.PermissionDeniedError as exc:
            raise ClaudeError(f"{stage}: API key lacks permission. {exc.message}") from exc
        except anthropic.BadRequestError as exc:
            raise ClaudeError(f"{stage}: bad request. {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            retry_after = "unknown"
            if exc.response is not None:
                retry_after = exc.response.headers.get("retry-after", "unknown")
            raise ClaudeError(
                f"{stage}: rate limited (retry after {retry_after}s)."
            ) from exc
        except anthropic.APIStatusError as exc:
            kind = "server" if exc.status_code >= 500 else "API"
            raise ClaudeError(f"{stage}: {kind} error {exc.status_code}. {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ClaudeError(f"{stage}: network error reaching the API. {exc}") from exc

    def _record_usage(self, message: Any) -> None:
        usage = getattr(message, "usage", None)
        if usage is None:
            return
        self._last_usage = Usage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
        log.debug("usage: %s", self._last_usage.describe())

    def _note_dry_run(self, stage: str, prompt: str) -> None:
        summary = prompt.strip().replace("\n", " ")[:160]
        if self.dry_run_hook:
            self.dry_run_hook(stage, summary)
        else:
            log.info("[dry-run] would call Claude for %s: %s...", stage, summary)


def stable_context_for(cfg, include: Iterable[str] = ("dossier", "base_cv", "rubric")) -> list[tuple[str, str]]:
    """Assemble the cacheable prefix from the user's source documents.

    Order is fixed (dossier, base CV, rubric) so the cache prefix is stable
    across every call in the tool.
    """
    parts: list[tuple[str, str]] = []
    wanted = list(include)
    if "dossier" in wanted:
        parts.append(("career_dossier", cfg.read_dossier()))
    if "base_cv" in wanted:
        parts.append(("base_cv_html", cfg.read_base_cv()))
    if "rubric" in wanted:
        parts.append(("search_strategy", cfg.read_search_strategy()))
    return parts
