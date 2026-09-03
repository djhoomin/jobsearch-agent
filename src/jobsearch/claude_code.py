"""A provider that runs the local Claude Code CLI headlessly.

A third sibling of :mod:`jobsearch.claude` and :mod:`jobsearch.openai_compat`,
satisfying the same small interface, so every stage calls one of the three
without knowing which. What it buys is billing: work goes through whatever the
`claude` binary is already authenticated with, which for a subscription means
included usage rather than metered API spend.

Facts this module is built on, all verified against the CLI rather than
assumed:

* **Structured output is real.** ``--json-schema`` validates server-side and the
  result envelope carries a parsed ``structured_output`` object, so there is no
  need for the text-scraping fallback the OpenAI-compatible path needs.
* **Prompt caching survives across invocations.** Each ``-p`` run is a separate
  process, but the ~40KB dossier prefix still hits the cache when the system
  prompt *and* the JSON schema are byte-identical between calls, at the 1h TTL.
  Measured: 9.5K tokens read from cache on the second call, list cost falling
  from $0.108 to $0.012. That is why the prefix is assembled with exactly the
  same block format as the Anthropic path, and why the schema is serialised
  deterministically.
* **stdin must be closed.** Without it the CLI waits 3s for piped input on every
  call and says so on stderr.
* **``--system-prompt`` replaces rather than appends.** The coding-agent persona,
  the CLAUDE.md discovery and the dynamic environment preamble are all gone, so
  the model sees the same instructions the API path sends it.
* **``--system-prompt-file`` exists** but is absent from ``--help``. It is used
  here because a 40KB dossier does not belong in argv.

Deliberate limitations, stated rather than hidden:

* ``max_tokens`` has no CLI equivalent. The model's own default output ceiling
  applies, and a truncated generation is detected from ``stop_reason`` instead.
* Every call pays process startup, roughly a second, and a small Haiku
  classifier call the CLI makes on its own.
* ``total_cost_usd`` in the envelope is list price for the tokens used. On a
  subscription nothing is billed, so it is reported as *avoided* cost.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .claude import ClaudeError, Usage

__all__ = ["ClaudeCodeClient", "ClaudeCodeError"]

log = logging.getLogger(__name__)

MODEL = "claude-opus-5"

#: Flags sent on every call. Each one is load-bearing:
#: ``--print`` is headless mode; ``--tools ""`` removes the built-in toolset so
#: the model can only answer; ``--safe-mode`` drops CLAUDE.md, skills, plugins,
#: hooks and MCP servers while leaving authentication alone (``--bare`` would
#: also drop OAuth, which is the thing we came for); ``--strict-mcp-config``
#: covers MCP servers configured outside settings; ``--no-session-persistence``
#: keeps a job search out of the resume picker.
BASE_ARGS: tuple[str, ...] = (
    "--print",
    "--tools", "",
    "--safe-mode",
    "--strict-mcp-config",
    "--no-session-persistence",
)


class ClaudeCodeError(ClaudeError):
    """A headless CLI failure the CLI should report, not traceback on."""


_PROMPT_DIR: tempfile.TemporaryDirectory | None = None


def _prompt_file(text: str) -> Path:
    """Write ``text`` to a content-addressed temp file and return the path.

    Content-addressed so a repeated prefix is written once per process, and so
    two calls with the same prefix are provably sending the same bytes - which
    is the whole precondition for a cache hit.
    """
    global _PROMPT_DIR
    if _PROMPT_DIR is None:
        _PROMPT_DIR = tempfile.TemporaryDirectory(prefix="jobsearch-cc-")
        atexit.register(_PROMPT_DIR.cleanup)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    path = Path(_PROMPT_DIR.name) / f"system-{digest}.txt"
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    return path


@dataclass
class ClaudeCodeClient:
    """Runs each stage as one headless ``claude -p`` invocation."""

    binary: str = "claude"
    model: str = MODEL
    effort: str = "high"
    #: A full CV generation runs for minutes. Generous, but not unbounded.
    timeout: int = 900
    #: Remove ``ANTHROPIC_API_KEY`` from the child environment. On by default:
    #: with a key present the CLI bills the API, which defeats the point of
    #: routing through the subscription in the first place.
    use_subscription: bool = True
    #: Appended verbatim to every invocation, for anything not modelled here.
    extra_args: list[str] = field(default_factory=list)
    stage_overrides: dict[str, dict[str, str]] = field(default_factory=dict)
    dry_run: bool = False
    dry_run_hook: Callable[[str, str], None] | None = None
    _last_usage: Usage = field(default_factory=Usage, repr=False)
    _last_cost_usd: float = field(default=0.0, repr=False)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_config(cls, cfg, dry_run: bool = False) -> "ClaudeCodeClient":
        section = cfg.section("claude")
        return cls(
            binary=str(section.get("binary", "claude")),
            model=str(section.get("model", MODEL)),
            effort=str(section.get("effort", "high")),
            timeout=int(section.get("timeout", 900)),
            use_subscription=bool(section.get("use_subscription", True)),
            extra_args=[str(a) for a in (section.get("extra_args", []) or [])],
            stage_overrides={
                str(name): dict(values)
                for name, values in (section.get("stages", {}) or {}).items()
                if isinstance(values, dict)
            },
            dry_run=dry_run,
        )

    # -- parity with ClaudeClient -----------------------------------------
    def model_for(self, stage: str) -> str:
        return str(self.stage_overrides.get(stage, {}).get("model", self.model))

    def effort_for(self, stage: str) -> str:
        return str(self.stage_overrides.get(stage, {}).get("effort", self.effort))

    @property
    def last_usage(self) -> Usage:
        return self._last_usage

    @property
    def last_cost_usd(self) -> float:
        """List price of the last call. Not billed on a subscription."""
        return self._last_cost_usd

    def system_prompt(
        self,
        instructions: str,
        stable_context: Sequence[tuple[str, str]] = (),
    ) -> str:
        """The full system prompt, block for block as the Anthropic path sends it.

        Same wrapper tags and same fixed order, so a prefix that caches on one
        provider caches on the other and the two are comparable.
        """
        parts = [instructions]
        parts.extend(f"<{label}>\n{text}\n</{label}>" for label, text in stable_context)
        return "\n\n".join(parts)

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

        ``max_tokens`` is accepted for interface parity and ignored: the CLI
        exposes no equivalent. Truncation is caught from ``stop_reason``.
        """
        if self.dry_run:
            self._note_dry_run(stage, user_content)
            return dry_run_value if dry_run_value is not None else {}

        envelope = self._run(
            stage,
            user_content,
            instructions,
            stable_context,
            extra=[
                "--output-format", "json",
                # sort_keys so an equal schema always serialises to equal bytes;
                # the schema lands in the cached prefix as a tool definition.
                "--json-schema", json.dumps(schema, sort_keys=True),
            ],
        )
        payload = envelope.get("structured_output")
        if isinstance(payload, dict):
            return payload
        text = str(envelope.get("result") or "")
        if not text.strip():
            raise ClaudeCodeError(f"{stage}: CLI returned no result text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ClaudeCodeError(
                f"{stage}: response was not valid JSON: {exc}"
            ) from exc

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

        chunks: list[str] = []

        def handle(event: dict[str, Any]) -> None:
            if event.get("type") != "stream_event":
                return
            inner = event.get("event") or {}
            if inner.get("type") != "content_block_delta":
                return
            delta = inner.get("delta") or {}
            if delta.get("type") != "text_delta":
                return
            text = str(delta.get("text") or "")
            if text:
                chunks.append(text)
                if on_delta:
                    on_delta(text)

        envelope = self._run(
            stage,
            user_content,
            instructions,
            stable_context,
            # --verbose is not optional here: the CLI refuses stream-json
            # without it. It only adds session-init lines, which are ignored.
            extra=[
                "--output-format", "stream-json",
                "--include-partial-messages",
                "--verbose",
            ],
            on_event=handle,
        )
        if envelope.get("stop_reason") == "max_tokens":
            raise ClaudeCodeError(
                f"{stage}: generation hit the model's output ceiling. The CLI "
                "exposes no max_tokens flag; split the stage or use "
                'provider = "anthropic" for it.'
            )
        # Prefer the accumulated deltas; fall back to the envelope for a run
        # that produced no partial events.
        return "".join(chunks) or str(envelope.get("result") or "")

    # -- process handling --------------------------------------------------
    def _argv(self, stage: str, system_file: Path, extra: Sequence[str]) -> list[str]:
        argv = [self._binary_path(), *BASE_ARGS]
        argv += ["--model", self.model_for(stage)]
        effort = self.effort_for(stage)
        if effort and effort != "n/a":
            argv += ["--effort", effort]
        argv += ["--system-prompt-file", str(system_file)]
        argv += list(extra)
        argv += self.extra_args
        return argv

    def _binary_path(self) -> str:
        found = shutil.which(self.binary)
        if not found:
            raise ClaudeCodeError(
                f"Claude Code CLI not found on PATH as {self.binary!r}. Install it, "
                'or set [claude].binary to its full path, or use provider = "anthropic".'
            )
        return found

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.use_subscription:
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("ANTHROPIC_AUTH_TOKEN", None)
        return env

    def _run(
        self,
        stage: str,
        user_content: str,
        instructions: str,
        stable_context: Sequence[tuple[str, str]],
        *,
        extra: Sequence[str],
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Invoke the CLI once and return the parsed result envelope."""
        system_file = _prompt_file(self.system_prompt(instructions, stable_context))
        argv = self._argv(stage, system_file, extra)
        # The prompt goes on stdin rather than argv: it can be long, and argv is
        # visible in `ps` to every other user on the machine.
        log.debug("%s: %s", stage, " ".join(argv))
        try:
            proc = subprocess.run(
                argv,
                input=user_content,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                # A temp cwd, so nothing in the working tree can be picked up.
                cwd=_PROMPT_DIR.name if _PROMPT_DIR else None,
                env=self._env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCodeError(
                f"{stage}: the CLI did not finish within {self.timeout}s. Raise "
                "[claude].timeout."
            ) from exc
        except OSError as exc:
            raise ClaudeCodeError(f"{stage}: could not run the CLI. {exc}") from exc

        envelope = self._envelope(stage, proc, on_event)
        self._record(envelope)
        return envelope

    def _envelope(
        self,
        stage: str,
        proc: subprocess.CompletedProcess,
        on_event: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        """Find the result object in stdout, dispatching stream events on the way."""
        result: dict[str, Any] | None = None
        limited = False
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "rate_limit_event":
                limited = True
            kind = event.get("type")
            if kind == "result" or (kind is None and "result" in event):
                result = event
            elif on_event:
                on_event(event)

        if result is None:
            detail = (proc.stderr or proc.stdout or "").strip()[-400:]
            hint = " Usage limits may be exhausted." if limited else ""
            raise ClaudeCodeError(
                f"{stage}: the CLI produced no result (exit {proc.returncode}).{hint} "
                f"{detail}"
            )
        if result.get("is_error"):
            raise ClaudeCodeError(
                f"{stage}: {result.get('subtype') or 'error'}. "
                f"{str(result.get('result') or '')[:400]}"
            )
        return result

    def _record(self, envelope: dict[str, Any]) -> None:
        usage = envelope.get("usage") or {}
        self._last_usage = Usage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        )
        self._last_cost_usd = float(envelope.get("total_cost_usd") or 0.0)
        log.debug(
            "usage: %s  list_cost=$%.4f", self._last_usage.describe(), self._last_cost_usd
        )

    def _note_dry_run(self, stage: str, prompt: str) -> None:
        summary = prompt.strip().replace("\n", " ")[:160]
        if self.dry_run_hook:
            self.dry_run_hook(stage, summary)
        else:
            log.info("[dry-run] would run claude -p for %s: %s...", stage, summary)
