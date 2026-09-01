"""An OpenAI-compatible provider: OpenRouter, Together, Groq, vLLM, Ollama.

A sibling of :mod:`jobsearch.claude`, not a replacement for it. The Anthropic
path keeps its native SDK, structured outputs and explicit ``cache_control``;
this one speaks the OpenAI chat-completions dialect. Both satisfy the same
small interface, so every stage calls one or the other without knowing which.

Two behavioural differences are worth knowing before switching:

* **Caching is not explicit.** The Anthropic path marks the dossier and base CV
  as a cached prefix and can prove a hit from ``cache_read_input_tokens``. Here
  caching is whatever the endpoint does on its own - often nothing. The prefix
  is ~27KB and is resent on every call, so the per-role cost can be higher on a
  nominally cheaper model.
* **Structured output is best-effort.** Endpoints vary in whether they honour
  ``response_format``. This sends the JSON-schema form, and falls back to
  extracting the first JSON object from the text when the endpoint ignores it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .claude import ClaudeError, Usage

__all__ = [
    "OpenAICompatibleClient",
    "extract_json",
    "normalise_base_url",
    "supports_explicit_cache",
]

#: Model prefixes whose providers need explicit ``cache_control`` breakpoints.
#: Everything else on OpenRouter that caches at all does so automatically
#: (OpenAI, DeepSeek, Grok, Groq, Moonshot, Z.AI, Gemini 2.5+), and only needs
#: a byte-identical prefix, which `messages_for` already guarantees.
EXPLICIT_CACHE_PREFIXES: tuple[str, ...] = ("anthropic/", "qwen/", "alibaba/")


def _default_session_id(cfg: Any) -> str:
    """A stable sticky-session key for this install.

    Derived from the config path so it survives restarts and stays the same
    across stages: the goal is one provider for all of this tool's traffic, not
    one per conversation.
    """
    import hashlib

    seed = str(getattr(cfg, "source", "") or getattr(cfg, "root", "jobsearch"))
    return "jobsearch-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def supports_explicit_cache(model: str) -> bool:
    """True when this model's provider requires cache_control breakpoints."""
    return str(model or "").strip().lower().startswith(EXPLICIT_CACHE_PREFIXES)

#: Paths the SDK appends itself. Pasting the full endpoint URL from a
#: provider's docs is the obvious mistake, and it produces a 404 that says
#: nothing about the cause.
_SDK_APPENDED_PATHS = (
    "/chat/completions",
    "/completions",
    "/v1/chat/completions",
    "/responses",
)


def normalise_base_url(url: str) -> str:
    """Strip a trailing endpoint path from a base URL.

    The OpenAI SDK appends "/chat/completions" itself, so a base_url that
    already ends in it requests ".../chat/completions/chat/completions" and
    404s. Providers document the full endpoint, so this is the natural thing
    to paste.
    """
    cleaned = (url or "").strip().rstrip("/")
    for path in _SDK_APPENDED_PATHS:
        if cleaned.lower().endswith(path):
            cleaned = cleaned[: -len(path)].rstrip("/")
            break
    return cleaned

JSON_INSTRUCTION = (
    "Respond with a single JSON object matching the schema below. "
    "No prose, no markdown fence, no commentary.\n\nSchema:\n"
)


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a response, fence or prose notwithstanding.

    Endpoints that ignore ``response_format`` tend to wrap the object in a
    ```json fence or introduce it with a sentence.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*\n", "", stripped)
        stripped = re.sub(r"\n```\s*$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    if start != -1:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start : index + 1])
                    except json.JSONDecodeError:
                        break
    raise ClaudeError(f"response was not valid JSON: {stripped[:200]}")


@dataclass
class OpenAICompatibleClient:
    """Chat-completions client with the same surface as ``ClaudeClient``."""

    model: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 16000
    streaming_max_tokens: int = 64000
    temperature: float | None = None
    cache_ttl: str = "1h"
    #: "auto" enables cache_control for providers that need it, off otherwise.
    explicit_cache: str = "auto"
    #: OpenRouter provider routing, e.g. {"only": ["DeepInfra"]}. A popular
    #: model can be served by twenty upstreams; a prompt cache is warmed on one
    #: of them, so without pinning consecutive calls land on different machines
    #: and every request is a cache miss.
    provider_routing: dict[str, Any] = field(default_factory=dict)
    #: OpenRouter sticky-session key. Without it, sticky routing only engages
    #: *after* a cache hit is observed, which never happens when the first call
    #: warms a cache the second call does not reach. With it, the provider is
    #: pinned from the first successful request. Preferred over pinning a
    #: single provider, which forfeits failover.
    session_id: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)
    stage_overrides: dict[str, dict[str, str]] = field(default_factory=dict)
    dry_run: bool = False
    dry_run_hook: Callable[[str, str], None] | None = None
    _client: Any = field(default=None, repr=False)
    _last_usage: Usage = field(default_factory=Usage, repr=False)

    # -- lifecycle ---------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:
                raise ClaudeError(
                    "An OpenAI-compatible endpoint needs the optional extra: "
                    "pip install -e '.[openai]'"
                ) from exc
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ClaudeError(
                    f"${self.api_key_env} is not set. Export it, or point "
                    f"[claude].api_key_env at the variable that holds the key."
                )
            self._client = OpenAI(
                api_key=key, base_url=self.base_url, default_headers=self.extra_headers or None
            )
        return self._client

    @classmethod
    def from_config(cls, cfg, dry_run: bool = False) -> "OpenAICompatibleClient":
        section = cfg.section("claude")
        base_url = section.get("base_url")
        if not base_url:
            raise ClaudeError(
                'provider = "openai_compatible" needs [claude].base_url, '
                'e.g. "https://openrouter.ai/api/v1"'
            )
        return cls(
            model=section.get("model", ""),
            base_url=normalise_base_url(str(base_url)),
            api_key_env=section.get("api_key_env", "OPENAI_API_KEY"),
            max_tokens=int(section.get("max_tokens", 16000)),
            streaming_max_tokens=int(section.get("streaming_max_tokens", 64000)),
            temperature=section.get("temperature"),
            cache_ttl=section.get("cache_ttl", "1h"),
            explicit_cache=str(section.get("explicit_cache", "auto")),
            provider_routing=dict(section.get("provider_routing", {}) or {}),
            session_id=str(section.get("session_id") or _default_session_id(cfg)),
            extra_headers=dict(section.get("extra_headers", {}) or {}),
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

    def effort_for(self, stage: str) -> str:  # pragma: no cover - not supported here
        """Effort has no equivalent in this dialect; reported for symmetry."""
        return str(self.stage_overrides.get(stage, {}).get("effort", "n/a"))

    @property
    def last_usage(self) -> Usage:
        return self._last_usage

    def uses_explicit_cache(self, stage: str) -> bool:
        """Whether to send cache_control breakpoints for this stage's model."""
        setting = str(self.explicit_cache).strip().lower()
        if setting in {"true", "yes", "on", "always"}:
            return True
        if setting in {"false", "no", "off", "never"}:
            return False
        return supports_explicit_cache(self.model_for(stage))

    def messages_for(
        self,
        instructions: str,
        stable_context: Sequence[tuple[str, str]],
        user_content: str,
        explicit_cache: bool = False,
    ) -> list[dict[str, Any]]:
        """Build the request messages, prefix first.

        Without ``explicit_cache`` the stable prefix is concatenated into one
        system message in fixed order. That is enough for the providers that
        cache automatically: they only need the prefix to be byte-identical
        between calls.

        With it, the prefix becomes a cache_control content block, which is
        what Anthropic and Qwen require through OpenRouter. Marking the prefix
        rather than the whole message is the point: the volatile per-role text
        must fall outside the breakpoint or nothing ever hits.
        """
        prefix = "\n\n".join(
            f"<{label}>\n{body}\n</{label}>" for label, body in stable_context
        )
        if not explicit_cache:
            parts = [instructions] + ([prefix] if prefix else [])
            return [
                {"role": "system", "content": "\n\n".join(parts)},
                {"role": "user", "content": user_content},
            ]

        blocks: list[dict[str, Any]] = []
        if prefix:
            blocks.append({
                "type": "text",
                "text": prefix,
                "cache_control": {"type": "ephemeral", "ttl": self.cache_ttl},
            })
        blocks.append({"type": "text", "text": user_content})
        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": blocks},
        ]

    def _params(self, stage: str, max_tokens: int) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model_for(stage), "max_tokens": max_tokens}
        if self.temperature is not None:
            params["temperature"] = float(self.temperature)
        extra: dict[str, Any] = {}
        if self.provider_routing:
            extra["provider"] = self.provider_routing
        if self.session_id:
            extra["session_id"] = self.session_id[:256]
        # Ask for detailed usage so cache hits are reported at all.
        extra["usage"] = {"include": True}
        params["extra_body"] = extra
        return params

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
        if self.dry_run:
            self._note_dry_run(stage, user_content)
            return dry_run_value if dry_run_value is not None else {}

        messages = self.messages_for(
            instructions + "\n\n" + JSON_INSTRUCTION + json.dumps(schema, indent=2),
            stable_context,
            user_content,
            explicit_cache=self.uses_explicit_cache(stage),
        )
        response = self.client.chat.completions.create(
            **self._params(stage, max_tokens or self.max_tokens),
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema, "strict": False},
            },
        )
        self._record_usage(response)
        text = (response.choices[0].message.content or "") if response.choices else ""
        if not text.strip():
            raise ClaudeError(f"{stage}: endpoint returned no content")
        return extract_json(text)

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
        if self.dry_run:
            self._note_dry_run(stage, user_content)
            return dry_run_value

        stream = self.client.chat.completions.create(
            **self._params(stage, max_tokens or self.streaming_max_tokens),
            messages=self.messages_for(
                instructions, stable_context, user_content,
                explicit_cache=self.uses_explicit_cache(stage),
            ),
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks: list[str] = []
        for event in stream:
            if getattr(event, "usage", None):
                self._record_usage(event)
            for choice in getattr(event, "choices", None) or []:
                piece = getattr(choice.delta, "content", None)
                if piece:
                    chunks.append(piece)
                    if on_delta:
                        on_delta(piece)
        text = "".join(chunks)
        if not text.strip():
            raise ClaudeError(f"{stage}: endpoint returned no content")
        return text

    # -- bookkeeping -------------------------------------------------------
    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        cached = written = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
            written = int(getattr(details, "cache_write_tokens", 0) or 0)
        self._last_usage = Usage(
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cache_read_input_tokens=cached,
            cache_creation_input_tokens=written,
        )

    def _note_dry_run(self, stage: str, prompt: str) -> None:
        if self.dry_run_hook:
            self.dry_run_hook(stage, prompt[:200])
