"""Editing the settings that decide what the search looks for.

Writes back into ``config.local.toml`` by replacing values in place rather than
re-serialising the document, so every explanatory comment in the file survives
an edit made from the UI. A TOML round-trip library would flatten them, and
those comments are most of what makes the config readable.

Only the settings worth changing often live here. Deeper judgement - what
"good" means, the archetypes, the domain ranking - is prose in
``search-strategy.md``, which the scorer reads directly; that belongs in an
editor, not a form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

__all__ = [
    "SettingSpec",
    "SETTINGS",
    "SettingsError",
    "apply_edits",
    "current_values",
    "set_array",
    "set_or_insert_scalar",
    "set_scalar",
    "validate",
]


class SettingsError(ValueError):
    """A proposed settings change is not valid."""


@dataclass(frozen=True)
class SettingSpec:
    """One editable setting: where it lives and how to parse it."""

    key: str
    section: str
    label: str
    kind: str  # "int" | "float" | "bool" | "list" | "str" | "choice"
    help: str = ""
    #: For kind == "choice": the permitted values.
    choices: tuple[str, ...] = ()
    #: Insert the key under this section when the config file lacks it.
    insert_if_missing: bool = False

    @property
    def is_list(self) -> bool:
        return self.kind == "list"


SETTINGS: tuple[SettingSpec, ...] = (
    SettingSpec(
        "title_include", "discover", "Titles to match", "list",
        "A posting whose title contains any of these is considered.",
    ),
    SettingSpec(
        "title_exclude", "discover", "Titles to reject", "list",
        "Checked after the include list; a hit here drops the posting.",
    ),
    SettingSpec(
        "comp_floor_eur", "constraints", "Base compensation floor (EUR)", "int",
        "Only a stated range whose top is below this eliminates a role.",
    ),
    SettingSpec(
        "non_compete_waiver_signed", "constraints", "Non-compete waiver signed", "bool",
        "While false, competing employers are gated.",
    ),
    SettingSpec(
        "travel_max_percent", "constraints", "Maximum travel (%)", "int",
        "Roles advertising more than this are eliminated.",
    ),
    SettingSpec(
        "allowed_location_patterns", "constraints", "Workable locations", "list",
        "A location matching one of these passes.",
    ),
    SettingSpec(
        "blocked_location_patterns", "constraints", "Blocked locations", "list",
        "A match here fails unless a European anchor also matches.",
    ),
    SettingSpec(
        "shortlist_threshold", "scoring", "Shortlist threshold", "float",
        "Roles scoring below this are not worth an application.",
    ),
    SettingSpec(
        "provider", "claude", "Provider", "choice",
        "anthropic uses the native SDK with prompt caching; openai_compatible "
        "speaks chat-completions to any OpenAI-shaped endpoint.",
        choices=("anthropic", "openai_compatible"),
        insert_if_missing=True,
    ),
    SettingSpec(
        "model", "claude", "Model", "str",
        "Model id. For openai_compatible, whatever the endpoint calls it.",
        insert_if_missing=True,
    ),
    SettingSpec(
        "base_url", "claude", "Base URL", "str",
        "openai_compatible only, e.g. https://openrouter.ai/api/v1",
        insert_if_missing=True,
    ),
    SettingSpec(
        "api_key_env", "claude", "API key variable", "str",
        "openai_compatible only: the environment variable holding the key.",
        insert_if_missing=True,
    ),
)

WEIGHT_KEYS: tuple[str, ...] = ("buyer", "role_fit", "company", "domain", "talent")


# --- reading ---------------------------------------------------------------


def current_values(raw: dict[str, Any]) -> dict[str, Any]:
    """Pull the editable settings, plus the weights, out of a parsed config."""
    values: dict[str, Any] = {}
    for spec in SETTINGS:
        values[spec.key] = raw.get(spec.section, {}).get(spec.key)
    for key in WEIGHT_KEYS:
        values[f"weight_{key}"] = raw.get("weights", {}).get(key)
    return values


# --- validation ------------------------------------------------------------

_PARSERS: dict[str, Callable[[str], Any]] = {
    "int": lambda v: int(str(v).replace(",", "").replace("_", "").strip()),
    "float": lambda v: float(str(v).strip()),
    "bool": lambda v: str(v).strip().lower() in {"true", "yes", "y", "1", "on"},
    "str": lambda v: str(v).strip(),
}


def validate(spec: SettingSpec, value: Any) -> Any:
    """Coerce a form value to the spec's type, or raise SettingsError."""
    if spec.kind == "choice":
        value = str(value).strip().lower()
        if value not in spec.choices:
            raise SettingsError(
                f"{spec.label}: must be one of {', '.join(spec.choices)}, got {value!r}"
            )
        return value
    if spec.kind == "str":
        return str(value).strip()
    if spec.is_list:
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",")]
        items = [str(v).strip() for v in value if str(v).strip()]
        if not items:
            raise SettingsError(f"{spec.label} cannot be empty")
        return items
    try:
        parsed = _PARSERS[spec.kind](value)
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"{spec.label}: expected {spec.kind}, got {value!r}") from exc
    if spec.kind in {"int", "float"} and parsed < 0:
        raise SettingsError(f"{spec.label} cannot be negative")
    return parsed


def validate_weights(weights: dict[str, float]) -> dict[str, float]:
    """Weights must cover every dimension and sum to 1.0."""
    missing = set(WEIGHT_KEYS) - set(weights)
    if missing:
        raise SettingsError(f"missing weight(s): {', '.join(sorted(missing))}")
    try:
        parsed = {k: float(weights[k]) for k in WEIGHT_KEYS}
    except (TypeError, ValueError) as exc:
        raise SettingsError("weights must be numbers") from exc
    total = sum(parsed.values())
    if abs(total - 1.0) > 1e-6:
        raise SettingsError(f"weights must sum to 1.0, got {total:.4f}")
    return parsed


# --- writing ---------------------------------------------------------------


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def set_scalar(text: str, key: str, value: Any) -> str:
    """Replace the value of the first ``key = ...`` line, keeping its comment."""
    pattern = re.compile(rf"^(\s*){re.escape(key)}(\s*)=\s*[^#\n]*(#.*)?$", re.MULTILINE)

    def repl(match: re.Match[str]) -> str:
        indent, spacing, comment = match.group(1), match.group(2), match.group(3)
        tail = f"  {comment}" if comment else ""
        return f"{indent}{key}{spacing}= {_toml_value(value)}{tail}"

    new_text, count = pattern.subn(repl, text, count=1)
    if not count:
        raise SettingsError(f"{key} not found in the config file")
    return new_text


def set_array(text: str, key: str, values: Sequence[str]) -> str:
    """Replace a ``key = [ ... ]`` array, however many lines it spans."""
    start = re.search(rf"^(\s*){re.escape(key)}\s*=\s*\[", text, re.MULTILINE)
    if not start:
        raise SettingsError(f"{key} not found in the config file")
    depth, index = 0, start.end() - 1
    while index < len(text):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                break
        index += 1
    else:  # pragma: no cover - malformed config
        raise SettingsError(f"{key} array is not closed")

    indent = start.group(1)
    body = "".join(f'{indent}  {_toml_value(v)},\n' for v in values)
    rendered = f"{indent}{key} = [\n{body}{indent}]"
    return text[: start.start()] + rendered + text[index + 1 :]


def set_or_insert_scalar(text: str, section: str, key: str, value: Any) -> str:
    """Set ``key``, adding it under ``[section]`` when the file lacks it.

    A config written before a setting existed has no line to replace, and
    failing on that would make the setting uneditable from the UI.
    """
    try:
        return set_scalar(text, key, value)
    except SettingsError:
        pass
    header = re.search(rf"^\[{re.escape(section)}\]\s*$", text, re.MULTILINE)
    if not header:
        raise SettingsError(f"no [{section}] section in the config file")
    insert_at = header.end()
    return text[:insert_at] + f"\n{key} = {_toml_value(value)}" + text[insert_at:]


def apply_edits(
    text: str,
    edits: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> str:
    """Apply validated edits to the config text. Returns the new text."""
    by_key = {spec.key: spec for spec in SETTINGS}
    for key, value in edits.items():
        spec = by_key.get(key)
        if spec is None:
            raise SettingsError(f"unknown setting: {key}")
        checked = validate(spec, value)
        if spec.is_list:
            text = set_array(text, key, checked)
        elif spec.insert_if_missing:
            text = set_or_insert_scalar(text, spec.section, key, checked)
        else:
            text = set_scalar(text, key, checked)
    base_url = str(edits.get("base_url", "")).strip().rstrip("/").lower()
    if base_url.endswith(("/chat/completions", "/completions", "/responses")):
        raise SettingsError(
            "Base URL should be the API root, not the endpoint: the client appends "
            '"/chat/completions" itself. Use https://openrouter.ai/api/v1'
        )

    if str(edits.get("provider", "")).strip().lower() == "openai_compatible" and not str(
        edits.get("base_url", "")
    ).strip():
        raise SettingsError(
            'provider "openai_compatible" needs a Base URL, '
            "e.g. https://openrouter.ai/api/v1"
        )

    if weights:
        for key, value in validate_weights(weights).items():
            text = set_scalar(text, key, value)
    return text
