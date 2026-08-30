"""Guided first-run setup.

Writes ``config.local.toml`` by filling in the committed
``config.example.toml`` template, so the generated file keeps every explanatory
comment from the template rather than being a bare dump of values.

Deliberately stdlib-only: this is the one command that has to run before the
user has installed extras or written any configuration, so it must not depend
on anything that could itself be missing.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .config import EXAMPLE_CONFIG_NAME, LOCAL_CONFIG_NAME

# --- terminal helpers ------------------------------------------------------

_NO_COLOUR = bool(os.environ.get("NO_COLOR")) or not sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return text if _NO_COLOUR else f"\033[{code}m{text}\033[0m"


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def red(t: str) -> str:
    return _c("31", t)


def heading(text: str) -> None:
    print(f"\n{bold(text)}")


def ok(text: str) -> None:
    print(f"  {green('ok')}   {text}")


def warn(text: str) -> None:
    print(f"  {yellow('warn')} {text}")


def bad(text: str) -> None:
    print(f"  {red('miss')} {text}")


class SetupAborted(RuntimeError):
    """The user cancelled, or the terminal cannot support prompting."""


# --- prompting -------------------------------------------------------------


def ask(
    prompt: str,
    default: str = "",
    *,
    reader: Callable[[str], str] = input,
    validate: Callable[[str], str | None] | None = None,
) -> str:
    """Prompt until the answer validates. Empty input accepts ``default``."""
    suffix = f" {dim('[' + default + ']')}" if default else ""
    while True:
        try:
            raw = reader(f"  {prompt}{suffix}: ").strip()
        except EOFError as exc:  # piped stdin with nothing left to read
            raise SetupAborted(
                "setup needs an interactive terminal; run it directly, or edit "
                f"{LOCAL_CONFIG_NAME} by hand"
            ) from exc
        value = raw or default
        if validate is None:
            return value
        problem = validate(value)
        if problem is None:
            return value
        print(f"       {red(problem)}")


def ask_yes_no(prompt: str, default: bool = False, *, reader: Callable[[str], str] = input) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = ask(f"{prompt} ({hint})", "", reader=reader).strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print(f"       {red('please answer y or n')}")


def _validate_existing_path(value: str) -> str | None:
    if not value:
        return "a path is required"
    if not Path(value).expanduser().exists():
        return f"no such file: {value}"
    return None


def _validate_int(value: str) -> str | None:
    try:
        int(value.replace(",", "").replace("_", ""))
    except ValueError:
        return "must be a whole number"
    return None


# --- document discovery ----------------------------------------------------


@dataclass(frozen=True)
class Guess:
    """A best-guess default for one source document."""

    key: str
    label: str
    patterns: tuple[str, ...]
    fallback: str


GUESSES: tuple[Guess, ...] = (
    Guess("dossier", "career dossier (the fact base)", ("*dossier*.md", "*career*.md"), "../career-dossier.md"),
    Guess("base_cv", "base CV (HTML template)", ("*CV*.html", "*cv*.html", "*resume*.html"), "../cv.html"),
    Guess("search_strategy", "search strategy", ("*strategy*.md", "*search*.md"), "../search-strategy.md"),
    Guess("target_companies", "target companies", ("*target*.md", "*compan*.md"), "../target-companies.md"),
    Guess("tracker_xlsx", "existing tracker spreadsheet", ("*track*.xlsx", "*.xlsx"), "../tracker.xlsx"),
)


def guess_document(root: Path, guess: Guess) -> str:
    """Find the likeliest existing file for one document, as a relative path.

    Searches the repo's parent directory, which is where these documents
    normally live relative to a checkout.
    """
    search_dir = (root / "..").resolve()
    for pattern in guess.patterns:
        matches = sorted(
            (p for p in search_dir.glob(pattern) if p.is_file() and not p.name.startswith(".")),
            key=lambda p: (-p.stat().st_mtime, p.name),
        )
        if matches:
            try:
                return os.path.relpath(matches[0], root)
            except ValueError:  # different drive on Windows
                return str(matches[0])
    return guess.fallback


CHROME_CANDIDATES: tuple[str, ...] = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)


def find_chrome() -> str | None:
    """Locate a Chrome-family binary for headless PDF rendering."""
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for name in ("google-chrome", "chromium", "chromium-browser", "brave"):
        found = shutil.which(name)
        if found:
            return found
    return None


def claude_credentials_state() -> tuple[bool, str]:
    """Report whether the Anthropic SDK will find a credential, and from where."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True, "ANTHROPIC_API_KEY is set"
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True, "ANTHROPIC_AUTH_TOKEN is set"
    profile = Path.home() / ".config" / "anthropic" / "credentials"
    if profile.is_dir() and any(profile.glob("*.json")):
        return True, f"stored profile in {profile}"
    return False, "no API key and no stored profile"


# --- template rendering ----------------------------------------------------


def _toml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_config(template: str, answers: dict[str, str]) -> str:
    """Substitute answers into the example template, preserving its comments.

    Replaces the value on each ``key = ...`` line rather than rewriting the
    file, so every explanatory comment survives into the user's own config.
    """
    replacements: dict[str, str] = {
        "dossier": _toml_str(answers["dossier"]),
        "base_cv": _toml_str(answers["base_cv"]),
        "search_strategy": _toml_str(answers["search_strategy"]),
        "target_companies": _toml_str(answers["target_companies"]),
        "tracker_xlsx": _toml_str(answers["tracker_xlsx"]),
        "name": _toml_str(answers["name"]),
        "email": _toml_str(answers["email"]),
        "location": _toml_str(answers["location"]),
        "linkedin": _toml_str(answers["linkedin"]),
        "current_title": _toml_str(answers["current_title"]),
        "current_company": _toml_str(answers["current_company"]),
        "comp_floor_eur": answers["comp_floor_eur"],
        "non_compete_waiver_signed": answers["non_compete_waiver_signed"],
        "chrome_binary": _toml_str(answers["chrome_binary"]),
        "user_agent": _toml_str(answers["user_agent"]),
    }
    seen: set[str] = set()
    out: list[str] = []
    for line in template.splitlines():
        stripped = line.lstrip()
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            # Only the first occurrence: later sections reuse some key names.
            if key in replacements and key not in seen:
                seen.add(key)
                indent = line[: len(line) - len(stripped)]
                out.append(f"{indent}{key} = {replacements[key]}")
                continue
        out.append(line)
    return "\n".join(out) + "\n"


# --- the wizard ------------------------------------------------------------


def run_setup(
    repo_root: Path,
    *,
    force: bool = False,
    reader: Callable[[str], str] = input,
    check_boards: Callable[[], Iterable[str]] | None = None,
) -> int:
    """Interactively write ``config.local.toml``. Returns a process exit code."""
    template_path = repo_root / EXAMPLE_CONFIG_NAME
    target = repo_root / LOCAL_CONFIG_NAME

    if not template_path.is_file():
        print(f"error: no {EXAMPLE_CONFIG_NAME} beside {repo_root}", file=sys.stderr)
        return 2

    print(bold("\njobsearch setup"))
    print(dim(f"  writes {target}"))
    print(dim("  press Enter to accept each [default]"))

    if target.exists() and not force:
        print()
        if not ask_yes_no(f"{LOCAL_CONFIG_NAME} already exists. Overwrite it?", False, reader=reader):
            print("\n  nothing changed")
            return 0

    answers: dict[str, str] = {}

    heading("Source documents")
    print(dim("  Paths are stored relative to the config file."))
    for guess in GUESSES:
        default = guess_document(repo_root, guess)
        exists = (repo_root / default).exists() if not Path(default).is_absolute() else Path(default).exists()
        if exists:
            ok(f"found {default}")
        answers[guess.key] = ask(guess.label, default, reader=reader)

    heading("About you")
    answers["name"] = ask("full name", "", reader=reader)
    answers["email"] = ask("email", "", reader=reader)
    answers["location"] = ask("where you are based", "", reader=reader)
    answers["linkedin"] = ask("LinkedIn URL", "", reader=reader)
    answers["current_title"] = ask("current title", "", reader=reader)
    answers["current_company"] = ask("current employer", "", reader=reader)
    answers["user_agent"] = (
        f"jobsearch-agent/0.1 (personal job search; +mailto:{answers['email'] or 'you@example.com'})"
    )

    heading("Hard constraints")
    print(dim("  These eliminate roles before anything is scored or sent to the API."))
    answers["comp_floor_eur"] = str(
        int(
            ask(
                "base compensation floor (EUR)",
                "100000",
                reader=reader,
                validate=_validate_int,
            )
            .replace(",", "")
            .replace("_", "")
        )
    )
    answers["non_compete_waiver_signed"] = (
        "true" if ask_yes_no("is your non-compete waiver signed?", False, reader=reader) else "false"
    )

    heading("Environment")
    chrome = find_chrome()
    if chrome:
        ok(f"Chrome for PDF rendering: {chrome}")
    else:
        bad("no Chrome-family browser found - PDF rendering will fail")
    answers["chrome_binary"] = ask(
        "Chrome binary", chrome or CHROME_CANDIDATES[0], reader=reader
    )

    have_creds, why = claude_credentials_state()
    if have_creds:
        ok(f"Claude credentials: {why}")
    else:
        bad(f"Claude credentials: {why}")
        print(dim("       run `ant auth login`, or export ANTHROPIC_API_KEY"))

    target.write_text(render_config(template_path.read_text(encoding="utf-8"), answers), encoding="utf-8")

    heading("Done")
    ok(f"wrote {target}")

    missing = [
        key
        for key in ("dossier", "base_cv", "search_strategy", "target_companies")
        if not (repo_root / answers[key]).exists()
    ]
    if missing:
        warn(f"these paths do not exist yet: {', '.join(missing)}")
        print(dim(f"       edit {LOCAL_CONFIG_NAME} once the documents are in place"))

    if check_boards is not None and ask_yes_no("\n  Verify board tokens against their live APIs now?", False, reader=reader):
        for line in check_boards():
            print(f"  {line}")

    print(f"\n  next: {bold('jobsearch doctor')}  then  {bold('jobsearch tui')}\n")
    return 0


def default_repo_root() -> Path:
    """The directory holding the packaged template."""
    return Path(__file__).resolve().parents[2]
