"""Configuration loading.

Everything the tool needs to know about *this* job search lives in a
``config.local.toml``.  Nothing is hardcoded to a particular machine: paths in the
config are resolved relative to the config file's own directory, so the repo is
portable as long as the user points ``[paths]`` at their own documents.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_NAME = "config.toml"
# Personal values live in config.local.toml, which is gitignored. The committed
# config.example.toml is a placeholder template, never loaded automatically.
LOCAL_CONFIG_NAME = "config.local.toml"
EXAMPLE_CONFIG_NAME = "config.example.toml"
ENV_CONFIG = "JOBSEARCH_CONFIG"


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or internally inconsistent."""


@dataclass(frozen=True)
class BoardRef:
    """A public ATS job board belonging to one target company."""

    company: str
    ats: str
    token: str
    tier: int = 3
    ind_sponsor: bool | str = "unknown"
    gaming: bool = False
    url: str | None = None  # for ats = "json_feed"

    @property
    def sponsor_state(self) -> str:
        """Normalise ``ind_sponsor`` to ``yes`` / ``no`` / ``unknown``."""
        if isinstance(self.ind_sponsor, bool):
            return "yes" if self.ind_sponsor else "no"
        return str(self.ind_sponsor).strip().lower() or "unknown"


@dataclass(frozen=True)
class Weights:
    """The weighted scoring rubric from ``search-strategy.md``."""

    buyer: float = 0.20
    role_fit: float = 0.25
    company: float = 0.25
    domain: float = 0.15
    talent: float = 0.15

    def __post_init__(self) -> None:
        total = self.buyer + self.role_fit + self.company + self.domain + self.talent
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"[weights] must sum to 1.0, got {total:.4f}")

    def as_dict(self) -> dict[str, float]:
        return {
            "buyer": self.buyer,
            "role_fit": self.role_fit,
            "company": self.company,
            "domain": self.domain,
            "talent": self.talent,
        }


@dataclass
class Config:
    """Parsed, path-resolved configuration."""

    root: Path
    raw: dict[str, Any]
    # The file this config was actually loaded from. Reported by `doctor`, which
    # must not guess the filename: it may be config.local.toml or config.toml.
    source: Path | None = None

    # -- resolved paths ----------------------------------------------------
    dossier: Path = field(init=False)
    base_cv: Path = field(init=False)
    search_strategy: Path = field(init=False)
    target_companies: Path = field(init=False)
    tracker_xlsx: Path = field(init=False)
    output_dir: Path = field(init=False)
    db_path: Path = field(init=False)

    weights: Weights = field(init=False)
    boards: list[BoardRef] = field(init=False)

    def __post_init__(self) -> None:
        paths = self.raw.get("paths", {})
        self.dossier = self._resolve(paths.get("dossier", "../career-dossier.md"))
        self.base_cv = self._resolve(paths.get("base_cv", "../cv.html"))
        self.search_strategy = self._resolve(paths.get("search_strategy", "../search-strategy.md"))
        self.target_companies = self._resolve(paths.get("target_companies", "../target-companies.md"))
        self.tracker_xlsx = self._resolve(paths.get("tracker_xlsx", "../job-search-tracker.xlsx"))
        self.output_dir = self._resolve(paths.get("output_dir", "output"))
        self.db_path = self._resolve(paths.get("db_path", "output/jobsearch.db"))

        self.weights = Weights(**self.raw.get("weights", {}))

        self.boards = []
        for entry in self.raw.get("discover", {}).get("boards", []):
            known = {f for f in BoardRef.__dataclass_fields__}
            self.boards.append(BoardRef(**{k: v for k, v in entry.items() if k in known}))

    # -- helpers -----------------------------------------------------------
    def _resolve(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.root / p).resolve()

    def section(self, name: str) -> dict[str, Any]:
        """Return a top-level config section, or an empty dict."""
        value = self.raw.get(name, {})
        return value if isinstance(value, dict) else {}

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.section(section).get(key, default)

    def board_for(self, company: str) -> BoardRef | None:
        """Case-insensitive lookup of a configured board by company name."""
        needle = company.strip().lower()
        for board in self.boards:
            if board.company.strip().lower() == needle:
                return board
        return None

    def reload(self) -> "Config":
        """Re-read the config file in place.

        Mutates this object rather than returning a new one: the TUI hands the
        same Config to every screen, so replacing it would leave stale copies
        behind after a settings change.
        """
        if self.source is None:
            raise ConfigError("this Config was not loaded from a file")
        with self.source.open("rb") as fh:
            self.raw = tomllib.load(fh)
        self.__post_init__()
        return self

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    # -- source documents --------------------------------------------------
    def read_dossier(self) -> str:
        return _read_required(self.dossier, "dossier")

    def read_base_cv(self) -> str:
        return _read_required(self.base_cv, "base_cv")

    def read_search_strategy(self) -> str:
        return _read_required(self.search_strategy, "search_strategy")

    def read_target_companies(self) -> str:
        return _read_required(self.target_companies, "target_companies")


def _read_required(path: Path, label: str) -> str:
    if not path.is_file():
        raise ConfigError(
            f"Configured [paths].{label} does not exist: {path}\n"
            f"Edit {LOCAL_CONFIG_NAME} and point it at your own document."
        )
    return path.read_text(encoding="utf-8")


def find_config(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate ``config.toml``.

    Order: explicit argument, ``$JOBSEARCH_CONFIG``, then the nearest
    ``config.toml`` walking up from the current directory.
    """
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise ConfigError(f"No config file at {p}")
        return p

    env = os.environ.get(ENV_CONFIG)
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_file():
            raise ConfigError(f"${ENV_CONFIG} points at a missing file: {p}")
        return p

    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        for name in (LOCAL_CONFIG_NAME, DEFAULT_CONFIG_NAME):
            p = candidate / name
            if p.is_file():
                return p

    # Fall back to the config shipped alongside the installed package's repo.
    repo_root = Path(__file__).resolve().parents[2]
    for name in (LOCAL_CONFIG_NAME, DEFAULT_CONFIG_NAME):
        packaged = repo_root / name
        if packaged.is_file():
            return packaged

    raise ConfigError(
        f"Could not find {LOCAL_CONFIG_NAME}. Copy {EXAMPLE_CONFIG_NAME} to "
        f"{LOCAL_CONFIG_NAME} and fill in your own details, then run from the "
        f"repo root, pass --config, or set ${ENV_CONFIG}."
    )


def load_config(explicit: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate configuration."""
    path = find_config(explicit)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return Config(root=path.parent, raw=raw, source=path)
