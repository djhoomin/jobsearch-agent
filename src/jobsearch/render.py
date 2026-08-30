"""HTML -> PDF rendering with headless Chrome.

Chrome is used rather than a Python PDF library because the base CV's layout is
CSS (``@page``, flexbox, print styles) and because Chrome's text layer is what
the ATS verifier is calibrated against. The exact invocation is the one verified
working on this machine::

    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \\
        --headless --disable-gpu --no-pdf-header-footer \\
        --print-to-pdf=OUT.pdf file://IN.html
"""

from __future__ import annotations

import shutil
import subprocess

from pathlib import Path
from typing import Sequence

DEFAULT_ARGS = ("--headless", "--disable-gpu", "--no-pdf-header-footer")

#: Tried in order when the configured binary is missing.
FALLBACK_BINARIES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
)


class RenderError(RuntimeError):
    """Chrome could not produce a PDF."""


def find_chrome(configured: str | None = None) -> str:
    """Locate a usable Chrome/Chromium binary."""
    candidates = [configured, *FALLBACK_BINARIES] if configured else list(FALLBACK_BINARIES)
    for candidate in candidates:
        if not candidate:
            continue
        if Path(candidate).is_file():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise RenderError(
        "No Chrome or Chromium binary found. Set [render].chrome_binary in "
        "config.toml to the full path of your browser executable."
    )


def html_to_pdf(
    html_path: str | Path,
    pdf_path: str | Path,
    *,
    chrome_binary: str | None = None,
    extra_args: Sequence[str] = (),
    timeout: float = 120.0,
) -> Path:
    """Render ``html_path`` to ``pdf_path``. Returns the PDF path."""
    html_path = Path(html_path).resolve()
    pdf_path = Path(pdf_path).resolve()
    if not html_path.is_file():
        raise RenderError(f"No such HTML file: {html_path}")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    binary = find_chrome(chrome_binary)
    args = list(extra_args) if extra_args else list(DEFAULT_ARGS)

    # Note: do NOT add --user-data-dir here. It looks like good hygiene but it
    # makes headless Chrome hang indefinitely on macOS instead of printing.
    # The flag set below is the one verified working on this machine.
    command = [binary, *args, f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise RenderError(f"Chrome timed out after {timeout}s rendering {html_path}") from exc

    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise RenderError(
            f"Chrome exited {result.returncode} without producing a PDF.\n"
            f"stderr: {result.stderr.strip()[:800]}"
        )
    return pdf_path


def render_from_config(cfg, html_path: str | Path, pdf_path: str | Path) -> Path:
    """Render using the binary and flags from ``config.toml``."""
    section = cfg.section("render")
    return html_to_pdf(
        html_path,
        pdf_path,
        chrome_binary=section.get("chrome_binary"),
        extra_args=section.get("chrome_args", DEFAULT_ARGS),
        timeout=float(section.get("timeout_seconds", 120)),
    )
