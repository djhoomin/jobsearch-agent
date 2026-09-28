"""Find job boards worth adding by sweeping the IND register of recognised sponsors.

Every employer on the register can sponsor a highly skilled migrant, but the
register carries no sector, so on its own it is ~13,000 names of noise. The
filter that works is cheap: guess each company's board token on the ATS
providers discovery can read (Ashby, Greenhouse, Lever), and keep the ones that
exist and list at least one role passing the title and location filters today.
A company with a public ATS board is usually a tech company or scale-up, which
is most of the targeting done for free.

    python -m jobsearch.sponsor_sweep            # full sweep, resumable
    python -m jobsearch.sponsor_sweep --limit 200

Output is a ranked CSV for review. It never edits config.local.toml: adding
300 boards would bury the good roles, so a person picks from the shortlist.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .config import BoardRef, Config, load_config
from .discover import filter_locations, filter_postings
from .discover.sources import PROBE_ATS, DiscoveryError, Fetcher, fetch_board

log = logging.getLogger(__name__)
_WRITE_LOCK = threading.Lock()  # three provider threads share one checkpoint file

REGISTER_URL = (
    "https://ind.nl/en/public-register-recognised-sponsors/"
    "public-register-regular-labour-and-highly-skilled-migrants"
)
# The IND site serves a stub to non-browser clients.
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# Names that are almost never a tech employer with an AI leadership opening.
SKIP_WORDS = (
    "stichting", "gemeente", "provincie", "ministerie", "universiteit", "university",
    "hogeschool", "school", "college", "academisch", "umc", "ziekenhuis", "hospital",
    "zorg", "care", "kliniek", "clinic", "tandarts", "dental", "huisarts", "apotheek",
    "pharmacy", "uitzend", "detachering", "staffing", "recruitment", "recruiting",
    "werving", "payroll", "employer of record", "vereniging", "church", "kerk",
    "restaurant", "horeca", "catering", "hotel", "bakkerij", "slagerij", "transport",
    "logistiek", "bouw", "construction", "installatie", "schoonmaak", "cleaning",
    "kinderopvang", "sportschool", "orkest", "theater", "museum",
)
LEGAL_SUFFIXES = re.compile(
    r"\b(b\.?\s?v\.?|n\.?\s?v\.?|v\.?\s?o\.?\s?f\.?|c\.?\s?v\.?|ltd|limited|inc|gmbh|s\.?a\.?|"
    r"holding|holdings|nederland|netherlands|the netherlands|europe|european|emea|"
    r"international|benelux|group|groep|services|branch|filiaal|nl)\b\.?",
    re.I,
)


@dataclass
class Company:
    name: str
    kvk: str
    base: str


def output_dir(cfg: Config) -> Path:
    db = Path(cfg.section("tracker").get("db_path", "output/jobsearch.db"))
    return db.parent


def load_register(cache: Path, refresh: bool = False, max_age_days: int = 7) -> list[Company]:
    """Parse the register table, caching the page so reruns do not refetch it."""
    stale = not cache.exists() or (time.time() - cache.stat().st_mtime) > max_age_days * 86400
    if refresh or stale:
        request = urllib.request.Request(REGISTER_URL, headers=BROWSER_HEADERS)
        with urllib.request.urlopen(request, timeout=60) as response:
            page = response.read().decode("utf-8", errors="replace")
        if "<table" not in page:
            raise RuntimeError("IND register page came back without a table; the site may be blocking")
        cache.write_text(page)
    page = cache.read_text()
    companies = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
        if len(cells) < 2 or cells[0] == "Organisation":
            continue
        name = cells[0].replace('""', '"').strip('" ')
        companies.append(Company(name=name, kvk=cells[1], base=base_name(name)))
    return companies


def base_name(name: str) -> str:
    base = re.sub(r"\(.*?\)", " ", name.lower())
    base = LEGAL_SUFFIXES.sub(" ", base)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9&\-. ]", " ", base)).strip(" .-&")


def worth_probing(company: Company) -> bool:
    lowered = company.name.lower()
    return len(company.base) >= 3 and not any(re.search(rf"\b{w}", lowered) for w in SKIP_WORDS)


def slug_candidates(base: str) -> list[str]:
    """Joined, hyphenated, and (for multi-word names) the first word if distinctive."""
    words = re.findall(r"[a-z0-9]+", base)
    if not words:
        return []
    slugs = ["".join(words), "-".join(words)]
    if len(words) > 1 and len(words[0]) >= 5:
        slugs.append(words[0])
    return list(dict.fromkeys(s for s in slugs if len(s) >= 3))


def probe_ats(ats: str, bases: list[str], cfg: Config, checkpoint: Path, rate: float) -> None:
    """Try every slug on one provider, appending each board found to the checkpoint.

    One thread per provider, each with its own throttle, so no host sees more
    than one request per ``rate`` seconds. Slugs already in the checkpoint are
    skipped, so a stopped sweep resumes where it left off.
    """
    fetcher = Fetcher(rate_limit_seconds=rate, timeout_seconds=20, respect_robots_txt=False,
                      user_agent=cfg.section("discover").get("user_agent", "jobsearch-agent/0.1"))
    done = _checkpoint_keys(checkpoint, ats)
    for base in bases:
        for slug in slug_candidates(base):
            if slug in done:
                continue
            done.add(slug)
            record = {"ats": ats, "slug": slug, "base": base, "found": False}
            board = BoardRef(company=base, ats=ats, token=slug, ind_sponsor=True)
            try:
                postings = fetch_board(board, fetcher)
            except DiscoveryError as exc:
                if "HTTP 404" not in str(exc):
                    record["error"] = str(exc)[:200]
                postings = None
            except Exception as exc:  # a malformed board must not stop the sweep
                record["error"] = f"{type(exc).__name__}: {exc}"[:200]
                postings = None
            if postings is not None:
                titled = filter_postings(postings, cfg)
                workable, _ = filter_locations(titled, cfg)
                record.update(found=True, total=len(postings), title_matches=len(titled),
                              workable=len(workable),
                              examples=[f"{p.title} ({p.location})" for p in workable[:5]])
            with _WRITE_LOCK, checkpoint.open("a") as fh:
                fh.write(json.dumps(record) + "\n")


def _checkpoint_keys(checkpoint: Path, ats: str) -> set[str]:
    if not checkpoint.exists():
        return set()
    keys = set()
    for line in checkpoint.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("ats") == ats:
            keys.add(row["slug"])
    return keys


def write_report(checkpoint: Path, register: list[Company], configured: set[str], out: Path) -> int:
    by_base = {}
    for company in register:
        by_base.setdefault(company.base, company)
    rows = []
    for line in checkpoint.read_text().splitlines():
        row = json.loads(line)
        if not row.get("found"):
            continue
        company = by_base.get(row["base"])
        rows.append({
            "company": company.name if company else row["base"],
            "kvk": company.kvk if company else "",
            "ats": row["ats"],
            "token": row["slug"],
            "already_configured": row["slug"] in configured,
            "workable_matches": row.get("workable", 0),
            "title_matches": row.get("title_matches", 0),
            "total_postings": row.get("total", 0),
            "examples": " | ".join(row.get("examples", [])),
        })
    rows.sort(key=lambda r: (-r["workable_matches"], -r["title_matches"], -r["total_postings"]))
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["company"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Probe only the first N companies")
    parser.add_argument("--rate", type=float, default=0.3, help="Seconds between requests per provider")
    parser.add_argument("--refresh", action="store_true", help="Refetch the register")
    parser.add_argument("--report-only", action="store_true", help="Rebuild the CSV from the checkpoint")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = load_config(args.config)
    out = output_dir(cfg)
    register = load_register(out / "ind-register.html", refresh=args.refresh)
    configured = {b.token for b in cfg.boards}
    candidates = [c for c in register if worth_probing(c)]
    bases = list(dict.fromkeys(c.base for c in candidates))
    if args.limit:
        bases = bases[: args.limit]
    checkpoint = out / "sponsor-sweep.jsonl"
    log.info("register: %d sponsors, %d after name filter, %d distinct names to probe",
             len(register), len(candidates), len(bases))

    if not args.report_only:
        with ThreadPoolExecutor(max_workers=len(PROBE_ATS)) as pool:
            for future in [pool.submit(probe_ats, ats, bases, cfg, checkpoint, args.rate) for ats in PROBE_ATS]:
                future.result()

    report = out / f"sponsor-sweep-{date.today().isoformat()}.csv"
    count = write_report(checkpoint, register, configured, report)
    log.info("%d boards found; report at %s", count, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
