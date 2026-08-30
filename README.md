# jobsearch-agent

An agent harness for one person's senior-AI-leadership job search. It finds
roles on public job-board APIs, scores them against a written strategy, tailors
a CV from a fact base without inventing anything, proves the resulting PDF
survives an ATS parser, drafts outreach, and tracks the pipeline in SQLite.

It is deliberately narrow. It is built around one candidate's documents, one
scoring rubric, and one set of hard constraints, all of which live in
`config.local.toml` and can be repointed at someone else's.

The most valuable thing in here is not the CV generation. It is
[`jobsearch verify`](#the-ats-verifier) — the check that a beautiful PDF is
still a machine-readable one.

---

## Install

Requires Python 3.11+ and, for PDF rendering, Google Chrome or Chromium.

```bash
cd jobsearch-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

jobsearch doctor          # checks documents, Chrome, credentials, weights
jobsearch doctor --boards # also pings every configured ATS board
```

For the optional Google sync: `pip install -e '.[google]'`.
For the tests: `pip install -e '.[dev]' && pytest`.

### Configure

```bash
jobsearch setup
```

A guided first run: it finds your source documents in the parent directory,
asks for your details and hard constraints, checks for Chrome and Claude
credentials, and writes `config.local.toml`. Press Enter to accept any
`[default]`. It fills in the committed template, so every explanatory
comment survives into your own config.

Prefer to do it by hand:

```bash
cp config.example.toml config.local.toml
```

Then edit `config.local.toml`: point `[paths]` at your own documents, set
`[candidate]`, and set your own hard constraints. `config.local.toml` is
gitignored — it holds personal details (compensation floor, notice-period
and non-compete status) that should never reach a remote. The committed
`config.example.toml` is a placeholder template and is never loaded
automatically.

### Claude API credentials

The client is constructed zero-arg, so it picks up whatever the environment
already has:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
ant auth login
```

No key is ever read from, written to, or prompted for by this code. Everything
except the four Claude-backed stages (`score`, `tailor`, `outreach`, `run`)
works with no credentials at all — including the entire ATS verifier.

---

## The TUI

```bash
pip install -e '.[tui]'
jobsearch tui
```

A pipeline browser: every role in one table, the selected one expanded below,
and a key per stage. It calls exactly the same functions as the subcommands, so
behaviour cannot drift between the two.

```
┌ jobsearch ─────────────────────────────────────────────┐
│ Pipeline (14)         [/] filter                       │
├────────────────────────────────────────────────────────┤
│ ▸ Northwind   Dir. Eng, AI ✓ Amsterdam ✓ ✓ 4.15 ● Applied│
│   Contoso AI  Head of AI    ✓ Remote (EU)  3.90 ○ New   │
│   Initech     Dir. Research ? Spain        3.55 ◐ Scored│
│   Example Co  VP AI         ✗ Remote - US  elim ✗ Elim  │
├─ Northwind — Director of Engineering, AI ──────────────┤
│ score 4.15   buyer 4 │ role_fit 5 │ company 4 │ ...    │
│ cv    output/cv/CV_Northwind.pdf                       │
└────────────────────────────────────────────────────────┘
```

| Key | Does |
|---|---|
| `s` | Score the selected role |
| `t` | Tailor a CV for it |
| `o` | Draft outreach |
| `v` | Run the ATS verifier on its CV |
| `enter` | Open the role page: constraints, per-dimension reasoning, notes, drafts |
| `w` | Open the posting in your browser |
| `c` | Attach a CV you already made (on the role page) |
| `n` | Add a note (on the role page) |
| `p` | Add a contact you found yourself (on the role page) |
| `a` | Set status (Applied, Rejected, …) — only legal transitions are offered |
| `n` | Add a role you found yourself; `ctrl+s` saves |
| `d` | Dismiss an irrelevant role — hidden, and it stays dismissed |
| `x` | Delete permanently (asks first) |
| `h` | Show dismissed roles again |
| `f` | Scan the boards — pick all tiers or one |
| `,` | Settings: title filters, constraints, rubric weights |
| `/` | Filter by company or title (`esc` clears) |
| `r` | Reload from the tracker |
| `q` | Quit |

The location column carries a fitness glyph from the same `check_location` the
scorer uses, so the table cannot disagree with what elimination will decide:
`✓` workable, `✗` outside Amsterdam / NL-hybrid / remote-EU, `?` unclassifiable.
A bare "remote" is not a pass — `Remote - United States` is a US role.

A scan (`f`) skips roles you have dismissed or are already working on, rather
than upserting them. `upsert_job` preserves status and scores anyway, but
skipping means a sweep provably cannot rewrite the posting text of a role you
have applied to. The summary counts them, so you can see the filter working:
`0 new · 8 refreshed · 52 left dismissed · 3 in progress, untouched`.

Settings (`,`) edits `config.local.toml` in place, replacing values rather than
re-serialising, so every comment in the file survives. Weights are validated to
sum to 1.0 before anything is written, and the config reloads without
restarting. `ctrl+e` opens `search-strategy.md` in `$EDITOR` — the scorer reads
that prose directly, so it is where judgement belongs, not a form.

**Dismiss (`d`) rather than delete (`x`).** Dismissing sets a status, and
`upsert_job` never clobbers the status of an existing job — so a dismissed role
stays dismissed the next time `discover` sweeps its board. A deleted one comes
straight back. Delete is there for genuine junk, and it asks first.

The role page binds the stage keys too (`s` `t` `o` `v`), so you can score,
tailor and draft without closing it — the page reprints itself when the stage
finishes. A modal's bindings shadow the app's, so a page that advertises a key
has to bind it itself.

Stages run on a worker thread, so a long tailoring call does not freeze the
table, and results stream into the log pane at the bottom. `--dry-run` works
here too: the app says so on start and makes no API calls.

## The pipeline

Each stage is independently invocable and chainable. `discover` writes to the
tracker; the others take a job id back out of it.

```bash
jobsearch discover --tier 1              # public ATS boards for tier-1 targets
jobsearch status                         # what is tracked
jobsearch score northwind-director        # hard constraints, then the rubric
jobsearch tailor northwind-director       # CV -> PDF -> grounding audit -> ATS check
jobsearch verify output/cv/CV_Northwind.pdf
jobsearch outreach northwind-director     # contacts + drafts, nothing sent
jobsearch track northwind-director --status Applied --reason "applied via careers page"
jobsearch export                         # xlsx in your tracker's column shape
jobsearch run https://jobs.ashbyhq.com/northwind/a1b2c3d4-...  # all of it, agentically
```

Job ids accept a unique prefix (`northwind-director`) or the posting URL.

`--dry-run` works on every command: it prints what would happen, makes no API
call, and writes nothing.

### 1. `discover`

Queries the **public job-board APIs** of the companies in `config.local.toml`:

| ATS | Endpoint |
|---|---|
| Greenhouse | `boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true` |
| Lever | `api.lever.co/v0/postings/{company}?mode=json` |
| Ashby | `api.ashbyhq.com/posting-api/job-board/{name}` |
| Anything else | a JSON feed the company publishes itself (`ats = "json_feed"`) |

Results are pre-filtered by title (`[discover].title_include` /
`title_exclude`) so only plausible roles reach the paid scoring stage. A board
that 404s is reported and skipped; one bad token does not abort the sweep.

```bash
jobsearch discover                        # everything configured
jobsearch discover --tier 1 --tier 2
jobsearch discover --company Northwind
jobsearch discover --web                  # also run a Claude web_search pass
jobsearch discover --web "sovereign AI leadership roles Netherlands"
```

`config.example.toml` ships three boards as a format illustration; you add
your own. Verify every token against its live API with `jobsearch doctor
--boards` — firms migrate between ATS vendors, and a stale token 404s
silently rather than erroring.

For a role on a site this tool will not scrape, enter it by hand:

```bash
jobsearch add --company Northwind --title "Director of Engineering, AI" \
  --location Munich --url https://... --file jd.txt
```

**Politeness.** A real User-Agent with a contact address, a configurable
inter-request delay, and a robots.txt check on every page fetch. The three ATS
API hosts are exempt from the robots check, because their `robots.txt` is
written for page crawlers while these endpoints are the vendors' own documented
public APIs, published so that listings can be consumed programmatically. That
exemption is a three-host allowlist, not a global override, and it is tested.

### 2. `score`

Encodes `search-strategy.md`.

**Hard constraints run first, and a role that fails one is never sent to the
model at all.** Each returns pass / fail / **unknown**, because "this posting
does not state a salary" is not the same as "this salary is too low", and
treating it as one throws away good roles.

| Constraint | Fails when |
|---|---|
| Visa | The posting explicitly rules out sponsorship. A company with no NL entity is `unknown`, not a fail: remote-EU or an EOR can still work, it just carries no kennismigrant status. |
| Non-compete | The employer is gaming-adjacent and `non_compete_waiver_signed = false`. Flip that flag the day it is signed and the gate disappears. |
| Compensation | A stated range whose **top** is below your configured floor. A range straddling the floor is negotiating room, not a fail. |
| Location | Outside Amsterdam / NL-hybrid / remote-EU. A bare "Remote" does **not** rescue a US role: only an explicit European anchor overrides a blocked location, which is why "San Francisco or Remote (Europe)" passes and "Remote - United States" does not. |
| Travel | Weekly travel, or a stated percentage above the cap. |

Survivors are scored 1-5 on the five weighted dimensions — Buyer 20%, Role fit
25%, Company 25%, Domain 15%, Talent density 15% — by Claude, with the rubric
text and the career dossier in the prompt. The arithmetic is done locally and is
unit-tested; the model supplies the judgement and, for each dimension, the
**reasoning and the evidence it used**. The CLI prints all of it, not just a
number.

### 3. `tailor`

The most careful stage, in three steps:

1. **Generate.** A streaming call produces a complete tailored HTML CV, using
   the base CV as the structural template and the career dossier as the fact
   base. It tailors the headline to the posting's exact job title, the summary,
   the bullet emphasis and ordering, and the skills-block vocabulary.
2. **Harden.** Deterministic post-processing re-applies the ATS CSS invariants
   the template depends on and warns loudly if the generated stylesheet
   reintroduces a known hazard.
3. **Verify grounding.** A second, structured call takes the generated CV back
   and audits every claim against the dossier and the base CV, returning the
   evidence for each. **Anything it cannot ground is printed in full and the
   command exits non-zero.** The tool would rather stop you than let a
   fabricated number reach an interview.

Then the PDF is rendered with headless Chrome and run through the ATS verifier.

```bash
jobsearch tailor northwind-director
jobsearch tailor northwind-director --stream     # watch it generate
jobsearch tailor northwind-director --no-render  # HTML only
```

Output lands in `output/cv/`. Your source documents are opened read-only and
never written to.

### 4. `verify` — the ATS verifier

Runnable standalone against any PDF. It reads the **text layer** with `pypdf` —
the same thing an ATS parser sees — and asserts seven properties. Each one was
learned from a real failure on this CV:

| Check | Why it exists |
|---|---|
| **Page count** ≤ 2 | Configurable. |
| **Section headings present** | A parser that cannot find "Education" does not mis-render it; it reports you as having no degree. |
| **No heading corrupted by letter-spacing** | The base CV originally had `letter-spacing` + `font-variant: small-caps` on `h2`. Chrome emits those as individually positioned glyphs, so extraction produced `Patents , Publications` and `E D U C A T I O N`, and Jobscan could not find the Education section. This check is that regression's test. |
| **Education entries on one line, with their dates** | The dates used to sit in a right-aligned flex column, which extraction reorders — detaching every date from its degree. |
| **Bullet markers inline** | An absolutely positioned `li::before` orphans every marker onto its own line, leaving the parser a column of dots and a column of unattributed sentences. |
| **No hyphenated keyword split across a wrap** | `human-in-` / `the-loop` is not a literal match for `human-in-the-loop`. Terms in `[ats].nowrap_keywords` are a **failure**; any other hyphen wrap is a **warning** naming the reconstructed word, so you can decide whether to wrap it in `<span class="nb">`. |
| **Keyword coverage** | JD terms present vs. missing, reported honestly. Missing terms are printed. A single reassuring percentage would be worse than useless. |

```bash
jobsearch verify ../cv.pdf                                  # your base CV: PASS
jobsearch verify output/cv/CV_Northwind.pdf --job-id northwind-director
jobsearch verify some.pdf --jd-file jd.txt --json
```

Exit code is 0 on pass, 1 on fail, so it drops straight into a pre-send check.

The test suite renders two fixture CVs with real headless Chrome: an
ATS-hardened one that must pass everything, and one carrying all five defects
above that must fail every corresponding check.

### 5. `outreach`

Infers the likely contact **roles** for this company and posting, emits a
LinkedIn **search URL you click yourself**, and drafts a connection note
(≤200 chars), a message (≤300 chars) and an email — grounded in the dossier and
the specific posting, and constrained by the same red-list guardrails.

```bash
jobsearch outreach northwind-director
jobsearch outreach northwind-director --gmail-draft --to person@example.com
```

`--gmail-draft` creates a Gmail **draft**. There is no send path in this
codebase.

### 6. `track` / `status` / `show` / `export`

SQLite (`output/jobsearch.db`) is the source of truth: job, company, source URL,
discovery date, weighted score plus the per-dimension breakdown, status, CV
paths, ATS report, contacts, outreach drafts, notes, timestamps.

Status transitions are **validated and logged**. You cannot record a job going
from Rejected back to Interviewing; the error tells you which moves are legal
from where you are.

```
Not started ─┬─> Outreach sent ─┬─> Applied ─> In conversation ─> Interviewing ─> Offer
             └─> Applied        └─> In conversation
any active state ─> Rejected / Withdrawn / Parked      Rejected ─> Parked (re-open)
```

```bash
jobsearch status --min-score 4.0 --status Applied
jobsearch show northwind-director
jobsearch track northwind-director --status Interviewing --reason "HM call booked" --history
jobsearch export
```

`export` writes an xlsx in the exact column shape of your existing
`job-search-tracker.xlsx` — Pipeline, Outreach, Interviews, Dashboard, plus one
extra `Jobs (jobsearch-agent)` sheet — into `output/`. It reads your file only
to learn its header row, so a column you added is followed rather than
overwritten, and **it refuses outright to write over your own tracker**.

### 7. `run` — agentic end to end

Hands the stages to Claude as tools via the SDK's Tool Runner and lets it work
one role from URL to tracked application, deciding the order and the stopping
point. It is instructed to stop when a hard constraint fails, to report every
ungrounded claim verbatim, and to say plainly when a role is not worth pursuing.

```bash
jobsearch run https://jobs.ashbyhq.com/northwind/a1b2c3d4-...
jobsearch run northwind-director --instruction "Aim outreach at the founder, not the posting"
```

---

## What this deliberately does NOT do

These are decisions, not gaps.

**It does not scrape LinkedIn, Indeed, Glassdoor, or any site whose terms
forbid automated access.** Those hosts are on a refusal list: passing one a URL
produces an error explaining why and pointing at `jobsearch add`, not a
best-effort fetch with a spoofed User-Agent. Discovery uses vendors' documented
public APIs. Outreach hands you a LinkedIn *search URL to click* — a human
opening a search page is ordinary use; a bot harvesting profiles is not. This
costs some coverage. Getting a LinkedIn account restricted mid-search would cost
considerably more.

**It never sends anything.** Not an application, not a connection request, not
an email. The Gmail scope requested is `gmail.compose`, which can create drafts
and *cannot send*; there is no `messages().send()` call anywhere in the
repository. Automated outreach at this altitude reads as exactly what it is.

**It does not auto-apply.** No form filling, no one-click apply. Every
application is a deliberate act by the user.

**It does not invent experience.** The tailoring stage may reword, reorder,
compress and re-emphasise; it may promote a fact the base CV omitted; it may
adopt the posting's vocabulary for something genuinely done. It may not add a
technology, a metric, a headcount or a responsibility that is not in the
sources. The grounding audit exists to catch it trying, and surfaces every
unverifiable claim instead of shipping it.

**It does not auto-update your status from your inbox.** `sync --replies` lists
recent messages; you decide what they mean. A recruiter's "we'll be in touch" is
not a state transition.

**It does not touch your source documents.** The dossier, base CV, strategy,
target list and tracker are opened read-only. Everything written goes to
`output/`.

**It does not pretend to know things it does not.** Unverified hard constraints
come back as `unknown` with the question you should ask, not as a confident
pass. Keyword coverage prints what is missing. Board tokens that could not be
found are commented stubs in the config, not silent omissions.

---

## Configuration

Everything lives in `config.local.toml`. Relative paths resolve against that file's
own directory, so nothing is bound to one machine.

```toml
[paths]                      # your source documents
[claude]                     # model, token budgets, cache TTL
[constraints]                # the five hard filters
[weights]                    # the rubric; must sum to 1.0 or startup fails
[ats]                        # page limit, required headings, nowrap keywords
[render]                     # Chrome binary and flags
[discover]                   # User-Agent, rate limit, title filters
[[discover.boards]]          # company -> ATS board mapping (extend freely)
[outreach]                   # contact titles, character caps
[google]                     # optional sync, off by default
```

Point it elsewhere with `jobsearch --config /path/to/config.local.toml` or
`JOBSEARCH_CONFIG`.

Adding a target company:

```toml
[[discover.boards]]
company = "Noxtua"
ats = "greenhouse"       # greenhouse | lever | ashby | json_feed
token = "noxtua"         # the slug in the API URL
tier = 2
ind_sponsor = false      # true | false | "unknown" - feeds the visa filter
gaming = false           # true gates it behind the non-compete waiver
```

Then `jobsearch doctor --boards` to confirm the token resolves.

---

## Prompt caching

The dossier, base CV and strategy are ~40KB of context re-sent on every scoring,
tailoring and grounding call. They are assembled in a **fixed order** into the
system blocks, ahead of all per-job content, with `cache_control` on the last
stable block. Per-job text goes into `messages`, after the breakpoint.

`ClaudeClient.last_usage` exposes `cache_read_input_tokens`, and the CLI prints
it after each call:

```
  tokens: in=812 out=1840 cache_read=41203 cache_write=0  [cache HIT]
```

If that reads `cache miss` on repeated calls, something has been inserted into
the prefix — the ordering is asserted in the tests for exactly this reason.

---

## Google setup (optional)

**Skip this entirely unless you want Gmail drafts, Drive upload, or a Sheets
mirror.** Everything else works with no Google credentials.

1. Go to <https://console.cloud.google.com/> and **create a project**
   (e.g. `jobsearch-agent`).
2. **APIs & Services → Library**, and enable:
   - Gmail API
   - Google Drive API
   - Google Sheets API (only for the Sheets mirror)
3. **APIs & Services → OAuth consent screen**:
   - User type: **External**
   - Fill in app name, your email as support and developer contact
   - **Scopes**: you can leave this empty; the app requests them at runtime
   - **Test users**: add your own Google address. Keep the app in **Testing**;
     it is a personal tool and never needs verification. Testing-mode refresh
     tokens expire every 7 days, so expect an occasional re-consent.
4. **APIs & Services → Credentials → Create credentials → OAuth client ID**:
   - Application type: **Desktop app**
   - Download the JSON, save it as `client_secret.json` in the repo root
5. Enable it in `config.local.toml`:

   ```toml
   [google]
   enabled = true
   client_secret_path = "client_secret.json"
   token_path = ".secrets/google_token.json"
   drive_folder_name = "Job Applications 2026 - Tailored CVs"
   gmail_create_drafts = true
   gmail_read_replies = false          # opt in separately; adds a read scope
   sheets_spreadsheet_id = ""          # the id from the spreadsheet's URL
   ```

6. First run opens a browser for consent:

   ```bash
   jobsearch sync --drive
   ```

   The token is written to `.secrets/google_token.json` with mode 600.

**Scopes requested, and why they are the narrow ones:**

| Scope | Grants | Deliberately excludes |
|---|---|---|
| `gmail.compose` | create drafts | **sending** |
| `drive.file` | only files this app created | the rest of your Drive |
| `spreadsheets` | the mirror sheet | — |
| `gmail.readonly` | reading replies; **only if you set `gmail_read_replies = true`** | — |

`client_secret*.json`, `token.json`, `credentials.json` and `.secrets/` are all
in `.gitignore`. Nothing secret is committed.

---

## Development

```bash
pip install -e '.[dev]'
pytest                    # ~200 tests, no live API calls
pytest -k ats             # just the verifier
```

**No test makes a live Claude API call.** `FakeClaude` in `tests/conftest.py`
returns canned structured payloads and records every call, so the tests can
assert not only on results but on prompt construction — including that the
cacheable prefix precedes the volatile per-job content, and that an eliminated
role costs no API call at all.

What is tested properly: the weighted scoring maths, all five hard-constraint
filters (including salary-range parsing and the remote-US trap), every ATS check
at unit level plus both fixture CVs rendered end to end through real Chrome, the
tracker's state machine, the xlsx export against the real tracker's columns, the
ATS-payload adapters, the no-scrape refusals, the robots exemption boundary, and
`--dry-run` making no call and no write.

Tests requiring Chrome skip cleanly where it is unavailable.

```
src/jobsearch/
  cli.py          argument parsing and every command
  config.py       config.local.toml loading, path resolution
  models.py       JobPosting, ScoreReport, Status + transition table
  claude.py       SDK wrapper: caching, structured output, typed errors
  scoring.py      hard constraints + weighted rubric
  tailor.py       CV generation, HTML hardening, grounding audit
  ats.py          the ATS verifier
  render.py       headless Chrome
  outreach.py     contact inference and drafting
  agent.py        the Tool Runner agentic loop
  discover/       sources.py (ATS adapters), single.py (one URL), websearch.py
  tracker/        db.py (SQLite), export.py (xlsx)
  sync/           google.py (optional Gmail/Drive/Sheets)
```

## Licence

MIT. Personal tool; no support implied.
