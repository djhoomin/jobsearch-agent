"""SQLite tracker: the source of truth for the search.

Tables
------
``job``            one row per posting, with the scoring breakdown denormalised
                   onto it so a status listing needs no joins
``status_history`` every status transition, with who/when/why
``contact``        inferred contacts per job
``outreach``       drafted messages per job (never sent from here)
``note``           free-text notes

Status transitions are validated against
:data:`jobsearch.models.TRANSITIONS`, so the database cannot record a job going
from Rejected straight back to Interviewing without an explicit re-open.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..models import (
    Contact,
    JobPosting,
    OutreachDraft,
    ScoreReport,
    Status,
    can_transition,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS job (
    job_id          TEXT PRIMARY KEY,
    company         TEXT NOT NULL,
    title           TEXT NOT NULL,
    url             TEXT,
    source          TEXT,
    location        TEXT,
    department      TEXT,
    description     TEXT,
    salary_text     TEXT,
    board_tier      INTEGER,
    ind_sponsor     TEXT,
    discovered_at   TEXT NOT NULL,
    status          TEXT NOT NULL,
    score_weighted  REAL,
    score_buyer     REAL,
    score_role_fit  REAL,
    score_company   REAL,
    score_domain    REAL,
    score_talent    REAL,
    score_json      TEXT,
    constraints_ok  INTEGER,
    recommendation  TEXT,
    cv_html_path    TEXT,
    cv_pdf_path     TEXT,
    ats_json        TEXT,
    warm_path       TEXT,
    next_action     TEXT,
    due             TEXT,
    notes           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS status_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES job(job_id) ON DELETE CASCADE,
    from_status TEXT,
    to_status   TEXT NOT NULL,
    reason      TEXT,
    changed_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS contact (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES job(job_id) ON DELETE CASCADE,
    name        TEXT,
    title       TEXT,
    rationale   TEXT,
    search_url  TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outreach (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL REFERENCES job(job_id) ON DELETE CASCADE,
    connection_note TEXT,
    linkedin_message TEXT,
    email_subject   TEXT,
    email_body      TEXT,
    sent            INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS note (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     TEXT NOT NULL REFERENCES job(job_id) ON DELETE CASCADE,
    body       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_status ON job(status);
CREATE INDEX IF NOT EXISTS idx_job_company ON job(company);
CREATE INDEX IF NOT EXISTS idx_history_job ON status_history(job_id);
"""


class TrackerError(RuntimeError):
    """An illegal or impossible tracker operation."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Tracker:
    """SQLite-backed application tracker."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @classmethod
    def from_config(cls, cfg) -> "Tracker":
        return cls(path=cfg.db_path)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- jobs --------------------------------------------------------------
    def upsert_job(
        self, posting: JobPosting, status: Status = Status.NOT_STARTED
    ) -> str:
        """Insert a posting, or refresh the mutable fields of an existing one.

        An existing job's status, scores and CV paths are never clobbered by a
        re-discovery: only the posting text and metadata are refreshed.
        """
        now = _now()
        existing = self.get_job(posting.job_id)
        with self._tx() as conn:
            if existing:
                conn.execute(
                    """
                    UPDATE job SET company=?, title=?, url=?, source=?, location=?,
                        department=?, description=?, salary_text=?, board_tier=?,
                        ind_sponsor=?, updated_at=?
                    WHERE job_id=?
                    """,
                    (
                        posting.company,
                        posting.title,
                        posting.url,
                        posting.source,
                        posting.location,
                        posting.department,
                        posting.description,
                        posting.salary_text,
                        posting.board_tier,
                        posting.ind_sponsor,
                        now,
                        posting.job_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO job (job_id, company, title, url, source, location,
                        department, description, salary_text, board_tier, ind_sponsor,
                        discovered_at, status, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        posting.job_id,
                        posting.company,
                        posting.title,
                        posting.url,
                        posting.source,
                        posting.location,
                        posting.department,
                        posting.description,
                        posting.salary_text,
                        posting.board_tier,
                        posting.ind_sponsor,
                        posting.discovered_at,
                        status.value,
                        now,
                        now,
                    ),
                )
                conn.execute(
                    "INSERT INTO status_history (job_id, from_status, to_status, reason, changed_at)"
                    " VALUES (?,?,?,?,?)",
                    (posting.job_id, None, status.value, "discovered", now),
                )
        return posting.job_id

    def get_job(self, job_id: str) -> sqlite3.Row | None:
        cur = self._conn.execute("SELECT * FROM job WHERE job_id = ?", (job_id,))
        return cur.fetchone()

    def resolve_job_id(self, needle: str) -> str:
        """Accept a full id, a unique prefix, or a URL."""
        row = self.get_job(needle)
        if row:
            return needle
        cur = self._conn.execute(
            "SELECT job_id FROM job WHERE job_id LIKE ? OR url = ?", (f"{needle}%", needle)
        )
        matches = [r["job_id"] for r in cur.fetchall()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise TrackerError(f"No tracked job matches {needle!r}. Run `jobsearch discover` first.")
        raise TrackerError(
            f"{needle!r} is ambiguous - matches {len(matches)}: {', '.join(matches[:5])}"
        )

    def get_posting(self, job_id: str) -> JobPosting:
        """Reconstruct a :class:`JobPosting` from the database."""
        row = self.get_job(job_id)
        if row is None:
            raise TrackerError(f"No tracked job with id {job_id!r}")
        return JobPosting(
            company=row["company"],
            title=row["title"],
            url=row["url"] or "",
            source=row["source"] or "manual",
            location=row["location"] or "",
            description=row["description"] or "",
            department=row["department"] or "",
            salary_text=row["salary_text"] or "",
            board_tier=row["board_tier"],
            ind_sponsor=row["ind_sponsor"] or "unknown",
            job_id=row["job_id"],
            discovered_at=row["discovered_at"],
        )

    def list_jobs(
        self,
        *,
        status: Status | None = None,
        company: str | None = None,
        min_score: float | None = None,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if company:
            clauses.append("LOWER(company) = ?")
            params.append(company.lower())
        if min_score is not None:
            clauses.append("score_weighted >= ?")
            params.append(min_score)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT * FROM job" + where
            + " ORDER BY score_weighted DESC NULLS LAST, board_tier ASC, company ASC"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        return list(self._conn.execute(sql, params).fetchall())

    def delete_job(self, job_id: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM job WHERE job_id = ?", (job_id,))

    # -- status ------------------------------------------------------------
    def set_status(self, job_id: str, target: Status, reason: str = "") -> Status:
        """Move a job's status, validating the transition and logging it."""
        row = self.get_job(job_id)
        if row is None:
            raise TrackerError(f"No tracked job with id {job_id!r}")
        current = Status.parse(row["status"])
        if not can_transition(current, target):
            allowed = ", ".join(sorted(s.value for s in _allowed_from(current)))
            raise TrackerError(
                f"Illegal status transition {current.value!r} -> {target.value!r}. "
                f"From {current.value!r} you can move to: {allowed}"
            )
        now = _now()
        with self._tx() as conn:
            conn.execute(
                "UPDATE job SET status = ?, updated_at = ? WHERE job_id = ?",
                (target.value, now, job_id),
            )
            if current is not target:
                conn.execute(
                    "INSERT INTO status_history (job_id, from_status, to_status, reason,"
                    " changed_at) VALUES (?,?,?,?,?)",
                    (job_id, current.value, target.value, reason, now),
                )
        return target

    def history(self, job_id: str) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM status_history WHERE job_id = ? ORDER BY id", (job_id,)
            ).fetchall()
        )

    # -- scores ------------------------------------------------------------
    def save_score(self, report: ScoreReport) -> None:
        scores = {d.name: d.score for d in report.dimensions}
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE job SET score_weighted=?, score_buyer=?, score_role_fit=?,
                    score_company=?, score_domain=?, score_talent=?, score_json=?,
                    constraints_ok=?, recommendation=?, notes=COALESCE(notes,'')||?,
                    updated_at=?
                WHERE job_id=?
                """,
                (
                    report.weighted,
                    scores.get("buyer"),
                    scores.get("role_fit"),
                    scores.get("company"),
                    scores.get("domain"),
                    scores.get("talent"),
                    json.dumps(report.to_dict()),
                    1 if report.constraints.passed else 0,
                    report.recommendation,
                    "",
                    _now(),
                    report.job_id,
                ),
            )

    # -- cv ----------------------------------------------------------------
    def save_cv(
        self, job_id: str, html_path: str, pdf_path: str | None, ats: dict[str, Any] | None = None
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE job SET cv_html_path=?, cv_pdf_path=?, ats_json=?, updated_at=?"
                " WHERE job_id=?",
                (html_path, pdf_path, json.dumps(ats) if ats else None, _now(), job_id),
            )

    # -- contacts and outreach --------------------------------------------
    def save_contacts(self, job_id: str, contacts: Sequence[Contact]) -> None:
        now = _now()
        with self._tx() as conn:
            conn.execute("DELETE FROM contact WHERE job_id = ?", (job_id,))
            conn.executemany(
                "INSERT INTO contact (job_id, name, title, rationale, search_url, created_at)"
                " VALUES (?,?,?,?,?,?)",
                [
                    (job_id, c.name, c.title, c.rationale, c.linkedin_search_url, now)
                    for c in contacts
                ],
            )

    def save_outreach(self, draft: OutreachDraft) -> None:
        self.save_contacts(draft.job_id, draft.contacts)
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO outreach (job_id, connection_note, linkedin_message,"
                " email_subject, email_body, sent, created_at) VALUES (?,?,?,?,?,0,?)",
                (
                    draft.job_id,
                    draft.linkedin_connection_note,
                    draft.linkedin_message,
                    draft.email_subject,
                    draft.email_body,
                    draft.created_at,
                ),
            )

    def latest_outreach(self, job_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM outreach WHERE job_id = ? ORDER BY id DESC LIMIT 1", (job_id,)
        ).fetchone()

    def job_ids_with_outreach(self) -> set[str]:
        """Every job that has a draft or a contact, in one query.

        The table needs this for a whole page of rows; asking per row would be
        one query per line.
        """
        rows = self._conn.execute(
            "SELECT job_id FROM outreach UNION SELECT job_id FROM contact"
        ).fetchall()
        return {str(r["job_id"]) for r in rows}

    def add_contact(self, job_id: str, contact: Contact) -> None:
        """Append one contact, keeping the existing ones.

        `save_contacts` replaces the whole set because a draft owns its
        contacts; a contact you found yourself has to be additive.
        """
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO contact (job_id, name, title, rationale, search_url, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (
                    job_id,
                    contact.name,
                    contact.title,
                    contact.rationale,
                    contact.linkedin_search_url,
                    _now(),
                ),
            )

    def contacts(self, job_id: str) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM contact WHERE job_id = ? ORDER BY id", (job_id,)
            ).fetchall()
        )

    # -- notes and planning ------------------------------------------------
    def add_note(self, job_id: str, body: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO note (job_id, body, created_at) VALUES (?,?,?)",
                (job_id, body, _now()),
            )

    def notes(self, job_id: str) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM note WHERE job_id = ? ORDER BY id", (job_id,)
            ).fetchall()
        )

    def set_fields(self, job_id: str, **fields: Any) -> None:
        """Update the free-form planning columns, or correct a posting field.

        `location` is included because a parser fix can leave stored locations
        wrong on rows a sweep will not touch (dismissed or in-flight ones), and
        a wrong location is what makes a workable role look unworkable.
        """
        allowed = {"warm_path", "next_action", "due", "notes", "location"}
        unknown = set(fields) - allowed
        if unknown:
            raise TrackerError(f"Cannot set unknown field(s): {', '.join(sorted(unknown))}")
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._tx() as conn:
            conn.execute(
                f"UPDATE job SET {assignments}, updated_at = ? WHERE job_id = ?",
                [*fields.values(), _now(), job_id],
            )

    # -- reporting ---------------------------------------------------------
    def counts_by_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM job GROUP BY status"
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def stats(self) -> dict[str, Any]:
        counts = self.counts_by_status()
        total = sum(counts.values())
        active = sum(
            n
            for s, n in counts.items()
            if s in {Status.OUTREACH_SENT.value, Status.APPLIED.value,
                     Status.IN_CONVERSATION.value, Status.INTERVIEWING.value}
        )
        row = self._conn.execute(
            "SELECT AVG(score_weighted) AS avg, MAX(score_weighted) AS max FROM job"
            " WHERE score_weighted IS NOT NULL"
        ).fetchone()
        return {
            "total": total,
            "by_status": counts,
            "active": active,
            "scored": self._conn.execute(
                "SELECT COUNT(*) AS n FROM job WHERE score_weighted IS NOT NULL"
            ).fetchone()["n"],
            "avg_score": round(row["avg"], 2) if row["avg"] is not None else None,
            "max_score": row["max"],
        }


def _allowed_from(status: Status) -> set[Status]:
    from ..models import TRANSITIONS

    return TRANSITIONS.get(status, set()) | {status}


__all__ = ["Tracker", "TrackerError"]
