"""Delisting detection must distinguish "gone" from "not checked".

The failure this guards against is a mass false positive: reporting every role
on a board nobody has swept lately as delisted, which would be worse than the
silence it replaces.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jobsearch.stale import find_stale

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def row(job_id, company, title="A role", status="Not started", last_seen=None):
    return {
        "job_id": job_id, "company": company, "title": title,
        "status": status, "last_seen_at": last_seen.isoformat() if last_seen else None,
    }


class TestDelistedVsUnchecked:
    def test_a_row_its_board_stopped_returning_is_delisted(self):
        swept = NOW - timedelta(hours=2)
        report = find_stale([
            row("a", "Acme", last_seen=swept),
            row("b", "Acme", last_seen=swept - timedelta(days=9)),
        ], now=NOW)
        assert [r.job_id for r in report.delisted] == ["b"]
        assert report.unknown == []

    def test_a_board_never_swept_yields_unknown_not_delisted(self):
        """The mass false positive. Old rows on an unswept board prove nothing."""
        report = find_stale([
            row("a", "Quiet Co"), row("b", "Quiet Co"), row("c", "Quiet Co"),
        ], now=NOW)
        assert report.delisted == []
        assert len(report.unknown) == 3

    def test_one_stale_board_does_not_poison_another(self):
        swept = NOW - timedelta(hours=1)
        report = find_stale([
            row("a", "Acme", last_seen=swept),
            row("b", "Acme", last_seen=swept - timedelta(days=5)),
            row("c", "Quiet Co"),
        ], now=NOW)
        assert [r.job_id for r in report.delisted] == ["b"]
        assert [r.job_id for r in report.unknown] == ["c"]

    def test_rows_seen_in_the_same_sweep_are_all_live(self):
        swept = NOW - timedelta(hours=3)
        report = find_stale([
            row("a", "Acme", last_seen=swept),
            row("b", "Acme", last_seen=swept - timedelta(minutes=20)),
        ], now=NOW)
        assert report.delisted == []

    def test_grace_hours_is_respected(self):
        swept = NOW
        rows = [row("a", "Acme", last_seen=swept),
                row("b", "Acme", last_seen=swept - timedelta(hours=40))]
        assert find_stale(rows, now=NOW, grace_hours=36).delisted
        assert not find_stale(rows, now=NOW, grace_hours=48).delisted


class TestStatusScope:
    def test_applied_and_later_are_never_reported(self):
        """A role vanishing after you applied means it was filled. That is
        information about the application, not clutter."""
        swept = NOW
        old = swept - timedelta(days=30)
        rows = [row("live", "Acme", last_seen=swept)]
        for i, status in enumerate(
            ["Applied", "In conversation", "Interviewing", "Offer", "Rejected"]
        ):
            rows.append(row(f"x{i}", "Acme", status=status, last_seen=old))
        assert find_stale(rows, now=NOW).delisted == []

    def test_parked_is_reported(self):
        swept = NOW
        report = find_stale([
            row("live", "Acme", last_seen=swept),
            row("p", "Acme", status="Parked", last_seen=swept - timedelta(days=9)),
        ], now=NOW)
        assert [r.job_id for r in report.delisted] == ["p"]


class TestRender:
    def test_it_says_so_when_nothing_has_been_swept(self):
        out = find_stale([row("a", "Acme")], now=NOW).render()
        assert "No company has been swept" in out
        assert "discover" in out

    def test_delisted_rows_are_listed_with_their_age(self):
        swept = NOW
        out = find_stale([
            row("a", "Acme", last_seen=swept),
            row("b", "Acme", title="Dead Role", last_seen=swept - timedelta(days=9)),
        ], now=NOW).render()
        assert "DELISTED (1)" in out
        assert "Dead Role" in out


class TestInFlightDelistings:
    """A role vanishing after you applied is news, so it is noted, never withdrawn."""

    def _row(self, job_id, company, status, seen):
        return {"job_id": job_id, "company": company, "title": job_id, "status": status, "last_seen_at": seen}

    def test_only_swept_in_flight_rows_with_a_sighting_count(self):
        from jobsearch.stale import in_flight_delisted

        rows = [
            self._row("fresh", "A", "Not started", "2026-09-29T10:00:00+00:00"),
            self._row("gone", "A", "Applied", "2026-09-20T10:00:00+00:00"),
            self._row("live", "A", "Applied", "2026-09-29T09:00:00+00:00"),
            self._row("old", "A", "Applied", None),
            self._row("parked", "A", "Parked", "2026-09-01T10:00:00+00:00"),
        ]
        assert [r.job_id for r in in_flight_delisted(rows)] == ["gone"]


class TestAutoWithdraw:
    """Withdrawing must stay narrow, leave a trace, and undo itself."""

    @staticmethod
    def _tracker(tmp_path, rows):
        from jobsearch.models import JobPosting, Status
        from jobsearch.tracker import Tracker

        t = Tracker(path=tmp_path / "t.db")
        for job_id, status, seen, next_action in rows:
            t.upsert_job(JobPosting(
                company="Acme", title=job_id, url=f"https://jobs.ashbyhq.com/acme/{job_id}",
                source="ashby", job_id=job_id,
            ))
            if status != "Not started":
                t.set_status(job_id, Status.PARKED if status == "Parked" else Status.APPLIED)
            t._conn.execute(
                "UPDATE job SET last_seen_at=?, next_action=? WHERE job_id=?",
                (seen.isoformat() if seen else None, next_action, job_id),
            )
        t._conn.commit()
        return t

    def test_withdraws_only_old_untouched_roles_and_logs_why(self, tmp_path):
        from jobsearch.stale import AUTO_WITHDRAW_REASON, withdraw_delisted

        now = datetime.now(timezone.utc)
        t = self._tracker(tmp_path, [
            ("live", "Not started", now, None),
            ("gone", "Not started", now - timedelta(days=20), None),
            ("parked", "Parked", now - timedelta(days=20), None),
            ("recent", "Not started", now - timedelta(days=3), None),
            ("planned", "Not started", now - timedelta(days=20), "Ask Dolf"),
            ("applied", "Applied", now - timedelta(days=20), None),
        ])
        closed = {r.job_id for r in withdraw_delisted(t, now=now)}
        assert closed == {"gone", "parked"}
        assert t.get_job("gone")["status"] == "Withdrawn"
        assert t.history("gone")[-1]["reason"].startswith(AUTO_WITHDRAW_REASON)
        for kept in ("live", "recent", "planned", "applied"):
            assert t.get_job(kept)["status"] != "Withdrawn"
        t.close()

    def test_dry_run_changes_nothing(self, tmp_path):
        from jobsearch.stale import withdraw_delisted

        now = datetime.now(timezone.utc)
        t = self._tracker(tmp_path, [
            ("live", "Not started", now, None),
            ("gone", "Not started", now - timedelta(days=20), None),
        ])
        assert [r.job_id for r in withdraw_delisted(t, now=now, dry_run=True)] == ["gone"]
        assert t.get_job("gone")["status"] == "Not started"
        t.close()

    def test_a_role_listed_again_is_reopened_but_a_manual_withdrawal_is_not(self, tmp_path):
        from jobsearch.models import Status
        from jobsearch.stale import restore_relisted, withdraw_delisted

        now = datetime.now(timezone.utc)
        t = self._tracker(tmp_path, [
            ("live", "Not started", now, None),
            ("gone", "Parked", now - timedelta(days=20), None),
            ("mine", "Not started", now - timedelta(days=20), None),
        ])
        t.set_status("mine", Status.WITHDRAWN, reason="not for me")
        withdraw_delisted(t, now=now)
        t._conn.execute(
            "UPDATE job SET last_seen_at=? WHERE job_id IN ('gone','mine')",
            ((now + timedelta(hours=1)).isoformat(),),
        )
        t._conn.commit()
        assert [r.job_id for r in restore_relisted(t)] == ["gone"]
        assert t.get_job("gone")["status"] == "Parked"
        assert t.get_job("mine")["status"] == "Withdrawn"
        t.close()
