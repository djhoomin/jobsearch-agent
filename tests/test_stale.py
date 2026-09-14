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
