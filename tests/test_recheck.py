"""Hand-added roles are checked at their own URL."""

from jobsearch.recheck import page_listed

LD = '<script type="application/ld+json">{"@type": "JobPosting", "title": "Head of AI"%s}</script>'


def test_a_page_with_a_job_posting_is_listed():
    assert page_listed(LD % "") is True


def test_a_stale_valid_through_does_not_mark_a_live_role_gone():
    assert page_listed(LD % ', "validThrough": "2026-08-31"') is True


def test_closed_wording_without_structured_data_is_gone():
    assert page_listed("<p>Deze vacature is gesloten.</p>") is False
    assert page_listed("<p>This position has been filled.</p>") is False


def test_a_plain_page_is_unknown():
    assert page_listed("<html><body>Careers</body></html>") is None


def test_stale_uses_the_row_check_for_single_role_companies():
    from jobsearch.stale import find_stale

    rows = [
        {"job_id": "a", "company": "Solo", "title": "a", "status": "Applied",
         "last_seen_at": "2026-09-20T10:00:00+00:00", "last_checked_at": "2026-09-29T10:00:00+00:00"},
        {"job_id": "b", "company": "Solo2", "title": "b", "status": "Applied",
         "last_seen_at": "2026-09-29T10:00:00+00:00", "last_checked_at": "2026-09-29T10:00:01+00:00"},
    ]
    assert [r.job_id for r in find_stale(rows, statuses=frozenset({"Applied"})).delisted] == ["a"]
