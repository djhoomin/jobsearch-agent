"""The System One (Jev) reader for the hard constraints.

The reader is a refinement of the deterministic checks, never a replacement:
a FAIL from the code stands, an UNKNOWN can be resolved, a PASS can only gain
an advisory, and any failure to reach Jev leaves the report untouched.
"""

from __future__ import annotations

import pytest

from jobsearch.models import ConstraintReport, ConstraintResult, JobPosting, Verdict
from jobsearch.scoring import DIMENSIONS, score_posting
from jobsearch.systemone import (
    NOULS,
    SystemOneSettings,
    apply_readings,
    posting_state,
    refine_constraints,
)


def report_with(**verdicts: Verdict) -> ConstraintReport:
    return ConstraintReport(
        results=[ConstraintResult(name, v, f"code: {name}") for name, v in verdicts.items()]
    )


SETTINGS = SystemOneSettings(enabled=True, fail_above=0.8, pass_below=0.2, flag_above=0.7)


class TestApplyReadings:
    def test_unknown_becomes_fail_above_threshold(self):
        report = report_with(visa=Verdict.UNKNOWN)
        touched = apply_readings(report, {"rules_out_sponsorship": 0.95}, SETTINGS)
        assert report.results[0].verdict is Verdict.FAIL
        assert not report.passed
        assert "Jev reads" in report.results[0].reason
        assert "code: visa" in report.results[0].reason, "the code's reason is kept"
        assert touched == report.results

    def test_unknown_becomes_pass_below_threshold(self):
        report = report_with(location=Verdict.UNKNOWN)
        apply_readings(report, {"outside_europe": 0.05}, SETTINGS)
        assert report.results[0].verdict is Verdict.PASS

    def test_visa_unknown_is_never_cleared_by_text(self):
        # UNKNOWN visa means "not on the IND register"; a posting cannot fix that.
        report = report_with(visa=Verdict.UNKNOWN)
        touched = apply_readings(report, {"rules_out_sponsorship": 0.02}, SETTINGS)
        assert report.results[0].verdict is Verdict.UNKNOWN
        assert touched == []

    def test_visa_unknown_can_still_fail(self):
        report = report_with(visa=Verdict.UNKNOWN)
        apply_readings(report, {"rules_out_sponsorship": 0.96}, SETTINGS)
        assert report.results[0].verdict is Verdict.FAIL

    def test_unknown_stays_unknown_in_between(self):
        report = report_with(travel=Verdict.UNKNOWN)
        touched = apply_readings(report, {"weekly_travel": 0.5}, SETTINGS)
        assert report.results[0].verdict is Verdict.UNKNOWN
        assert "inconclusive" in report.results[0].reason
        assert touched == []

    def test_fail_is_never_overturned(self):
        report = report_with(location=Verdict.FAIL)
        apply_readings(report, {"outside_europe": 0.0}, SETTINGS)
        assert report.results[0].verdict is Verdict.FAIL
        assert report.results[0].advisory == ""

    def test_pass_keeps_verdict_but_gains_advisory(self):
        # Adyen San Francisco: IND-listed employer, posting requires US authorisation.
        report = report_with(visa=Verdict.PASS)
        apply_readings(report, {"rules_out_sponsorship": 0.97}, SETTINGS)
        assert report.results[0].verdict is Verdict.PASS
        assert report.passed
        assert "p=0.97" in report.results[0].advisory

    def test_pass_below_flag_threshold_is_untouched(self):
        report = report_with(visa=Verdict.PASS)
        apply_readings(report, {"rules_out_sponsorship": 0.3}, SETTINGS)
        assert report.results[0].advisory == ""
        assert report.results[0].evidence == ""

    def test_missing_answers_and_constraints_are_skipped(self):
        report = report_with(visa=Verdict.UNKNOWN)
        apply_readings(report, {"weekly_travel": 0.99}, SETTINGS)
        assert report.results[0].verdict is Verdict.UNKNOWN

    def test_advisory_survives_serialisation(self):
        report = report_with(visa=Verdict.PASS)
        apply_readings(report, {"rules_out_sponsorship": 0.9}, SETTINGS)
        assert report.to_dict()["results"][0]["advisory"].startswith("posting text suggests")


class TestPostingState:
    def test_only_posting_fields_are_sent(self):
        posting = JobPosting(
            company="Northwind", title="Head of AI", url="https://example.invalid/1",
            location="Amsterdam", description="x" * 50, salary_text="",
        )
        state = posting_state(posting, SystemOneSettings(max_description_chars=10))
        assert set(state) == {
            "company", "title", "location", "department", "stated_compensation", "description"
        }
        assert state["description"] == "x" * 10
        assert state["stated_compensation"] == "not stated"

    def test_every_noul_maps_to_a_real_constraint(self):
        assert {q.constraint for q in NOULS.values()} == {"visa", "travel", "location"}

    def test_visa_question_describes_boilerplate_as_a_no(self):
        assert "boilerplate" in NOULS["rules_out_sponsorship"].criteria["false"]


class TestRefineConstraints:
    def test_disabled_never_calls_the_reader(self, cfg):
        calls = []
        report = report_with(visa=Verdict.UNKNOWN)
        cfg.raw["systemone"] = {"enabled": False}
        assert refine_constraints(
            JobPosting(company="A", title="B", url="u"), cfg, report,
            reader=lambda p: calls.append(p) or {"rules_out_sponsorship": 0.99},
        ) is False
        assert calls == []
        assert report.results[0].verdict is Verdict.UNKNOWN

    def test_enabled_without_key_or_sdk_is_a_noop(self, cfg, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        cfg.raw["systemone"] = {"enabled": True}
        report = report_with(visa=Verdict.UNKNOWN)
        assert refine_constraints(JobPosting(company="A", title="B", url="u"), cfg, report) is False
        assert report.results[0].verdict is Verdict.UNKNOWN

    def test_reader_error_leaves_the_report_alone(self, cfg, capsys):
        cfg.raw["systemone"] = {"enabled": True}
        report = report_with(visa=Verdict.UNKNOWN)

        def boom(posting):
            raise RuntimeError("rate limited")

        assert refine_constraints(JobPosting(company="A", title="B", url="u"), cfg, report, reader=boom) is False
        assert report.results[0].verdict is Verdict.UNKNOWN
        assert "rate limited" in capsys.readouterr().err

    def test_thresholds_come_from_config(self, cfg):
        cfg.raw["systemone"] = {"enabled": True, "fail_above": 0.5}
        report = report_with(visa=Verdict.UNKNOWN)
        refine_constraints(
            JobPosting(company="A", title="B", url="u"), cfg, report,
            reader=lambda p: {"rules_out_sponsorship": 0.6},
        )
        assert report.results[0].verdict is Verdict.FAIL


class TestScorePostingIntegration:
    def _scored(self, fake_claude):
        fake_claude.structured_responses["score"] = {
            **{d: {"score": 3, "reasoning": "", "evidence": []} for d in DIMENSIONS},
            "recommendation": "park", "notes": "", "open_questions": [],
        }

    def test_jev_fail_on_unknown_eliminates_before_the_model(self, cfg, fake_claude):
        cfg.raw["systemone"] = {"enabled": True}
        self._scored(fake_claude)
        posting = JobPosting(
            company="Nowhere Inc", title="Head of AI", url="https://example.invalid/2",
            location="Amsterdam",  # visa is UNKNOWN: no sponsor flag, no phrase
            description="Applicants must hold a valid US work permit.",
        )
        report = score_posting(
            posting, cfg, fake_claude, systemone_reader=lambda p: {"rules_out_sponsorship": 0.95}
        )
        assert report.eliminated
        assert fake_claude.calls == [], "a Jev fail must save the model call too"

    def test_dry_run_skips_jev(self, cfg, fake_claude):
        cfg.raw["systemone"] = {"enabled": True}
        self._scored(fake_claude)
        fake_claude.dry_run = True
        calls = []
        posting = JobPosting(company="A", title="Head of AI", url="u", location="Amsterdam")
        score_posting(posting, cfg, fake_claude, systemone_reader=lambda p: calls.append(p) or {})
        assert calls == []

    def test_unknowns_resolved_by_jev_are_not_listed_as_unverified(self, cfg, fake_claude, posting):
        cfg.raw["systemone"] = {"enabled": True}
        self._scored(fake_claude)
        report = score_posting(
            posting, cfg, fake_claude,
            systemone_reader=lambda p: {"rules_out_sponsorship": 0.01, "weekly_travel": 0.02, "outside_europe": 0.03},
        )
        assert not report.eliminated
        assert "travel" not in report.notes and "location" not in report.notes
