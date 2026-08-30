"""Hard-constraint filters and weighted scoring maths.

These are the deterministic half of the scoring stage and the half that must
never drift: a bug here either eliminates a viable role silently or lets an
unviable one through to a paid model call.
"""

from __future__ import annotations

import pytest

from jobsearch.models import JobPosting, Verdict
from jobsearch.scoring import (
    DIMENSIONS,
    build_dimensions,
    check_comp,
    check_constraints,
    check_location,
    check_noncompete,
    check_travel,
    check_visa,
    parse_salaries,
    score_posting,
    weighted_score,
)


def make(**kwargs) -> JobPosting:
    base = dict(company="Test Co", title="Head of AI", url="https://example.invalid/1")
    base.update(kwargs)
    return JobPosting(**base)


# ---------------------------------------------------------------------------
# Weighted maths
# ---------------------------------------------------------------------------


class TestWeightedScore:
    WEIGHTS = {"buyer": 0.20, "role_fit": 0.25, "company": 0.25, "domain": 0.15, "talent": 0.15}

    def test_all_fives_is_five(self):
        assert weighted_score({d: 5 for d in DIMENSIONS}, self.WEIGHTS) == 5.0

    def test_all_ones_is_one(self):
        assert weighted_score({d: 1 for d in DIMENSIONS}, self.WEIGHTS) == 1.0

    def test_known_mix(self):
        # DataSnipper's row in the real tracker: 5, 5, 3.5, 5, 3.5 -> 4.4
        scores = {"buyer": 5, "role_fit": 5, "company": 3.5, "domain": 5, "talent": 3.5}
        assert weighted_score(scores, self.WEIGHTS) == 4.4

    def test_weights_are_applied_not_averaged(self):
        # A 5 on the two 25% dimensions must beat a 5 on the two 15% ones.
        heavy = {"buyer": 1, "role_fit": 5, "company": 5, "domain": 1, "talent": 1}
        light = {"buyer": 1, "role_fit": 1, "company": 1, "domain": 5, "talent": 5}
        assert weighted_score(heavy, self.WEIGHTS) > weighted_score(light, self.WEIGHTS)

    def test_missing_dimension_raises(self):
        with pytest.raises(ValueError, match="Missing dimension"):
            weighted_score({"buyer": 5}, self.WEIGHTS)

    def test_out_of_range_score_raises(self):
        with pytest.raises(ValueError, match="outside 1-5"):
            weighted_score({**{d: 3 for d in DIMENSIONS}, "buyer": 9}, self.WEIGHTS)

    def test_build_dimensions_carries_weight_and_reasoning(self):
        payload = {d: {"score": 4, "reasoning": f"because {d}", "evidence": ["x"]} for d in DIMENSIONS}
        dims = build_dimensions(payload, self.WEIGHTS)
        assert [d.name for d in dims] == list(DIMENSIONS)
        assert dims[0].reasoning == "because buyer"
        assert dims[0].contribution == pytest.approx(4 * 0.20)


# ---------------------------------------------------------------------------
# Salary parsing
# ---------------------------------------------------------------------------


class TestSalaryParsing:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("EUR 150,000 - 180,000", [150000]),
            ("€120.000 tot €140.000", [120000, 140000]),
            ("$180,000 - $220,000 base", [180000, 220000]),
            ("110k - 140k EUR", [110000]),
            ("no salary here", []),
            ("team of 25 people, 500 customers", []),
        ],
    )
    def test_parses(self, text, expected):
        found = parse_salaries(text)
        for value in expected:
            assert value in found

    def test_ignores_implausible_amounts(self):
        assert parse_salaries("$15 per hour, $5,000,000 raised") == []


# ---------------------------------------------------------------------------
# Individual constraints
# ---------------------------------------------------------------------------


class TestVisaConstraint:
    def test_explicit_refusal_fails(self, cfg):
        posting = make(description="Note: we do not sponsor visas for this role.")
        assert check_visa(posting, cfg).verdict is Verdict.FAIL

    def test_known_sponsor_passes(self, cfg):
        assert check_visa(make(ind_sponsor="yes"), cfg).verdict is Verdict.PASS

    def test_offer_of_sponsorship_passes(self, cfg):
        posting = make(description="We provide visa sponsorship and relocation support.")
        assert check_visa(posting, cfg).verdict is Verdict.PASS

    def test_non_sponsor_is_unknown_not_fail(self, cfg):
        # A company with no NL entity may still work as remote-EU or via an EOR.
        result = check_visa(make(ind_sponsor="no"), cfg)
        assert result.verdict is Verdict.UNKNOWN
        assert "EOR" in result.reason

    def test_silence_is_unknown(self, cfg):
        assert check_visa(make(description="Great team."), cfg).verdict is Verdict.UNKNOWN


class TestNonCompeteConstraint:
    def test_gaming_employer_is_gated(self, cfg):
        posting = make(company="Studio X", description="We are a mobile games publisher.")
        result = check_noncompete(posting, cfg)
        assert result.verdict is Verdict.FAIL
        assert "non-compete waiver" in result.reason

    def test_board_flag_gates_even_without_keywords(self, cfg):
        assert check_noncompete(make(gaming=True), cfg).verdict is Verdict.FAIL

    def test_waiver_signed_lifts_the_gate(self, cfg):
        cfg.raw["constraints"]["non_compete_waiver_signed"] = True
        posting = make(description="We are a video game studio.")
        assert check_noncompete(posting, cfg).verdict is Verdict.PASS

    def test_non_gaming_passes(self, cfg):
        assert check_noncompete(make(description="Fintech compliance AI"), cfg).verdict is Verdict.PASS


class TestCompConstraint:
    def test_range_below_floor_fails(self, cfg):
        # Set the floor explicitly rather than inheriting whatever the shipped
        # template happens to use, so the assertion cannot drift with it.
        cfg.raw["constraints"]["comp_floor_eur"] = 200000
        posting = make(salary_text="EUR 90,000 - 110,000")
        result = check_comp(posting, cfg)
        assert result.verdict is Verdict.FAIL
        assert "110,000" in result.reason

    def test_range_above_floor_passes(self, cfg):
        assert check_comp(make(salary_text="EUR 150,000 - 180,000"), cfg).verdict is Verdict.PASS

    def test_top_of_range_is_what_counts(self, cfg):
        # A range straddling the floor is not a fail: the top is negotiable room.
        assert check_comp(make(salary_text="EUR 110,000 - 140,000"), cfg).verdict is Verdict.PASS

    def test_unstated_salary_is_unknown(self, cfg):
        assert check_comp(make(description="Competitive salary"), cfg).verdict is Verdict.UNKNOWN

    def test_unstated_can_be_made_fatal(self, cfg):
        cfg.raw["constraints"]["comp_fail_on_unknown"] = True
        assert check_comp(make(), cfg).verdict is Verdict.FAIL


class TestLocationConstraint:
    @pytest.mark.parametrize(
        "location",
        ["Amsterdam, Netherlands", "Remote - Europe", "Utrecht (hybrid)", "EMEA"],
    )
    def test_workable_locations_pass(self, cfg, location):
        assert check_location(make(location=location), cfg).verdict is Verdict.PASS

    @pytest.mark.parametrize("location", ["San Francisco, CA", "New York, United States"])
    def test_us_onsite_fails(self, cfg, location):
        assert check_location(make(location=location), cfg).verdict is Verdict.FAIL

    def test_us_company_with_remote_eu_option_passes(self, cfg):
        posting = make(location="San Francisco, USA or Remote (Europe)")
        assert check_location(posting, cfg).verdict is Verdict.PASS

    def test_bare_remote_does_not_rescue_a_us_role(self, cfg):
        """"Remote - United States" is a US role, not a remote-EU one."""
        posting = make(
            location="New York, United States; Remote - United States; San Francisco"
        )
        result = check_location(posting, cfg)
        assert result.verdict is Verdict.FAIL
        assert result.evidence == "united states"

    def test_short_patterns_match_on_word_boundaries(self, cfg):
        """A bare "nl" pattern must not match inside "Finland"."""
        result = check_location(make(location="Helsinki, Finland"), cfg)
        assert result.verdict is Verdict.UNKNOWN

    def test_missing_location_is_unknown(self, cfg):
        assert check_location(make(location=""), cfg).verdict is Verdict.UNKNOWN


class TestTravelConstraint:
    def test_weekly_travel_fails(self, cfg):
        posting = make(description="This role involves weekly travel to customer sites.")
        assert check_travel(posting, cfg).verdict is Verdict.FAIL

    def test_high_percentage_travel_fails(self, cfg):
        posting = make(description="Expect 50% travel across the region.")
        assert check_travel(posting, cfg).verdict is Verdict.FAIL

    def test_unquantified_travel_is_unknown(self, cfg):
        posting = make(description="Some travel to our Berlin office.")
        assert check_travel(posting, cfg).verdict is Verdict.UNKNOWN

    def test_no_travel_mentioned_passes(self, cfg):
        assert check_travel(make(description="Fully remote team."), cfg).verdict is Verdict.PASS


# ---------------------------------------------------------------------------
# Composite behaviour
# ---------------------------------------------------------------------------


class TestConstraintReport:
    def test_all_checks_run_in_stable_order(self, cfg):
        report = check_constraints(make(), cfg)
        assert [r.name for r in report.results] == [
            "visa",
            "non_compete",
            "compensation",
            "location",
            "travel",
        ]

    def test_one_failure_eliminates(self, cfg, posting):
        posting.description += " This role requires weekly travel."
        report = check_constraints(posting, cfg)
        assert not report.passed
        assert [r.name for r in report.failures] == ["travel"]

    def test_unknowns_do_not_eliminate(self, cfg):
        report = check_constraints(make(location="", description="Competitive salary"), cfg)
        assert report.passed
        assert len(report.unknowns) >= 2

    def test_viable_posting_passes_everything(self, cfg, posting):
        assert check_constraints(posting, cfg).passed


class TestScorePosting:
    """The stage function: constraints gate the model call."""

    def test_eliminated_posting_never_calls_the_model(self, cfg, fake_claude):
        posting = make(description="We do not sponsor visas.")
        report = score_posting(posting, cfg, fake_claude)
        assert report.eliminated
        assert report.weighted is None
        assert fake_claude.calls == [], "an eliminated role must not cost an API call"

    def test_surviving_posting_is_scored(self, cfg, fake_claude, posting):
        fake_claude.structured_responses["score"] = {
            **{
                d: {"score": 4, "reasoning": f"r-{d}", "evidence": ["e"]}
                for d in DIMENSIONS
            },
            "recommendation": "apply",
            "notes": "verify runway",
            "open_questions": ["What is the burn?"],
        }
        report = score_posting(posting, cfg, fake_claude)
        assert report.weighted == 4.0
        assert report.recommendation == "apply"
        assert "What is the burn?" in report.notes
        assert len(report.dimensions) == 5

    def test_prompt_puts_stable_context_before_job_content(self, cfg, fake_claude, posting):
        fake_claude.structured_responses["score"] = {
            **{d: {"score": 3, "reasoning": "", "evidence": []} for d in DIMENSIONS},
            "recommendation": "park",
            "notes": "",
            "open_questions": [],
        }
        score_posting(posting, cfg, fake_claude)
        call = fake_claude.calls[0]
        labels = [label for label, _ in call["stable_context"]]
        assert labels == ["career_dossier", "base_cv_html", "search_strategy"]
        assert "Weaviate" not in "".join(text for _, text in call["stable_context"])
        assert "Weaviate" in call["user_content"]

    def test_cache_breakpoint_is_on_the_last_stable_block(self, cfg, fake_claude):
        blocks = fake_claude.system_blocks(
            "instructions", [("a", "aaa"), ("b", "bbb")]
        )
        assert "cache_control" not in blocks[0]
        assert "cache_control" not in blocks[1]
        assert blocks[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_unknown_constraints_surface_in_notes(self, cfg, fake_claude):
        fake_claude.structured_responses["score"] = {
            **{d: {"score": 3, "reasoning": "", "evidence": []} for d in DIMENSIONS},
            "recommendation": "park",
            "notes": "",
            "open_questions": [],
        }
        report = score_posting(make(location="Amsterdam"), cfg, fake_claude)
        assert "Unverified" in report.notes


def test_dimension_schema_uses_only_supported_keywords():
    """Structured outputs reject `minimum`/`maximum` on integer types with a 400.

    Regression guard: a live `jobsearch score` failed with
    "output_config.format.schema: For 'integer' type, properties maximum,
    minimum are not supported". Any numeric bound must be expressed as an enum.
    """
    from jobsearch.scoring import SCORE_SCHEMA, DIMENSIONS

    banned = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}

    def walk(node, path="root"):
        if isinstance(node, dict):
            if node.get("type") == "integer":
                offenders = banned & node.keys()
                assert not offenders, f"{path}: unsupported keyword(s) {offenders}"
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(SCORE_SCHEMA)
    score = SCORE_SCHEMA["properties"][DIMENSIONS[0]]["properties"]["score"]
    assert score["enum"] == [1, 2, 3, 4, 5]


def test_clamp_score_bounds_out_of_range_values():
    from jobsearch.scoring import _clamp_score

    assert _clamp_score(7) == 5.0
    assert _clamp_score(-2) == 0.0
    assert _clamp_score(3) == 3.0
    assert _clamp_score("not a number") == 0.0


class TestDismissalIsReversible:
    """A role dismissed on a skim is sometimes one you later apply to.

    Withdrawn used to allow only Parked, which made the TUI's `d` effectively
    one-way and contradicted the "press a to restore" hint it printed.
    """

    def test_withdrawn_can_return_to_the_pipeline(self):
        from jobsearch.models import Status, can_transition

        assert can_transition(Status.WITHDRAWN, Status.NOT_STARTED)
        assert can_transition(Status.WITHDRAWN, Status.APPLIED)
        assert can_transition(Status.WITHDRAWN, Status.PARKED)

    def test_transitions_has_no_duplicate_keys(self):
        """A duplicated dict key silently overrides the earlier one."""
        import re
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "src/jobsearch/models.py").read_text()
        block = source[source.index("TRANSITIONS"): source.index("def can_transition")]
        keys = re.findall(r"Status\.([A-Z_]+):", block)
        assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"
