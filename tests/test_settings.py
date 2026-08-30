"""Editing settings must never damage the config file.

The file is hand-written and full of explanatory comments; an edit made from
the UI has to leave all of them in place.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from jobsearch.settings import (
    SETTINGS,
    WEIGHT_KEYS,
    SettingsError,
    apply_edits,
    current_values,
    set_array,
    set_scalar,
    validate,
    validate_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (REPO_ROOT / "config.example.toml").read_text(encoding="utf-8")


class TestScalars:
    def test_it_replaces_a_number(self):
        out = set_scalar(TEMPLATE, "comp_floor_eur", 175000)
        assert tomllib.loads(out)["constraints"]["comp_floor_eur"] == 175000

    def test_it_replaces_a_bool(self):
        out = set_scalar(TEMPLATE, "non_compete_waiver_signed", True)
        assert tomllib.loads(out)["constraints"]["non_compete_waiver_signed"] is True

    def test_it_keeps_the_trailing_comment(self):
        text = "  travel_max_percent = 30  # roughly monthly\n"
        assert set_scalar(text, "travel_max_percent", 45) == (
            "  travel_max_percent = 45  # roughly monthly\n"
        )

    def test_an_unknown_key_is_an_error_not_a_silent_no_op(self):
        with pytest.raises(SettingsError, match="not found"):
            set_scalar(TEMPLATE, "no_such_key", 1)


class TestArrays:
    def test_it_replaces_a_multiline_array(self):
        out = set_array(TEMPLATE, "title_exclude", ["intern", "junior"])
        assert tomllib.loads(out)["discover"]["title_exclude"] == ["intern", "junior"]

    def test_it_does_not_disturb_neighbouring_keys(self):
        before = tomllib.loads(TEMPLATE)["discover"]["title_include"]
        out = set_array(TEMPLATE, "title_exclude", ["intern"])
        assert tomllib.loads(out)["discover"]["title_include"] == before

    def test_quotes_in_a_value_survive(self):
        out = set_array(TEMPLATE, "title_exclude", ['say "no"'])
        assert tomllib.loads(out)["discover"]["title_exclude"] == ['say "no"']


class TestValidation:
    def test_a_list_accepts_a_comma_separated_string(self):
        spec = next(s for s in SETTINGS if s.key == "title_exclude")
        assert validate(spec, "intern, junior , ") == ["intern", "junior"]

    def test_an_empty_list_is_rejected(self):
        spec = next(s for s in SETTINGS if s.key == "title_include")
        with pytest.raises(SettingsError, match="cannot be empty"):
            validate(spec, "   ")

    def test_a_number_tolerates_thousands_separators(self):
        spec = next(s for s in SETTINGS if s.key == "comp_floor_eur")
        assert validate(spec, "125,000") == 125000

    def test_a_negative_number_is_rejected(self):
        spec = next(s for s in SETTINGS if s.key == "comp_floor_eur")
        with pytest.raises(SettingsError, match="negative"):
            validate(spec, "-5")

    def test_rubbish_in_a_number_field_is_reported_clearly(self):
        spec = next(s for s in SETTINGS if s.key == "comp_floor_eur")
        with pytest.raises(SettingsError, match="expected int"):
            validate(spec, "lots")

    @pytest.mark.parametrize("raw,expected", [("true", True), ("yes", True), ("no", False), ("", False)])
    def test_bools(self, raw, expected):
        spec = next(s for s in SETTINGS if s.key == "non_compete_waiver_signed")
        assert validate(spec, raw) is expected

    def test_weights_must_sum_to_one(self):
        with pytest.raises(SettingsError, match="sum to 1.0"):
            validate_weights({k: 0.5 for k in WEIGHT_KEYS})

    def test_missing_weights_are_named(self):
        with pytest.raises(SettingsError, match="missing weight"):
            validate_weights({"buyer": 1.0})


class TestApplyEdits:
    def test_every_comment_survives_a_save(self):
        out = apply_edits(
            TEMPLATE,
            {"comp_floor_eur": "150000", "title_exclude": ["intern"]},
            weights={"buyer": 0.3, "role_fit": 0.2, "company": 0.25, "domain": 0.15, "talent": 0.1},
        )
        assert out.count("#") == TEMPLATE.count("#"), "explanatory comments were lost"
        assert tomllib.loads(out)["weights"]["buyer"] == 0.3

    def test_the_result_still_loads_as_a_Config(self, tmp_path):
        from jobsearch.config import Config

        out = apply_edits(TEMPLATE, {"comp_floor_eur": "150000"})
        Config(root=tmp_path, raw=tomllib.loads(out))

    def test_a_bad_weight_set_leaves_the_file_untouched(self):
        """Validation happens before anything is written."""
        with pytest.raises(SettingsError):
            apply_edits(TEMPLATE, {"comp_floor_eur": "150000"}, weights={"buyer": 9.0})

    def test_unknown_settings_are_refused(self):
        with pytest.raises(SettingsError, match="unknown setting"):
            apply_edits(TEMPLATE, {"nope": 1})

    def test_current_values_round_trips(self):
        values = current_values(tomllib.loads(TEMPLATE))
        assert values["comp_floor_eur"] == 100000
        assert isinstance(values["title_include"], list)
        assert values["weight_role_fit"] == 0.25


class TestProviderSettings:
    def test_provider_is_inserted_when_the_file_lacks_it(self):
        """A config written before the setting existed has no line to replace."""
        import tomllib

        text = "[claude]\nmodel = \"claude-opus-5\"\n"
        out = apply_edits(text, {"provider": "anthropic"})
        assert tomllib.loads(out)["claude"]["provider"] == "anthropic"

    def test_an_existing_provider_line_is_replaced_not_duplicated(self):
        import tomllib

        text = '[claude]\nprovider = "anthropic"\nmodel = "x"\n'
        out = apply_edits(text, {"provider": "openai_compatible", "base_url": "http://x"})
        assert out.count("provider =") == 1
        assert tomllib.loads(out)["claude"]["provider"] == "openai_compatible"

    def test_an_unknown_provider_is_refused(self):
        with pytest.raises(SettingsError, match="must be one of"):
            apply_edits(TEMPLATE, {"provider": "carrier pigeon"})

    def test_openai_compatible_requires_a_base_url(self):
        with pytest.raises(SettingsError, match="needs a Base URL"):
            apply_edits(TEMPLATE, {"provider": "openai_compatible", "base_url": "  "})

    def test_anthropic_does_not_require_a_base_url(self):
        apply_edits(TEMPLATE, {"provider": "anthropic", "base_url": ""})

    def test_a_missing_section_is_a_clear_error(self):
        with pytest.raises(SettingsError, match=r"no \[claude\] section"):
            apply_edits("[other]\nx = 1\n", {"provider": "anthropic"})
