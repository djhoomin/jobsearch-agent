"""Name handling for the IND register sweep."""

from jobsearch.sponsor_sweep import Company, base_name, slug_candidates, worth_probing


def test_base_name_strips_legal_forms_and_country_words():
    assert base_name("Adyen N.V.") == "adyen"
    assert base_name("Picnic Technologies B.V.") == "picnic technologies"
    assert base_name("Backbase Europe B.V.") == "backbase"


def test_slugs_cover_joined_hyphenated_and_first_word():
    assert slug_candidates("picnic technologies") == ["picnictechnologies", "picnic-technologies", "picnic"]
    assert slug_candidates("bunq") == ["bunq"]


def test_short_first_words_are_not_guessed_alone():
    assert "ing" not in slug_candidates("ing bank")


def test_public_and_care_sector_names_are_skipped():
    for name in ("Gemeente Amsterdam", "Stichting Zorgpartners", "Randstad Uitzendbureau B.V."):
        assert not worth_probing(Company(name=name, kvk="1", base=base_name(name)))
    assert worth_probing(Company(name="Mollie B.V.", kvk="1", base="mollie"))
