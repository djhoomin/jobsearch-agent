"""Discovery: ATS payload parsing, title filtering, and the no-scrape refusals.

No network access happens in these tests: the ATS adapters are pure functions
over recorded payload shapes.
"""

from __future__ import annotations

import pytest

from jobsearch.config import BoardRef
from jobsearch.discover import DiscoveryError, filter_postings, title_matches
from jobsearch.discover.single import FORBIDDEN_HOSTS, fetch_single_posting
from jobsearch.discover.sources import (
    board_url,
    parse_ashby,
    parse_greenhouse,
    parse_json_feed,
    parse_lever,
    strip_html,
)

GREENHOUSE = BoardRef(company="DataSnipper", ats="greenhouse", token="datasnipper", tier=1, ind_sponsor=True)
LEVER = BoardRef(company="Mistral AI", ats="lever", token="mistral", tier=1, ind_sponsor=True)
ASHBY = BoardRef(company="ElevenLabs", ats="ashby", token="elevenlabs", tier=1, ind_sponsor="unknown")


class TestBoardUrls:
    def test_greenhouse(self):
        assert board_url(GREENHOUSE) == (
            "https://boards-api.greenhouse.io/v1/boards/datasnipper/jobs?content=true"
        )

    def test_lever(self):
        assert board_url(LEVER) == "https://api.lever.co/v0/postings/mistral?mode=json"

    def test_ashby(self):
        assert board_url(ASHBY).startswith(
            "https://api.ashbyhq.com/posting-api/job-board/elevenlabs"
        )

    def test_json_feed_requires_a_url(self):
        with pytest.raises(DiscoveryError, match="requires a `url`"):
            board_url(BoardRef(company="X", ats="json_feed", token="x"))

    def test_unknown_ats_is_reported_clearly(self):
        with pytest.raises(DiscoveryError, match="unknown ats"):
            board_url(BoardRef(company="X", ats="workday", token="x"))


class TestGreenhouseParsing:
    PAYLOAD = {
        "jobs": [
            {
                "id": 4321,
                "title": "Director of AI",
                "absolute_url": "https://boards.greenhouse.io/datasnipper/jobs/4321",
                "location": {"name": "Amsterdam"},
                "offices": [{"name": "Amsterdam HQ"}],
                "departments": [{"name": "Engineering"}],
                "content": "<p>Own the AI org.</p><ul><li>Ship agents</li></ul>",
                "updated_at": "2026-08-20T10:00:00Z",
            }
        ]
    }

    def test_maps_core_fields(self):
        posting = parse_greenhouse(self.PAYLOAD, GREENHOUSE)[0]
        assert posting.company == "DataSnipper"
        assert posting.title == "Director of AI"
        assert posting.location == "Amsterdam"
        assert posting.department == "Engineering"
        assert posting.posted_at == "2026-08-20"
        assert posting.source == "greenhouse"

    def test_carries_board_metadata_into_the_constraints(self):
        posting = parse_greenhouse(self.PAYLOAD, GREENHOUSE)[0]
        assert posting.ind_sponsor == "yes"
        assert posting.board_tier == 1

    def test_html_description_is_flattened(self):
        posting = parse_greenhouse(self.PAYLOAD, GREENHOUSE)[0]
        assert "<p>" not in posting.description
        assert "Own the AI org." in posting.description
        assert "- Ship agents" in posting.description

    def test_empty_payload(self):
        assert parse_greenhouse({}, GREENHOUSE) == []


class TestLeverParsing:
    PAYLOAD = [
        {
            "text": "Applied AI Technical Lead, EMEA",
            "hostedUrl": "https://jobs.lever.co/mistral/abc",
            "categories": {"location": "Paris, France", "team": "Applied AI"},
            "descriptionPlain": "Lead the applied AI team.",
            "lists": [{"text": "Requirements", "content": "<li>10 years</li>"}],
            "createdAt": 1755600000000,
        }
    ]

    def test_maps_core_fields(self):
        posting = parse_lever(self.PAYLOAD, LEVER)[0]
        assert posting.title == "Applied AI Technical Lead, EMEA"
        assert posting.location == "Paris, France"
        assert posting.department == "Applied AI"
        assert "Lead the applied AI team." in posting.description

    def test_list_sections_are_appended(self):
        posting = parse_lever(self.PAYLOAD, LEVER)[0]
        assert "Requirements" in posting.description
        assert "10 years" in posting.description

    def test_epoch_millis_become_a_date(self):
        assert parse_lever(self.PAYLOAD, LEVER)[0].posted_at.startswith("2025-")


class TestAshbyParsing:
    PAYLOAD = {
        "jobs": [
            {
                "title": "Deployment Strategist Lead",
                "jobUrl": "https://jobs.ashbyhq.com/elevenlabs/xyz",
                "location": "Netherlands",
                "department": "Go to Market",
                "descriptionPlain": "Operate as if a co-founder.",
                "publishedAt": "2026-08-01T00:00:00Z",
                "isRemote": True,
                "compensation": {"compensationTierSummary": "EUR 140K - 180K"},
            }
        ]
    }

    def test_maps_core_fields(self):
        posting = parse_ashby(self.PAYLOAD, ASHBY)[0]
        assert posting.title == "Deployment Strategist Lead"
        assert posting.location == "Netherlands"
        assert posting.remote is True

    def test_compensation_is_extracted_for_the_comp_filter(self):
        posting = parse_ashby(self.PAYLOAD, ASHBY)[0]
        assert "140K" in posting.salary_text

    def test_unknown_sponsor_state_is_preserved(self):
        assert parse_ashby(self.PAYLOAD, ASHBY)[0].ind_sponsor == "unknown"


class TestJsonFeedParsing:
    def test_bare_list(self):
        payload = [{"title": "Head of AI", "url": "https://x/1", "location": "Amsterdam"}]
        board = BoardRef(company="X", ats="json_feed", token="x", url="https://x/feed.json")
        assert parse_json_feed(payload, board)[0].title == "Head of AI"

    def test_wrapped_list(self):
        payload = {"results": [{"name": "Head of AI", "link": "https://x/1"}]}
        board = BoardRef(company="X", ats="json_feed", token="x", url="https://x/feed.json")
        assert parse_json_feed(payload, board)[0].url == "https://x/1"

    def test_unrecognisable_payload_yields_nothing(self):
        board = BoardRef(company="X", ats="json_feed", token="x", url="https://x")
        assert parse_json_feed({"nope": 1}, board) == []


class TestTitleFiltering:
    def test_matches_target_titles(self):
        assert title_matches("Head of AI", ["head of ai"], [])
        assert title_matches("Director, AI Platform", ["director, ai"], [])

    def test_excludes_junior_roles(self):
        assert not title_matches("AI Lead Intern", ["ai lead"], ["intern"])

    def test_exclusion_beats_inclusion(self):
        assert not title_matches("Junior Head of AI", ["head of ai"], ["junior"])

    def test_no_include_list_matches_everything(self):
        assert title_matches("Anything", [], [])

    def test_include_terms_match_whole_words_only(self):
        """The defect this replaced: "ai" was a substring match, so it fired on
        chAIn, retAIl and mAIntenance. 213 postings entered the pipeline whose
        only qualification was letters sitting inside an unrelated word."""
        assert title_matches("AI Scientist", ["ai"], [])
        assert title_matches("Head of AI", ["ai"], [])
        assert not title_matches("Supply Chain Data Engineer", ["ai"], [])
        assert not title_matches("Sr. Solutions Architect - Retail", ["ai"], [])
        assert not title_matches("Maintenance Planner", ["ai"], [])

    def test_ml_does_not_fire_on_html(self):
        assert title_matches("Staff ML Engineer", ["ml"], [])
        assert not title_matches("HTML Email Developer", ["ml"], [])

    def test_exclude_terms_match_whole_words_only(self):
        """Cuts both ways: substring excludes were over-eager as well."""
        assert not title_matches("Data Science Intern", ["data"], ["intern"])
        assert title_matches("Internal Tools Data Lead", ["data"], ["intern"])

    def test_phrases_still_match_across_words(self):
        assert title_matches("Sr. Forward Deployed Engineer", ["forward deployed"], [])
        assert title_matches("Head of Solutions Engineering", ["solutions engineering"], [])

    def test_require_any_is_a_second_axis_not_a_second_chance(self):
        """`include` is seniority, `require_any` is subject. The AND only bites
        while the lists stay disjoint - a seniority word in both collapses it
        back to an OR, which is how "Director Global Benefits" got in."""
        seniority, subject = ["director"], ["ai", "machine learning"]
        assert title_matches("Director of AI", seniority, [], subject)
        assert not title_matches("Director Global Benefits", seniority, [], subject)
        # The regression itself: "director" present in both lists lets anything
        # with the word "director" through, however unrelated.
        assert title_matches("Director Global Benefits", seniority, [], subject + ["director"])

    def test_leadership_phrases_admit_the_role_without_admitting_the_noise(self):
        """A target company's "Director of Product" is the AI seat even when the
        title never says AI. Caught as a phrase, so the HR director stays out."""
        subject = ["ai", "director of product"]
        assert title_matches("Director of Product", ["director"], [], subject)
        assert not title_matches("Director Global Benefits", ["director"], [], subject)

    def test_filter_postings_uses_config(self, cfg):
        from jobsearch.models import JobPosting

        postings = [
            JobPosting(company="X", title="Head of AI", url="https://x/1"),
            JobPosting(company="X", title="Account Executive", url="https://x/2"),
        ]
        kept = filter_postings(postings, cfg)
        assert [p.title for p in kept] == ["Head of AI"]


class TestNoScrapePolicy:
    """The tool must refuse hosts whose terms forbid automated access."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.linkedin.com/jobs/view/1234567890",
            "https://uk.indeed.com/viewjob?jk=abc",
            "https://www.glassdoor.com/job-listing/x",
        ],
    )
    def test_refuses_forbidden_hosts(self, url):
        with pytest.raises(DiscoveryError, match="terms of service"):
            fetch_single_posting(url, fetcher=None, cfg=None)

    def test_the_refusal_suggests_the_manual_path(self):
        with pytest.raises(DiscoveryError, match="jobsearch add"):
            fetch_single_posting("https://www.linkedin.com/jobs/view/1", None, None)

    def test_linkedin_is_on_the_list(self):
        assert "linkedin.com" in FORBIDDEN_HOSTS
        assert "indeed.com" in FORBIDDEN_HOSTS


class TestStripHtml:
    def test_removes_tags_and_keeps_text(self):
        assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_list_items_become_dashes(self):
        assert "- one" in strip_html("<ul><li>one</li><li>two</li></ul>")

    def test_scripts_are_dropped(self):
        assert "alert" not in strip_html("<script>alert(1)</script><p>text</p>")

    def test_entities_are_decoded(self):
        assert strip_html("<p>R&amp;D</p>") == "R&D"


class TestRobotsPolicy:
    """robots.txt is enforced for pages, not for the documented ATS APIs."""

    def test_the_three_ats_apis_are_exempt(self):
        from jobsearch.discover.sources import is_public_ats_api

        assert is_public_ats_api("https://api.ashbyhq.com/posting-api/job-board/weaviate")
        assert is_public_ats_api("https://boards-api.greenhouse.io/v1/boards/x/jobs")
        assert is_public_ats_api("https://api.lever.co/v0/postings/x?mode=json")

    def test_everything_else_is_still_robots_checked(self):
        from jobsearch.discover.sources import is_public_ats_api

        assert not is_public_ats_api("https://careers.example.com/jobs.json")
        assert not is_public_ats_api("https://www.linkedin.com/jobs/view/1")

    def test_a_page_fetch_honours_a_disallow(self, monkeypatch):
        from jobsearch.discover.sources import DiscoveryError, Fetcher

        fetcher = Fetcher(user_agent="test")
        monkeypatch.setattr(fetcher, "allowed", lambda url: False)
        with pytest.raises(DiscoveryError, match="robots.txt disallows"):
            fetcher.get("https://careers.example.com/jobs.json")

    def test_the_exemption_is_explicit_not_a_global_override(self, monkeypatch):
        """Passing check_robots=False is what skips it - nothing else does."""
        from jobsearch.discover.sources import Fetcher

        fetcher = Fetcher(user_agent="test")
        monkeypatch.setattr(fetcher, "allowed", lambda url: False)
        monkeypatch.setattr(fetcher, "_throttle", lambda: None)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("should reach the network")),
        )
        with pytest.raises(AssertionError, match="should reach the network"):
            fetcher.get("https://api.ashbyhq.com/x", check_robots=False)


class TestAshbyLocation:
    """Ashby posts one record per multi-city role; the extras live in
    `secondaryLocations`. Dropping them mislabels a role you can actually do.
    """

    def test_it_joins_primary_and_secondary(self):
        from jobsearch.discover.sources import ashby_location

        job = {
            "location": "Paris",
            "secondaryLocations": [{"location": "Amsterdam"}, {"location": "London"}],
        }
        assert ashby_location(job) == "Paris / Amsterdam / London"

    def test_primary_only(self):
        from jobsearch.discover.sources import ashby_location

        assert ashby_location({"location": "Amsterdam"}) == "Amsterdam"

    def test_it_deduplicates(self):
        from jobsearch.discover.sources import ashby_location

        job = {"location": "Paris", "secondaryLocations": [{"location": "Paris"}]}
        assert ashby_location(job) == "Paris"

    def test_it_tolerates_bare_strings(self):
        from jobsearch.discover.sources import ashby_location

        assert ashby_location({"location": "Paris", "secondaryLocations": ["Amsterdam"]}) == "Paris / Amsterdam"

    def test_the_remote_flag_is_not_folded_into_the_location(self):
        """Appending "remote" would make every non-blocklisted country pass.

        "remote" is an allowed location pattern, so "Australia (remote)" would
        read as workable. The flag stays on JobPosting.remote instead.
        """
        from jobsearch.discover.sources import ashby_location

        assert ashby_location({"location": "Australia", "isRemote": True}) == "Australia"

    def test_empty_is_safe(self):
        from jobsearch.discover.sources import ashby_location

        assert ashby_location({}) == ""
        assert ashby_location({"secondaryLocations": None}) == ""

    def test_a_secondary_amsterdam_rescues_an_eu_role(self, cfg):
        """The bug in the wild: an EMEA role headed 'Paris' also open in Amsterdam."""
        from jobsearch.discover.sources import ashby_location
        from jobsearch.tui import location_cell

        job = {"location": "Paris", "secondaryLocations": [{"location": "Amsterdam"}]}
        assert location_cell({"location": ashby_location(job)}, cfg).startswith("✓")


class TestTitleRequireAny:
    """Seniority words alone are ambiguous. 'Staff Software Engineer - Backend'
    and 'Staff Applied AI Researcher' both match 'staff'; only one is the job.
    """

    INCLUDE = ["staff", "principal", "director", "head of", "forward deployed"]
    REQUIRE = ["ai", "ml", "research", "forward deployed", "solutions", "director", "product"]

    def _match(self, title, require=None):
        from jobsearch.discover.sources import title_matches

        return title_matches(title, self.INCLUDE, [], self.REQUIRE if require is None else require)

    def test_generic_platform_engineering_is_dropped(self):
        for title in (
            "Staff Software Engineer - Backend",
            "Senior Staff Software Engineer - Delta",
            "Principal Software Engineer - Postgres",
            "Principal Engineer - Privacy",
        ):
            assert not self._match(title), title

    def test_ai_flavoured_seniority_is_kept(self):
        for title in (
            "Staff / Principal Applied AI Researcher (Agentic Search)",
            "Principal Software Engineer - AI Poland",
            "Forward Deployed Engineer",
            "Director of Product & Engineering",
        ):
            assert self._match(title), title

    def test_it_matches_on_word_boundaries(self):
        """'ai' must not fire on 'maintain', 'ml' must not fire on 'html'."""
        assert not self._match("Staff Engineer - Maintenance Tooling")
        assert not self._match("Principal Engineer, HTML Rendering")

    def test_an_empty_require_list_disables_the_gate(self):
        assert self._match("Staff Software Engineer - Backend", require=[])

    def test_exclude_still_wins_over_everything(self):
        from jobsearch.discover.sources import title_matches

        assert not title_matches(
            "Junior AI Researcher", self.INCLUDE + ["junior"], ["junior"], self.REQUIRE
        )

    def test_include_is_still_required(self):
        """The gate narrows; it does not admit titles the include list rejects."""
        assert not self._match("Data Analyst")
