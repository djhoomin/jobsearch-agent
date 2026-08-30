

class TestStageOverrides:
    """Per-stage model and effort, so a cheap stage can use a cheap model
    without downgrading the ones where quality matters.
    """

    def test_a_stage_without_an_override_uses_the_defaults(self):
        from jobsearch.claude import ClaudeClient

        client = ClaudeClient(model="claude-opus-5", effort="high")
        assert client.model_for("tailor") == "claude-opus-5"
        assert client.effort_for("tailor") == "high"

    def test_an_override_applies_only_to_its_stage(self):
        from jobsearch.claude import ClaudeClient

        client = ClaudeClient(
            model="claude-opus-5",
            effort="high",
            stage_overrides={"ground": {"model": "claude-haiku-4-5", "effort": "low"}},
        )
        assert client.model_for("ground") == "claude-haiku-4-5"
        assert client.effort_for("ground") == "low"
        assert client.model_for("tailor") == "claude-opus-5"
        assert client.effort_for("tailor") == "high"

    def test_a_partial_override_falls_back_for_the_rest(self):
        from jobsearch.claude import ClaudeClient

        client = ClaudeClient(
            model="claude-opus-5", effort="high",
            stage_overrides={"score": {"effort": "medium"}},
        )
        assert client.model_for("score") == "claude-opus-5"
        assert client.effort_for("score") == "medium"

    def test_overrides_are_read_from_config(self, cfg):
        from jobsearch.claude import ClaudeClient

        cfg.raw.setdefault("claude", {})["stages"] = {
            "ground": {"model": "claude-sonnet-5"}
        }
        client = ClaudeClient.from_config(cfg, dry_run=True)
        assert client.model_for("ground") == "claude-sonnet-5"
        assert client.model_for("tailor") == client.model

    def test_a_malformed_override_is_ignored_not_fatal(self, cfg):
        from jobsearch.claude import ClaudeClient

        cfg.raw.setdefault("claude", {})["stages"] = {"ground": "sonnet"}
        client = ClaudeClient.from_config(cfg, dry_run=True)
        assert client.model_for("ground") == client.model
