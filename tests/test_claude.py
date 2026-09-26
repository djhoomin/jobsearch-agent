

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


class TestTruncation:
    """A structured answer cut off at max_tokens is half a JSON object; say so."""

    def test_a_structured_call_at_the_limit_names_the_limit(self, monkeypatch):
        from types import SimpleNamespace

        import pytest

        from jobsearch.claude import ClaudeClient, ClaudeError

        response = SimpleNamespace(
            stop_reason="max_tokens",
            content=[SimpleNamespace(type="text", text='{"claims": [')],
            usage=None,
        )
        fake = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))
        client = ClaudeClient(model="claude-opus-5")
        monkeypatch.setattr(type(client), "client", property(lambda self: fake))
        with pytest.raises(ClaudeError, match="output limit of 32000 tokens"):
            client.structured(instructions="i", stable_context=[], user_content="u", schema={})
