"""The OpenAI-compatible provider must satisfy the same contract as the
Anthropic one, without either knowing about the other.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from jobsearch.claude import ClaudeClient, ClaudeError, make_client
from jobsearch.openai_compat import OpenAICompatibleClient, extract_json


class TestExtractJson:
    def test_plain_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_with_leading_prose(self):
        """Endpoints that ignore response_format often introduce the object."""
        assert extract_json('Sure! Here it is:\n{"a": 1}\nHope that helps.') == {"a": 1}

    def test_braces_inside_strings_do_not_confuse_it(self):
        assert extract_json('prefix {"a": "an { unbalanced brace"} tail') == {
            "a": "an { unbalanced brace"
        }

    def test_nested_objects(self):
        assert extract_json('{"a": {"b": {"c": 1}}}') == {"a": {"b": {"c": 1}}}

    def test_unparseable_raises_with_a_snippet(self):
        with pytest.raises(ClaudeError, match="not valid JSON"):
            extract_json("no json here at all")


class TestProviderSelection:
    def test_anthropic_is_the_default(self, cfg):
        assert isinstance(make_client(cfg, dry_run=True), ClaudeClient)

    def test_openai_compatible_is_selected_by_config(self, cfg):
        cfg.raw.setdefault("claude", {}).update(
            {"provider": "openai_compatible", "base_url": "https://openrouter.ai/api/v1"}
        )
        assert isinstance(make_client(cfg, dry_run=True), OpenAICompatibleClient)

    def test_a_missing_base_url_is_a_clear_error(self, cfg):
        cfg.raw.setdefault("claude", {})["provider"] = "openai_compatible"
        cfg.raw["claude"].pop("base_url", None)
        with pytest.raises(ClaudeError, match="base_url"):
            make_client(cfg, dry_run=True)

    def test_an_unknown_provider_is_refused(self, cfg):
        cfg.raw.setdefault("claude", {})["provider"] = "carrier pigeon"
        with pytest.raises(ClaudeError, match="Unknown"):
            make_client(cfg, dry_run=True)


class TestInterfaceParity:
    """Whatever the pipeline calls on one client must exist on the other."""

    def test_the_two_clients_share_a_surface(self):
        methods = ("structured", "stream_text", "last_usage", "model_for",
                   "effort_for", "from_config")
        for name in methods:
            assert hasattr(ClaudeClient, name), f"ClaudeClient lacks {name}"
            assert hasattr(OpenAICompatibleClient, name), f"OpenAICompatibleClient lacks {name}"

        # Data attributes are dataclass fields; one without a default is not a
        # class attribute, so hasattr on the class is the wrong check.
        for name in ("model", "max_tokens", "streaming_max_tokens", "dry_run",
                     "dry_run_hook", "stage_overrides"):
            assert name in ClaudeClient.__dataclass_fields__, f"ClaudeClient lacks {name}"
            assert name in OpenAICompatibleClient.__dataclass_fields__, (
                f"OpenAICompatibleClient lacks {name}"
            )

    def test_dry_run_needs_no_credentials(self):
        client = OpenAICompatibleClient(model="x", base_url="http://local", dry_run=True)
        assert client.structured(
            instructions="i", stable_context=[], user_content="u",
            schema={}, dry_run_value={"ok": True},
        ) == {"ok": True}
        assert client.stream_text(
            instructions="i", stable_context=[], user_content="u", dry_run_value="hello"
        ) == "hello"

    def test_the_stable_prefix_keeps_its_order(self):
        """Byte-identical prefixes are the only way an endpoint's own prefix
        caching can hit, since this dialect has no explicit cache_control."""
        client = OpenAICompatibleClient(model="x", base_url="http://local")
        messages = client.messages_for("INSTR", [("dossier", "D"), ("cv", "C")], "ASK")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].index("<dossier>") < messages[0]["content"].index("<cv>")
        assert messages[1] == {"role": "user", "content": "ASK"}


class TestCalls:
    def _client(self, monkeypatch, response):
        client = OpenAICompatibleClient(model="m", base_url="http://local")
        fake = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: response)
            )
        )
        monkeypatch.setattr(type(client), "client", property(lambda self: fake))
        return client

    def test_structured_parses_and_records_usage(self, monkeypatch):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"score": 4}'))],
            usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=20,
                prompt_tokens_details=SimpleNamespace(cached_tokens=80),
            ),
        )
        client = self._client(monkeypatch, response)
        assert client.structured(
            instructions="i", stable_context=[], user_content="u", schema={"type": "object"}
        ) == {"score": 4}
        assert client.last_usage.input_tokens == 100
        assert client.last_usage.cache_read_input_tokens == 80

    def test_an_empty_response_is_an_error_not_an_empty_dict(self, monkeypatch):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=""))], usage=None
        )
        client = self._client(monkeypatch, response)
        with pytest.raises(ClaudeError, match="no content"):
            client.structured(
                instructions="i", stable_context=[], user_content="u", schema={}
            )

    def test_missing_openai_package_explains_the_extra(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "openai":
                raise ModuleNotFoundError("no openai")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        client = OpenAICompatibleClient(model="m", base_url="http://local")
        with pytest.raises(ClaudeError, match=r"\.\[openai\]"):
            _ = client.client


class TestNormaliseBaseUrl:
    """Providers document the full endpoint; the SDK appends the path itself.
    Pasting the documented URL produces a 404 that says nothing about why.
    """

    def test_it_strips_the_endpoint_path(self):
        from jobsearch.openai_compat import normalise_base_url

        assert normalise_base_url("https://openrouter.ai/api/v1/chat/completions") == (
            "https://openrouter.ai/api/v1"
        )

    def test_a_correct_base_url_is_untouched(self):
        from jobsearch.openai_compat import normalise_base_url

        assert normalise_base_url("https://openrouter.ai/api/v1") == "https://openrouter.ai/api/v1"

    def test_a_trailing_slash_is_removed(self):
        from jobsearch.openai_compat import normalise_base_url

        assert normalise_base_url("https://openrouter.ai/api/v1/") == "https://openrouter.ai/api/v1"

    def test_a_local_endpoint_keeps_its_v1(self):
        """Ollama and vLLM are served at /v1; that is the root, not an endpoint."""
        from jobsearch.openai_compat import normalise_base_url

        assert normalise_base_url("http://localhost:11434/v1") == "http://localhost:11434/v1"

    def test_it_is_applied_when_building_from_config(self, cfg):
        from jobsearch.openai_compat import OpenAICompatibleClient

        cfg.raw.setdefault("claude", {}).update({
            "provider": "openai_compatible",
            "base_url": "https://openrouter.ai/api/v1/chat/completions",
            "model": "z-ai/glm-5.3",
        })
        client = OpenAICompatibleClient.from_config(cfg, dry_run=True)
        assert client.base_url == "https://openrouter.ai/api/v1"


class TestPromptCaching:
    """OpenRouter caches two ways: automatically for most providers, and via
    explicit cache_control breakpoints for Anthropic and Qwen.
    """

    def test_providers_needing_breakpoints_are_detected(self):
        from jobsearch.openai_compat import supports_explicit_cache

        assert supports_explicit_cache("anthropic/claude-opus-4.1")
        assert supports_explicit_cache("qwen/qwen3-max")
        assert not supports_explicit_cache("z-ai/glm-5.3")
        assert not supports_explicit_cache("openai/gpt-5")
        assert not supports_explicit_cache("")

    def test_automatic_providers_get_one_stable_system_message(self):
        """They need only a byte-identical prefix, not breakpoints."""
        from jobsearch.openai_compat import OpenAICompatibleClient

        client = OpenAICompatibleClient(model="z-ai/glm-5.3", base_url="http://x")
        messages = client.messages_for("INSTR", [("dossier", "D")], "ASK")
        assert isinstance(messages[0]["content"], str)
        assert "cache_control" not in str(messages)

    def test_the_breakpoint_covers_the_prefix_and_not_the_question(self):
        """The volatile per-role text must fall outside the breakpoint or
        nothing ever hits."""
        from jobsearch.openai_compat import OpenAICompatibleClient

        client = OpenAICompatibleClient(model="anthropic/claude-opus-4.1", base_url="http://x")
        blocks = client.messages_for(
            "INSTR", [("dossier", "D"), ("base_cv", "C")], "ASK", explicit_cache=True
        )[1]["content"]
        assert "cache_control" in blocks[0]
        assert "dossier" in blocks[0]["text"] and "base_cv" in blocks[0]["text"]
        assert blocks[1] == {"type": "text", "text": "ASK"}
        assert "cache_control" not in blocks[1]

    def test_the_prefix_is_byte_identical_between_calls(self):
        """Any drift in the prefix means a permanent cache miss."""
        from jobsearch.openai_compat import OpenAICompatibleClient

        client = OpenAICompatibleClient(model="anthropic/claude-opus-4.1", base_url="http://x")
        first = client.messages_for("I", [("dossier", "D")], "question one", explicit_cache=True)
        second = client.messages_for("I", [("dossier", "D")], "question two", explicit_cache=True)
        assert first[1]["content"][0] == second[1]["content"][0]

    def test_the_setting_can_be_forced_either_way(self):
        from jobsearch.openai_compat import OpenAICompatibleClient

        on = OpenAICompatibleClient(model="z-ai/glm-5.3", base_url="http://x", explicit_cache="true")
        off = OpenAICompatibleClient(
            model="anthropic/claude-opus-4.1", base_url="http://x", explicit_cache="false"
        )
        assert on.uses_explicit_cache("score")
        assert not off.uses_explicit_cache("score")

    def test_cache_writes_are_recorded_as_well_as_reads(self):
        from types import SimpleNamespace

        from jobsearch.openai_compat import OpenAICompatibleClient

        client = OpenAICompatibleClient(model="m", base_url="http://x")
        client._record_usage(SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=10339, completion_tokens=50,
            prompt_tokens_details=SimpleNamespace(cached_tokens=10318, cache_write_tokens=21),
        )))
        assert client.last_usage.cache_read_input_tokens == 10318
        assert client.last_usage.cache_creation_input_tokens == 21
        assert client.last_usage.cache_hit


class TestStickySessions:
    """Without a session id, OpenRouter pins a provider only *after* it sees a
    cache hit, which cannot happen when the first call warms a cache the second
    call never reaches.
    """

    def test_a_session_id_is_sent(self, cfg):
        from jobsearch.openai_compat import OpenAICompatibleClient

        cfg.raw.setdefault("claude", {}).update({"base_url": "http://x", "model": "m"})
        client = OpenAICompatibleClient.from_config(cfg, dry_run=True)
        assert client._params("score", 100)["extra_body"]["session_id"]

    def test_it_is_stable_across_clients(self, cfg):
        """Two runs of the tool must land on the same provider."""
        from jobsearch.openai_compat import OpenAICompatibleClient

        cfg.raw.setdefault("claude", {}).update({"base_url": "http://x", "model": "m"})
        first = OpenAICompatibleClient.from_config(cfg, dry_run=True)
        second = OpenAICompatibleClient.from_config(cfg, dry_run=True)
        assert first.session_id == second.session_id

    def test_it_is_the_same_for_every_stage(self, cfg):
        """One provider for all traffic, not one per stage."""
        from jobsearch.openai_compat import OpenAICompatibleClient

        cfg.raw.setdefault("claude", {}).update({"base_url": "http://x", "model": "m"})
        client = OpenAICompatibleClient.from_config(cfg, dry_run=True)
        ids = {client._params(s, 100)["extra_body"]["session_id"] for s in ("score", "tailor", "letter")}
        assert len(ids) == 1

    def test_an_explicit_session_id_wins(self, cfg):
        from jobsearch.openai_compat import OpenAICompatibleClient

        cfg.raw.setdefault("claude", {}).update(
            {"base_url": "http://x", "model": "m", "session_id": "mine"}
        )
        assert OpenAICompatibleClient.from_config(cfg, dry_run=True).session_id == "mine"

    def test_it_is_capped_at_the_documented_limit(self):
        from jobsearch.openai_compat import OpenAICompatibleClient

        client = OpenAICompatibleClient(model="m", base_url="http://x", session_id="x" * 400)
        assert len(client._params("score", 100)["extra_body"]["session_id"]) == 256
