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
