"""The headless-CLI provider must satisfy the same contract as the other two.

Nothing here shells out. The CLI is replaced by a fake ``subprocess.run`` that
returns the envelopes the real binary was observed to produce, so the tests
assert on argv construction and envelope handling rather than on the network.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from jobsearch.claude import ClaudeClient, ClaudeError, make_client
from jobsearch.claude_code import ClaudeCodeClient, ClaudeCodeError
from jobsearch.openai_compat import OpenAICompatibleClient


def envelope(**overrides):
    """A result object shaped like the real ``--output-format json`` envelope."""
    base = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "stop_reason": "end_turn",
        "result": '{"score": 4}',
        "structured_output": {"score": 4},
        "total_cost_usd": 0.0164,
        "usage": {
            "input_tokens": 2,
            "output_tokens": 56,
            "cache_read_input_tokens": 22041,
            "cache_creation_input_tokens": 303,
        },
    }
    base.update(overrides)
    return base


class Recorder:
    """Stands in for subprocess.run, keeping the argv it was handed."""

    def __init__(self, stdout: str, returncode: int = 0, stderr: str = ""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr
        self.argv: list[str] = []
        self.kwargs: dict = {}

    def __call__(self, argv, **kwargs):
        self.argv, self.kwargs = list(argv), kwargs
        return subprocess.CompletedProcess(
            argv, self.returncode, stdout=self.stdout, stderr=self.stderr
        )


def fake_cli(monkeypatch, stdout: str, returncode: int = 0, stderr: str = "") -> Recorder:
    recorder = Recorder(stdout, returncode, stderr)
    monkeypatch.setattr("jobsearch.claude_code.subprocess.run", recorder)
    monkeypatch.setattr("jobsearch.claude_code.shutil.which", lambda name: f"/bin/{name}")
    return recorder


class TestProviderSelection:
    def test_claude_code_is_selected_by_config(self, cfg):
        cfg.raw.setdefault("claude", {})["provider"] = "claude_code"
        assert isinstance(make_client(cfg, dry_run=True), ClaudeCodeClient)

    def test_the_aliases_work_too(self, cfg):
        for alias in ("cli", "headless"):
            cfg.raw.setdefault("claude", {})["provider"] = alias
            assert isinstance(make_client(cfg, dry_run=True), ClaudeCodeClient)

    def test_the_unknown_provider_message_lists_all_three(self, cfg):
        cfg.raw.setdefault("claude", {})["provider"] = "carrier pigeon"
        with pytest.raises(ClaudeError, match="claude_code"):
            make_client(cfg, dry_run=True)


class TestInterfaceParity:
    def test_all_three_clients_share_a_surface(self):
        for name in ("structured", "stream_text", "last_usage", "model_for",
                     "effort_for", "from_config"):
            for client in (ClaudeClient, OpenAICompatibleClient, ClaudeCodeClient):
                assert hasattr(client, name), f"{client.__name__} lacks {name}"
        for name in ("model", "dry_run", "dry_run_hook", "stage_overrides"):
            assert name in ClaudeCodeClient.__dataclass_fields__

    def test_dry_run_needs_no_binary(self):
        client = ClaudeCodeClient(binary="definitely-not-installed", dry_run=True)
        assert client.structured(
            instructions="i", stable_context=[], user_content="u",
            schema={}, dry_run_value={"ok": True},
        ) == {"ok": True}
        assert client.stream_text(
            instructions="i", stable_context=[], user_content="u", dry_run_value="hello"
        ) == "hello"

    def test_the_prefix_matches_the_anthropic_path_block_for_block(self):
        """The two providers must build the same prefix text, so a prefix that
        caches on one caches on the other and results stay comparable."""
        context = [("career_dossier", "D"), ("base_cv_html", "C")]
        api = ClaudeClient().system_blocks("INSTR", context)
        cli = ClaudeCodeClient().system_prompt("INSTR", context)
        assert cli == "\n\n".join(block["text"] for block in api)


class TestArgv:
    def test_the_load_bearing_flags_are_all_sent(self, monkeypatch):
        cli = fake_cli(monkeypatch, json.dumps(envelope()))
        ClaudeCodeClient().structured(
            instructions="i", stable_context=[], user_content="u", schema={"type": "object"}
        )
        for flag in ("--print", "--tools", "--safe-mode", "--strict-mcp-config",
                     "--no-session-persistence", "--output-format", "--json-schema"):
            assert flag in cli.argv, f"{flag} missing"
        # An empty --tools value is the documented way to disable the toolset;
        # it must survive as its own empty argv element.
        assert cli.argv[cli.argv.index("--tools") + 1] == ""

    def test_stage_overrides_reach_the_command_line(self, monkeypatch):
        cli = fake_cli(monkeypatch, json.dumps(envelope()))
        client = ClaudeCodeClient(
            model="claude-opus-5", effort="high",
            stage_overrides={"score": {"model": "claude-sonnet-5", "effort": "medium"}},
        )
        client.structured(instructions="i", stable_context=[], user_content="u",
                          schema={}, stage="score")
        assert cli.argv[cli.argv.index("--model") + 1] == "claude-sonnet-5"
        assert cli.argv[cli.argv.index("--effort") + 1] == "medium"

    def test_the_prompt_goes_on_stdin_not_argv(self, monkeypatch):
        """argv is world-readable through ps; a dossier-shaped prompt is not."""
        cli = fake_cli(monkeypatch, json.dumps(envelope()))
        ClaudeCodeClient().structured(
            instructions="i", stable_context=[], user_content="SECRET BRIEF", schema={}
        )
        assert cli.kwargs["input"] == "SECRET BRIEF"
        assert "SECRET BRIEF" not in cli.argv

    def test_the_system_prompt_goes_to_a_file_not_argv(self, monkeypatch):
        cli = fake_cli(monkeypatch, json.dumps(envelope()))
        ClaudeCodeClient().structured(
            instructions="INSTR", stable_context=[("career_dossier", "D" * 5000)],
            user_content="u", schema={},
        )
        path = Path(cli.argv[cli.argv.index("--system-prompt-file") + 1])
        assert path.read_text().startswith("INSTR")
        assert "D" * 5000 in path.read_text()

    def test_the_schema_serialises_deterministically(self, monkeypatch):
        """The schema lands inside the cached prefix. Equal schemas must produce
        equal bytes or every call is a cache miss."""
        cli = fake_cli(monkeypatch, json.dumps(envelope()))
        client = ClaudeCodeClient()
        sent = []
        for schema in ({"b": 1, "a": 2}, {"a": 2, "b": 1}):
            client.structured(instructions="i", stable_context=[], user_content="u",
                              schema=schema)
            sent.append(cli.argv[cli.argv.index("--json-schema") + 1])
        assert sent[0] == sent[1]

    def test_extra_args_are_appended(self, monkeypatch):
        cli = fake_cli(monkeypatch, json.dumps(envelope()))
        ClaudeCodeClient(extra_args=["--fallback-model", "sonnet"]).structured(
            instructions="i", stable_context=[], user_content="u", schema={}
        )
        assert cli.argv[-2:] == ["--fallback-model", "sonnet"]

    def test_a_stray_api_key_is_kept_out_of_the_child(self, monkeypatch):
        """The point of this provider is subscription usage. A key in the
        environment would silently bill the API instead."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
        cli = fake_cli(monkeypatch, json.dumps(envelope()))
        ClaudeCodeClient().structured(instructions="i", stable_context=[],
                                      user_content="u", schema={})
        assert "ANTHROPIC_API_KEY" not in cli.kwargs["env"]

    def test_the_key_is_kept_when_the_switch_is_off(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-deliberate")
        cli = fake_cli(monkeypatch, json.dumps(envelope()))
        ClaudeCodeClient(use_subscription=False).structured(
            instructions="i", stable_context=[], user_content="u", schema={}
        )
        assert cli.kwargs["env"]["ANTHROPIC_API_KEY"] == "sk-deliberate"


class TestStructured:
    def test_it_prefers_the_validated_structured_output(self, monkeypatch):
        fake_cli(monkeypatch, json.dumps(envelope(
            result="prose the model wrapped it in", structured_output={"score": 9}
        )))
        assert ClaudeCodeClient().structured(
            instructions="i", stable_context=[], user_content="u", schema={}
        ) == {"score": 9}

    def test_it_falls_back_to_parsing_the_result_text(self, monkeypatch):
        payload = envelope(result='{"score": 7}')
        payload.pop("structured_output")
        fake_cli(monkeypatch, json.dumps(payload))
        assert ClaudeCodeClient().structured(
            instructions="i", stable_context=[], user_content="u", schema={}
        ) == {"score": 7}

    def test_usage_and_avoided_cost_are_recorded(self, monkeypatch):
        fake_cli(monkeypatch, json.dumps(envelope()))
        client = ClaudeCodeClient()
        client.structured(instructions="i", stable_context=[], user_content="u", schema={})
        assert client.last_usage.cache_read_input_tokens == 22041
        assert client.last_usage.cache_hit
        assert client.last_cost_usd == pytest.approx(0.0164)

    def test_an_empty_result_is_an_error_not_an_empty_dict(self, monkeypatch):
        payload = envelope(result="")
        payload.pop("structured_output")
        fake_cli(monkeypatch, json.dumps(payload))
        with pytest.raises(ClaudeCodeError, match="no result text"):
            ClaudeCodeClient().structured(instructions="i", stable_context=[],
                                          user_content="u", schema={})


class TestStreaming:
    def _stream(self, deltas, **result):
        lines = [json.dumps({"type": "system", "subtype": "init"})]
        lines += [
            json.dumps({
                "type": "stream_event",
                "event": {"type": "content_block_delta",
                          "delta": {"type": "text_delta", "text": text}},
            })
            for text in deltas
        ]
        lines.append(json.dumps(envelope(**result)))
        return "\n".join(lines)

    def test_deltas_are_forwarded_and_joined(self, monkeypatch):
        fake_cli(monkeypatch, self._stream(["<html>", "hello", "</html>"]))
        seen: list[str] = []
        text = ClaudeCodeClient().stream_text(
            instructions="i", stable_context=[], user_content="u", on_delta=seen.append
        )
        assert seen == ["<html>", "hello", "</html>"]
        assert text == "<html>hello</html>"

    def test_thinking_deltas_are_not_treated_as_output(self, monkeypatch):
        stdout = "\n".join([
            json.dumps({"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "hmm"}}}),
            json.dumps({"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "real"}}}),
            json.dumps(envelope()),
        ])
        fake_cli(monkeypatch, stdout)
        assert ClaudeCodeClient().stream_text(
            instructions="i", stable_context=[], user_content="u"
        ) == "real"

    def test_verbose_is_sent_because_the_cli_demands_it(self, monkeypatch):
        cli = fake_cli(monkeypatch, self._stream(["x"]))
        ClaudeCodeClient().stream_text(instructions="i", stable_context=[], user_content="u")
        assert "--verbose" in cli.argv
        assert cli.argv[cli.argv.index("--output-format") + 1] == "stream-json"

    def test_truncation_is_reported_rather_than_silently_returned(self, monkeypatch):
        fake_cli(monkeypatch, self._stream(["half a c"], stop_reason="max_tokens"))
        with pytest.raises(ClaudeCodeError, match="output ceiling"):
            ClaudeCodeClient().stream_text(instructions="i", stable_context=[],
                                           user_content="u")

    def test_a_run_with_no_partial_events_still_returns_its_text(self, monkeypatch):
        fake_cli(monkeypatch, json.dumps(envelope(result="the whole thing")))
        assert ClaudeCodeClient().stream_text(
            instructions="i", stable_context=[], user_content="u"
        ) == "the whole thing"


class TestFailures:
    def test_a_missing_binary_names_the_way_out(self, monkeypatch):
        monkeypatch.setattr("jobsearch.claude_code.shutil.which", lambda name: None)
        with pytest.raises(ClaudeCodeError, match="not found on PATH"):
            ClaudeCodeClient(binary="nope").structured(
                instructions="i", stable_context=[], user_content="u", schema={}
            )

    def test_an_error_envelope_is_raised_not_parsed(self, monkeypatch):
        fake_cli(monkeypatch, json.dumps(envelope(
            is_error=True, subtype="error_during_execution", result="upstream said no"
        )))
        with pytest.raises(ClaudeCodeError, match="upstream said no"):
            ClaudeCodeClient().structured(instructions="i", stable_context=[],
                                          user_content="u", schema={})

    def test_no_envelope_reports_the_stderr(self, monkeypatch):
        fake_cli(monkeypatch, "", returncode=1, stderr="Error: unknown option --nope")
        with pytest.raises(ClaudeCodeError, match="unknown option"):
            ClaudeCodeClient().structured(instructions="i", stable_context=[],
                                          user_content="u", schema={})

    def test_a_rate_limit_event_is_mentioned_in_the_failure(self, monkeypatch):
        """Running out of included usage is the expected failure of this
        provider, so it must not surface as a bare exit code."""
        fake_cli(monkeypatch, json.dumps({"type": "rate_limit_event"}), returncode=1)
        with pytest.raises(ClaudeCodeError, match="Usage limits"):
            ClaudeCodeClient().structured(instructions="i", stable_context=[],
                                          user_content="u", schema={})

    def test_a_timeout_names_the_setting_that_fixes_it(self, monkeypatch):
        monkeypatch.setattr("jobsearch.claude_code.shutil.which", lambda name: "/bin/claude")

        def boom(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 900)

        monkeypatch.setattr("jobsearch.claude_code.subprocess.run", boom)
        with pytest.raises(ClaudeCodeError, match="timeout"):
            ClaudeCodeClient().structured(instructions="i", stable_context=[],
                                          user_content="u", schema={})
