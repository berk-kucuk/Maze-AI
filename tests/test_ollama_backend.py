"""Tests for the Ollama backend: capabilities, payload shaping, errors."""

from __future__ import annotations

import json

import pytest
import requests

from maze_ai.llm import ollama_backend as ob
from maze_ai.llm.base import LLMError
from maze_ai.llm.ollama_backend import OllamaBackend, auto_context, parse_size_hint, short_size


class FakeResponse:
    def __init__(self, payload=None, status=200, lines=None, text=""):
        self._payload = payload if payload is not None else {}
        self.status_code = status
        self._lines = lines or []
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def iter_lines(self, decode_unicode=False):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass


@pytest.fixture
def captured(monkeypatch):
    """Capture the payload the backend would send."""
    sent: dict = {}

    def fake_request(method, url, **kwargs):
        sent["method"] = method
        sent["url"] = url
        sent["json"] = kwargs.get("json")
        return FakeResponse({"message": {"role": "assistant", "content": "ok"},
                             "eval_count": 10, "eval_duration": 1_000_000_000})

    monkeypatch.setattr(ob, "request_with_retry", fake_request)
    return sent


# ── context sizing ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("ram,limit,expected", [
    (64, 262144, 32768),    # big machine, huge model → our ceiling
    (32, 262144, 32768),
    (16, 262144, 16384),    # mid machine → smaller slice
    (8, 131072, 8192),
    (4, 131072, 4096),      # small machine → the floor
    (64, 4096, 4096),       # model's own limit wins when it is lower
    (16, 0, 16384),         # unknown limit → the machine's budget
])
def test_auto_context(ram, limit, expected):
    assert auto_context(limit, ram_gb=ram) == expected


def test_resolved_ctx_prefers_an_explicit_number():
    backend = OllamaBackend(model="m", num_ctx=12000)
    assert backend.resolved_ctx() == 12000


def test_resolved_ctx_auto_uses_the_model_limit(monkeypatch):
    backend = OllamaBackend(model="m", num_ctx="auto")
    backend._info_cache["m"] = {"context_length": 8192, "capabilities": []}
    monkeypatch.setattr(ob, "_system_ram_gb", lambda: 64)
    assert backend.resolved_ctx() == 8192


# ── capabilities ───────────────────────────────────────────────────────────
def test_capabilities_come_from_the_server(monkeypatch):
    backend = OllamaBackend(model="mystery-model")
    monkeypatch.setattr(ob.requests, "post", lambda *a, **k: FakeResponse({
        "capabilities": ["completion", "tools", "vision"],
        "details": {"context_length": 40960, "parameter_size": "8B"},
    }))
    assert backend.supports_native_tools
    assert backend.supports_vision
    assert not backend.supports_thinking
    assert backend.context_limit() == 40960


def test_capabilities_fall_back_to_name_markers(monkeypatch):
    def boom(*a, **k):
        raise requests.RequestException("no server")

    monkeypatch.setattr(ob.requests, "post", boom)
    assert OllamaBackend(model="llava:7b").supports_vision
    assert OllamaBackend(model="qwen2.5:7b").supports_native_tools
    assert not OllamaBackend(model="some-unknown-model").supports_native_tools


def test_model_info_is_cached(monkeypatch):
    calls = []

    def counting_post(*a, **k):
        calls.append(1)
        return FakeResponse({"capabilities": ["tools"], "details": {}})

    monkeypatch.setattr(ob.requests, "post", counting_post)
    backend = OllamaBackend(model="m")
    assert backend.supports_native_tools
    assert backend.supports_native_tools
    assert len(calls) == 1


# ── payload shaping ────────────────────────────────────────────────────────
def test_payload_carries_keep_alive_and_context(captured):
    backend = OllamaBackend(model="m", num_ctx=6000, keep_alive="2h")
    backend._info_cache["m"] = {"capabilities": [], "context_length": 8192}
    backend.chat_ex([{"role": "user", "content": "hi"}])
    payload = captured["json"]
    assert payload["keep_alive"] == "2h"
    assert payload["options"]["num_ctx"] == 6000
    assert payload["stream"] is False
    assert "tools" not in payload and "format" not in payload


def test_payload_includes_tools_and_schema(captured):
    backend = OllamaBackend(model="m")
    backend._info_cache["m"] = {"capabilities": ["tools"], "context_length": 8192}
    tools = [{"type": "function", "function": {"name": "run_command"}}]
    schema = {"type": "object"}
    backend.chat_ex([{"role": "user", "content": "hi"}], tools=tools, schema=schema)
    assert captured["json"]["tools"] == tools
    assert captured["json"]["format"] == schema


def test_think_is_only_sent_to_thinking_models(captured):
    backend = OllamaBackend(model="m", think=True)
    backend._info_cache["m"] = {"capabilities": ["completion"], "context_length": 4096}
    backend.chat_ex([{"role": "user", "content": "hi"}])
    assert "think" not in captured["json"]

    backend._info_cache["m"] = {"capabilities": ["thinking"], "context_length": 4096}
    backend.chat_ex([{"role": "user", "content": "hi"}])
    assert captured["json"]["think"] is True


def test_prepare_passes_tool_messages_through():
    prepared = OllamaBackend._prepare([
        {"role": "assistant", "content": "",
         "tool_calls": [{"name": "run_command", "arguments": {"command": "ls"}}]},
        {"role": "tool", "tool_name": "run_command", "content": "a\nb"},
    ])
    assert prepared[0]["tool_calls"][0]["function"]["name"] == "run_command"
    assert prepared[1] == {"role": "tool", "content": "a\nb", "tool_name": "run_command"}


# ── replies ────────────────────────────────────────────────────────────────
def test_tool_calls_are_parsed(monkeypatch):
    monkeypatch.setattr(ob, "request_with_retry", lambda *a, **k: FakeResponse({
        "message": {"role": "assistant", "content": "", "thinking": "hmm",
                    "tool_calls": [{"id": "c1", "function": {
                        "name": "run_command", "arguments": {"command": "df -h"}}}]},
        "eval_count": 20, "eval_duration": 2_000_000_000,
    }))
    reply = OllamaBackend(model="m").chat_ex([{"role": "user", "content": "x"}])
    assert reply.tool_calls[0].name == "run_command"
    assert reply.tool_calls[0].arguments == {"command": "df -h"}
    assert reply.thinking == "hmm"
    assert reply.tokens_per_second == 10.0


def test_tool_call_arguments_may_arrive_as_a_string(monkeypatch):
    monkeypatch.setattr(ob, "request_with_retry", lambda *a, **k: FakeResponse({
        "message": {"content": "", "tool_calls": [
            {"function": {"name": "list_dir", "arguments": '{"path": "~"}'}}]},
    }))
    reply = OllamaBackend(model="m").chat_ex([{"role": "user", "content": "x"}])
    assert reply.tool_calls[0].arguments == {"path": "~"}


def test_streaming_collects_text_thinking_and_calls(monkeypatch):
    lines = [
        json.dumps({"message": {"thinking": "let me "}}),
        json.dumps({"message": {"thinking": "think"}}),
        json.dumps({"message": {"content": "Hello"}}),
        json.dumps({"message": {"content": " there"}}),
        json.dumps({"message": {"content": "", "tool_calls": [
            {"function": {"name": "notify", "arguments": {"message": "hi"}}}]}}),
        json.dumps({"done": True, "eval_count": 4, "eval_duration": 1_000_000_000,
                    "prompt_eval_count": 30, "prompt_eval_duration": 500_000_000}),
    ]
    monkeypatch.setattr(ob, "request_with_retry",
                        lambda *a, **k: FakeResponse(lines=lines))
    text, thoughts = [], []
    reply = OllamaBackend(model="m").chat_ex(
        [{"role": "user", "content": "x"}], stream=True,
        on_text=text.append, on_thinking=thoughts.append,
    )
    assert "".join(text) == "Hello there"
    assert "".join(thoughts) == "let me think"
    assert reply.tool_calls[0].name == "notify"
    assert reply.metrics["eval_count"] == 4
    # Prefill speed is reported too — it is the first thing to sag when a
    # model spills out of VRAM.
    assert reply.prompt_tokens_per_second == 60.0


# ── errors ─────────────────────────────────────────────────────────────────
def test_missing_model_says_how_to_pull(monkeypatch):
    monkeypatch.setattr(ob, "request_with_retry",
                        lambda *a, **k: FakeResponse(status=404, text="not found"))
    with pytest.raises(LLMError, match="ollama pull"):
        OllamaBackend(model="ghost").chat_ex([{"role": "user", "content": "x"}])


def test_unreachable_server_says_how_to_start_it(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(ob, "request_with_retry", boom)
    with pytest.raises(LLMError, match="systemctl --user start ollama"):
        OllamaBackend(model="m").chat_ex([{"role": "user", "content": "x"}])


def test_context_overflow_error_is_explained(monkeypatch):
    monkeypatch.setattr(ob, "request_with_retry", lambda *a, **k: FakeResponse(
        status=400, text="input length exceeds the available context size"))
    with pytest.raises(LLMError, match="ran out of context"):
        OllamaBackend(model="m").chat_ex([{"role": "user", "content": "x"}])


# ── formatting helpers ─────────────────────────────────────────────────────
def test_short_size():
    assert short_size(5_347_929_166) == "5.0 GB"
    assert short_size(900_000).endswith("KB")
    assert short_size(50_000_000).endswith("MB")


def test_parse_size_hint():
    assert parse_size_hint("8.8B") == 8.8
    assert parse_size_hint("500M") == 0.5
    assert parse_size_hint("") == 0.0


# ── placement and hardware awareness ───────────────────────────────────────
def test_runtime_reports_the_gpu_split(monkeypatch):
    monkeypatch.setattr(ob.requests, "get", lambda *a, **k: FakeResponse({"models": [{
        "name": "m", "size": 8_143_623_398, "size_vram": 4_854_980_279,
        "context_length": 16384, "expires_at": "",
    }]}))
    runtime = OllamaBackend(model="m").runtime()
    assert runtime.loaded
    assert runtime.processor() == "40%/60% CPU/GPU"
    assert runtime.spilled and not runtime.on_gpu


def test_runtime_when_fully_on_the_gpu(monkeypatch):
    monkeypatch.setattr(ob.requests, "get", lambda *a, **k: FakeResponse({"models": [{
        "name": "m", "size": 5_000_000_000, "size_vram": 5_000_000_000,
    }]}))
    runtime = OllamaBackend(model="m").runtime()
    assert runtime.on_gpu and runtime.processor() == "100% GPU"


def test_runtime_when_not_loaded(monkeypatch):
    monkeypatch.setattr(ob.requests, "get", lambda *a, **k: FakeResponse({"models": []}))
    runtime = OllamaBackend(model="m").runtime()
    assert not runtime.loaded and runtime.processor() == ""


def test_expires_in_counts_down():
    from datetime import datetime, timedelta

    soon = (datetime.now().astimezone() + timedelta(minutes=5)).isoformat()
    runtime = ob.ModelRuntime(loaded=True, expires_at=soon)
    assert 250 < runtime.expires_in() < 310


def test_usable_vram_excludes_other_applications(monkeypatch):
    from maze_ai.llm.hardware import GPU, MB

    # 8 GB card, 6 GB in use, of which 4 GB is our model → 2 GB is somebody else.
    monkeypatch.setattr(ob, "detect_gpus", lambda: [GPU("card", 8192, 6144)])
    monkeypatch.setattr(ob, "total_vram_bytes", lambda: 8192 * MB)
    monkeypatch.setattr(ob.requests, "get", lambda *a, **k: FakeResponse(
        {"models": [{"name": "m", "size_vram": 4096 * MB}]}))
    usable, other = OllamaBackend(model="m").usable_vram()
    assert other == 2048 * MB
    assert usable == 6144 * MB


def test_auto_context_shrinks_to_keep_the_model_on_the_gpu():
    from maze_ai.llm.hardware import GB

    # 7 GB of weights in an 8 GB card: a 32k window would spill.
    chosen = ob.auto_context(262144, ram_gb=64, weights_bytes=7 * GB, vram_bytes=8 * GB)
    assert chosen < 32768


def test_auto_context_ignores_vram_when_the_model_cannot_fit_anyway():
    from maze_ai.llm.hardware import GB

    # 20 GB of weights will run from RAM; a tiny window would help nobody.
    chosen = ob.auto_context(131072, ram_gb=64, weights_bytes=20 * GB, vram_bytes=8 * GB)
    assert chosen == 32768


def test_benchmark_reports_the_numbers_that_matter(monkeypatch):
    monkeypatch.setattr(ob, "request_with_retry", lambda *a, **k: FakeResponse({
        "message": {"content": "one, two"},
        "eval_count": 50, "eval_duration": 5_000_000_000,
        "prompt_eval_count": 20, "prompt_eval_duration": 500_000_000,
        "load_duration": 3_000_000_000,
    }))
    monkeypatch.setattr(ob.requests, "get", lambda *a, **k: FakeResponse({"models": [
        {"name": "m", "size": 1000, "size_vram": 1000}]}))
    backend = OllamaBackend(model="m", num_ctx=8192)
    result = backend.benchmark(tokens=50)
    assert result["generate_tps"] == 10.0
    assert result["prefill_tps"] == 40.0
    assert result["load_seconds"] == 3.0
    assert result["processor"] == "100% GPU"
    assert not result["spilled"]
