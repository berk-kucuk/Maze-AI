"""On a hosted model, a sensitive read says where its contents go (audit MA4).

Whatever a tool returns is part of the next request to the model. On a local
model that stays on the machine; on Gemini or a remote OpenAI-compatible or
Ollama server it does not — and the approval for reading ~/.ssh/id_rsa said
only that the path was sensitive.
"""
from __future__ import annotations

import pytest

from maze_ai.agent.agent import (
    REASON_SCREEN, REASON_SCREEN_CLOUD, REASON_SENSITIVE, REASON_SENSITIVE_CLOUD,
    UNSKIPPABLE_REASONS, Agent,
)
from maze_ai.config import MODE_ASK, MODE_AUTO
from maze_ai.llm.base import remote_provider
from maze_ai.llm.gemini_backend import GeminiBackend
from maze_ai.llm.ollama_backend import OllamaBackend
from maze_ai.llm.openai_backend import OpenAIBackend


def _openai(url):
    return OpenAIBackend(base_url=url, api_key="k", model="m")


@pytest.mark.parametrize("backend, expected", [
    (lambda: _openai("https://api.openai.com/v1"), "api.openai.com"),
    (lambda: _openai("https://openrouter.ai/api/v1"), "openrouter.ai"),
    (lambda: _openai("http://localhost:1234/v1"), ""),
    (lambda: _openai("http://127.0.0.1:8080/v1"), ""),
    (lambda: _openai("http://[::1]:8080/v1"), ""),
    (lambda: OllamaBackend(host="http://127.0.0.1:11434"), ""),
    (lambda: OllamaBackend(host="http://gpu-box.lan:11434"), "gpu-box.lan"),
    (lambda: GeminiBackend(api_key="k"), "Google (Gemini)"),
])
def test_remote_provider(backend, expected):
    assert remote_provider(backend()) == expected


@pytest.mark.parametrize("mode", [MODE_ASK, MODE_AUTO])
def test_sensitive_read_names_the_provider(mode, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    agent = Agent(backend=_openai("https://api.openai.com/v1"), mode=mode)
    reason, detail = agent._approval_reason("read_file", {"path": "~/.ssh/id_rsa"})
    assert reason == REASON_SENSITIVE_CLOUD
    assert detail.endswith("→ api.openai.com")
    assert reason in UNSKIPPABLE_REASONS, "never 'always allow'"


def test_local_model_keeps_the_plain_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    agent = Agent(backend=OllamaBackend(), mode=MODE_ASK)
    reason, _ = agent._approval_reason("read_file", {"path": "~/.ssh/id_rsa"})
    assert reason == REASON_SENSITIVE


def test_screen_capture_names_the_provider():
    agent = Agent(backend=GeminiBackend(api_key="k"), mode=MODE_ASK)
    tool = next(iter(__import__("maze_ai.agent.agent", fromlist=["x"]).SCREEN_TOOLS))
    assert agent._approval_reason(tool, {}) == (REASON_SCREEN_CLOUD, "Google (Gemini)")
    agent.backend = OllamaBackend()
    assert agent._approval_reason(tool, {})[0] == REASON_SCREEN
