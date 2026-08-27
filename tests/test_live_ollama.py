"""End-to-end tests against a real local Ollama server.

Skipped unless ``MAZE_AI_LIVE=1`` is set, because they need a running server
with a pulled model and take real seconds. Run them after changing anything in
the Ollama path, or to check whether a newly pulled model behaves as an agent::

    MAZE_AI_LIVE=1 MAZE_AI_LIVE_MODEL=qwen2.5 pytest tests/test_live_ollama.py -v
"""

from __future__ import annotations

import os

import pytest

from maze_ai.agent.agent import Agent, AgentEvent
from maze_ai.config import MODE_AUTO
from maze_ai.llm.ollama_backend import OllamaBackend

pytestmark = pytest.mark.skipif(
    os.environ.get("MAZE_AI_LIVE") != "1",
    reason="set MAZE_AI_LIVE=1 to run against a real Ollama server",
)


@pytest.fixture(scope="module")
def backend() -> OllamaBackend:
    server = OllamaBackend()
    if not server.is_reachable():
        pytest.skip("no Ollama server on this machine")
    models = server.available_models()
    if not models:
        pytest.skip("no models installed")
    wanted = os.environ.get("MAZE_AI_LIVE_MODEL")
    model = next((m for m in models if wanted and m.startswith(wanted)), models[0])
    return OllamaBackend(model=model, keep_alive="5m")


def test_server_reports_capabilities(backend):
    info = backend.model_info()
    assert info, "the server should describe the active model"
    assert "capabilities" in info


def test_context_is_sized_for_the_machine(backend):
    assert 2048 <= backend.resolved_ctx() <= 32768


def test_a_real_turn_uses_a_tool(backend):
    agent = Agent(backend, mode=MODE_AUTO, max_steps=4,
                  tool_groups=["shell"], stream_responses=True)
    events: list[AgentEvent] = []
    answer = agent.run(
        "Run a shell command to print exactly the word MAZE, then tell me what it printed.",
        events.append, lambda request: True,
    )
    kinds = [e.kind for e in events]
    assert "tool_call" in kinds, f"the model never called a tool: {answer!r}"
    assert "MAZE" in answer.upper()
    assert [m["role"] for m in agent.history] == ["user", "assistant"]


def test_generation_metrics_are_reported(backend):
    reply = backend.chat_ex([{"role": "user", "content": "Say OK."}])
    assert reply.tokens_per_second > 0
