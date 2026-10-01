"""Tests for personality presets, creativity and the emoji → emoticon filter."""

from __future__ import annotations

import json

import pytest

from maze_ai.agent.agent import Agent
from maze_ai.agent.prompts import build_system_prompt
from maze_ai.config import MODE_AUTO, MODE_CHAT, Config
from maze_ai.llm import build_backend
from maze_ai.llm.base import LLMBackend
from maze_ai.style import (
    CREATIVITY,
    PERSONAS,
    EmojiFilter,
    persona_block,
    strip_emoji,
    temperature_for,
)


# ── emoji filter ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("Done! 😀", "Done! :D"),
    ("Hmm 😕 that failed", "Hmm :/ that failed"),
    ("Just kidding 😛", "Just kidding :P"),
    ("Sorry 😢", "Sorry :'("),
    ("I ❤️ Arch", "I <3 Arch"),
    ("nice😉", "nice ;)"),
])
def test_faces_become_emoticons(text, expected):
    assert strip_emoji(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("## 🚀 Features", "## Features"),
    ("- ✅ backup done", "- backup done"),
    ("✨ Ready", "Ready"),
    ("⚠️ Careful", "Careful"),
    ("Turkey 🇹🇷 flag", "Turkey flag"),
    ("dev 👨‍💻 mode", "dev mode"),
    ("wave 👋🏽 hi", "wave o/ hi"),
    ("1️⃣ first", "1 first"),
    ("**⚠️ Careful**", "**Careful**"),
    ("_✨ new_", "_new_"),
])
def test_other_pictographs_are_removed_cleanly(text, expected):
    assert strip_emoji(text) == expected


@pytest.mark.parametrize("text", [
    "Paths → /etc/pacman.conf",
    "Box ┌─┐ drawing │ stays",
    "Türkçe karakterler: ğüşıöç İ",
    "Check ✓ and ✗ and ⚠ without emoji style",
    "Math: x ≤ 5, π ≈ 3.14, ∑",
    "def f(): return {'a': 1}  # code",
    "中文 日本語 русский العربية",
    "Copyright © 2026",
])
def test_ordinary_text_is_untouched(text):
    assert strip_emoji(text) == text


def test_streaming_matches_the_whole_string():
    text = "Selam 😀! ⚠️ Dikkat: ## 🚀 Başlık\n- ✅ tamam 👨‍💻 ok → bitti ❤️"
    whole = strip_emoji(text)
    for size in (1, 2, 3, 5):
        flt = EmojiFilter()
        out = "".join(flt.feed(text[i:i + size]) for i in range(0, len(text), size))
        out += flt.flush()
        assert out == whole, size


def test_a_symbol_split_from_its_variation_selector_is_still_caught():
    flt = EmojiFilter()
    out = flt.feed("Warning ⚠") + flt.feed("️ now") + flt.flush()
    assert out == "Warning now"


# ── personality ────────────────────────────────────────────────────────────
def test_every_persona_has_a_label_description_and_prompt():
    for pid, (label, desc, prompt) in PERSONAS.items():
        assert label and desc and prompt, pid


def test_persona_block_asks_for_emoticons_not_emoji():
    block = persona_block("friendly", no_emoji=True)
    assert "Never use emoji" in block
    assert ":)" in block and ":P" in block and ":/" in block


def test_professional_persona_uses_no_emoticons_either():
    assert "Never use emoji or emoticons" in persona_block("professional", True)


def test_unknown_persona_falls_back_to_balanced():
    assert persona_block("nonsense") == persona_block("balanced")


def test_persona_lands_in_every_prompt_variant():
    concise = PERSONAS["concise"][2]
    for kwargs in ({"enable_tools": False}, {"enable_tools": True},
                   {"enable_tools": True, "native_tools": True}):
        prompt = build_system_prompt(persona="concise", **kwargs)
        assert concise in prompt


def test_compact_prompt_drops_the_easter_egg():
    full = build_system_prompt(native_tools=True)
    small = build_system_prompt(native_tools=True, compact=True)
    assert "Pop-culture" in full and "Pop-culture" not in small
    assert len(small) < len(full)


def test_creativity_maps_to_temperature():
    assert temperature_for("precise") < temperature_for("balanced") < temperature_for("creative")
    assert temperature_for(None) == CREATIVITY["balanced"][1]


@pytest.mark.parametrize("backend", ["ollama", "gemini", "openai"])
def test_creativity_reaches_every_backend(backend, monkeypatch):
    config = Config.__new__(Config)
    from maze_ai.config import DEFAULTS
    config._data = dict(DEFAULTS, backend=backend, creativity="creative")
    assert build_backend(config).temperature == temperature_for("creative")


# ── the agent applies it ───────────────────────────────────────────────────
class _Scripted(LLMBackend):
    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.systems: list[str] = []

    def chat(self, messages):
        self.systems.append(messages[0]["content"])
        return self.replies.pop(0)

    def chat_stream(self, messages):
        self.systems.append(messages[0]["content"])
        reply = self.replies.pop(0)
        for i in range(0, len(reply), 3):
            yield reply[i:i + 3]

    def available_models(self):
        return []


def _run(agent: Agent, text: str = "selam"):
    events = []
    answer = agent.run(text, events.append, lambda _req: True)
    return answer, events


def test_agent_strips_emoji_from_answer_stream_and_history():
    agent = Agent(_Scripted(["Selam! 😀 🚀 Hazırım"]), mode=MODE_CHAT,
                  stream_responses=True)
    answer, events = _run(agent)
    streamed = "".join(e.text for e in events if e.kind == "stream")
    final = next(e.text for e in events if e.kind == "final")
    assert final == "Selam! :D Hazırım"
    assert streamed == final
    assert agent.history[-1]["content"] == final
    assert "😀" not in answer


def test_agent_strips_emoji_from_protocol_answers():
    reply = json.dumps({"thought": "t", "action": "final_answer",
                        "action_input": {"answer": "Tamam ✅ bitti 😉"}})
    agent = Agent(_Scripted([reply]), mode=MODE_AUTO, stream_responses=False)
    answer, _ = _run(agent)
    assert answer == "Tamam bitti ;)"
    assert agent.history[-1]["content"] == "Tamam bitti ;)"


def test_emoji_can_be_allowed():
    agent = Agent(_Scripted(["Selam 😀"]), mode=MODE_CHAT, stream_responses=False,
                  no_emoji=False)
    answer, _ = _run(agent)
    assert answer == "Selam 😀"


def test_agent_sends_the_chosen_persona():
    backend = _Scripted(["ok"])
    agent = Agent(backend, mode=MODE_CHAT, stream_responses=False, persona="hacker")
    _run(agent)
    assert PERSONAS["hacker"][2] in backend.systems[0]


def test_apply_config_reads_personality(monkeypatch):
    from maze_ai.config import DEFAULTS

    config = Config.__new__(Config)
    config._data = dict(DEFAULTS, persona="detailed", no_emoji=False, agent_mode=MODE_CHAT)
    agent = Agent(_Scripted([]))
    agent.apply_config(config, rebuild_backend=False)
    assert agent.persona == "detailed"
    assert agent.no_emoji is False
    assert agent.mode == MODE_CHAT
