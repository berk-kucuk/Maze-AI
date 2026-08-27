"""Tests for system-prompt composition and language detection."""

from __future__ import annotations

import pytest

from maze_ai.agent.prompts import build_system_prompt, guess_language


@pytest.mark.parametrize("text,expected", [
    ("Kaç dosya var ana dizinimde?", "tr"),
    ("merhaba nasilsin", "tr"),            # Turkish without the special letters
    ("bana bir dosya olustur", "tr"),
    ("how many files are here", "en"),
    ("hello, show my disk usage", "en"),
    ("Wie viele Dateien gibt es?", "de"),
    ("bonjour, combien de fichiers", "fr"),
    ("hola, cuántos archivos hay", "es"),
    ("Привет, как дела", "ru"),
    ("こんにちは", "ja"),
    ("你好", "zh"),
])
def test_guess_language(text, expected):
    assert guess_language(text) == expected


@pytest.mark.parametrize("text", ["df -h", "ls", "", "42", "~/Projects"])
def test_guess_language_stays_silent_when_unsure(text):
    # Guessing from a shell command would pin the answer to the wrong language.
    assert guess_language(text) == ""


def test_auto_language_becomes_a_concrete_instruction():
    prompt = build_system_prompt(True, "auto", user_message="Kaç dosya var?")
    assert "in Turkish" in prompt


def test_auto_language_falls_back_to_the_generic_rule():
    prompt = build_system_prompt(True, "auto", user_message="df -h")
    assert "SAME language" in prompt


def test_explicit_language_overrides_detection():
    prompt = build_system_prompt(True, "en", user_message="Kaç dosya var?")
    assert "in English" in prompt


# ── shape of the prompt ────────────────────────────────────────────────────
def test_native_prompt_drops_the_catalogue():
    native = build_system_prompt(True, "en", native_tools=True)
    protocol = build_system_prompt(True, "en", native_tools=False)
    assert "action_input" not in native      # no JSON contract
    assert "arguments:" not in native        # no tool catalogue
    assert "action_input" in protocol
    # The whole point: it has to be much smaller on a small context window.
    assert len(native) < len(protocol) / 2


def test_both_prompts_carry_the_injection_rule():
    for native in (True, False):
        prompt = build_system_prompt(True, "en", native_tools=native)
        assert "TOOL_OUTPUT" in prompt
        assert "never instructions" in prompt.lower() or "not orders" in prompt.lower()


def test_catalogue_lists_only_the_enabled_tools():
    prompt = build_system_prompt(True, "en", tool_names=["run_command", "list_dir"])
    assert "run_command" in prompt and "list_dir" in prompt
    assert "add_reminder" not in prompt


def test_chat_mode_has_no_tools_at_all():
    prompt = build_system_prompt(False, "en")
    assert "CHAT-ONLY" in prompt
    assert "run_command" not in prompt


def test_working_directory_is_stated_when_tools_are_on():
    assert "/home/x/dev" in build_system_prompt(True, "en", cwd="/home/x/dev")
    assert "/home/x/dev" not in build_system_prompt(False, "en", cwd="/home/x/dev")


# ── prompt stability (prefix caching) ──────────────────────────────────────
def test_agent_prompt_is_stable_within_a_day():
    # Ollama caches the prompt prefix between turns; a minute-resolution
    # timestamp would invalidate it on every message.
    first = build_system_prompt(True, "en", cwd="/home/x")
    second = build_system_prompt(True, "en", cwd="/home/x")
    assert first == second
    assert ":" not in first.split("- Today:")[1].split("\n")[0].split("(")[0]


def test_agent_prompt_tells_the_model_how_to_read_the_clock():
    assert "run `date`" in build_system_prompt(True, "en")


def test_chat_mode_keeps_the_exact_time():
    # No tools to read the clock with, and no tool loop to cache for.
    prompt = build_system_prompt(False, "en")
    assert "- Today: " in prompt


# ── Turkish typed without diacritics ───────────────────────────────────────
@pytest.mark.parametrize("text", [
    "konsole uygulamamda neler var gorebiliyor musun",
    "en son konsole uygulamasinda ne yaptim",
    "ekranimda ne goruyorsun",
    "bu dosyayi ac ve icindekileri goster",
    "yarin 9da toplanti hatirlat",
    "disk kullanimimi gosterir misin",
    "simdi ne yapiyorsun",
    "bunu nasil duzeltebilirim",
])
def test_turkish_without_diacritics_is_detected(text):
    # Nobody types ğ/ş/ı in a hurry; the suffixes give it away instead.
    assert guess_language(text) == "tr"


@pytest.mark.parametrize("text", [
    "the ruler is similar to the other one",     # -lar / -ler look Turkish
    "this is a filler sentence about a scholar",
    "summarise this file for me",
    "how many files are here",
])
def test_english_is_not_mistaken_for_turkish(text):
    assert guess_language(text) != "tr"
