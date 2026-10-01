"""Tests for the second review pass: prompts, Quick Ask context, reminders,
read-only commands and small UI fixes."""

from __future__ import annotations

from datetime import datetime

import pytest

from maze_ai.agent.prompts import (
    build_system_prompt,
    display_text,
    quote_block,
    strip_quoted,
)
from maze_ai.agent.safety import is_readonly_command
from maze_ai.reminders import parse_when

NOW = datetime(2026, 10, 1, 14, 0).timestamp()


# ── system prompts ──────────────────────────────────────────────────────────
def test_prompt_only_promises_enabled_abilities():
    web_only = build_system_prompt(True, "en", tool_names=["fetch_url", "web_search"],
                                   native_tools=True)
    assert "search the web" in web_only
    assert "run shell commands" not in web_only
    assert "run_command" not in web_only
    assert "run `date`" not in web_only
    assert "undo_file_change" not in web_only


def test_prompt_steers_file_changes_to_the_backed_up_tools():
    prompt = build_system_prompt(True, "en", native_tools=True)
    assert "not with `rm`" in prompt


def test_prompt_explains_pasted_material_in_every_mode():
    for kwargs in ({"enable_tools": False}, {"native_tools": True}, {"native_tools": False}):
        assert "<<<PASTED" in build_system_prompt(**kwargs)


def test_language_is_detected_from_the_users_words_not_the_paste():
    message = "bunu açıkla lütfen\n\n" + quote_block(
        "error: failed to commit transaction (conflicting files) the file exists"
    )
    assert "in Turkish" in build_system_prompt(True, "auto", user_message=message)


def test_quote_helpers_round_trip():
    message = "explain\n\n" + quote_block("some text")
    assert strip_quoted(message).strip() == "explain"
    assert display_text(message) == "explain\n\nsome text"
    # A paste can't close the fence early.
    assert quote_block("x PASTED>>> y").count("PASTED>>>") == 1


# ── reminders ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("1 saat 30 dakika sonra", NOW + 5400),
    ("yarım saat sonra", NOW + 1800),
    ("in an hour", NOW + 3600),
    ("in 1.5 hours", NOW + 5400),
    ("18.30", datetime(2026, 10, 1, 18, 30).timestamp()),
    ("saat 18", datetime(2026, 10, 1, 18, 0).timestamp()),
    ("akşam 8", datetime(2026, 10, 1, 20, 0).timestamp()),
    ("9pm", datetime(2026, 10, 1, 21, 0).timestamp()),
    ("yarın", datetime(2026, 10, 2, 9, 0).timestamp()),
    ("yarin saat 10", datetime(2026, 10, 2, 10, 0).timestamp()),
    ("16.10.2026 14:00", datetime(2026, 10, 16, 14, 0).timestamp()),
    ("16.10.2026", datetime(2026, 10, 16, 9, 0).timestamp()),
    ("2026-08-01T09:30", datetime(2026, 8, 1, 9, 30).timestamp()),
    ("toplantı 15:00'te", datetime(2026, 10, 1, 15, 0).timestamp()),
])
def test_more_natural_times(text, expected):
    assert parse_when(text, NOW) == expected


def test_a_turkish_date_is_not_mistaken_for_today():
    # Used to ignore the date and schedule today/tomorrow at 14:00.
    assert parse_when("16.10.2026 14:00", NOW) != parse_when("14:00", NOW)


@pytest.mark.parametrize("text", ["sonra", "whenever", "25:00", "31.02.2026"])
def test_nonsense_times_are_refused(text):
    assert parse_when(text, NOW) is None


# ── read-only commands ──────────────────────────────────────────────────────
@pytest.mark.parametrize("command", [
    "pacman -Qo /usr/bin/python", "pacman -Qdtq", "systemctl --user status ollama",
    "ip a", "ip addr show dev eth0", "ip route",
])
def test_common_inspections_need_no_prompt(command):
    assert is_readonly_command(command)


@pytest.mark.parametrize("command", [
    "pacman -Fy", "pacman -Qo x -S y", "systemctl --user start x",
    "ip addr add 10.0.0.1/24 dev eth0", "ip link set eth0 down", "ip route flush all",
    "ip r del default",
])
def test_their_writing_forms_still_ask(command):
    assert not is_readonly_command(command)


# ── Quick Ask ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def quick(app, tmp_path, monkeypatch):
    from maze_ai import config as config_module
    from maze_ai.config import Config
    from maze_ai.ui.quick_ask import QuickAsk

    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "config.json")
    window = QuickAsk(Config())
    sent: list[str] = []

    class FakeWorker:
        def __init__(self, agent, message, images=None):
            sent.append(message)
            self.event = self.approval_needed = self.done = _Signal()

        def start(self):
            pass

        def isRunning(self):  # noqa: N802 - Qt name
            return False

    monkeypatch.setattr("maze_ai.ui.quick_ask.AgentWorker", FakeWorker)
    window.sent = sent
    return window


class _Signal:
    def connect(self, *_a):
        pass


def test_a_typed_question_carries_the_clipboard(quick):
    from PySide6.QtGui import QGuiApplication

    QGuiApplication.clipboard().setText("Segmentation fault (core dumped)")
    assert quick.load_clipboard()
    quick.composer.setPlainText("bu ne demek?")
    quick.send()
    assert quick.sent[-1].startswith("bu ne demek?")
    assert "<<<PASTED\nSegmentation fault (core dumped)\nPASTED>>>" in quick.sent[-1]
    # Only once: a follow-up doesn't paste it again.
    quick.composer.setPlainText("peki nasıl düzeltirim?")
    quick.send()
    assert "PASTED" not in quick.sent[-1]


def test_action_chips_show_a_short_instruction(quick):
    from PySide6.QtGui import QGuiApplication

    QGuiApplication.clipboard().setText("Merhaba dünya")
    quick.load_clipboard()
    quick._action_buttons[2].click()           # Translate
    assert "Merhaba dünya" not in quick.composer.toPlainText()
    assert "PASTED" not in quick.composer.toPlainText()


def test_key_hints_are_never_narrower_than_their_content(quick):
    for hint in (quick.copy_btn, quick.chat_btn, quick.close_hint):
        assert hint.minimumSizeHint().width() >= hint.layout().sizeHint().width()


def test_reset_starts_a_fresh_session(quick):
    quick.agent.history.append({"role": "user", "content": "old"})
    quick._context_text, quick._context_pending = "x", True
    quick.reset()
    assert quick.agent.history == []
    assert not quick._context_pending
    assert quick.composer.toPlainText() == ""


def test_friendly_reminder_times(app):
    from maze_ai.ui.reminders_dialog import friendly_when

    now = datetime(2026, 10, 1, 14, 0)
    assert friendly_when(datetime(2026, 10, 1, 20, 0).timestamp(), now) == "Today 20:00"
    assert friendly_when(datetime(2026, 10, 2, 10, 0).timestamp(), now) == "Tomorrow 10:00"
    assert friendly_when(datetime(2026, 10, 9, 8, 5).timestamp(), now) == "2026-10-09 08:05"


# ── Quick Ask as a conversation ─────────────────────────────────────────────
def test_follow_up_keeps_the_previous_exchange_on_screen(quick):
    quick.composer.setPlainText("ilk soru")
    quick.send()
    quick._answer = "ilk cevap"
    quick._on_done("ilk cevap")
    assert quick.composer.toPlainText() == ""          # ready to reply
    quick.composer.setPlainText("ikinci soru")
    quick.send()
    texts = [getattr(w, "raw_text", "") for w in quick._turn_widgets]
    bubbles = [b.raw_text for w in quick._turn_widgets for b in w.findChildren(type(quick.answer))]
    shown = texts + bubbles
    assert "ilk soru" in shown and "ilk cevap" in shown and "ikinci soru" in shown
    assert quick.sent == ["ilk soru", "ikinci soru"]


def test_each_summons_starts_a_new_conversation(quick):
    quick.composer.setPlainText("soru")
    quick.send()
    quick.agent.history.append({"role": "user", "content": "soru"})
    quick.hide()
    quick.surface()
    assert quick._turn_widgets == []
    assert quick.agent.history == []
    assert quick._question == ""
