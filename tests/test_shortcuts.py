"""Keyboard shortcuts in the chat window — they must work from the message box.

The message box is where focus lives, and a QPlainTextEdit claims many key
combinations for itself (Ctrl+K deletes to the end of the line on Linux,
Ctrl+Backspace a word, …). Every binding here is sent *to the composer*, the
way a user actually presses it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from maze_ai.config import Config

CTRL = Qt.KeyboardModifier.ControlModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier
ALT = Qt.KeyboardModifier.AltModifier


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    from maze_ai import config as config_module
    from maze_ai import history

    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "config" / "c.json")
    monkeypatch.setattr(history, "CHATS_DIR", tmp_path / "chats")
    from maze_ai.ui.main_window import MainWindow

    win = MainWindow(Config())
    win.show()
    win.raise_()
    win.activateWindow()
    assert QTest.qWaitForWindowActive(win), "window shortcuts need an active window"
    win.input_bar.focus_input()
    QTest.qWait(30)
    yield win
    win._really_quit = True
    win.close()
    win.deleteLater()
    QTest.qWait(10)


def press(window, key, mods=Qt.KeyboardModifier.NoModifier):
    QTest.keyClick(window.input_bar.composer, key, mods)
    QTest.qWait(20)


def saved_chat(window, text="hello"):
    from maze_ai.history import Conversation

    conv = Conversation(messages=[{"role": "user", "content": text},
                                  {"role": "assistant", "content": "hi"}])
    window.store.save(conv)
    window._refresh_sidebar()
    return conv


def test_every_listed_shortcut_is_bound(window):
    bound = {s.key().toString() for s in window.shortcuts}
    listed = {
        QKeySequence(alt).toString()
        for _, items in window.shortcut_table()[:3]
        for _, keys in items
        for alt in keys.split(" / ")
    }
    # Keys a widget handles itself (typing, paging, recall) are documented
    # but not window shortcuts.
    handled_by_widgets = {"Return", "Shift+Return", "Up", "PgUp", "PgDown"}
    assert listed - handled_by_widgets <= bound


def test_no_shortcut_is_bound_twice(window):
    keys = [s.key().toString() for s in window.shortcuts]
    assert len(keys) == len(set(keys))


def test_ctrl_b_toggles_the_sidebar(window):
    before = window.sidebar.isVisible()
    press(window, Qt.Key.Key_B, CTRL)
    assert window.sidebar.isVisible() is not before
    press(window, Qt.Key.Key_B, CTRL)
    assert window.sidebar.isVisible() is before


def test_ctrl_n_starts_a_new_chat(window):
    conv = saved_chat(window)
    window.load_chat(conv.id)
    press(window, Qt.Key.Key_N, CTRL)
    assert window.conversation.id != conv.id and window.conversation.is_empty


def test_ctrl_f_focuses_search(window):
    press(window, Qt.Key.Key_F, CTRL)
    assert window.sidebar.search.hasFocus()


def test_esc_in_search_clears_then_returns_to_the_composer(window):
    window.focus_search()
    QTest.keyClicks(window.sidebar.search, "abc")
    QTest.keyClick(window.sidebar.search, Qt.Key.Key_Escape)
    assert window.sidebar.search.text() == ""
    QTest.keyClick(window.sidebar.search, Qt.Key.Key_Escape)
    QTest.qWait(20)
    assert window.input_bar.composer.hasFocus()


def test_alt_down_and_up_walk_the_history(window):
    first = saved_chat(window, "first")
    second = saved_chat(window, "second")
    window.load_chat(second.id)
    press(window, Qt.Key.Key_Down, ALT)
    assert window.conversation.id == first.id
    press(window, Qt.Key.Key_Up, ALT)
    assert window.conversation.id == second.id


def test_delete_shortcut_asks_first(window, monkeypatch):
    from maze_ai.ui import main_window as mw

    conv = saved_chat(window)
    window.load_chat(conv.id)
    monkeypatch.setattr(mw.ConfirmDialog, "exec", lambda self: 0)   # user says no
    press(window, Qt.Key.Key_Backspace, CTRL | SHIFT)
    assert window.store.load(conv.id) is not None
    monkeypatch.setattr(mw.ConfirmDialog, "exec", lambda self: 1)   # user says yes
    press(window, Qt.Key.Key_Backspace, CTRL | SHIFT)
    assert window.store.load(conv.id) is None


def test_up_in_an_empty_box_recalls_the_last_message(window):
    window._last_user_message = "list my files"
    press(window, Qt.Key.Key_Up)
    assert window.input_bar.composer.toPlainText() == "list my files"


def test_up_does_nothing_while_typing(window):
    window._last_user_message = "old"
    QTest.keyClicks(window.input_bar.composer, "new")
    press(window, Qt.Key.Key_Up)
    assert window.input_bar.composer.toPlainText() == "new"


def test_ctrl_shift_c_copies_the_last_answer(window):
    window.chat.add_ai("the answer")
    press(window, Qt.Key.Key_C, CTRL | SHIFT)
    assert QApplication.clipboard().text() == "the answer"


def test_shortcuts_dialog_opens(window, monkeypatch):
    from maze_ai.ui import main_window as mw

    opened = []
    monkeypatch.setattr(mw.ShortcutsDialog, "exec", lambda self: opened.append(self) or 1)
    press(window, Qt.Key.Key_Slash, CTRL)
    press(window, Qt.Key.Key_F1)
    assert len(opened) == 2


def test_empty_state_suggestions_send(window, monkeypatch):
    sent = []
    monkeypatch.setattr(window, "on_send", lambda text: sent.append(text))
    window.chat.suggestion_clicked.disconnect()
    window.chat.suggestion_clicked.connect(window.on_send)
    window.new_chat()
    assert window.chat.empty_state is not None
    window.chat.empty_state.cards[0].click()
    assert sent and sent[0]
