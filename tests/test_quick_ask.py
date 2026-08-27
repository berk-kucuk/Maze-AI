"""Tests for Quick Ask and the composer's drop/paste handling."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, QUrl
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtWidgets import QApplication

from maze_ai.config import Config
from maze_ai.ui.input_bar import InputBar
from maze_ai.ui.quick_ask import CLIPBOARD_ACTIONS, QuickAsk


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def config(tmp_path, monkeypatch):
    from maze_ai import config as config_module

    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "config.json")
    return Config()


# ── quick ask ──────────────────────────────────────────────────────────────
def test_prefill_puts_the_question_in_the_box(app, config):
    window = QuickAsk(config)
    window.prefill("what is a symlink")
    assert window.composer.toPlainText() == "what is a symlink"


def test_clipboard_offers_actions(app, config):
    QGuiApplication.clipboard().setText("sudo pacman -Syu --noconfrm")
    window = QuickAsk(config)
    assert window.load_clipboard()
    labels = [
        window.actions_row.itemAt(i).widget().text()
        for i in range(window.actions_row.count())
        if window.actions_row.itemAt(i).widget()
    ]
    assert len(labels) == len(CLIPBOARD_ACTIONS)
    assert "noconfrm" in window.context_label.text()


def test_empty_clipboard_says_so(app, config):
    QGuiApplication.clipboard().setText("")
    window = QuickAsk(config)
    assert not window.load_clipboard()
    assert window.status.text()


def test_attaching_an_image_sets_a_question(app, config, tmp_path):
    shot = tmp_path / "shot.png"
    QImage(4, 4, QImage.Format.Format_RGB32).save(str(shot))
    window = QuickAsk(config)
    window.attach_image(str(shot))
    assert window._images == [str(shot)]
    assert window.composer.toPlainText()
    assert "shot.png" in window.context_label.text()


def test_an_empty_question_is_not_sent(app, config):
    window = QuickAsk(config)
    window.composer.setPlainText("   ")
    window.send()
    assert window.worker is None


def test_continue_in_chat_emits_the_exchange(app, config):
    window = QuickAsk(config)
    window._question = "q"
    window._answer = "a"
    seen = []
    window.open_in_chat.connect(lambda q, a: seen.append((q, a)))
    window._to_chat()
    assert seen == [("q", "a")]


# ── dropping files on the composer ─────────────────────────────────────────
def drop(bar: InputBar, paths: list[str]):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
    bar.add_files_or_images([url.toLocalFile() for url in mime.urls()])


def test_dropping_a_document_puts_its_path_in_the_message(app, tmp_path):
    bar = InputBar()
    doc = tmp_path / "notes.txt"
    doc.write_text("hi")
    drop(bar, [str(doc)])
    assert str(doc) in bar.composer.toPlainText()
    assert bar.take_attachments() == []


def test_dropping_an_image_attaches_it_when_the_model_can_see(app, tmp_path):
    bar = InputBar()
    bar.set_vision(True)
    shot = tmp_path / "shot.png"
    QImage(4, 4, QImage.Format.Format_RGB32).save(str(shot))
    drop(bar, [str(shot)])
    assert bar.take_attachments() == [str(shot)]


def test_dropping_an_image_without_vision_falls_back_to_the_path(app, tmp_path):
    # A text-only model can still OCR it — but only if it knows the path.
    bar = InputBar()
    bar.set_vision(False)
    shot = tmp_path / "shot.png"
    QImage(4, 4, QImage.Format.Format_RGB32).save(str(shot))
    drop(bar, [str(shot)])
    assert str(shot) in bar.composer.toPlainText()


def test_paths_with_spaces_are_quoted(app, tmp_path):
    bar = InputBar()
    doc = tmp_path / "my notes.txt"
    doc.write_text("hi")
    drop(bar, [str(doc)])
    assert f'"{doc}"' in bar.composer.toPlainText()


def test_pasted_image_is_saved_to_a_file(app, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    image = QImage(8, 8, QImage.Format.Format_RGB32)
    image.fill(0)
    QGuiApplication.clipboard().setImage(image)
    saved = InputBar().composer._save_pasted_image()
    assert saved and saved.endswith(".png")
    assert (tmp_path / "maze-ai" / "pasted").exists()


def test_nothing_is_saved_when_the_clipboard_has_no_image(app):
    QGuiApplication.clipboard().setText("just text")
    assert InputBar().composer._save_pasted_image() == ""


# ── files handed over by the file manager ──────────────────────────────────
def test_a_text_file_is_summarised(app, config, tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("# notes")
    window = QuickAsk(config)
    window.load_files([str(doc)])
    question = window.composer.toPlainText()
    assert str(doc) in question and "summarise" in question.lower()
    assert "notes.md" in window.context_label.text()


def test_a_folder_asks_what_is_inside(app, config, tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    window = QuickAsk(config)
    window.load_files([str(folder)])
    assert "folder" in window.composer.toPlainText().lower()


def test_an_unknown_binary_file_asks_what_it_is(app, config, tmp_path):
    blob = tmp_path / "thing.bin"
    blob.write_bytes(b"\x00\x01")
    window = QuickAsk(config)
    window.load_files([str(blob)])
    assert "what this file is" in window.composer.toPlainText().lower()


def test_several_files_are_listed(app, config, tmp_path):
    made = []
    for name in ("a.txt", "b.txt", "c.txt", "d.txt"):
        path = tmp_path / name
        path.write_text("x")
        made.append(str(path))
    window = QuickAsk(config)
    window.load_files(made)
    question = window.composer.toPlainText()
    assert all(path in question for path in made)
    assert "+1" in window.context_label.text()   # only three names are shown


def test_an_image_is_attached_for_a_vision_model(app, config, tmp_path, monkeypatch):
    shot = tmp_path / "shot.png"
    QImage(4, 4, QImage.Format.Format_RGB32).save(str(shot))
    window = QuickAsk(config)
    monkeypatch.setattr(type(window.agent.backend), "supports_vision", True,
                        raising=False)
    window.load_files([str(shot)])
    assert window._images == [str(shot)]


def test_an_image_falls_back_to_ocr_without_vision(app, config, tmp_path, monkeypatch):
    shot = tmp_path / "shot.png"
    QImage(4, 4, QImage.Format.Format_RGB32).save(str(shot))
    window = QuickAsk(config)
    monkeypatch.setattr(type(window.agent.backend), "supports_vision", False,
                        raising=False)
    window.load_files([str(shot)])
    assert window._images == []
    assert str(shot) in window.composer.toPlainText()


def test_paths_with_spaces_are_quoted_in_the_question(app, config, tmp_path):
    doc = tmp_path / "my file.txt"
    doc.write_text("x")
    window = QuickAsk(config)
    window.load_files([str(doc)])
    assert f'"{doc}"' in window.composer.toPlainText()


def test_no_paths_is_a_no_op(app, config):
    window = QuickAsk(config)
    window.load_files([])
    assert window.composer.toPlainText() == ""


def test_explain_action_sends_immediately(app, config, tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr("maze_ai.ui.quick_ask.QuickAsk.send",
                        lambda self: sent.append(self.composer.toPlainText()))
    doc = tmp_path / "a.txt"
    doc.write_text("x")
    window = QuickAsk(config)
    window.load_files([str(doc)], action="explain")
    # prefill defers the send by a tick so the box is filled first.
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(50, loop.quit)
    loop.exec()
    assert sent and str(doc) in sent[0]


# ── layout: the bar is a bar, not a half-empty dialog ──────────────────────
def test_the_empty_bar_has_no_dead_space(app, config):
    window = QuickAsk(config)
    # Room for the ask row and the hint line, and nothing else.
    assert window.height() <= 160
    assert not window.body.isVisible()


def test_the_input_and_the_button_share_one_line(app, config):
    window = QuickAsk(config)
    window.show()
    assert window.composer.height() == window.send_btn.height(), \
        "the field and the button must be the same height, or nothing lines up"


def test_the_input_centres_its_text(app, config):
    window = QuickAsk(config)
    window.composer.setPlainText("bir satır")
    top = window.composer.viewportMargins().top()
    # A single line in a 44px box sits ~12px down, not at the very top.
    assert top > 4


def test_the_field_grows_with_a_long_question(app, config):
    window = QuickAsk(config)
    single = window.composer.height()
    window.composer.setPlainText("uzun bir soru\n" * 4)
    assert window.composer.height() > single
    assert window.composer.height() <= window.composer.MAX_HEIGHT


def test_content_below_the_row_stays_inside_the_card(app, config):
    # The card reserves margins for its drop shadow; content outside them is
    # painted past the rounded edge and clipped.
    from maze_ai.ui.effects import MARGIN

    window = QuickAsk(config)
    margins = window.ask_row.parentWidget().layout().contentsMargins()
    assert margins.bottom() >= MARGIN
    assert margins.top() >= MARGIN


def test_clipboard_mode_opens_the_body(app, config):
    QGuiApplication.clipboard().setText("systemctl statuss ollama")
    window = QuickAsk(config)
    window.load_clipboard()
    assert window.body.isVisible() or window.context_label.isVisibleTo(window)


# ── rounding: nothing square inside a rounded window ───────────────────────
def test_the_field_is_a_pill(app, config):
    from maze_ai.ui.quick_ask import _ROW_HEIGHT

    window = QuickAsk(config)
    radius = int(window.ask_row.styleSheet().split("border-radius:")[1].split("px")[0])
    assert radius * 2 == _ROW_HEIGHT, \
        "the radius has to be half the row height, or the ends are not round"
    assert window.ask_row.minimumHeight() in (0, _ROW_HEIGHT)


def test_the_input_paints_nothing_of_its_own(app, config):
    # The app-wide sheet gives every QPlainTextEdit a panel background and its
    # own 10px radius — a second box inside the pill.
    from PySide6.QtWidgets import QFrame

    window = QuickAsk(config)
    assert window.composer.frameShape() == QFrame.Shape.NoFrame
    assert "background: transparent" in window.composer.styleSheet()
    assert "border: none" in window.composer.styleSheet()
    assert not window.composer.viewport().autoFillBackground()


def test_the_button_is_round_too(app, config):
    window = QuickAsk(config)
    radius = int(window.send_btn.styleSheet().split("border-radius:")[1].split("px")[0])
    assert radius * 2 == window.send_btn.height(), "the button should be a pill too"
