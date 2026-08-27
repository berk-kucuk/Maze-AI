"""The single-instance socket: one running app, many entry points."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from maze_ai import app as app_module
from maze_ai.config import Config


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
    yield win
    win.close_quick_ask()
    win.close()


def pump(milliseconds: int = 300) -> None:
    """Run the event loop briefly so queued signals are delivered."""
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def send(key: str, message: str) -> None:
    sock = QLocalSocket()
    sock.connectToServer(key)
    assert sock.waitForConnected(500), "no listener on the instance socket"
    sock.write(message.encode())
    sock.flush()
    sock.waitForBytesWritten(500)
    sock.disconnectFromServer()


@pytest.fixture
def listening(window, monkeypatch):
    """A server wired the way app.main wires it."""
    key = f"maze-ai-test-{os.getpid()}"
    QLocalServer.removeServer(key)
    server = QLocalServer()

    def on_connection() -> None:
        conn = server.nextPendingConnection()
        if conn is None:
            return
        conn.readyRead.connect(
            lambda: app_module._dispatch(
                window, bytes(conn.readAll().data()).decode()
            )
        )

    server.newConnection.connect(on_connection)
    assert server.listen(key)
    yield key
    server.close()
    QLocalServer.removeServer(key)


def test_ask_message_opens_quick_ask(listening, window):
    assert getattr(window, "_quick", None) is None
    send(listening, "ask")
    pump()
    assert window._quick is not None


def test_ask_with_text_prefills_the_question(listening, window, monkeypatch):
    # Don't actually call a model: just check the text lands in the box.
    monkeypatch.setattr("maze_ai.ui.quick_ask.QuickAsk.send", lambda self: None)
    send(listening, "ask:how do I list open ports")
    pump()
    assert window._quick.composer.toPlainText() == "how do I list open ports"


def test_clipboard_message_loads_the_clipboard(listening, window):
    from PySide6.QtGui import QGuiApplication

    QGuiApplication.clipboard().setText("systemctl status ollam")
    send(listening, "clipboard")
    pump()
    assert "ollam" in window._quick.context_label.text()


def test_show_message_surfaces_the_chat_window(listening, window):
    window.hide()
    send(listening, "show")
    pump()
    assert window.isVisible()
    assert getattr(window, "_quick", None) is None


def test_file_message_opens_quick_ask_with_the_selection(listening, window, tmp_path):
    doc = tmp_path / "report.md"
    doc.write_text("# report")
    message = app_module._request_from(["--file", str(doc)])
    send(listening, message)
    pump()
    assert window._quick is not None
    assert str(doc) in window._quick.composer.toPlainText()


def test_file_message_survives_spaces_in_paths(listening, window, tmp_path):
    doc = tmp_path / "my notes.md"
    doc.write_text("# notes")
    send(listening, app_module._request_from(["--file", str(doc)]))
    pump()
    assert str(doc) in window._quick.composer.toPlainText()
