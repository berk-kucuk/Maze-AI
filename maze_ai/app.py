"""Application entry point: wires the window, tray and config together."""

from __future__ import annotations

import logging
import os
import signal
import socket
import stat
import struct
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIcon
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from . import autostart, logs
from .config import Config
from .i18n import set_language, tr
from .ui.main_window import MainWindow
from .ui.theme import LOGO_PATH, STYLESHEET
from .ui.tray import Tray

log = logging.getLogger(__name__)

_USAGE = """\
Maze AI — agentic assistant for Maze Linux

  maze-ai                    open the chat window
  maze-ai --ask [text]       pop up Quick Ask over whatever you're doing
  maze-ai --clipboard        ask about what you just copied
  maze-ai --screenshot       drag a box on screen and ask about it
  maze-ai --fix [command]    print the corrected version of a failed command
  maze-ai --ask-cli <text>   answer a question in the terminal, no window
  maze-ai --file <paths>     ask about files (used by the file-manager menu)
  maze-ai --shell-init zsh   print the shell integration (mz, mzask, mzfix)
  maze-ai --install-menus    add "Ask Maze AI" to your file manager
  maze-ai --remove-menus     take it out again

Options:
  --hidden, --tray   start minimized to the system tray
  --quiet            with --fix: print only the corrected command
  --debug            verbose logging to the console and the log file
  --version          print the version and exit
  -h, --help         show this help

Bind `maze-ai --ask` to a key in your desktop's shortcut settings and the
assistant is one keystroke away from anywhere.
"""


# Options that carry no value of their own, so they may sit between a flag and
# its text: `maze-ai --fix --quiet "npm instal"` is the shape the shell
# integration produces.
_STANDALONE_FLAGS = {"--quiet", "--debug", "--hidden", "--tray", "--version"}


def _flag_value(argv: list[str], flag: str) -> str:
    """The words following ``flag`` up to the next value-taking option."""
    if flag not in argv:
        return ""
    words: list[str] = []
    for word in argv[argv.index(flag) + 1:]:
        if word in _STANDALONE_FLAGS:
            continue
        if word.startswith("--"):
            break
        words.append(word)
    return " ".join(words).strip()

# Per-user single-instance rendezvous socket. A second launch connects to this,
# asks the running instance to surface its window, and exits — so the app never
# spawns a duplicate window or tray icon.
#
# The socket is a control channel: an "ask:" message makes the assistant act
# on its text. So it lives in the user's private runtime directory (0700,
# owned by them) rather than the shared /tmp, is created owner-only, and every
# connection's peer is checked to be this same user.


def _instance_key() -> str:
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    try:
        info = os.stat(runtime) if runtime else None
    except OSError:
        info = None
    if (
        info is not None
        and stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.getuid()
        and not info.st_mode & 0o077
    ):
        return os.path.join(runtime, "maze-ai.sock")
    return f"maze-ai-{os.getuid()}"


_INSTANCE_KEY = _instance_key()

#: Longest instance message accepted (a --file selection can be long, but not
#: this long). Anything bigger is dropped rather than buffered without bound.
_MAX_MESSAGE = 256 * 1024


def _peer_uid(conn: QLocalSocket) -> int | None:
    """The uid on the other end of a local socket (Linux SO_PEERCRED)."""
    fd = int(conn.socketDescriptor())
    if fd < 0 or not hasattr(socket, "SO_PEERCRED"):
        return None
    try:
        with socket.socket(fileno=os.dup(fd)) as peer:
            raw = peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        _pid, uid, _gid = struct.unpack("3i", raw)
        return uid
    except OSError:
        return None


def _activate_running_instance(message: str = "show") -> bool:
    """Hand a request to the already-running instance, if there is one.

    The message is what the second launch wanted: surface the window, open
    Quick Ask, ask about the clipboard, grab a region of the screen. One
    running app, many entry points.
    """
    sock = QLocalSocket()
    sock.connectToServer(_INSTANCE_KEY)
    if sock.waitForConnected(300):
        sock.write(message.encode("utf-8"))
        sock.flush()
        sock.waitForBytesWritten(300)
        sock.waitForDisconnected(200)
        sock.disconnectFromServer()
        return True
    return False


#: Separates the parts of a socket message (never appears in a path).
_SEP = "\x1f"


def _file_paths(argv: list[str]) -> list[str]:
    """Everything after ``--file``: the selection the file manager passed."""
    if "--file" not in argv:
        return []
    paths: list[str] = []
    for word in argv[argv.index("--file") + 1:]:
        if word.startswith("--"):
            break
        if word:
            paths.append(word)
    return paths


def _request_from(argv: list[str]) -> str:
    """Translate command-line flags into an instance message."""
    paths = _file_paths(argv)
    if paths:
        action = _flag_value(argv, "--file-action") or "ask"
        return _SEP.join(["file", action, *paths])
    if "--clipboard" in argv:
        return "clipboard"
    if "--screenshot" in argv:
        return "screenshot"
    if "--ask" in argv:
        text = _flag_value(argv, "--ask")
        return f"ask:{text}" if text else "ask"
    return "show"


def _dispatch(window: MainWindow, message: str) -> None:
    """Act on a message from another launch."""
    message = (message or "show").strip()
    log.debug("instance message: %s", message[:80])
    if message.startswith("file" + _SEP):
        _, action, *paths = message.split(_SEP)
        window.quick_ask(mode="files", paths=paths, action=action)
        return
    if message.startswith("ask:"):
        window.quick_ask(text=message[4:])
    elif message == "ask":
        window.quick_ask()
    elif message == "clipboard":
        window.quick_ask(mode="clipboard")
    elif message == "screenshot":
        window.quick_ask(mode="screenshot")
    else:
        _surface(window)


def _surface(window: MainWindow) -> None:
    """Bring the (possibly tray-hidden or minimized) window to the front."""
    window.show()
    window.setWindowState(
        (window.windowState() & ~Qt.WindowState.WindowMinimized)
        | Qt.WindowState.WindowActive
    )
    window.raise_()
    window.activateWindow()


def main() -> int:
    from . import __version__

    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(_USAGE)
        return 0
    if "--version" in argv:
        print(f"Maze AI {__version__}")
        return 0

    # ── headless paths: no Qt, no window, no running instance needed ────
    if "--shell-init" in argv:
        from .cli import shell_init

        return shell_init(_flag_value(argv, "--shell-init") or "zsh")
    if "--fix" in argv:
        from .cli import fix_command

        logs.setup("--debug" in argv)
        return fix_command(_flag_value(argv, "--fix"), quiet="--quiet" in argv)
    if "--ask-cli" in argv:
        from .cli import ask_cli

        logs.setup("--debug" in argv)
        return ask_cli(_flag_value(argv, "--ask-cli"))
    if "--install-menus" in argv or "--remove-menus" in argv:
        from . import desktop_integration

        # Menu labels follow the interface language (module-level import: a
        # second import here would shadow it for the whole function).
        set_language(Config().get("ui_language"))
        if "--remove-menus" in argv:
            removed = desktop_integration.uninstall()
            print("Removed: " + (", ".join(removed) or "nothing"))
            return 0
        results = desktop_integration.install()
        if not results:
            print("No supported file manager found.", file=sys.stderr)
            return 1
        for key, error in results.items():
            print(f"{key}: {error or 'installed'}")
        return 0 if not any(results.values()) else 1

    debug = "--debug" in argv
    log_path = logs.setup(debug)
    log.info("Maze AI %s starting (debug=%s)", __version__, debug)
    if debug and log_path:
        print(f"[maze-ai] logging to {log_path}")

    # Let Ctrl+C terminate the app cleanly when run from a terminal.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    app.setApplicationName("Maze AI")
    app.setApplicationDisplayName("Maze AI")
    app.setDesktopFileName("maze-ai")
    app.setWindowIcon(QIcon(LOGO_PATH))
    app.setQuitOnLastWindowClosed(False)  # keep running in the tray
    font = QFont("Inter")
    font.setStyleHint(QFont.StyleHint.SansSerif)
    font.setPointSizeF(10.5)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)
    # App-wide, so menus, tool tips and file dialogs match the windows.
    app.setStyleSheet(STYLESHEET)

    # Single instance: hand our request to it and bow out.
    request = _request_from(argv)
    if _activate_running_instance(request):
        return 0

    config = Config()
    # Interface language before any widget is built — Qt strings are captured
    # at construction time, so this has to happen first.
    set_language(config.get("ui_language"))
    # Keep the per-user login autostart entry in sync with the preference, so a
    # fresh install starts in the tray next login and toggling it off removes it.
    autostart.sync(config.get("autostart"))
    window = MainWindow(config)

    # Own the single-instance socket. Clear any stale one left by a crash, then
    # listen: later launches connect here and we raise the window instead.
    QLocalServer.removeServer(_INSTANCE_KEY)
    instance_server = QLocalServer(app)
    instance_server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)

    def _on_second_launch() -> None:
        conn = instance_server.nextPendingConnection()
        if conn is None:
            return
        uid = _peer_uid(conn)
        if uid is not None and uid != os.getuid():
            log.warning("refused an instance message from uid %s", uid)
            conn.abort()
            conn.deleteLater()
            return
        buffer = bytearray()

        def _read() -> None:
            buffer.extend(bytes(conn.readAll().data()))
            if len(buffer) > _MAX_MESSAGE:
                log.warning("dropped an oversized instance message")
                buffer.clear()
                conn.abort()

        def _done() -> None:
            if buffer:
                _dispatch(window, bytes(buffer).decode("utf-8", "replace"))
            conn.deleteLater()

        conn.readyRead.connect(_read)
        conn.disconnected.connect(_done)

    instance_server.newConnection.connect(_on_second_launch)
    instance_server.listen(_INSTANCE_KEY)

    tray_ok = QSystemTrayAvailable()
    tray = Tray(window, app) if tray_ok else None
    if tray:
        tray.show()
        window.tray = tray  # lets the window/reminders notify through the tray
        window.hidden_to_tray.connect(
            lambda: tray.notify("Maze AI", tr("Still running in the tray."))
        )

    # Start in the tray when asked — either via the saved preference or the
    # --hidden/--tray flag the autostart entry passes at login.
    start_hidden = (
        config.get("start_hidden")
        or "--hidden" in argv
        or "--tray" in argv
    )
    if request != "show":
        # Launched as a shortcut: Quick Ask is the only window wanted, whether
        # or not this desktop has a tray to hide the chat window in.
        pass
    elif start_hidden and tray_ok:
        # Start minimized to tray — no window is shown.
        pass
    else:
        window.show()
        QTimer.singleShot(60, window.input_bar.focus_input)
        QTimer.singleShot(400, window.maybe_onboard)

    if not tray_ok:
        # No tray: closing should really quit.
        config.set("close_to_tray", False)

    if request != "show":
        # Launched straight into Quick Ask: don't show the chat window at all.
        QTimer.singleShot(120, lambda: _dispatch(window, request))

    # If the saved Ollama model isn't installed, fall back to one that is —
    # off the UI thread, because it is an HTTP call to a server that may not be
    # running, and the window must not wait on its timeout to appear.
    QTimer.singleShot(0, window.resolve_model_async)

    # Greet the user (in their language) and surface any reminders that came
    # due while the app was closed, shortly after the tray is up.
    QTimer.singleShot(1200, window.greet)
    QTimer.singleShot(1500, window._check_reminders)

    return app.exec()


def QSystemTrayAvailable() -> bool:
    from PySide6.QtWidgets import QSystemTrayIcon

    return QSystemTrayIcon.isSystemTrayAvailable()


if __name__ == "__main__":
    raise SystemExit(main())
