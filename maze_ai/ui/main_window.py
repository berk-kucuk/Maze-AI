"""Main window: frameless glass card — history sidebar, chat column, composer."""

from __future__ import annotations

import html
import threading
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..agent import Agent, AgentEvent, ApprovalRequest
from ..agent.tools import set_reminder_store
from ..config import Config
from ..history import ChatStore, Conversation, export_markdown
from ..i18n import tr
from ..llm import OllamaBackend, build_backend, resolve_ollama_model
from ..llm.hardware import GB
from ..reminders import ReminderStore
from .approval import ApprovalDialog
from .chat_view import COLUMN_MAX, ChatView
from .dialogs import ConfirmDialog, PromptDialog, ShortcutsDialog, notify_toast, shortcut_text
from .effects import AuroraCard, GlowDot
from .input_bar import InputBar
from .onboarding import OnboardingDialog
from .reminders_dialog import RemindersDialog
from .richtext import harden_labels, plain_label
from .settings_dialog import SettingsDialog
from .sidebar import HistorySidebar
from .theme import LOGO_PATH, OK, STYLESHEET, TEXT_FAINT
from .title_bar import TitleBar
from .worker import AgentWorker, ModelResolveWorker

# The first screen of a new chat.
EMPTY_TITLE = "How can I help?"
EMPTY_SUBTITLE = (
    "Commands, files, apps, the web and reminders on Maze Linux — "
    "anything that changes your system asks you first."
)
SUGGESTIONS: list[tuple[str, str, str]] = [
    ("terminal", "Show my disk usage", "Show my disk usage and the biggest folders in my home."),
    ("sparkle", "Open Firefox", "Open Firefox."),
    ("file", "Create a Python venv in ~/dev", "Create a Python virtual environment in ~/dev/venv."),
    ("alarm", "Remind me to take a break in 30 minutes",
     "Remind me to take a break in 30 minutes."),
]

# Localized startup greetings, keyed by output-language code.
GREETINGS = {
    "en": "Hello! Maze AI is ready to help.",
    "tr": "Merhaba! Maze AI yardıma hazır.",
    "de": "Hallo! Maze AI ist bereit zu helfen.",
    "fr": "Bonjour ! Maze AI est prêt à vous aider.",
    "es": "¡Hola! Maze AI está listo para ayudar.",
    "it": "Ciao! Maze AI è pronto ad aiutarti.",
    "pt": "Olá! O Maze AI está pronto para ajudar.",
    "ru": "Привет! Maze AI готов помочь.",
    "ar": "مرحبًا! Maze AI جاهز للمساعدة.",
    "zh": "你好！Maze AI 已准备好为你服务。",
    "ja": "こんにちは！Maze AI の準備ができました。",
}

#: Width of the invisible band around the card that resizes the window.
_RESIZE_BAND = 10


class MainWindow(QWidget):
    hidden_to_tray = Signal()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.agent = Agent(
            build_backend(config),
            mode=config.get("agent_mode"),
            max_steps=int(config.get("max_steps")),
            command_timeout=int(config.get("command_timeout")),
            language=config.get("output_language"),
            custom_instructions=config.get("custom_instructions"),
            context_char_budget=int(config.get("context_char_budget")),
            stream_responses=bool(config.get("stream_responses")),
            block_dangerous=bool(config.get("block_dangerous_commands")),
            auto_approve_readonly=bool(config.get("auto_approve_readonly")),
            always_allow=list(config.get("always_allow") or []),
            guard_secrets=bool(config.get("guard_secrets")),
            confirm_egress=bool(config.get("confirm_network_egress")),
            native_tools=bool(config.get("native_tools")),
            constrain_json=bool(config.get("constrain_json")),
            tool_groups=list(config.get("tool_groups") or []),
        )
        self.worker: AgentWorker | None = None
        self._resolver: ModelResolveWorker | None = None
        # Local-model telemetry and server health, filled in off the UI thread.
        self._last_metrics: dict = {}
        self._ollama_ok: bool | None = None
        self._runtime = None          # ModelRuntime: where the model is running
        self._warming = False
        self._thinking_text = ""
        self._last_user_message = ""  # for the Regenerate action
        self.shortcuts: list[QShortcut] = []

        # Chat history: start on a fresh (unsaved) conversation. The agent's
        # message list IS the conversation's, so every turn is remembered and
        # persisted; the model replays it as context on the next turn.
        self.store = ChatStore()
        self.conversation = Conversation()
        self.agent.history = self.conversation.messages

        # Reminders: shared store between the agent's tools and the UI's checker.
        self.reminders = ReminderStore()
        set_reminder_store(self.reminders)
        self.tray = None  # set by app.py once the tray exists

        self.setWindowTitle("Maze AI")
        self.setWindowIcon(QIcon(LOGO_PATH))
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(STYLESHEET)
        self.resize(1160, 820)
        self.setMinimumSize(720, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.card = AuroraCard(self)
        root.addWidget(self.card)
        # Resizing a frameless window: the band around the card acts as edges.
        self.card.setMouseTracking(True)
        self.card.installEventFilter(self)

        body = QHBoxLayout(self.card)
        body.setContentsMargins(self.card.margin, self.card.margin,
                                self.card.margin, self.card.margin)
        body.setSpacing(0)
        self._body = body

        self.sidebar = HistorySidebar()
        self.sidebar.new_chat_requested.connect(self.new_chat)
        self.sidebar.chat_selected.connect(self.load_chat)
        self.sidebar.chat_deleted.connect(self.delete_chat)
        self.sidebar.chat_renamed.connect(self.rename_chat)
        self.sidebar.chat_exported.connect(self.export_chat)
        self.sidebar.search_left.connect(self.focus_composer)
        self.sidebar.setVisible(bool(config.get("sidebar_visible")))
        body.addWidget(self.sidebar)

        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 12)
        column.setSpacing(0)

        # title bar (spans the chat column)
        self.title_bar = TitleBar()
        self.title_bar.minimize_clicked.connect(self.showMinimized)
        self.title_bar.maximize_clicked.connect(self.toggle_maximized)
        self.title_bar.close_clicked.connect(self._on_close_button)
        self.title_bar.settings_clicked.connect(self.open_settings)
        self.title_bar.sidebar_clicked.connect(self.toggle_sidebar)
        self.title_bar.reminders_clicked.connect(self.open_reminders)
        self.title_bar.shortcuts_clicked.connect(self.show_shortcuts)
        column.addWidget(self.title_bar)

        self.chat = ChatView()
        self.chat.suggestion_clicked.connect(self.on_send)
        self.chat.regenerate_requested.connect(self.regenerate)
        column.addWidget(self.chat, 1)

        # composer + status, centred under the reading column
        dock = QHBoxLayout()
        dock.setContentsMargins(24, 4, 24, 0)
        dock.addStretch(1)
        dock_col = QVBoxLayout()
        dock_col.setSpacing(6)

        self.input_bar = InputBar()
        self.input_bar.send.connect(self.on_send)
        self.input_bar.stop.connect(self.on_stop)
        self.input_bar.composer.recall_requested.connect(self.recall_last_message)
        self.input_bar.composer.page_requested.connect(self.chat.scroll_page)
        dock_col.addWidget(self.input_bar)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(6, 0, 2, 0)
        status_row.setSpacing(8)
        self.status_dot = GlowDot(OK)
        status_row.addWidget(self.status_dot)
        self.status_label = plain_label("")
        self.status_label.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 8.5pt;")
        status_row.addWidget(self.status_label, 1)
        self.shrink_ctx_btn = QPushButton("")
        self.shrink_ctx_btn.setObjectName("chip")
        self.shrink_ctx_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.shrink_ctx_btn.clicked.connect(self.shrink_context)
        self.shrink_ctx_btn.hide()
        status_row.addWidget(self.shrink_ctx_btn)

        self.start_ollama_btn = QPushButton(tr("Start Ollama"))
        self.start_ollama_btn.setObjectName("chip")
        self.start_ollama_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_ollama_btn.setToolTip(
            tr("Run `systemctl --user start ollama` and check again")
        )
        self.start_ollama_btn.clicked.connect(self.start_ollama)
        self.start_ollama_btn.hide()
        status_row.addWidget(self.start_ollama_btn)

        hint = plain_label(
            tr("{keys} for shortcuts").format(keys=shortcut_text("Ctrl+/"))
        )
        hint.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 8.5pt;")
        status_row.addWidget(hint)
        dock_col.addLayout(status_row)

        dock_host = QWidget()
        dock_host.setLayout(dock_col)
        dock_host.setMaximumWidth(COLUMN_MAX + 8)
        dock.addWidget(dock_host, 100)
        dock.addStretch(1)
        column.addLayout(dock)

        body.addLayout(column, 1)

        self._install_shortcuts()
        self._refresh_status()
        self._render_conversation()
        self._refresh_sidebar()
        harden_labels(self)

        # Poll for due reminders and fire desktop notifications.
        self._reminder_timer = QTimer(self)
        self._reminder_timer.setInterval(20_000)  # every 20s
        self._reminder_timer.timeout.connect(self._check_reminders)
        self._reminder_timer.start()

    # ── keyboard ─────────────────────────────────────────────────────────
    def shortcut_table(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """Every shortcut, grouped — the single source for the help dialog.

        Entries whose handler is ``None`` are handled by a widget (Enter in
        the composer, Meta+M by the desktop) and are listed for reference.
        """
        return [
            (tr("Chats"), [
                (tr("New chat"), "Ctrl+N"),
                (tr("Search chats"), "Ctrl+F"),
                (tr("Previous / next chat"), "Alt+Up / Alt+Down"),
                (tr("Rename chat"), "F2"),
                (tr("Export chat"), "Ctrl+E"),
                (tr("Delete chat"), "Ctrl+Shift+Backspace"),
                (tr("Show / hide history"), "Ctrl+B"),
            ]),
            (tr("Messages"), [
                (tr("Send"), "Return"),
                (tr("New line"), "Shift+Return"),
                (tr("Stop generating"), "Esc"),
                (tr("Regenerate the last answer"), "Ctrl+R"),
                (tr("Copy the last answer"), "Ctrl+Shift+C"),
                (tr("Edit the last message"), "Up"),
                (tr("Attach an image"), "Ctrl+O"),
                (tr("Focus the message box"), "Ctrl+L"),
                (tr("Scroll the conversation"), "PgUp / PgDown"),
            ]),
            (tr("Window"), [
                (tr("Settings"), "Ctrl+,"),
                (tr("Reminders"), "Ctrl+Shift+R"),
                (tr("Keyboard shortcuts"), "Ctrl+/ / F1"),
                (tr("Maximize / restore"), "F11"),
                (tr("Hide the window"), "Ctrl+W"),
                (tr("Quit Maze AI"), "Ctrl+Q"),
            ]),
            (tr("Anywhere on the desktop"), [
                (tr("Quick Ask"), "Meta+M"),
            ]),
            (tr("Quick Ask"), [
                (tr("Ask"), "Return"),
                (tr("Close"), "Esc"),
                (tr("Copy the answer"), "Ctrl+Shift+C"),
                (tr("Continue in chat"), "Ctrl+Shift+Return"),
                (tr("Clipboard actions"), "Alt+1 / Alt+4"),
            ]),
            (tr("Approval dialog"), [
                (tr("Approve"), "Ctrl+Return"),
                (tr("Deny"), "Esc"),
                (tr("Always allow this command"), "Alt+A"),
            ]),
        ]

    def _install_shortcuts(self) -> None:
        bindings: list[tuple[str, object]] = [
            ("Ctrl+N", self.new_chat),
            ("Ctrl+F", self.focus_search),
            ("Alt+Up", lambda: self.switch_chat(-1)),
            ("Alt+Down", lambda: self.switch_chat(1)),
            ("F2", lambda: self.rename_chat(self.conversation.id)),
            ("Ctrl+E", lambda: self.export_chat(self.conversation.id)),
            ("Ctrl+Shift+Backspace", lambda: self.delete_chat(self.conversation.id)),
            ("Ctrl+B", self.toggle_sidebar),
            ("Esc", self.on_escape),
            ("Ctrl+R", self.regenerate),
            ("Ctrl+Shift+C", self.copy_last_answer),
            ("Ctrl+O", self.input_bar.attach),
            ("Ctrl+L", self.focus_composer),
            ("Ctrl+,", self.open_settings),
            ("Ctrl+Shift+R", self.open_reminders),
            ("Ctrl+/", self.show_shortcuts),
            ("F1", self.show_shortcuts),
            ("F11", self.toggle_maximized),
            ("Ctrl+W", self._on_close_button),
            ("Ctrl+Q", self.quit_app),
        ]
        for keys, handler in bindings:
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(handler)
            self.shortcuts.append(shortcut)

    def show_shortcuts(self) -> None:
        ShortcutsDialog(self.shortcut_table(), self).exec()

    def focus_composer(self) -> None:
        self.input_bar.focus_input()

    def focus_search(self) -> None:
        if not self.sidebar.isVisible():
            self.sidebar.show()
        self.sidebar.focus_search()

    def on_escape(self) -> None:
        """Esc: stop a running answer; otherwise put the window away."""
        if self._busy():
            self.on_stop()
        elif self.config.get("close_to_tray"):
            self._on_close_button()

    def switch_chat(self, step: int) -> None:
        target = self.sidebar.neighbour(self.conversation.id, step)
        if target:
            self.load_chat(target)

    def copy_last_answer(self) -> None:
        text = self.chat.last_answer()
        if not text:
            for msg in reversed(self.conversation.messages):
                if msg.get("role") == "assistant":
                    text = msg.get("content", "")
                    break
        if text:
            QApplication.clipboard().setText(text)
            notify_toast(self, tr("Answer copied"), kind="ok")

    def recall_last_message(self) -> None:
        """↑ in an empty box: edit and resend the last message."""
        if self._busy() or not self._last_user_message:
            return
        self.input_bar.set_text(self._last_user_message)

    def _busy(self) -> bool:
        return bool(self.worker and self.worker.isRunning())

    # ── frameless window: maximize and edge resizing ────────────────────
    def toggle_maximized(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def changeEvent(self, event) -> None:  # noqa: N802
        if event.type() == QEvent.Type.WindowStateChange:
            maximized = self.isMaximized() or self.isFullScreen()
            self.card.set_flush(maximized)
            m = self.card.margin
            self._body.setContentsMargins(m, m, m, m)
            self.title_bar.set_maximized(maximized)
        super().changeEvent(event)

    def _edges_at(self, pos: QPoint) -> Qt.Edge:
        edges = Qt.Edge(0)
        if self.isMaximized() or self.isFullScreen():
            return edges
        m = self.card.margin
        lo, w, h = m - _RESIZE_BAND, self.width(), self.height()
        x, y = pos.x(), pos.y()
        if lo <= x <= m + 2:
            edges |= Qt.Edge.LeftEdge
        elif w - m - 2 <= x <= w - lo:
            edges |= Qt.Edge.RightEdge
        if lo <= y <= m + 2:
            edges |= Qt.Edge.TopEdge
        elif h - m - 2 <= y <= h - lo:
            edges |= Qt.Edge.BottomEdge
        return edges

    _CURSORS = {
        Qt.Edge.LeftEdge: Qt.CursorShape.SizeHorCursor,
        Qt.Edge.RightEdge: Qt.CursorShape.SizeHorCursor,
        Qt.Edge.TopEdge: Qt.CursorShape.SizeVerCursor,
        Qt.Edge.BottomEdge: Qt.CursorShape.SizeVerCursor,
    }

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self.card:
            etype = event.type()
            if etype == QEvent.Type.MouseMove and not event.buttons():
                edges = self._edges_at(event.position().toPoint())
                if not edges:
                    self.card.unsetCursor()
                elif edges in (Qt.Edge.LeftEdge | Qt.Edge.TopEdge,
                               Qt.Edge.RightEdge | Qt.Edge.BottomEdge):
                    self.card.setCursor(Qt.CursorShape.SizeFDiagCursor)
                elif edges in (Qt.Edge.RightEdge | Qt.Edge.TopEdge,
                               Qt.Edge.LeftEdge | Qt.Edge.BottomEdge):
                    self.card.setCursor(Qt.CursorShape.SizeBDiagCursor)
                else:
                    self.card.setCursor(self._CURSORS.get(edges, Qt.CursorShape.ArrowCursor))
            elif etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                edges = self._edges_at(event.position().toPoint())
                handle = self.windowHandle()
                if edges and handle is not None and handle.startSystemResize(edges):
                    return True
            elif etype == QEvent.Type.Leave:
                self.card.unsetCursor()
        return super().eventFilter(obj, event)

    # ── Quick Ask ────────────────────────────────────────────────────────
    def quick_ask(
        self,
        mode: str = "",
        text: str = "",
        paths: list[str] | None = None,
        action: str = "ask",
    ) -> None:
        """Pop the floating ask bar over whatever the user is doing.

        ``mode`` picks the entry point: "" for a plain question, "clipboard" to
        act on what they just copied, "screenshot" to drag a box on screen and
        ask about that.
        """
        from .quick_ask import QuickAsk  # local: the chat window shouldn't need it

        window = getattr(self, "_quick", None)
        if window is None:
            window = QuickAsk(self.config)
            window.open_in_chat.connect(self._quick_to_chat)
            self._quick = window
        window.surface()

        if mode == "clipboard":
            window.load_clipboard()
        elif mode == "screenshot":
            self._quick_screenshot(window)
        elif mode == "files":
            window.load_files(paths or [], action)
        elif text:
            window.prefill(text, send=True)

    def _quick_screenshot(self, window) -> None:
        """Let the user drag a box, then ask about the captured area."""
        from ..agent.tools import capture_region

        window.set_status(tr("Drag a box on the screen…"))
        window.hide()          # get out of the way of the selection overlay
        captured: dict = {}

        def work() -> None:
            captured["result"] = capture_region()

        threading.Thread(target=work, daemon=True, name="region-capture").start()

        def check() -> None:
            result = captured.get("result")
            if result is None:
                QTimer.singleShot(250, check)
                return
            window.surface()
            if result.ok:
                window.set_status("")
                window.attach_image(result.output)
            else:
                window.set_status(result.output)

        QTimer.singleShot(400, check)

    def _quick_to_chat(self, question: str, answer: str) -> None:
        """Carry a Quick Ask exchange into a real, saved conversation."""
        if not question:
            return
        self._save_current()
        self.conversation = Conversation()
        self.conversation.messages.append({"role": "user", "content": question})
        if answer:
            self.conversation.messages.append({"role": "assistant", "content": answer})
        self.conversation.touch()
        self.agent.history = self.conversation.messages
        self._render_conversation()
        self._save_current()
        self._refresh_sidebar()
        self._sync_regen_state()
        self.show()
        self.raise_()
        self.activateWindow()
        self.input_bar.focus_input()

    # ── startup checks ───────────────────────────────────────────────────
    def resolve_model_async(self) -> None:
        """Verify the configured Ollama model exists, without blocking the UI."""
        if self.config.get("backend") != "ollama":
            return
        self._resolver = ModelResolveWorker(self.config)
        self._resolver.resolved.connect(self._on_model_resolved)
        self._resolver.start()

    def _stop_workers(self) -> None:
        """Stop every background thread before the process goes away.

        A QThread that is still running when its object is destroyed aborts the
        whole application, so quitting seconds after launch (while the model
        check is still talking to Ollama) would crash on the way out.
        """
        for attr, grace in (("worker", 1500), ("_resolver", 6000)):
            thread = getattr(self, attr, None)
            if thread is None:
                continue
            try:
                if thread.isRunning():
                    thread.requestInterruption()
                    thread.quit()
                    thread.wait(grace)
            except RuntimeError:
                pass  # already destroyed by Qt

    def _on_model_resolved(self, model: str) -> None:
        # Only rebuild when the resolver actually switched models on us.
        if model and model != getattr(self.agent.backend, "model", model):
            self.agent.backend = build_backend(self.config)
        self._refresh_status()
        self.check_ollama_async()

    # ── Ollama health & warm-up ──────────────────────────────────────────
    def ollama_backend(self) -> OllamaBackend | None:
        """The active backend, when it is Ollama."""
        backend = self.agent.backend
        return backend if isinstance(backend, OllamaBackend) else None

    def check_ollama_async(self) -> None:
        """Ping the server and (optionally) preload the model, off the UI thread.

        Plain daemon threads rather than QThreads: a preload can take fifteen
        seconds on a cold 12B model, and nothing here is worth making the user
        wait for at shutdown.
        """
        backend = self.ollama_backend()
        if backend is None:
            self._ollama_ok = None
            self._warming = False
            self._sync_ollama_ui()
            return
        preload = bool(self.config.get("ollama_preload"))
        self._warming = preload

        def work() -> None:
            ok = backend.is_reachable()
            self._ollama_ok = ok
            if ok and preload:
                loaded = backend.model in backend.loaded_models()
                if not loaded:
                    backend.preload()
            if ok:
                self._runtime = backend.runtime()
            self._warming = False

        threading.Thread(target=work, daemon=True, name="ollama-check").start()
        # The thread only flips plain attributes; the UI reads them here, on the
        # UI thread, so no cross-thread signal can outlive the window.
        self._ollama_timer = QTimer(self)
        self._ollama_timer.setInterval(600)
        self._ollama_timer.timeout.connect(self._poll_ollama)
        self._ollama_timer.start()

    def refresh_runtime_async(self) -> None:
        """Ask the server where the model is running, off the UI thread."""
        backend = self.ollama_backend()
        if backend is None:
            self._runtime = None
            return

        def work() -> None:
            self._runtime = backend.runtime()

        threading.Thread(target=work, daemon=True, name="ollama-runtime").start()
        QTimer.singleShot(700, self._sync_ollama_ui)

    def _explain_placement(self, runtime) -> None:
        """Put the why behind the CPU/GPU split into the status tooltip."""
        backend = self.ollama_backend()
        if backend is None or not runtime.spilled:
            self.status_label.setToolTip("")
            return
        try:
            report = backend.fit()
            usable, _ = backend.usable_vram()
            lines = [
                tr("{model} needs about {need} GB; {usable} GB of VRAM is usable, "
                   "so part of it runs on the CPU.").format(
                    model=backend.model,
                    need=f"{report.need_bytes / GB:.1f}",
                    usable=f"{usable / GB:.1f}",
                )
            ]
            alternatives = [m for m in backend.models_that_fit() if m != backend.model]
            if alternatives:
                lines.append(
                    tr("Installed models that would fit: {models}").format(
                        models=", ".join(alternatives[:3])
                    )
                )
        except Exception:  # noqa: BLE001 - an explanation must never break the UI
            return
        self.status_label.setToolTip("\n".join(lines))

    def shrink_context(self) -> None:
        """Drop the context window to what fits in VRAM, and reload."""
        backend = self.ollama_backend()
        if backend is None:
            return
        best = backend.best_context()
        self.config.set("ollama_num_ctx", int(best))
        self.config.save()
        self.agent.backend = build_backend(self.config)
        self.shrink_ctx_btn.hide()
        self.status_label.setText(
            tr("Context set to {tokens}. The model will reload on the next message.")
            .format(tokens=best)
        )
        self.check_ollama_async()

    def _poll_ollama(self) -> None:
        self._sync_ollama_ui()
        if not self._warming and self._ollama_ok is not None:
            self._ollama_timer.stop()

    def _sync_ollama_ui(self) -> None:
        backend = self.ollama_backend()
        self.start_ollama_btn.setVisible(bool(backend) and self._ollama_ok is False)
        # Offer the one-click fix only when it would actually help: the model
        # spilled onto the CPU *and* a smaller window would keep it on the GPU.
        runtime = self._runtime
        show_fix = False
        if backend is not None and runtime is not None and runtime.spilled:
            try:
                best = backend.best_context()
                show_fix = best < backend.resolved_ctx()
                if show_fix:
                    self.shrink_ctx_btn.setText(
                        tr("Shrink context to {tokens}").format(tokens=best)
                    )
                    self.shrink_ctx_btn.setToolTip(
                        tr("Part of the model is running on the CPU. A smaller "
                           "context window keeps it in VRAM, which is several "
                           "times faster.")
                    )
            except Exception:  # noqa: BLE001 - a hint must never break the UI
                show_fix = False
        self.shrink_ctx_btn.setVisible(show_fix)
        self._refresh_status()

    def start_ollama(self) -> None:
        """Try to bring the local Ollama server up, then re-check."""
        import subprocess

        for argv in (["systemctl", "--user", "start", "ollama"], ["ollama", "serve"]):
            try:
                subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
                break
            except (OSError, subprocess.SubprocessError):
                continue
        self.status_label.setText(tr("Starting Ollama…"))
        QTimer.singleShot(2500, self.check_ollama_async)

    # ── notifications / reminders ────────────────────────────────────────
    def notify(self, title: str, message: str) -> None:
        """Send a desktop notification via the tray (or notify-send fallback).

        Notification servers render a subset of HTML in the body (links,
        images, bold). Reminder text comes from the agent, so it is escaped:
        a notification must show text, never fetch or link to anything.
        """
        body = html.escape(message or "", quote=False)
        if self.tray is not None:
            self.tray.notify(title, body)
            return
        try:
            import subprocess
            subprocess.Popen(
                ["notify-send", "-a", "Maze AI", "--", title, body],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:  # noqa: BLE001
            pass

    def _check_reminders(self) -> None:
        self.reminders.load()  # pick up reminders added by the agent tools
        for r in self.reminders.due():
            self.notify(tr("⏰ Reminder"), r.text)

    def greet(self) -> None:
        """Send a one-time greeting notification in the chosen language."""
        if not self.config.get("greet_on_start"):
            return
        lang = self.config.get("output_language") or "auto"
        text = GREETINGS.get(lang, GREETINGS["en"])
        # Also surface any reminders the user missed while the app was closed.
        self.notify("Maze AI", text)

    # ── status / config ──────────────────────────────────────────────────
    def _refresh_status(self) -> None:
        desc = self.agent.backend.describe()
        mode = self.config.get("agent_mode")
        self.title_bar.set_backend_badge(desc)
        parts = [f"{tr('mode')}: {mode}"]
        if self.ollama_backend() is not None:
            if self._ollama_ok is False:
                parts.append(tr("Ollama is not running"))
            elif self._warming:
                parts.append(tr("warming up the model…"))
            elif self.agent.uses_native_tools():
                # Worth showing: it is the difference between a model that can
                # really use tools and one being coaxed through JSON.
                parts.append(tr("native tools"))
        runtime = self._runtime
        if runtime is not None and runtime.loaded:
            parts.append(runtime.processor())
            self._explain_placement(runtime)
        speed = self._last_metrics.get("tokens_per_second") or 0
        if speed:
            parts.append(f"{speed:.1f} tok/s")
        used = self._last_metrics.get("prompt_tokens") or 0
        limit = self._last_metrics.get("context_limit") or 0
        if used and limit:
            parts.append(
                tr("context {used}/{limit}").format(
                    used=f"{used / 1000:.1f}k" if used >= 1000 else used,
                    limit=f"{limit // 1024}k" if limit >= 1024 else limit,
                )
            )
        self.status_label.setText("  ·  ".join(parts))
        down = self.ollama_backend() is not None and self._ollama_ok is False
        self.status_dot.set_color("#ff6b6b" if down else OK)
        self.input_bar.set_vision(getattr(self.agent.backend, "supports_vision", False))

    def open_settings(self) -> None:
        dlg = SettingsDialog(self.config, self)
        if dlg.exec():
            # Rebuild backend but keep the conversation.
            resolve_ollama_model(self.config)
            self.agent.backend = build_backend(self.config)
            self.agent.mode = self.config.get("agent_mode")
            self.agent.max_steps = int(self.config.get("max_steps"))
            self.agent.command_timeout = int(self.config.get("command_timeout"))
            self.agent.language = self.config.get("output_language")
            self.agent.custom_instructions = self.config.get("custom_instructions")
            self.agent.context_char_budget = int(self.config.get("context_char_budget"))
            self.agent.stream_responses = bool(self.config.get("stream_responses"))
            self.agent.block_dangerous = bool(self.config.get("block_dangerous_commands"))
            self.agent.auto_approve_readonly = bool(self.config.get("auto_approve_readonly"))
            self.agent.always_allow = list(self.config.get("always_allow") or [])
            self.agent.guard_secrets = bool(self.config.get("guard_secrets"))
            self.agent.confirm_egress = bool(self.config.get("confirm_network_egress"))
            self.agent.native_tools = bool(self.config.get("native_tools"))
            self.agent.constrain_json = bool(self.config.get("constrain_json"))
            self.agent.tool_groups = list(self.config.get("tool_groups") or [])
            self._last_metrics = {}
            self._refresh_status()
            self.check_ollama_async()

    def open_reminders(self) -> None:
        RemindersDialog(self.reminders, self).exec()

    def maybe_onboard(self) -> None:
        """Show the first-run welcome once; open Settings if the user opts in."""
        if self.config.get("onboarded"):
            return
        self.config.set("onboarded", True)
        self.config.save()
        if OnboardingDialog(self).exec() == QDialog.DialogCode.Accepted:
            self.open_settings()

    # ── conversation management ──────────────────────────────────────────
    def _render_conversation(self) -> None:
        """Draw the current conversation's messages, or the welcome screen."""
        self.chat.clear()
        self.title_bar.set_title(
            tr("New chat") if self.conversation.is_empty
            else (self.conversation.title if self.conversation.title != "New chat"
                  else tr("New chat"))
        )
        if self.conversation.is_empty:
            self.chat.show_empty_state(
                tr(EMPTY_TITLE),
                tr(EMPTY_SUBTITLE),
                [(icon, tr(label), tr(prompt)) for icon, label, prompt in SUGGESTIONS],
                hint=tr("Press {keys} to see every keyboard shortcut.").format(
                    keys=shortcut_text("Ctrl+/")
                ),
            )
            return
        for msg in self.conversation.messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                self.chat.add_user(content)
            elif role == "assistant":
                if content.startswith("(error)"):
                    self.chat.add_error(content[len("(error)"):].strip())
                else:
                    self.chat.add_ai(content)

    def _save_current(self) -> None:
        """Persist the current conversation (no-op if it's still empty)."""
        self.store.save(self.conversation)

    def _refresh_sidebar(self) -> None:
        self.sidebar.set_conversations(
            self.store.list_conversations(), self.conversation.id
        )
        if not self.conversation.is_empty:
            self.title_bar.set_title(self.conversation.title)

    def _sync_regen_state(self) -> None:
        """Point Regenerate at the CURRENT conversation's last user message.

        Without this, switching chats (or opening a fresh, empty one) left the
        button pointing at the previous chat's message — so Regenerate would
        re-run an old prompt in the new session.
        """
        last_user = ""
        for msg in self.conversation.messages:
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
        self._last_user_message = last_user
        self.chat.set_regenerate_available(bool(last_user))

    def toggle_sidebar(self) -> None:
        visible = not self.sidebar.isVisible()
        self.sidebar.setVisible(visible)
        self.config.set("sidebar_visible", visible)
        try:
            self.config.save()
        except OSError:
            pass

    def new_chat(self) -> None:
        if self._busy():
            return
        # Keep the existing chat — just save it and start a fresh one.
        self._save_current()
        self.conversation = Conversation()
        self.agent.history = self.conversation.messages
        self._render_conversation()
        self._refresh_sidebar()
        self._sync_regen_state()
        self.input_bar.focus_input()

    def load_chat(self, conv_id: str) -> None:
        if self._busy():
            notify_toast(self, tr("Wait for the answer to finish, or press Esc to stop it."))
            return
        if conv_id == self.conversation.id:
            return
        self._save_current()
        loaded = self.store.load(conv_id)
        if loaded is None:
            self._refresh_sidebar()
            return
        self.conversation = loaded
        self.agent.history = self.conversation.messages
        self._render_conversation()
        self._refresh_sidebar()
        self._sync_regen_state()
        self.input_bar.focus_input()

    def _find_title(self, conv_id: str) -> str:
        if conv_id == self.conversation.id:
            return self.conversation.title
        loaded = self.store.load(conv_id)
        return loaded.title if loaded else ""

    def delete_chat(self, conv_id: str, *, confirm: bool = True) -> None:
        is_current = conv_id == self.conversation.id
        if is_current and self.conversation.is_empty:
            return  # nothing saved to delete
        if is_current and self._busy():
            return
        if confirm:
            dialog = ConfirmDialog(
                tr("Delete this chat?"),
                tr("It will be removed from your history. This can't be undone."),
                detail=self._find_title(conv_id),
                confirm=tr("Delete"),
                danger=True,
                parent=self,
            )
            if not dialog.exec():
                return
        self.store.delete(conv_id)
        if is_current:
            # Deleting the open chat drops us onto a fresh one.
            self.conversation = Conversation()
            self.agent.history = self.conversation.messages
            self._render_conversation()
            self._sync_regen_state()
        self._refresh_sidebar()
        self.input_bar.focus_input()
        notify_toast(self, tr("Chat deleted"))

    def rename_chat(self, conv_id: str) -> None:
        if conv_id == self.conversation.id and self.conversation.is_empty:
            return
        current = self._find_title(conv_id)
        dialog = PromptDialog(tr("Rename chat"), tr("New title:"), current, parent=self)
        if not dialog.exec():
            return
        title = dialog.value()
        if not title:
            return
        if conv_id == self.conversation.id:
            self.conversation.title = title[:80]
            self._save_current()
        else:
            self.store.rename(conv_id, title)
        self._refresh_sidebar()

    def export_chat(self, conv_id: str) -> None:
        conv = self.conversation if conv_id == self.conversation.id else self.store.load(conv_id)
        if conv is None or conv.is_empty:
            return
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in conv.title)[:40].strip()
        path, _ = QFileDialog.getSaveFileName(
            self, tr("Export chat"), f"{safe or 'chat'}.md", "Markdown (*.md)"
        )
        if not path:
            return
        try:
            Path(path).write_text(export_markdown(conv), encoding="utf-8")
            notify_toast(self, tr("Chat exported to {path}").format(path=path), kind="ok")
        except OSError as exc:
            notify_toast(self, tr("Export failed: {error}").format(error=exc), kind="danger")

    # ── sending ──────────────────────────────────────────────────────────
    def on_send(self, text: str) -> None:
        if self._busy():
            return
        text = (text or "").strip()
        if not text:
            return
        images = self.input_bar.take_attachments()
        self._last_user_message = text
        self._thinking_text = ""
        # Bump the conversation's timestamp now, at send time, so the sidebar
        # orders by the last message — not by incidental saves on chat switch.
        self.conversation.touch()
        self.chat.set_regenerate_available(False)
        attached = tr("{count} image(s) attached").format(count=len(images))
        shown = text + (f"\n\n📎 {attached}" if images else "")
        self.chat.add_user(shown)
        self.input_bar.set_busy(True)
        self.status_dot.set_active(True)
        self.status_label.setText(tr("Working…"))
        self.chat.start_thinking()

        self.worker = AgentWorker(self.agent, text, images)
        self.worker.event.connect(self._on_event)
        self.worker.approval_needed.connect(self._on_approval)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def on_stop(self) -> None:
        """Ask the running turn to stop."""
        if self._busy():
            self.status_label.setText(tr("Stopping…"))
            self.worker.cancel()

    def regenerate(self) -> None:
        """Re-run the last user message (drops the previous answer from history)."""
        if self._busy():
            return
        if not self._last_user_message:
            return
        # Drop the trailing assistant answer and the user turn we're about to
        # re-send, so history isn't duplicated.
        hist = self.agent.history
        if hist and hist[-1].get("role") == "assistant":
            hist.pop()
        if hist and hist[-1].get("role") == "user":
            hist.pop()
        # Wipe the visible bubbles for that exchange too, then re-run.
        self.chat.pop_last_turn()
        self.on_send(self._last_user_message)

    def _on_event(self, ev: AgentEvent) -> None:
        if ev.kind == "thought":
            self.chat.update_thinking(ev.text or tr("Thinking…"))
            self.chat.add_step("thought", text=ev.text)
        elif ev.kind == "tool_call":
            self.chat.update_thinking(self._tool_status(ev))
            req = ApprovalRequest(ev.tool, ev.args or {})
            self.chat.add_step("tool_call", tool=ev.tool, text=req.describe())
        elif ev.kind == "tool_result":
            self.chat.add_step("tool_result", tool=ev.tool, text=ev.text, ok=ev.ok)
        elif ev.kind == "denied":
            self.chat.add_step("denied", tool=ev.tool,
                               text=tr("Action denied by user."), ok=False)
        elif ev.kind == "thinking":
            # Live reasoning from a thinking model: keep it in the status row,
            # never in the answer bubble.
            self._thinking_text = (self._thinking_text + ev.text)[-400:]
            tail = " ".join(self._thinking_text.split())[-90:]
            self.chat.update_thinking(f"{tr('Thinking…')}  {tail}")
        elif ev.kind == "metrics":
            self._last_metrics = dict(ev.args or {})
            self._refresh_status()
        elif ev.kind == "stream":
            self.chat.append_stream(ev.text)
        elif ev.kind == "stream_end":
            # The model narrated, then reached for a tool: close that bubble so
            # the eventual answer starts a fresh one.
            self.chat.finalize_stream(ev.text)
        elif ev.kind == "final":
            self.chat.stop_thinking()
            # If the answer was streamed, finalise that live bubble; otherwise
            # add a fresh one.
            if not self.chat.finalize_stream(ev.text):
                self.chat.add_ai(ev.text)
        elif ev.kind == "error":
            self.chat.stop_thinking()
            self.chat.finalize_stream("")
            self.chat.add_error(ev.text)

    @staticmethod
    def _tool_status(ev: AgentEvent) -> str:
        """What to show while a tool runs — some of them need the user."""
        args = ev.args or {}
        if ev.tool in ("read_screen", "screenshot"):
            if args.get("region"):
                # Without this the app just looks frozen while a crosshair
                # waits for a drag somewhere else on the screen.
                return tr("Drag a box on the screen…")
            return tr("Taking a screenshot…")
        return tr("Running {tool}…").format(tool=ev.tool)

    def _on_approval(self, request: ApprovalRequest) -> None:
        # The request may arrive while the window is in the tray: bring it up,
        # an approval nobody can see is a hung turn.
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()
        dlg = ApprovalDialog(request, self)
        approved = dlg.exec() == QDialog.DialogCode.Accepted
        # Remember "always allow" choices for this exact command.
        if approved and getattr(dlg, "always_allow", False) and request.tool == "run_command":
            cmd = request.args.get("command", "").strip()
            if cmd and cmd not in self.agent.always_allow:
                self.agent.always_allow.append(cmd)
                self.config.set("always_allow", list(self.agent.always_allow))
                self.config.save()
        if self.worker:
            self.worker.provide_approval(approved)

    def _on_done(self, _answer: str) -> None:
        self.chat.stop_thinking()
        self.chat.finalize_stream("")  # safety: close any dangling stream bubble
        self.input_bar.set_busy(False)
        self.status_dot.set_active(False)
        self.input_bar.focus_input()
        self._refresh_status()
        # Offer a re-run of the last message.
        self.chat.set_regenerate_available(bool(self._last_user_message))
        self.refresh_runtime_async()
        # Persist the turn and reflect the (possibly new) title in the sidebar.
        self._save_current()
        self._refresh_sidebar()

    # ── close / tray ─────────────────────────────────────────────────────
    def _on_close_button(self) -> None:
        if self.config.get("close_to_tray"):
            self.hide()
            self.hidden_to_tray.emit()
        else:
            self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_current()
        if not self.config.get("close_to_tray") or getattr(self, "_really_quit", False):
            self._stop_workers()
        if self.config.get("close_to_tray") and not getattr(self, "_really_quit", False):
            event.ignore()
            self.hide()
            self.hidden_to_tray.emit()
        else:
            event.accept()

    def close_quick_ask(self) -> None:
        window = getattr(self, "_quick", None)
        if window is not None:
            window.close()

    def quit_app(self) -> None:
        self.close_quick_ask()
        # Stop any running agent work, then really quit. The app runs with
        # quitOnLastWindowClosed=False (to live in the tray), so closing the
        # window alone won't end the event loop — quit the application.
        self._really_quit = True
        self._stop_workers()
        self._save_current()
        self.close()
        QApplication.instance().quit()
