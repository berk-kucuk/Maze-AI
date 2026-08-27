"""Main window: frameless aurora card hosting the chat + composer."""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
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
from .chat_view import ChatView
from .effects import AuroraCard, GlowDot
from .input_bar import InputBar
from .onboarding import OnboardingDialog
from .reminders_dialog import RemindersDialog
from .settings_dialog import SettingsDialog
from .sidebar import HistorySidebar
from .theme import LOGO_PATH, STYLESHEET, TEXT_DIM
from .title_bar import TitleBar
from .worker import AgentWorker, ModelResolveWorker

WELCOME = (
    "**Maze AI** is ready. I can run commands, manage files, launch apps, fetch "
    "web pages, send notifications and set reminders on Maze Linux.\n\nTry: "
    "*“show my disk usage”*, *“open firefox”*, *“create a python venv in ~/dev”* "
    "or *“remind me to take a break in 30 minutes”*."
)

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
        self.resize(920, 820)
        self.setMinimumSize(680, 580)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        card = AuroraCard(self)
        root.addWidget(card)

        outer = QVBoxLayout(card)
        outer.setContentsMargins(30, 24, 30, 26)
        outer.setSpacing(10)

        # title bar (spans the full width)
        self.title_bar = TitleBar()
        self.title_bar.minimize_clicked.connect(self.showMinimized)
        self.title_bar.close_clicked.connect(self._on_close_button)
        self.title_bar.settings_clicked.connect(self.open_settings)
        self.title_bar.sidebar_clicked.connect(self.toggle_sidebar)
        self.title_bar.reminders_clicked.connect(self.open_reminders)
        outer.addWidget(self.title_bar)

        # main body: history sidebar + chat column
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(14)

        self.sidebar = HistorySidebar()
        self.sidebar.new_chat_requested.connect(self.new_chat)
        self.sidebar.chat_selected.connect(self.load_chat)
        self.sidebar.chat_deleted.connect(self.delete_chat)
        self.sidebar.chat_renamed.connect(self.rename_chat)
        self.sidebar.chat_exported.connect(self.export_chat)
        body.addWidget(self.sidebar)

        chat_col = QVBoxLayout()
        chat_col.setContentsMargins(0, 0, 0, 0)
        chat_col.setSpacing(10)

        self.chat = ChatView()
        chat_col.addWidget(self.chat, 1)

        # status row
        status_row = QHBoxLayout()
        self.status_dot = GlowDot("#7CFC9A")
        status_row.addWidget(self.status_dot)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        status_row.addWidget(self.status_label)
        status_row.addStretch(1)
        self.shrink_ctx_btn = QPushButton("")
        self.shrink_ctx_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.shrink_ctx_btn.clicked.connect(self.shrink_context)
        self.shrink_ctx_btn.hide()
        status_row.addWidget(self.shrink_ctx_btn)

        self.start_ollama_btn = QPushButton(tr("Start Ollama"))
        self.start_ollama_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_ollama_btn.setToolTip(
            tr("Run `systemctl --user start ollama` and check again")
        )
        self.start_ollama_btn.clicked.connect(self.start_ollama)
        self.start_ollama_btn.hide()
        status_row.addWidget(self.start_ollama_btn)

        self.regen_btn = QPushButton(tr("↻ Regenerate"))
        self.regen_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.regen_btn.setToolTip(tr("Re-run the last message"))
        self.regen_btn.clicked.connect(self.regenerate)
        self.regen_btn.hide()
        status_row.addWidget(self.regen_btn)
        chat_col.addLayout(status_row)

        # composer
        self.input_bar = InputBar()
        self.input_bar.send.connect(self.on_send)
        self.input_bar.stop.connect(self.on_stop)
        chat_col.addWidget(self.input_bar)

        body.addLayout(chat_col, 1)
        outer.addLayout(body, 1)

        self._refresh_status()
        self._render_conversation()
        self._refresh_sidebar()

        # Poll for due reminders and fire desktop notifications.
        self._reminder_timer = QTimer(self)
        self._reminder_timer.setInterval(20_000)  # every 20s
        self._reminder_timer.timeout.connect(self._check_reminders)
        self._reminder_timer.start()

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
        """Send a desktop notification via the tray (or notify-send fallback)."""
        if self.tray is not None:
            self.tray.notify(title, message)
            return
        try:
            import subprocess
            subprocess.Popen(
                ["notify-send", "-a", "Maze AI", title, message],
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
        parts = [desc, f"{tr('mode')}: {mode}"]
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
        if self.conversation.is_empty:
            self.chat.add_ai(tr(WELCOME))
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
        self.regen_btn.setVisible(bool(last_user))

    def toggle_sidebar(self) -> None:
        self.sidebar.setVisible(not self.sidebar.isVisible())

    def new_chat(self) -> None:
        if self.worker and self.worker.isRunning():
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
        if self.worker and self.worker.isRunning():
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

    def delete_chat(self, conv_id: str) -> None:
        self.store.delete(conv_id)
        if conv_id == self.conversation.id:
            # Deleting the open chat drops us onto a fresh one.
            self.conversation = Conversation()
            self.agent.history = self.conversation.messages
            self._render_conversation()
            self._sync_regen_state()
        self._refresh_sidebar()

    def rename_chat(self, conv_id: str) -> None:
        current = self.conversation.title if conv_id == self.conversation.id else ""
        if not current:
            loaded = self.store.load(conv_id)
            current = loaded.title if loaded else ""
        title, ok = QInputDialog.getText(
            self, tr("Rename chat"), tr("New title:"), text=current
        )
        if not ok or not title.strip():
            return
        if conv_id == self.conversation.id:
            self.conversation.title = title.strip()[:80]
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
            self.notify("Maze AI", tr("Chat exported to {path}").format(path=path))
        except OSError as exc:
            self.notify("Maze AI", tr("Export failed: {error}").format(error=exc))

    # ── sending ──────────────────────────────────────────────────────────
    def on_send(self, text: str) -> None:
        if self.worker and self.worker.isRunning():
            return
        images = self.input_bar.take_attachments()
        self._last_user_message = text
        self._thinking_text = ""
        # Bump the conversation's timestamp now, at send time, so the sidebar
        # orders by the last message — not by incidental saves on chat switch.
        self.conversation.touch()
        self.regen_btn.hide()
        attached = tr("{count} image(s) attached").format(count=len(images))
        shown = text + (f"\n\n*📎 {attached}*" if images else "")
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
        if self.worker and self.worker.isRunning():
            self.status_label.setText(tr("Stopping…"))
            self.worker.cancel()

    def regenerate(self) -> None:
        """Re-run the last user message (drops the previous answer from history)."""
        if self.worker and self.worker.isRunning():
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
        self.regen_btn.setVisible(bool(self._last_user_message))
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
