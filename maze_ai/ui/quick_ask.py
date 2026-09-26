"""Quick Ask — the assistant, one keystroke away from anywhere.

A small floating bar that appears over whatever you were doing, answers, and
gets out of the way. It is bound to a desktop shortcut running ``maze-ai
--ask``; the same window backs "explain what I just copied" (``--clipboard``)
and "explain this part of my screen" (``--screenshot``).

It runs its own agent over the same configuration and backend, so the main
chat's conversation is never polluted by a one-off question — and anything
worth keeping can be pushed into a real chat with one click.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..agent import Agent, AgentEvent, ApprovalRequest
from ..config import Config
from ..i18n import tr
from ..llm import build_backend
from .approval import ApprovalDialog
from . import icons
from .chat_view import _Bubble
from .dialogs import shortcut_text
from .effects import MARGIN as CARD_MARGIN
from .effects import AuroraCard, GlowDot
from .input_bar import _Composer
from .richtext import harden_labels, plain_label
from .theme import LINE, STYLESHEET, TEXT, TEXT_DIM, TEXT_FAINT
from .worker import AgentWorker

# The ask row's height, and the window height before anything is asked: the
# card's own margins (it reserves them for its drop shadow) plus the row and
# the hint line, with nothing left over to look like a hole.
_ROW_HEIGHT = 56
_EMPTY_HEIGHT = (CARD_MARGIN + 4) + _ROW_HEIGHT + 30 + (CARD_MARGIN + 2)

# Prompt templates for the one-click actions on copied text.
CLIPBOARD_ACTIONS: list[tuple[str, str]] = [
    ("Explain", "Explain this clearly and briefly:\n\n{text}"),
    ("Fix", "This failed or is wrong. Say what's wrong and give the corrected "
            "version:\n\n{text}"),
    ("Translate", "Translate this. If it is in English, translate to the user's "
                  "language; otherwise translate to English:\n\n{text}"),
    ("Summarise", "Summarise this in a few bullet points:\n\n{text}"),
]


class _AskInput(_Composer):
    """The Quick Ask field: one line by default, and vertically centred.

    A QPlainTextEdit draws its first line at the top of the widget, so in a
    44 px box a single line of text floats above the logo and the button next
    to it. Centring the document inside the viewport is what makes the row read
    as one horizontal line instead of three things at different heights.
    """

    MIN_HEIGHT = 44
    MAX_HEIGHT = 148

    def __init__(self) -> None:
        super().__init__()
        self.document().setDocumentMargin(0)
        self.setFixedHeight(self.MIN_HEIGHT)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # The app-wide sheet gives every QPlainTextEdit a panel background, a
        # border and a 10 px radius. Inside a rounded field that draws a second,
        # differently-rounded box — which is exactly what makes it look boxy.
        # The stylesheet and the frame shape both have to say "none".
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            "QPlainTextEdit { background: transparent; background-color: transparent;"
            f"border: none; border-radius: 0; padding: 0; color: {TEXT};"
            "font-size: 13.5pt; }"
        )
        self.viewport().setAutoFillBackground(False)

    def _content_height(self) -> float:
        lines = max(1.0, self.document().size().height())
        return lines * self.fontMetrics().lineSpacing()

    def _autosize(self) -> None:
        wanted = int(self._content_height()) + 2 * self.frameWidth() + 14
        self.setFixedHeight(max(self.MIN_HEIGHT, min(self.MAX_HEIGHT, wanted)))
        self._centre()

    def _centre(self) -> None:
        top = max(0, int((self.height() - self._content_height()) / 2) - 1)
        self.setViewportMargins(0, top, 0, 0)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._centre()


class QuickAsk(QDialog):
    """A one-shot question window that floats above everything else."""

    open_in_chat = Signal(str, str)   # question, answer

    def __init__(self, config: Config, parent: QWidget | None = None) -> None:
        super().__init__(parent)
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
        self._grow: QPropertyAnimation | None = None
        self._question = ""
        self._answer = ""
        self._pending = ""
        self._images: list[str] = []
        # Tokens are revealed on a timer rather than the instant they arrive —
        # a local model emits them in lumps, and painting each lump looks like
        # stuttering rather than typing.
        self._reveal_timer = QTimer(self)
        self._reveal_timer.setInterval(16)
        self._reveal_timer.timeout.connect(self._reveal)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Dialog
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(STYLESHEET)
        # A command bar, not a dialog: as tall as the row plus the hint line,
        # growing only once there is an answer to show.
        self.resize(760, _EMPTY_HEIGHT)
        self.setMinimumWidth(520)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        card = AuroraCard(self)
        root.addWidget(card)

        lay = QVBoxLayout(card)
        # AuroraCard reserves MARGIN px on every side for its drop shadow, so
        # content has to start inside that — anything less and the footer is
        # painted past the rounded edge and clipped.
        lay.setContentsMargins(CARD_MARGIN + 8, CARD_MARGIN + 4,
                               CARD_MARGIN + 8, CARD_MARGIN + 2)
        lay.setSpacing(0)

        # ── the ask row ──────────────────────────────────────────────────
        # Logo, one big input, one button. Nothing else competes with it.
        # One field, one button, both exactly _ROW_HEIGHT tall and centred on
        # the same line — the logo included.
        self.ask_row = QFrame()
        self.ask_row.setObjectName("askrow")
        # A true pill: radius exactly half the height, and enough contrast that
        # the shape is actually visible. At 4% white the rounding was invisible
        # and the field read as a rectangle.
        self.ask_row.setStyleSheet(
            f"QFrame#askrow {{ background: rgba(255,255,255,0.075);"
            f"border: 1px solid #3a3a42; border-radius: {_ROW_HEIGHT // 2}px; }}"
        )
        row = QHBoxLayout(self.ask_row)
        row.setContentsMargins(14, 6, 6, 6)
        row.setSpacing(12)

        # A glyph rather than the app icon: the logo carries its own black
        # plate, which inside a rounded field looks like a hole punched in it.
        mark = QLabel()
        mark.setPixmap(icons.pixmap("sparkle", TEXT_DIM, 18))
        mark.setFixedWidth(20)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setStyleSheet("background: transparent;")
        row.addWidget(mark, 0, Qt.AlignmentFlag.AlignVCenter)

        self.composer = _AskInput()
        self.composer.setPlaceholderText(tr("Ask anything…"))
        self.composer.submit.connect(self.send)
        row.addWidget(self.composer, 1, Qt.AlignmentFlag.AlignVCenter)

        self.dot = GlowDot("#7CFC9A")
        self.dot.setFixedWidth(14)
        self.dot.hide()
        row.addWidget(self.dot, 0, Qt.AlignmentFlag.AlignVCenter)

        self.send_btn = QPushButton(tr("Ask"))
        self.send_btn.setObjectName("primary")
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setFixedHeight(_ROW_HEIGHT - 12)
        self.send_btn.setMinimumWidth(86)
        # Matches the field it sits in: a pill inside a pill.
        self.send_btn.setStyleSheet(
            f"QPushButton {{ border-radius: {(_ROW_HEIGHT - 12) // 2}px; "
            "padding: 0 18px; font-weight: 600; }"
        )
        self.send_btn.clicked.connect(self.send)
        row.addWidget(self.send_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.ask_row)

        # Everything below the ask row lives in one collapsible body. Hiding
        # children one by one leaves their spacers behind, which is what put a
        # hole under the bar when there was nothing to show.
        self.body = QWidget()
        body_lay = QVBoxLayout(self.body)
        body_lay.setContentsMargins(0, 12, 0, 0)
        body_lay.setSpacing(10)
        self.body.hide()

        # ── context strip (clipboard text / attached capture / files) ────
        # Clipboard text and file names: shown literally, never as markup.
        self.context_label = plain_label("")
        self.context_label.setWordWrap(True)
        self.context_label.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 9pt; background: rgba(255,255,255,0.04);"
            f"border: 1px solid {LINE}; border-radius: 14px; padding: 9px 14px;"
        )
        self.context_label.hide()
        body_lay.addWidget(self.context_label)

        # ── one-click actions on copied text ─────────────────────────────
        self.actions_row = QHBoxLayout()
        self.actions_row.setContentsMargins(0, 0, 0, 0)
        self.actions_row.setSpacing(8)
        self.actions_widget = QWidget()
        self.actions_widget.setLayout(self.actions_row)
        self.actions_widget.hide()
        body_lay.addWidget(self.actions_widget)

        # ── the answer ───────────────────────────────────────────────────
        self.separator = QFrame()
        self.separator.setFixedHeight(1)
        self.separator.setStyleSheet(f"background: {LINE}; border: none;")
        self.separator.hide()
        body_lay.addWidget(self.separator)

        self.answer_area = QScrollArea()
        self.answer_area.setWidgetResizable(True)
        self.answer_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.answer_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.answer_area.setStyleSheet("background: transparent;")
        holder = QWidget()
        holder.setStyleSheet("background: transparent;")
        holder_lay = QVBoxLayout(holder)
        holder_lay.setContentsMargins(2, 4, 8, 0)
        holder_lay.setSpacing(0)
        # The same renderer the chat uses, so a shell command comes out as a
        # proper code block with its own copy button rather than grey text.
        self.answer = _Bubble("", user=False)
        self.answer.setStyleSheet(
            "QFrame#bubble { background: transparent; border: none; }"
        )
        self.answer.layout().setContentsMargins(0, 0, 0, 0)
        holder_lay.addWidget(self.answer)
        holder_lay.addStretch(1)
        self.answer_holder = holder
        self.answer_area.setWidget(holder)
        self.answer_area.hide()
        body_lay.addWidget(self.answer_area, 1)
        lay.addWidget(self.body, 1)

        # ── footer ───────────────────────────────────────────────────────
        self.footer = QWidget()
        self.footer.setFixedHeight(30)
        footer = QHBoxLayout(self.footer)
        footer.setContentsMargins(6, 9, 2, 0)
        footer.setSpacing(10)
        self.hint = plain_label(tr("Enter to ask  ·  Esc to close"))
        self.hint.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 8.5pt;")
        footer.addWidget(self.hint)
        self.status = plain_label("")
        self.status.setStyleSheet(f"color: {TEXT_DIM}; font-size: 8.5pt;")
        footer.addWidget(self.status, 1, Qt.AlignmentFlag.AlignRight)
        self.copy_btn = self._footer_button(tr("⧉ Copy"), self._copy)
        self.copy_btn.setToolTip(shortcut_text("Ctrl+Shift+C"))
        footer.addWidget(self.copy_btn)
        self.chat_btn = self._footer_button(tr("Continue in chat →"), self._to_chat)
        self.chat_btn.setToolTip(shortcut_text("Ctrl+Shift+Return"))
        footer.addWidget(self.chat_btn)
        lay.addWidget(self.footer)

        QShortcut(QKeySequence("Escape"), self, activated=self.close)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.send)
        QShortcut(QKeySequence("Ctrl+Shift+C"), self, activated=self._copy_if_any)
        QShortcut(QKeySequence("Ctrl+Shift+Return"), self, activated=self._to_chat_if_any)
        # Alt+1…4 run the clipboard actions (Explain, Fix, Translate, Summarise).
        for index in range(len(CLIPBOARD_ACTIONS)):
            QShortcut(QKeySequence(f"Alt+{index + 1}"), self,
                      activated=lambda i=index: self._run_action(i))
        self._action_buttons: list[QPushButton] = []
        harden_labels(self)

        # Let the layout say how tall the empty bar has to be, rather than
        # trusting arithmetic that a font change would quietly invalidate.
        self._empty_height = max(_EMPTY_HEIGHT, card.sizeHint().height())
        self.resize(760, self._empty_height)

    def _footer_button(self, text: str, slot) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setStyleSheet(
            f"QToolButton {{ background: transparent; border: none; color: {TEXT_DIM};"
            "font-size: 9pt; padding: 3px 8px; border-radius: 7px; }"
            f"QToolButton:hover {{ color: {TEXT}; background: rgba(255,255,255,0.08); }}"
        )
        button.clicked.connect(slot)
        button.hide()
        return button

    # ── entry points ─────────────────────────────────────────────────────
    def prefill(self, text: str = "", *, send: bool = False) -> None:
        """Put a question in the box, optionally sending it straight away."""
        if text:
            self.composer.setPlainText(text)
            self.composer.moveCursor(self.composer.textCursor().MoveOperation.End)
        if send and text.strip():
            QTimer.singleShot(0, self.send)

    def load_clipboard(self) -> bool:
        """Offer one-click actions on whatever the user just copied."""
        clipboard = QGuiApplication.clipboard()
        text = (clipboard.text() or "").strip()
        if not text:
            self.set_status(tr("The clipboard is empty."))
            return False
        self._context_text = text
        self.body.show()
        preview = " ".join(text.split())
        self.context_label.setText(
            "📋 " + (preview[:220] + "…" if len(preview) > 220 else preview)
        )
        self.context_label.show()
        self._build_actions(text)
        return True

    # File types we can meaningfully summarise by reading them directly.
    TEXT_SUFFIXES = {
        ".txt", ".md", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml",
        ".toml", ".ini", ".conf", ".cfg", ".sh", ".bash", ".zsh", ".fish",
        ".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".cpp", ".hpp", ".rs",
        ".go", ".java", ".kt", ".rb", ".php", ".sql", ".html", ".css", ".xml",
        ".desktop", ".service", ".patch", ".diff",
    }
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

    def load_files(self, paths: list[str], action: str = "ask") -> None:
        """Take a selection from the file manager and frame a question about it.

        The point is that the user has already said what they mean by choosing
        the files — so the box opens with a sensible question already written,
        and for "Explain this" it just runs.
        """
        paths = [p for p in paths if p]
        if not paths:
            return
        first = Path(paths[0])
        vision = getattr(self.agent.backend, "supports_vision", False)
        images = [p for p in paths if Path(p).suffix.lower() in self.IMAGE_SUFFIXES]

        listed = " ".join(f'"{p}"' if " " in p else p for p in paths)
        names = ", ".join(Path(p).name for p in paths[:3])
        if len(paths) > 3:
            names += f" +{len(paths) - 3}"
        self.body.show()
        self.context_label.setText(f"📁 {names}")
        self.context_label.show()

        if images and vision:
            # A vision model should look at the picture, not read its path.
            self._images = images
            question = tr("What does this show?")
        elif images:
            question = tr("Read the text in this image and explain it: {path}").format(
                path=listed
            )
        elif len(paths) == 1 and first.is_dir():
            question = tr("What is in this folder, and what stands out? {path}").format(
                path=listed
            )
        elif len(paths) == 1 and first.suffix.lower() in self.TEXT_SUFFIXES:
            question = tr("Read this file and summarise it: {path}").format(path=listed)
        elif len(paths) == 1:
            question = tr("Tell me what this file is: {path}").format(path=listed)
        else:
            question = tr("Here are some files. Tell me what they are: {path}").format(
                path=listed
            )
        self.prefill(question, send=action == "explain")

    def attach_image(self, path: str) -> None:
        """Attach a screenshot (or any image) as the subject of the question."""
        self._images = [path]
        self.body.show()
        self.context_label.setText(f"🖼 {Path(path).name}")
        self.context_label.show()
        vision = getattr(self.agent.backend, "supports_vision", False)
        self.composer.setPlainText(
            tr("What does this show?") if vision
            else tr("Read the text in this image and explain it.")
        )

    def _build_actions(self, text: str) -> None:
        while self.actions_row.count():
            item = self.actions_row.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._action_buttons = []
        for number, (label, template) in enumerate(CLIPBOARD_ACTIONS, start=1):
            button = QPushButton(tr(label))
            button.setToolTip(shortcut_text(f"Alt+{number}"))
            self._action_buttons.append(button)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFixedHeight(30)
            button.setStyleSheet(
                f"QPushButton {{ background: rgba(255,255,255,0.05); border: 1px solid {LINE};"
                f"border-radius: 15px; color: {TEXT}; padding: 0 16px; font-size: 9.5pt; }}"
                "QPushButton:hover { background: rgba(255,255,255,0.13); }"
                "QPushButton:pressed { background: rgba(255,255,255,0.18); }"
            )
            button.clicked.connect(
                lambda _=False, tpl=template: self.prefill(
                    tpl.format(text=text), send=True
                )
            )
            self.actions_row.addWidget(button)
        self.actions_row.addStretch(1)
        self.actions_widget.show()
        self._resize_to(self.width(), 210)

    def _run_action(self, index: int) -> None:
        if self.actions_widget.isVisible() and index < len(self._action_buttons):
            self._action_buttons[index].click()

    def _copy_if_any(self) -> None:
        if self._answer:
            self._copy()

    def _to_chat_if_any(self) -> None:
        if self._question and self._answer and not (self.worker and self.worker.isRunning()):
            self._to_chat()

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def _fit_to_answer(self) -> None:
        """Shrink or grow the window to the answer, capped to the screen.

        A one-line answer in a half-empty 460 px window looks unfinished; a long
        one should not run off the bottom of the display.
        """
        screen = QGuiApplication.screenAt(self.pos()) or QGuiApplication.primaryScreen()
        limit = int(screen.availableGeometry().height() * 0.7) if screen else 620
        # heightForWidth, not sizeHint: a word-wrapped label's size hint assumes
        # a much narrower box and comes out more than twice too tall, which
        # would leave the window padded with empty space under short answers.
        viewport = max(1, self.answer_area.viewport().width())
        content = self.answer_holder.heightForWidth(viewport)
        if content <= 0:
            content = self.answer_holder.sizeHint().height()
        chrome = self.height() - self.answer_area.height()
        wanted = chrome + content + 24
        self._resize_to(self.width(), max(200, min(wanted, limit)), shrink=True)

    def _resize_to(self, width: int, height: int, shrink: bool = False) -> None:
        """Grow (or settle) to fit new content, gliding rather than jumping."""
        if not shrink and self.height() >= height:
            return
        if abs(self.height() - height) < 8:
            return
        # Stop whatever is still moving: a grow that finishes after a shrink
        # would put the window back where it started.
        running = getattr(self, "_grow", None)
        if running is not None:
            running.stop()
        self._grow = QPropertyAnimation(self, b"geometry", self)
        start = self.geometry()
        end = QRect(start.x(), start.y(), width, height)
        self._grow.setDuration(160)
        self._grow.setStartValue(start)
        self._grow.setEndValue(end)
        self._grow.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._grow.start()

    # ── running a question ───────────────────────────────────────────────
    def send(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        text = self.composer.toPlainText().strip()
        if not text:
            return
        self._question = text
        self._answer = ""
        self._pending = ""
        self.answer.set_streaming_text("")
        self.body.show()
        self.answer_area.show()
        self.separator.show()
        self.copy_btn.hide()
        self.chat_btn.hide()
        self.actions_widget.hide()
        self._resize_to(self.width(), 460)
        self.send_btn.setEnabled(False)
        self.send_btn.setText(tr("Asking…"))
        self.dot.show()
        self.dot.set_active(True)
        self.set_status(tr("Thinking…"))

        images, self._images = self._images, []
        self.worker = AgentWorker(self.agent, text, images)
        self.worker.event.connect(self._on_event)
        self.worker.approval_needed.connect(self._on_approval)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _reveal(self) -> None:
        """Show the next slice of queued text, so it reads as typing."""
        if not self._pending:
            self._reveal_timer.stop()
            return
        take = max(1, min(len(self._pending), -(-len(self._pending) // 6), 24))
        self._answer += self._pending[:take]
        self._pending = self._pending[take:]
        self.answer.set_streaming_text(self._answer)
        bar = self.answer_area.verticalScrollBar()
        gap = bar.maximum() - bar.value()
        if gap > 0:
            bar.setValue(bar.value() + max(1, int(gap * 0.35)))

    def _on_event(self, ev: AgentEvent) -> None:
        if ev.kind == "stream":
            self._pending += ev.text
            if not self._reveal_timer.isActive():
                self._reveal_timer.start()
                self._reveal()
        elif ev.kind in ("final", "stream_end"):
            self._reveal_timer.stop()
            self._pending = ""
            self._answer = ev.text or self._answer
            self.answer.set_text(self._answer)
        elif ev.kind == "error":
            self.answer.set_text(f"**{tr('Error:')}** {ev.text}")
        elif ev.kind == "tool_call":
            self.set_status(tr("Running {tool}…").format(tool=ev.tool))
        elif ev.kind == "thought":
            self.set_status(" ".join((ev.text or "").split())[:80])

    def _on_approval(self, request: ApprovalRequest) -> None:
        self.surface()
        dialog = ApprovalDialog(request, self)
        approved = dialog.exec() == QDialog.DialogCode.Accepted
        if self.worker:
            self.worker.provide_approval(approved)

    def _on_done(self, _answer: str) -> None:
        self._reveal_timer.stop()
        if self._pending:
            self._answer += self._pending
            self._pending = ""
        if self._answer:
            self.answer.set_text(self._answer)
            # One tick later: the code blocks have to be laid out before their
            # height means anything.
            QTimer.singleShot(0, self._fit_to_answer)
        self.send_btn.setEnabled(True)
        self.send_btn.setText(tr("Ask"))
        self.dot.set_active(False)
        self.dot.hide()
        self.set_status("")
        self.copy_btn.setVisible(bool(self._answer))
        self.chat_btn.setVisible(bool(self._answer))
        self.composer.selectAll()          # ready for the next question

    # ── footer actions ───────────────────────────────────────────────────
    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self._answer)
        self.copy_btn.setText(tr("✓ Copied"))
        QTimer.singleShot(1200, lambda: self.copy_btn.setText(tr("⧉ Copy")))

    def _to_chat(self) -> None:
        self.open_in_chat.emit(self._question, self._answer)
        self.close()

    # ── window behaviour ─────────────────────────────────────────────────
    def surface(self) -> None:
        """Show near the top of the screen, focused and ready to type."""
        screen = QGuiApplication.screenAt(self.pos()) or QGuiApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            self.move(
                area.center().x() - self.width() // 2,
                area.top() + max(60, area.height() // 6),
            )
        self.show()
        self.raise_()
        self.activateWindow()
        self.composer.setFocus()

    def closeEvent(self, event) -> None:  # noqa: N802
        worker = self.worker
        if worker is not None and worker.isRunning():
            self.agent.request_cancel()
            worker.cancel()
            worker.wait(1500)
        super().closeEvent(event)
