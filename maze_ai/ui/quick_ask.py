"""Quick Ask — the assistant, one keystroke away from anywhere.

A small floating command bar (Meta+M) that appears over whatever you were
doing, answers, and gets out of the way. It is bound to a desktop shortcut
running ``maze-ai --ask``; the same window backs "explain what I just copied"
(``--clipboard``), "explain this part of my screen" (``--screenshot``) and
the file manager's "Ask Maze AI" menu.

Layout, top to bottom — each part appears only when it has something to say:

* the ask row: mark, one big field, a round send / stop button;
* a context card for what was handed over (clipboard text, files, a capture)
  with one-click actions on it;
* the answer, under a small header that shows what the assistant is doing;
* a footer: the model on the left, the keys that work right now on the right.

It runs its own agent over the same configuration and backend, so the main
chat's conversation is never polluted by a one-off question — and anything
worth keeping can be pushed into a real chat with one key.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QGuiApplication, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..agent import Agent, AgentEvent, ApprovalRequest
from ..config import Config
from ..i18n import tr
from ..llm import build_backend
from . import icons
from .approval import ApprovalDialog
from .chat_view import ErrorCard, _Bubble
from .dialogs import keycap, shortcut_text
from .effects import MARGIN as CARD_MARGIN
from .effects import AuroraCard, GlowDot, TypingDots
from .input_bar import _Composer
from .richtext import harden_labels, plain_label
from .theme import (
    BG,
    DANGER,
    FONT_MONO,
    LINE,
    LINE_HI,
    LOGO_PATH,
    OK,
    STYLESHEET,
    TEXT,
    TEXT_DIM,
    TEXT_FAINT,
    WHITE,
)
from .worker import AgentWorker

# The ask row's height, and the window height before anything is asked: the
# card's own margins (it reserves them for its drop shadow) plus the row and
# the footer, with nothing left over to look like a hole.
_ROW_HEIGHT = 56
_FOOTER_HEIGHT = 34
_EMPTY_HEIGHT = (CARD_MARGIN + 6) + _ROW_HEIGHT + _FOOTER_HEIGHT + (CARD_MARGIN + 6)
_BUTTON = 40            # the round send / stop button
_WIDTH = 780

# Prompt templates for the one-click actions on copied text, with their icon.
CLIPBOARD_ACTIONS: list[tuple[str, str]] = [
    ("Explain", "Explain this clearly and briefly:\n\n{text}"),
    ("Fix", "This failed or is wrong. Say what's wrong and give the corrected "
            "version:\n\n{text}"),
    ("Translate", "Translate this. If it is in English, translate to the user's "
                  "language; otherwise translate to English:\n\n{text}"),
    ("Summarise", "Summarise this in a few bullet points:\n\n{text}"),
]
_ACTION_ICONS = {"Explain": "chat", "Fix": "edit", "Translate": "globe", "Summarise": "file"}


class _AskInput(_Composer):
    """The Quick Ask field: one line by default, and vertically centred.

    A QPlainTextEdit draws its first line at the top of the widget, so in a
    44 px box a single line of text floats above the mark and the button next
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
        # border and a 10 px radius. Inside the bar that draws a second box —
        # the stylesheet and the frame shape both have to say "none".
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            "QPlainTextEdit { background: transparent; background-color: transparent;"
            f"border: none; border-radius: 0; padding: 0; color: {TEXT};"
            "font-size: 14.5pt; }"
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


class _KeyHint(QPushButton):
    """A footer hint that is also a button: "[Ctrl+Shift+C] Copy"."""

    def __init__(self, keys: str, label: str) -> None:
        super().__init__()
        self.setObjectName("keyhint")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFlat(True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 0)
        lay.setSpacing(6)
        self.caption = plain_label(label)
        self.caption.setStyleSheet(f"color: {TEXT_DIM}; font-size: 8.5pt; background: transparent;")
        self.caption.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        cap = keycap(shortcut_text(keys) or keys)
        cap.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay.addWidget(cap)
        lay.addWidget(self.caption)
        self.setFixedHeight(26)
        self.setMinimumWidth(lay.sizeHint().width())
        self.setStyleSheet(
            "QPushButton#keyhint { background: transparent; border: none; border-radius: 7px;"
            "padding: 0; }"
            "QPushButton#keyhint:hover { background: rgba(255,255,255,0.06); }"
        )

    def text(self) -> str:  # noqa: D401 - the caption is what the hint says
        return self.caption.text()


class _ActionChip(QPushButton):
    """One-click action on the handed-over text: icon, label, its Alt+N key."""

    def __init__(self, icon_name: str, label: str, number: int) -> None:
        super().__init__()
        self.setObjectName("actionchip")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedHeight(36)
        self._label = label
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 8, 0)
        lay.setSpacing(8)
        ic = QLabel()
        ic.setPixmap(icons.pixmap(icon_name, TEXT, 15))
        ic.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay.addWidget(ic)
        name = plain_label(label)
        name.setStyleSheet(f"color: {TEXT}; font-size: 9.5pt; background: transparent;")
        name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay.addWidget(name)
        cap = keycap(f"Alt+{number}")
        cap.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay.addWidget(cap)
        self.setMinimumWidth(lay.sizeHint().width() + 4)
        self.setToolTip(shortcut_text(f"Alt+{number}"))
        self.setStyleSheet(
            f"QPushButton#actionchip {{ background: rgba(255,255,255,0.04);"
            f"border: 1px solid {LINE}; border-radius: 11px; padding: 0; }}"
            f"QPushButton#actionchip:hover {{ background: rgba(255,255,255,0.09);"
            f"border-color: {LINE_HI}; }}"
            "QPushButton#actionchip:pressed { background: rgba(255,255,255,0.14); }"
        )

    def text(self) -> str:  # noqa: D401 - the label is what the chip says
        return self._label


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
        self._fade: QPropertyAnimation | None = None
        self._question = ""
        self._answer = ""
        self._pending = ""
        self._images: list[str] = []
        self._context_text = ""
        self._last_question = ""
        self._action_buttons: list[QPushButton] = []
        # Tokens are revealed on a timer rather than the instant they arrive —
        # a local model emits them in lumps, and painting each lump looks like
        # stuttering rather than typing.
        self._reveal_timer = QTimer(self)
        self._reveal_timer.setInterval(16)
        self._reveal_timer.timeout.connect(self._reveal)

        self.setWindowTitle(tr("Quick Ask"))
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Dialog
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(STYLESHEET)
        # A command bar, not a dialog: as tall as the row plus the footer,
        # growing only once there is something to show.
        self.resize(_WIDTH, _EMPTY_HEIGHT)
        self.setMinimumWidth(560)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        card = AuroraCard(self)
        root.addWidget(card)

        lay = QVBoxLayout(card)
        # AuroraCard reserves MARGIN px on every side for its drop shadow, so
        # content has to start inside that — anything less and the footer is
        # painted past the rounded edge and clipped.
        lay.setContentsMargins(CARD_MARGIN + 10, CARD_MARGIN + 6,
                               CARD_MARGIN + 10, CARD_MARGIN + 6)
        lay.setSpacing(0)

        # ── the ask row ──────────────────────────────────────────────────
        # Mark, one big field, one round button — all on one centre line.
        self.ask_row = QFrame()
        self.ask_row.setObjectName("askrow")
        self.ask_row.setFixedHeight(_ROW_HEIGHT)
        self.ask_row.setStyleSheet("QFrame#askrow { background: transparent; border: none; }")
        row = QHBoxLayout(self.ask_row)
        row.setContentsMargins(4, 0, 2, 0)
        row.setSpacing(14)

        self.mark = QLabel()
        self.mark.setFixedSize(34, 34)
        self.mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mark.setPixmap(icons.pixmap("sparkle", TEXT, 18))
        self.mark.setStyleSheet(
            f"background: rgba(255,255,255,0.06); border: 1px solid {LINE_HI};"
            "border-radius: 10px;"
        )
        row.addWidget(self.mark, 0, Qt.AlignmentFlag.AlignVCenter)

        self.composer = _AskInput()
        self.composer.setPlaceholderText(tr("Ask anything…"))
        self.composer.submit.connect(self.send)
        self.composer.recall_requested.connect(self._recall)
        row.addWidget(self.composer, 1, Qt.AlignmentFlag.AlignVCenter)

        # Kept for callers of the old status dot; the footer carries status now.
        self.dot = GlowDot(OK)
        self.dot.hide()

        self.send_btn = QPushButton()
        self.send_btn.setObjectName("asksend")
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setFixedSize(_BUTTON, _BUTTON)
        self.send_btn.setIconSize(QSize(18, 18))
        self.send_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.send_btn.clicked.connect(self._send_or_stop)
        row.addWidget(self.send_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self._set_send_mode(busy=False)
        lay.addWidget(self.ask_row)

        # Everything below the ask row lives in one collapsible body. Hiding
        # children one by one leaves their spacers behind, which is what put a
        # hole under the bar when there was nothing to show.
        self.body = QWidget()
        body_lay = QVBoxLayout(self.body)
        body_lay.setContentsMargins(0, 6, 0, 10)
        body_lay.setSpacing(12)
        self.body.hide()

        self.separator = QFrame()
        self.separator.setObjectName("hsep")
        body_lay.addWidget(self.separator)

        # ── context card (clipboard text / attached capture / files) ─────
        self.context_card = QFrame()
        self.context_card.setObjectName("ctxcard")
        self.context_card.setStyleSheet(
            f"QFrame#ctxcard {{ background: rgba(255,255,255,0.03); border: 1px solid {LINE};"
            "border-radius: 12px; }"
        )
        ctx = QHBoxLayout(self.context_card)
        ctx.setContentsMargins(12, 10, 14, 10)
        ctx.setSpacing(12)
        self.context_icon = QLabel()
        self.context_icon.setFixedSize(30, 30)
        self.context_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.context_icon.setStyleSheet(
            f"background: rgba(255,255,255,0.05); border: 1px solid {LINE}; border-radius: 8px;"
        )
        ctx.addWidget(self.context_icon, 0, Qt.AlignmentFlag.AlignTop)
        ctx_text = QVBoxLayout()
        ctx_text.setSpacing(2)
        self.context_title = plain_label("")
        self.context_title.setStyleSheet(
            f"color: {TEXT_FAINT}; font-size: 7.5pt; font-weight: 700; letter-spacing: 0.8px;"
            "background: transparent;"
        )
        ctx_text.addWidget(self.context_title)
        # Clipboard text and file names: shown literally, never as markup.
        self.context_label = plain_label("")
        self.context_label.setWordWrap(True)
        self.context_label.setStyleSheet(
            f"color: {TEXT}; font-family: {FONT_MONO}; font-size: 9pt; background: transparent;"
        )
        ctx_text.addWidget(self.context_label)
        ctx.addLayout(ctx_text, 1)
        self.context_card.hide()
        body_lay.addWidget(self.context_card)

        # ── one-click actions on copied text ─────────────────────────────
        self.actions_row = QHBoxLayout()
        self.actions_row.setContentsMargins(0, 0, 0, 0)
        self.actions_row.setSpacing(8)
        self.actions_widget = QWidget()
        self.actions_widget.setLayout(self.actions_row)
        self.actions_widget.hide()
        body_lay.addWidget(self.actions_widget)

        # ── the answer ───────────────────────────────────────────────────
        self.answer_head = QWidget()
        head = QHBoxLayout(self.answer_head)
        head.setContentsMargins(2, 0, 0, 0)
        head.setSpacing(8)
        avatar = QLabel()
        pix = QPixmap(LOGO_PATH)
        if not pix.isNull():
            scaled = pix.scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
            scaled.setDevicePixelRatio(2.0)
            avatar.setPixmap(scaled)
        avatar.setFixedSize(20, 20)
        head.addWidget(avatar)
        who = plain_label("Maze AI")
        who.setStyleSheet(f"color: {TEXT}; font-size: 9.5pt; font-weight: 600;")
        head.addWidget(who)
        self.typing = TypingDots(TEXT_DIM)
        head.addWidget(self.typing)
        self.activity = plain_label("")
        self.activity.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 9pt;")
        head.addWidget(self.activity, 1)
        self.answer_head.hide()
        body_lay.addWidget(self.answer_head)

        self.error_card: ErrorCard | None = None
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
        holder_lay.setContentsMargins(2, 0, 10, 0)
        holder_lay.setSpacing(8)
        # The same renderer the chat uses, so a shell command comes out as a
        # proper code block with its own copy button rather than grey text.
        self.answer = _Bubble("", user=False)
        self.answer.layout().setContentsMargins(0, 0, 0, 0)
        holder_lay.addWidget(self.answer)
        holder_lay.addStretch(1)
        self._holder_lay = holder_lay
        self.answer_holder = holder
        self.answer_area.setWidget(holder)
        self.answer_area.hide()
        body_lay.addWidget(self.answer_area, 1)
        lay.addWidget(self.body, 1)

        # ── footer ───────────────────────────────────────────────────────
        self.footer = QFrame()
        self.footer.setObjectName("askfooter")
        self.footer.setFixedHeight(_FOOTER_HEIGHT)
        self.footer.setStyleSheet(
            f"QFrame#askfooter {{ background: transparent; border: none;"
            f"border-top: 1px solid {LINE}; }}"
        )
        footer = QHBoxLayout(self.footer)
        footer.setContentsMargins(4, 6, 0, 0)
        footer.setSpacing(4)
        self.status_dot = GlowDot(OK)
        footer.addWidget(self.status_dot, 0, Qt.AlignmentFlag.AlignVCenter)
        self.model_label = plain_label(self.agent.backend.describe())
        self.model_label.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 8.5pt;")
        footer.addWidget(self.model_label, 0, Qt.AlignmentFlag.AlignVCenter)
        self.status = plain_label("")
        self.status.setStyleSheet(f"color: {TEXT_DIM}; font-size: 8.5pt; padding-left: 8px;")
        footer.addWidget(self.status, 1, Qt.AlignmentFlag.AlignVCenter)

        self.ask_hint = _KeyHint("Return", tr("Ask"))
        self.ask_hint.clicked.connect(self.send)
        footer.addWidget(self.ask_hint)
        self.copy_btn = _KeyHint("Ctrl+Shift+C", tr("Copy"))
        self.copy_btn.clicked.connect(self._copy)
        self.copy_btn.hide()
        footer.addWidget(self.copy_btn)
        self.chat_btn = _KeyHint("Ctrl+Shift+Return", tr("Continue in chat"))
        self.chat_btn.clicked.connect(self._to_chat)
        self.chat_btn.hide()
        footer.addWidget(self.chat_btn)
        self.close_hint = _KeyHint("Esc", tr("Close"))
        self.close_hint.clicked.connect(self.close)
        footer.addWidget(self.close_hint)
        # The old one-line hint, kept for callers; the key hints replace it.
        self.hint = self.ask_hint
        lay.addWidget(self.footer)

        QShortcut(QKeySequence("Escape"), self, activated=self.close)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.send)
        QShortcut(QKeySequence("Ctrl+Shift+C"), self, activated=self._copy_if_any)
        QShortcut(QKeySequence("Ctrl+Shift+Return"), self, activated=self._to_chat_if_any)
        # Alt+1…4 run the clipboard actions (Explain, Fix, Translate, Summarise).
        for index in range(len(CLIPBOARD_ACTIONS)):
            QShortcut(QKeySequence(f"Alt+{index + 1}"), self,
                      activated=lambda i=index: self._run_action(i))
        harden_labels(self)

        # Let the layout say how tall the empty bar has to be, rather than
        # trusting arithmetic that a font change would quietly invalidate.
        self._empty_height = max(_EMPTY_HEIGHT, card.sizeHint().height())
        self.resize(_WIDTH, self._empty_height)

    # ── the send / stop button ───────────────────────────────────────────
    def _set_send_mode(self, busy: bool) -> None:
        radius = _BUTTON // 2
        if busy:
            self.send_btn.setIcon(icons.icon("stop", TEXT, 14))
            self.send_btn.setToolTip(tr("Stop"))
            self.send_btn.setStyleSheet(
                f"QPushButton {{ background: #26262c; border: 1px solid {LINE_HI};"
                f"border-radius: {radius}px; padding: 0; }}"
                "QPushButton:hover { background: #33333a; }"
            )
        else:
            self.send_btn.setIcon(icons.icon("arrow-up", BG, 18, stroke=2.3))
            self.send_btn.setToolTip(tr("Ask") + f"  ({shortcut_text('Return')})")
            self.send_btn.setStyleSheet(
                f"QPushButton {{ background: {WHITE}; border: none;"
                f"border-radius: {radius}px; padding: 0; }}"
                "QPushButton:hover { background: #e2e2e6; }"
                "QPushButton:pressed { background: #cfcfd4; }"
            )

    def _busy(self) -> bool:
        return bool(self.worker and self.worker.isRunning())

    def _send_or_stop(self) -> None:
        if self._busy():
            self.set_status(tr("Stopping…"))
            self.worker.cancel()
        else:
            self.send()

    # ── entry points ─────────────────────────────────────────────────────
    def prefill(self, text: str = "", *, send: bool = False) -> None:
        """Put a question in the box, optionally sending it straight away."""
        if text:
            self.composer.setPlainText(text)
            self.composer.moveCursor(self.composer.textCursor().MoveOperation.End)
        if send and text.strip():
            QTimer.singleShot(0, self.send)

    def _show_context(self, icon_name: str, title: str, text: str) -> None:
        self.body.show()
        self.separator.show()
        self.context_icon.setPixmap(icons.pixmap(icon_name, TEXT_DIM, 16))
        self.context_title.setText(title.upper())
        self.context_label.setText(text)
        self.context_card.show()

    def load_clipboard(self) -> bool:
        """Offer one-click actions on whatever the user just copied."""
        clipboard = QGuiApplication.clipboard()
        text = (clipboard.text() or "").strip()
        if not text:
            self.set_status(tr("The clipboard is empty."))
            return False
        self._context_text = text
        lines = text.splitlines()
        preview = "\n".join(line[:140] for line in lines[:3])
        if len(lines) > 3 or len(preview) > 360:
            preview = preview[:360] + " …"
        title = tr("Clipboard · {count} characters").format(count=len(text))
        self._show_context("copy", title, preview)
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
        title = (tr("Folder") if len(paths) == 1 and first.is_dir()
                 else tr("{count} file(s)").format(count=len(paths)))
        self._show_context("image" if images else "file", title, names)

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
        self._resize_to(self.width(), self._empty_height + 90)
        self.prefill(question, send=action == "explain")

    def attach_image(self, path: str) -> None:
        """Attach a screenshot (or any image) as the subject of the question."""
        self._images = [path]
        self._show_context("image", tr("Screen capture"), Path(path).name)
        self._resize_to(self.width(), self._empty_height + 90)
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
            button = _ActionChip(_ACTION_ICONS.get(label, "sparkle"), tr(label), number)
            self._action_buttons.append(button)
            button.clicked.connect(
                lambda _=False, tpl=template: self.prefill(
                    tpl.format(text=text), send=True
                )
            )
            self.actions_row.addWidget(button)
        self.actions_row.addStretch(1)
        self.actions_widget.show()
        self._resize_to(self.width(), self._empty_height + 150)

    def _run_action(self, index: int) -> None:
        if self.actions_widget.isVisible() and index < len(self._action_buttons):
            self._action_buttons[index].click()

    def _copy_if_any(self) -> None:
        if self._answer:
            self._copy()

    def _to_chat_if_any(self) -> None:
        if self._question and self._answer and not self._busy():
            self._to_chat()

    def _recall(self) -> None:
        """↑ in an empty box brings back the last question."""
        if self._last_question and not self._busy():
            self.prefill(self._last_question)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def _set_activity(self, text: str) -> None:
        self.activity.setText(text)

    def _wanted_height(self) -> tuple[int, int]:
        """(height the answer needs, the most the screen allows)."""
        screen = QGuiApplication.screenAt(self.pos()) or QGuiApplication.primaryScreen()
        limit = int(screen.availableGeometry().height() * 0.7) if screen else 620
        viewport = max(1, self.answer_area.viewport().width())
        content = self.answer_holder.heightForWidth(viewport)
        if content <= 0:
            content = self.answer_holder.sizeHint().height()
        chrome = self.height() - self.answer_area.height()
        return chrome + content + 16, limit

    def _follow_content(self) -> None:
        """While the answer streams in, let the window grow with it (never shrink)."""
        wanted, limit = self._wanted_height()
        self._resize_to(self.width(), min(wanted, limit))

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
        wanted = chrome + content + 16
        self._resize_to(self.width(), max(220, min(wanted, limit)), shrink=True)

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
        self._grow.setDuration(170)
        self._grow.setStartValue(start)
        self._grow.setEndValue(end)
        self._grow.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._grow.start()

    # ── running a question ───────────────────────────────────────────────
    def _clear_error(self) -> None:
        if self.error_card is not None:
            self._holder_lay.removeWidget(self.error_card)
            self.error_card.deleteLater()
            self.error_card = None

    def send(self) -> None:
        if self._busy():
            return
        text = self.composer.toPlainText().strip()
        if not text:
            return
        self._question = text
        self._last_question = text
        self._answer = ""
        self._pending = ""
        self._clear_error()
        self.answer.set_streaming_text("")
        self.answer.hide()
        self.body.show()
        self.separator.show()
        self.answer_head.show()
        self.typing.show()
        self._set_activity(tr("Thinking…"))
        # The answer area appears with the first words; until then the window
        # is just the question and the "thinking" line.
        self.answer_area.hide()
        self.copy_btn.hide()
        self.chat_btn.hide()
        self.actions_widget.hide()
        # Just room for the "thinking" line; the window grows with the answer.
        self._resize_to(self.width(), self._empty_height + 52, shrink=True)
        self._frames = 0
        self._set_send_mode(busy=True)
        self.ask_hint.hide()
        self.status_dot.set_active(True)
        self.set_status("")

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
        self.answer_area.show()
        self.answer.show()
        self.answer.set_streaming_text(self._answer)
        self._frames = getattr(self, "_frames", 0) + 1
        if self._frames % 8 == 1:
            self._follow_content()
        bar = self.answer_area.verticalScrollBar()
        gap = bar.maximum() - bar.value()
        if gap > 0:
            bar.setValue(bar.value() + max(1, int(gap * 0.35)))

    def _on_event(self, ev: AgentEvent) -> None:
        if ev.kind == "stream":
            self._pending += ev.text
            self._set_activity(tr("Writing…"))
            if not self._reveal_timer.isActive():
                self._reveal_timer.start()
                self._reveal()
        elif ev.kind in ("final", "stream_end"):
            self._reveal_timer.stop()
            self._pending = ""
            self._answer = ev.text or self._answer
            self.answer_area.show()
            self.answer.show()
            self.answer.set_text(self._answer)
        elif ev.kind == "error":
            self._clear_error()
            self.error_card = ErrorCard(ev.text)
            self._holder_lay.insertWidget(0, self.error_card)
            self.answer_area.show()
        elif ev.kind == "tool_call":
            self._set_activity(tr("Running {tool}…").format(tool=ev.tool))
        elif ev.kind in ("thought", "thinking"):
            tail = " ".join((ev.text or "").split())[:90]
            if tail:
                self._set_activity(tail)

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
            self.answer_area.show()
            self.answer.show()
            self.answer.set_text(self._answer)
        self.typing.hide()
        self._set_activity("" if self._answer or self.error_card else tr("No answer."))
        if self._answer or self.error_card is not None:
            # One tick later: the code blocks have to be laid out before their
            # height means anything.
            QTimer.singleShot(0, self._fit_to_answer)
        self._set_send_mode(busy=False)
        self.status_dot.set_active(False)
        self.status_dot.set_color(DANGER if self.error_card is not None else OK)
        self.set_status("")
        # With an answer on screen the useful keys are copy and continue;
        # Enter still asks again, it just doesn't need advertising.
        self.ask_hint.setVisible(not self._answer)
        self.copy_btn.setVisible(bool(self._answer))
        self.chat_btn.setVisible(bool(self._answer))
        self.composer.selectAll()          # ready for the next question

    # ── footer actions ───────────────────────────────────────────────────
    def _copy(self) -> None:
        if not self._answer:
            return
        QGuiApplication.clipboard().setText(self._answer)
        self.copy_btn.caption.setText(tr("Copied"))
        self.copy_btn.caption.setStyleSheet(f"color: {OK}; font-size: 8.5pt;")
        QTimer.singleShot(1300, self._reset_copy)

    def _reset_copy(self) -> None:
        try:
            self.copy_btn.caption.setText(tr("Copy"))
            self.copy_btn.caption.setStyleSheet(f"color: {TEXT_DIM}; font-size: 8.5pt;")
        except RuntimeError:
            pass

    def _to_chat(self) -> None:
        self.open_in_chat.emit(self._question, self._answer)
        self.close()

    # ── window behaviour ─────────────────────────────────────────────────
    def surface(self) -> None:
        """Show near the top of the screen, focused and ready to type."""
        was_visible = self.isVisible()
        screen = QGuiApplication.screenAt(self.pos()) or QGuiApplication.primaryScreen()
        if screen is not None and not was_visible:
            area = screen.availableGeometry()
            self.move(
                area.center().x() - self.width() // 2,
                area.top() + max(60, area.height() // 6),
            )
        self.model_label.setText(self.agent.backend.describe())
        if not was_visible:
            # A short fade reads as "summoned", not "a window popped up".
            self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        self.activateWindow()
        self.composer.setFocus()
        if not was_visible:
            self._fade = QPropertyAnimation(self, b"windowOpacity", self)
            self._fade.setDuration(140)
            self._fade.setStartValue(0.0)
            self._fade.setEndValue(1.0)
            self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._fade.start()

    def closeEvent(self, event) -> None:  # noqa: N802
        worker = self.worker
        if worker is not None and worker.isRunning():
            self.agent.request_cancel()
            worker.cancel()
            worker.wait(1500)
        super().closeEvent(event)

