"""Chat transcript: messages, the agent's live activity trail, the empty state.

Layout: one centred reading column (never wider than is comfortable to
read), the user's messages as compact bubbles on the right, the assistant's
answers as open text under its mark on the left — the answer is the content,
it doesn't need a box around it.

Security: nothing in here renders outside text as markup except through
:mod:`.richtext` — the model's Markdown with raw HTML and images disabled,
links opened only after the user has seen the real address.
"""

from __future__ import annotations

import re

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFontMetrics, QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from . import icons
from .effects import TypingDots
from .richtext import markdown_to_html, plain_label, wire_links
from .theme import (
    DANGER,
    DANGER_BG,
    FONT_MONO,
    LINE,
    LINE_HI,
    LOGO_PATH,
    OK,
    TEXT,
    TEXT_DIM,
    TEXT_FAINT,
    USER_BUBBLE,
    USER_TEXT,
)

#: Icon per tool, for the activity trail.
_TOOL_ICON = {
    "run_command": "terminal",
    "launch_app": "sparkle",
    "read_file": "file",
    "write_file": "edit",
    "edit_file": "edit",
    "append_file": "edit",
    "list_dir": "file",
    "search_files": "search",
    "fetch_url": "globe",
    "web_search": "search",
    "clipboard_copy": "copy",
    "screenshot": "image",
    "read_screen": "image",
    "ocr_image": "image",
    "undo_file_change": "refresh",
    "delete_path": "trash",
    "move_path": "chevron-right",
    "copy_path": "copy",
    "create_dir": "plus",
    "notify": "alarm",
    "add_reminder": "alarm",
}

#: Widest the reading column gets, and how much of a long tool output the
#: trail shows before it offers "Show more".
COLUMN_MAX = 860
_STEP_PREVIEW_LINES = 6
_STEP_PREVIEW_CHARS = 700

# Streaming feel. 16 ms is one frame at 60 Hz; the backlog is drained over
# roughly six frames (~100 ms), which keeps up with any model without the text
# arriving in visible lumps.
_FRAME_MS = 16
_DRAIN_FRAMES = 6
_MAX_PER_FRAME = 24
_SCROLL_EASE = 0.35          # fraction of the remaining gap covered per frame
_IDLE_FRAMES = 40            # ~0.6 s of nothing new: park the timer

_FENCE_RE = re.compile(r"```([\w+#.-]*)[ \t]*\n?(.*?)(?:```|\Z)", re.DOTALL)
#: Only fences that have actually closed — no ``\Z`` fallback. Used mid-stream
#: to tell "this code block is finished" from "still being typed", since the
#: closing fence's own three backticks arrive one character at a time too.
_CLOSED_FENCE_RE = re.compile(r"```([\w+#.-]*)[ \t]*\n?(.*?)```", re.DOTALL)


def _action_button(text: str, icon_name: str, tip: str = "") -> QToolButton:
    """A small text+icon action (Copy, Regenerate, Show more)."""
    button = QToolButton()
    button.setObjectName("action")
    button.setText(text)
    button.setIcon(icons.icon(icon_name, TEXT_FAINT, 14, hover=TEXT))
    button.setIconSize(QSize(14, 14))
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    if tip:
        button.setToolTip(tip)
    return button


def _flash_copied(button: QToolButton, idle_text: str, idle_icon: str) -> None:
    button.setText(tr("Copied"))
    button.setIcon(icons.icon("check", OK, 14))

    def restore() -> None:
        try:
            button.setText(idle_text)
            button.setIcon(icons.icon(idle_icon, TEXT_FAINT, 14, hover=TEXT))
        except RuntimeError:
            pass  # the message was removed in the meantime

    QTimer.singleShot(1300, restore)


class CodeBlock(QFrame):
    """A fenced code block: language label, its own copy button, monospace body.

    Markdown inside a QLabel renders code as undifferentiated grey text with no
    way to copy just the command — which is most of what this assistant
    outputs. Splitting fenced blocks into their own widget makes them readable
    and, more importantly, actionable.
    """

    def __init__(self, code: str, language: str = "") -> None:
        super().__init__()
        self.code = code
        self.setObjectName("codeblock")
        self.setStyleSheet(
            f"QFrame#codeblock {{ background: #0a0a0d; border: 1px solid {LINE};"
            "border-radius: 10px; }"
            f"QFrame#codehead {{ background: rgba(255,255,255,0.025);"
            f"border: none; border-bottom: 1px solid {LINE};"
            "border-top-left-radius: 10px; border-top-right-radius: 10px; }"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        head_frame = QFrame()
        head_frame.setObjectName("codehead")
        head = QHBoxLayout(head_frame)
        head.setContentsMargins(12, 3, 4, 3)
        head.setSpacing(6)
        lang = plain_label((language or tr("code")).lower())
        lang.setStyleSheet(
            f"color: {TEXT_FAINT}; font-size: 8.5pt; font-family: {FONT_MONO};"
            "background: transparent;"
        )
        head.addWidget(lang)
        head.addStretch(1)
        self.copy_btn = _action_button(tr("Copy"), "copy", tr("Copy code"))
        self.copy_btn.clicked.connect(self._copy)
        head.addWidget(self.copy_btn)
        lay.addWidget(head_frame)

        body = plain_label(code)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.setContentsMargins(14, 10, 14, 12)
        body.setStyleSheet(
            f"color: {TEXT}; background: transparent; font-family: {FONT_MONO}; "
            "font-size: 9.5pt;"
        )
        lay.addWidget(body)

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.code)
        _flash_copied(self.copy_btn, tr("Copy"), "copy")


def split_code_blocks(text: str) -> list[tuple[str, str, str]]:
    """Split Markdown into ("text"|"code", content, language) segments.

    An unterminated fence (the usual state mid-stream) is still returned as a
    code segment, so a finished answer never leaves stray back-ticks behind.
    """
    segments: list[tuple[str, str, str]] = []
    pos = 0
    for match in _FENCE_RE.finditer(text or ""):
        before = text[pos:match.start()]
        if before.strip():
            segments.append(("text", before.strip("\n"), ""))
        code = match.group(2)
        if code.strip():
            segments.append(("code", code.rstrip("\n"), match.group(1) or ""))
        pos = match.end()
    rest = (text or "")[pos:]
    if rest.strip():
        segments.append(("text", rest.strip("\n"), ""))
    return segments


def _closed_segments(text: str) -> tuple[list[tuple[str, str, str]], str]:
    """Segments for fences that have already closed, plus the still-open tail.

    The tail — everything after the last closed fence, including one that has
    only just been opened — is still being typed and is not resolved into a
    segment yet, so a code block doesn't flicker into existence while its own
    closing ``` is arriving character by character.
    """
    segments: list[tuple[str, str, str]] = []
    pos = 0
    for match in _CLOSED_FENCE_RE.finditer(text or ""):
        before = text[pos:match.start()]
        if before.strip():
            segments.append(("text", before.strip("\n"), ""))
        code = match.group(2)
        if code.strip():
            segments.append(("code", code.rstrip("\n"), match.group(1) or ""))
        pos = match.end()
    # Matches the "\n"-stripping every other text segment already gets, so
    # the tail looks identical once it becomes a real segment at finalize —
    # otherwise a stray leading blank line would vanish right as the message
    # lands, one more small version of the exact glitch this is fixing.
    return segments, (text or "")[pos:].strip("\n")


class _Bubble(QFrame):
    """One message. The user's is a bubble; the assistant's is open text."""

    def __init__(self, text: str, *, user: bool, actions: bool = False) -> None:
        super().__init__()
        self.user = user
        fg = USER_TEXT if user else TEXT
        # Scope to this frame by object name: a bare `QFrame` selector would also
        # match the inner QLabel (QLabel is-a QFrame) and draw a nested box.
        self.setObjectName("bubble")
        if user:
            self.setStyleSheet(
                f"QFrame#bubble {{ background: {USER_BUBBLE}; border-radius: 16px;"
                f"border: 1px solid {LINE_HI}; border-bottom-right-radius: 5px; }}"
            )
        else:
            self.setStyleSheet("QFrame#bubble { background: transparent; border: none; }")
        self.raw_text = text
        self._fg = fg
        lay = QVBoxLayout(self)
        if user:
            lay.setContentsMargins(15, 10, 15, 11)
        else:
            lay.setContentsMargins(0, 2, 0, 0)
        lay.setSpacing(6)

        # Segments (prose labels and code blocks) live in their own layout so
        # the action row always stays at the bottom of the message.
        self._body = QVBoxLayout()
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(10)
        lay.addLayout(self._body)

        self.label = self._new_label()
        self._body.addWidget(self.label)
        self._extra: list[QWidget] = []
        # Streaming-only bookkeeping: how many fences have already closed (so
        # a rebuild only happens when a new one does) and the label showing
        # whatever prose is still being typed after them.
        self._closed_segment_count = 0
        self._tail_label: QLabel | None = None
        if text:
            self.set_text(text)

        # Action row under assistant messages: copy, and (on the last answer)
        # regenerate.
        self.copy_btn: QToolButton | None = None
        self.regen_btn: QToolButton | None = None
        self._actions: QWidget | None = None
        if actions:
            self._actions = QWidget()
            arow = QHBoxLayout(self._actions)
            arow.setContentsMargins(0, 0, 0, 0)
            arow.setSpacing(2)
            self.copy_btn = _action_button(tr("Copy"), "copy", tr("Copy the answer"))
            self.copy_btn.clicked.connect(self._copy)
            arow.addWidget(self.copy_btn)
            self.regen_btn = _action_button(tr("Regenerate"), "refresh",
                                            tr("Re-run the last message"))
            self.regen_btn.hide()
            arow.addWidget(self.regen_btn)
            arow.addStretch(1)
            lay.addWidget(self._actions)

    def _new_label(self) -> QLabel:
        label = QLabel()
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        wire_links(label)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        label.setStyleSheet(
            f"background: transparent; color: {self._fg}; font-size: 10.5pt;"
        )
        return label

    def _set_prose(self, label: QLabel, text: str) -> None:
        """Final rendering of a prose segment: safe Markdown for the model,
        literal text for the user (what they typed is what they see)."""
        label.setProperty("source", text)
        if self.user:
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setText(text)
        else:
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setText(markdown_to_html(text))

    def natural_width(self) -> int:
        """Width the text would like on one line per paragraph (user bubbles)."""
        fm = QFontMetrics(self.label.font())
        longest = max((fm.horizontalAdvance(line) for line in
                       (self.raw_text or " ").splitlines() or [" "]), default=0)
        margins = self.layout().contentsMargins()
        return longest + margins.left() + margins.right() + 6

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.raw_text)
        if self.copy_btn is not None:
            _flash_copied(self.copy_btn, tr("Copy"), "copy")

    def _clear_extra(self) -> None:
        for widget in self._extra:
            self._body.removeWidget(widget)
            widget.deleteLater()
        self._extra = []

    def _render_segments(
        self, segments: list[tuple[str, str, str]], *, final: bool
    ) -> None:
        """Lay out (prose | code) segments in the order they were written.

        The permanent label is only reused when prose comes *first*, because it
        sits at the top of the body layout and everything else is appended
        after it. Reusing it for a later prose chunk — the explanation that
        follows an answer opening with a code block, which is how most of these
        answers are shaped — hoisted that text above the code it was describing.
        """
        self._clear_extra()
        reuse_label = bool(segments) and segments[0][0] == "text"
        for index, (kind, content, language) in enumerate(segments):
            if reuse_label and index == 0:
                if final:
                    self._set_prose(self.label, content)
                else:
                    self.label.setTextFormat(Qt.TextFormat.PlainText)
                    self.label.setText(content)
                self.label.setVisible(True)
                continue
            widget: QWidget
            if kind == "code":
                widget = CodeBlock(content, language)
            else:
                widget = self._new_label()
                self._set_prose(widget, content)
            self._body.addWidget(widget)
            self._extra.append(widget)
        if not reuse_label:
            self.label.setVisible(False)

    def set_text(self, text: str) -> None:
        """Set the final text: prose as safe Markdown, fenced code as code blocks."""
        self.raw_text = text
        self._closed_segment_count = 0
        self._tail_label = None
        segments = split_code_blocks(text) if not self.user else []
        if not any(kind == "code" for kind, _, _ in segments):
            self._clear_extra()
            self._set_prose(self.label, text)
            self.label.setVisible(bool(text))
            return
        self._render_segments(segments, final=True)

    def set_streaming_text(self, text: str) -> None:
        """Set partial text while the answer is still arriving.

        Prose stays plain rather than re-parsed as Markdown on every token:
        that would make a long answer quadratically slower to render, and
        half-typed syntax (an unclosed ``**`` or code fence) flickers as it
        resolves. But a fenced code block that has already *closed* is shown
        as a real, monospaced code block immediately — without this, every
        code answer rendered as a plain paragraph with literal backticks for
        the whole reply, then snapped into a completely different look (font,
        box, spacing) the instant the message finished, which read as the
        message visibly glitching right when it landed.
        """
        self.raw_text = text
        closed, tail = _closed_segments(text)
        if not closed:
            self._clear_extra()
            self.label.setVisible(True)
            self.label.setTextFormat(Qt.TextFormat.PlainText)
            self.label.setText(text)
            return

        # Rebuilding the closed segments is only needed when a new one has
        # just closed — otherwise the code block(s) are already in place and
        # only the still-typing tail below them needs updating, every frame,
        # for as long as the model keeps writing prose after the code.
        if len(closed) != self._closed_segment_count:
            self._render_segments(closed, final=False)
            self._closed_segment_count = len(closed)
            self._tail_label = None

        if tail.strip():
            if self._tail_label is None:
                self._tail_label = self._new_label()
                self._tail_label.setTextFormat(Qt.TextFormat.PlainText)
                self._body.addWidget(self._tail_label)
                self._extra.append(self._tail_label)
            self._tail_label.setText(tail)
        elif self._tail_label is not None:
            self._tail_label.setVisible(False)


def source_text(label: QLabel) -> str:
    """What a prose label shows, as written — before Markdown rendering."""
    source = label.property("source")
    return source if isinstance(source, str) and label.textFormat() == Qt.TextFormat.RichText \
        else label.text()


class StepLine(QFrame):
    """One compact line of agent activity: a tool call, its result, a thought.

    Long tool output is folded to a few lines with a "Show more" toggle, so
    one ``cat`` of a big file can't push the actual answer off the screen.
    """

    def __init__(self, kind: str, tool: str = "", text: str = "", ok: bool = True) -> None:
        super().__init__()
        # Object-name scoped so the border doesn't bleed onto the child QLabels
        # (which inherit from QFrame) and draw a nested box inside the chip.
        self.setObjectName("stepline")
        border = "#3a1d21" if kind in ("denied",) or (kind == "tool_result" and not ok) else LINE
        self.setStyleSheet(
            f"QFrame#stepline {{ background: rgba(255,255,255,0.022); "
            f"border: 1px solid {border}; border-radius: 10px; }}"
        )
        self.full_text = text or ""
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 7, 8, 7)
        lay.setSpacing(9)

        icon_name, color = self._style(kind, tool, ok)
        icon = QLabel()
        icon.setPixmap(icons.pixmap(icon_name, color, 14))
        icon.setFixedSize(16, 18)
        icon.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)

        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        # Plain text, always: this is tool output and model narration — the
        # most attacker-reachable text in the whole app.
        self.body = plain_label("")
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        mono = f"font-family: {FONT_MONO}; font-size: 9pt;"
        tone = TEXT_FAINT if kind == "thought" else TEXT_DIM
        self.body.setStyleSheet(f"color: {tone}; background: transparent; {mono}")
        col.addWidget(self.body)

        self._expanded = False
        self.more_btn: QToolButton | None = None
        preview = self._preview(self.full_text)
        if preview != self.full_text:
            self.more_btn = _action_button(tr("Show more"), "chevron-down")
            self.more_btn.clicked.connect(self.toggle)
            col.addWidget(self.more_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.body.setText(preview)
        lay.addLayout(col, 1)

    @staticmethod
    def _preview(text: str) -> str:
        lines = text.splitlines()
        cut = "\n".join(lines[:_STEP_PREVIEW_LINES])
        if len(cut) > _STEP_PREVIEW_CHARS:
            cut = cut[:_STEP_PREVIEW_CHARS]
        return cut + ("…" if cut != text else "")

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self.body.setText(self.full_text if self._expanded else self._preview(self.full_text))
        if self.more_btn is not None:
            self.more_btn.setText(tr("Show less") if self._expanded else tr("Show more"))

    @staticmethod
    def _style(kind: str, tool: str, ok: bool) -> tuple[str, str]:
        if kind == "thought":
            return "sparkle", TEXT_FAINT
        if kind == "tool_call":
            return _TOOL_ICON.get(tool, "chevron-right"), TEXT
        if kind == "tool_result":
            return ("check", OK) if ok else ("close", DANGER)
        if kind == "denied":
            return "shield", DANGER
        return "chevron-right", TEXT_DIM


class ThinkingRow(QFrame):
    def __init__(self) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 4, 2, 4)
        lay.setSpacing(10)
        self.dots = TypingDots(TEXT_DIM)
        lay.addWidget(self.dots, 0, Qt.AlignmentFlag.AlignVCenter)
        # Live reasoning from the model ends up here: plain text only.
        self.label = plain_label(tr("Thinking…"))
        self.label.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; font-size: 9.5pt;")
        lay.addWidget(self.label, 1)

    def set_text(self, text: str) -> None:
        self.label.setText(text)


class ErrorCard(QFrame):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.setObjectName("errorcard")
        self.setStyleSheet(
            f"QFrame#errorcard {{ background: {DANGER_BG}; border: 1px solid #5a2229;"
            "border-radius: 12px; }"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 14, 10)
        lay.setSpacing(10)
        icon = QLabel()
        icon.setPixmap(icons.pixmap("warning", DANGER, 16))
        icon.setFixedSize(18, 20)
        lay.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
        self.label = plain_label(f"{tr('Error:')} {text}")
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setStyleSheet("color: #ffb3b3; background: transparent; font-size: 10pt;")
        lay.addWidget(self.label, 1)
        self.raw_text = text


class _Row(QWidget):
    """A horizontal row that aligns its content left (AI) or right (user)."""

    def __init__(self, content: QWidget, *, user: bool, avatar: bool = False) -> None:
        super().__init__()
        # Exposed so ChatView can cap the bubble to the current column width
        # (a fixed cap wider than the window pushed user bubbles off the right
        # edge, where they were clipped and appeared missing).
        self.content = content
        self.is_user = user
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        if user:
            content.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            lay.addStretch(1)
            lay.addWidget(content)
        else:
            if avatar:
                av = QLabel()
                pix = QPixmap(LOGO_PATH)
                if not pix.isNull():
                    ratio = 2.0
                    scaled = pix.scaled(int(26 * ratio), int(26 * ratio),
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
                    scaled.setDevicePixelRatio(ratio)
                    av.setPixmap(scaled)
                av.setFixedSize(28, 28)
                av.setAlignment(Qt.AlignmentFlag.AlignCenter)
                av.setStyleSheet(
                    f"background: #0c0c0f; border: 1px solid {LINE}; border-radius: 9px;"
                )
                lay.addWidget(av, 0, Qt.AlignmentFlag.AlignTop)
            else:
                lay.addSpacing(40)
            content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            lay.addWidget(content, 1)


class EmptyState(QWidget):
    """The first screen: what this is, and a few things to try."""

    suggestion_clicked = Signal(str)

    def __init__(self, title: str, subtitle: str,
                 suggestions: list[tuple[str, str, str]], hint: str = "") -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 40, 0, 20)
        lay.setSpacing(0)
        lay.addStretch(2)

        logo = QLabel()
        pix = QPixmap(LOGO_PATH)
        if not pix.isNull():
            scaled = pix.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
            scaled.setDevicePixelRatio(2.0)
            logo.setPixmap(scaled)
        logo.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(logo)
        lay.addSpacing(18)

        heading = plain_label(title)
        heading.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        heading.setStyleSheet("font-size: 19pt; font-weight: 700; letter-spacing: -0.3px;")
        lay.addWidget(heading)
        lay.addSpacing(8)

        sub = plain_label(subtitle)
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        sub.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10.5pt;")
        lay.addWidget(sub)
        lay.addSpacing(28)

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        self.cards: list[QPushButton] = []
        for i, (icon_name, label, prompt) in enumerate(suggestions):
            card = QPushButton()
            card.setObjectName("suggestion")
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card.setMinimumHeight(64)
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            inner = QHBoxLayout(card)
            inner.setContentsMargins(14, 10, 14, 10)
            inner.setSpacing(12)
            ic = QLabel()
            ic.setPixmap(icons.pixmap(icon_name, TEXT_DIM, 18))
            ic.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            inner.addWidget(ic, 0, Qt.AlignmentFlag.AlignVCenter)
            text = plain_label(label)
            text.setWordWrap(True)
            text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            text.setStyleSheet(f"color: {TEXT}; font-size: 10pt; background: transparent;")
            inner.addWidget(text, 1)
            card.setStyleSheet(
                f"QPushButton#suggestion {{ background: rgba(255,255,255,0.025);"
                f"border: 1px solid {LINE}; border-radius: 12px; text-align: left; }}"
                f"QPushButton#suggestion:hover {{ background: rgba(255,255,255,0.06);"
                f"border-color: {LINE_HI}; }}"
                "QPushButton#suggestion:focus { border-color: #6a6a74; }"
            )
            card.clicked.connect(lambda _=False, p=prompt: self.suggestion_clicked.emit(p))
            grid.addWidget(card, i // 2, i % 2)
            self.cards.append(card)
        lay.addWidget(grid_host)

        if hint:
            lay.addSpacing(22)
            tip = plain_label(hint)
            tip.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            tip.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 9pt;")
            lay.addWidget(tip)
        lay.addStretch(3)


class ChatView(QScrollArea):
    suggestion_clicked = Signal(str)
    regenerate_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background: transparent;")

        # Outer host centres a width-capped column; rows live in the column.
        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        host = QHBoxLayout(self._container)
        host.setContentsMargins(24, 8, 24, 20)
        host.setSpacing(0)
        host.addStretch(1)
        self._column = QWidget()
        self._column.setMaximumWidth(COLUMN_MAX)
        self._column.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        host.addWidget(self._column, 100)
        host.addStretch(1)
        self._layout = QVBoxLayout(self._column)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(14)
        self._layout.addStretch(1)
        self.setWidget(self._container)

        self._thinking: ThinkingRow | None = None
        self._stream_bubble: _Bubble | None = None
        self._stream_text = ""
        self._empty: EmptyState | None = None
        self._last_ai: _Bubble | None = None
        # Streamed tokens arrive far faster than a human reads. Repaint on a
        # timer instead of per chunk, so a fast local model can't pin the UI
        # thread laying out text nobody has read yet.
        # Tokens arrive in lumps — a local model emits nothing for 300 ms, then
        # a whole word. Painting them the moment they land looks like stuttering.
        # Instead they queue here and are revealed at a steady frame rate, which
        # is what makes the text look like it is being typed rather than pasted.
        self._stream_pending = ""
        self._stream_timer = QTimer(self)
        self._stream_timer.setInterval(_FRAME_MS)
        self._stream_timer.timeout.connect(self._flush_stream)
        self._idle_frames = 0

        # "Jump to latest" — appears once the reader has scrolled up.
        self.jump_btn = QToolButton(self)
        self.jump_btn.setObjectName("jump")
        self.jump_btn.setIcon(icons.icon("chevron-down", TEXT, 18))
        self.jump_btn.setIconSize(QSize(18, 18))
        self.jump_btn.setFixedSize(36, 36)
        self.jump_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.jump_btn.setToolTip(tr("Jump to the latest message"))
        self.jump_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.jump_btn.setStyleSheet(
            f"QToolButton#jump {{ background: #17171c; border: 1px solid {LINE_HI};"
            "border-radius: 18px; }"
            "QToolButton#jump:hover { background: #222229; }"
        )
        self.jump_btn.clicked.connect(self._scroll_to_bottom)
        self.jump_btn.hide()
        self.verticalScrollBar().valueChanged.connect(self._sync_jump)
        self.verticalScrollBar().rangeChanged.connect(lambda *_: self._sync_jump())

    # ── responsive bubble width ──────────────────────────────────────────
    def _bubble_cap(self) -> int:
        column = min(COLUMN_MAX, max(0, self.viewport().width() - 48))
        return max(220, int(column * 0.82))

    def _apply_cap(self, row: QWidget, cap: int | None = None) -> None:
        content = getattr(row, "content", None)
        if content is None:
            return
        if getattr(row, "is_user", False) and isinstance(content, _Bubble):
            width = min(cap if cap is not None else self._bubble_cap(), content.natural_width())
            content.setFixedWidth(max(60, width))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        cap = self._bubble_cap()
        for i in range(self._layout.count()):
            w = self._layout.itemAt(i).widget()
            if w is not None:
                self._apply_cap(w, cap)
        self._place_jump()

    def _place_jump(self) -> None:
        vp = self.viewport().geometry()
        self.jump_btn.move(vp.center().x() - self.jump_btn.width() // 2,
                           vp.bottom() - self.jump_btn.height() - 12)

    def _sync_jump(self) -> None:
        show = not self._at_bottom(slack=240) and self.verticalScrollBar().maximum() > 0
        if show != self.jump_btn.isVisible():
            self._place_jump()
            self.jump_btn.setVisible(show)
            self.jump_btn.raise_()

    # ── insertion helpers ────────────────────────────────────────────────
    def _insert(self, widget: QWidget) -> None:
        self._drop_empty()
        # Insert before the trailing stretch.
        self._layout.insertWidget(self._layout.count() - 1, widget)
        self._apply_cap(widget)
        QTimer.singleShot(10, self._scroll_to_bottom)

    def _drop_empty(self) -> None:
        if self._empty is not None:
            self._layout.removeWidget(self._empty)
            self._empty.deleteLater()
            self._empty = None

    def show_empty_state(self, title: str, subtitle: str,
                         suggestions: list[tuple[str, str, str]], hint: str = "") -> None:
        self.clear()
        self._empty = EmptyState(title, subtitle, suggestions, hint)
        self._empty.suggestion_clicked.connect(self.suggestion_clicked.emit)
        self._layout.insertWidget(0, self._empty)

    @property
    def empty_state(self) -> EmptyState | None:
        return self._empty

    def _set_last_ai(self, bubble: _Bubble | None) -> None:
        """Only the latest answer offers Regenerate."""
        if self._last_ai is not None:
            try:
                if self._last_ai.regen_btn is not None:
                    self._last_ai.regen_btn.hide()
            except RuntimeError:
                pass
        self._last_ai = bubble
        if bubble is not None and bubble.regen_btn is not None:
            bubble.regen_btn.show()

    def set_regenerate_available(self, available: bool) -> None:
        if self._last_ai is not None and self._last_ai.regen_btn is not None:
            try:
                self._last_ai.regen_btn.setVisible(available)
            except RuntimeError:
                self._last_ai = None

    def add_user(self, text: str) -> None:
        self._insert(_Row(_Bubble(text, user=True), user=True))

    def _ai_bubble(self, text: str) -> _Bubble:
        bubble = _Bubble(text, user=False, actions=True)
        if bubble.regen_btn is not None:
            bubble.regen_btn.clicked.connect(self.regenerate_requested.emit)
        return bubble

    def add_ai(self, text: str) -> None:
        bubble = self._ai_bubble(text)
        self._insert(_Row(bubble, user=False, avatar=True))
        self._set_last_ai(bubble)

    def last_answer(self) -> str:
        try:
            return self._last_ai.raw_text if self._last_ai is not None else ""
        except RuntimeError:
            return ""

    # ── streaming ────────────────────────────────────────────────────────
    def append_stream(self, chunk: str) -> None:
        """Queue a chunk of the answer; the timer reveals it smoothly."""
        if self._stream_bubble is None:
            self.stop_thinking()
            self._stream_text = ""
            self._stream_pending = ""
            self._stream_bubble = self._ai_bubble("")
            self._insert(_Row(self._stream_bubble, user=False, avatar=True))
            self._set_last_ai(None)
            if self._stream_bubble._actions is not None:
                self._stream_bubble._actions.hide()
        self._stream_pending += chunk
        self._idle_frames = 0
        if not self._stream_timer.isActive():
            self._stream_timer.start()
            self._flush_stream()          # first token shows without waiting

    @staticmethod
    def _reveal_size(pending: int) -> int:
        """How many characters to reveal this frame.

        Proportional to the backlog, so a burst catches up quickly while a slow
        model still gets a character at a time instead of a stutter. Never more
        than a short burst per frame, or long answers would snap into place.
        """
        if pending <= 0:
            return 0
        return max(1, min(pending, -(-pending // _DRAIN_FRAMES), _MAX_PER_FRAME))

    def _flush_stream(self) -> None:
        """Reveal the next slice of queued text and glide the view down."""
        if self._stream_bubble is None:
            self._stream_timer.stop()
            return
        if not self._stream_pending:
            # Idle for a while (the model is thinking): stop burning frames.
            self._idle_frames += 1
            if self._idle_frames > _IDLE_FRAMES:
                self._stream_timer.stop()
            return
        self._idle_frames = 0
        take = self._reveal_size(len(self._stream_pending))
        self._stream_text += self._stream_pending[:take]
        self._stream_pending = self._stream_pending[take:]
        self._stream_bubble.set_streaming_text(self._stream_text)
        self._glide_to_bottom()

    def finalize_stream(self, text: str) -> bool:
        """Finish a streamed answer with its full text. Returns True if a live
        streaming bubble was updated (so the caller shouldn't add another)."""
        self._stream_timer.stop()
        if self._stream_bubble is None:
            self._stream_pending = ""
            return False
        # Whatever is still queued belongs to this answer: show it all at once.
        self._stream_text += self._stream_pending
        self._stream_pending = ""
        bubble = self._stream_bubble
        final = text or self._stream_text
        bubble.set_text(final)
        if bubble._actions is not None:
            bubble._actions.setVisible(bool(final.strip()))
        self._stream_bubble = None
        self._stream_text = ""
        if final.strip():
            self._set_last_ai(bubble)
        self._scroll_to_bottom()
        return True

    def add_error(self, text: str) -> None:
        self._insert(_Row(ErrorCard(text), user=False, avatar=False))

    def add_step(self, kind: str, tool: str = "", text: str = "", ok: bool = True) -> None:
        self._insert(_Row(StepLine(kind, tool, text, ok), user=False))

    def start_thinking(self, text: str = "") -> None:
        self.stop_thinking()
        self._thinking = ThinkingRow()
        self._thinking.set_text(text or tr("Thinking…"))
        self._insert(_Row(self._thinking, user=False, avatar=True))

    def update_thinking(self, text: str) -> None:
        if self._thinking:
            self._thinking.set_text(text)

    def stop_thinking(self) -> None:
        if self._thinking is not None:
            row = self._thinking.parentWidget()
            if row is not None:
                self._layout.removeWidget(row)
                row.setParent(None)
                row.deleteLater()
            self._thinking = None

    def pop_last_turn(self) -> None:
        """Remove the trailing exchange — every row back to and including the
        most recent user message (its steps and answer bubble). Used by
        Regenerate so the re-run doesn't stack a duplicate on the transcript."""
        self.stop_thinking()
        self._stream_timer.stop()
        self._stream_bubble = None
        self._stream_text = ""
        self._stream_pending = ""
        self._last_ai = None
        i = self._layout.count() - 2  # -1 is the trailing stretch
        while i >= 0:
            w = self._layout.itemAt(i).widget()
            self._layout.takeAt(i)
            removed_user = isinstance(w, _Row) and getattr(w, "is_user", False)
            if w is not None:
                w.deleteLater()
            i -= 1
            if removed_user:
                break

    def clear(self) -> None:
        self.stop_thinking()
        self._stream_timer.stop()
        self._stream_bubble = None
        self._stream_text = ""
        self._stream_pending = ""
        self._last_ai = None
        self._empty = None
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _at_bottom(self, slack: int = 140) -> bool:
        """Is the view following the conversation, or has the user scrolled up?"""
        bar = self.verticalScrollBar()
        return bar.maximum() - bar.value() <= slack

    def _glide_to_bottom(self) -> None:
        """Ease the view down while text streams in.

        Snapping to the bottom on every frame makes the whole transcript twitch;
        moving a fraction of the remaining distance reads as a smooth follow.
        And if the user has scrolled up to read something, they are left alone.
        """
        if not self._at_bottom():
            return
        bar = self.verticalScrollBar()
        gap = bar.maximum() - bar.value()
        if gap <= 0:
            return
        bar.setValue(bar.value() + max(1, int(gap * _SCROLL_EASE)))

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def scroll_page(self, direction: int) -> None:
        """Page the transcript up (-1) or down (+1) — from the keyboard."""
        bar = self.verticalScrollBar()
        bar.setValue(bar.value() + direction * max(40, int(self.viewport().height() * 0.85)))

    def scroll_to_top(self) -> None:
        self.verticalScrollBar().setValue(0)
