"""Chat transcript: message bubbles and the agent's live activity trail."""

from __future__ import annotations

import re

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from .effects import GlowDot
from .theme import (
    AI_BUBBLE,
    DANGER,
    LINE,
    LOGO_PATH,
    OK,
    TEXT,
    TEXT_DIM,
    TEXT_FAINT,
    USER_BUBBLE,
    USER_TEXT,
)

_TOOL_GLYPH = {
    "run_command": "$",
    "launch_app": "▶",
    "read_file": "◎",
    "write_file": "✎",
    "edit_file": "✎",
    "append_file": "✎",
    "list_dir": "☰",
    "search_files": "⌕",
    "fetch_url": "⇲",
    "web_search": "⌕",
    "clipboard_copy": "⧉",
    "screenshot": "▨",
    "ocr_image": "⎘",
    "undo_file_change": "↶",
    "delete_path": "✕",
    "move_path": "→",
    "copy_path": "⧉",
    "create_dir": "▸",
    "notify": "◆",
    "add_reminder": "⏰",
}


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
_MONO = "'JetBrains Mono','DejaVu Sans Mono',monospace"


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
            f"QFrame#codeblock {{ background: #08080a; border: 1px solid {LINE};"
            "border-radius: 10px; }"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 10)
        lay.setSpacing(5)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        lang = QLabel((language or tr("code")).lower())
        lang.setStyleSheet(
            f"color: {TEXT_FAINT}; font-size: 8pt; letter-spacing: 0.6px; "
            "background: transparent;"
        )
        head.addWidget(lang)
        head.addStretch(1)
        self.copy_btn = QToolButton()
        self.copy_btn.setText(tr("⧉ Copy"))
        self.copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_btn.setStyleSheet(
            f"QToolButton {{ background: transparent; border: none; color: {TEXT_FAINT};"
            "font-size: 8.5pt; padding: 1px 4px; border-radius: 6px; }"
            f"QToolButton:hover {{ color: {TEXT}; background: rgba(255,255,255,0.07); }}"
        )
        self.copy_btn.clicked.connect(self._copy)
        head.addWidget(self.copy_btn)
        lay.addLayout(head)

        body = QLabel(code)
        body.setTextFormat(Qt.TextFormat.PlainText)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.setStyleSheet(
            f"color: {TEXT}; background: transparent; font-family: {_MONO}; "
            "font-size: 9.5pt;"
        )
        lay.addWidget(body)

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.code)
        self.copy_btn.setText(tr("✓ Copied"))
        QTimer.singleShot(1200, lambda: self.copy_btn.setText(tr("⧉ Copy")))


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
    def __init__(self, text: str, *, user: bool, actions: bool = False) -> None:
        super().__init__()
        bg = USER_BUBBLE if user else AI_BUBBLE
        fg = USER_TEXT if user else TEXT
        border = "none" if user else f"1px solid {LINE}"
        # Scope to this frame by object name: a bare `QFrame` selector would also
        # match the inner QLabel (QLabel is-a QFrame) and draw a nested box.
        self.setObjectName("bubble")
        self.setStyleSheet(
            f"QFrame#bubble {{ background: {bg}; border-radius: 14px; border: {border}; }}"
        )
        self.raw_text = text
        self._fg = fg
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(4)

        # Segments (prose labels and code blocks) live in their own layout so
        # the action row always stays at the bottom of the bubble.
        self._body = QVBoxLayout()
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(8)
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

        # An always-visible action row (AI messages only): copy the raw text.
        self.copy_btn: QToolButton | None = None
        if actions:
            arow = QHBoxLayout()
            arow.setContentsMargins(0, 0, 0, 0)
            arow.setSpacing(6)
            arow.addStretch(1)
            self.copy_btn = QToolButton()
            self.copy_btn.setText(tr("⧉ Copy"))
            self.copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.copy_btn.setStyleSheet(
                f"QToolButton {{ background: transparent; border: none; color: {TEXT_FAINT};"
                "font-size: 8.5pt; padding: 2px 4px; border-radius: 6px; }"
                f"QToolButton:hover {{ color: {TEXT}; background: rgba(255,255,255,0.06); }}"
            )
            self.copy_btn.clicked.connect(self._copy)
            arow.addWidget(self.copy_btn)
            lay.addLayout(arow)

    def _new_label(self) -> QLabel:
        label = QLabel()
        label.setTextFormat(Qt.TextFormat.MarkdownText)
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        label.setOpenExternalLinks(True)
        label.setStyleSheet(f"background: transparent; color: {self._fg};")
        return label

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.raw_text)
        if self.copy_btn is not None:
            self.copy_btn.setText(tr("✓ Copied"))
            QTimer.singleShot(1200, lambda: self.copy_btn and self.copy_btn.setText(tr("⧉ Copy")))

    def _clear_extra(self) -> None:
        for widget in self._extra:
            self._body.removeWidget(widget)
            widget.deleteLater()
        self._extra = []

    def _render_segments(
        self, segments: list[tuple[str, str, str]], *, first_format: Qt.TextFormat
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
                self.label.setTextFormat(first_format)
                self.label.setText(content)
                self.label.setVisible(True)
                continue
            widget: QWidget
            if kind == "code":
                widget = CodeBlock(content, language)
            else:
                widget = self._new_label()
                widget.setTextFormat(Qt.TextFormat.MarkdownText)
                widget.setText(content)
            self._body.addWidget(widget)
            self._extra.append(widget)
        if not reuse_label:
            self.label.setVisible(False)

    def set_text(self, text: str) -> None:
        """Set the final text: prose as Markdown, fenced code as code blocks."""
        self.raw_text = text
        segments = split_code_blocks(text)
        if not any(kind == "code" for kind, _, _ in segments):
            self._clear_extra()
            self.label.setTextFormat(Qt.TextFormat.MarkdownText)
            self.label.setText(text)
            self.label.setVisible(bool(text))
            return
        self._render_segments(segments, first_format=Qt.TextFormat.MarkdownText)

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
            self._render_segments(closed, first_format=Qt.TextFormat.PlainText)
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


class StepLine(QFrame):
    """One faint, monospace line describing agent activity."""

    def __init__(self, kind: str, tool: str = "", text: str = "", ok: bool = True) -> None:
        super().__init__()
        # Object-name scoped so the border doesn't bleed onto the child QLabels
        # (which inherit from QFrame) and draw a nested box inside the chip.
        self.setObjectName("stepline")
        self.setStyleSheet(
            f"QFrame#stepline {{ background: rgba(255,255,255,0.03); "
            f"border: 1px solid {LINE}; border-radius: 10px; }}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 7, 12, 7)
        lay.setSpacing(9)

        glyph, color = self._style(kind, tool, ok)
        icon = QLabel(glyph)
        icon.setStyleSheet(f"color: {color}; font-weight: 700; background: transparent;")
        icon.setFixedWidth(16)
        icon.setAlignment(Qt.AlignmentFlag.AlignTop)
        lay.addWidget(icon)

        body = QLabel(text)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        mono = "font-family: 'JetBrains Mono','DejaVu Sans Mono',monospace; font-size: 9.5pt;"
        body.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; {mono}")
        lay.addWidget(body, 1)

    @staticmethod
    def _style(kind: str, tool: str, ok: bool) -> tuple[str, str]:
        if kind == "thought":
            return "…", TEXT_FAINT
        if kind == "tool_call":
            return _TOOL_GLYPH.get(tool, "●"), TEXT
        if kind == "tool_result":
            return ("✓", OK) if ok else ("✕", DANGER)
        if kind == "denied":
            return "⊘", DANGER
        return "•", TEXT_DIM


class ThinkingRow(QFrame):
    def __init__(self) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(8)
        self.dot = GlowDot("#ffffff")
        self.dot.set_active(True)
        lay.addWidget(self.dot)
        self.label = QLabel(tr("Thinking…"))
        self.label.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")
        lay.addWidget(self.label)
        lay.addStretch(1)

    def set_text(self, text: str) -> None:
        self.label.setText(text)


class _Row(QWidget):
    """A horizontal row that aligns its content left (AI) or right (user)."""

    def __init__(self, content: QWidget, *, user: bool, avatar: bool = False) -> None:
        super().__init__()
        # Exposed so ChatView can cap the bubble to the current viewport width
        # (a fixed cap wider than the window pushed user bubbles off the right
        # edge, where they were clipped and appeared missing).
        self.content = content
        self.is_user = user
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        content.setMaximumWidth(680)
        content.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        if user:
            lay.addStretch(1)
            lay.addWidget(content)
        else:
            if avatar:
                av = QLabel()
                pix = QPixmap(LOGO_PATH)
                if not pix.isNull():
                    av.setPixmap(pix.scaled(24, 24, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))
                av.setFixedSize(24, 24)
                av.setAlignment(Qt.AlignmentFlag.AlignTop)
                lay.addWidget(av)
            else:
                lay.addSpacing(34)
            lay.addWidget(content)
            lay.addStretch(1)


class ChatView(QScrollArea):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background: transparent;")

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(18, 12, 18, 12)
        self._layout.setSpacing(12)
        self._layout.addStretch(1)
        self.setWidget(self._container)

        self._thinking: ThinkingRow | None = None
        self._stream_bubble: _Bubble | None = None
        self._stream_text = ""
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

    # ── responsive bubble width ──────────────────────────────────────────
    def _bubble_cap(self) -> int:
        # Leave room for the avatar/indent, layout margins and a right-edge gap
        # so bubbles never spill past the viewport at any window size.
        return max(220, min(680, self.viewport().width() - 96))

    def _apply_cap(self, row: QWidget, cap: int | None = None) -> None:
        content = getattr(row, "content", None)
        if content is not None:
            content.setMaximumWidth(cap if cap is not None else self._bubble_cap())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        cap = self._bubble_cap()
        for i in range(self._layout.count()):
            w = self._layout.itemAt(i).widget()
            if w is not None:
                self._apply_cap(w, cap)

    # ── insertion helpers ────────────────────────────────────────────────
    def _insert(self, widget: QWidget) -> None:
        # Insert before the trailing stretch.
        self._layout.insertWidget(self._layout.count() - 1, widget)
        self._apply_cap(widget)
        QTimer.singleShot(10, self._scroll_to_bottom)

    def add_user(self, text: str) -> None:
        self._insert(_Row(_Bubble(text, user=True), user=True))

    def add_ai(self, text: str) -> None:
        self._insert(_Row(_Bubble(text, user=False, actions=True), user=False, avatar=True))

    # ── streaming ────────────────────────────────────────────────────────
    def append_stream(self, chunk: str) -> None:
        """Queue a chunk of the answer; the timer reveals it smoothly."""
        if self._stream_bubble is None:
            self.stop_thinking()
            self._stream_text = ""
            self._stream_pending = ""
            self._stream_bubble = _Bubble("", user=False, actions=True)
            self._insert(_Row(self._stream_bubble, user=False, avatar=True))
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
        self._stream_bubble.set_text(text or self._stream_text)
        self._stream_bubble = None
        self._stream_text = ""
        self._scroll_to_bottom()
        return True

    def add_error(self, text: str) -> None:
        bubble = _Bubble(f"**{tr('Error:')}** {text}", user=False)
        bubble.label.setStyleSheet(f"background: transparent; color: {DANGER};")
        self._insert(_Row(bubble, user=False, avatar=True))

    def add_step(self, kind: str, tool: str = "", text: str = "", ok: bool = True) -> None:
        self._insert(_Row(StepLine(kind, tool, text, ok), user=False))

    def start_thinking(self, text: str = "") -> None:
        self.stop_thinking()
        self._thinking = ThinkingRow()
        self._thinking.set_text(text or tr("Thinking…"))
        self._insert(_Row(self._thinking, user=False))

    def update_thinking(self, text: str) -> None:
        if self._thinking:
            self._thinking.set_text(text)

    def stop_thinking(self) -> None:
        if self._thinking is not None:
            row = self._thinking.parentWidget()
            if row is not None:
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
