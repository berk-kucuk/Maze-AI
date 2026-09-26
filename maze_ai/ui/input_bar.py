"""Composer: a multi-line message field on a card, with its own toolbar.

Layout — the text on top, a slim toolbar underneath: attach on the left, a
key hint and the round send / stop button on the right. Everything the
buttons do is also on the keyboard (see the shortcuts dialog).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from ..private import private_dir, tighten
from . import icons
from .dialogs import with_shortcut
from .richtext import plain_label
from .theme import BG, LINE, LINE_HI, PANEL, TEXT, TEXT_DIM, TEXT_FAINT, WHITE

# Files that make sense as a vision/OCR attachment rather than a path.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


class _Composer(QPlainTextEdit):
    submit = Signal()
    image_pasted = Signal(str)      # a screenshot pasted straight from the clipboard
    #: Up arrow in an empty box: bring the last message back for editing.
    recall_requested = Signal()
    #: Page Up / Page Down scroll the transcript, not the (short) text box.
    page_requested = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setPlaceholderText(tr("Message Maze AI…"))
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTabChangesFocus(True)
        self.setFixedHeight(48)
        self.textChanged.connect(self._autosize)

    def _autosize(self) -> None:
        # QPlainTextEdit's document().size().height() is a LINE count (visual
        # lines, wrapping included) — NOT pixels — so turn it into a real pixel
        # height via the font's line spacing. Without this the box stayed frozen
        # at one line and typed text scrolled out of view, looking like it was
        # lost. Grows from ~1 line up to ~8 lines, then scrolls.
        lines = max(1.0, self.document().size().height())
        line_px = self.fontMetrics().lineSpacing()
        chrome = 2 * self.document().documentMargin() + 2 * self.frameWidth() + 12
        height = int(lines * line_px + chrome)
        self.setFixedHeight(max(48, min(200, height)))

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        mods = event.modifiers()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
            mods & Qt.KeyboardModifier.ShiftModifier
        ):
            event.accept()
            self.submit.emit()
            return
        if key == Qt.Key.Key_Up and not mods and not self.toPlainText():
            event.accept()
            self.recall_requested.emit()
            return
        if key in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown) and not (
            mods & Qt.KeyboardModifier.ControlModifier
        ):
            event.accept()
            self.page_requested.emit(-1 if key == Qt.Key.Key_PageUp else 1)
            return
        if event.matches(QKeySequence.StandardKey.Paste):
            # Take a screenshot with the system shortcut, hit Ctrl+V here, ask
            # about it. Without this, pasting an image would do nothing at all.
            saved = self._save_pasted_image()
            if saved:
                event.accept()
                self.image_pasted.emit(saved)
                return
        super().keyPressEvent(event)

    @staticmethod
    def _save_pasted_image() -> str:
        """Write an image sitting on the clipboard to a file; "" if there is none."""
        mime = QGuiApplication.clipboard().mimeData()
        if mime is None or not mime.hasImage():
            return ""
        image = QGuiApplication.clipboard().image()
        if image.isNull():
            return ""
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        try:
            # A pasted screenshot shows the user's screen: owner-only.
            private_dir(cache / "maze-ai")
            target = private_dir(cache / "maze-ai" / "pasted")
            path = target / f"paste-{datetime.now():%Y%m%d-%H%M%S-%f}.png"
            if image.save(str(path), "PNG"):
                tighten(path)
                return str(path)
        except OSError:
            pass
        return ""


class InputBar(QFrame):
    send = Signal(str)
    stop = Signal()

    #: Emitted when files are dropped that aren't images — the composer gets
    #: their paths so the agent can read them.
    files_dropped = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._attachments: list[str] = []
        self._vision = False  # whether the active backend accepts images
        self._busy = False
        # Drag a file onto the composer to work with it. Expected of any modern
        # chat window, and the shortest path from "this file" to "do something".
        self.setAcceptDrops(True)
        self.setObjectName("inputbar")
        self._style(focused=False)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 8, 10, 8)
        outer.setSpacing(2)

        # A small strip showing attached images (hidden when there are none).
        self.attach_label = QPushButton("")
        self.attach_label.setObjectName("chip")
        self.attach_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.attach_label.setIcon(icons.icon("image", TEXT_DIM, 14))
        self.attach_label.setToolTip(tr("Click to clear attachments"))
        self.attach_label.clicked.connect(self.clear_attachments)
        self.attach_label.hide()
        outer.addWidget(self.attach_label, 0, Qt.AlignmentFlag.AlignLeft)

        self.composer = _Composer()
        self.composer.setStyleSheet(
            "QPlainTextEdit { background: transparent; border: none; padding: 6px 2px;"
            "font-size: 11pt; }"
        )
        self.composer.submit.connect(self._emit)
        self.composer.image_pasted.connect(self.add_files_or_images)
        self.composer.installEventFilter(self)
        outer.addWidget(self.composer)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(6)

        self.attach_btn = QToolButton()
        self.attach_btn.setObjectName("icon")
        self.attach_btn.setIcon(
            icons.icon("image", TEXT_DIM, 18, hover=TEXT, disabled="#3a3a42")
        )
        self.attach_btn.setIconSize(QSize(18, 18))
        self.attach_btn.setToolTip(with_shortcut(tr("Attach an image (vision models)"), "Ctrl+O"))
        self.attach_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.attach_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.attach_btn.setFixedSize(32, 32)
        self.attach_btn.clicked.connect(self.attach)
        bar.addWidget(self.attach_btn)
        bar.addStretch(1)

        self.hint = plain_label(tr("Enter to send · Shift+Enter for a new line"))
        self.hint.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 8.5pt;")
        bar.addWidget(self.hint)
        bar.addSpacing(6)

        # While the agent is working, the Send button becomes a Stop button.
        self.stop_btn = QToolButton()
        self.stop_btn.setObjectName("stopbtn")
        self.stop_btn.setIcon(icons.icon("stop", TEXT, 14))
        self.stop_btn.setIconSize(QSize(14, 14))
        self.stop_btn.setFixedSize(34, 34)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setToolTip(with_shortcut(tr("Stop"), "Esc"))
        self.stop_btn.setStyleSheet(
            f"QToolButton#stopbtn {{ background: #26262c; border: 1px solid {LINE_HI};"
            "border-radius: 17px; }"
            "QToolButton#stopbtn:hover { background: #33333a; }"
        )
        self.stop_btn.clicked.connect(self.stop.emit)
        self.stop_btn.hide()
        bar.addWidget(self.stop_btn)

        self.send_btn = QToolButton()
        self.send_btn.setObjectName("sendbtn")
        self.send_btn.setIcon(
            icons.icon("arrow-up", BG, 18, disabled=TEXT_FAINT, stroke=2.2)
        )
        self.send_btn.setIconSize(QSize(18, 18))
        self.send_btn.setFixedSize(34, 34)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setToolTip(with_shortcut(tr("Send"), "Return"))
        self.send_btn.setStyleSheet(
            f"QToolButton#sendbtn {{ background: {WHITE}; border: none; border-radius: 17px; }}"
            "QToolButton#sendbtn:hover { background: #e2e2e6; }"
            "QToolButton#sendbtn:pressed { background: #cfcfd4; }"
            "QToolButton#sendbtn:disabled { background: #232329; }"
        )
        self.send_btn.clicked.connect(self._emit)
        bar.addWidget(self.send_btn)
        outer.addLayout(bar)

        self.composer.textChanged.connect(self._sync_send)
        self._sync_send()

    def _style(self, focused: bool) -> None:
        border = "#4a4a54" if focused else LINE
        self.setStyleSheet(
            f"QFrame#inputbar {{ background: {PANEL}; border: 1px solid {border};"
            "border-radius: 18px; }"
        )

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self.composer and event.type() in (event.Type.FocusIn, event.Type.FocusOut):
            self._style(event.type() == event.Type.FocusIn)
        return False

    def _sync_send(self) -> None:
        self.send_btn.setEnabled(bool(self.composer.toPlainText().strip()) and not self._busy)

    # ── attachments ──────────────────────────────────────────────────────
    def set_vision(self, enabled: bool) -> None:
        """Enable/disable the attach button based on the backend's capability."""
        self._vision = enabled
        self.attach_btn.setEnabled(enabled and not self._busy)
        self.attach_btn.setToolTip(
            with_shortcut(tr("Attach an image"), "Ctrl+O") if enabled
            else tr("The current model doesn't support images")
        )
        if not enabled:
            self.clear_attachments()

    # ── drag & drop ──────────────────────────────────────────────────────
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls() or event.mimeData().hasImage():
            event.acceptProposedAction()
            self._style(focused=True)

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._style(focused=self.composer.hasFocus())

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls() or event.mimeData().hasImage():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        mime = event.mimeData()
        paths = [
            url.toLocalFile() for url in mime.urls() if url.isLocalFile()
        ] if mime.hasUrls() else []
        if not paths and mime.hasImage():
            saved = self.composer._save_pasted_image()
            if saved:
                paths = [saved]
        if paths:
            self.add_files_or_images(paths)
            event.acceptProposedAction()

    def add_files_or_images(self, paths: list[str] | str) -> None:
        """Route dropped/pasted files: images become attachments, the rest paths.

        A non-image file is mentioned by path in the message instead — the agent
        can read, edit or run it with the tools it already has, which works for
        any file type and any model.
        """
        if isinstance(paths, str):
            paths = [paths]
        images: list[str] = []
        others: list[str] = []
        for raw in paths:
            path = Path(raw)
            if not raw:
                continue
            if path.suffix.lower() in IMAGE_SUFFIXES and self._vision:
                images.append(str(path))
            else:
                others.append(str(path))
        if images:
            self._attachments.extend(images)
            self._refresh_attach_label()
        if others:
            existing = self.composer.toPlainText()
            listed = " ".join(f'"{p}"' if " " in p else p for p in others)
            self.composer.setPlainText(
                (existing + ("\n" if existing and not existing.endswith("\n") else "")
                 + listed + " ").lstrip()
            )
            self.composer.moveCursor(self.composer.textCursor().MoveOperation.End)
            self.files_dropped.emit(others)
        self.composer.setFocus()

    def attach(self) -> None:
        if not self.attach_btn.isEnabled():
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("Attach image(s)"), "",
            tr("Images (*.png *.jpg *.jpeg *.gif *.webp *.bmp)"),
        )
        if paths:
            self._attachments.extend(paths)
            self._refresh_attach_label()

    # Kept for callers of the old name.
    _attach = attach

    def clear_attachments(self) -> None:
        self._attachments = []
        self._refresh_attach_label()

    def take_attachments(self) -> list[str]:
        """Return the current attachments and clear them."""
        items = list(self._attachments)
        self.clear_attachments()
        return items

    def _refresh_attach_label(self) -> None:
        n = len(self._attachments)
        if n:
            names = ", ".join(p.rsplit("/", 1)[-1] for p in self._attachments[:3])
            more = f" +{n - 3}" if n > 3 else ""
            self.attach_label.setText(f" {names}{more}   ✕")
            self.attach_label.show()
        else:
            self.attach_label.hide()

    def _emit(self) -> None:
        if self._busy:
            return
        text = self.composer.toPlainText().strip()
        if text:
            self.composer.clear()
            self.send.emit(text)

    def set_text(self, text: str) -> None:
        self.composer.setPlainText(text)
        self.composer.moveCursor(self.composer.textCursor().MoveOperation.End)
        self.composer.setFocus()

    def set_busy(self, busy: bool) -> None:
        # The box stays editable while the agent works, so the next message can
        # be drafted; only sending waits.
        self._busy = busy
        self.send_btn.setVisible(not busy)
        self.stop_btn.setVisible(busy)
        self.attach_btn.setEnabled(self._vision and not busy)
        self.hint.setText(
            tr("Working… Esc to stop") if busy
            else tr("Enter to send · Shift+Enter for a new line")
        )
        self._sync_send()

    def focus_input(self) -> None:
        self.composer.setFocus()

